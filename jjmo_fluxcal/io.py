"""
JJMO Spectrograph Pipeline -- Step 1: Data Ingestion & Standardization

This module provides format-specific readers and a dispatcher that auto-detects
file types to produce a uniform list of specutils.Spectrum1D objects with
standardised metadata.

Supported formats
-----------------
1. Paired .fit/.txt (Sirius-style): 2D FITS image + tab-delimited wavelength file.
2. CSV (Betelgeuse-style): three-column (index, wavelength, flux) CSV.
3. Generic 1D FITS: standard WCS keywords CRVAL1/CDELT1/CRPIX1.

All readers return Spectrum1D with a ``meta`` dictionary containing at least
the keys defined in ``REQUIRED_META_KEYS`` (set to None when unavailable).
"""

import logging
import warnings
from pathlib import Path

import numpy as np
import yaml
from astropy.io import fits
import astropy.units as u
try:
    from specutils import Spectrum as Spectrum1D  # specutils >= 2.3
except ImportError:
    from specutils import Spectrum1D  # specutils < 2.3

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Metadata schema
# ---------------------------------------------------------------------------

# Keys that every Spectrum1D.meta dict must contain (value may be None).
REQUIRED_META_KEYS = (
    "wavelength_unit",   # str, e.g. "Angstrom"
    "flux_unit",         # str, e.g. "counts" or "counts/s"
    "exptime",           # float [seconds] or None
    "airmass",           # float or None
    "date_obs",          # str ISO-8601 or None
    "instrument",        # str or None
    "segment_id",        # str identifier for this wavelength segment
    "source_file",       # str, path to the primary file that was read
)

# Minimum number of valid (finite) pixels to accept a segment.
MIN_VALID_PIXELS = 10

# Air wavelength convention is used throughout (JJMO data is in air).
WAVELENGTH_CONVENTION = "air"


def _empty_meta():
    """Return a metadata dict initialised with None for every required key."""
    return {k: None for k in REQUIRED_META_KEYS}


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _validate_spectrum(wavelength, flux, source_label=""):
    """Validate a wavelength/flux pair and return a boolean mask.

    Checks performed:
    - NaN / Inf values -> masked (True = bad pixel).
    - Wavelength monotonically increasing; sorted if not.
    - Minimum pixel count.

    Returns
    -------
    wavelength : np.ndarray
        Sorted wavelength array.
    flux : np.ndarray
        Correspondingly sorted flux array.
    mask : np.ndarray of bool
        True where the pixel is invalid (NaN, Inf, or non-positive wavelength).

    Raises
    ------
    ValueError
        If fewer than MIN_VALID_PIXELS valid pixels remain after masking.
    """
    wavelength = np.asarray(wavelength, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64)

    if wavelength.shape != flux.shape:
        raise ValueError(
            f"Wavelength and flux arrays must have the same shape, "
            f"got {wavelength.shape} vs {flux.shape} ({source_label})"
        )

    # Mask non-finite values
    bad = ~np.isfinite(wavelength) | ~np.isfinite(flux)
    # Mask non-positive wavelengths
    bad |= wavelength <= 0

    n_bad = int(np.sum(bad))
    if n_bad > 0:
        logger.info(
            "%s: masked %d non-finite or invalid pixels", source_label, n_bad
        )

    # Ensure monotonically increasing wavelength
    if not np.all(np.diff(wavelength[~bad]) >= 0):
        logger.info("%s: wavelength not monotonic -- sorting", source_label)
        order = np.argsort(wavelength)
        wavelength = wavelength[order]
        flux = flux[order]
        bad = bad[order]

    n_valid = int(np.sum(~bad))
    if n_valid < MIN_VALID_PIXELS:
        raise ValueError(
            f"Only {n_valid} valid pixels in segment ({source_label}); "
            f"minimum is {MIN_VALID_PIXELS}"
        )

    return wavelength, flux, bad


def _make_spectrum1d(wavelength, flux, mask, meta):
    """Wrap validated arrays into a Spectrum1D with standardised metadata.

    Parameters
    ----------
    wavelength : array-like  [Angstrom]
    flux : array-like        [counts]
    mask : array-like bool   (True = bad)
    meta : dict              Must contain all REQUIRED_META_KEYS.
    """
    # Ensure all required keys are present
    for key in REQUIRED_META_KEYS:
        meta.setdefault(key, None)

    # specutils requires strictly monotonic spectral_axis even for masked
    # pixels, so replace masked wavelengths with interpolated values to keep
    # the axis valid while preserving the mask.
    wavelength = np.asarray(wavelength, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if np.any(mask):
        good = ~mask
        if np.sum(good) >= 2:
            wavelength[mask] = np.interp(
                np.where(mask)[0], np.where(good)[0], wavelength[good]
            )

    spec = Spectrum1D(
        flux=flux * u.ct,
        spectral_axis=wavelength * u.AA,
        mask=mask,
        meta=meta,
    )
    return spec


# ---------------------------------------------------------------------------
# FITS header helpers (handle SBIG keyword variations)
# ---------------------------------------------------------------------------

def _extract_fits_header_meta(header):
    """Pull metadata from a FITS header, handling SBIG keyword variants.

    The SBIG ST-7 CCD software used at JJMO wrote different keyword names
    depending on version: e.g. EXPTIME vs EXPOSURE, DATE-OBS vs SBIGDATE,
    INSTRUME vs USER_1.  This function normalises them.
    """
    meta = _empty_meta()

    # Exposure time
    exptime = header.get("EXPTIME", header.get("EXPOSURE"))
    if exptime is not None:
        meta["exptime"] = float(exptime)

    # Observation date -- normalise MM/DD/YY to ISO
    date_obs = header.get("DATE-OBS", header.get("DATE", header.get("SBIGDATE")))
    if date_obs is not None:
        date_obs = _normalise_date(str(date_obs))
    # Append time if available
    time_obs = header.get("TIME-OBS", header.get("SBIGTIME"))
    if date_obs and time_obs:
        meta["date_obs"] = f"{date_obs}T{time_obs}"
    elif date_obs:
        meta["date_obs"] = date_obs

    # Instrument
    meta["instrument"] = header.get("INSTRUME", header.get("USER_1"))

    # Airmass (not present in JJMO data, but standard keyword)
    meta["airmass"] = header.get("AIRMASS")

    return meta


def _normalise_date(date_str):
    """Convert MM/DD/YY dates to ISO-8601 YYYY-MM-DD.

    Already-ISO dates pass through unchanged.
    """
    if "-" in date_str and len(date_str) >= 10:
        return date_str  # already ISO
    parts = date_str.split("/")
    if len(parts) == 3:
        mm, dd, yy = parts
        year = int(yy)
        # Two-digit year: pivot at 50 -> 19xx vs 20xx
        if year < 100:
            year = year + 2000 if year < 50 else year + 1900
        return f"{year:04d}-{int(mm):02d}-{int(dd):02d}"
    return date_str  # unrecognised format, return as-is


# ---------------------------------------------------------------------------
# Reader: paired .fit + .txt  (Sirius-style)
# ---------------------------------------------------------------------------

def read_fit_txt_pair(fit_path, txt_path=None, *, metadata_overrides=None):
    """Read a 2D FITS image and its paired wavelength text file.

    The 2D image is collapsed along the spatial axis (axis=0) with nansum.
    The txt file is expected to be tab-delimited with columns
    (index, wavelength, intensity); wavelength may be in descending order.

    Parameters
    ----------
    fit_path : str or Path
        Path to the .fit FITS file.
    txt_path : str or Path, optional
        Path to the paired .txt file.  If None, inferred by replacing the
        .fit extension with .txt.
    metadata_overrides : dict, optional
        Extra metadata to merge (overrides header values).

    Returns
    -------
    Spectrum1D
    """
    fit_path = Path(fit_path)
    if txt_path is None:
        txt_path = fit_path.with_suffix(".txt")
    else:
        txt_path = Path(txt_path)

    if not fit_path.exists():
        raise FileNotFoundError(f"FITS file not found: {fit_path}")
    if not txt_path.exists():
        raise FileNotFoundError(
            f"Wavelength text file not found: {txt_path} "
            f"(expected companion to {fit_path})"
        )

    # --- Read 2D FITS and collapse spatial axis ---
    with fits.open(fit_path) as hdul:
        data_2d = hdul[0].data.astype(np.float64)
        header = hdul[0].header.copy()

    if data_2d.ndim != 2:
        raise ValueError(
            f"Expected 2D FITS image, got {data_2d.ndim}D ({fit_path})"
        )

    # Collapse along spatial axis (rows) to get 1D spectrum
    flux = np.nansum(data_2d, axis=0)

    # --- Read wavelength from text file ---
    # Format: tab-delimited, first line is a header (quoted string),
    # columns: index, wavelength, intensity.  Trailing blank lines are
    # silently dropped via invalid_raise=False + NaN row filtering.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # suppress genfromtxt trailing-line warnings
        txt_data = np.genfromtxt(
            txt_path, delimiter="\t", skip_header=1, invalid_raise=False
        )
    # Drop any all-NaN rows produced by trailing blank lines
    if txt_data.ndim == 2:
        good_rows = ~np.all(np.isnan(txt_data), axis=1)
        txt_data = txt_data[good_rows]

    if txt_data.ndim == 1:
        # Single-column fallback
        wavelength = txt_data
    elif txt_data.ndim == 2 and txt_data.shape[1] >= 2:
        wavelength = txt_data[:, 1]
    else:
        raise ValueError(f"Cannot parse wavelength from {txt_path}")

    # The txt wavelengths are often in descending order; the FITS pixel order
    # matches the txt row order, so if wavelength is descending we flip both.
    if len(wavelength) > 1 and wavelength[0] > wavelength[-1]:
        wavelength = wavelength[::-1]
        flux = flux[::-1]

    # Trim to matching lengths (txt may have trailing blank lines)
    n = min(len(wavelength), len(flux))
    wavelength = wavelength[:n]
    flux = flux[:n]

    # --- Metadata ---
    meta = _extract_fits_header_meta(header)
    meta["wavelength_unit"] = "Angstrom"
    meta["flux_unit"] = "counts"
    meta["source_file"] = str(fit_path)
    meta["segment_id"] = fit_path.stem  # e.g. "3900"

    if metadata_overrides:
        meta.update(metadata_overrides)

    # --- Validate and return ---
    wavelength, flux, mask = _validate_spectrum(
        wavelength, flux, source_label=str(fit_path)
    )
    return _make_spectrum1d(wavelength, flux, mask, meta)


# ---------------------------------------------------------------------------
# Reader: CSV  (Betelgeuse-style)
# ---------------------------------------------------------------------------

def read_csv(csv_path, *, wavelength_col=1, flux_col=2,
             delimiter=",", skip_header=0, metadata_overrides=None):
    """Read a CSV spectrum file.

    Default layout is three columns: (index, wavelength, flux).

    Parameters
    ----------
    csv_path : str or Path
    wavelength_col, flux_col : int
        Column indices (0-based) for wavelength and flux.
    delimiter : str
    skip_header : int
        Number of header lines to skip.
    metadata_overrides : dict, optional

    Returns
    -------
    Spectrum1D
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    data = np.genfromtxt(
        csv_path, delimiter=delimiter, skip_header=skip_header,
        invalid_raise=False
    )

    if data.ndim == 1:
        raise ValueError(
            f"CSV file appears to have only one column ({csv_path})"
        )
    if data.shape[1] <= max(wavelength_col, flux_col):
        raise ValueError(
            f"CSV file has {data.shape[1]} columns but need columns "
            f"{wavelength_col} and {flux_col} ({csv_path})"
        )

    wavelength = data[:, wavelength_col]
    flux = data[:, flux_col]

    meta = _empty_meta()
    meta["wavelength_unit"] = "Angstrom"
    meta["flux_unit"] = "counts"
    meta["source_file"] = str(csv_path)
    # Derive segment_id from filename: e.g. "Betelgeuse_4400" -> "4400"
    stem = csv_path.stem
    parts = stem.rsplit("_", 1)
    meta["segment_id"] = parts[-1] if len(parts) > 1 else stem

    if metadata_overrides:
        meta.update(metadata_overrides)

    wavelength, flux, mask = _validate_spectrum(
        wavelength, flux, source_label=str(csv_path)
    )
    return _make_spectrum1d(wavelength, flux, mask, meta)


# ---------------------------------------------------------------------------
# Reader: generic 1D FITS with WCS
# ---------------------------------------------------------------------------

def read_fits_1d(fits_path, *, hdu_index=0, metadata_overrides=None):
    """Read a 1D FITS spectrum using WCS keywords (CRVAL1/CDELT1/CRPIX1).

    This covers already-reduced spectra from other observatories or the
    output of the existing ``import_to_fits()`` function.

    Parameters
    ----------
    fits_path : str or Path
    hdu_index : int
        Index of the HDU containing the spectrum.
    metadata_overrides : dict, optional

    Returns
    -------
    Spectrum1D
    """
    fits_path = Path(fits_path)
    if not fits_path.exists():
        raise FileNotFoundError(f"FITS file not found: {fits_path}")

    with fits.open(fits_path) as hdul:
        hdu = hdul[hdu_index]
        flux = np.asarray(hdu.data, dtype=np.float64)
        header = hdu.header.copy()

    if flux.ndim != 1:
        raise ValueError(
            f"Expected 1D FITS data, got {flux.ndim}D ({fits_path})"
        )

    # Build wavelength from WCS keywords
    crval1 = header.get("CRVAL1")
    cdelt1 = header.get("CDELT1", header.get("CD1_1"))
    crpix1 = header.get("CRPIX1", 1.0)

    if crval1 is None or cdelt1 is None:
        raise ValueError(
            f"FITS header missing CRVAL1/CDELT1 wavelength keywords ({fits_path})"
        )

    n_pix = len(flux)
    # Standard FITS WCS: lambda_i = CRVAL1 + (i + 1 - CRPIX1) * CDELT1
    pixel_indices = np.arange(1, n_pix + 1, dtype=np.float64)
    wavelength = crval1 + (pixel_indices - crpix1) * cdelt1

    meta = _extract_fits_header_meta(header)
    meta["wavelength_unit"] = "Angstrom"
    meta["flux_unit"] = "counts"
    meta["source_file"] = str(fits_path)
    meta["segment_id"] = fits_path.stem

    if metadata_overrides:
        meta.update(metadata_overrides)

    wavelength, flux, mask = _validate_spectrum(
        wavelength, flux, source_label=str(fits_path)
    )
    return _make_spectrum1d(wavelength, flux, mask, meta)


# ---------------------------------------------------------------------------
# Metadata from companion YAML config
# ---------------------------------------------------------------------------

def load_metadata_config(config_path):
    """Load a YAML metadata config file.

    Expected format::

        global:
            instrument: "SBIG ST-7"
            date_obs: "2004-01-29"
            airmass: 1.05
        segments:
            3900:
                exptime: 30.0
            4400:
                exptime: 30.0

    The ``global`` block is applied to every segment; per-segment overrides
    are merged on top (segment keys are matched against ``segment_id``).

    Returns
    -------
    dict
        ``{"global": {...}, "segments": {...}}``
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as fh:
        cfg = yaml.safe_load(fh)

    if not isinstance(cfg, dict):
        raise ValueError(f"Config file must be a YAML mapping ({config_path})")

    # Normalize segment keys to strings (YAML may parse bare numbers as int)
    raw_segments = cfg.get("segments", {})
    segments = {str(k): v for k, v in raw_segments.items()}

    return {
        "global": cfg.get("global", {}),
        "segments": segments,
    }


def _apply_config_metadata(spectrum, config):
    """Merge YAML config metadata into a Spectrum1D.meta dict.

    Global keys are applied first, then segment-specific overrides.
    Existing non-None values in meta are NOT overwritten by config
    (header values take priority over config for fields like exptime).
    """
    if config is None:
        return

    # Apply global metadata (only fill in missing values)
    for key, val in config.get("global", {}).items():
        if spectrum.meta.get(key) is None:
            spectrum.meta[key] = val

    # Apply segment-specific overrides
    seg_id = spectrum.meta.get("segment_id")
    seg_cfg = config.get("segments", {}).get(seg_id, {})
    # Segment overrides DO replace global values (more specific wins)
    for key, val in seg_cfg.items():
        spectrum.meta[key] = val


# ---------------------------------------------------------------------------
# Format auto-detection
# ---------------------------------------------------------------------------

def _detect_format(path):
    """Guess the format of a spectral data file.

    Returns
    -------
    str
        One of ``"fit_txt"``, ``"csv"``, ``"fits_1d"``.

    Raises
    ------
    ValueError
        If format cannot be determined.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        return "csv"

    if suffix == ".fit":
        # Paired .fit + .txt (2D image)
        txt_companion = path.with_suffix(".txt")
        if txt_companion.exists():
            return "fit_txt"
        # Fallback: might be a 1D FITS with .fit extension
        with fits.open(path) as hdul:
            if hdul[0].data is not None and hdul[0].data.ndim == 1:
                return "fits_1d"
        raise ValueError(
            f"File {path} is a .fit file but no companion .txt found "
            f"and it is not a 1D spectrum"
        )

    if suffix in (".fits", ".fts"):
        with fits.open(path) as hdul:
            if hdul[0].data is not None and hdul[0].data.ndim == 1:
                return "fits_1d"
            elif hdul[0].data is not None and hdul[0].data.ndim == 2:
                txt_companion = path.with_suffix(".txt")
                if txt_companion.exists():
                    return "fit_txt"
                raise ValueError(
                    f"2D FITS {path} has no companion .txt wavelength file"
                )
        raise ValueError(f"Cannot determine spectrum format for {path}")

    raise ValueError(f"Unrecognised file extension: {suffix} ({path})")


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def read_spectrum(path, *, format=None, metadata_overrides=None):
    """Read a single spectrum file, auto-detecting format if needed.

    Parameters
    ----------
    path : str or Path
    format : str, optional
        Force a specific format (``"fit_txt"``, ``"csv"``, ``"fits_1d"``).
    metadata_overrides : dict, optional

    Returns
    -------
    Spectrum1D
    """
    path = Path(path)
    if format is None:
        format = _detect_format(path)

    if format == "fit_txt":
        return read_fit_txt_pair(path, metadata_overrides=metadata_overrides)
    elif format == "csv":
        return read_csv(path, metadata_overrides=metadata_overrides)
    elif format == "fits_1d":
        return read_fits_1d(path, metadata_overrides=metadata_overrides)
    else:
        raise ValueError(f"Unknown format: {format!r}")


def read_directory(directory, *, extensions=None, config_path=None,
                   metadata_overrides=None):
    """Discover and read all spectra in a directory.

    Files are auto-detected by extension.  For .fit files the companion .txt
    is expected alongside.  Results are sorted by minimum wavelength.

    Parameters
    ----------
    directory : str or Path
    extensions : list of str, optional
        File extensions to consider, e.g. [".fit", ".csv"].
        Default: [".fit", ".csv", ".fits", ".fts"].
    config_path : str or Path, optional
        Path to a YAML metadata config file (see ``load_metadata_config``).
    metadata_overrides : dict, optional
        Applied to every segment (lowest priority).

    Returns
    -------
    list of Spectrum1D
        Ordered by minimum wavelength of each segment.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    if extensions is None:
        extensions = [".fit", ".csv", ".fits", ".fts"]

    config = None
    if config_path is not None:
        config = load_metadata_config(config_path)

    # Collect candidate files (skip .txt -- those are companions, not primary)
    candidates = sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in extensions
    )

    if not candidates:
        warnings.warn(f"No spectral data files found in {directory}")
        return []

    spectra = []
    for fpath in candidates:
        try:
            fmt = _detect_format(fpath)
        except ValueError as exc:
            logger.warning("Skipping %s: %s", fpath, exc)
            continue

        try:
            spec = read_spectrum(
                fpath, format=fmt, metadata_overrides=metadata_overrides
            )
        except (ValueError, FileNotFoundError) as exc:
            logger.warning("Failed to read %s: %s", fpath, exc)
            continue

        if config is not None:
            _apply_config_metadata(spec, config)

        spectra.append(spec)

    # Sort by minimum wavelength of each segment
    spectra.sort(key=lambda s: s.spectral_axis.value.min())

    # Warn about gaps or unexpected overlaps
    _check_segment_coverage(spectra)

    logger.info(
        "Loaded %d spectral segments from %s", len(spectra), directory
    )
    return spectra


def _check_segment_coverage(spectra):
    """Log warnings about gaps or overlaps between consecutive segments."""
    for i in range(len(spectra) - 1):
        max_i = spectra[i].spectral_axis.value.max()
        min_next = spectra[i + 1].spectral_axis.value.min()

        seg_i = spectra[i].meta.get("segment_id", i)
        seg_next = spectra[i + 1].meta.get("segment_id", i + 1)

        gap = min_next - max_i
        if gap > 50:  # > 50 Angstrom gap
            logger.warning(
                "Gap of %.1f A between segments %s and %s",
                gap, seg_i, seg_next,
            )
        elif gap < -200:  # large overlap (> 200 A)
            logger.info(
                "Overlap of %.1f A between segments %s and %s",
                abs(gap), seg_i, seg_next,
            )
