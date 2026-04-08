"""
calibrate.py — Flux Calibration Application
============================================
Step 7 of the JJMO Spectral Flux Calibration Pipeline.

Applies a derived sensitivity function S(λ) to observed spectra to produce
flux-calibrated spectra in physical units (erg/s/cm²/Å).  Handles:

- Per-segment and global sensitivity function application (§7.1)
- Differential atmospheric extinction correction (§7.2)
- Slit-loss documentation and flagging (§7.3)
- Self-calibration residual check (§7.4)
- FITS and CSV output with full provenance headers (§7.5)
- Batch-mode calibration of multiple science targets (§7.6)

Works with specutils Spectrum1D objects and numpy arrays.  Follows the
conventions established in io.py (Step 1) and stitching.py (Step 4).

Authors: JJMO Pipeline
"""

import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.interpolate import interp1d

from astropy.io import fits
from astropy.nddata import StdDevUncertainty
from astropy.table import Table
import astropy.units as u

try:
    from specutils import Spectrum as Spectrum1D  # specutils >= 2.3
except ImportError:
    from specutils import Spectrum1D  # specutils < 2.3

logger = logging.getLogger(__name__)

# Physical flux unit used throughout
FLUX_UNIT = u.erg / u.s / u.cm**2 / u.AA   # erg s⁻¹ cm⁻² Å⁻¹


# ==========================================================================
# Data structures
# ==========================================================================

@dataclass
class SensitivityFunction:
    """Container for a sensitivity function S(λ).

    The sensitivity function converts observed count rates to physical flux:
        F(λ) = C_obs(λ) / (t_exp × S(λ))

    Attributes
    ----------
    wavelength : np.ndarray
        Wavelength grid in Angstroms at which the sensitivity is sampled.
    sensitivity : np.ndarray
        Sensitivity values S(λ) in counts/s per (erg/s/cm²/Å), i.e. the
        instrumental throughput in counts per unit flux.
    uncertainty : np.ndarray or None
        1-sigma uncertainty on the sensitivity values.
    meta : dict
        Provenance metadata (standard star, fit method, extinction law, etc.).
    _interpolator : callable or None
        Cached interpolation function for evaluation at arbitrary wavelengths.
    """
    wavelength: np.ndarray
    sensitivity: np.ndarray
    uncertainty: Optional[np.ndarray] = None
    meta: Dict = field(default_factory=dict)
    _interpolator: Optional[Callable] = field(default=None, repr=False)

    def __post_init__(self):
        self.wavelength = np.asarray(self.wavelength, dtype=np.float64)
        self.sensitivity = np.asarray(self.sensitivity, dtype=np.float64)
        if self.uncertainty is not None:
            self.uncertainty = np.asarray(self.uncertainty, dtype=np.float64)
        self._build_interpolator()

    def _build_interpolator(self):
        """Build a linear interpolator with extrapolation fill=NaN."""
        valid = np.isfinite(self.sensitivity) & (self.sensitivity > 0)
        if np.sum(valid) < 2:
            raise ValueError(
                "Sensitivity function has fewer than 2 valid points; "
                "cannot build interpolator."
            )
        self._interpolator = interp1d(
            self.wavelength[valid], self.sensitivity[valid],
            kind="linear", bounds_error=False, fill_value=np.nan,
        )
        if self.uncertainty is not None:
            self._unc_interpolator = interp1d(
                self.wavelength[valid], self.uncertainty[valid],
                kind="linear", bounds_error=False, fill_value=np.nan,
            )
        else:
            self._unc_interpolator = None

    def evaluate(self, wavelength):
        """Evaluate S(λ) at arbitrary wavelengths.

        Parameters
        ----------
        wavelength : array-like
            Wavelength values in Angstroms.

        Returns
        -------
        sens : np.ndarray
            Sensitivity values.  NaN where extrapolation is needed.
        sens_unc : np.ndarray or None
            Uncertainty on the sensitivity, or None if unavailable.
        """
        wave = np.asarray(wavelength, dtype=np.float64)
        sens = self._interpolator(wave)
        sens_unc = None
        if self._unc_interpolator is not None:
            sens_unc = self._unc_interpolator(wave)
        return sens, sens_unc

    @classmethod
    def from_fits(cls, filepath):
        """Load a sensitivity function from a FITS table.

        Expected columns: WAVELENGTH, SENSITIVITY, and optionally
        SENSITIVITY_ERR.  Metadata is read from the header.
        """
        filepath = Path(filepath)
        with fits.open(filepath) as hdul:
            # Try binary table extension first, then primary
            if len(hdul) > 1 and isinstance(hdul[1], fits.BinTableHDU):
                tbl = hdul[1]
            else:
                tbl = hdul[0]

            data = tbl.data
            header = tbl.header

            wavelength = np.asarray(data["WAVELENGTH"], dtype=np.float64)
            sensitivity = np.asarray(data["SENSITIVITY"], dtype=np.float64)
            uncertainty = None
            if "SENSITIVITY_ERR" in data.dtype.names:
                uncertainty = np.asarray(data["SENSITIVITY_ERR"],
                                         dtype=np.float64)

        meta = dict(header)
        meta["source_file"] = str(filepath)
        return cls(wavelength=wavelength, sensitivity=sensitivity,
                   uncertainty=uncertainty, meta=meta)

    def to_fits(self, filepath, overwrite=True):
        """Write the sensitivity function to a FITS binary table."""
        filepath = Path(filepath)
        cols = [
            fits.Column(name="WAVELENGTH", format="D",
                        array=self.wavelength, unit="Angstrom"),
            fits.Column(name="SENSITIVITY", format="D",
                        array=self.sensitivity, unit="ct/s/(erg/s/cm2/A)"),
        ]
        if self.uncertainty is not None:
            cols.append(
                fits.Column(name="SENSITIVITY_ERR", format="D",
                            array=self.uncertainty,
                            unit="ct/s/(erg/s/cm2/A)")
            )
        hdu = fits.BinTableHDU.from_columns(cols)
        # Write provenance metadata
        for key, val in self.meta.items():
            if isinstance(val, str) and len(val) > 68:
                continue  # skip overly long values
            try:
                hdu.header[key[:8]] = val
            except (ValueError, TypeError):
                pass
        hdu.header["EXTNAME"] = "SENSITIVITY"
        primary = fits.PrimaryHDU()
        hdul = fits.HDUList([primary, hdu])
        hdul.writeto(filepath, overwrite=overwrite)
        logger.info("Wrote sensitivity function to %s", filepath)

    @classmethod
    def from_callable(cls, func, wavelength_grid, *, meta=None,
                      uncertainty_func=None):
        """Create a SensitivityFunction by evaluating a callable on a grid.

        Parameters
        ----------
        func : callable
            S(λ) function accepting a wavelength array.
        wavelength_grid : array-like
            Wavelength points to sample.
        meta : dict, optional
        uncertainty_func : callable, optional
            Function returning sensitivity uncertainty at given wavelengths.
        """
        wave = np.asarray(wavelength_grid, dtype=np.float64)
        sens = np.asarray(func(wave), dtype=np.float64)
        unc = None
        if uncertainty_func is not None:
            unc = np.asarray(uncertainty_func(wave), dtype=np.float64)
        return cls(wavelength=wave, sensitivity=sens, uncertainty=unc,
                   meta=meta or {})


@dataclass
class CalibrationResult:
    """Container for a flux-calibrated spectrum and diagnostics.

    Attributes
    ----------
    wavelength : np.ndarray
        Wavelength in Angstroms.
    flux : np.ndarray
        Calibrated flux in erg/s/cm²/Å.
    uncertainty : np.ndarray
        1-sigma uncertainty on the calibrated flux.
    mask : np.ndarray
        Boolean mask; True = bad/unreliable pixel.
    meta : dict
        Provenance metadata from the calibration.
    """
    wavelength: np.ndarray
    flux: np.ndarray
    uncertainty: np.ndarray
    mask: np.ndarray
    meta: Dict = field(default_factory=dict)

    def to_spectrum1d(self):
        """Convert to a specutils Spectrum1D with physical flux units."""
        spec_flux = self.flux * FLUX_UNIT
        unc = StdDevUncertainty(self.uncertainty * FLUX_UNIT)
        return Spectrum1D(
            spectral_axis=self.wavelength * u.AA,
            flux=spec_flux,
            uncertainty=unc,
            mask=self.mask,
            meta=self.meta,
        )


# ==========================================================================
# §7.1  Core calibration: apply sensitivity function to a spectrum
# ==========================================================================

def _unpack_observed(spectrum):
    """Extract arrays from a Spectrum1D, dict, or tuple.

    Returns
    -------
    wavelength, flux, uncertainty, mask : np.ndarray
        mask convention: True = bad pixel (specutils convention).
    meta : dict
    """
    if isinstance(spectrum, Spectrum1D):
        wave = spectrum.spectral_axis.to(u.AA).value
        flux = spectrum.flux.value
        if spectrum.uncertainty is not None:
            unc = spectrum.uncertainty.array
            # If uncertainty carries units, strip them
            if hasattr(unc, "value"):
                unc = unc.value
        else:
            unc = np.sqrt(np.maximum(np.abs(flux), 1.0))
        mask = spectrum.mask if spectrum.mask is not None else np.zeros(
            len(wave), dtype=bool)
        meta = dict(spectrum.meta) if spectrum.meta else {}
        return wave, flux, unc, mask, meta

    if isinstance(spectrum, dict):
        wave = np.asarray(spectrum["wavelength"], dtype=np.float64)
        flux = np.asarray(spectrum["flux"], dtype=np.float64)
        unc = np.asarray(
            spectrum.get("uncertainty",
                         np.sqrt(np.maximum(np.abs(flux), 1.0))),
            dtype=np.float64
        )
        mask = np.asarray(
            spectrum.get("mask", np.zeros(len(wave), dtype=bool)),
            dtype=bool
        )
        meta = dict(spectrum.get("meta", {}))
        return wave, flux, unc, mask, meta

    # tuple/list: (wave, flux[, unc[, mask]])
    parts = list(spectrum)
    wave = np.asarray(parts[0], dtype=np.float64)
    flux = np.asarray(parts[1], dtype=np.float64)
    unc = (np.asarray(parts[2], dtype=np.float64) if len(parts) >= 3
           else np.sqrt(np.maximum(np.abs(flux), 1.0)))
    mask = (np.asarray(parts[3], dtype=bool) if len(parts) >= 4
            else np.zeros(len(wave), dtype=bool))
    meta = {}
    return wave, flux, unc, mask, meta


def apply_sensitivity(observed, sensfunc, *, exptime=None, airmass_obs=None,
                      airmass_std=None, extinction_curve=None):
    """Apply a sensitivity function to produce flux-calibrated spectrum.

    Implements the calibration equation:

        F(λ) = C_obs(λ) / (t_exp × S(λ)) × 10^(0.4 × Δairmass × k(λ))

    where Δairmass = airmass_obs - airmass_std accounts for the difference
    between the science and standard-star observations.

    Parameters
    ----------
    observed : Spectrum1D, dict, or tuple
        Observed spectrum in instrumental counts.
    sensfunc : SensitivityFunction
        The sensitivity function derived from a standard star.
    exptime : float, optional
        Exposure time in seconds.  If None, read from spectrum metadata
        or default to 1.0 (count rate already normalised).
    airmass_obs : float, optional
        Airmass of the observation.  If None, read from metadata.
    airmass_std : float, optional
        Airmass at which the standard star was observed (from sensfunc
        metadata).  If None, read from sensfunc.meta.
    extinction_curve : callable or tuple, optional
        Atmospheric extinction k(λ) in mag/airmass.  Either a callable
        accepting wavelength in Angstroms, or a (wavelength, k) tuple
        for interpolation.  If None, no differential extinction is applied.

    Returns
    -------
    CalibrationResult
        Flux-calibrated spectrum with uncertainties and provenance metadata.
    """
    wave, counts, counts_unc, mask, meta = _unpack_observed(observed)

    # --- Resolve exposure time ---
    if exptime is None:
        exptime = meta.get("exptime")
    if exptime is None:
        warnings.warn(
            "Exposure time not provided and not found in metadata; "
            "assuming counts are already in counts/s (exptime=1).",
            UserWarning
        )
        exptime = 1.0
    exptime = float(exptime)
    if exptime <= 0:
        raise ValueError(f"Exposure time must be positive, got {exptime}")

    # Convert to count rate
    count_rate = counts / exptime
    count_rate_unc = counts_unc / exptime

    # --- Evaluate sensitivity function at observed wavelengths ---
    sens, sens_unc = sensfunc.evaluate(wave)

    # Flag pixels where sensitivity is invalid (NaN, zero, or negative)
    bad_sens = ~np.isfinite(sens) | (sens <= 0)
    if np.any(bad_sens):
        n_bad = int(np.sum(bad_sens))
        logger.warning(
            "%d pixels have invalid sensitivity (out of %d); flagging as bad.",
            n_bad, len(wave)
        )
        mask = mask | bad_sens
        # Set safe values so the division doesn't produce warnings
        sens[bad_sens] = 1.0
        if sens_unc is not None:
            sens_unc[bad_sens] = np.nan

    # --- Apply differential atmospheric extinction correction (§7.2) ---
    extinction_factor = _compute_extinction_factor(
        wave, airmass_obs=airmass_obs, airmass_std=airmass_std,
        extinction_curve=extinction_curve, meta=meta,
        sensfunc_meta=sensfunc.meta
    )

    # --- Core calibration: divide count rate by sensitivity ---
    flux_cal = (count_rate / sens) * extinction_factor

    # --- Propagate uncertainties (§7.1, quadrature sum of fractional errors) ---
    # F = (C/t) / S * E
    # (σ_F/F)² = (σ_C/C)² + (σ_S/S)²
    # The extinction factor uncertainty is a systematic handled in Step 8.
    frac_counts = np.zeros_like(flux_cal)
    frac_sens = np.zeros_like(flux_cal)

    good = ~mask & (count_rate != 0)
    frac_counts[good] = (count_rate_unc[good] / np.abs(count_rate[good]))**2

    if sens_unc is not None:
        good_s = ~mask & (sens > 0)
        frac_sens[good_s] = (sens_unc[good_s] / sens[good_s])**2

    flux_unc = np.abs(flux_cal) * np.sqrt(frac_counts + frac_sens)

    # Set masked pixels to NaN for clarity
    flux_cal[mask] = np.nan
    flux_unc[mask] = np.nan

    # --- Build provenance metadata ---
    cal_meta = dict(meta)
    cal_meta["flux_unit"] = "erg/s/cm2/A"
    cal_meta["wavelength_unit"] = "Angstrom"
    cal_meta["exptime_applied"] = exptime
    cal_meta["sensfunc_source"] = sensfunc.meta.get("source_file", "unknown")
    cal_meta["standard_star"] = sensfunc.meta.get("standard_star", "unknown")
    cal_meta["extinction_law"] = sensfunc.meta.get("extinction_law", "none")
    cal_meta["airmass_obs"] = airmass_obs
    cal_meta["airmass_std"] = airmass_std or sensfunc.meta.get("airmass_std")
    cal_meta["extinction_applied"] = extinction_factor is not None
    cal_meta["calibration_step"] = "step_07_flux_calibration"
    # Flag slit-loss systematic (§7.3)
    cal_meta["slit_loss_corrected"] = False
    cal_meta["slit_loss_warning"] = (
        "No slit-loss correction applied. If the standard and science "
        "targets were observed with the same slit/fiber setup, slit losses "
        "partially cancel. A residual grey offset and wavelength-dependent "
        "term may remain. See §7.3 documentation."
    )

    return CalibrationResult(
        wavelength=wave, flux=flux_cal, uncertainty=flux_unc,
        mask=mask, meta=cal_meta,
    )


def apply_sensitivity_per_segment(segments, sensfunc, **kwargs):
    """Apply the sensitivity function to each segment independently.

    This is the recommended workflow: calibrate each ~500 Å segment before
    stitching, so that any per-segment sensitivity variations are handled
    correctly.

    Parameters
    ----------
    segments : list of Spectrum1D, dict, or tuple
        Individual spectral segments in instrumental counts.
    sensfunc : SensitivityFunction
        A single (global) or per-segment sensitivity function.
    **kwargs
        Forwarded to ``apply_sensitivity`` (exptime, airmass, etc.).

    Returns
    -------
    list of CalibrationResult
    """
    results = []
    for i, seg in enumerate(segments):
        logger.info("Calibrating segment %d/%d", i + 1, len(segments))
        result = apply_sensitivity(seg, sensfunc, **kwargs)
        result.meta["segment_index"] = i
        results.append(result)
    return results


# ==========================================================================
# §7.2  Atmospheric extinction correction
# ==========================================================================

# Default mean atmospheric extinction curve (mag/airmass) for a generic
# mid-latitude observatory.  Values from Hayes & Latham (1975) and
# Straizhys & Sviderskene (1972), linearly interpolated.  This is a
# coarse fallback; users should supply their own site-specific curve.
_DEFAULT_EXTINCTION_WAVE = np.array([
    3200., 3400., 3600., 3800., 4000., 4200., 4400., 4600., 4800.,
    5000., 5200., 5400., 5600., 5800., 6000., 6200., 6400., 6600.,
    6800., 7000., 7200., 7400., 7600., 7800., 8000., 8500., 9000.,
])
_DEFAULT_EXTINCTION_K = np.array([
    1.084, 0.756, 0.552, 0.420, 0.332, 0.272, 0.228, 0.195, 0.170,
    0.149, 0.134, 0.121, 0.110, 0.101, 0.093, 0.087, 0.081, 0.077,
    0.074, 0.071, 0.069, 0.068, 0.067, 0.066, 0.065, 0.063, 0.061,
])
_DEFAULT_EXTINCTION_INTERP = interp1d(
    _DEFAULT_EXTINCTION_WAVE, _DEFAULT_EXTINCTION_K,
    kind="linear", bounds_error=False,
    fill_value=(_DEFAULT_EXTINCTION_K[0], _DEFAULT_EXTINCTION_K[-1]),
)


def default_extinction_curve(wavelength):
    """Evaluate the default atmospheric extinction k(λ) in mag/airmass.

    Parameters
    ----------
    wavelength : array-like
        Wavelength in Angstroms.

    Returns
    -------
    k : np.ndarray
        Extinction coefficient in magnitudes per airmass.
    """
    return _DEFAULT_EXTINCTION_INTERP(np.asarray(wavelength, dtype=np.float64))


def _resolve_extinction_curve(extinction_curve):
    """Normalise the extinction_curve argument to a callable.

    Accepts:
    - None → default extinction curve
    - callable → used directly
    - (wavelength, k) tuple → interpolated
    """
    if extinction_curve is None:
        return default_extinction_curve

    if callable(extinction_curve):
        return extinction_curve

    # Assume (wavelength, k) pair
    try:
        wave_ext, k_ext = extinction_curve
        wave_ext = np.asarray(wave_ext, dtype=np.float64)
        k_ext = np.asarray(k_ext, dtype=np.float64)
        return interp1d(wave_ext, k_ext, kind="linear",
                        bounds_error=False,
                        fill_value=(k_ext[0], k_ext[-1]))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "extinction_curve must be None, a callable, or a "
            "(wavelength, k) tuple of arrays."
        ) from exc


def _compute_extinction_factor(wavelength, *, airmass_obs, airmass_std,
                               extinction_curve, meta, sensfunc_meta):
    """Compute the differential extinction correction factor.

    The sensitivity function was derived from a standard star observed at
    airmass_std.  When applying to a science target at airmass_obs, we
    correct for the difference:

        factor = 10^(0.4 × (airmass_obs - airmass_std) × k(λ))

    This removes the standard's extinction from S(λ) and applies the
    science target's extinction.

    Returns
    -------
    factor : np.ndarray
        Multiplicative correction factor (1.0 if no correction applied).
    """
    # Resolve airmass values from arguments or metadata
    if airmass_obs is None:
        airmass_obs = meta.get("airmass")
    if airmass_std is None:
        airmass_std = sensfunc_meta.get("airmass_std",
                                        sensfunc_meta.get("airmass"))

    ones = np.ones(len(wavelength), dtype=np.float64)

    if airmass_obs is None and airmass_std is None:
        logger.info(
            "No airmass information available; skipping extinction correction."
        )
        return ones

    if airmass_obs is None:
        warnings.warn(
            "Science airmass unknown; cannot apply differential extinction "
            "correction. The calibrated flux will be correct only if the "
            "science and standard observations were at the same airmass.",
            UserWarning
        )
        return ones

    if airmass_std is None:
        warnings.warn(
            "Standard-star airmass unknown; cannot apply differential "
            "extinction correction. Using science airmass alone.",
            UserWarning
        )
        return ones

    airmass_obs = float(airmass_obs)
    airmass_std = float(airmass_std)
    delta_airmass = airmass_obs - airmass_std

    if abs(delta_airmass) < 1e-6:
        logger.info(
            "Science and standard airmass are equal (%.4f); "
            "no differential extinction needed.", airmass_obs
        )
        return ones

    k_func = _resolve_extinction_curve(extinction_curve)
    k_values = k_func(wavelength)

    factor = 10.0 ** (0.4 * delta_airmass * k_values)

    logger.info(
        "Applied differential extinction: Δairmass=%.4f, "
        "correction range [%.4f, %.4f]",
        delta_airmass, np.nanmin(factor), np.nanmax(factor)
    )

    return factor


# ==========================================================================
# §7.3  Slit-loss and aperture corrections (documentation / flagging)
# ==========================================================================

SLIT_LOSS_DOCUMENTATION = """
Slit-Loss and Aperture Correction Notes for JJMO Spectrograph
=============================================================

Overview
--------
Slit losses occur when the point-spread function (PSF) of the star is wider
than the slit or fiber aperture, causing a fraction of the light to be
rejected.  This introduces:

  1. A **grey (wavelength-independent) offset**: the total lost flux fraction
     when the PSF is symmetric and smaller than the slit at all wavelengths.

  2. A **chromatic (wavelength-dependent) term**: seeing typically improves
     at longer wavelengths (FWHM ∝ λ^{-1/5} for Kolmogorov turbulence),
     so less light is lost in the red than in the blue.  If the seeing disc
     approaches or exceeds the slit width at short wavelengths, this effect
     steepens the observed spectral slope relative to the true spectrum.

Cancellation for Matched Observations
--------------------------------------
If the standard star and the science target are observed through the **same
slit/fiber** under **similar seeing** conditions, both spectra suffer the
same wavelength-dependent losses.  In the ratio C_obs(standard) / C_obs(science),
the slit losses largely cancel.  The residual is second-order: differences
in airmass-dependent seeing, guiding errors, or focus drift between the two
observations.

JJMO Spectrograph Specifics
----------------------------
The JJMO spectrograph uses a **fixed fiber-fed input** (no adjustable slit).
Because the fiber aperture is constant, slit losses are dominated by the
seeing-to-fiber ratio and pointing accuracy.  For the existing dataset:

- The fiber diameter is large enough that most starlight is captured under
  typical seeing conditions (>2 arcsec seeing, ~3 arcsec fiber).
- No automatic differential correction is attempted.
- The slit-loss contribution to the error budget is estimated to be a
  ~5-15% grey offset (absorbed into the sensitivity function) plus a
  ~1-5% chromatic term (not corrected, treated as a systematic).

Users performing precision photometry should obtain photometric measurements
alongside spectroscopy to anchor the absolute flux scale independently of
slit losses.

Implementation Status
---------------------
- **No automatic slit-loss correction is applied** in this pipeline.
- The calibration metadata includes ``slit_loss_corrected = False`` and a
  warning string in ``slit_loss_warning`` for downstream consumers.
- Future work could implement an empirical correction by comparing the
  integrated calibrated spectrum to broadband photometry (synthetic
  photometry check).
"""


def get_slit_loss_documentation():
    """Return the slit-loss and aperture correction documentation string.

    This documents known systematics from slit/fiber losses that are NOT
    automatically corrected in the pipeline.  See §7.3 in the spec.
    """
    return SLIT_LOSS_DOCUMENTATION


def flag_slit_loss_systematic(result):
    """Ensure a CalibrationResult carries slit-loss warnings.

    Parameters
    ----------
    result : CalibrationResult
        Modified in-place to include slit-loss metadata flags.
    """
    result.meta.setdefault("slit_loss_corrected", False)
    result.meta.setdefault("slit_loss_warning",
                           "No slit-loss correction applied. See §7.3.")


# ==========================================================================
# §7.4  Self-calibration check
# ==========================================================================

@dataclass
class SelfCalibrationReport:
    """Results of the self-calibration sanity check.

    Applying the sensitivity function to the standard star itself should
    recover the reference spectrum to within the noise.

    Attributes
    ----------
    wavelength : np.ndarray
        Common wavelength grid.
    residual_frac : np.ndarray
        Fractional residual (calibrated - reference) / reference.
    rms_residual : float
        RMS of the fractional residual over unmasked pixels.
    median_residual : float
        Median absolute fractional residual.
    max_residual : float
        Maximum absolute fractional residual.
    mask : np.ndarray
        True where the comparison is unreliable (masked pixels).
    """
    wavelength: np.ndarray
    residual_frac: np.ndarray
    rms_residual: float
    median_residual: float
    max_residual: float
    mask: np.ndarray


def self_calibration_check(observed_standard, sensfunc, reference_flux,
                           reference_wavelength=None, *,
                           exptime=None, airmass_obs=None, airmass_std=None,
                           extinction_curve=None):
    """Apply the sensitivity function to the standard star and compare
    to the known reference spectrum.

    This is the primary quality metric for the calibration (§7.4).

    Parameters
    ----------
    observed_standard : Spectrum1D, dict, or tuple
        The standard star observation in instrumental counts.
    sensfunc : SensitivityFunction
        The sensitivity function derived from this standard.
    reference_flux : array-like or Spectrum1D
        The true reference flux (erg/s/cm²/Å).  If a Spectrum1D, the
        spectral_axis and flux are extracted automatically.
    reference_wavelength : array-like, optional
        Wavelength grid for reference_flux (Angstroms).  Required if
        reference_flux is a plain array; ignored if reference_flux is
        Spectrum1D.
    exptime, airmass_obs, airmass_std, extinction_curve
        Forwarded to ``apply_sensitivity``.

    Returns
    -------
    SelfCalibrationReport
    """
    # Calibrate the standard star
    cal = apply_sensitivity(
        observed_standard, sensfunc,
        exptime=exptime, airmass_obs=airmass_obs, airmass_std=airmass_std,
        extinction_curve=extinction_curve,
    )

    # Extract reference spectrum
    if isinstance(reference_flux, Spectrum1D):
        ref_wave = reference_flux.spectral_axis.to(u.AA).value
        ref_flux = reference_flux.flux.value
        # Handle units: if flux has physical units, strip them
        if hasattr(reference_flux.flux, "unit") and reference_flux.flux.unit is not None:
            ref_flux = reference_flux.flux.to(FLUX_UNIT).value
    else:
        ref_flux = np.asarray(reference_flux, dtype=np.float64)
        if reference_wavelength is None:
            raise ValueError(
                "reference_wavelength required when reference_flux is an array"
            )
        ref_wave = np.asarray(reference_wavelength, dtype=np.float64)

    # Interpolate reference to the calibrated wavelength grid
    ref_interp = interp1d(ref_wave, ref_flux, kind="linear",
                          bounds_error=False, fill_value=np.nan)
    ref_on_grid = ref_interp(cal.wavelength)

    # Compute fractional residuals where both are valid
    combined_mask = (cal.mask
                     | ~np.isfinite(ref_on_grid)
                     | ~np.isfinite(cal.flux)
                     | (ref_on_grid <= 0))

    residual_frac = np.full_like(cal.flux, np.nan)
    good = ~combined_mask
    residual_frac[good] = (cal.flux[good] - ref_on_grid[good]) / ref_on_grid[good]

    # Summary statistics
    if np.any(good):
        rms = float(np.sqrt(np.nanmean(residual_frac[good]**2)))
        med = float(np.nanmedian(np.abs(residual_frac[good])))
        mx = float(np.nanmax(np.abs(residual_frac[good])))
    else:
        rms = med = mx = np.nan
        warnings.warn(
            "Self-calibration check: no valid pixels for comparison.",
            UserWarning
        )

    logger.info(
        "Self-calibration check: RMS=%.4f, median=%.4f, max=%.4f "
        "(%d/%d valid pixels)",
        rms, med, mx, int(np.sum(good)), len(cal.wavelength)
    )

    return SelfCalibrationReport(
        wavelength=cal.wavelength,
        residual_frac=residual_frac,
        rms_residual=rms,
        median_residual=med,
        max_residual=mx,
        mask=combined_mask,
    )


# ==========================================================================
# §7.5  Output calibrated spectra (FITS + CSV)
# ==========================================================================

def write_calibrated_fits(result, filepath, *, overwrite=True):
    """Write a flux-calibrated spectrum to a FITS file.

    Produces a FITS file with:
    - Primary HDU: calibrated flux array with WCS wavelength keywords.
    - Extension 1 (UNCERTAINTY): uncertainty array.
    - Extension 2 (MASK): pixel mask (1 = bad).
    - Extension 3 (WAVELENGTH): explicit wavelength array.

    Header keywords document the full calibration provenance.

    Parameters
    ----------
    result : CalibrationResult
        The calibrated spectrum.
    filepath : str or Path
        Output file path.
    overwrite : bool
        Overwrite existing file.
    """
    filepath = Path(filepath)
    wave = result.wavelength
    flux = result.flux
    unc = result.uncertainty
    mask = result.mask.astype(np.uint8)

    # --- Primary HDU: flux with WCS ---
    hdr = fits.Header()

    # WCS for wavelength (assumes uniform grid; if not, we provide the
    # explicit WAVELENGTH extension)
    if len(wave) >= 2:
        # Check if the grid is approximately uniform
        dw = np.diff(wave)
        if np.allclose(dw, dw[0], rtol=1e-4):
            hdr["CRVAL1"] = float(wave[0])
            hdr["CDELT1"] = float(dw[0])
            hdr["CRPIX1"] = 1.0
            hdr["CTYPE1"] = "WAVE"
            hdr["CUNIT1"] = "Angstrom"
        else:
            hdr["COMMENT"] = ("Wavelength grid is non-uniform; "
                              "see WAVELENGTH extension.")

    # Flux metadata
    hdr["BUNIT"] = "erg/s/cm2/A"
    hdr["BTYPE"] = "FLUX"

    # Provenance keywords from calibration metadata
    _prov_keys = [
        ("EXPTIME", "exptime_applied", "Exposure time [s]"),
        ("SENSFILE", "sensfunc_source", "Sensitivity function file"),
        ("STDSTAR", "standard_star", "Standard star used"),
        ("EXTLAW", "extinction_law", "Extinction law applied"),
        ("AIRMOBS", "airmass_obs", "Science observation airmass"),
        ("AIRMSTD", "airmass_std", "Standard star airmass"),
        ("SLITCOR", "slit_loss_corrected", "Slit-loss correction applied?"),
        ("CALSTEP", "calibration_step", "Pipeline step identifier"),
        ("FLXUNIT", "flux_unit", "Flux unit"),
        ("WAVUNIT", "wavelength_unit", "Wavelength unit"),
    ]
    for fits_key, meta_key, comment in _prov_keys:
        val = result.meta.get(meta_key)
        if val is not None:
            try:
                hdr[fits_key] = (val, comment)
            except (ValueError, TypeError):
                hdr[fits_key] = (str(val)[:68], comment)

    primary = fits.PrimaryHDU(data=flux, header=hdr)

    # --- Uncertainty extension ---
    unc_hdr = fits.Header()
    unc_hdr["BUNIT"] = "erg/s/cm2/A"
    unc_hdr["BTYPE"] = "FLUX_ERR"
    unc_hdr["EXTNAME"] = "UNCERTAINTY"
    unc_hdu = fits.ImageHDU(data=unc, header=unc_hdr)

    # --- Mask extension ---
    mask_hdr = fits.Header()
    mask_hdr["EXTNAME"] = "MASK"
    mask_hdr["COMMENT"] = "Pixel mask: 1 = bad/unreliable, 0 = good"
    mask_hdu = fits.ImageHDU(data=mask, header=mask_hdr)

    # --- Explicit wavelength extension ---
    wave_hdr = fits.Header()
    wave_hdr["BUNIT"] = "Angstrom"
    wave_hdr["BTYPE"] = "WAVELENGTH"
    wave_hdr["EXTNAME"] = "WAVELENGTH"
    wave_hdu = fits.ImageHDU(data=wave, header=wave_hdr)

    hdul = fits.HDUList([primary, unc_hdu, mask_hdu, wave_hdu])
    hdul.writeto(filepath, overwrite=overwrite)
    logger.info("Wrote calibrated FITS to %s", filepath)


def write_calibrated_csv(result, filepath, *, delimiter=","):
    """Write a flux-calibrated spectrum to a CSV/ASCII table.

    Columns: wavelength [Angstrom], flux [erg/s/cm²/Å], uncertainty, mask.

    Parameters
    ----------
    result : CalibrationResult
        The calibrated spectrum.
    filepath : str or Path
        Output file path.
    delimiter : str
        Column delimiter (default: comma).
    """
    filepath = Path(filepath)

    # Build header comment block with provenance
    header_lines = [
        "# Flux-calibrated spectrum from JJMO pipeline (step_07)",
        f"# Standard star: {result.meta.get('standard_star', 'unknown')}",
        f"# Sensitivity source: {result.meta.get('sensfunc_source', 'unknown')}",
        f"# Exposure time: {result.meta.get('exptime_applied', 'unknown')} s",
        f"# Airmass (obs): {result.meta.get('airmass_obs', 'unknown')}",
        f"# Airmass (std): {result.meta.get('airmass_std', 'unknown')}",
        f"# Flux unit: erg/s/cm2/A",
        f"# Wavelength unit: Angstrom",
        f"# Columns: wavelength{delimiter}flux{delimiter}uncertainty{delimiter}mask",
    ]
    header_text = "\n".join(header_lines)

    data = np.column_stack([
        result.wavelength,
        result.flux,
        result.uncertainty,
        result.mask.astype(np.int32),
    ])

    np.savetxt(filepath, data, delimiter=delimiter, header=header_text,
               fmt=["%.4f", "%.8e", "%.8e", "%d"],
               comments="")
    logger.info("Wrote calibrated CSV to %s", filepath)


def read_calibrated_fits(filepath):
    """Read a flux-calibrated spectrum written by ``write_calibrated_fits``.

    Parameters
    ----------
    filepath : str or Path
        Path to the FITS file.

    Returns
    -------
    CalibrationResult
    """
    filepath = Path(filepath)
    with fits.open(filepath) as hdul:
        flux = np.asarray(hdul[0].data, dtype=np.float64)
        header = hdul[0].header

        # Read explicit wavelength if available
        wave = None
        unc = None
        mask = None
        for ext in hdul[1:]:
            name = ext.header.get("EXTNAME", "")
            if name == "WAVELENGTH":
                wave = np.asarray(ext.data, dtype=np.float64)
            elif name == "UNCERTAINTY":
                unc = np.asarray(ext.data, dtype=np.float64)
            elif name == "MASK":
                mask = np.asarray(ext.data, dtype=bool)

        # Fallback: reconstruct wavelength from WCS
        if wave is None:
            crval1 = header.get("CRVAL1")
            cdelt1 = header.get("CDELT1")
            crpix1 = header.get("CRPIX1", 1.0)
            if crval1 is not None and cdelt1 is not None:
                pix = np.arange(1, len(flux) + 1, dtype=np.float64)
                wave = crval1 + (pix - crpix1) * cdelt1
            else:
                raise ValueError(
                    f"Cannot determine wavelength from {filepath}; "
                    "no WAVELENGTH extension and no WCS keywords."
                )

        if unc is None:
            unc = np.full_like(flux, np.nan)
        if mask is None:
            mask = np.zeros(len(flux), dtype=bool)

    meta = dict(header)
    meta["source_file"] = str(filepath)

    return CalibrationResult(
        wavelength=wave, flux=flux, uncertainty=unc, mask=mask, meta=meta
    )


# ==========================================================================
# §7.6  Batch mode
# ==========================================================================

def calibrate_batch(spectra, sensfunc, *, exptimes=None, airmasses=None,
                    airmass_std=None, extinction_curve=None,
                    output_dir=None, output_format="both",
                    filename_template="calibrated_{index:03d}"):
    """Calibrate a list of science spectra in batch mode.

    Supports different airmasses and exposure times per spectrum.

    Parameters
    ----------
    spectra : list
        Science spectra (Spectrum1D, dict, or tuple).
    sensfunc : SensitivityFunction
        The sensitivity function to apply.
    exptimes : list of float, optional
        Per-spectrum exposure times.  If None, read from metadata.
    airmasses : list of float, optional
        Per-spectrum airmasses.  If None, read from metadata.
    airmass_std : float, optional
        Standard-star airmass (from sensfunc metadata if not given).
    extinction_curve : callable or tuple, optional
        Atmospheric extinction k(λ).
    output_dir : str or Path, optional
        If given, write calibrated spectra to this directory.
    output_format : str
        One of "fits", "csv", or "both" (default: "both").
    filename_template : str
        Template for output filenames.  Must contain ``{index}``.

    Returns
    -------
    list of CalibrationResult
    """
    n = len(spectra)
    if exptimes is not None and len(exptimes) != n:
        raise ValueError(
            f"Length of exptimes ({len(exptimes)}) does not match "
            f"number of spectra ({n})."
        )
    if airmasses is not None and len(airmasses) != n:
        raise ValueError(
            f"Length of airmasses ({len(airmasses)}) does not match "
            f"number of spectra ({n})."
        )

    results = []
    for i, spec in enumerate(spectra):
        logger.info("Batch calibrating spectrum %d/%d", i + 1, n)
        et = exptimes[i] if exptimes is not None else None
        am = airmasses[i] if airmasses is not None else None

        cal = apply_sensitivity(
            spec, sensfunc,
            exptime=et, airmass_obs=am, airmass_std=airmass_std,
            extinction_curve=extinction_curve,
        )
        cal.meta["batch_index"] = i
        results.append(cal)

        # Write output if requested
        if output_dir is not None:
            outdir = Path(output_dir)
            outdir.mkdir(parents=True, exist_ok=True)
            base = filename_template.format(index=i)

            if output_format in ("fits", "both"):
                write_calibrated_fits(cal, outdir / f"{base}.fits")
            if output_format in ("csv", "both"):
                write_calibrated_csv(cal, outdir / f"{base}.csv")

    logger.info("Batch calibration complete: %d spectra processed.", n)
    return results


# ==========================================================================
# Diagnostic plotting
# ==========================================================================

def plot_calibrated_spectrum(result, *, ax=None, show_uncertainty=True,
                             title=None, wavelength_range=None):
    """Plot a calibrated spectrum with optional uncertainty band.

    Parameters
    ----------
    result : CalibrationResult
        The calibrated spectrum.
    ax : matplotlib Axes, optional
        If None, creates a new figure.
    show_uncertainty : bool
        Show the ±1σ uncertainty shaded band.
    title : str, optional
    wavelength_range : tuple of (min, max), optional
        Restrict wavelength display range.

    Returns
    -------
    ax : matplotlib Axes
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 5))

    wave = result.wavelength
    flux = result.flux
    unc = result.uncertainty
    good = ~result.mask

    if wavelength_range is not None:
        sel = good & (wave >= wavelength_range[0]) & (wave <= wavelength_range[1])
    else:
        sel = good

    ax.plot(wave[sel], flux[sel], "k-", linewidth=0.8, label="Calibrated flux")

    if show_uncertainty and unc is not None:
        ax.fill_between(
            wave[sel],
            flux[sel] - unc[sel],
            flux[sel] + unc[sel],
            alpha=0.25, color="steelblue", label="±1σ uncertainty"
        )

    ax.set_xlabel("Wavelength [Å]")
    ax.set_ylabel("Flux [erg/s/cm²/Å]")
    ax.set_title(title or "Flux-Calibrated Spectrum")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    return ax


def plot_self_calibration(report, *, ax=None, title=None):
    """Plot the self-calibration residuals.

    Parameters
    ----------
    report : SelfCalibrationReport
    ax : matplotlib Axes, optional
    title : str, optional

    Returns
    -------
    ax : matplotlib Axes
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 4))

    good = ~report.mask
    wave = report.wavelength[good]
    resid = report.residual_frac[good] * 100  # convert to percent

    ax.plot(wave, resid, "k.", markersize=2, alpha=0.5)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)

    # Show ±RMS band
    rms_pct = report.rms_residual * 100
    ax.axhline(rms_pct, color="red", linestyle=":", linewidth=0.8,
               label=f"+RMS = {rms_pct:.2f}%")
    ax.axhline(-rms_pct, color="red", linestyle=":", linewidth=0.8,
               label=f"-RMS = {rms_pct:.2f}%")

    ax.set_xlabel("Wavelength [Å]")
    ax.set_ylabel("Residual [(cal - ref) / ref] (%)")
    ax.set_title(
        title or
        f"Self-Calibration Check (RMS={rms_pct:.2f}%, "
        f"median={report.median_residual*100:.2f}%)"
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    return ax
