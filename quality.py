"""
quality.py — Segment Quality Assessment & Bad Region Rejection

Step 3 of the JJMO Spectral Flux Calibration Pipeline.

Provides automated quality assessment for noisy, segmented spectra from
educational observatory spectrographs. Identifies and masks:
  - Edge artifacts (vignetting / grating rolloff)
  - Cosmic rays and hot pixels (sigma-clipping)
  - Telluric absorption bands
  - Stellar absorption lines (for sensitivity function fitting)
  - Low-SNR regions and segments

All masking functions return boolean arrays where True = GOOD (unmasked) pixel
and False = BAD (masked) pixel, following the numpy masked array convention
where the mask marks *invalid* data when inverted.

Usage
-----
    from quality import assess_segment, QualityReport

    wavelength, flux = ...  # numpy arrays
    report = assess_segment(wavelength, flux)
    # report.mask_good   — combined boolean mask (True = good pixel)
    # report.snr_median  — median SNR in unmasked region
    # report.summary()   — print a human-readable summary
"""

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter
from dataclasses import dataclass, field
from typing import Optional
import warnings


# ---------------------------------------------------------------------------
# Line lists (mirrored from JJMO_functions.py for standalone use)
# ---------------------------------------------------------------------------

BALMER_LINES = np.array([
    6562.8,  # Hα
    4861.3,  # Hβ
    4340.5,  # Hγ
    4101.7,  # Hδ
    3970.1,  # Hε
    3889.1,  # Hζ
    3835.4,  # Hη
])

METAL_LINES = np.array([
    3933.7,  # Ca II K
    3968.5,  # Ca II H
    4481.2,  # Mg II
    4383.6,  # Fe I
    4923.9,  # Fe II
    5018.4,  # Fe II
    5169.0,  # Fe II
    4552.6,  # Si III
    4077.7,  # Sr II
])

# Default masking half-widths (Angstroms) for each line category
BALMER_HALF_WIDTH = 15.0  # broad lines
METAL_HALF_WIDTH = 5.0    # narrow metal lines

# ---------------------------------------------------------------------------
# Telluric absorption bands: (lower_wavelength, upper_wavelength) in Angstroms
# From spec: O2 A-band, O2 B-band, H2O bands
# ---------------------------------------------------------------------------

TELLURIC_BANDS = [
    (6270.0, 6290.0),   # O2 A-band
    (6860.0, 6880.0),   # O2 B-band (partial)
    (7150.0, 7300.0),   # H2O band
    (7590.0, 7650.0),   # O2 B-band (deep)
    (8100.0, 8400.0),   # H2O band (far red, beyond JJMO range but included)
]


# ===================================================================
# 3.1  Edge Trimming
# ===================================================================

def trim_edges(wavelength, flux, threshold_frac=0.20, min_run=5):
    """Detect where flux drops below a fraction of the segment median,
    indicating vignetting or grating rolloff at segment edges.

    Parameters
    ----------
    wavelength : array-like
        Wavelength array (Angstroms). Must be monotonically increasing.
    flux : array-like
        Flux array (counts or counts/s).
    threshold_frac : float
        Fraction of the segment's median flux used as the cutoff.
        Pixels below ``median_flux * threshold_frac`` at the edges are trimmed.
    min_run : int
        Minimum number of consecutive good pixels required before the edge
        is considered to start. This prevents a single noisy spike from
        fooling the trimmer.

    Returns
    -------
    mask : ndarray of bool
        True for pixels within the good wavelength range, False for trimmed edges.
    trim_blue : float
        Wavelength at which the blue (low-wavelength) trim begins.
    trim_red : float
        Wavelength at which the red (high-wavelength) trim ends.
    """
    flux = np.asarray(flux, dtype=float)
    wavelength = np.asarray(wavelength, dtype=float)
    n = len(flux)

    # Use a lightly smoothed version to avoid noise-driven false edges
    kernel = max(5, n // 50)
    if kernel % 2 == 0:
        kernel += 1
    flux_smooth = median_filter(flux, size=kernel)

    # Threshold based on the interior median (middle 60% of pixels)
    interior = flux_smooth[n // 5 : 4 * n // 5]
    med = np.median(interior)
    threshold = med * threshold_frac

    above = flux_smooth >= threshold

    # Find first index where min_run consecutive pixels are above threshold
    left = 0
    for i in range(n - min_run + 1):
        if np.all(above[i : i + min_run]):
            left = i
            break

    # Find last such index from the right
    right = n - 1
    for i in range(n - 1, min_run - 2, -1):
        if np.all(above[i - min_run + 1 : i + 1]):
            right = i
            break

    mask = np.zeros(n, dtype=bool)
    mask[left : right + 1] = True

    trim_blue = wavelength[left]
    trim_red = wavelength[right]

    return mask, trim_blue, trim_red


# ===================================================================
# 3.2  Cosmic Ray / Hot Pixel Detection
# ===================================================================

def detect_cosmics_1d(flux, sigma_thresh=5.0, window_size=21):
    """Detect cosmic rays and hot pixels in a 1D spectrum using
    sigma-clipping against a local running median.

    Parameters
    ----------
    flux : array-like
        1D flux array.
    sigma_thresh : float
        Number of standard deviations above the local median to flag
        a pixel as a cosmic ray hit.
    window_size : int
        Size of the running median window (pixels). Must be odd.

    Returns
    -------
    mask : ndarray of bool
        True = good pixel, False = cosmic ray / hot pixel.
    n_flagged : int
        Number of pixels flagged.
    """
    flux = np.asarray(flux, dtype=float)
    if window_size % 2 == 0:
        window_size += 1

    # Running median as the local continuum estimate
    local_median = median_filter(flux, size=window_size)
    residual = flux - local_median

    # Use a LOCAL noise estimate (running MAD) rather than a global one.
    # In structured spectral regions (e.g., near the Balmer limit), a global
    # MAD is biased high by systematic residuals from unresolved features,
    # causing over-flagging. A local MAD adapts to the per-region noise.
    abs_residual = np.abs(residual)
    local_noise_window = window_size * 3  # wider window for noise estimate
    if local_noise_window % 2 == 0:
        local_noise_window += 1
    local_mad = median_filter(abs_residual, size=local_noise_window)
    local_sigma = local_mad * 1.4826

    # Fall back to global MAD where local estimate is zero (constant regions)
    global_mad = np.median(abs_residual)
    global_sigma = global_mad * 1.4826
    if global_sigma < 1e-10:
        return np.ones(len(flux), dtype=bool), 0
    local_sigma = np.where(local_sigma > 1e-10, local_sigma, global_sigma)

    # Flag positive outliers only (cosmic rays are bright spikes).
    # Also require the spike to be narrow: check that neighboring pixels
    # are NOT similarly elevated, which would indicate a spectral feature.
    bad = residual > sigma_thresh * local_sigma

    # Reject candidates where both neighbors are also above 2-sigma
    # (likely a broad spectral feature, not a cosmic ray)
    neighbor_thresh = 2.0
    for i in np.where(bad)[0]:
        if i > 0 and i < len(flux) - 1:
            left_high = residual[i - 1] > neighbor_thresh * local_sigma[i - 1]
            right_high = residual[i + 1] > neighbor_thresh * local_sigma[i + 1]
            if left_high and right_high:
                bad[i] = False

    mask = ~bad
    return mask, int(np.sum(bad))


def detect_cosmics_2d(image, sigma_thresh=5.0, gain=1.0, readnoise=5.0):
    """Detect cosmic rays in a 2D CCD image using a simplified
    Laplacian edge detection method (inspired by L.A.Cosmic).

    This is a lightweight version suitable for the JJMO 2D spectral images
    (typically 20 x 765 pixels). For production use, consider astroscrappy.

    Parameters
    ----------
    image : 2D ndarray
        Raw CCD image (counts).
    sigma_thresh : float
        Detection threshold in units of the noise model.
    gain : float
        CCD gain (e-/ADU). Used for Poisson noise estimate.
    readnoise : float
        Read noise (e-). Used in the noise model.

    Returns
    -------
    cr_mask : 2D ndarray of bool
        True = good pixel, False = cosmic ray.
    n_flagged : int
        Number of pixels flagged.
    """
    image = np.asarray(image, dtype=float)

    from scipy.ndimage import laplace, median_filter as mf2d

    # Median-smooth the image to estimate the background.
    # Use a kernel that spans several pixels in the spatial direction
    # but is narrow in the dispersion direction to preserve spectral features.
    ny, nx = image.shape
    med_size = (min(5, ny), 5)
    med_img = mf2d(image, size=med_size)

    # Subtract background so the Laplacian only sees sharp residuals.
    # This prevents extended spectral features from triggering the detector.
    residual_img = image - med_img

    # Laplacian of the residual emphasises truly sharp features (cosmic rays)
    lap = laplace(residual_img)

    # Noise model: Poisson + readnoise per pixel
    noise = np.sqrt(np.maximum(med_img * gain, 0) + readnoise**2) / gain
    # Floor the noise to avoid division by near-zero
    noise = np.maximum(noise, 1.0)

    # Significance map
    with np.errstate(divide='ignore', invalid='ignore'):
        signif = lap / noise

    # Cosmic rays show as strong positive Laplacian (sharp bright points).
    # Also require the pixel itself to be significantly above the local median,
    # to avoid flagging noise fluctuations on the edges of spectral features.
    pixel_excess = residual_img / noise
    bad = (signif > sigma_thresh) & (pixel_excess > sigma_thresh * 0.5)

    cr_mask = ~bad
    return cr_mask, int(np.sum(bad))


# ===================================================================
# 3.3  Telluric Region Masking
# ===================================================================

def mask_telluric(wavelength, bands=None):
    """Mask known telluric absorption bands.

    Parameters
    ----------
    wavelength : array-like
        Wavelength array (Angstroms).
    bands : list of (float, float), optional
        List of (lower, upper) wavelength pairs defining telluric bands.
        Defaults to the standard O2 and H2O bands.

    Returns
    -------
    mask : ndarray of bool
        True = outside telluric bands (good), False = inside a band.
    bands_in_segment : list of (float, float)
        Which telluric bands actually overlap this segment.
    """
    wavelength = np.asarray(wavelength, dtype=float)
    if bands is None:
        bands = TELLURIC_BANDS

    mask = np.ones(len(wavelength), dtype=bool)
    wmin, wmax = wavelength.min(), wavelength.max()
    bands_in_segment = []

    for lo, hi in bands:
        if hi < wmin or lo > wmax:
            continue  # band doesn't overlap this segment
        bands_in_segment.append((lo, hi))
        in_band = (wavelength >= lo) & (wavelength <= hi)
        mask[in_band] = False

    return mask, bands_in_segment


def mask_custom_regions(wavelength, regions):
    """Mask user-specified wavelength regions.

    Parameters
    ----------
    wavelength : array-like
        Wavelength array (Angstroms).
    regions : list of (float, float)
        List of (lower, upper) wavelength pairs to mask.

    Returns
    -------
    mask : ndarray of bool
        True = outside custom regions (good).
    """
    wavelength = np.asarray(wavelength, dtype=float)
    mask = np.ones(len(wavelength), dtype=bool)

    for lo, hi in regions:
        in_region = (wavelength >= lo) & (wavelength <= hi)
        mask[in_region] = False

    return mask


# ===================================================================
# 3.4  Stellar Absorption Line Masking
# ===================================================================

def mask_stellar_lines(wavelength, velocity_shift=0.0,
                       balmer_lines=None, metal_lines=None,
                       balmer_half_width=None, metal_half_width=None,
                       extra_lines=None):
    """Mask stellar absorption lines at their observed (shifted) wavelengths.

    Intended for use before fitting the smooth sensitivity function, where
    absorption features would bias the fit.

    Parameters
    ----------
    wavelength : array-like
        Wavelength array (Angstroms).
    velocity_shift : float
        Radial velocity of the star in km/s. Lines are shifted by
        ``lambda_obs = lambda_rest * (1 + v/c)``.
    balmer_lines : array-like, optional
        Rest wavelengths of Balmer lines. Defaults to the standard list.
    metal_lines : array-like, optional
        Rest wavelengths of metal lines. Defaults to the standard list.
    balmer_half_width : float, optional
        Half-width (Angstroms) to mask around each Balmer line. Default 15 A.
    metal_half_width : float, optional
        Half-width (Angstroms) to mask around each metal line. Default 5 A.
    extra_lines : list of (float, float), optional
        Additional (rest_wavelength, half_width) pairs to mask.

    Returns
    -------
    mask : ndarray of bool
        True = outside absorption lines (good).
    lines_masked : list of dict
        Info about each line that was actually masked in this segment.
    """
    wavelength = np.asarray(wavelength, dtype=float)
    if balmer_lines is None:
        balmer_lines = BALMER_LINES
    if metal_lines is None:
        metal_lines = METAL_LINES
    if balmer_half_width is None:
        balmer_half_width = BALMER_HALF_WIDTH
    if metal_half_width is None:
        metal_half_width = METAL_HALF_WIDTH

    c_kms = 299792.458  # speed of light in km/s
    doppler_factor = 1.0 + velocity_shift / c_kms

    mask = np.ones(len(wavelength), dtype=bool)
    wmin, wmax = wavelength.min(), wavelength.max()
    lines_masked = []

    # Mask Balmer lines
    for rest_wave in balmer_lines:
        obs_wave = rest_wave * doppler_factor
        if obs_wave < wmin - balmer_half_width or obs_wave > wmax + balmer_half_width:
            continue
        in_line = np.abs(wavelength - obs_wave) <= balmer_half_width
        if np.any(in_line):
            mask[in_line] = False
            lines_masked.append({
                'rest_wave': rest_wave,
                'obs_wave': obs_wave,
                'half_width': balmer_half_width,
                'type': 'balmer',
                'n_pixels': int(np.sum(in_line)),
            })

    # Mask metal lines
    for rest_wave in metal_lines:
        obs_wave = rest_wave * doppler_factor
        if obs_wave < wmin - metal_half_width or obs_wave > wmax + metal_half_width:
            continue
        in_line = np.abs(wavelength - obs_wave) <= metal_half_width
        if np.any(in_line):
            mask[in_line] = False
            lines_masked.append({
                'rest_wave': rest_wave,
                'obs_wave': obs_wave,
                'half_width': metal_half_width,
                'type': 'metal',
                'n_pixels': int(np.sum(in_line)),
            })

    # Mask extra user-specified lines
    if extra_lines:
        for rest_wave, half_w in extra_lines:
            obs_wave = rest_wave * doppler_factor
            if obs_wave < wmin - half_w or obs_wave > wmax + half_w:
                continue
            in_line = np.abs(wavelength - obs_wave) <= half_w
            if np.any(in_line):
                mask[in_line] = False
                lines_masked.append({
                    'rest_wave': rest_wave,
                    'obs_wave': obs_wave,
                    'half_width': half_w,
                    'type': 'extra',
                    'n_pixels': int(np.sum(in_line)),
                })

    return mask, lines_masked


# ===================================================================
# 3.5  SNR Estimation
# ===================================================================

def estimate_snr(flux, mask_good=None, method='auto'):
    """Estimate per-pixel and summary SNR for a spectral segment.

    Parameters
    ----------
    flux : array-like
        Flux array (counts).
    mask_good : array-like of bool, optional
        Boolean mask where True = valid pixel. SNR is computed only for
        these pixels; the returned array has the same length as flux
        with NaN for masked pixels.
    method : str
        'poisson' — SNR = sqrt(counts), valid for photon-dominated regime.
        'rms'     — SNR = median(flux) / RMS(flux) in continuum regions,
                     better for read-noise-dominated regime.
        'auto'    — Use Poisson if median counts > 100, else RMS.

    Returns
    -------
    snr_per_pixel : ndarray
        Per-pixel SNR estimate (NaN for masked pixels).
    snr_median : float
        Median SNR across unmasked pixels.
    snr_method : str
        Which method was actually used.
    """
    flux = np.asarray(flux, dtype=float)
    n = len(flux)

    if mask_good is None:
        mask_good = np.ones(n, dtype=bool)
    mask_good = np.asarray(mask_good, dtype=bool)

    valid_flux = flux[mask_good]

    if len(valid_flux) == 0:
        return np.full(n, np.nan), 0.0, 'none'

    med_flux = np.median(valid_flux)

    # Choose method
    if method == 'auto':
        method = 'poisson' if med_flux > 100 else 'rms'

    snr_per_pixel = np.full(n, np.nan)

    if method == 'poisson':
        # SNR ~ sqrt(counts) for Poisson-dominated regime
        good_vals = flux[mask_good]
        snr_vals = np.where(good_vals > 0, np.sqrt(good_vals), 0.0)
        snr_per_pixel[mask_good] = snr_vals

    elif method == 'rms':
        # Estimate noise from RMS of residuals after local smoothing
        # to isolate the noise component from the signal
        kernel = min(51, max(5, n // 10))
        if kernel % 2 == 0:
            kernel += 1
        flux_smooth = median_filter(flux, size=kernel)
        residuals = flux - flux_smooth
        # Use MAD-based sigma for robustness against remaining features
        mad = np.median(np.abs(residuals[mask_good]))
        noise_est = mad * 1.4826
        if noise_est > 0:
            snr_per_pixel[mask_good] = flux[mask_good] / noise_est
        else:
            # Zero residual noise (e.g., constant flux) — fall back to
            # Poisson estimate as a lower bound
            good_vals = flux[mask_good]
            snr_per_pixel[mask_good] = np.where(
                good_vals > 0, np.sqrt(good_vals), 0.0)

    snr_median = float(np.nanmedian(snr_per_pixel[mask_good]))

    return snr_per_pixel, snr_median, method


# ===================================================================
# 3.6  Quality Report
# ===================================================================

@dataclass
class QualityReport:
    """Container for the full quality assessment of a single spectral segment."""

    # Identification
    segment_id: str = ''

    # Wavelength range (after trimming)
    wave_min: float = 0.0
    wave_max: float = 0.0
    wave_min_original: float = 0.0
    wave_max_original: float = 0.0

    # Pixel counts
    n_pixels_total: int = 0
    n_pixels_good: int = 0

    # Edge trimming
    trim_blue: float = 0.0
    trim_red: float = 0.0
    n_pixels_edge_trimmed: int = 0

    # Cosmic rays
    n_cosmic_rays: int = 0

    # Telluric masking
    telluric_bands_found: list = field(default_factory=list)
    n_pixels_telluric: int = 0

    # Stellar line masking
    lines_masked: list = field(default_factory=list)
    n_pixels_stellar: int = 0

    # SNR
    snr_median: float = 0.0
    snr_method: str = ''
    snr_per_pixel: Optional[np.ndarray] = field(default=None, repr=False)

    # Individual masks (True = good)
    mask_edge: Optional[np.ndarray] = field(default=None, repr=False)
    mask_cosmic: Optional[np.ndarray] = field(default=None, repr=False)
    mask_telluric: Optional[np.ndarray] = field(default=None, repr=False)
    mask_stellar: Optional[np.ndarray] = field(default=None, repr=False)
    mask_custom: Optional[np.ndarray] = field(default=None, repr=False)

    # Combined mask (True = good pixel)
    mask_good: Optional[np.ndarray] = field(default=None, repr=False)

    # Whether the whole segment is flagged as unusable
    usable: bool = True
    unusable_reason: str = ''

    @property
    def frac_masked(self):
        """Fraction of total pixels that are masked."""
        if self.n_pixels_total == 0:
            return 1.0
        return 1.0 - self.n_pixels_good / self.n_pixels_total

    def summary(self):
        """Return a human-readable summary string."""
        lines = [
            f"=== Quality Report: {self.segment_id} ===",
            f"  Wavelength range: {self.wave_min:.1f} -- {self.wave_max:.1f} A "
            f"(original: {self.wave_min_original:.1f} -- {self.wave_max_original:.1f} A)",
            f"  Total pixels: {self.n_pixels_total}",
            f"  Good pixels:  {self.n_pixels_good} ({100*(1-self.frac_masked):.1f}%)",
            f"  Edge trimmed: {self.n_pixels_edge_trimmed} pixels "
            f"(blue to {self.trim_blue:.1f} A, red to {self.trim_red:.1f} A)",
            f"  Cosmic rays:  {self.n_cosmic_rays} pixels flagged",
            f"  Telluric:     {self.n_pixels_telluric} pixels in "
            f"{len(self.telluric_bands_found)} band(s)",
            f"  Stellar lines: {self.n_pixels_stellar} pixels across "
            f"{len(self.lines_masked)} line(s)",
            f"  Median SNR:   {self.snr_median:.1f} ({self.snr_method})",
            f"  Usable:       {'YES' if self.usable else 'NO — ' + self.unusable_reason}",
        ]
        return '\n'.join(lines)


# ===================================================================
# Unified Pipeline: assess_segment()
# ===================================================================

def assess_segment(wavelength, flux, segment_id='',
                   # Edge trimming parameters
                   edge_threshold_frac=0.20,
                   edge_min_run=5,
                   manual_trim=None,
                   # Cosmic ray parameters
                   cosmic_sigma=5.0,
                   cosmic_window=21,
                   # Telluric parameters
                   telluric_bands=None,
                   mask_telluric_flag=True,
                   # Stellar line parameters
                   mask_stellar_flag=True,
                   velocity_shift=0.0,
                   balmer_half_width=None,
                   metal_half_width=None,
                   extra_lines=None,
                   # Custom regions
                   custom_mask_regions=None,
                   # SNR parameters
                   snr_method='auto',
                   snr_threshold=3.0,
                   # General
                   max_masked_frac=0.80):
    """Run the full quality assessment pipeline on a single spectral segment.

    Parameters
    ----------
    wavelength : array-like
        Wavelength array (Angstroms). Will be sorted to ascending if needed.
    flux : array-like
        Flux array (counts or counts/s), same length as wavelength.
    segment_id : str
        Identifier for this segment (e.g., '3900', 'seg_01').
    edge_threshold_frac : float
        Fraction of median flux for edge trimming threshold.
    edge_min_run : int
        Minimum consecutive good pixels for edge detection.
    manual_trim : tuple of (float, float), optional
        If provided, (blue_limit, red_limit) in Angstroms — overrides
        automatic edge trimming.
    cosmic_sigma : float
        Sigma threshold for cosmic ray detection.
    cosmic_window : int
        Window size for running median in cosmic ray detection.
    telluric_bands : list of (float, float), optional
        Custom telluric band definitions. Uses defaults if None.
    mask_telluric_flag : bool
        Whether to apply telluric masking.
    mask_stellar_flag : bool
        Whether to apply stellar absorption line masking.
    velocity_shift : float
        Stellar radial velocity in km/s for line shift correction.
    balmer_half_width : float, optional
        Half-width for Balmer line masking (default 15 A).
    metal_half_width : float, optional
        Half-width for metal line masking (default 5 A).
    extra_lines : list of (float, float), optional
        Additional (rest_wavelength, half_width) pairs to mask.
    custom_mask_regions : list of (float, float), optional
        Additional wavelength regions to mask.
    snr_method : str
        SNR estimation method: 'poisson', 'rms', or 'auto'.
    snr_threshold : float
        Minimum median SNR for a segment to be considered usable.
    max_masked_frac : float
        If more than this fraction of pixels are masked, flag segment unusable.

    Returns
    -------
    report : QualityReport
        Full quality assessment results including all masks and diagnostics.
    """
    wavelength = np.asarray(wavelength, dtype=float)
    flux = np.asarray(flux, dtype=float)

    if len(wavelength) != len(flux):
        raise ValueError(
            f"wavelength and flux must have the same length, "
            f"got {len(wavelength)} and {len(flux)}")

    # Ensure ascending wavelength order
    if len(wavelength) > 1 and wavelength[0] > wavelength[-1]:
        sort_idx = np.argsort(wavelength)
        wavelength = wavelength[sort_idx]
        flux = flux[sort_idx]

    n = len(wavelength)
    report = QualityReport(
        segment_id=segment_id,
        wave_min_original=float(wavelength[0]),
        wave_max_original=float(wavelength[-1]),
        n_pixels_total=n,
    )

    # --- 3.1: Edge trimming ---
    if manual_trim is not None:
        blue_lim, red_lim = manual_trim
        mask_edge = (wavelength >= blue_lim) & (wavelength <= red_lim)
        trim_blue, trim_red = blue_lim, red_lim
    else:
        mask_edge, trim_blue, trim_red = trim_edges(
            wavelength, flux,
            threshold_frac=edge_threshold_frac,
            min_run=edge_min_run)

    report.mask_edge = mask_edge
    report.trim_blue = trim_blue
    report.trim_red = trim_red
    report.n_pixels_edge_trimmed = int(np.sum(~mask_edge))

    # --- 3.2: Cosmic ray detection (on the edge-trimmed region only) ---
    mask_cosmic = np.ones(n, dtype=bool)
    flux_for_cr = flux.copy()
    # Only check for cosmics within the edge-trimmed region
    idx_good_edge = np.where(mask_edge)[0]
    if len(idx_good_edge) > 0:
        cr_mask_sub, n_cr = detect_cosmics_1d(
            flux[idx_good_edge],
            sigma_thresh=cosmic_sigma,
            window_size=cosmic_window)
        mask_cosmic[idx_good_edge[~cr_mask_sub]] = False
    else:
        n_cr = 0

    report.mask_cosmic = mask_cosmic
    report.n_cosmic_rays = n_cr

    # --- 3.3: Telluric region masking ---
    if mask_telluric_flag:
        mask_tell, bands_found = mask_telluric(wavelength, bands=telluric_bands)
    else:
        mask_tell = np.ones(n, dtype=bool)
        bands_found = []

    report.mask_telluric = mask_tell
    report.telluric_bands_found = bands_found
    report.n_pixels_telluric = int(np.sum(~mask_tell))

    # --- 3.4: Stellar absorption line masking ---
    if mask_stellar_flag:
        mask_star, lines_found = mask_stellar_lines(
            wavelength,
            velocity_shift=velocity_shift,
            balmer_half_width=balmer_half_width,
            metal_half_width=metal_half_width,
            extra_lines=extra_lines)
    else:
        mask_star = np.ones(n, dtype=bool)
        lines_found = []

    report.mask_stellar = mask_star
    report.lines_masked = lines_found
    report.n_pixels_stellar = int(np.sum(~mask_star))

    # --- Custom mask regions ---
    if custom_mask_regions:
        mask_cust = mask_custom_regions(wavelength, custom_mask_regions)
    else:
        mask_cust = np.ones(n, dtype=bool)

    report.mask_custom = mask_cust

    # --- Combine all masks ---
    mask_good = mask_edge & mask_cosmic & mask_tell & mask_star & mask_cust
    report.mask_good = mask_good
    report.n_pixels_good = int(np.sum(mask_good))

    # --- Wavelength range of good pixels ---
    good_idx = np.where(mask_good)[0]
    if len(good_idx) > 0:
        report.wave_min = float(wavelength[good_idx[0]])
        report.wave_max = float(wavelength[good_idx[-1]])
    else:
        report.wave_min = report.wave_min_original
        report.wave_max = report.wave_max_original

    # --- 3.5: SNR estimation (on good pixels) ---
    snr_per_pixel, snr_median, used_method = estimate_snr(
        flux, mask_good=mask_good, method=snr_method)

    report.snr_per_pixel = snr_per_pixel
    report.snr_median = snr_median
    report.snr_method = used_method

    # --- Check segment usability ---
    report.usable = True
    if report.frac_masked > max_masked_frac:
        report.usable = False
        report.unusable_reason = (
            f"Too many pixels masked ({report.frac_masked*100:.0f}% > "
            f"{max_masked_frac*100:.0f}%)")
    elif snr_median < snr_threshold:
        report.usable = False
        report.unusable_reason = (
            f"SNR too low ({snr_median:.1f} < {snr_threshold:.1f})")
    elif report.n_pixels_good < 20:
        report.usable = False
        report.unusable_reason = (
            f"Too few good pixels ({report.n_pixels_good})")

    return report


# ===================================================================
# Batch Processing
# ===================================================================

def assess_segments(wavelengths, fluxes, segment_ids=None, **kwargs):
    """Run quality assessment on multiple segments.

    Parameters
    ----------
    wavelengths : list of array-like
        Wavelength arrays for each segment.
    fluxes : list of array-like
        Flux arrays for each segment.
    segment_ids : list of str, optional
        Identifiers for each segment.
    **kwargs
        Passed through to ``assess_segment()``.

    Returns
    -------
    reports : list of QualityReport
        One report per segment.
    """
    n_seg = len(wavelengths)
    if segment_ids is None:
        segment_ids = [f'seg_{i:02d}' for i in range(n_seg)]

    reports = []
    for i in range(n_seg):
        report = assess_segment(
            wavelengths[i], fluxes[i],
            segment_id=segment_ids[i],
            **kwargs)
        reports.append(report)

    return reports


def print_quality_table(reports):
    """Print a summary table of quality reports.

    Parameters
    ----------
    reports : list of QualityReport
    """
    header = (f"{'Segment':<10} {'Wave Range':<22} {'Good/Total':<12} "
              f"{'Masked%':<8} {'CR':<4} {'Tell':<5} {'Lines':<6} "
              f"{'SNR':<8} {'Usable':<6}")
    print(header)
    print('-' * len(header))

    for r in reports:
        wave_range = f"{r.wave_min:.0f}--{r.wave_max:.0f}"
        pix = f"{r.n_pixels_good}/{r.n_pixels_total}"
        masked = f"{r.frac_masked*100:.1f}%"
        usable = "YES" if r.usable else "NO"
        print(f"{r.segment_id:<10} {wave_range:<22} {pix:<12} "
              f"{masked:<8} {r.n_cosmic_rays:<4} {r.n_pixels_telluric:<5} "
              f"{len(r.lines_masked):<6} {r.snr_median:<8.1f} {usable:<6}")


# ===================================================================
# Diagnostic Plotting
# ===================================================================

def plot_segment_quality(wavelength, flux, report, save_path=None, show=False):
    """Plot a single segment with masked regions shaded for visual verification.

    Parameters
    ----------
    wavelength : array-like
        Wavelength array (Angstroms).
    flux : array-like
        Flux array (counts).
    report : QualityReport
        Quality assessment report for this segment.
    save_path : str, optional
        Path to save the figure. If None, figure is not saved.
    show : bool
        Whether to call plt.show().

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    wavelength = np.asarray(wavelength, dtype=float)
    flux = np.asarray(flux, dtype=float)

    # Ensure ascending order (matching assess_segment behavior)
    if len(wavelength) > 1 and wavelength[0] > wavelength[-1]:
        sort_idx = np.argsort(wavelength)
        wavelength = wavelength[sort_idx]
        flux = flux[sort_idx]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                              gridspec_kw={'height_ratios': [3, 1]})

    # --- Top panel: flux with masked regions ---
    ax1 = axes[0]
    ax1.plot(wavelength, flux, color='0.3', lw=0.5, alpha=0.7, label='Raw flux')

    # Shade edge-trimmed regions
    if report.mask_edge is not None:
        _shade_mask(ax1, wavelength, ~report.mask_edge, color='gray',
                    alpha=0.3, label='Edge trimmed')

    # Shade telluric bands
    if report.mask_telluric is not None:
        _shade_mask(ax1, wavelength, ~report.mask_telluric, color='blue',
                    alpha=0.2, label='Telluric')

    # Shade stellar lines
    if report.mask_stellar is not None:
        _shade_mask(ax1, wavelength, ~report.mask_stellar, color='red',
                    alpha=0.2, label='Stellar lines')

    # Mark cosmic rays as vertical lines
    if report.mask_cosmic is not None:
        cr_idx = np.where(~report.mask_cosmic)[0]
        if len(cr_idx) > 0:
            for idx in cr_idx:
                ax1.axvline(wavelength[idx], color='magenta', lw=0.8,
                           alpha=0.5, zorder=5)
            # Add a single label entry for the legend
            ax1.axvline(wavelength[cr_idx[0]], color='magenta', lw=0.8,
                       alpha=0.5, label='Cosmic ray')

    # Plot good pixels highlighted
    if report.mask_good is not None:
        ax1.plot(wavelength[report.mask_good], flux[report.mask_good],
                color='green', lw=0.6, alpha=0.5, label='Good pixels')

    ax1.set_ylabel('Flux (counts)')
    title = f'Segment {report.segment_id}'
    if not report.usable:
        title += f'  [UNUSABLE: {report.unusable_reason}]'
    ax1.set_title(title)
    ax1.legend(loc='upper right', fontsize=8)

    # --- Bottom panel: SNR per pixel ---
    ax2 = axes[1]
    if report.snr_per_pixel is not None:
        valid = ~np.isnan(report.snr_per_pixel)
        ax2.plot(wavelength[valid], report.snr_per_pixel[valid],
                color='darkorange', lw=0.5, alpha=0.8)
        ax2.axhline(report.snr_median, color='darkorange', ls='--', lw=1,
                    label=f'Median SNR = {report.snr_median:.1f}')
        ax2.legend(loc='upper right', fontsize=8)

    ax2.set_xlabel('Wavelength (Å)')
    ax2.set_ylabel('SNR')
    ax2.set_ylim(bottom=0)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_quality_overview(wavelengths, fluxes, reports, save_path=None, show=False):
    """Plot an overview of all segments with quality color coding.

    Parameters
    ----------
    wavelengths : list of array-like
    fluxes : list of array-like
    reports : list of QualityReport
    save_path : str, optional
    show : bool
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n = len(reports)
    fig, axes = plt.subplots(n, 1, figsize=(14, 2.5 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for i, (wave, flux, report) in enumerate(zip(wavelengths, fluxes, reports)):
        wave = np.asarray(wave, dtype=float)
        flux = np.asarray(flux, dtype=float)
        if len(wave) > 1 and wave[0] > wave[-1]:
            sort_idx = np.argsort(wave)
            wave = wave[sort_idx]
            flux = flux[sort_idx]

        ax = axes[i]
        color = 'green' if report.usable else 'red'
        ax.plot(wave, flux, color='0.5', lw=0.4, alpha=0.5)
        if report.mask_good is not None:
            ax.plot(wave[report.mask_good], flux[report.mask_good],
                   color=color, lw=0.6, alpha=0.7)
        ax.set_ylabel(f'{report.segment_id}', fontsize=9)
        ax.tick_params(labelsize=7)

        info = (f"SNR={report.snr_median:.0f}  "
                f"Good={100*(1-report.frac_masked):.0f}%  "
                f"CR={report.n_cosmic_rays}")
        ax.text(0.98, 0.85, info, transform=ax.transAxes,
               fontsize=7, ha='right', va='top',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    axes[-1].set_xlabel('Wavelength (Å)')
    fig.suptitle('Segment Quality Overview', fontsize=12, y=1.01)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def _shade_mask(ax, wavelength, bad_mask, color='gray', alpha=0.3, label=None):
    """Shade contiguous masked regions on an axis."""
    if not np.any(bad_mask):
        return
    # Find contiguous runs of True in bad_mask
    diff = np.diff(bad_mask.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1

    # Handle edge cases
    if bad_mask[0]:
        starts = np.concatenate([[0], starts])
    if bad_mask[-1]:
        ends = np.concatenate([ends, [len(bad_mask)]])

    labeled = False
    for s, e in zip(starts, ends):
        lbl = label if not labeled else None
        ax.axvspan(wavelength[s], wavelength[min(e, len(wavelength)-1)],
                  color=color, alpha=alpha, label=lbl, zorder=0)
        labeled = True
