"""
sensitivity.py -- Sensitivity Function Derivation
==================================================
Step 6 of the JJMO Spectral Flux Calibration Pipeline.

Derives the instrumental sensitivity function S(lambda) by comparing observed
counts from a known standard star to its trusted reference flux:

    S(lambda) = F_true(lambda) / C_obs(lambda)

where F_true is the reference flux (erg/s/cm^2/A) and C_obs is the observed
count rate (counts/s/A), corrected for atmospheric extinction.

The sensitivity function is smooth (representing optics + detector + atmosphere,
not stellar features), so it is fit with a low-order function after masking
stellar and telluric features.

Supports:
  - Per-segment and stitched-first sensitivity derivation
  - Multiple fit methods: Chebyshev, Legendre, cubic spline, Savitzky-Golay
  - Iterative sigma-clipping during fitting
  - Grey-shift combination across segments
  - Serialization to/from JSON
  - Comprehensive diagnostic plotting

Authors: JJMO Pipeline
"""

import json
import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.interpolate import UnivariateSpline, interp1d
from scipy.signal import savgol_filter

logger = logging.getLogger(__name__)


# ============================================================================
# Constants
# ============================================================================

# Telluric absorption bands (Angstroms) -- duplicated from quality.py for
# standalone use; these define default regions to exclude from sensitivity fits.
TELLURIC_BANDS = [
    (6270.0, 6290.0),   # O2 near 6280 A
    (6860.0, 6880.0),   # O2 B-band
    (7150.0, 7300.0),   # H2O band
    (7590.0, 7650.0),   # O2 A-band (deep)
    (8100.0, 8400.0),   # H2O far red
]

# Balmer and metal lines to mask for sensitivity fitting
# (sensitivity should be smooth, not follow stellar features)
BALMER_LINES_AA = np.array([
    6562.8, 4861.3, 4340.5, 4101.7, 3970.1, 3889.1, 3835.4,
])
METAL_LINES_AA = np.array([
    3933.7, 3968.5, 4481.2, 4383.6, 4923.9, 5018.4, 5169.0, 4552.6, 4077.7,
])

# Default masking half-widths
DEFAULT_BALMER_HALF_WIDTH = 15.0   # Angstroms
DEFAULT_METAL_HALF_WIDTH = 5.0     # Angstroms

# Supported fit methods
VALID_FIT_METHODS = ('chebyshev', 'legendre', 'spline', 'savgol')

# Minimum number of unmasked pixels needed to attempt a fit
MIN_FIT_PIXELS = 10


# ============================================================================
# 6.1  Compute the raw sensitivity ratio per segment
# ============================================================================

def compute_sensitivity_ratio(
    wavelength_obs: np.ndarray,
    flux_obs: np.ndarray,
    wavelength_ref: np.ndarray,
    flux_ref: np.ndarray,
    exptime: Optional[float] = None,
    mask: Optional[np.ndarray] = None,
    airmass: Optional[float] = None,
    extinction_curve: Optional[Callable] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the raw sensitivity ratio S(lambda) = F_ref / C_obs.

    The reference spectrum is resampled onto the observed wavelength grid
    before division.  Exposure-time normalization and atmospheric extinction
    correction are applied to the observed flux if the relevant parameters
    are provided.

    Parameters
    ----------
    wavelength_obs : array (N,)
        Observed wavelength grid in Angstroms.
    flux_obs : array (N,)
        Observed flux in counts (or counts/s if already normalized).
    wavelength_ref : array (M,)
        Reference spectrum wavelength grid in Angstroms.
    flux_ref : array (M,)
        Reference flux in physical units (e.g. erg/s/cm^2/A).
    exptime : float, optional
        Exposure time in seconds.  If provided, flux_obs is divided by
        exptime to convert to count rate.
    mask : array of bool (N,), optional
        Boolean mask where True = good pixel.  If None, all pixels are good.
    airmass : float, optional
        Airmass of the observation.  Used with ``extinction_curve`` to
        correct for atmospheric extinction.
    extinction_curve : callable, optional
        Function that takes wavelength (Angstrom) and returns extinction
        in magnitudes per airmass (k(lambda)).

    Returns
    -------
    wavelength : array (N,)
        The observed wavelength grid (unchanged).
    ratio : array (N,)
        The raw sensitivity ratio F_ref / C_obs (may contain NaN/Inf).
    mask_out : array of bool (N,)
        Updated mask (True = good). Pixels where the ratio is non-finite
        or C_obs <= 0 are masked False.
    """
    wavelength_obs = np.asarray(wavelength_obs, dtype=np.float64)
    flux_obs = np.asarray(flux_obs, dtype=np.float64).copy()
    wavelength_ref = np.asarray(wavelength_ref, dtype=np.float64)
    flux_ref = np.asarray(flux_ref, dtype=np.float64)

    if mask is None:
        mask = np.ones(len(wavelength_obs), dtype=bool)
    else:
        mask = np.asarray(mask, dtype=bool).copy()

    # Exposure-time normalization: counts -> count rate
    if exptime is not None and exptime > 0:
        flux_obs = flux_obs / exptime

    # Atmospheric extinction correction: restore the above-atmosphere flux
    # by dividing out the extinction attenuation (i.e. multiply observed
    # flux by 10^(0.4 * airmass * k(lambda))).
    if airmass is not None and extinction_curve is not None:
        k_lambda = extinction_curve(wavelength_obs)
        extinction_factor = 10.0 ** (0.4 * airmass * k_lambda)
        flux_obs = flux_obs * extinction_factor

    # Resample reference spectrum onto observed wavelength grid
    # using linear interpolation (sufficient for smooth reference spectra).
    ref_resampled = _resample_reference(
        wavelength_ref, flux_ref, wavelength_obs
    )

    # Mask pixels outside the reference spectrum coverage
    outside = (wavelength_obs < wavelength_ref.min()) | \
              (wavelength_obs > wavelength_ref.max())
    mask[outside] = False

    # Mask non-positive observed flux (division would be meaningless)
    bad_obs = flux_obs <= 0
    mask[bad_obs] = False

    # Mask non-positive reference flux
    bad_ref = ref_resampled <= 0
    mask[bad_ref] = False

    # Compute the ratio
    ratio = np.full_like(flux_obs, np.nan)
    good = mask & (flux_obs > 0)
    ratio[good] = ref_resampled[good] / flux_obs[good]

    # Final mask update: flag non-finite ratios
    mask[~np.isfinite(ratio)] = False

    n_good = int(np.sum(mask))
    logger.info(
        "Sensitivity ratio: %d/%d good pixels (%.1f%%)",
        n_good, len(mask), 100.0 * n_good / max(len(mask), 1)
    )

    return wavelength_obs, ratio, mask


def _resample_reference(
    wave_ref: np.ndarray,
    flux_ref: np.ndarray,
    wave_target: np.ndarray,
) -> np.ndarray:
    """Resample a reference spectrum onto a target wavelength grid.

    Uses linear interpolation, which is adequate for smooth stellar model
    or CALSPEC reference spectra being resampled onto a coarser grid.
    Extrapolated values are set to NaN.
    """
    interp_func = interp1d(
        wave_ref, flux_ref,
        kind='linear',
        bounds_error=False,
        fill_value=np.nan,
    )
    return interp_func(wave_target)


# ============================================================================
# 6.2  Fit a smooth sensitivity function per segment
# ============================================================================

@dataclass
class SensitivityFit:
    """Container for a fitted sensitivity function.

    Stores the fit coefficients/parameters and provides a callable interface
    to evaluate the sensitivity at arbitrary wavelengths.

    Attributes
    ----------
    method : str
        Fit method ('chebyshev', 'legendre', 'spline', 'savgol').
    order : int
        Polynomial order or spline knot count or Savgol window size.
    coefficients : list
        Fit coefficients (polynomial) or smoothed values (savgol).
    wave_min, wave_max : float
        Domain of the fit (Angstroms).
    sigma_clip_threshold : float
        Sigma-clip threshold used during fitting.
    n_iterations : int
        Number of sigma-clip iterations performed.
    n_rejected : int
        Total number of points rejected by sigma clipping.
    n_points_used : int
        Number of data points in the final fit.
    rms_residual : float
        RMS of the fit residuals on unmasked data.
    segment_id : str
        Identifier for the segment this fit belongs to.
    grey_shift : float
        Multiplicative grey-shift factor applied (1.0 = no shift).
    _wavelength_data : list
        Wavelength grid (for savgol or spline evaluation lookups).
    _flux_data : list
        Smoothed flux values (for savgol, or spline knot values).
    metadata : dict
        Additional metadata (standard star, reference source, etc.).
    """
    method: str = ''
    order: int = 0
    coefficients: list = field(default_factory=list)
    wave_min: float = 0.0
    wave_max: float = 0.0
    sigma_clip_threshold: float = 3.0
    n_iterations: int = 0
    n_rejected: int = 0
    n_points_used: int = 0
    rms_residual: float = 0.0
    segment_id: str = ''
    grey_shift: float = 1.0
    _wavelength_data: list = field(default_factory=list)
    _flux_data: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def __call__(self, wavelength: np.ndarray) -> np.ndarray:
        """Evaluate the sensitivity function at given wavelengths.

        Parameters
        ----------
        wavelength : array-like
            Wavelengths in Angstroms.

        Returns
        -------
        sensitivity : array
            Evaluated sensitivity values.  NaN for wavelengths outside
            the fit domain.
        """
        wavelength = np.asarray(wavelength, dtype=np.float64)
        result = np.full_like(wavelength, np.nan)

        # Only evaluate within the fit domain
        in_domain = (wavelength >= self.wave_min) & (wavelength <= self.wave_max)

        if not np.any(in_domain):
            return result

        w = wavelength[in_domain]

        if self.method in ('chebyshev', 'legendre'):
            # Coefficients are stored for the normalized domain [-1, 1]
            w_norm = _normalize_wavelength(w, self.wave_min, self.wave_max)
            if self.method == 'chebyshev':
                result[in_domain] = np.polynomial.chebyshev.chebval(
                    w_norm, self.coefficients
                )
            else:
                result[in_domain] = np.polynomial.legendre.legval(
                    w_norm, self.coefficients
                )

        elif self.method == 'spline':
            # Reconstruct the spline from stored knots/coefficients
            if len(self._wavelength_data) > 0 and len(self._flux_data) > 0:
                interp_func = interp1d(
                    self._wavelength_data, self._flux_data,
                    kind='cubic', bounds_error=False, fill_value=np.nan,
                )
                result[in_domain] = interp_func(w)

        elif self.method == 'savgol':
            # Savgol stores the smoothed values; interpolate for evaluation
            if len(self._wavelength_data) > 0 and len(self._flux_data) > 0:
                interp_func = interp1d(
                    self._wavelength_data, self._flux_data,
                    kind='linear', bounds_error=False, fill_value=np.nan,
                )
                result[in_domain] = interp_func(w)

        # Apply grey shift
        result *= self.grey_shift

        return result

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dictionary."""
        return {
            'method': self.method,
            'order': self.order,
            'coefficients': list(self.coefficients),
            'wave_min': self.wave_min,
            'wave_max': self.wave_max,
            'sigma_clip_threshold': self.sigma_clip_threshold,
            'n_iterations': self.n_iterations,
            'n_rejected': self.n_rejected,
            'n_points_used': self.n_points_used,
            'rms_residual': self.rms_residual,
            'segment_id': self.segment_id,
            'grey_shift': self.grey_shift,
            '_wavelength_data': [float(x) for x in self._wavelength_data],
            '_flux_data': [float(x) for x in self._flux_data],
            'metadata': self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'SensitivityFit':
        """Reconstruct from a dictionary."""
        return cls(
            method=d['method'],
            order=d['order'],
            coefficients=d['coefficients'],
            wave_min=d['wave_min'],
            wave_max=d['wave_max'],
            sigma_clip_threshold=d.get('sigma_clip_threshold', 3.0),
            n_iterations=d.get('n_iterations', 0),
            n_rejected=d.get('n_rejected', 0),
            n_points_used=d.get('n_points_used', 0),
            rms_residual=d.get('rms_residual', 0.0),
            segment_id=d.get('segment_id', ''),
            grey_shift=d.get('grey_shift', 1.0),
            _wavelength_data=d.get('_wavelength_data', []),
            _flux_data=d.get('_flux_data', []),
            metadata=d.get('metadata', {}),
        )

    def to_magnitude(self, wavelength: np.ndarray) -> np.ndarray:
        """Evaluate the sensitivity in magnitude units (zeropoint curve).

        s(lambda) = -2.5 * log10(S(lambda))

        Useful for comparison with IRAF/PypeIt conventions.
        """
        s = self(wavelength)
        with np.errstate(divide='ignore', invalid='ignore'):
            mag = -2.5 * np.log10(s)
        return mag


def _normalize_wavelength(
    wavelength: np.ndarray,
    wave_min: float,
    wave_max: float,
) -> np.ndarray:
    """Map wavelength to the normalized domain [-1, 1]."""
    return 2.0 * (wavelength - wave_min) / (wave_max - wave_min) - 1.0


def _denormalize_wavelength(
    w_norm: np.ndarray,
    wave_min: float,
    wave_max: float,
) -> np.ndarray:
    """Map from [-1, 1] back to wavelength in Angstroms."""
    return (w_norm + 1.0) / 2.0 * (wave_max - wave_min) + wave_min


def fit_sensitivity(
    wavelength: np.ndarray,
    ratio: np.ndarray,
    mask: Optional[np.ndarray] = None,
    method: str = 'chebyshev',
    order: int = 5,
    sigma_clip: float = 3.0,
    max_iter: int = 5,
    segment_id: str = '',
    metadata: Optional[dict] = None,
) -> SensitivityFit:
    """Fit a smooth function to the raw sensitivity ratio.

    Implements iterative sigma-clipping: fit, reject outliers > N-sigma,
    refit, until convergence or max iterations reached.

    Parameters
    ----------
    wavelength : array (N,)
        Wavelength grid in Angstroms.
    ratio : array (N,)
        Raw sensitivity ratio S(lambda) = F_ref / C_obs.
    mask : array of bool (N,), optional
        True = good pixel to include in the fit.
    method : str
        Fit method: 'chebyshev', 'legendre', 'spline', or 'savgol'.
    order : int
        For chebyshev/legendre: polynomial order (3--8 typical).
        For spline: number of interior knots.
        For savgol: window length in pixels (must be odd, >= 5).
    sigma_clip : float
        Sigma threshold for iterative outlier rejection.
    max_iter : int
        Maximum number of sigma-clip iterations (0 = no clipping).
    segment_id : str
        Identifier for this segment.
    metadata : dict, optional
        Additional metadata to store with the fit.

    Returns
    -------
    SensitivityFit
        Callable object representing the fitted sensitivity function.

    Raises
    ------
    ValueError
        If fewer than MIN_FIT_PIXELS remain after masking, or if
        method/order is invalid.
    """
    wavelength = np.asarray(wavelength, dtype=np.float64)
    ratio = np.asarray(ratio, dtype=np.float64)

    if method not in VALID_FIT_METHODS:
        raise ValueError(
            f"Unknown fit method '{method}'. "
            f"Valid methods: {VALID_FIT_METHODS}"
        )

    if mask is None:
        mask = np.isfinite(ratio)
    else:
        mask = np.asarray(mask, dtype=bool) & np.isfinite(ratio)

    n_good = int(np.sum(mask))
    if n_good < MIN_FIT_PIXELS:
        raise ValueError(
            f"Only {n_good} unmasked pixels for segment '{segment_id}'; "
            f"need at least {MIN_FIT_PIXELS}"
        )

    wave_min = float(wavelength[mask].min())
    wave_max = float(wavelength[mask].max())

    # Iterative sigma-clipping loop
    fit_mask = mask.copy()
    total_rejected = 0

    for iteration in range(max(max_iter, 1)):
        w_fit = wavelength[fit_mask]
        r_fit = ratio[fit_mask]

        if len(w_fit) < MIN_FIT_PIXELS:
            warnings.warn(
                f"Sigma clipping reduced to {len(w_fit)} points "
                f"(segment '{segment_id}'); stopping."
            )
            break

        # Perform the fit
        if method == 'chebyshev':
            fitted_vals = _fit_chebyshev(w_fit, r_fit, order, wave_min, wave_max)
        elif method == 'legendre':
            fitted_vals = _fit_legendre(w_fit, r_fit, order, wave_min, wave_max)
        elif method == 'spline':
            fitted_vals = _fit_spline(w_fit, r_fit, order, wave_min, wave_max)
        elif method == 'savgol':
            fitted_vals = _fit_savgol(w_fit, r_fit, order)

        # Compute residuals
        residuals = r_fit - fitted_vals
        rms = np.std(residuals)

        if max_iter == 0 or rms == 0:
            break

        # Reject outliers
        outlier = np.abs(residuals) > sigma_clip * rms
        n_reject_this = int(np.sum(outlier))

        if n_reject_this == 0:
            logger.info(
                "Sigma clip converged after %d iterations (segment '%s')",
                iteration + 1, segment_id
            )
            break

        # Update the mask: un-flag the outlier points
        fit_indices = np.where(fit_mask)[0]
        fit_mask[fit_indices[outlier]] = False
        total_rejected += n_reject_this

        logger.debug(
            "Iteration %d: rejected %d outliers (segment '%s')",
            iteration + 1, n_reject_this, segment_id
        )

    # Final fit on the clipped data
    w_final = wavelength[fit_mask]
    r_final = ratio[fit_mask]

    # Build the SensitivityFit object
    fit_obj = SensitivityFit(
        method=method,
        order=order,
        wave_min=wave_min,
        wave_max=wave_max,
        sigma_clip_threshold=sigma_clip,
        n_iterations=iteration + 1 if max_iter > 0 else 0,
        n_rejected=total_rejected,
        n_points_used=len(w_final),
        segment_id=segment_id,
        metadata=metadata or {},
    )

    if method == 'chebyshev':
        w_norm = _normalize_wavelength(w_final, wave_min, wave_max)
        coeffs = np.polynomial.chebyshev.chebfit(w_norm, r_final, order)
        fit_obj.coefficients = coeffs.tolist()
        fitted_final = np.polynomial.chebyshev.chebval(w_norm, coeffs)

    elif method == 'legendre':
        w_norm = _normalize_wavelength(w_final, wave_min, wave_max)
        coeffs = np.polynomial.legendre.legfit(w_norm, r_final, order)
        fit_obj.coefficients = coeffs.tolist()
        fitted_final = np.polynomial.legendre.legval(w_norm, coeffs)

    elif method == 'spline':
        fitted_final, wave_data, flux_data = _fit_spline_full(
            w_final, r_final, order, wave_min, wave_max
        )
        fit_obj._wavelength_data = wave_data.tolist()
        fit_obj._flux_data = flux_data.tolist()

    elif method == 'savgol':
        fitted_final = _fit_savgol(w_final, r_final, order)
        fit_obj._wavelength_data = w_final.tolist()
        fit_obj._flux_data = fitted_final.tolist()

    # Compute final RMS
    residuals_final = r_final - fitted_final
    fit_obj.rms_residual = float(np.std(residuals_final))

    logger.info(
        "Fit '%s' order=%d for segment '%s': %d pts, %d rejected, "
        "RMS=%.4e",
        method, order, segment_id, len(w_final), total_rejected,
        fit_obj.rms_residual
    )

    return fit_obj


def _fit_chebyshev(
    w: np.ndarray, r: np.ndarray, order: int,
    wave_min: float, wave_max: float,
) -> np.ndarray:
    """Fit Chebyshev polynomial and return fitted values at data points."""
    w_norm = _normalize_wavelength(w, wave_min, wave_max)
    coeffs = np.polynomial.chebyshev.chebfit(w_norm, r, order)
    return np.polynomial.chebyshev.chebval(w_norm, coeffs)


def _fit_legendre(
    w: np.ndarray, r: np.ndarray, order: int,
    wave_min: float, wave_max: float,
) -> np.ndarray:
    """Fit Legendre polynomial and return fitted values at data points."""
    w_norm = _normalize_wavelength(w, wave_min, wave_max)
    coeffs = np.polynomial.legendre.legfit(w_norm, r, order)
    return np.polynomial.legendre.legval(w_norm, coeffs)


def _fit_spline(
    w: np.ndarray, r: np.ndarray, n_knots: int,
    wave_min: float, wave_max: float,
) -> np.ndarray:
    """Fit a cubic spline with n_knots interior knots, return fitted values."""
    try:
        inner_knots = np.linspace(
            w.min() + 1e-3, w.max() - 1e-3, n_knots
        )
        spline = _safe_spline_fit(w, r, inner_knots)
        return spline(w)
    except Exception:
        # Fallback: use smoothing spline
        spline = UnivariateSpline(w, r, k=3, s=len(w))
        return spline(w)


def _fit_spline_full(
    w: np.ndarray, r: np.ndarray, n_knots: int,
    wave_min: float, wave_max: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit a cubic spline and return (fitted_at_data, wave_dense, flux_dense).

    The dense grid is used for serialization/evaluation later.
    """
    from scipy.interpolate import LSQUnivariateSpline

    try:
        inner_knots = np.linspace(w.min() + 1e-3, w.max() - 1e-3, n_knots)
        spline = _safe_spline_fit(w, r, inner_knots)
    except Exception:
        spline = UnivariateSpline(w, r, k=3, s=len(w))

    fitted_at_data = spline(w)

    # Store a dense evaluation for serialization
    wave_dense = np.linspace(wave_min, wave_max, max(500, 2 * len(w)))
    flux_dense = spline(wave_dense)

    return fitted_at_data, wave_dense, flux_dense


def _safe_spline_fit(w, r, inner_knots):
    """Fit LSQUnivariateSpline, removing any knots that coincide with data edges."""
    from scipy.interpolate import LSQUnivariateSpline

    # Ensure knots are strictly inside data range
    inner_knots = inner_knots[
        (inner_knots > w.min()) & (inner_knots < w.max())
    ]
    if len(inner_knots) == 0:
        raise ValueError("No valid interior knots")
    return LSQUnivariateSpline(w, r, inner_knots, k=3)


def _fit_savgol(
    w: np.ndarray, r: np.ndarray, window_length: int,
) -> np.ndarray:
    """Apply Savitzky-Golay smoothing to the sensitivity ratio.

    Parameters
    ----------
    w : wavelength array (for sorting, not used in savgol itself)
    r : ratio values to smooth
    window_length : int
        Must be odd and >= 5.  If even, incremented by 1.
    """
    if window_length < 5:
        window_length = 5
    if window_length % 2 == 0:
        window_length += 1
    # Ensure window doesn't exceed data length
    if window_length > len(r):
        window_length = len(r)
        if window_length % 2 == 0:
            window_length -= 1
        if window_length < 5:
            return r.copy()  # too few points to smooth

    # polyorder = 3 is a good default for smooth sensitivity curves
    return savgol_filter(r, window_length, polyorder=3)


# ============================================================================
# 6.3  Combine per-segment sensitivity functions into a global curve
# ============================================================================

@dataclass
class GlobalSensitivity:
    """Container for the combined global sensitivity function.

    Stores both the per-segment fits and the combined global fit.
    """
    segment_fits: List[SensitivityFit] = field(default_factory=list)
    global_fit: Optional[SensitivityFit] = None
    grey_shifts: Dict[str, float] = field(default_factory=dict)
    wave_min: float = 0.0
    wave_max: float = 0.0
    approach: str = ''  # 'per_segment' or 'stitched'
    metadata: dict = field(default_factory=dict)

    def __call__(self, wavelength: np.ndarray) -> np.ndarray:
        """Evaluate the global sensitivity function.

        If a global fit exists, use it.  Otherwise, evaluate the
        per-segment fits and combine (using the segment whose domain
        contains each wavelength).
        """
        wavelength = np.asarray(wavelength, dtype=np.float64)

        if self.global_fit is not None:
            return self.global_fit(wavelength)

        # Fall back to per-segment evaluation
        return self._evaluate_piecewise(wavelength)

    def _evaluate_piecewise(self, wavelength: np.ndarray) -> np.ndarray:
        """Evaluate piecewise from per-segment fits.

        For wavelengths in overlap regions, average the two segment values.
        """
        result = np.full_like(wavelength, np.nan)
        counts = np.zeros_like(wavelength, dtype=np.float64)

        for fit in self.segment_fits:
            vals = fit(wavelength)
            valid = np.isfinite(vals)
            result[valid & (counts == 0)] = 0.0  # initialize accumulator
            # Use nansum-like logic
            finite_result = np.where(np.isfinite(result), result, 0.0)
            result = np.where(valid, finite_result + vals, result)
            counts[valid] += 1.0

        # Average where multiple segments contributed
        multi = counts > 1
        result[multi] /= counts[multi]

        return result

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dictionary."""
        return {
            'segment_fits': [f.to_dict() for f in self.segment_fits],
            'global_fit': self.global_fit.to_dict() if self.global_fit else None,
            'grey_shifts': self.grey_shifts,
            'wave_min': self.wave_min,
            'wave_max': self.wave_max,
            'approach': self.approach,
            'metadata': self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'GlobalSensitivity':
        """Reconstruct from a dictionary."""
        seg_fits = [SensitivityFit.from_dict(f) for f in d['segment_fits']]
        gf = None
        if d.get('global_fit') is not None:
            gf = SensitivityFit.from_dict(d['global_fit'])
        return cls(
            segment_fits=seg_fits,
            global_fit=gf,
            grey_shifts=d.get('grey_shifts', {}),
            wave_min=d.get('wave_min', 0.0),
            wave_max=d.get('wave_max', 0.0),
            approach=d.get('approach', ''),
            metadata=d.get('metadata', {}),
        )


def combine_segment_sensitivities(
    segment_fits: List[SensitivityFit],
    fit_global: bool = True,
    global_method: str = 'chebyshev',
    global_order: int = 6,
    sigma_clip: float = 3.0,
    max_iter: int = 3,
    n_eval_points: int = 1000,
    metadata: Optional[dict] = None,
) -> GlobalSensitivity:
    """Combine per-segment sensitivity fits into a global sensitivity curve.

    Steps:
    1. Estimate grey-shift corrections between adjacent segments by
       comparing their values in overlap regions.
    2. Apply grey shifts to bring all segments to a common scale.
    3. Optionally fit a single global smooth function across the combined data.

    Parameters
    ----------
    segment_fits : list of SensitivityFit
        Per-segment sensitivity fits, ordered by wavelength.
    fit_global : bool
        If True, fit a single global function across all segments.
    global_method : str
        Fit method for the global function.
    global_order : int
        Order for the global fit.
    sigma_clip : float
        Sigma-clip threshold for the global fit.
    max_iter : int
        Max iterations for global fit sigma clipping.
    n_eval_points : int
        Number of points to evaluate each segment fit on for building
        the global fit dataset.
    metadata : dict, optional

    Returns
    -------
    GlobalSensitivity
    """
    if len(segment_fits) == 0:
        raise ValueError("No segment fits provided")

    # Sort by wave_min
    segment_fits = sorted(segment_fits, key=lambda f: f.wave_min)

    # Step 1: Estimate grey shifts between adjacent segments
    grey_shifts = _estimate_grey_shifts(segment_fits, n_eval_points)

    # Apply grey shifts to the segment fits
    for fit in segment_fits:
        shift = grey_shifts.get(fit.segment_id, 1.0)
        fit.grey_shift = shift

    # Compute global domain
    wave_min = min(f.wave_min for f in segment_fits)
    wave_max = max(f.wave_max for f in segment_fits)

    result = GlobalSensitivity(
        segment_fits=segment_fits,
        grey_shifts=grey_shifts,
        wave_min=wave_min,
        wave_max=wave_max,
        approach='per_segment',
        metadata=metadata or {},
    )

    # Step 2: Optionally fit a global function
    if fit_global and len(segment_fits) > 1:
        # Build a combined dataset from all segment fits
        all_wave = []
        all_sens = []
        for fit in segment_fits:
            w = np.linspace(fit.wave_min, fit.wave_max, n_eval_points)
            s = fit(w)
            valid = np.isfinite(s)
            all_wave.append(w[valid])
            all_sens.append(s[valid])

        all_wave = np.concatenate(all_wave)
        all_sens = np.concatenate(all_sens)

        # Sort by wavelength
        order = np.argsort(all_wave)
        all_wave = all_wave[order]
        all_sens = all_sens[order]

        try:
            global_fit = fit_sensitivity(
                all_wave, all_sens,
                method=global_method,
                order=global_order,
                sigma_clip=sigma_clip,
                max_iter=max_iter,
                segment_id='global',
                metadata=metadata,
            )
            result.global_fit = global_fit
            logger.info(
                "Global sensitivity fit: RMS=%.4e over %.0f--%.0f A",
                global_fit.rms_residual, wave_min, wave_max
            )
        except ValueError as exc:
            warnings.warn(
                f"Could not fit global sensitivity function: {exc}. "
                f"Using piecewise per-segment evaluation."
            )

    return result


def _estimate_grey_shifts(
    segment_fits: List[SensitivityFit],
    n_eval: int = 200,
) -> Dict[str, float]:
    """Estimate multiplicative grey-shift corrections between adjacent segments.

    The first segment is used as the reference (shift = 1.0). Each subsequent
    segment is shifted to match the previous one in the overlap region.

    Returns a dict mapping segment_id -> grey_shift_factor.
    """
    shifts = {}
    if len(segment_fits) == 0:
        return shifts

    # First segment is the reference
    shifts[segment_fits[0].segment_id] = 1.0

    for i in range(1, len(segment_fits)):
        prev = segment_fits[i - 1]
        curr = segment_fits[i]

        # Find overlap region
        overlap_start = max(prev.wave_min, curr.wave_min)
        overlap_end = min(prev.wave_max, curr.wave_max)

        if overlap_end <= overlap_start:
            # No overlap -- propagate previous shift
            shifts[curr.segment_id] = shifts.get(prev.segment_id, 1.0)
            logger.warning(
                "No overlap between segments '%s' and '%s'; "
                "carrying forward grey shift",
                prev.segment_id, curr.segment_id
            )
            continue

        # Evaluate both fits in the overlap region
        w_overlap = np.linspace(overlap_start, overlap_end, n_eval)
        s_prev = prev(w_overlap)
        s_curr = curr(w_overlap)

        # Use only finite values
        valid = np.isfinite(s_prev) & np.isfinite(s_curr) & (s_curr > 0)
        if np.sum(valid) < 3:
            shifts[curr.segment_id] = shifts.get(prev.segment_id, 1.0)
            continue

        # Grey shift = median ratio of prev to curr in overlap
        ratios = s_prev[valid] / s_curr[valid]
        grey = float(np.median(ratios))

        # Compound with the previous segment's shift
        prev_shift = shifts.get(prev.segment_id, 1.0)
        shifts[curr.segment_id] = prev_shift * grey

        logger.info(
            "Grey shift segment '%s' -> '%s': %.4f (cumulative: %.4f)",
            prev.segment_id, curr.segment_id, grey,
            shifts[curr.segment_id]
        )

    return shifts


# ============================================================================
# 6.4  Per-segment vs. stitched-first approaches
# ============================================================================

def derive_sensitivity_per_segment(
    segments_obs: List[Tuple[np.ndarray, np.ndarray]],
    wavelength_ref: np.ndarray,
    flux_ref: np.ndarray,
    masks: Optional[List[np.ndarray]] = None,
    exptimes: Optional[List[float]] = None,
    airmasses: Optional[List[float]] = None,
    extinction_curve: Optional[Callable] = None,
    segment_ids: Optional[List[str]] = None,
    fit_method: str = 'chebyshev',
    fit_order: int = 5,
    sigma_clip: float = 3.0,
    max_iter: int = 5,
    fit_global: bool = True,
    global_method: str = 'chebyshev',
    global_order: int = 6,
    metadata: Optional[dict] = None,
) -> GlobalSensitivity:
    """Derive the sensitivity function per-segment, then combine.

    This is the recommended approach: derive S(lambda) for each segment
    independently, then combine with grey-shift corrections. More robust
    to stitching artifacts.

    Parameters
    ----------
    segments_obs : list of (wavelength, flux) tuples
        Each element is (wavelength_array, flux_array) for one segment.
    wavelength_ref : array
        Reference spectrum wavelength grid.
    flux_ref : array
        Reference flux (physical units).
    masks : list of bool arrays, optional
        Per-segment masks (True = good).
    exptimes : list of float, optional
        Per-segment exposure times in seconds.
    airmasses : list of float, optional
        Per-segment airmasses.
    extinction_curve : callable, optional
    segment_ids : list of str, optional
    fit_method, fit_order, sigma_clip, max_iter : fit parameters
    fit_global : bool
        Whether to also fit a global sensitivity curve.
    global_method, global_order : global fit parameters
    metadata : dict, optional

    Returns
    -------
    GlobalSensitivity
    """
    n_seg = len(segments_obs)
    if segment_ids is None:
        segment_ids = [f'seg_{i:02d}' for i in range(n_seg)]
    if masks is None:
        masks = [None] * n_seg
    if exptimes is None:
        exptimes = [None] * n_seg
    if airmasses is None:
        airmasses = [None] * n_seg

    segment_fits = []

    for i in range(n_seg):
        w_obs, f_obs = segments_obs[i]

        logger.info(
            "Processing segment %d/%d ('%s')", i + 1, n_seg, segment_ids[i]
        )

        # Compute raw ratio
        w, ratio, mask_out = compute_sensitivity_ratio(
            w_obs, f_obs,
            wavelength_ref, flux_ref,
            exptime=exptimes[i],
            mask=masks[i],
            airmass=airmasses[i],
            extinction_curve=extinction_curve,
        )

        # Fit smooth function
        try:
            fit = fit_sensitivity(
                w, ratio, mask=mask_out,
                method=fit_method,
                order=fit_order,
                sigma_clip=sigma_clip,
                max_iter=max_iter,
                segment_id=segment_ids[i],
                metadata=metadata,
            )
            segment_fits.append(fit)
        except ValueError as exc:
            logger.warning(
                "Could not fit segment '%s': %s", segment_ids[i], exc
            )

    if len(segment_fits) == 0:
        raise ValueError("No segments could be fit")

    # Combine into a global sensitivity
    result = combine_segment_sensitivities(
        segment_fits,
        fit_global=fit_global,
        global_method=global_method,
        global_order=global_order,
        sigma_clip=sigma_clip,
        max_iter=max_iter,
        metadata=metadata,
    )
    result.approach = 'per_segment'

    return result


def derive_sensitivity_stitched(
    wavelength_stitched: np.ndarray,
    flux_stitched: np.ndarray,
    wavelength_ref: np.ndarray,
    flux_ref: np.ndarray,
    mask: Optional[np.ndarray] = None,
    exptime: Optional[float] = None,
    airmass: Optional[float] = None,
    extinction_curve: Optional[Callable] = None,
    fit_method: str = 'chebyshev',
    fit_order: int = 6,
    sigma_clip: float = 3.0,
    max_iter: int = 5,
    metadata: Optional[dict] = None,
) -> GlobalSensitivity:
    """Derive the sensitivity function from an already-stitched spectrum.

    Simpler code path: stitch the observed segments first (Step 4),
    then derive one global S(lambda). More sensitive to stitching artifacts
    but simpler.

    Parameters
    ----------
    wavelength_stitched : array
        Stitched wavelength grid.
    flux_stitched : array
        Stitched flux (counts or counts/s).
    wavelength_ref, flux_ref : arrays
        Reference spectrum.
    mask : array of bool, optional
    exptime : float, optional
    airmass : float, optional
    extinction_curve : callable, optional
    fit_method, fit_order, sigma_clip, max_iter : fit parameters
    metadata : dict, optional

    Returns
    -------
    GlobalSensitivity
    """
    w, ratio, mask_out = compute_sensitivity_ratio(
        wavelength_stitched, flux_stitched,
        wavelength_ref, flux_ref,
        exptime=exptime,
        mask=mask,
        airmass=airmass,
        extinction_curve=extinction_curve,
    )

    fit = fit_sensitivity(
        w, ratio, mask=mask_out,
        method=fit_method,
        order=fit_order,
        sigma_clip=sigma_clip,
        max_iter=max_iter,
        segment_id='stitched',
        metadata=metadata,
    )

    result = GlobalSensitivity(
        segment_fits=[fit],
        global_fit=fit,
        wave_min=float(w[mask_out].min()),
        wave_max=float(w[mask_out].max()),
        approach='stitched',
        metadata=metadata or {},
    )

    return result


# ============================================================================
# 6.5  Telluric feature handling in the sensitivity function
# ============================================================================

def build_sensitivity_mask(
    wavelength: np.ndarray,
    quality_mask: Optional[np.ndarray] = None,
    mask_telluric: bool = True,
    mask_stellar: bool = True,
    telluric_bands: Optional[List[Tuple[float, float]]] = None,
    stellar_lines: Optional[np.ndarray] = None,
    stellar_half_widths: Optional[np.ndarray] = None,
    balmer_half_width: float = DEFAULT_BALMER_HALF_WIDTH,
    metal_half_width: float = DEFAULT_METAL_HALF_WIDTH,
    edge_fraction: float = 0.05,
) -> np.ndarray:
    """Build a comprehensive mask for sensitivity function fitting.

    Combines masks from the quality assessment step with additional
    masking of telluric bands and stellar lines. The sensitivity function
    should be smooth, so stellar and telluric features must be excluded.

    Parameters
    ----------
    wavelength : array (N,)
        Wavelength grid in Angstroms.
    quality_mask : array of bool (N,), optional
        Mask from Step 3 quality assessment (True = good).
        If None, all pixels start as good.
    mask_telluric : bool
        Whether to mask default telluric bands.
    mask_stellar : bool
        Whether to mask known stellar absorption lines.
    telluric_bands : list of (float, float), optional
        Custom telluric bands.  Uses defaults if None.
    stellar_lines : array, optional
        Custom stellar line wavelengths.
    stellar_half_widths : array, optional
        Half-widths for custom stellar lines.
    balmer_half_width : float
        Half-width for Balmer line masking.
    metal_half_width : float
        Half-width for metal line masking.
    edge_fraction : float
        Fraction of segment to mask at each edge (0--0.5).

    Returns
    -------
    mask : array of bool (N,)
        Combined mask (True = good pixel for sensitivity fitting).
    """
    wavelength = np.asarray(wavelength)
    n = len(wavelength)

    if quality_mask is not None:
        mask = np.asarray(quality_mask, dtype=bool).copy()
    else:
        mask = np.ones(n, dtype=bool)

    # Mask edges
    if edge_fraction > 0:
        n_edge = max(1, int(n * edge_fraction))
        mask[:n_edge] = False
        mask[-n_edge:] = False

    # Mask telluric bands
    if mask_telluric:
        bands = telluric_bands if telluric_bands is not None else TELLURIC_BANDS
        for band_lo, band_hi in bands:
            in_band = (wavelength >= band_lo) & (wavelength <= band_hi)
            mask[in_band] = False

    # Mask stellar absorption lines
    if mask_stellar:
        # Balmer lines
        for line_center in BALMER_LINES_AA:
            in_line = np.abs(wavelength - line_center) <= balmer_half_width
            mask[in_line] = False

        # Metal lines
        for line_center in METAL_LINES_AA:
            in_line = np.abs(wavelength - line_center) <= metal_half_width
            mask[in_line] = False

        # Custom lines
        if stellar_lines is not None:
            stellar_lines = np.asarray(stellar_lines)
            if stellar_half_widths is None:
                stellar_half_widths = np.full(len(stellar_lines), 5.0)
            stellar_half_widths = np.asarray(stellar_half_widths)

            for lc, hw in zip(stellar_lines, stellar_half_widths):
                in_line = np.abs(wavelength - lc) <= hw
                mask[in_line] = False

    return mask


# ============================================================================
# 6.6  Non-photometric condition handling
# ============================================================================

def estimate_grey_shift_multi_obs(
    sensitivity_fits: List[SensitivityFit],
    reference_fit: Optional[SensitivityFit] = None,
    wave_range: Optional[Tuple[float, float]] = None,
    n_eval: int = 200,
) -> Tuple[List[float], float]:
    """Estimate grey-shift corrections from multiple observations.

    If multiple observations of the same standard exist (e.g. at different
    airmasses), the residual grey offset between them can be determined.
    This implements the IRAF-style grey-shift approach.

    For the single-observation case, this returns a unity shift and documents
    the systematic.

    Parameters
    ----------
    sensitivity_fits : list of SensitivityFit
        Sensitivity fits from different observations.
    reference_fit : SensitivityFit, optional
        A reference fit to normalize against.  If None, the first fit
        is used as the reference.
    wave_range : tuple of (float, float), optional
        Wavelength range over which to compute the grey shift.
    n_eval : int
        Number of evaluation points.

    Returns
    -------
    shifts : list of float
        Grey-shift factors for each observation (first = 1.0).
    rms_scatter : float
        RMS scatter among the shifts (indicates photometric stability).
    """
    if len(sensitivity_fits) == 0:
        return [], 0.0

    if len(sensitivity_fits) == 1:
        logger.info(
            "Single observation -- grey shift is undefined. "
            "Calibration is relative, not absolute, unless external "
            "photometry is available."
        )
        return [1.0], 0.0

    # Use the first fit as reference unless one is specified
    if reference_fit is None:
        reference_fit = sensitivity_fits[0]

    # Determine evaluation range
    if wave_range is None:
        wave_range = (reference_fit.wave_min, reference_fit.wave_max)

    w_eval = np.linspace(wave_range[0], wave_range[1], n_eval)
    ref_vals = reference_fit(w_eval)
    valid_ref = np.isfinite(ref_vals) & (ref_vals > 0)

    shifts = []
    for fit in sensitivity_fits:
        vals = fit(w_eval)
        valid = valid_ref & np.isfinite(vals) & (vals > 0)

        if np.sum(valid) < 3:
            shifts.append(1.0)
            continue

        # Grey shift = median ratio of reference to this observation
        ratios = ref_vals[valid] / vals[valid]
        shifts.append(float(np.median(ratios)))

    rms_scatter = float(np.std(shifts)) if len(shifts) > 1 else 0.0

    logger.info(
        "Grey shifts from %d observations: %s (RMS scatter: %.4f)",
        len(sensitivity_fits),
        [f'{s:.4f}' for s in shifts],
        rms_scatter
    )

    return shifts, rms_scatter


# ============================================================================
# 6.7  Sensitivity function output format (serialization)
# ============================================================================

def save_sensitivity(
    sensitivity: Union[SensitivityFit, GlobalSensitivity],
    filepath: Union[str, Path],
) -> None:
    """Save a sensitivity function to a JSON file.

    Parameters
    ----------
    sensitivity : SensitivityFit or GlobalSensitivity
        The sensitivity function to save.
    filepath : str or Path
        Output file path (should end in .json).
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    data = sensitivity.to_dict()

    # Add a type tag so we know what to reconstruct
    if isinstance(sensitivity, GlobalSensitivity):
        data['_type'] = 'GlobalSensitivity'
    else:
        data['_type'] = 'SensitivityFit'

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

    logger.info("Saved sensitivity function to %s", filepath)


def load_sensitivity(
    filepath: Union[str, Path],
) -> Union[SensitivityFit, GlobalSensitivity]:
    """Load a sensitivity function from a JSON file.

    Parameters
    ----------
    filepath : str or Path

    Returns
    -------
    SensitivityFit or GlobalSensitivity
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Sensitivity file not found: {filepath}")

    with open(filepath, 'r') as f:
        data = json.load(f)

    type_tag = data.pop('_type', 'SensitivityFit')

    if type_tag == 'GlobalSensitivity':
        return GlobalSensitivity.from_dict(data)
    else:
        return SensitivityFit.from_dict(data)


# ============================================================================
# 6.8  Diagnostic outputs (plotting)
# ============================================================================

def plot_sensitivity_ratio(
    wavelength: np.ndarray,
    ratio: np.ndarray,
    mask: np.ndarray,
    fit: Optional[SensitivityFit] = None,
    segment_id: str = '',
    ax=None,
    save_path: Optional[str] = None,
):
    """Plot the raw sensitivity ratio with the smooth fit overlaid.

    Parameters
    ----------
    wavelength : array
    ratio : array
    mask : array of bool (True = good)
    fit : SensitivityFit, optional
        If provided, the fitted curve is overlaid.
    segment_id : str
    ax : matplotlib Axes, optional
    save_path : str, optional
        If provided, save the figure to this path.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        own_fig = True
    else:
        fig = ax.get_figure()
        own_fig = False

    # Plot masked points lightly
    masked = ~mask
    if np.any(masked):
        ax.plot(
            wavelength[masked], ratio[masked],
            '.', color='lightgray', alpha=0.3, ms=2, label='Masked',
        )

    # Plot good points
    if np.any(mask):
        ax.plot(
            wavelength[mask], ratio[mask],
            '.', color='steelblue', alpha=0.5, ms=3, label='Data',
        )

    # Overlay fit
    if fit is not None:
        w_dense = np.linspace(wavelength.min(), wavelength.max(), 500)
        s_dense = fit(w_dense)
        ax.plot(
            w_dense, s_dense,
            '-', color='crimson', lw=2,
            label=f'{fit.method} order={fit.order}',
        )

    ax.set_xlabel('Wavelength (A)')
    ax.set_ylabel('Sensitivity (F_ref / C_obs)')
    title = 'Sensitivity Ratio'
    if segment_id:
        title += f' -- {segment_id}'
    ax.set_title(title)
    ax.legend(fontsize=8)

    if save_path and own_fig:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info("Saved ratio plot to %s", save_path)

    return ax


def plot_sensitivity_residuals(
    wavelength: np.ndarray,
    ratio: np.ndarray,
    mask: np.ndarray,
    fit: SensitivityFit,
    segment_id: str = '',
    ax=None,
    save_path: Optional[str] = None,
):
    """Plot residuals (data - fit) to assess fit quality.

    Parameters
    ----------
    wavelength, ratio, mask, fit, segment_id : see plot_sensitivity_ratio
    ax : matplotlib Axes, optional
    save_path : str, optional
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 3))
        own_fig = True
    else:
        fig = ax.get_figure()
        own_fig = False

    w_good = wavelength[mask]
    r_good = ratio[mask]
    fit_vals = fit(w_good)
    residuals = r_good - fit_vals

    ax.plot(w_good, residuals, '.', color='steelblue', alpha=0.5, ms=3)
    ax.axhline(0, color='crimson', lw=1, ls='--')
    ax.set_xlabel('Wavelength (A)')
    ax.set_ylabel('Residual')
    title = 'Sensitivity Fit Residuals'
    if segment_id:
        title += f' -- {segment_id}'
    ax.set_title(title)

    # Annotate with RMS
    ax.text(
        0.02, 0.95,
        f'RMS = {fit.rms_residual:.4e}',
        transform=ax.transAxes, fontsize=9, va='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
    )

    if save_path and own_fig:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info("Saved residuals plot to %s", save_path)

    return ax


def plot_global_sensitivity(
    global_sens: GlobalSensitivity,
    show_segments: bool = True,
    magnitude_units: bool = False,
    ax=None,
    save_path: Optional[str] = None,
):
    """Plot the combined global sensitivity function.

    Parameters
    ----------
    global_sens : GlobalSensitivity
    show_segments : bool
        If True, also show per-segment fits in lighter colors.
    magnitude_units : bool
        If True, plot in magnitude units (the "zeropoint curve").
    ax : matplotlib Axes, optional
    save_path : str, optional
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 5))
        own_fig = True
    else:
        fig = ax.get_figure()
        own_fig = False

    w_global = np.linspace(global_sens.wave_min, global_sens.wave_max, 2000)

    # Plot per-segment fits
    if show_segments:
        colors = plt.cm.tab10(np.linspace(0, 1, len(global_sens.segment_fits)))
        for i, fit in enumerate(global_sens.segment_fits):
            w_seg = np.linspace(fit.wave_min, fit.wave_max, 500)
            if magnitude_units:
                s_seg = fit.to_magnitude(w_seg)
            else:
                s_seg = fit(w_seg)
            ax.plot(
                w_seg, s_seg,
                '-', color=colors[i], alpha=0.5, lw=1,
                label=f'{fit.segment_id}',
            )

    # Plot global fit
    if global_sens.global_fit is not None:
        if magnitude_units:
            s_global = global_sens.global_fit.to_magnitude(w_global)
        else:
            s_global = global_sens.global_fit(w_global)
        ax.plot(
            w_global, s_global,
            '-', color='black', lw=2, label='Global fit',
        )
    else:
        # Plot piecewise evaluation
        if magnitude_units:
            s_global = np.full_like(w_global, np.nan)
            for fit in global_sens.segment_fits:
                vals = fit.to_magnitude(w_global)
                valid = np.isfinite(vals)
                s_global[valid] = vals[valid]
        else:
            s_global = global_sens(w_global)
        ax.plot(
            w_global, s_global,
            '-', color='black', lw=2, label='Piecewise',
        )

    ax.set_xlabel('Wavelength (A)')
    ylabel = 'Sensitivity (mag)' if magnitude_units else 'Sensitivity'
    ax.set_ylabel(ylabel)
    title = f'Global Sensitivity Function ({global_sens.approach})'
    ax.set_title(title)
    ax.legend(fontsize=8, ncol=2)

    if save_path and own_fig:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info("Saved global sensitivity plot to %s", save_path)

    return ax


def plot_sensitivity_diagnostic(
    wavelength: np.ndarray,
    ratio: np.ndarray,
    mask: np.ndarray,
    fit: SensitivityFit,
    segment_id: str = '',
    save_path: Optional[str] = None,
):
    """Combined diagnostic plot: ratio + fit (top) and residuals (bottom).

    Parameters
    ----------
    wavelength, ratio, mask, fit, segment_id : see plot_sensitivity_ratio
    save_path : str, optional
    """
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 7), gridspec_kw={'height_ratios': [3, 1]},
        sharex=True,
    )

    plot_sensitivity_ratio(
        wavelength, ratio, mask, fit=fit, segment_id=segment_id, ax=ax1
    )
    plot_sensitivity_residuals(
        wavelength, ratio, mask, fit, segment_id=segment_id, ax=ax2
    )

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info("Saved diagnostic plot to %s", save_path)

    return fig


def plot_method_comparison(
    wavelength: np.ndarray,
    ratio: np.ndarray,
    mask: np.ndarray,
    methods: Optional[List[str]] = None,
    orders: Optional[Dict[str, List[int]]] = None,
    segment_id: str = '',
    save_path: Optional[str] = None,
):
    """Compare different fit methods and orders on the same data.

    Useful for the paper: test several methods and report which performs
    best for JJMO-quality data.

    Parameters
    ----------
    wavelength, ratio, mask : arrays
    methods : list of str, optional
        Fit methods to compare.  Default: all four.
    orders : dict mapping method -> list of orders, optional
        Orders to test for each method.
    segment_id : str
    save_path : str, optional
    """
    import matplotlib.pyplot as plt

    if methods is None:
        methods = ['chebyshev', 'legendre', 'spline', 'savgol']
    if orders is None:
        orders = {
            'chebyshev': [3, 5, 7],
            'legendre': [3, 5, 7],
            'spline': [3, 5, 8],
            'savgol': [31, 51, 101],
        }

    fig, axes = plt.subplots(
        len(methods), 1, figsize=(12, 4 * len(methods)), sharex=True
    )
    if len(methods) == 1:
        axes = [axes]

    for ax, method in zip(axes, methods):
        # Plot data
        ax.plot(
            wavelength[mask], ratio[mask],
            '.', color='lightgray', alpha=0.3, ms=2,
        )

        method_orders = orders.get(method, [5])
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(method_orders)))

        for color, order in zip(colors, method_orders):
            try:
                fit = fit_sensitivity(
                    wavelength, ratio, mask=mask,
                    method=method, order=order,
                    segment_id=f'{segment_id}_{method}_{order}',
                )
                w_dense = np.linspace(wavelength.min(), wavelength.max(), 500)
                s_dense = fit(w_dense)
                ax.plot(
                    w_dense, s_dense,
                    '-', color=color, lw=1.5,
                    label=f'order={order} (RMS={fit.rms_residual:.3e})',
                )
            except ValueError as exc:
                ax.text(
                    0.5, 0.5, f'Failed: {exc}',
                    transform=ax.transAxes, ha='center',
                )

        ax.set_ylabel('Sensitivity')
        ax.set_title(f'{method.capitalize()}')
        ax.legend(fontsize=7)

    axes[-1].set_xlabel('Wavelength (A)')
    fig.suptitle(
        f'Fit Method Comparison -- {segment_id}',
        fontsize=13, y=1.01,
    )
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info("Saved method comparison to %s", save_path)

    return fig


# ============================================================================
# Convenience: full pipeline entry point
# ============================================================================

def derive_sensitivity(
    segments_obs: List[Tuple[np.ndarray, np.ndarray]],
    wavelength_ref: np.ndarray,
    flux_ref: np.ndarray,
    approach: str = 'per_segment',
    masks: Optional[List[np.ndarray]] = None,
    exptimes: Optional[List[float]] = None,
    airmasses: Optional[List[float]] = None,
    extinction_curve: Optional[Callable] = None,
    segment_ids: Optional[List[str]] = None,
    fit_method: str = 'chebyshev',
    fit_order: int = 5,
    sigma_clip: float = 3.0,
    max_iter: int = 5,
    fit_global: bool = True,
    global_method: str = 'chebyshev',
    global_order: int = 6,
    stitcher: Optional[Callable] = None,
    metadata: Optional[dict] = None,
) -> GlobalSensitivity:
    """Main entry point: derive sensitivity using either approach.

    Parameters
    ----------
    segments_obs : list of (wavelength, flux) tuples
    wavelength_ref, flux_ref : reference spectrum arrays
    approach : str
        'per_segment' (recommended) or 'stitched'.
    masks : list of bool arrays, optional
    exptimes : list of float, optional
    airmasses : list of float, optional
    extinction_curve : callable, optional
    segment_ids : list of str, optional
    fit_method, fit_order, sigma_clip, max_iter : fit params
    fit_global : bool
    global_method, global_order : global fit params
    stitcher : callable, optional
        Function that stitches segments.  Required if approach='stitched'.
        Signature: stitcher(segments_obs) -> (wavelength, flux, mask)
    metadata : dict, optional

    Returns
    -------
    GlobalSensitivity
    """
    if approach == 'per_segment':
        return derive_sensitivity_per_segment(
            segments_obs=segments_obs,
            wavelength_ref=wavelength_ref,
            flux_ref=flux_ref,
            masks=masks,
            exptimes=exptimes,
            airmasses=airmasses,
            extinction_curve=extinction_curve,
            segment_ids=segment_ids,
            fit_method=fit_method,
            fit_order=fit_order,
            sigma_clip=sigma_clip,
            max_iter=max_iter,
            fit_global=fit_global,
            global_method=global_method,
            global_order=global_order,
            metadata=metadata,
        )

    elif approach == 'stitched':
        if stitcher is None:
            # Simple concatenation + sort as a basic stitcher
            all_w = np.concatenate([s[0] for s in segments_obs])
            all_f = np.concatenate([s[1] for s in segments_obs])
            order = np.argsort(all_w)
            w_stitched = all_w[order]
            f_stitched = all_f[order]
            m_stitched = None
        else:
            w_stitched, f_stitched, m_stitched = stitcher(segments_obs)

        # Combine masks if provided
        if masks is not None and m_stitched is None:
            m_stitched = np.concatenate(masks)
            order = np.argsort(np.concatenate([s[0] for s in segments_obs]))
            m_stitched = m_stitched[order]

        # Use mean exptime and airmass for the stitched case
        mean_exptime = None
        if exptimes is not None:
            valid_exp = [e for e in exptimes if e is not None]
            if valid_exp:
                mean_exptime = np.mean(valid_exp)

        mean_airmass = None
        if airmasses is not None:
            valid_am = [a for a in airmasses if a is not None]
            if valid_am:
                mean_airmass = np.mean(valid_am)

        return derive_sensitivity_stitched(
            wavelength_stitched=w_stitched,
            flux_stitched=f_stitched,
            wavelength_ref=wavelength_ref,
            flux_ref=flux_ref,
            mask=m_stitched,
            exptime=mean_exptime,
            airmass=mean_airmass,
            extinction_curve=extinction_curve,
            fit_method=fit_method,
            fit_order=fit_order,
            sigma_clip=sigma_clip,
            max_iter=max_iter,
            metadata=metadata,
        )

    else:
        raise ValueError(
            f"Unknown approach '{approach}'. Use 'per_segment' or 'stitched'."
        )
