"""
stitching.py - Segment Stitching & Cross-Normalization
=======================================================

Step 4 of the JJMO Spectral Flux Calibration Pipeline.

Combines individual ~500 A spectral segments into a single continuous
spectrum, handling overlapping regions, gaps, and segment-to-segment
flux offsets. Supports both pre-calibration (raw counts) and
post-calibration (flux-calibrated) stitching workflows.

The module works with specutils Spectrum1D objects as the canonical
container, but also provides numpy-level functions for flexibility.
"""

import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import median_filter

try:
    from astropy import units as u
    from astropy.nddata import StdDevUncertainty
    from specutils import Spectrum1D
    HAS_SPECUTILS = True
except ImportError:
    HAS_SPECUTILS = False
    warnings.warn("specutils not available; Spectrum1D support disabled.")

try:
    from spectres import spectres
    HAS_SPECTRES = True
except ImportError:
    HAS_SPECTRES = False
    warnings.warn("spectres not available; flux-conserving resampling disabled.")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class OverlapInfo:
    """Describes the overlap (or gap) between two adjacent segments."""
    idx_left: int              # Index of left segment
    idx_right: int             # Index of right segment
    wave_start: float          # Start wavelength of overlap region
    wave_end: float            # End wavelength of overlap region
    overlap_width: float       # Positive = overlap in Angstroms; negative = gap
    is_gap: bool               # True if there is a gap (no overlap)

    def __repr__(self):
        kind = "GAP" if self.is_gap else "OVERLAP"
        return (f"OverlapInfo(seg {self.idx_left}-{self.idx_right}: "
                f"{kind} {abs(self.overlap_width):.1f} A, "
                f"{self.wave_start:.1f}-{self.wave_end:.1f} A)")


@dataclass
class NormFactor:
    """Cross-normalization factor applied to a segment."""
    segment_idx: int
    factor: float               # Multiplicative factor applied
    factor_uncertainty: float   # Uncertainty on the factor
    method: str                 # 'median_ratio' or 'polynomial'
    overlap_idx: int            # Which overlap region informed this factor


@dataclass
class StitchResult:
    """Container for the stitched spectrum and diagnostic metadata."""
    wavelength: np.ndarray
    flux: np.ndarray
    uncertainty: np.ndarray
    mask: np.ndarray            # True = valid pixel, False = masked/interpolated
    interpolated: np.ndarray    # True = pixel was interpolated across a gap
    overlaps: List[OverlapInfo] = field(default_factory=list)
    norm_factors: List[NormFactor] = field(default_factory=list)
    reference_segment: int = -1

    def to_spectrum1d(self):
        """Convert to a specutils Spectrum1D object."""
        if not HAS_SPECUTILS:
            raise ImportError("specutils is required for Spectrum1D conversion.")
        wave = self.wavelength * u.AA
        flux_val = self.flux * u.ct  # counts; downstream can change units
        unc = StdDevUncertainty(self.uncertainty)
        # Encode mask: specutils mask convention is True = bad pixel
        spec_mask = ~self.mask
        return Spectrum1D(spectral_axis=wave, flux=flux_val,
                          uncertainty=unc, mask=spec_mask)


# ---------------------------------------------------------------------------
# Helper: extract numpy arrays from various segment formats
# ---------------------------------------------------------------------------

def _unpack_segment(segment):
    """
    Extract (wavelength, flux, uncertainty) arrays from a segment.

    Accepts:
      - Spectrum1D object
      - dict with keys 'wavelength', 'flux', and optionally 'uncertainty'/'mask'
      - tuple/list of (wavelength, flux) or (wavelength, flux, uncertainty)

    Returns:
      wavelength, flux, uncertainty, mask  (all numpy arrays)
      Uncertainty defaults to sqrt(|flux|) if not provided (Poisson estimate).
      Mask defaults to all True (all valid).
    """
    if HAS_SPECUTILS and isinstance(segment, Spectrum1D):
        wave = segment.spectral_axis.to(u.AA).value
        flux = segment.flux.value
        if segment.uncertainty is not None:
            unc = segment.uncertainty.array
        else:
            unc = np.sqrt(np.maximum(np.abs(flux), 1.0))
        if segment.mask is not None:
            # specutils convention: True = bad pixel; we invert to True = good
            mask = ~segment.mask
        else:
            mask = np.ones(len(wave), dtype=bool)
        return wave, flux, unc, mask

    if isinstance(segment, dict):
        wave = np.asarray(segment['wavelength'], dtype=float)
        flux = np.asarray(segment['flux'], dtype=float)
        unc = np.asarray(segment.get('uncertainty',
                                     np.sqrt(np.maximum(np.abs(flux), 1.0))),
                         dtype=float)
        mask = np.asarray(segment.get('mask', np.ones(len(wave), dtype=bool)),
                          dtype=bool)
        return wave, flux, unc, mask

    # tuple/list of arrays
    seg = list(segment)
    wave = np.asarray(seg[0], dtype=float)
    flux = np.asarray(seg[1], dtype=float)
    if len(seg) >= 3:
        unc = np.asarray(seg[2], dtype=float)
    else:
        unc = np.sqrt(np.maximum(np.abs(flux), 1.0))
    if len(seg) >= 4:
        mask = np.asarray(seg[3], dtype=bool)
    else:
        mask = np.ones(len(wave), dtype=bool)
    return wave, flux, unc, mask


def _ensure_sorted(wave, flux, unc, mask):
    """Ensure arrays are sorted by ascending wavelength."""
    if len(wave) < 2:
        return wave, flux, unc, mask
    if wave[0] > wave[-1]:
        # Descending order; reverse
        wave = wave[::-1].copy()
        flux = flux[::-1].copy()
        unc = unc[::-1].copy()
        mask = mask[::-1].copy()
    elif not np.all(np.diff(wave) > 0):
        # Not monotonic; sort
        order = np.argsort(wave)
        wave = wave[order]
        flux = flux[order]
        unc = unc[order]
        mask = mask[order]
    return wave, flux, unc, mask


# ---------------------------------------------------------------------------
# 4.1  Identify overlapping regions between adjacent segments
# ---------------------------------------------------------------------------

def find_overlaps(segments: list) -> List[OverlapInfo]:
    """
    Identify overlapping regions (or gaps) between adjacent spectral segments.

    Parameters
    ----------
    segments : list
        Ordered list of spectral segments (Spectrum1D, dict, or array tuples).
        Must be sorted by ascending wavelength (by starting wavelength).

    Returns
    -------
    list of OverlapInfo
        One entry per adjacent pair describing the overlap or gap.
    """
    overlaps = []
    for i in range(len(segments) - 1):
        w_left, _, _, _ = _ensure_sorted(*_unpack_segment(segments[i]))
        w_right, _, _, _ = _ensure_sorted(*_unpack_segment(segments[i + 1]))

        # Overlap region: max of the two start wavelengths to min of the two end wavelengths
        overlap_start = max(w_left.min(), w_right.min())
        overlap_end = min(w_left.max(), w_right.max())
        overlap_width = overlap_end - overlap_start

        is_gap = overlap_width <= 0
        if is_gap:
            # Gap: record the gap boundaries
            gap_start = min(w_left.max(), w_right.max())
            gap_end = max(w_left.min(), w_right.min())
            overlaps.append(OverlapInfo(
                idx_left=i, idx_right=i + 1,
                wave_start=gap_start, wave_end=gap_end,
                overlap_width=overlap_width, is_gap=True
            ))
        else:
            overlaps.append(OverlapInfo(
                idx_left=i, idx_right=i + 1,
                wave_start=overlap_start, wave_end=overlap_end,
                overlap_width=overlap_width, is_gap=False
            ))

    return overlaps


# ---------------------------------------------------------------------------
# 4.2  Cross-normalization in overlap regions
# ---------------------------------------------------------------------------

def estimate_segment_snr(flux, uncertainty=None, mask=None):
    """
    Estimate the median SNR of a segment.

    Parameters
    ----------
    flux : array
    uncertainty : array, optional
        If None, uses sqrt(|flux|) as Poisson estimate.
    mask : array of bool, optional
        True = valid pixel.

    Returns
    -------
    float : median SNR over valid pixels
    """
    if mask is None:
        mask = np.ones(len(flux), dtype=bool)
    if uncertainty is None:
        uncertainty = np.sqrt(np.maximum(np.abs(flux), 1.0))

    valid = mask & (uncertainty > 0) & np.isfinite(flux) & np.isfinite(uncertainty)
    if not np.any(valid):
        return 0.0
    snr = np.abs(flux[valid]) / uncertainty[valid]
    return float(np.median(snr))


def compute_normalization_factor(
    wave_left, flux_left, unc_left, mask_left,
    wave_right, flux_right, unc_right, mask_right,
    overlap: OverlapInfo,
    method: str = 'median_ratio'
) -> Tuple[float, float]:
    """
    Compute the multiplicative factor to bring the right segment's flux scale
    to match the left segment in the overlap region.

    Parameters
    ----------
    wave_left, flux_left, unc_left, mask_left : arrays for left segment
    wave_right, flux_right, unc_right, mask_right : arrays for right segment
    overlap : OverlapInfo
    method : str
        'median_ratio' - median of flux_left/flux_right in overlap
        'polynomial'   - low-order polynomial fit to flux ratio

    Returns
    -------
    factor : float
        Multiply right segment's flux by this to match left.
    factor_unc : float
        Uncertainty on the factor.
    """
    if overlap.is_gap:
        return 1.0, 0.0

    # Select overlap region in each segment
    ol_mask_l = ((wave_left >= overlap.wave_start) &
                 (wave_left <= overlap.wave_end) & mask_left)
    ol_mask_r = ((wave_right >= overlap.wave_start) &
                 (wave_right <= overlap.wave_end) & mask_right)

    if np.sum(ol_mask_l) < 5 or np.sum(ol_mask_r) < 5:
        warnings.warn(f"Too few valid pixels in overlap {overlap}; "
                      "using factor=1.0")
        return 1.0, 0.0

    # Interpolate right segment onto left segment's wavelength grid in overlap
    w_common = wave_left[ol_mask_l]
    f_left = flux_left[ol_mask_l]
    u_left = unc_left[ol_mask_l]

    # Linear interpolation of right segment onto common grid
    interp_func = interp1d(wave_right[ol_mask_r], flux_right[ol_mask_r],
                           kind='linear', bounds_error=False, fill_value=np.nan)
    f_right_interp = interp_func(w_common)

    interp_unc_func = interp1d(wave_right[ol_mask_r], unc_right[ol_mask_r],
                               kind='linear', bounds_error=False, fill_value=np.nan)
    u_right_interp = interp_unc_func(w_common)

    # Filter out NaN/invalid
    valid = (np.isfinite(f_right_interp) & np.isfinite(f_left) &
             (f_right_interp > 0) & (f_left > 0) &
             np.isfinite(u_left) & np.isfinite(u_right_interp))

    if np.sum(valid) < 3:
        warnings.warn(f"Too few valid overlap pixels after interpolation; "
                      "using factor=1.0")
        return 1.0, 0.0

    f_l = f_left[valid]
    f_r = f_right_interp[valid]
    u_l = u_left[valid]
    u_r = u_right_interp[valid]

    if method == 'median_ratio':
        ratios = f_l / f_r
        factor = float(np.median(ratios))
        # Uncertainty: MAD-based robust scatter divided by sqrt(N)
        mad = np.median(np.abs(ratios - factor))
        factor_unc = float(1.4826 * mad / np.sqrt(len(ratios)))

    elif method == 'polynomial':
        # Fit a low-order polynomial to the ratio vs wavelength
        # Use order 1 (linear) to capture mild wavelength-dependent correction
        w = w_common[valid]
        ratios = f_l / f_r
        # Inverse-variance weights from error propagation on ratio
        ratio_var = (u_l / f_r)**2 + (f_l * u_r / f_r**2)**2
        weights = np.where(ratio_var > 0, 1.0 / ratio_var, 1.0)
        coeffs = np.polyfit(w, ratios, deg=1, w=np.sqrt(weights))
        # Evaluate at midpoint for the representative factor
        w_mid = 0.5 * (overlap.wave_start + overlap.wave_end)
        factor = float(np.polyval(coeffs, w_mid))
        # Uncertainty from residuals
        residuals = ratios - np.polyval(coeffs, w)
        factor_unc = float(np.std(residuals) / np.sqrt(len(residuals)))
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    if factor <= 0 or not np.isfinite(factor):
        warnings.warn(f"Invalid normalization factor {factor}; using 1.0")
        return 1.0, 0.0

    return factor, factor_unc


def cross_normalize(
    segments: list,
    overlaps: Optional[List[OverlapInfo]] = None,
    reference: Union[int, str] = 'auto',
    method: str = 'median_ratio'
) -> Tuple[list, List[NormFactor]]:
    """
    Apply cross-normalization to bring all segments to a common flux scale.

    Strategy: choose a reference segment (highest SNR by default), then
    propagate normalization outward to neighbors via the overlap regions.

    Parameters
    ----------
    segments : list of spectral segments
    overlaps : list of OverlapInfo, optional
        If None, computed automatically.
    reference : int or 'auto'
        Index of the reference segment, or 'auto' to choose highest-SNR.
    method : str
        'median_ratio' or 'polynomial'

    Returns
    -------
    normalized_segments : list of (wavelength, flux, uncertainty, mask) tuples
    norm_factors : list of NormFactor objects
    """
    if overlaps is None:
        overlaps = find_overlaps(segments)

    n_seg = len(segments)
    # Unpack all segments
    unpacked = []
    for seg in segments:
        w, f, u, m = _ensure_sorted(*_unpack_segment(seg))
        unpacked.append((w, f.copy(), u.copy(), m.copy()))

    # Choose reference segment
    if reference == 'auto':
        snrs = [estimate_segment_snr(f, u, m) for w, f, u, m in unpacked]
        ref_idx = int(np.argmax(snrs))
    else:
        ref_idx = int(reference)

    # Cumulative normalization factors (all relative to reference)
    cum_factors = np.ones(n_seg)
    cum_unc = np.zeros(n_seg)
    norm_records = []

    # Propagate normalization from reference toward lower indices
    for i in range(ref_idx - 1, -1, -1):
        ol = overlaps[i]  # overlap between segment i and i+1
        wl, fl, ul, ml = unpacked[i]
        wr, fr, ur, mr = unpacked[i + 1]
        # Scale right segment by its cumulative factor before computing ratio
        fr_scaled = fr * cum_factors[i + 1]
        ur_scaled = ur * cum_factors[i + 1]
        # factor: multiply left to match right
        factor, factor_unc = compute_normalization_factor(
            wr, fr_scaled, ur_scaled, mr,  # "left" = already-normalized neighbor
            wl, fl, ul, ml,                 # "right" = segment to normalize
            ol, method=method
        )
        cum_factors[i] = factor
        cum_unc[i] = factor_unc
        norm_records.append(NormFactor(
            segment_idx=i, factor=factor, factor_uncertainty=factor_unc,
            method=method, overlap_idx=i
        ))

    # Propagate normalization from reference toward higher indices
    for i in range(ref_idx + 1, n_seg):
        ol = overlaps[i - 1]  # overlap between segment i-1 and i
        wl, fl, ul, ml = unpacked[i - 1]
        wr, fr, ur, mr = unpacked[i]
        # Scale left segment by its cumulative factor
        fl_scaled = fl * cum_factors[i - 1]
        ul_scaled = ul * cum_factors[i - 1]
        factor, factor_unc = compute_normalization_factor(
            wl, fl_scaled, ul_scaled, ml,
            wr, fr, ur, mr,
            ol, method=method
        )
        cum_factors[i] = factor
        cum_unc[i] = factor_unc
        norm_records.append(NormFactor(
            segment_idx=i, factor=factor, factor_uncertainty=factor_unc,
            method=method, overlap_idx=i - 1
        ))

    # Apply cumulative factors
    normalized = []
    for i, (w, f, u, m) in enumerate(unpacked):
        f_norm = f * cum_factors[i]
        # Propagate uncertainty: u_norm^2 = (f*sigma_c)^2 + (c*sigma_f)^2
        u_norm = np.sqrt((f * cum_unc[i])**2 + (cum_factors[i] * u)**2)
        normalized.append((w, f_norm, u_norm, m))

    return normalized, norm_records


# ---------------------------------------------------------------------------
# 4.3  Combine overlapping pixels (inverse-variance weighting)
# ---------------------------------------------------------------------------

def combine_overlap_region(
    wave_left, flux_left, unc_left, mask_left,
    wave_right, flux_right, unc_right, mask_right,
    overlap: OverlapInfo
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Combine two segments in their overlap region using inverse-variance weighting.

    In the overlap, interpolate both segments onto a common wavelength grid
    (the finer of the two), then combine with weights ~ 1/sigma^2.

    Outside the overlap, each segment contributes its own data.

    Parameters
    ----------
    wave_left, flux_left, unc_left, mask_left : arrays for left segment
    wave_right, flux_right, unc_right, mask_right : arrays for right segment
    overlap : OverlapInfo

    Returns
    -------
    wave_combined, flux_combined, unc_combined, mask_combined
    """
    if overlap.is_gap:
        # No overlap; just concatenate with gap handling done separately
        return (np.concatenate([wave_left, wave_right]),
                np.concatenate([flux_left, flux_right]),
                np.concatenate([unc_left, unc_right]),
                np.concatenate([mask_left, mask_right]))

    ol_start = overlap.wave_start
    ol_end = overlap.wave_end

    # --- Left-only region ---
    left_only = wave_left < ol_start
    # --- Right-only region ---
    right_only = wave_right > ol_end
    # --- Overlap regions in each segment ---
    left_in_ol = (wave_left >= ol_start) & (wave_left <= ol_end)
    right_in_ol = (wave_right >= ol_start) & (wave_right <= ol_end)

    # Build a common wavelength grid in the overlap from the finer segment
    spacing_left = np.median(np.diff(wave_left[left_in_ol])) if np.sum(left_in_ol) > 1 else 999
    spacing_right = np.median(np.diff(wave_right[right_in_ol])) if np.sum(right_in_ol) > 1 else 999
    use_spacing = min(abs(spacing_left), abs(spacing_right))

    n_ol = max(int(np.ceil((ol_end - ol_start) / use_spacing)), 2)
    wave_ol = np.linspace(ol_start, ol_end, n_ol)

    # Interpolate left segment onto overlap grid
    f_left_interp = np.interp(wave_ol, wave_left[left_in_ol & mask_left],
                              flux_left[left_in_ol & mask_left],
                              left=np.nan, right=np.nan)
    u_left_interp = np.interp(wave_ol, wave_left[left_in_ol & mask_left],
                              unc_left[left_in_ol & mask_left],
                              left=np.nan, right=np.nan)
    m_left_interp = np.isfinite(f_left_interp) & np.isfinite(u_left_interp)

    # Interpolate right segment onto overlap grid
    f_right_interp = np.interp(wave_ol, wave_right[right_in_ol & mask_right],
                               flux_right[right_in_ol & mask_right],
                               left=np.nan, right=np.nan)
    u_right_interp = np.interp(wave_ol, wave_right[right_in_ol & mask_right],
                               unc_right[right_in_ol & mask_right],
                               left=np.nan, right=np.nan)
    m_right_interp = np.isfinite(f_right_interp) & np.isfinite(u_right_interp)

    # Inverse-variance combination
    flux_ol = np.full_like(wave_ol, np.nan)
    unc_ol = np.full_like(wave_ol, np.nan)
    mask_ol = np.zeros(len(wave_ol), dtype=bool)

    for j in range(len(wave_ol)):
        have_left = m_left_interp[j] and u_left_interp[j] > 0
        have_right = m_right_interp[j] and u_right_interp[j] > 0

        if have_left and have_right:
            # Inverse-variance weighting
            var_l = u_left_interp[j]**2
            var_r = u_right_interp[j]**2
            w_l = 1.0 / var_l
            w_r = 1.0 / var_r
            w_total = w_l + w_r
            flux_ol[j] = (w_l * f_left_interp[j] + w_r * f_right_interp[j]) / w_total
            unc_ol[j] = np.sqrt(1.0 / w_total)
            mask_ol[j] = True
        elif have_left:
            flux_ol[j] = f_left_interp[j]
            unc_ol[j] = u_left_interp[j]
            mask_ol[j] = True
        elif have_right:
            flux_ol[j] = f_right_interp[j]
            unc_ol[j] = u_right_interp[j]
            mask_ol[j] = True
        # else: stays NaN/masked

    # Assemble: left-only + overlap + right-only
    wave_combined = np.concatenate([wave_left[left_only],
                                    wave_ol,
                                    wave_right[right_only]])
    flux_combined = np.concatenate([flux_left[left_only],
                                    flux_ol,
                                    flux_right[right_only]])
    unc_combined = np.concatenate([unc_left[left_only],
                                   unc_ol,
                                   unc_right[right_only]])
    mask_combined = np.concatenate([mask_left[left_only],
                                    mask_ol,
                                    mask_right[right_only]])

    return wave_combined, flux_combined, unc_combined, mask_combined


# ---------------------------------------------------------------------------
# 4.4  Handle gaps
# ---------------------------------------------------------------------------

def handle_gap(
    wave_left, flux_left, unc_left, mask_left,
    wave_right, flux_right, unc_right, mask_right,
    gap: OverlapInfo,
    max_gap_angstrom: float = 50.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Bridge a gap between two non-overlapping segments.

    If the gap is smaller than max_gap_angstrom, linearly interpolate across it.
    Otherwise, insert NaN/masked values.

    Parameters
    ----------
    wave_left, flux_left, unc_left, mask_left : arrays for left segment
    wave_right, flux_right, unc_right, mask_right : arrays for right segment
    gap : OverlapInfo with is_gap=True
    max_gap_angstrom : float
        Maximum gap width (in Angstroms) to interpolate across.

    Returns
    -------
    wave_combined, flux_combined, unc_combined, mask_combined, interpolated_mask
        interpolated_mask: True where pixels were interpolated (not real data).
    """
    gap_width = abs(gap.overlap_width)
    gap_start = min(wave_left.max(), wave_right.min())
    gap_end = max(wave_left.max(), wave_right.min())

    # Average pixel spacing from both segments
    avg_spacing = 0.5 * (np.median(np.abs(np.diff(wave_left))) +
                         np.median(np.abs(np.diff(wave_right))))
    n_gap = max(int(np.ceil(gap_width / avg_spacing)), 2)
    wave_gap = np.linspace(gap_start, gap_end, n_gap + 2)[1:-1]  # exclude endpoints

    interp_flag_left = np.zeros(len(wave_left), dtype=bool)
    interp_flag_right = np.zeros(len(wave_right), dtype=bool)

    if gap_width <= max_gap_angstrom and gap_width > 0:
        # Linear interpolation across small gap
        # Use last few points of left and first few points of right
        n_anchor = min(10, len(wave_left), len(wave_right))
        w_anchor = np.concatenate([wave_left[-n_anchor:], wave_right[:n_anchor]])
        f_anchor = np.concatenate([flux_left[-n_anchor:], flux_right[:n_anchor]])
        u_anchor = np.concatenate([unc_left[-n_anchor:], unc_right[:n_anchor]])

        flux_gap = np.interp(wave_gap, w_anchor, f_anchor)
        unc_gap = np.interp(wave_gap, w_anchor, u_anchor)
        # Inflate uncertainty for interpolated pixels (scale by 2x)
        unc_gap *= 2.0
        mask_gap = np.ones(len(wave_gap), dtype=bool)
        interp_flag_gap = np.ones(len(wave_gap), dtype=bool)
    else:
        # Too large to interpolate; fill with NaN and mask
        flux_gap = np.full(len(wave_gap), np.nan)
        unc_gap = np.full(len(wave_gap), np.nan)
        mask_gap = np.zeros(len(wave_gap), dtype=bool)
        interp_flag_gap = np.zeros(len(wave_gap), dtype=bool)

    wave_combined = np.concatenate([wave_left, wave_gap, wave_right])
    flux_combined = np.concatenate([flux_left, flux_gap, flux_right])
    unc_combined = np.concatenate([unc_left, unc_gap, unc_right])
    mask_combined = np.concatenate([mask_left, mask_gap, mask_right])
    interp_combined = np.concatenate([interp_flag_left, interp_flag_gap,
                                      interp_flag_right])

    return wave_combined, flux_combined, unc_combined, mask_combined, interp_combined


# ---------------------------------------------------------------------------
# 4.5  Resample to a uniform wavelength grid
# ---------------------------------------------------------------------------

def resample_to_uniform_grid(
    wavelength: np.ndarray,
    flux: np.ndarray,
    uncertainty: np.ndarray,
    mask: np.ndarray,
    grid_start: Optional[float] = None,
    grid_end: Optional[float] = None,
    grid_step: Optional[float] = None,
    method: str = 'spectres'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Resample the stitched spectrum onto a uniform wavelength grid.

    Parameters
    ----------
    wavelength, flux, uncertainty, mask : arrays
        The stitched spectrum (potentially non-uniform spacing).
    grid_start : float, optional
        Start wavelength. Default: min of valid wavelengths.
    grid_end : float, optional
        End wavelength. Default: max of valid wavelengths.
    grid_step : float, optional
        Pixel spacing in Angstroms. Default: median spacing of input, rounded up
        to match or slightly oversample the coarsest segment resolution.
    method : str
        'spectres' (flux-conserving) or 'interp' (linear interpolation fallback)

    Returns
    -------
    wave_new, flux_new, unc_new, mask_new
    """
    valid = mask & np.isfinite(flux) & np.isfinite(wavelength)
    if np.sum(valid) < 2:
        raise ValueError("Too few valid pixels for resampling.")

    if grid_start is None:
        grid_start = float(wavelength[valid].min())
    if grid_end is None:
        grid_end = float(wavelength[valid].max())
    if grid_step is None:
        # Use the coarsest spacing found in the data, slightly oversampled
        spacings = np.abs(np.diff(wavelength[valid]))
        grid_step = float(np.percentile(spacings, 90))  # near-coarsest

    wave_new = np.arange(grid_start, grid_end + grid_step * 0.5, grid_step)

    if method == 'spectres' and HAS_SPECTRES:
        # spectres does flux-conserving resampling
        # It requires sorted, unique wavelengths with the new grid strictly
        # within the old grid bounds
        w_valid = wavelength[valid]
        f_valid = flux[valid]
        u_valid = uncertainty[valid]

        # Ensure unique wavelengths (spectres requires strictly increasing)
        unique_mask = np.concatenate([[True], np.diff(w_valid) > 0])
        w_valid = w_valid[unique_mask]
        f_valid = f_valid[unique_mask]
        u_valid = u_valid[unique_mask]

        # Trim new grid to be within old grid bounds (spectres requirement)
        in_range = (wave_new >= w_valid.min()) & (wave_new <= w_valid.max())
        wave_trimmed = wave_new[in_range]

        if len(wave_trimmed) < 2:
            raise ValueError("New wavelength grid has no overlap with data.")

        try:
            flux_new_trimmed = spectres(wave_trimmed, w_valid, f_valid)
            unc_new_trimmed = spectres(wave_trimmed, w_valid, u_valid)
        except Exception as e:
            warnings.warn(f"spectres failed ({e}); falling back to interpolation.")
            return resample_to_uniform_grid(wavelength, flux, uncertainty, mask,
                                            grid_start, grid_end, grid_step,
                                            method='interp')

        # Embed trimmed result back into full grid
        flux_new = np.full(len(wave_new), np.nan)
        unc_new = np.full(len(wave_new), np.nan)
        mask_new = np.zeros(len(wave_new), dtype=bool)
        flux_new[in_range] = flux_new_trimmed
        unc_new[in_range] = unc_new_trimmed
        mask_new[in_range] = np.isfinite(flux_new_trimmed)

    elif method == 'interp':
        w_valid = wavelength[valid]
        f_valid = flux[valid]
        u_valid = uncertainty[valid]
        flux_new = np.interp(wave_new, w_valid, f_valid, left=np.nan, right=np.nan)
        unc_new = np.interp(wave_new, w_valid, u_valid, left=np.nan, right=np.nan)
        mask_new = np.isfinite(flux_new)
    else:
        if not HAS_SPECTRES:
            warnings.warn("spectres not installed; using linear interpolation.")
        return resample_to_uniform_grid(wavelength, flux, uncertainty, mask,
                                        grid_start, grid_end, grid_step,
                                        method='interp')

    return wave_new, flux_new, unc_new, mask_new


# ---------------------------------------------------------------------------
# Main stitching function
# ---------------------------------------------------------------------------

def stitch_segments(
    segments: list,
    mode: str = 'pre_calibration',
    normalize: bool = True,
    norm_method: str = 'median_ratio',
    reference: Union[int, str] = 'auto',
    resample: bool = True,
    grid_start: Optional[float] = None,
    grid_end: Optional[float] = None,
    grid_step: Optional[float] = None,
    resample_method: str = 'spectres',
    max_gap_angstrom: float = 50.0
) -> StitchResult:
    """
    Stitch a list of spectral segments into a single continuous spectrum.

    This is the main entry point for segment stitching. It orchestrates
    overlap detection, cross-normalization, overlap combination, gap handling,
    and (optionally) resampling.

    Parameters
    ----------
    segments : list
        Ordered list of spectral segments. Each can be a Spectrum1D, dict,
        or tuple of (wavelength, flux[, uncertainty[, mask]]).
    mode : str
        'pre_calibration'  - stitching raw/normalized counts before deriving
                             a sensitivity function.
        'post_calibration' - stitching already flux-calibrated segments.
        The main difference: pre-calibration applies cross-normalization by
        default; post-calibration may skip it if segments are already on a
        common flux scale.
    normalize : bool
        Whether to apply cross-normalization. Default True for pre_calibration,
        can be set False for post_calibration.
    norm_method : str
        'median_ratio' or 'polynomial'
    reference : int or 'auto'
        Reference segment index for normalization.
    resample : bool
        Whether to resample to a uniform grid at the end.
    grid_start, grid_end, grid_step : float, optional
        Uniform grid parameters. Defaults chosen from data.
    resample_method : str
        'spectres' or 'interp'
    max_gap_angstrom : float
        Maximum gap width to interpolate across.

    Returns
    -------
    StitchResult
        Contains stitched wavelength, flux, uncertainty, mask, and diagnostics.
    """
    if len(segments) == 0:
        raise ValueError("No segments provided.")
    if len(segments) == 1:
        w, f, u, m = _ensure_sorted(*_unpack_segment(segments[0]))
        return StitchResult(
            wavelength=w, flux=f, uncertainty=u, mask=m,
            interpolated=np.zeros(len(w), dtype=bool),
            reference_segment=0
        )

    # 1. Unpack and sort segments by ascending starting wavelength
    unpacked = []
    for seg in segments:
        w, f, u, m = _ensure_sorted(*_unpack_segment(seg))
        unpacked.append((w, f, u, m))
    # Sort by min wavelength
    order = np.argsort([w.min() for w, f, u, m in unpacked])
    unpacked = [unpacked[i] for i in order]

    # 2. Find overlaps
    # Build temporary list for overlap detection
    overlaps = find_overlaps(unpacked)

    # 3. Cross-normalize if requested
    norm_factors_list = []
    if normalize and mode == 'pre_calibration':
        unpacked, norm_factors_list = cross_normalize(
            unpacked, overlaps, reference=reference, method=norm_method
        )
        ref_idx = (int(np.argmax([estimate_segment_snr(f, u, m)
                                  for w, f, u, m in unpacked]))
                   if reference == 'auto' else int(reference))
    elif normalize and mode == 'post_calibration':
        # For post-calibration, cross-normalize only if large offsets detected
        # Check if median fluxes differ by >10% between adjacent segments
        needs_norm = False
        for ol in overlaps:
            if ol.is_gap:
                continue
            wl, fl, ul, ml = unpacked[ol.idx_left]
            wr, fr, ur, mr = unpacked[ol.idx_right]
            ol_l = (wl >= ol.wave_start) & (wl <= ol.wave_end) & ml
            ol_r = (wr >= ol.wave_start) & (wr <= ol.wave_end) & mr
            if np.sum(ol_l) > 5 and np.sum(ol_r) > 5:
                ratio = np.median(fl[ol_l]) / np.median(fr[ol_r])
                if abs(ratio - 1.0) > 0.10:
                    needs_norm = True
                    break
        if needs_norm:
            unpacked, norm_factors_list = cross_normalize(
                unpacked, overlaps, reference=reference, method=norm_method
            )
        ref_idx = 0
    else:
        ref_idx = 0

    # 4. Sequentially combine adjacent segment pairs
    # Start with the leftmost segment and progressively merge rightward
    w_running, f_running, u_running, m_running = unpacked[0]
    interp_running = np.zeros(len(w_running), dtype=bool)

    for i in range(len(overlaps)):
        ol = overlaps[i]
        w_next, f_next, u_next, m_next = unpacked[i + 1]

        if ol.is_gap:
            w_running, f_running, u_running, m_running, interp_new = handle_gap(
                w_running, f_running, u_running, m_running,
                w_next, f_next, u_next, m_next,
                ol, max_gap_angstrom=max_gap_angstrom
            )
            interp_running = np.concatenate([
                interp_running,
                interp_new[len(interp_running):]
            ])
            # Pad interpolated array to match
            if len(interp_running) < len(w_running):
                interp_running = np.concatenate([
                    interp_running,
                    np.zeros(len(w_running) - len(interp_running), dtype=bool)
                ])
        else:
            # Recompute overlap between running result and next segment
            ol_start = max(w_running.min(), w_next.min())
            ol_end = min(w_running.max(), w_next.max())
            if ol_end > ol_start:
                running_ol = OverlapInfo(
                    idx_left=i, idx_right=i + 1,
                    wave_start=ol_start, wave_end=ol_end,
                    overlap_width=ol_end - ol_start, is_gap=False
                )
                w_comb, f_comb, u_comb, m_comb = combine_overlap_region(
                    w_running, f_running, u_running, m_running,
                    w_next, f_next, u_next, m_next,
                    running_ol
                )
            else:
                # Edge case: after combination, no actual overlap
                w_comb = np.concatenate([w_running, w_next])
                f_comb = np.concatenate([f_running, f_next])
                u_comb = np.concatenate([u_running, u_next])
                m_comb = np.concatenate([m_running, m_next])

            # Update interpolated flag array
            interp_comb = np.zeros(len(w_comb), dtype=bool)
            # Transfer old interpolated flags for the left-only region
            left_only_n = np.sum(w_comb < ol_start) if ol_end > ol_start else len(w_running)
            if left_only_n <= len(interp_running):
                interp_comb[:left_only_n] = interp_running[:left_only_n]

            w_running = w_comb
            f_running = f_comb
            u_running = u_comb
            m_running = m_comb
            interp_running = interp_comb

    # Ensure monotonic wavelength order in final result
    sort_idx = np.argsort(w_running)
    w_running = w_running[sort_idx]
    f_running = f_running[sort_idx]
    u_running = u_running[sort_idx]
    m_running = m_running[sort_idx]
    interp_running = interp_running[sort_idx]

    # Remove duplicate wavelengths (can occur at overlap boundaries)
    unique_mask = np.concatenate([[True], np.diff(w_running) > 1e-6])
    w_running = w_running[unique_mask]
    f_running = f_running[unique_mask]
    u_running = u_running[unique_mask]
    m_running = m_running[unique_mask]
    interp_running = interp_running[unique_mask]

    # 5. Resample to uniform grid
    if resample:
        w_final, f_final, u_final, m_final = resample_to_uniform_grid(
            w_running, f_running, u_running, m_running,
            grid_start=grid_start, grid_end=grid_end, grid_step=grid_step,
            method=resample_method
        )
        # Interpolated flag: propagate by nearest-neighbor mapping
        if len(interp_running) > 0 and np.any(interp_running):
            interp_func = interp1d(w_running, interp_running.astype(float),
                                   kind='nearest', bounds_error=False,
                                   fill_value=0)
            interp_final = interp_func(w_final) > 0.5
        else:
            interp_final = np.zeros(len(w_final), dtype=bool)
    else:
        w_final = w_running
        f_final = f_running
        u_final = u_running
        m_final = m_running
        interp_final = interp_running

    return StitchResult(
        wavelength=w_final, flux=f_final, uncertainty=u_final,
        mask=m_final, interpolated=interp_final,
        overlaps=overlaps, norm_factors=norm_factors_list,
        reference_segment=ref_idx if normalize else -1
    )


# ---------------------------------------------------------------------------
# 4.6  Diagnostic outputs / plotting
# ---------------------------------------------------------------------------

def plot_stitched_spectrum(
    result: StitchResult,
    segments: Optional[list] = None,
    title: str = 'Stitched Spectrum',
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (14, 8)
):
    """
    Plot the stitched spectrum with segment boundaries and overlap regions.

    Parameters
    ----------
    result : StitchResult
    segments : list of original segments (for overlay), optional
    title : str
    save_path : str, optional
        If provided, save figure to this path.
    figsize : tuple
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True,
                             gridspec_kw={'height_ratios': [3, 1, 1]})

    # --- Panel 1: Stitched spectrum + individual segments ---
    ax1 = axes[0]
    valid = result.mask & np.isfinite(result.flux)
    ax1.plot(result.wavelength[valid], result.flux[valid],
             color='black', lw=0.8, label='Stitched', zorder=10)

    # Mark interpolated regions
    if np.any(result.interpolated):
        interp_valid = result.interpolated & result.mask
        ax1.scatter(result.wavelength[interp_valid], result.flux[interp_valid],
                    color='orange', s=5, label='Interpolated', zorder=11)

    # Overlay individual segments if provided
    if segments is not None:
        cmap = plt.cm.tab10
        for i, seg in enumerate(segments):
            w, f, u, m = _ensure_sorted(*_unpack_segment(seg))
            color = cmap(i % 10)
            ax1.plot(w[m], f[m], color=color, alpha=0.4, lw=0.5,
                     label=f'Seg {i} ({w[m].min():.0f}-{w[m].max():.0f} A)')

    # Shade overlap regions
    for ol in result.overlaps:
        if not ol.is_gap:
            ax1.axvspan(ol.wave_start, ol.wave_end, alpha=0.1, color='blue')
        else:
            ax1.axvspan(ol.wave_start, ol.wave_end, alpha=0.2, color='red')

    ax1.set_ylabel('Flux (counts)')
    ax1.set_title(title)
    ax1.legend(fontsize=7, ncol=3, loc='upper right')

    # --- Panel 2: Uncertainty ---
    ax2 = axes[1]
    ax2.plot(result.wavelength[valid], result.uncertainty[valid],
             color='gray', lw=0.5)
    ax2.set_ylabel('Uncertainty')

    # --- Panel 3: SNR ---
    ax3 = axes[2]
    snr = np.zeros_like(result.flux)
    good = valid & (result.uncertainty > 0)
    snr[good] = np.abs(result.flux[good]) / result.uncertainty[good]
    ax3.plot(result.wavelength[good], snr[good], color='green', lw=0.5)
    ax3.set_ylabel('SNR')
    ax3.set_xlabel('Wavelength (A)')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved stitched spectrum plot to {save_path}")
    plt.close(fig)
    return fig


def plot_normalization_factors(
    segments: list,
    norm_factors: List[NormFactor],
    overlaps: List[OverlapInfo],
    title: str = 'Cross-Normalization Factors',
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (12, 5)
):
    """
    Plot the cross-normalization factors as a function of segment/wavelength.

    Parameters
    ----------
    segments : list of original segments
    norm_factors : list of NormFactor
    overlaps : list of OverlapInfo
    title : str
    save_path : str, optional
    figsize : tuple
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Left panel: factor vs segment index
    indices = [nf.segment_idx for nf in norm_factors]
    factors = [nf.factor for nf in norm_factors]
    uncs = [nf.factor_uncertainty for nf in norm_factors]
    ax1.errorbar(indices, factors, yerr=uncs, fmt='o-', capsize=3)
    ax1.axhline(1.0, color='gray', ls='--', lw=0.5)
    ax1.set_xlabel('Segment Index')
    ax1.set_ylabel('Normalization Factor')
    ax1.set_title('Factor per Segment')

    # Right panel: factor vs overlap midpoint wavelength
    mid_waves = []
    for nf in norm_factors:
        ol_idx = nf.overlap_idx
        if ol_idx < len(overlaps):
            mid = 0.5 * (overlaps[ol_idx].wave_start + overlaps[ol_idx].wave_end)
        else:
            mid = 0.0
        mid_waves.append(mid)
    ax2.errorbar(mid_waves, factors, yerr=uncs, fmt='s-', capsize=3, color='tab:orange')
    ax2.axhline(1.0, color='gray', ls='--', lw=0.5)
    ax2.set_xlabel('Overlap Midpoint Wavelength (A)')
    ax2.set_ylabel('Normalization Factor')
    ax2.set_title('Factor vs Wavelength')

    fig.suptitle(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved normalization factors plot to {save_path}")
    plt.close(fig)
    return fig


def plot_overlap_quality(
    segments: list,
    overlaps: List[OverlapInfo],
    title: str = 'Overlap Region Quality',
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (14, 10)
):
    """
    For each overlap region, plot the two contributing segments overlaid
    to visualize their agreement.

    Parameters
    ----------
    segments : list of original segments
    overlaps : list of OverlapInfo
    title : str
    save_path : str, optional
    figsize : tuple
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n_overlaps = sum(1 for ol in overlaps if not ol.is_gap)
    if n_overlaps == 0:
        print("No overlapping regions to plot.")
        return None

    fig, axes = plt.subplots(n_overlaps, 1, figsize=figsize, squeeze=False)
    panel_idx = 0

    for ol in overlaps:
        if ol.is_gap:
            continue

        ax = axes[panel_idx, 0]
        w_left, f_left, _, m_left = _ensure_sorted(*_unpack_segment(segments[ol.idx_left]))
        w_right, f_right, _, m_right = _ensure_sorted(*_unpack_segment(segments[ol.idx_right]))

        # Plot in overlap range with some margin
        margin = 0.1 * ol.overlap_width
        xlim = (ol.wave_start - margin, ol.wave_end + margin)

        in_range_l = (w_left >= xlim[0]) & (w_left <= xlim[1]) & m_left
        in_range_r = (w_right >= xlim[0]) & (w_right <= xlim[1]) & m_right

        ax.plot(w_left[in_range_l], f_left[in_range_l],
                color='blue', alpha=0.7, lw=0.8,
                label=f'Seg {ol.idx_left}')
        ax.plot(w_right[in_range_r], f_right[in_range_r],
                color='red', alpha=0.7, lw=0.8,
                label=f'Seg {ol.idx_right}')

        ax.axvspan(ol.wave_start, ol.wave_end, alpha=0.08, color='green')
        ax.set_ylabel('Flux')
        ax.legend(fontsize=8)
        ax.set_title(f'Overlap {ol.idx_left}-{ol.idx_right}: '
                     f'{ol.wave_start:.0f}-{ol.wave_end:.0f} A '
                     f'({ol.overlap_width:.0f} A)')
        panel_idx += 1

    axes[-1, 0].set_xlabel('Wavelength (A)')
    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved overlap quality plot to {save_path}")
    plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# Convenience: load JJMO data directly (bridge until Step 1 io.py exists)
# ---------------------------------------------------------------------------

def load_jjmo_sirius(data_dir: str) -> list:
    """
    Load Sirius segments from the JJMO data directory.

    Parameters
    ----------
    data_dir : str
        Path to the Sirius data directory containing .fit and .txt files.

    Returns
    -------
    list of (wavelength, flux) tuples, sorted by ascending start wavelength.
    """
    from astropy.io import fits as afits
    import os

    segment_centers = [3900, 4400, 4900, 5400, 5900, 6400, 6900, 7400]
    segments = []

    for center in segment_centers:
        fit_path = os.path.join(data_dir, f'{center}.fit')
        txt_path = os.path.join(data_dir, f'{center}.txt')

        if not os.path.exists(fit_path) or not os.path.exists(txt_path):
            warnings.warn(f"Missing data for segment {center}; skipping.")
            continue

        wave = np.genfromtxt(txt_path, delimiter='\t', usecols=(1),
                             invalid_raise=False)
        wave = wave[~np.isnan(wave)]

        with afits.open(fit_path) as hdul:
            data_2d = hdul[0].data
            flux = np.flip(np.nansum(data_2d, axis=0))

        # Ensure same length (trim to shorter if needed)
        n = min(len(wave), len(flux))
        wave = wave[:n]
        flux = flux[:n]

        # Sort ascending
        if wave[0] > wave[-1]:
            wave = wave[::-1]
            flux = flux[::-1]

        segments.append((wave, flux))

    # Sort by starting wavelength
    segments.sort(key=lambda s: s[0].min())
    return segments


def load_jjmo_betelgeuse(data_dir: str) -> list:
    """
    Load Betelgeuse segments from the JJMO data directory.

    Parameters
    ----------
    data_dir : str
        Path to the Betelgeuse data directory containing .csv files.

    Returns
    -------
    list of (wavelength, flux) tuples, sorted by ascending start wavelength.
    """
    import os

    segment_centers = [4400, 4900, 5400, 5900, 6400, 6900, 7400]
    segments = []

    for center in segment_centers:
        csv_path = os.path.join(data_dir, f'Betelgeuse_{center}.csv')
        if not os.path.exists(csv_path):
            warnings.warn(f"Missing data for Betelgeuse segment {center}; skipping.")
            continue

        data = np.genfromtxt(csv_path, delimiter=',')
        wave = data[:, 1]
        flux = data[:, 2]

        valid = ~(np.isnan(wave) | np.isnan(flux))
        wave = wave[valid]
        flux = flux[valid]

        # Sort ascending
        if wave[0] > wave[-1]:
            wave = wave[::-1]
            flux = flux[::-1]

        segments.append((wave, flux))

    segments.sort(key=lambda s: s[0].min())
    return segments
