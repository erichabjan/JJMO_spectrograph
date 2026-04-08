"""
wavelength.py — Wavelength Calibration & Velocity Correction
=============================================================
Step 2 of the JJMO Spectral Flux Calibration Pipeline.

Provides robust absorption-line detection, line matching against known
rest wavelengths, instrumental offset measurement via telluric lines,
radial velocity determination, cross-correlation verification, and
wavelength correction for segmented spectrograph data.

Works with plain numpy arrays (wavelength, flux) and is forward-compatible
with specutils Spectrum1D containers when Step 1 (io.py) is built.

Authors: JJMO Pipeline
"""

import numpy as np
from scipy.signal import savgol_filter, correlate
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import warnings

# ============================================================================
# Known line lists (air wavelengths in Angstroms)
# ============================================================================

BALMER_LINES = {
    'H-alpha': 6562.8,
    'H-beta':  4861.3,
    'H-gamma': 4340.5,
    'H-delta': 4101.7,
    'H-eps':   3970.1,
    'H-zeta':  3889.1,
    'H-eta':   3835.4,
}

METAL_LINES = {
    'Ca II K':  3933.7,
    'Ca II H':  3968.5,
    'Mg II':    4481.2,
    'Fe I 4384': 4383.6,
    'Fe II 4924': 4923.9,
    'Fe II 5018': 5018.4,
    'Fe II 5169': 5169.0,
    'Si III':   4552.6,
    'Sr II':    4077.7,
}

TELLURIC_LINES = {
    'O2 6277':        6277.0,   # O2 absorption near 6280 A
    'O2 B-band 6867': 6867.0,   # Atmospheric O2 B band head
    'H2O 7186':       7186.0,   # Water vapor band
    'O2 A-band 7620': 7620.0,   # O2 A band — effective center at JJMO resolution
    # Note: at ~1 A/pixel resolution, the O2 A-band (7594-7630) appears
    # as a single blended trough; 7620 A is the approximate centroid.
}

# Aggregate line catalogues for convenience
ALL_STELLAR_LINES = {**BALMER_LINES, **METAL_LINES}
ALL_KNOWN_LINES = {**ALL_STELLAR_LINES, **TELLURIC_LINES}

# Speed of light in km/s
C_KMS = 299792.458


# ============================================================================
# 2.1 — Improved absorption-line finder
# ============================================================================

def estimate_snr(flux):
    """Estimate signal-to-noise ratio from the flux array.

    Uses the median flux divided by the standard deviation of the
    first-difference (which approximates pixel-to-pixel noise without
    being biased by broad spectral features).
    """
    diff_noise = np.std(np.diff(flux)) / np.sqrt(2)
    if diff_noise == 0:
        return np.inf
    return np.median(flux) / diff_noise


def _parabolic_centroid(wavelength, flux, idx, half_width=2):
    """Refine a minimum location to sub-pixel precision via parabolic fit.

    Fits a parabola to 2*half_width+1 points centered on idx, returns
    the wavelength of the parabola vertex and an estimated uncertainty.

    Parameters
    ----------
    wavelength : array
    flux : array
    idx : int
        Index of the approximate minimum.
    half_width : int
        Number of points on each side of idx to include in the fit.

    Returns
    -------
    centroid : float
        Sub-pixel wavelength of the refined minimum.
    uncertainty : float
        Estimated 1-sigma uncertainty on the centroid (Angstroms).
    """
    lo = max(0, idx - half_width)
    hi = min(len(wavelength) - 1, idx + half_width)
    w = wavelength[lo:hi + 1]
    f = flux[lo:hi + 1]

    if len(w) < 3:
        return wavelength[idx], np.nan

    # Fit parabola: f = a*(w - w0)^2 + b*(w - w0) + c
    # Shift to local coords for numerical stability
    w0 = wavelength[idx]
    dw = w - w0
    try:
        coeffs = np.polyfit(dw, f, 2)
    except (np.linalg.LinAlgError, ValueError):
        return wavelength[idx], np.nan

    a, b, c = coeffs
    if a <= 0:
        # Not a minimum (concave down) — fall back to grid value
        return wavelength[idx], np.nan

    # Vertex of parabola: dw_min = -b / (2a)
    dw_min = -b / (2 * a)
    centroid = w0 + dw_min

    # Clamp to the fitting window to avoid runaway extrapolation
    if centroid < w[0] or centroid > w[-1]:
        centroid = wavelength[idx]
        dw_min = 0.0

    # Uncertainty estimate: from the curvature of the parabola and local noise
    # sigma_centroid ~ sigma_flux / sqrt(2 * a * N) where N is the number of points
    residuals = np.polyval(coeffs, dw) - f
    rms_resid = np.sqrt(np.mean(residuals**2))
    if a > 0:
        sigma = rms_resid / (2 * a * np.sqrt(len(w)))
    else:
        sigma = np.nan

    return centroid, sigma


def find_absorption_lines(wavelength, flux, smoothing_window=None,
                          smoothing_polyorder=None, d_range=None,
                          end_buffer=None, min_depth_sigma=3.0,
                          min_depth_frac=0.01,
                          parabolic_half_width=2, dedup_tolerance=None):
    """Find absorption-line minima with sub-pixel precision.

    Improved version of the original JJMO derivative-based finder.
    Parameters scale automatically with estimated SNR and spectral
    resolution when not explicitly provided.

    Parameters
    ----------
    wavelength : array-like
        Wavelength array in Angstroms (ascending order).
    flux : array-like
        Flux array (counts or counts/s).
    smoothing_window : int, optional
        Savitzky-Golay window length (pixels). If None, auto-scaled
        from SNR: higher noise → wider smoothing.
    smoothing_polyorder : int, optional
        Savitzky-Golay polynomial order. Default: 2.
    d_range : int, optional
        Number of consecutive derivative points that must be negative
        (before minimum) or positive (after). If None, auto-scaled.
    end_buffer : int, optional
        Pixels to skip at each end of the spectrum. If None, auto-scaled.
    min_depth_sigma : float
        Minimum line depth in units of local noise sigma to accept.
        Set to 0 to disable depth filtering.
    min_depth_frac : float
        Minimum line depth as a fraction of the local continuum.
        Lines shallower than this fraction of the continuum are rejected.
        Default 0.01 (1%). Set to 0 to disable.
    parabolic_half_width : int
        Half-width for parabolic centroid refinement (points on each side).
    dedup_tolerance : float, optional
        Minimum separation (Angstroms) between distinct lines. Lines
        closer than this are merged, keeping the deeper one. If None,
        set to 3× the pixel scale.

    Returns
    -------
    centroids : ndarray
        Sub-pixel wavelengths of detected absorption-line minima.
    uncertainties : ndarray
        Estimated 1-sigma uncertainties on each centroid (Angstroms).
    indices : ndarray of int
        Nearest-pixel indices into the wavelength array.
    depths : ndarray
        Depth of each line relative to the local continuum (positive = deeper).
    """
    wavelength = np.asarray(wavelength, dtype=float)
    flux = np.asarray(flux, dtype=float)

    # Pixel scale (Angstrom/pixel)
    pixel_scale = np.median(np.abs(np.diff(wavelength)))
    snr = estimate_snr(flux)

    # Auto-scale parameters based on SNR.
    # Higher SNR → smaller smoothing window (data is cleaner), but d_range
    # stays moderately large to avoid flagging pixel-scale noise as lines.
    if smoothing_window is None:
        smoothing_window = int(np.clip(60 / max(snr, 1), 5, 31))
        if smoothing_window % 2 == 0:
            smoothing_window += 1
    if smoothing_polyorder is None:
        smoothing_polyorder = min(2, smoothing_window - 1)
    if d_range is None:
        # At least 4 consecutive derivative points of the same sign
        # (original JJMO code used 5). Scales up for noisier data.
        d_range = int(np.clip(60 / max(snr, 1), 4, 12))
    if end_buffer is None:
        end_buffer = max(10, smoothing_window + d_range)
    if dedup_tolerance is None:
        dedup_tolerance = 3.0 * pixel_scale

    # Smooth the flux
    flux_smooth = savgol_filter(flux, window_length=smoothing_window,
                                polyorder=smoothing_polyorder)

    # Compute derivatives
    dflux = np.gradient(flux_smooth, wavelength)
    d2flux = np.gradient(dflux, wavelength)

    # Estimate local noise for depth filtering
    noise = np.std(np.diff(flux)) / np.sqrt(2)

    # Find zero-crossings: derivative goes negative→positive with positive curvature
    raw_indices = []
    search_lo = d_range + end_buffer
    search_hi = len(dflux) - d_range - end_buffer

    for i in range(search_lo, search_hi):
        if (np.all(dflux[i - d_range:i] < 0) and
                np.all(dflux[i + 1:i + d_range + 1] > 0) and
                d2flux[i] > 0):
            raw_indices.append(i)

    if len(raw_indices) == 0:
        return (np.array([]), np.array([]), np.array([], dtype=int),
                np.array([]))

    # Refine each minimum with parabolic interpolation
    centroids = []
    uncertainties = []
    indices = []
    depths = []

    for idx in raw_indices:
        centroid, sigma = _parabolic_centroid(
            wavelength, flux_smooth, idx, half_width=parabolic_half_width)

        # Estimate line depth: local continuum minus minimum
        cont_lo = max(0, idx - 20)
        cont_hi = min(len(flux) - 1, idx + 20)
        # Local continuum as the max of the smoothed flux in the neighborhood
        local_cont = np.max(flux_smooth[cont_lo:cont_hi + 1])
        depth = local_cont - flux_smooth[idx]

        # Depth filters: absolute (sigma-based) and relative (fraction of continuum)
        if min_depth_sigma > 0 and noise > 0 and depth < min_depth_sigma * noise:
            continue
        if min_depth_frac > 0 and local_cont > 0 and depth / local_cont < min_depth_frac:
            continue

        centroids.append(centroid)
        uncertainties.append(sigma)
        indices.append(idx)
        depths.append(depth)

    if len(centroids) == 0:
        return (np.array([]), np.array([]), np.array([], dtype=int),
                np.array([]))

    centroids = np.array(centroids)
    uncertainties = np.array(uncertainties)
    indices = np.array(indices, dtype=int)
    depths = np.array(depths)

    # Deduplicate: merge lines closer than dedup_tolerance, keeping deeper one
    if len(centroids) > 1:
        keep = np.ones(len(centroids), dtype=bool)
        sort_idx = np.argsort(centroids)
        centroids = centroids[sort_idx]
        uncertainties = uncertainties[sort_idx]
        indices = indices[sort_idx]
        depths = depths[sort_idx]

        i = 0
        while i < len(centroids) - 1:
            if centroids[i + 1] - centroids[i] < dedup_tolerance:
                # Keep the deeper line
                if depths[i] >= depths[i + 1]:
                    keep[i + 1] = False
                else:
                    keep[i] = False
            i += 1

        centroids = centroids[keep]
        uncertainties = uncertainties[keep]
        indices = indices[keep]
        depths = depths[keep]

    return centroids, uncertainties, indices, depths


# ============================================================================
# 2.2 — Line-matching engine
# ============================================================================

def _lines_in_range(line_dict, wmin, wmax, margin=0.0):
    """Return subset of a line dictionary whose wavelengths fall in [wmin-margin, wmax+margin]."""
    return {name: wl for name, wl in line_dict.items()
            if wmin - margin <= wl <= wmax + margin}


def match_lines(observed_centroids, rest_line_dict, tolerance=50.0,
                observed_uncertainties=None):
    """Match observed absorption centroids to known rest wavelengths.

    For each observed centroid, finds the closest rest wavelength within
    the tolerance window. Then tests multiple velocity hypotheses to
    maximize the number of consistent matches.

    Parameters
    ----------
    observed_centroids : array-like
        Detected line wavelengths (Angstroms).
    rest_line_dict : dict
        {name: rest_wavelength} of known lines to match against.
    tolerance : float
        Maximum allowed offset in Angstroms for a match candidate.
    observed_uncertainties : array-like, optional
        Uncertainties on the observed centroids.

    Returns
    -------
    matches : list of dict
        Each dict has keys:
            'obs_wave': observed wavelength,
            'rest_wave': rest wavelength,
            'line_name': name from the catalogue,
            'offset': obs - rest (Angstroms),
            'velocity': offset/rest * c (km/s),
            'quality': inverse of |offset| normalized by tolerance (0-1, higher=better)
        Sorted by rest wavelength.
    """
    observed = np.asarray(observed_centroids, dtype=float)
    if len(observed) == 0 or len(rest_line_dict) == 0:
        return []

    rest_names = list(rest_line_dict.keys())
    rest_waves = np.array([rest_line_dict[n] for n in rest_names])

    # Build all candidate pairings within tolerance
    candidates = []
    for i, obs in enumerate(observed):
        for j, (name, rest) in enumerate(zip(rest_names, rest_waves)):
            offset = obs - rest
            if abs(offset) <= tolerance:
                candidates.append({
                    'obs_idx': i,
                    'rest_idx': j,
                    'obs_wave': obs,
                    'rest_wave': rest,
                    'line_name': name,
                    'offset': offset,
                    'velocity': (offset / rest) * C_KMS,
                })

    if not candidates:
        return []

    # Extract unique velocity hypotheses from the candidate offsets.
    # Group candidates that imply similar velocities (within 50 km/s).
    cand_velocities = np.array([c['velocity'] for c in candidates])

    # Cluster velocities by sorting and grouping
    sort_idx = np.argsort(cand_velocities)
    velocity_groups = []
    current_group = [sort_idx[0]]
    for k in range(1, len(sort_idx)):
        if cand_velocities[sort_idx[k]] - cand_velocities[sort_idx[k - 1]] < 50.0:
            current_group.append(sort_idx[k])
        else:
            velocity_groups.append(current_group)
            current_group = [sort_idx[k]]
    velocity_groups.append(current_group)

    # For each velocity cluster, compute a representative velocity and
    # find the best set of one-to-one matches at that velocity.
    best_matches = []
    best_score = -1

    for group in velocity_groups:
        group_cands = [candidates[k] for k in group]
        v_median = np.median([c['velocity'] for c in group_cands])

        # At this velocity, shift all rest wavelengths and find best 1:1 matches
        shifted_rest = rest_waves * (1 + v_median / C_KMS)
        matches_at_v = []
        used_obs = set()
        used_rest = set()

        # Greedily match: smallest offset first
        pairs = []
        for i, obs in enumerate(observed):
            for j, sr in enumerate(shifted_rest):
                d = abs(obs - sr)
                if d <= tolerance:
                    pairs.append((d, i, j))
        pairs.sort()

        for d, i, j in pairs:
            if i not in used_obs and j not in used_rest:
                abs_offset = abs(observed[i] - rest_waves[j])
                matches_at_v.append({
                    'obs_wave': observed[i],
                    'rest_wave': rest_waves[j],
                    'line_name': rest_names[j],
                    'offset': observed[i] - rest_waves[j],
                    'velocity': ((observed[i] - rest_waves[j]) / rest_waves[j]) * C_KMS,
                    # Quality based on absolute offset from unshifted rest,
                    # NOT from shifted rest (which is always small by construction)
                    'quality': 1.0 - abs_offset / tolerance,
                })
                used_obs.add(i)
                used_rest.add(j)

        # Score: count of matches plus mean quality (0-1). The quality
        # term breaks ties when the number of matches is equal, preferring
        # the velocity hypothesis where lines are closer to their rest
        # wavelengths. This is critical for segments with only 1 stellar line.
        if len(matches_at_v) == 0:
            continue
        mean_quality = np.mean([m['quality'] for m in matches_at_v])
        residuals = np.array([abs(m['offset'] - np.median([mm['offset'] for mm in matches_at_v]))
                              for m in matches_at_v])
        score = len(matches_at_v) + mean_quality - np.mean(residuals) / tolerance

        if score > best_score:
            best_score = score
            best_matches = matches_at_v

    # Sort by rest wavelength
    best_matches.sort(key=lambda m: m['rest_wave'])
    return best_matches


# ============================================================================
# 2.3 — Instrumental offset & radial velocity separation
# ============================================================================

def measure_instrumental_offset(observed_centroids, wavelength_range,
                                observed_uncertainties=None,
                                telluric_dict=None, tolerance=50.0):
    """Measure the instrumental wavelength zero-point offset from telluric lines.

    Telluric absorption features are at rest in the observatory frame,
    so any observed shift is purely instrumental.

    Parameters
    ----------
    observed_centroids : array-like
        Detected line wavelengths for one segment.
    wavelength_range : tuple (wmin, wmax)
        Wavelength range of this segment.
    observed_uncertainties : array-like, optional
    telluric_dict : dict, optional
        Override for the telluric line catalogue.
    tolerance : float
        Matching tolerance in Angstroms.

    Returns
    -------
    offset : float
        Instrumental offset in Angstroms (observed - true). NaN if no
        telluric lines are matched.
    offset_err : float
        Uncertainty on the offset. NaN if not determined.
    telluric_matches : list of dict
        The matched telluric line pairs.
    """
    if telluric_dict is None:
        telluric_dict = TELLURIC_LINES

    wmin, wmax = wavelength_range
    local_telluric = _lines_in_range(telluric_dict, wmin, wmax, margin=tolerance)

    if not local_telluric:
        return np.nan, np.nan, []

    matches = match_lines(observed_centroids, local_telluric,
                          tolerance=tolerance,
                          observed_uncertainties=observed_uncertainties)

    if not matches:
        return np.nan, np.nan, []

    offsets = np.array([m['offset'] for m in matches])

    # Weighted mean if uncertainties available, else simple median
    offset = np.median(offsets)
    if len(offsets) > 1:
        offset_err = np.std(offsets) / np.sqrt(len(offsets))
    else:
        offset_err = np.nan

    return offset, offset_err, matches


def measure_radial_velocity(observed_centroids, wavelength_range,
                            instrumental_offset=0.0,
                            observed_uncertainties=None,
                            stellar_dict=None, tolerance=50.0):
    """Measure radial velocity from stellar absorption lines.

    After removing the instrumental offset, the remaining shift of
    stellar lines gives the radial velocity.

    Parameters
    ----------
    observed_centroids : array-like
        Detected line wavelengths (before offset correction).
    wavelength_range : tuple (wmin, wmax)
    instrumental_offset : float
        Already-measured instrumental offset to subtract first.
    observed_uncertainties : array-like, optional
    stellar_dict : dict, optional
        Override for the stellar line catalogue.
    tolerance : float

    Returns
    -------
    rv_kms : float
        Radial velocity in km/s (positive = receding). NaN if no lines matched.
    rv_err_kms : float
        Uncertainty on rv. NaN if not determined.
    stellar_matches : list of dict
        Matched stellar line pairs (after offset correction).
    """
    if stellar_dict is None:
        stellar_dict = ALL_STELLAR_LINES

    wmin, wmax = wavelength_range
    local_stellar = _lines_in_range(stellar_dict, wmin, wmax, margin=tolerance)

    if not local_stellar:
        return np.nan, np.nan, []

    # Remove instrumental offset before matching
    corrected = np.asarray(observed_centroids, dtype=float) - instrumental_offset

    matches = match_lines(corrected, local_stellar,
                          tolerance=tolerance,
                          observed_uncertainties=observed_uncertainties)

    if not matches:
        return np.nan, np.nan, []

    velocities = np.array([m['velocity'] for m in matches])

    rv_kms = np.median(velocities)
    if len(velocities) > 1:
        rv_err_kms = np.std(velocities) / np.sqrt(len(velocities))
    else:
        rv_err_kms = np.nan

    return rv_kms, rv_err_kms, matches


def estimate_offset_from_stellar(observed_centroids, wavelength_range,
                                 stellar_dict=None, tolerance=50.0):
    """Estimate instrumental offset from stellar absorption lines.

    When no telluric lines are available, the median offset of matched
    stellar lines gives a reasonable estimate of the instrumental zero-point.
    This works because the true radial velocity shift (typically a few km/s)
    contributes <0.5 Å, which is negligible compared to the ~5-20 Å
    instrumental offsets typical of educational spectrographs.

    Parameters
    ----------
    observed_centroids : array-like
        Detected absorption-line wavelengths.
    wavelength_range : tuple (wmin, wmax)
    stellar_dict : dict, optional
    tolerance : float

    Returns
    -------
    offset : float
        Estimated instrumental offset (Angstroms). NaN if no matches.
    offset_err : float
        Scatter of the individual offsets (Angstroms). NaN if <2 matches.
    matches : list of dict
        Matched stellar line pairs.
    """
    if stellar_dict is None:
        stellar_dict = ALL_STELLAR_LINES

    wmin, wmax = wavelength_range
    local_stellar = _lines_in_range(stellar_dict, wmin, wmax, margin=tolerance)

    if not local_stellar:
        return np.nan, np.nan, []

    matches = match_lines(observed_centroids, local_stellar, tolerance=tolerance)

    if not matches:
        return np.nan, np.nan, []

    offsets = np.array([m['offset'] for m in matches])
    offset = np.median(offsets)
    if len(offsets) > 1:
        offset_err = np.std(offsets) / np.sqrt(len(offsets))
    else:
        offset_err = np.nan

    return offset, offset_err, matches


def propagate_offsets(segment_offsets):
    """Fill NaN instrumental offsets by interpolating from calibrated neighbors.

    For segments that lack telluric lines, this propagates the offset
    measured from the nearest segment(s) that do have telluric coverage.

    Parameters
    ----------
    segment_offsets : array-like
        Per-segment offsets; NaN where telluric calibration was not possible.

    Returns
    -------
    filled_offsets : ndarray
        Offsets with NaN values replaced by interpolation/extrapolation.
    interpolated_mask : ndarray of bool
        True where the offset was interpolated (not directly measured).
    """
    offsets = np.array(segment_offsets, dtype=float)
    interpolated = np.isnan(offsets)

    if np.all(interpolated):
        warnings.warn("No segments have telluric calibration; cannot propagate offsets.")
        return np.zeros_like(offsets), interpolated

    if not np.any(interpolated):
        return offsets.copy(), interpolated

    # Indices where we have measurements
    good = np.where(~interpolated)[0]
    # Segment indices (0, 1, 2, ...)
    seg_idx = np.arange(len(offsets))

    if len(good) == 1:
        # Only one calibrated segment: use its offset for all
        offsets[interpolated] = offsets[good[0]]
    else:
        # Linear interpolation, with nearest-neighbor extrapolation at edges
        interp_func = interp1d(good, offsets[good], kind='linear',
                               fill_value='extrapolate')
        offsets[interpolated] = interp_func(seg_idx[interpolated])

    return offsets, interpolated


# ============================================================================
# 2.4 — Cross-correlation method
# ============================================================================

def cross_correlate_velocity(wavelength, flux, template_wavelength,
                             template_flux, velocity_range=(-500, 500),
                             velocity_step=1.0):
    """Measure velocity shift by cross-correlating observed and template spectra.

    Resamples both spectra onto a common log-wavelength grid (so a
    velocity shift corresponds to a constant pixel shift), computes
    the cross-correlation function, and finds its peak.

    Parameters
    ----------
    wavelength : array
        Observed wavelength (Angstroms).
    flux : array
        Observed flux.
    template_wavelength : array
        Template (rest-frame) wavelength.
    template_flux : array
        Template flux.
    velocity_range : tuple (v_min, v_max)
        Range to search in km/s.
    velocity_step : float
        Velocity resolution in km/s.

    Returns
    -------
    best_velocity : float
        Best-fit velocity in km/s (positive = redshift).
    ccf_peak : float
        Peak normalized cross-correlation value (0-1).
    velocities : ndarray
        Velocity grid.
    ccf : ndarray
        Cross-correlation function values.
    """
    # Find overlapping wavelength range
    wmin = max(wavelength.min(), template_wavelength.min())
    wmax = min(wavelength.max(), template_wavelength.max())

    if wmax <= wmin:
        warnings.warn("No wavelength overlap between observation and template.")
        return np.nan, 0.0, np.array([]), np.array([])

    # Build a uniform log-wavelength grid
    # Pixel scale in ln(lambda) corresponds to a constant velocity shift
    dlnw = velocity_step / C_KMS  # delta(ln lambda) per velocity step
    lnw_grid = np.arange(np.log(wmin), np.log(wmax), dlnw)
    w_grid = np.exp(lnw_grid)

    # Interpolate both spectra onto the log-wavelength grid
    obs_interp = interp1d(wavelength, flux, bounds_error=False, fill_value=np.nan)
    tpl_interp = interp1d(template_wavelength, template_flux,
                          bounds_error=False, fill_value=np.nan)

    obs_resamp = obs_interp(w_grid)
    tpl_resamp = tpl_interp(w_grid)

    # Mask NaNs
    valid = np.isfinite(obs_resamp) & np.isfinite(tpl_resamp)
    if np.sum(valid) < 10:
        return np.nan, 0.0, np.array([]), np.array([])

    obs_resamp[~valid] = 0.0
    tpl_resamp[~valid] = 0.0

    # Subtract means (zero-mean for correlation)
    obs_resamp -= np.mean(obs_resamp[valid])
    tpl_resamp -= np.mean(tpl_resamp[valid])

    # Compute cross-correlation via scipy
    ccf_full = correlate(obs_resamp, tpl_resamp, mode='full')

    # Normalize by the geometric mean of autocorrelation peaks
    norm = np.sqrt(np.sum(obs_resamp**2) * np.sum(tpl_resamp**2))
    if norm > 0:
        ccf_full /= norm

    # The lag axis in pixels
    n = len(obs_resamp)
    lags = np.arange(-(n - 1), n)

    # Convert lag to velocity
    velocities_full = lags * velocity_step

    # Restrict to the requested velocity range
    v_mask = (velocities_full >= velocity_range[0]) & (velocities_full <= velocity_range[1])
    velocities = velocities_full[v_mask]
    ccf = ccf_full[v_mask]

    if len(ccf) == 0:
        return np.nan, 0.0, velocities, ccf

    # Find the peak with parabolic refinement
    peak_idx = np.argmax(ccf)
    ccf_peak = ccf[peak_idx]
    best_velocity = velocities[peak_idx]

    # Parabolic refinement around the peak
    if 1 <= peak_idx < len(ccf) - 1:
        y0, y1, y2 = ccf[peak_idx - 1], ccf[peak_idx], ccf[peak_idx + 1]
        denom = 2 * (2 * y1 - y0 - y2)
        if denom != 0:
            delta = (y0 - y2) / denom
            best_velocity += delta * velocity_step
            ccf_peak = y1 - 0.25 * (y0 - y2) * delta

    return best_velocity, ccf_peak, velocities, ccf


# ============================================================================
# 2.5 — Wavelength correction
# ============================================================================

class WavelengthSolution:
    """Container for a wavelength calibration solution for one segment.

    Stores the original and corrected wavelength arrays along with the
    calibration parameters that produced them.
    """

    def __init__(self, segment_id, wavelength_original, flux,
                 instrumental_offset=0.0, radial_velocity_kms=0.0):
        self.segment_id = segment_id
        self.wavelength_original = np.array(wavelength_original, dtype=float)
        self.flux = np.array(flux, dtype=float)
        self.instrumental_offset = instrumental_offset  # Angstroms
        self.radial_velocity_kms = radial_velocity_kms  # km/s

        # Metadata populated during calibration
        self.instrumental_offset_err = np.nan
        self.rv_err_kms = np.nan
        self.n_lines_detected = 0
        self.n_lines_matched = 0
        self.telluric_matches = []
        self.stellar_matches = []
        self.detected_centroids = np.array([])
        self.detected_uncertainties = np.array([])
        self.rms_residual = np.nan
        self.offset_interpolated = False

        # Corrected wavelengths (computed on demand)
        self._wave_observatory = None
        self._wave_stellar_rest = None

    @property
    def wavelength_observatory(self):
        """Wavelength corrected for instrumental offset only (observatory frame)."""
        if self._wave_observatory is None:
            self._wave_observatory = self.wavelength_original - self.instrumental_offset
        return self._wave_observatory

    @property
    def wavelength_stellar_rest(self):
        """Wavelength corrected to the stellar rest frame.

        Removes both instrumental offset and Doppler shift from the
        stellar radial velocity.
        """
        if self._wave_stellar_rest is None:
            # First remove instrumental offset
            w = self.wavelength_original - self.instrumental_offset
            # Then remove Doppler shift: lambda_rest = lambda_obs / (1 + v/c)
            if np.isfinite(self.radial_velocity_kms) and self.radial_velocity_kms != 0:
                w = w / (1 + self.radial_velocity_kms / C_KMS)
            self._wave_stellar_rest = w
        return self._wave_stellar_rest

    def get_corrected_wavelength(self, frame='observatory'):
        """Return corrected wavelength array.

        Parameters
        ----------
        frame : str
            'observatory' — remove only instrumental offset
            'stellar_rest' — remove instrumental offset + radial velocity
            'original' — no correction
        """
        if frame == 'observatory':
            return self.wavelength_observatory
        elif frame == 'stellar_rest':
            return self.wavelength_stellar_rest
        elif frame == 'original':
            return self.wavelength_original.copy()
        else:
            raise ValueError(f"Unknown frame '{frame}'. "
                             "Use 'observatory', 'stellar_rest', or 'original'.")

    def summary_dict(self):
        """Return a summary dictionary for diagnostic tables."""
        return {
            'segment_id': self.segment_id,
            'wave_min': self.wavelength_original.min(),
            'wave_max': self.wavelength_original.max(),
            'n_detected': self.n_lines_detected,
            'n_matched': self.n_lines_matched,
            'inst_offset_A': self.instrumental_offset,
            'inst_offset_err_A': self.instrumental_offset_err,
            'rv_kms': self.radial_velocity_kms,
            'rv_err_kms': self.rv_err_kms,
            'rms_residual_A': self.rms_residual,
            'offset_interpolated': self.offset_interpolated,
        }


def apply_wavelength_correction(wavelength, instrumental_offset=0.0,
                                radial_velocity_kms=0.0,
                                frame='observatory'):
    """Apply wavelength correction to an array.

    Convenience function when you don't need the full WavelengthSolution object.

    Parameters
    ----------
    wavelength : array
    instrumental_offset : float
        In Angstroms (observed - true).
    radial_velocity_kms : float
        Stellar radial velocity in km/s.
    frame : str
        'observatory' or 'stellar_rest'.

    Returns
    -------
    corrected : ndarray
    """
    w = np.asarray(wavelength, dtype=float) - instrumental_offset
    if frame == 'stellar_rest' and np.isfinite(radial_velocity_kms):
        w = w / (1 + radial_velocity_kms / C_KMS)
    return w


# ============================================================================
# 2.6 — Diagnostic outputs
# ============================================================================

def plot_segment_diagnostics(solution, ax=None, show_labels=True):
    """Plot a single segment with detected/matched lines annotated.

    Parameters
    ----------
    solution : WavelengthSolution
    ax : matplotlib Axes, optional
    show_labels : bool
        Label matched lines with their identifications.

    Returns
    -------
    fig : matplotlib Figure (only if ax was None)
    """
    import matplotlib.pyplot as plt

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 4))
        created_fig = True

    wave = solution.wavelength_original
    flux = solution.flux

    ax.plot(wave, flux, 'k-', lw=0.8, alpha=0.8, label='Observed flux')

    # Mark detected lines
    if len(solution.detected_centroids) > 0:
        for c in solution.detected_centroids:
            ax.axvline(c, color='blue', alpha=0.3, lw=0.5)

    # Mark matched telluric lines
    for m in solution.telluric_matches:
        ax.axvline(m['obs_wave'], color='green', alpha=0.7, lw=1.5,
                   linestyle='--')
        if show_labels:
            ax.text(m['obs_wave'], ax.get_ylim()[1] * 0.95,
                    m['line_name'], rotation=90, fontsize=6,
                    color='green', va='top', ha='right')

    # Mark matched stellar lines
    for m in solution.stellar_matches:
        ax.axvline(m['obs_wave'] + solution.instrumental_offset,
                   color='red', alpha=0.7, lw=1.5, linestyle='-.')
        if show_labels:
            ax.text(m['obs_wave'] + solution.instrumental_offset,
                    ax.get_ylim()[1] * 0.90,
                    m['line_name'], rotation=90, fontsize=6,
                    color='red', va='top', ha='right')

    # Annotate the applied shift
    info = (f"Seg {solution.segment_id}: "
            f"offset={solution.instrumental_offset:.2f} Å, "
            f"RV={solution.radial_velocity_kms:.1f} km/s")
    ax.set_title(info, fontsize=10)
    ax.set_xlabel('Wavelength (Å)')
    ax.set_ylabel('Flux (counts)')

    if created_fig:
        fig.tight_layout()
        return fig
    return None


def print_calibration_table(solutions):
    """Print a summary table of calibration results.

    Parameters
    ----------
    solutions : list of WavelengthSolution
    """
    header = (f"{'Seg':>3s}  {'Range (Å)':>18s}  {'Det':>3s}  {'Mat':>3s}  "
              f"{'Offset (Å)':>10s}  {'RV (km/s)':>10s}  {'RMS (Å)':>8s}  "
              f"{'Source':>10s}")
    print(header)
    print('-' * len(header))

    for sol in solutions:
        s = sol.summary_dict()
        source = getattr(sol, '_offset_source', 'telluric' if not s['offset_interpolated'] else 'interp')
        off_str = f"{s['inst_offset_A']:+.2f}" if np.isfinite(s['inst_offset_A']) else '   N/A'
        rv_str = f"{s['rv_kms']:+.1f}" if np.isfinite(s['rv_kms']) else '    N/A'
        rms_str = f"{s['rms_residual_A']:.3f}" if np.isfinite(s['rms_residual_A']) else '  N/A'
        print(f"{s['segment_id']:3d}  "
              f"{s['wave_min']:8.1f}-{s['wave_max']:8.1f}  "
              f"{s['n_detected']:3d}  {s['n_matched']:3d}  "
              f"{off_str:>10s}  {rv_str:>10s}  {rms_str:>8s}  "
              f"{source:>10s}")


def plot_all_segments(solutions, frame='observatory', save_path=None):
    """Plot all segments with diagnostic info, one subplot per segment.

    Parameters
    ----------
    solutions : list of WavelengthSolution
    frame : str
        Which corrected frame to show.
    save_path : str, optional
        If given, save the figure to this path.

    Returns
    -------
    fig : matplotlib Figure
    """
    import matplotlib.pyplot as plt

    n = len(solutions)
    fig, axes = plt.subplots(n, 1, figsize=(14, 3 * n), sharex=False)
    if n == 1:
        axes = [axes]

    for sol, ax in zip(solutions, axes):
        plot_segment_diagnostics(sol, ax=ax)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


# ============================================================================
# Top-level calibration pipeline
# ============================================================================

def calibrate_segments(wavelengths, fluxes, segment_ids=None,
                       frame='observatory',
                       line_finder_kwargs=None,
                       match_tolerance=50.0,
                       telluric_dict=None,
                       stellar_dict=None,
                       cross_correlate=False,
                       template_wavelength=None,
                       template_flux=None,
                       verbose=True):
    """Run the full wavelength calibration pipeline on a set of spectral segments.

    This is the main entry point. For each segment it:
    1. Detects absorption lines (improved derivative method)
    2. Matches telluric lines to measure instrumental offset
    3. Matches stellar lines to measure radial velocity
    4. Optionally cross-correlates against a template
    5. Propagates offsets to segments without telluric coverage
    6. Builds WavelengthSolution objects with corrected wavelengths

    Parameters
    ----------
    wavelengths : list of array
        Wavelength arrays for each segment.
    fluxes : list of array
        Flux arrays for each segment.
    segment_ids : list of int, optional
        Segment identifiers. Defaults to 0, 1, 2, ...
    frame : str
        'observatory' or 'stellar_rest'.
    line_finder_kwargs : dict, optional
        Extra keyword arguments for find_absorption_lines().
    match_tolerance : float
        Tolerance for line matching (Angstroms).
    telluric_dict, stellar_dict : dict, optional
        Override line catalogues.
    cross_correlate : bool
        If True, also run cross-correlation (requires template).
    template_wavelength, template_flux : array, optional
        Template spectrum for cross-correlation.
    verbose : bool
        Print progress and summary table.

    Returns
    -------
    solutions : list of WavelengthSolution
        One per segment, in the same order as the input.
    """
    n_seg = len(wavelengths)
    if segment_ids is None:
        segment_ids = list(range(n_seg))
    if line_finder_kwargs is None:
        line_finder_kwargs = {}
    if telluric_dict is None:
        telluric_dict = TELLURIC_LINES
    if stellar_dict is None:
        stellar_dict = ALL_STELLAR_LINES

    # Per-segment storage
    all_centroids = []
    all_uncertainties = []
    offset_source = []  # 'telluric', 'stellar', 'interpolated', or 'none'

    # ---- Pass 1: detect lines and measure instrumental offsets ----
    # Three-tier offset estimation:
    #   Tier 1 (best):  telluric lines → zero-velocity reference
    #   Tier 2 (good):  median stellar-line offset (works because RV shift
    #                   is <<1 Å, negligible vs the ~10 Å instrumental offset)
    #   Tier 3 (last resort): interpolate from neighbors
    if verbose:
        print("--- Pass 1: line detection & offset measurement ---")

    raw_offsets = np.full(n_seg, np.nan)
    raw_offset_errs = np.full(n_seg, np.nan)
    all_tell_matches = []
    all_star_offset_matches = []  # stellar matches used for offset estimation

    for i in range(n_seg):
        wave = np.asarray(wavelengths[i], dtype=float)
        flux = np.asarray(fluxes[i], dtype=float)
        seg_id = segment_ids[i]

        if verbose:
            print(f"[Seg {seg_id}] {wave.min():.0f}-{wave.max():.0f} Å ...", end=' ')

        centroids, uncertainties, indices, depths = find_absorption_lines(
            wave, flux, **line_finder_kwargs)

        all_centroids.append(centroids)
        all_uncertainties.append(uncertainties)

        wrange = (wave.min(), wave.max())

        # Tier 1: try telluric lines
        inst_off, inst_off_err, tell_matches = measure_instrumental_offset(
            centroids, wrange,
            observed_uncertainties=uncertainties,
            telluric_dict=telluric_dict,
            tolerance=match_tolerance)
        all_tell_matches.append(tell_matches)

        if np.isfinite(inst_off):
            raw_offsets[i] = inst_off
            raw_offset_errs[i] = inst_off_err
            offset_source.append('telluric')
            all_star_offset_matches.append([])
            if verbose:
                print(f"det={len(centroids)}, telluric_matched={len(tell_matches)}, "
                      f"offset={inst_off:+.2f} Å (telluric)")
            continue

        # Tier 2: use stellar-line median offset
        stell_off, stell_off_err, stell_matches = estimate_offset_from_stellar(
            centroids, wrange,
            stellar_dict=stellar_dict,
            tolerance=match_tolerance)
        all_star_offset_matches.append(stell_matches)

        if np.isfinite(stell_off) and len(stell_matches) >= 2:
            raw_offsets[i] = stell_off
            raw_offset_errs[i] = stell_off_err
            offset_source.append('stellar')
            if verbose:
                print(f"det={len(centroids)}, stellar_matched={len(stell_matches)}, "
                      f"offset={stell_off:+.2f} Å (stellar median)")
            continue

        # No offset measurable from this segment
        offset_source.append('none')
        all_star_offset_matches.append([])
        if verbose:
            print(f"det={len(centroids)}, no offset measurable")

    # Tier 3: propagate offsets to remaining uncalibrated segments
    filled_offsets, interp_mask = propagate_offsets(raw_offsets)
    for i in range(n_seg):
        if offset_source[i] == 'none':
            offset_source[i] = 'interpolated'
            if verbose:
                print(f"[Seg {segment_ids[i]}] Offset interpolated: "
                      f"{filled_offsets[i]:+.2f} Å")

    # ---- Pass 2: measure radial velocity with calibrated offsets ----
    if verbose:
        print("\n--- Pass 2: radial velocity measurement ---")

    solutions = []
    raw_rvs = np.full(n_seg, np.nan)

    for i in range(n_seg):
        wave = np.asarray(wavelengths[i], dtype=float)
        flux = np.asarray(fluxes[i], dtype=float)
        seg_id = segment_ids[i]
        centroids = all_centroids[i]
        uncertainties = all_uncertainties[i]
        inst_off = filled_offsets[i]

        wrange = (wave.min(), wave.max())
        tell_matches = all_tell_matches[i]

        if offset_source[i] in ('stellar', 'stellar (re-cal)'):
            # For stellar-offset segments: the offset was the median of
            # stellar match offsets. Compute RV from the residuals around
            # that median using the SAME matches from pass 1.
            star_matches = all_star_offset_matches[i]
            if len(star_matches) >= 2:
                offsets_arr = np.array([m['offset'] for m in star_matches])
                # RV = median residual (individual offset - median offset) / wavelength * c
                residuals = offsets_arr - inst_off
                # Convert each residual to velocity at its wavelength
                vels = np.array([(offsets_arr[j] - inst_off) / star_matches[j]['rest_wave'] * C_KMS
                                 for j in range(len(star_matches))])
                rv = np.median(vels)
                rv_err = np.std(vels) / np.sqrt(len(vels)) if len(vels) > 1 else np.nan
            else:
                # Only 1 stellar line: RV is 0 by construction
                rv = 0.0
                rv_err = np.nan
                star_matches = all_star_offset_matches[i]
        else:
            # For telluric-offset segments: measure RV independently
            rv, rv_err, star_matches = measure_radial_velocity(
                centroids, wrange,
                instrumental_offset=inst_off,
                observed_uncertainties=uncertainties,
                stellar_dict=stellar_dict,
                tolerance=match_tolerance)

        # Optional cross-correlation
        if cross_correlate and template_wavelength is not None:
            ccf_velocity, ccf_peak, _, _ = cross_correlate_velocity(
                wave, flux, template_wavelength, template_flux)
            if verbose and np.isfinite(ccf_velocity):
                print(f"[Seg {seg_id}] CCF: {ccf_velocity:+.1f} km/s "
                      f"(peak={ccf_peak:.3f})")

        raw_rvs[i] = rv

        # Build WavelengthSolution
        sol = WavelengthSolution(seg_id, wave, flux,
                                 instrumental_offset=inst_off,
                                 radial_velocity_kms=rv if np.isfinite(rv) else 0.0)
        sol.instrumental_offset_err = raw_offset_errs[i] if np.isfinite(raw_offset_errs[i]) else np.nan
        sol.rv_err_kms = rv_err
        sol.n_lines_detected = len(centroids)
        sol.n_lines_matched = len(tell_matches) + len(star_matches)
        sol.telluric_matches = tell_matches
        sol.stellar_matches = star_matches
        sol.detected_centroids = centroids
        sol.detected_uncertainties = uncertainties
        sol.offset_interpolated = (offset_source[i] != 'telluric')
        sol._offset_source = offset_source[i]

        # RMS residual of matched lines after removing the median offset
        all_match_offsets = [m['offset'] for m in tell_matches] + [m['offset'] for m in star_matches]
        if all_match_offsets:
            med = np.median(all_match_offsets)
            sol.rms_residual = np.sqrt(np.mean((np.array(all_match_offsets) - med)**2))

        solutions.append(sol)

        if verbose:
            rv_str = f"{rv:+.1f}" if np.isfinite(rv) else "N/A"
            n_star = len(star_matches)
            print(f"[Seg {seg_id}] stellar_matched={n_star}, RV={rv_str} km/s "
                  f"(offset via {offset_source[i]})")

    # ---- Pass 3: outlier detection and re-calibration ----
    # When a telluric-calibrated segment gives an RV wildly different from
    # segments calibrated via stellar lines, the telluric offset likely
    # suffers from an intra-segment dispersion error. Re-estimate the
    # offset from stellar lines instead.
    good_rvs = raw_rvs[np.isfinite(raw_rvs)]
    if len(good_rvs) >= 3:
        median_rv = np.median(good_rvs)
        mad_rv = np.median(np.abs(good_rvs - median_rv))
        rv_threshold = max(50.0, 5 * max(mad_rv, 1.0))  # km/s

        for i in range(n_seg):
            if (np.isfinite(raw_rvs[i]) and
                    offset_source[i] == 'telluric' and
                    abs(raw_rvs[i] - median_rv) > rv_threshold):
                # This segment's telluric-based RV is an outlier.
                # Re-estimate offset from stellar lines.
                centroids = all_centroids[i]
                wave = np.asarray(wavelengths[i], dtype=float)
                wrange = (wave.min(), wave.max())
                stell_off, stell_off_err, stell_matches = estimate_offset_from_stellar(
                    centroids, wrange, stellar_dict=stellar_dict,
                    tolerance=match_tolerance)

                if np.isfinite(stell_off):
                    old_off = filled_offsets[i]
                    filled_offsets[i] = stell_off
                    offset_source[i] = 'stellar (re-cal)'

                    # Re-measure RV
                    rv, rv_err, star_matches = measure_radial_velocity(
                        centroids, wrange,
                        instrumental_offset=stell_off,
                        stellar_dict=stellar_dict,
                        tolerance=match_tolerance)

                    raw_rvs[i] = rv
                    sol = solutions[i]
                    sol.instrumental_offset = stell_off
                    sol.radial_velocity_kms = rv if np.isfinite(rv) else 0.0
                    sol.rv_err_kms = rv_err
                    sol.stellar_matches = star_matches
                    sol.instrumental_offset_err = stell_off_err
                    sol._offset_source = offset_source[i]
                    sol._wave_observatory = None
                    sol._wave_stellar_rest = None

                    if verbose:
                        print(f"[Seg {solutions[i].segment_id}] RV outlier detected "
                              f"({raw_rvs[i]:+.0f} vs median {median_rv:+.0f} km/s). "
                              f"Re-calibrated offset: {old_off:+.2f} -> {stell_off:+.2f} Å")

    # Consensus RV from all segments that measured one
    good_rvs = raw_rvs[np.isfinite(raw_rvs)]
    consensus_rv = np.median(good_rvs) if len(good_rvs) > 0 else 0.0

    # Fill consensus RV for segments that couldn't measure their own
    for i in range(n_seg):
        if not np.isfinite(raw_rvs[i]) and consensus_rv != 0:
            solutions[i].radial_velocity_kms = consensus_rv
            solutions[i]._wave_stellar_rest = None

    if verbose:
        print("\n=== Wavelength Calibration Summary ===")
        print_calibration_table(solutions)
        if len(good_rvs) > 0:
            print(f"\nConsensus radial velocity: {consensus_rv:+.1f} km/s "
                  f"(from {len(good_rvs)} segments)")

    return solutions


# ============================================================================
# Convenience loaders for JJMO data (until Step 1 io.py is available)
# ============================================================================

def load_sirius_segments(data_dir='/home/habjan.e/JJMO_home/Data/Sirius',
                         segment_names=None):
    """Load Sirius data using the existing import_to_fits mechanism.

    Returns wavelength and flux arrays suitable for calibrate_segments().
    """
    import sys
    import os
    # Import existing JJMO functions
    spec_dir = os.path.dirname(os.path.abspath(__file__))
    if spec_dir not in sys.path:
        sys.path.insert(0, spec_dir)
    from JJMO_functions import import_to_fits

    if segment_names is None:
        segment_names = [3900, 4400, 4900, 5400, 5900, 6400, 6900, 7400]

    paths = [os.path.join(data_dir, str(s)) for s in segment_names]
    hdulist = import_to_fits(paths)

    wavelengths = []
    fluxes = []
    for hdu in hdulist:
        crval = hdu.header['CRVAL1']
        cdelt = hdu.header['CDELT1']
        naxis = hdu.header['NAXIS1']
        wave = np.linspace(crval, crval + naxis * cdelt, naxis)
        wavelengths.append(wave)
        fluxes.append(hdu.data.astype(float))

    return wavelengths, fluxes, list(range(len(wavelengths)))


def load_betelgeuse_segments(data_dir='/home/habjan.e/JJMO_home/Data/Betelgeuse',
                             segment_names=None):
    """Load Betelgeuse CSV data.

    Returns wavelength and flux arrays suitable for calibrate_segments().
    """
    import os

    if segment_names is None:
        segment_names = [4400, 4900, 5400, 5900, 6400, 6900, 7400]

    wavelengths = []
    fluxes = []
    for s in segment_names:
        path = os.path.join(data_dir, f'Betelgeuse_{s}.csv')
        data = np.genfromtxt(path, delimiter=',', skip_header=1)
        wavelengths.append(data[:, 1])
        fluxes.append(data[:, 2])

    return wavelengths, fluxes, list(range(len(wavelengths)))
