"""
uncertainties.py — Uncertainty Propagation & Error Budget
=========================================================
Step 8 of the JJMO Spectral Flux Calibration Pipeline.

Tracks and propagates uncertainties through every pipeline step, from raw
counts to final flux-calibrated spectrum.  Produces a realistic error budget
identifying the dominant sources of uncertainty for JJMO-quality data.

Subsections
-----------
8.1  Photon noise and read noise propagation
8.2  Wavelength calibration uncertainty
8.3  Sensitivity function fit uncertainty (including bootstrap)
8.4  Stitching / cross-normalization uncertainty
8.5  Systematic error sources (extinction law, reference spectrum, airmass,
     telluric residuals, slit losses)
8.6  Total error budget table construction
8.7  Monte Carlo end-to-end validation

Authors: JJMO Pipeline
"""

import logging
import warnings
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.interpolate import interp1d

logger = logging.getLogger(__name__)


# ============================================================================
# Constants
# ============================================================================

# Default SBIG ST-7 CCD parameters (JJMO spectrograph)
DEFAULT_READ_NOISE_E = 15.0    # electrons RMS per pixel
DEFAULT_GAIN = 2.3             # e-/ADU (ST-7 typical gain)
DEFAULT_DARK_CURRENT = 1.0     # e-/s/pixel at -10°C (typical for ST-7)

# Telluric bands — regions where sensitivity function is interpolated
# and thus carries larger uncertainty.
TELLURIC_BANDS = [
    (6270.0, 6290.0),
    (6860.0, 6880.0),
    (7150.0, 7300.0),
    (7590.0, 7650.0),
    (8100.0, 8400.0),
]

# Speed of light in km/s
C_KMS = 299792.458


# ============================================================================
# 8.1  Photon noise and read noise propagation
# ============================================================================

def estimate_pixel_uncertainty(counts_2d, gain=None, read_noise=None,
                               dark_current=None, exptime=None):
    """Estimate per-pixel uncertainty from a 2D CCD image.

    Combines Poisson photon noise, read noise, and dark current noise
    in quadrature.

    Parameters
    ----------
    counts_2d : np.ndarray
        2D CCD image in ADU (analog-to-digital units).
    gain : float, optional
        CCD gain in e-/ADU. Defaults to the ST-7 value.
    read_noise : float, optional
        Read noise in electrons RMS. Defaults to ST-7 value.
    dark_current : float, optional
        Dark current in e-/s/pixel. Defaults to ST-7 value.
    exptime : float, optional
        Exposure time in seconds. Needed for dark current term.

    Returns
    -------
    uncertainty_2d : np.ndarray
        Per-pixel 1-sigma uncertainty in ADU, same shape as counts_2d.
    """
    if gain is None:
        gain = DEFAULT_GAIN
    if read_noise is None:
        read_noise = DEFAULT_READ_NOISE_E
    if dark_current is None:
        dark_current = DEFAULT_DARK_CURRENT

    counts_2d = np.asarray(counts_2d, dtype=np.float64)

    # Photon noise: variance = counts_in_electrons / gain^2 (in ADU^2)
    # counts_in_electrons = max(counts_ADU, 0) * gain
    electrons = np.maximum(counts_2d, 0.0) * gain
    photon_var = electrons / (gain ** 2)  # variance in ADU^2

    # Read noise: sigma_read in electrons -> variance in ADU^2
    read_var = (read_noise / gain) ** 2

    # Dark current: variance = dark_current * exptime / gain^2
    dark_var = 0.0
    if exptime is not None and exptime > 0:
        dark_var = (dark_current * exptime) / (gain ** 2)

    total_var = photon_var + read_var + dark_var
    return np.sqrt(total_var)


def propagate_spatial_collapse(uncertainty_2d, collapse_axis=0):
    """Propagate uncertainty through spatial collapse (row summation).

    When summing rows of a 2D CCD frame to produce a 1D spectrum,
    uncertainties add in quadrature.

    Parameters
    ----------
    uncertainty_2d : np.ndarray
        2D per-pixel uncertainty array (same shape as the CCD frame).
    collapse_axis : int
        Axis along which to sum (0 = sum rows = spatial collapse).

    Returns
    -------
    uncertainty_1d : np.ndarray
        1D uncertainty array after spatial collapse.
    """
    uncertainty_2d = np.asarray(uncertainty_2d, dtype=np.float64)
    # Quadrature sum along the collapse axis
    return np.sqrt(np.nansum(uncertainty_2d ** 2, axis=collapse_axis))


def estimate_1d_uncertainty(flux_1d, gain=None, read_noise=None,
                            n_rows=1, dark_current=None, exptime=None):
    """Estimate uncertainty for an already-collapsed 1D spectrum.

    Use this when the 2D frame is unavailable. Reconstructs the uncertainty
    from the 1D counts under the assumption of uniform distribution across
    n_rows spatial rows.

    Parameters
    ----------
    flux_1d : np.ndarray
        1D flux array in ADU (after spatial collapse / row sum).
    gain : float, optional
        CCD gain in e-/ADU.
    read_noise : float, optional
        Read noise in electrons RMS per pixel.
    n_rows : int
        Number of spatial rows that were summed.
    dark_current : float, optional
        Dark current in e-/s/pixel.
    exptime : float, optional
        Exposure time in seconds.

    Returns
    -------
    uncertainty_1d : np.ndarray
        1D per-pixel uncertainty in ADU.
    """
    if gain is None:
        gain = DEFAULT_GAIN
    if read_noise is None:
        read_noise = DEFAULT_READ_NOISE_E
    if dark_current is None:
        dark_current = DEFAULT_DARK_CURRENT

    flux_1d = np.asarray(flux_1d, dtype=np.float64)

    # Photon noise from total counts (already summed over rows)
    electrons_total = np.maximum(flux_1d, 0.0) * gain
    photon_var = electrons_total / (gain ** 2)

    # Read noise: each of n_rows pixels contributes independently
    read_var = n_rows * (read_noise / gain) ** 2

    # Dark current from all summed pixels
    dark_var = 0.0
    if exptime is not None and exptime > 0:
        dark_var = n_rows * (dark_current * exptime) / (gain ** 2)

    total_var = photon_var + read_var + dark_var
    return np.sqrt(total_var)


def propagate_division(flux, flux_unc, divisor, divisor_unc=None):
    """Propagate uncertainty through division: result = flux / divisor.

    Uses standard quadrature addition of fractional uncertainties:
        (sigma_result / result)^2 = (sigma_flux / flux)^2 + (sigma_div / div)^2

    Parameters
    ----------
    flux, flux_unc : np.ndarray
        Numerator and its uncertainty.
    divisor : float or np.ndarray
        Denominator.
    divisor_unc : float or np.ndarray, optional
        Uncertainty on the denominator.

    Returns
    -------
    result : np.ndarray
    result_unc : np.ndarray
    """
    flux = np.asarray(flux, dtype=np.float64)
    flux_unc = np.asarray(flux_unc, dtype=np.float64)
    divisor = np.asarray(divisor, dtype=np.float64)

    # Avoid division by zero
    safe_div = np.where(divisor != 0, divisor, np.nan)
    result = flux / safe_div

    frac_sq = np.zeros_like(result)
    good_flux = np.abs(flux) > 0
    frac_sq[good_flux] = (flux_unc[good_flux] / flux[good_flux]) ** 2

    if divisor_unc is not None:
        divisor_unc = np.asarray(divisor_unc, dtype=np.float64)
        good_div = np.abs(divisor) > 0
        frac_sq[good_div] += (divisor_unc[good_div] / divisor[good_div]) ** 2

    result_unc = np.abs(result) * np.sqrt(frac_sq)
    return result, result_unc


def propagate_multiplication(a, a_unc, b, b_unc=None):
    """Propagate uncertainty through multiplication: result = a * b.

    Parameters
    ----------
    a, a_unc : np.ndarray
        First factor and its uncertainty.
    b : float or np.ndarray
        Second factor.
    b_unc : float or np.ndarray, optional
        Uncertainty on the second factor.

    Returns
    -------
    result : np.ndarray
    result_unc : np.ndarray
    """
    a = np.asarray(a, dtype=np.float64)
    a_unc = np.asarray(a_unc, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    result = a * b

    frac_sq = np.zeros_like(result)
    good_a = np.abs(a) > 0
    frac_sq[good_a] = (a_unc[good_a] / a[good_a]) ** 2

    if b_unc is not None:
        b_unc = np.asarray(b_unc, dtype=np.float64)
        good_b = np.abs(b) > 0
        frac_sq[good_b] += (b_unc[good_b] / b[good_b]) ** 2

    result_unc = np.abs(result) * np.sqrt(frac_sq)
    return result, result_unc


# ============================================================================
# 8.2  Wavelength calibration uncertainty
# ============================================================================

def wavelength_to_flux_uncertainty(wavelength, flux, delta_lambda):
    """Convert wavelength uncertainty into flux uncertainty.

    At each pixel, the flux error from a wavelength shift is:
        delta_F = |dF/dlambda| * delta_lambda

    Near steep spectral features (line wings), this contribution
    can dominate the error budget.

    Parameters
    ----------
    wavelength : np.ndarray
        Wavelength array in Angstroms.
    flux : np.ndarray
        Flux array.
    delta_lambda : float or np.ndarray
        Wavelength uncertainty in Angstroms. Can be a single value
        (applied uniformly) or per-pixel.

    Returns
    -------
    flux_unc_from_wavelength : np.ndarray
        Flux uncertainty contribution from wavelength errors.
    """
    wavelength = np.asarray(wavelength, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64)
    delta_lambda = np.asarray(delta_lambda, dtype=np.float64)

    # Compute |dF/dlambda| using central differences
    # np.gradient requires at least 2 points; for a single point, return 0
    if len(wavelength) < 2:
        return np.zeros_like(flux)
    dfdl = np.gradient(flux, wavelength)

    return np.abs(dfdl) * delta_lambda


def wavelength_calibration_uncertainty(wavelength, flux, rms_residual,
                                       line_centroids=None,
                                       line_uncertainties=None):
    """Compute the flux uncertainty from wavelength calibration errors.

    Uses the RMS residual of the wavelength solution fit as the
    characteristic wavelength uncertainty at each pixel.

    Parameters
    ----------
    wavelength : np.ndarray
        Calibrated wavelength array.
    flux : np.ndarray
        Flux array.
    rms_residual : float
        RMS residual of the wavelength solution in Angstroms.
    line_centroids : np.ndarray, optional
        Wavelengths of identified calibration lines (for reporting).
    line_uncertainties : np.ndarray, optional
        Per-line centroid uncertainties (for reporting).

    Returns
    -------
    flux_unc : np.ndarray
        Flux uncertainty from wavelength calibration.
    report : dict
        Summary containing rms_residual, mean_line_uncertainty, and
        the per-pixel flux uncertainty statistics.
    """
    flux_unc = wavelength_to_flux_uncertainty(wavelength, flux, rms_residual)

    report = {
        "rms_residual_angstrom": float(rms_residual),
        "mean_flux_uncertainty_from_wavelength": float(np.nanmean(flux_unc)),
        "max_flux_uncertainty_from_wavelength": float(np.nanmax(flux_unc)),
    }

    if line_centroids is not None and line_uncertainties is not None:
        report["n_calibration_lines"] = len(line_centroids)
        report["mean_line_centroid_uncertainty"] = float(
            np.nanmean(line_uncertainties)
        )

    return flux_unc, report


# ============================================================================
# 8.3  Sensitivity function fit uncertainty (including bootstrap)
# ============================================================================

def sensitivity_fit_covariance(wavelength, ratio, mask, method='chebyshev',
                               order=5, wave_min=None, wave_max=None):
    """Compute the covariance matrix of the sensitivity fit parameters.

    For polynomial fits (Chebyshev/Legendre), the covariance matrix
    is computed from the least-squares solution. The diagonal gives
    the variance of each coefficient; off-diagonal entries give
    correlations.

    Parameters
    ----------
    wavelength : np.ndarray
        Wavelength grid (masked pixels already excluded).
    ratio : np.ndarray
        Sensitivity ratio at the wavelength points.
    mask : np.ndarray of bool
        True = good pixel to include.
    method : str
        'chebyshev' or 'legendre'.
    order : int
        Polynomial order.
    wave_min, wave_max : float, optional
        Domain bounds. If None, derived from the data.

    Returns
    -------
    coefficients : np.ndarray
        Best-fit coefficients.
    covariance : np.ndarray
        Covariance matrix of shape (order+1, order+1).
    residual_rms : float
        RMS of fit residuals.
    """
    wavelength = np.asarray(wavelength, dtype=np.float64)
    ratio = np.asarray(ratio, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)

    w = wavelength[mask]
    r = ratio[mask]

    if wave_min is None:
        wave_min = w.min()
    if wave_max is None:
        wave_max = w.max()

    # Normalize wavelength to [-1, 1]
    w_norm = 2.0 * (w - wave_min) / (wave_max - wave_min) - 1.0

    if method == 'chebyshev':
        # Build the Chebyshev Vandermonde matrix
        vander = np.polynomial.chebyshev.chebvander(w_norm, order)
    elif method == 'legendre':
        vander = np.polynomial.legendre.legvander(w_norm, order)
    else:
        raise ValueError(
            f"Covariance computation only supported for 'chebyshev' and "
            f"'legendre', got '{method}'"
        )

    # Least-squares fit: r = V @ c  =>  c = (V^T V)^{-1} V^T r
    vtv = vander.T @ vander
    vtr = vander.T @ r

    try:
        vtv_inv = np.linalg.inv(vtv)
    except np.linalg.LinAlgError:
        warnings.warn("Singular Vandermonde matrix; using pseudoinverse.")
        vtv_inv = np.linalg.pinv(vtv)

    coefficients = vtv_inv @ vtr
    fitted = vander @ coefficients
    residuals = r - fitted
    residual_rms = float(np.std(residuals))

    # Covariance: Cov(c) = sigma^2 * (V^T V)^{-1}
    # where sigma^2 is estimated from the residuals
    n = len(r)
    p = order + 1
    dof = max(n - p, 1)
    sigma_sq = float(np.sum(residuals ** 2) / dof)
    covariance = sigma_sq * vtv_inv

    return coefficients, covariance, residual_rms


def propagate_sensitivity_fit_uncertainty(wavelength, coefficients, covariance,
                                          method='chebyshev',
                                          wave_min=None, wave_max=None):
    """Propagate fit parameter covariance to sensitivity uncertainty at each λ.

    The sensitivity at wavelength λ is a function of the fit coefficients c:
        S(λ) = Σ_i c_i T_i(λ_norm)
    where T_i are Chebyshev/Legendre basis polynomials.

    The variance is:
        Var(S(λ)) = basis(λ)^T @ Cov(c) @ basis(λ)

    Parameters
    ----------
    wavelength : np.ndarray
        Wavelengths at which to evaluate the uncertainty.
    coefficients : np.ndarray
        Fit coefficients.
    covariance : np.ndarray
        Covariance matrix of the coefficients.
    method : str
        'chebyshev' or 'legendre'.
    wave_min, wave_max : float
        Domain bounds for normalization.

    Returns
    -------
    sensitivity : np.ndarray
        Evaluated sensitivity values.
    sensitivity_unc : np.ndarray
        1-sigma uncertainty on the sensitivity at each wavelength.
    """
    wavelength = np.asarray(wavelength, dtype=np.float64)
    order = len(coefficients) - 1

    if wave_min is None or wave_max is None:
        raise ValueError("wave_min and wave_max are required")

    w_norm = 2.0 * (wavelength - wave_min) / (wave_max - wave_min) - 1.0

    if method == 'chebyshev':
        vander = np.polynomial.chebyshev.chebvander(w_norm, order)
        sensitivity = np.polynomial.chebyshev.chebval(w_norm, coefficients)
    elif method == 'legendre':
        vander = np.polynomial.legendre.legvander(w_norm, order)
        sensitivity = np.polynomial.legendre.legval(w_norm, coefficients)
    else:
        raise ValueError(f"Unsupported method: {method}")

    # Var(S(λ_i)) = V_i @ Cov @ V_i^T  for each row V_i of the Vandermonde
    # Vectorized: diag(V @ Cov @ V^T)
    variance = np.sum((vander @ covariance) * vander, axis=1)
    sensitivity_unc = np.sqrt(np.maximum(variance, 0.0))

    return sensitivity, sensitivity_unc


def bootstrap_sensitivity_uncertainty(wavelength, ratio, mask,
                                      fit_func, n_bootstrap=200,
                                      random_state=None):
    """Estimate sensitivity function uncertainty via bootstrap resampling.

    Resamples the unmasked data points with replacement, refits, and
    computes the spread across bootstrap realizations.

    Parameters
    ----------
    wavelength : np.ndarray
        Full wavelength grid.
    ratio : np.ndarray
        Sensitivity ratio.
    mask : np.ndarray of bool
        True = good pixel.
    fit_func : callable
        Function that takes (wavelength, ratio, mask) and returns
        an array of fitted values on the full wavelength grid.
        Signature: fit_func(wavelength, ratio, mask) -> np.ndarray
    n_bootstrap : int
        Number of bootstrap realizations.
    random_state : int or np.random.RandomState, optional
        Random seed for reproducibility.

    Returns
    -------
    bootstrap_mean : np.ndarray
        Mean of bootstrap realizations.
    bootstrap_std : np.ndarray
        Standard deviation (uncertainty) across realizations.
    bootstrap_samples : np.ndarray
        Full array of shape (n_bootstrap, len(wavelength)) with all
        realizations, for further analysis.
    """
    if random_state is None:
        rng = np.random.RandomState(42)
    elif isinstance(random_state, int):
        rng = np.random.RandomState(random_state)
    else:
        rng = random_state

    good_idx = np.where(mask)[0]
    n_good = len(good_idx)

    if n_good < 10:
        raise ValueError(
            f"Only {n_good} unmasked pixels; need at least 10 for bootstrap."
        )

    n_wave = len(wavelength)
    samples = np.full((n_bootstrap, n_wave), np.nan)

    for i in range(n_bootstrap):
        # Resample good indices with replacement
        boot_idx = rng.choice(good_idx, size=n_good, replace=True)
        boot_mask = np.zeros(n_wave, dtype=bool)
        boot_mask[boot_idx] = True

        try:
            fitted = fit_func(wavelength, ratio, boot_mask)
            samples[i, :] = fitted
        except Exception as e:
            logger.debug("Bootstrap iteration %d failed: %s", i, e)
            continue

    # Compute statistics ignoring failed iterations
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        bootstrap_mean = np.nanmean(samples, axis=0)
        bootstrap_std = np.nanstd(samples, axis=0)

    return bootstrap_mean, bootstrap_std, samples


# ============================================================================
# 8.4  Stitching / cross-normalization uncertainty
# ============================================================================

def normalization_factor_uncertainty(factor, factor_unc, n_hops):
    """Compute the accumulated normalization uncertainty after n hops.

    When segments are cross-normalized by propagating from a reference
    segment through overlap regions, each hop introduces additional
    uncertainty. For n hops, the total fractional uncertainty grows as
    sqrt(n) * sigma_single, assuming independent errors.

    Parameters
    ----------
    factor : float
        The cumulative normalization factor applied to this segment.
    factor_unc : float
        Single-hop normalization factor uncertainty.
    n_hops : int
        Number of normalization hops from the reference segment.

    Returns
    -------
    cumulative_unc : float
        Total fractional uncertainty on the normalization factor.
    """
    if n_hops <= 0:
        return 0.0
    # Fractional uncertainty per hop
    if abs(factor) < 1e-30:
        return 0.0
    frac_per_hop = abs(factor_unc / factor) if factor_unc > 0 else 0.0
    # Accumulate in quadrature over n hops
    cumulative_frac = frac_per_hop * np.sqrt(n_hops)
    return cumulative_frac * abs(factor)


def stitching_uncertainty(wavelength, flux, norm_factors, reference_idx):
    """Compute the systematic uncertainty floor from cross-normalization.

    Segments farther from the reference accumulate more normalization
    uncertainty. This function returns a per-pixel uncertainty array
    representing the normalization systematic.

    Parameters
    ----------
    wavelength : np.ndarray
        Stitched wavelength grid.
    flux : np.ndarray
        Stitched flux array.
    norm_factors : list of dict or NormFactor-like objects
        Each must have attributes/keys: segment_idx, factor, factor_uncertainty.
    reference_idx : int
        Index of the reference segment (no normalization uncertainty).

    Returns
    -------
    stitch_unc : np.ndarray
        Per-pixel uncertainty from stitching systematics.
    segment_unc_map : dict
        Maps segment_idx -> (n_hops, cumulative_frac_unc).
    """
    # Build a map from segment_idx to its normalization info
    seg_info = {}
    for nf in norm_factors:
        if hasattr(nf, 'segment_idx'):
            idx = nf.segment_idx
            fac = nf.factor
            fac_unc = nf.factor_uncertainty
        else:
            idx = nf['segment_idx']
            fac = nf['factor']
            fac_unc = nf['factor_uncertainty']
        seg_info[idx] = (fac, fac_unc)

    # Determine the number of hops from reference for each segment.
    # The reference has 0 hops; neighbors have 1 hop, etc.
    all_indices = sorted(seg_info.keys())
    if reference_idx not in seg_info and len(all_indices) > 0:
        all_indices = sorted(set(all_indices) | {reference_idx})

    segment_unc_map = {}
    for idx in all_indices:
        n_hops = abs(idx - reference_idx)
        if idx == reference_idx:
            segment_unc_map[idx] = (0, 0.0)
        elif idx in seg_info:
            fac, fac_unc = seg_info[idx]
            cum_unc = normalization_factor_uncertainty(fac, fac_unc, n_hops)
            frac = cum_unc / abs(fac) if abs(fac) > 0 else 0.0
            segment_unc_map[idx] = (n_hops, frac)
        else:
            segment_unc_map[idx] = (n_hops, 0.0)

    # Apply a uniform fractional uncertainty across the full stitched spectrum.
    # Use the maximum fractional uncertainty across all segments as a
    # conservative floor.
    max_frac = max(frac for _, frac in segment_unc_map.values())
    stitch_unc = np.abs(flux) * max_frac

    return stitch_unc, segment_unc_map


# ============================================================================
# 8.5  Systematic error sources
# ============================================================================

def extinction_law_uncertainty(wavelength, flux, airmass, rv_values=None,
                               extinction_laws=None):
    """Estimate flux uncertainty from extinction law choice.

    Computes the flux-calibrated spectrum under different R_V values
    and/or extinction law parameterizations, and reports the spread.

    Parameters
    ----------
    wavelength : np.ndarray
        Wavelength in Angstroms.
    flux : np.ndarray
        Flux that has been corrected with the baseline extinction law.
    airmass : float
        Airmass of the observation.
    rv_values : list of float, optional
        R_V values to try. Default: [2.5, 3.1, 3.5, 4.0].
    extinction_laws : list of callable, optional
        Functions k(wavelength) returning extinction in mag/airmass.
        If None, a simple Rayleigh+ozone parameterization is used
        with varying R_V.

    Returns
    -------
    extinction_unc : np.ndarray
        Per-pixel uncertainty from extinction law variation.
    report : dict
        Describes the extinction variants tested.
    """
    wavelength = np.asarray(wavelength, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64)

    if rv_values is None:
        rv_values = [2.5, 3.1, 3.5, 4.0]

    if extinction_laws is None:
        # Use a simple parametric atmospheric extinction model that
        # depends on R_V (proxy for aerosol scattering):
        # k(λ) = k_rayleigh(λ) + k_aerosol(λ, R_V)
        # k_rayleigh ∝ λ^{-4}, k_aerosol ∝ λ^{-1.3} * (3.1/R_V)
        extinction_laws = []
        for rv in rv_values:
            def make_law(rv_val):
                def law(w):
                    w_um = np.asarray(w, dtype=np.float64) / 10000.0  # A -> μm
                    # Rayleigh scattering (wavelength-independent of R_V)
                    k_ray = 0.00877 * w_um ** (-4.05)
                    # Aerosol component scales with R_V
                    k_aer = 0.025 * (3.1 / rv_val) * w_um ** (-1.3)
                    return k_ray + k_aer
                return law
            extinction_laws.append(make_law(rv))

    # Compute correction factor for each law and evaluate the spread
    flux_variants = []
    for k_func in extinction_laws:
        k_vals = k_func(wavelength)
        # The correction factor relative to the baseline (assumed R_V=3.1)
        k_baseline = extinction_laws[1](wavelength) if len(extinction_laws) > 1 \
            else k_func(wavelength)
        delta_k = k_vals - k_baseline
        # Correction: 10^(0.4 * airmass * delta_k)
        correction = 10.0 ** (0.4 * airmass * delta_k)
        flux_variants.append(flux * correction)

    flux_variants = np.array(flux_variants)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        extinction_unc = np.nanstd(flux_variants, axis=0)

    report = {
        "rv_values_tested": rv_values,
        "n_laws_tested": len(extinction_laws),
        "airmass": float(airmass),
        "mean_extinction_uncertainty": float(np.nanmean(extinction_unc)),
        "max_extinction_uncertainty": float(np.nanmax(extinction_unc)),
    }

    return extinction_unc, report


def reference_spectrum_uncertainty(wavelength, sens_from_calspec,
                                   sens_from_model):
    """Estimate uncertainty from reference spectrum choice.

    Compares sensitivity functions derived from two different reference
    spectra (e.g., CALSPEC empirical vs. PHOENIX model) and reports
    the difference as a systematic uncertainty.

    Parameters
    ----------
    wavelength : np.ndarray
        Common wavelength grid.
    sens_from_calspec : np.ndarray
        Sensitivity derived using CALSPEC reference.
    sens_from_model : np.ndarray
        Sensitivity derived using PHOENIX/Kurucz model.

    Returns
    -------
    ref_unc : np.ndarray
        Per-pixel fractional uncertainty (|S_calspec - S_model| / S_mean).
    report : dict
    """
    wavelength = np.asarray(wavelength, dtype=np.float64)
    s1 = np.asarray(sens_from_calspec, dtype=np.float64)
    s2 = np.asarray(sens_from_model, dtype=np.float64)

    s_mean = 0.5 * (s1 + s2)
    diff = np.abs(s1 - s2)

    with np.errstate(divide='ignore', invalid='ignore'):
        frac_diff = np.where(s_mean > 0, diff / s_mean, 0.0)

    report = {
        "mean_fractional_difference": float(np.nanmean(frac_diff)),
        "max_fractional_difference": float(np.nanmax(frac_diff)),
        "median_fractional_difference": float(np.nanmedian(frac_diff)),
    }

    return frac_diff, report


def airmass_uncertainty(wavelength, flux, airmass, delta_airmass,
                        extinction_curve=None):
    """Propagate airmass uncertainty through the extinction correction.

    If the airmass is approximate (e.g., from an estimated altitude),
    the extinction correction carries additional uncertainty.

    Parameters
    ----------
    wavelength : np.ndarray
        Wavelength in Angstroms.
    flux : np.ndarray
        Extinction-corrected flux.
    airmass : float
        Nominal airmass.
    delta_airmass : float
        Uncertainty on the airmass.
    extinction_curve : callable, optional
        k(λ) function. If None, a default simple model is used.

    Returns
    -------
    airmass_unc : np.ndarray
        Per-pixel flux uncertainty from airmass uncertainty.
    """
    wavelength = np.asarray(wavelength, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64)

    if extinction_curve is None:
        # Simple default: Rayleigh + aerosol
        def extinction_curve(w):
            w_um = np.asarray(w, dtype=np.float64) / 10000.0
            return 0.00877 * w_um ** (-4.05) + 0.025 * w_um ** (-1.3)

    k = extinction_curve(wavelength)

    # The extinction correction factor is 10^(0.4 * X * k(λ))
    # Its derivative with respect to airmass:
    # d(factor)/dX = factor * 0.4 * ln(10) * k(λ)
    # So delta_flux = |flux| * 0.4 * ln(10) * k(λ) * delta_X
    airmass_unc = np.abs(flux) * 0.4 * np.log(10) * k * delta_airmass

    return airmass_unc


def telluric_residual_uncertainty(wavelength, sensitivity_unc,
                                  telluric_bands=None,
                                  inflation_factor=3.0):
    """Inflate uncertainty in telluric absorption regions.

    In masked telluric bands, the sensitivity function is interpolated
    rather than directly fit. The uncertainty there should be larger.

    Parameters
    ----------
    wavelength : np.ndarray
        Wavelength grid.
    sensitivity_unc : np.ndarray
        Baseline sensitivity uncertainty.
    telluric_bands : list of (float, float), optional
        Telluric band boundaries in Angstroms.
    inflation_factor : float
        Factor by which to inflate uncertainty in telluric regions.

    Returns
    -------
    inflated_unc : np.ndarray
        Sensitivity uncertainty with telluric regions inflated.
    in_telluric : np.ndarray of bool
        True for pixels inside a telluric band.
    """
    wavelength = np.asarray(wavelength, dtype=np.float64)
    sensitivity_unc = np.asarray(sensitivity_unc, dtype=np.float64)

    if telluric_bands is None:
        telluric_bands = TELLURIC_BANDS

    in_telluric = np.zeros(len(wavelength), dtype=bool)
    for wmin, wmax in telluric_bands:
        in_telluric |= (wavelength >= wmin) & (wavelength <= wmax)

    inflated_unc = sensitivity_unc.copy()
    inflated_unc[in_telluric] *= inflation_factor

    return inflated_unc, in_telluric


def slit_loss_uncertainty(wavelength, fractional_grey=0.10,
                          fractional_chromatic=0.03):
    """Estimate slit/fiber loss systematic uncertainty.

    Slit losses produce both a grey (constant) and a chromatic
    (wavelength-dependent) systematic. Neither is corrected in this
    pipeline; we only quantify the magnitude.

    Parameters
    ----------
    wavelength : np.ndarray
        Wavelength grid in Angstroms.
    fractional_grey : float
        Estimated fractional grey offset (e.g. 0.10 for 10%).
    fractional_chromatic : float
        Estimated fractional chromatic term at the blue edge,
        scaling as 1/λ relative to the red edge.

    Returns
    -------
    slit_unc : np.ndarray
        Per-pixel fractional uncertainty from slit losses.
    """
    wavelength = np.asarray(wavelength, dtype=np.float64)

    # The grey term is a constant fractional offset
    grey = np.full_like(wavelength, fractional_grey)

    # The chromatic term: worse at short wavelengths
    # Model as linear in 1/λ, normalized so that the blue edge
    # has the full chromatic value and the red edge has ~0.
    if len(wavelength) > 1:
        w_min, w_max = wavelength.min(), wavelength.max()
        # Normalize so chromatic varies from fractional_chromatic to 0
        chromatic = fractional_chromatic * (1.0 / wavelength - 1.0 / w_max) / \
                    (1.0 / w_min - 1.0 / w_max + 1e-30)
    else:
        chromatic = np.full_like(wavelength, fractional_chromatic)

    # Total slit-loss uncertainty: grey and chromatic in quadrature
    slit_unc = np.sqrt(grey ** 2 + chromatic ** 2)

    return slit_unc


# ============================================================================
# 8.6  Total error budget table
# ============================================================================

@dataclass
class ErrorBudget:
    """Container for the total error budget at each wavelength.

    Stores individual uncertainty contributions and the combined total.
    """
    wavelength: np.ndarray
    flux: np.ndarray

    # Individual contributions (all in the same units as flux, or fractional)
    photon_noise: np.ndarray = field(default_factory=lambda: np.array([]))
    read_noise: np.ndarray = field(default_factory=lambda: np.array([]))
    wavelength_cal: np.ndarray = field(default_factory=lambda: np.array([]))
    sensitivity_fit: np.ndarray = field(default_factory=lambda: np.array([]))
    stitching: np.ndarray = field(default_factory=lambda: np.array([]))
    extinction: np.ndarray = field(default_factory=lambda: np.array([]))
    reference_spec: np.ndarray = field(default_factory=lambda: np.array([]))
    airmass_term: np.ndarray = field(default_factory=lambda: np.array([]))
    telluric: np.ndarray = field(default_factory=lambda: np.array([]))
    slit_loss: np.ndarray = field(default_factory=lambda: np.array([]))

    total: np.ndarray = field(default_factory=lambda: np.array([]))
    dominant_source: np.ndarray = field(default_factory=lambda: np.array([]))

    def compute_total(self):
        """Compute the total uncertainty by adding all components in quadrature.

        Also identifies the dominant source at each wavelength.
        """
        n = len(self.wavelength)

        # Collect all non-empty contributions
        components = {}
        for name in ['photon_noise', 'read_noise', 'wavelength_cal',
                      'sensitivity_fit', 'stitching', 'extinction',
                      'reference_spec', 'airmass_term', 'telluric',
                      'slit_loss']:
            arr = getattr(self, name)
            if len(arr) == n:
                components[name] = arr
            elif len(arr) == 0:
                components[name] = np.zeros(n)
            else:
                logger.warning(
                    "ErrorBudget.%s has length %d, expected %d; padding with zeros.",
                    name, len(arr), n
                )
                padded = np.zeros(n)
                padded[:min(len(arr), n)] = arr[:min(len(arr), n)]
                components[name] = padded

        # Quadrature sum
        total_var = np.zeros(n, dtype=np.float64)
        for arr in components.values():
            total_var += np.asarray(arr, dtype=np.float64) ** 2
        self.total = np.sqrt(total_var)

        # Identify dominant source at each pixel
        names = list(components.keys())
        stacked = np.array([components[k] for k in names])
        dominant_idx = np.argmax(np.abs(stacked), axis=0)
        self.dominant_source = np.array([names[i] for i in dominant_idx])

        return self.total

    def to_table(self, n_bins=20):
        """Produce a binned error budget summary table.

        Parameters
        ----------
        n_bins : int
            Number of wavelength bins.

        Returns
        -------
        table : list of dict
            Each entry contains the bin center wavelength and the
            fractional contribution of each error source.
        """
        if len(self.total) == 0:
            self.compute_total()

        n = len(self.wavelength)
        bin_edges = np.linspace(self.wavelength.min(), self.wavelength.max(),
                                n_bins + 1)

        table = []
        for i in range(n_bins):
            in_bin = ((self.wavelength >= bin_edges[i]) &
                      (self.wavelength < bin_edges[i + 1]))
            if not np.any(in_bin):
                continue

            row = {
                "wave_center": float(0.5 * (bin_edges[i] + bin_edges[i + 1])),
                "wave_min": float(bin_edges[i]),
                "wave_max": float(bin_edges[i + 1]),
                "n_pixels": int(np.sum(in_bin)),
            }

            # Mean flux and SNR in this bin
            mean_flux = float(np.nanmean(self.flux[in_bin]))
            mean_total = float(np.nanmean(self.total[in_bin]))
            row["mean_flux"] = mean_flux
            row["mean_total_unc"] = mean_total
            if mean_total > 0:
                row["snr"] = mean_flux / mean_total
            else:
                row["snr"] = np.inf

            # Fractional contribution of each source
            for name in ['photon_noise', 'read_noise', 'wavelength_cal',
                          'sensitivity_fit', 'stitching', 'extinction',
                          'reference_spec', 'airmass_term', 'telluric',
                          'slit_loss']:
                arr = getattr(self, name)
                if len(arr) == n:
                    mean_comp = float(np.nanmean(arr[in_bin]))
                    row[f"unc_{name}"] = mean_comp
                    if mean_total > 0:
                        row[f"frac_{name}"] = (mean_comp / mean_total) ** 2
                    else:
                        row[f"frac_{name}"] = 0.0
                else:
                    row[f"unc_{name}"] = 0.0
                    row[f"frac_{name}"] = 0.0

            # Dominant source
            frac_keys = [f"frac_{name}" for name in
                         ['photon_noise', 'read_noise', 'wavelength_cal',
                          'sensitivity_fit', 'stitching', 'extinction',
                          'reference_spec', 'airmass_term', 'telluric',
                          'slit_loss']]
            comp_names = ['photon_noise', 'read_noise', 'wavelength_cal',
                          'sensitivity_fit', 'stitching', 'extinction',
                          'reference_spec', 'airmass_term', 'telluric',
                          'slit_loss']
            frac_vals = [row.get(k, 0.0) for k in frac_keys]
            row["dominant_source"] = comp_names[np.argmax(frac_vals)]

            table.append(row)

        return table

    def summary(self):
        """Return a human-readable summary of the error budget."""
        if len(self.total) == 0:
            self.compute_total()

        flux = self.flux
        total = self.total
        good = np.isfinite(flux) & np.isfinite(total) & (total > 0)

        if not np.any(good):
            return "Error budget: no valid pixels."

        snr = np.abs(flux[good]) / total[good]

        lines = [
            "Error Budget Summary",
            "=" * 50,
            f"Wavelength range: {self.wavelength.min():.1f} - "
            f"{self.wavelength.max():.1f} A",
            f"Number of pixels: {len(self.wavelength)}",
            f"Median SNR:       {np.median(snr):.1f}",
            f"Min SNR:          {np.min(snr):.1f}",
            f"Max SNR:          {np.max(snr):.1f}",
            "",
            "Mean contribution (fraction of total variance):",
        ]

        total_var = np.mean(total[good] ** 2)
        for name in ['photon_noise', 'read_noise', 'wavelength_cal',
                      'sensitivity_fit', 'stitching', 'extinction',
                      'reference_spec', 'airmass_term', 'telluric',
                      'slit_loss']:
            arr = getattr(self, name)
            if len(arr) == len(self.wavelength):
                mean_var = np.mean(arr[good] ** 2)
                frac = mean_var / total_var if total_var > 0 else 0.0
                lines.append(f"  {name:20s}: {100 * frac:6.2f}%")

        # Dominant source
        if len(self.dominant_source) > 0:
            from collections import Counter
            counts = Counter(self.dominant_source[good])
            most_common = counts.most_common(1)[0]
            lines.append(f"\nDominant source at most pixels: "
                         f"{most_common[0]} ({most_common[1]} pixels)")

        return "\n".join(lines)


def build_error_budget(wavelength, flux, *,
                       photon_noise=None,
                       read_noise=None,
                       wavelength_cal_unc=None,
                       sensitivity_fit_unc=None,
                       stitching_unc=None,
                       extinction_unc=None,
                       reference_spec_unc=None,
                       airmass_unc=None,
                       telluric_unc=None,
                       slit_loss_unc=None):
    """Construct a complete error budget from individual components.

    All uncertainty arrays should be in the same units as flux
    (absolute uncertainty, not fractional).

    Parameters
    ----------
    wavelength : np.ndarray
    flux : np.ndarray
    photon_noise, read_noise, ... : np.ndarray or None
        Individual uncertainty contributions. None values are treated
        as zero.

    Returns
    -------
    ErrorBudget
    """
    n = len(wavelength)

    def _ensure(arr):
        if arr is None:
            return np.zeros(n)
        arr = np.asarray(arr, dtype=np.float64)
        if len(arr) != n:
            raise ValueError(
                f"Uncertainty array length {len(arr)} != wavelength length {n}"
            )
        return arr

    budget = ErrorBudget(
        wavelength=np.asarray(wavelength, dtype=np.float64),
        flux=np.asarray(flux, dtype=np.float64),
        photon_noise=_ensure(photon_noise),
        read_noise=_ensure(read_noise),
        wavelength_cal=_ensure(wavelength_cal_unc),
        sensitivity_fit=_ensure(sensitivity_fit_unc),
        stitching=_ensure(stitching_unc),
        extinction=_ensure(extinction_unc),
        reference_spec=_ensure(reference_spec_unc),
        airmass_term=_ensure(airmass_unc),
        telluric=_ensure(telluric_unc),
        slit_loss=_ensure(slit_loss_unc),
    )

    budget.compute_total()

    return budget


# ============================================================================
# 8.7  Monte Carlo end-to-end validation
# ============================================================================

@dataclass
class MonteCarloResult:
    """Container for Monte Carlo validation results.

    Attributes
    ----------
    wavelength : np.ndarray
        Wavelength grid.
    flux_mean : np.ndarray
        Mean calibrated flux across MC realizations.
    flux_std : np.ndarray
        Standard deviation of calibrated flux (empirical uncertainty).
    analytic_unc : np.ndarray
        Analytically propagated uncertainty for comparison.
    ratio : np.ndarray
        flux_std / analytic_unc — should be ~1.0 if propagation is correct.
    n_realizations : int
        Number of MC realizations completed.
    """
    wavelength: np.ndarray
    flux_mean: np.ndarray
    flux_std: np.ndarray
    analytic_unc: np.ndarray
    ratio: np.ndarray
    n_realizations: int

    def is_consistent(self, tolerance=0.3):
        """Check whether MC and analytic uncertainties agree.

        Parameters
        ----------
        tolerance : float
            Fractional tolerance. Default 0.3 means the median ratio
            must be within [0.7, 1.3].

        Returns
        -------
        consistent : bool
        median_ratio : float
        """
        good = np.isfinite(self.ratio) & (self.ratio > 0)
        if not np.any(good):
            return False, np.nan
        med = float(np.nanmedian(self.ratio[good]))
        return abs(med - 1.0) < tolerance, med

    def summary(self):
        """Human-readable summary."""
        consistent, med_ratio = self.is_consistent()
        status = "PASS" if consistent else "FAIL"

        good = np.isfinite(self.ratio) & (self.ratio > 0)
        lines = [
            "Monte Carlo Validation Summary",
            "=" * 50,
            f"Realizations: {self.n_realizations}",
            f"Wavelength range: {self.wavelength.min():.1f} - "
            f"{self.wavelength.max():.1f} A",
            f"Median MC/analytic ratio: {med_ratio:.3f}",
            f"Status: {status}",
        ]

        if np.any(good):
            lines.append(
                f"Ratio range: [{np.nanmin(self.ratio[good]):.3f}, "
                f"{np.nanmax(self.ratio[good]):.3f}]"
            )
            lines.append(
                f"25th/75th percentile: "
                f"[{np.nanpercentile(self.ratio[good], 25):.3f}, "
                f"{np.nanpercentile(self.ratio[good], 75):.3f}]"
            )

        return "\n".join(lines)


def monte_carlo_validation(wavelength, flux_counts, uncertainty_counts,
                           sensitivity, sensitivity_unc,
                           exptime=1.0,
                           wavelength_unc=0.0,
                           n_realizations=200,
                           random_state=None):
    """Run Monte Carlo end-to-end validation.

    Perturbs the input data at each step (counts noise, wavelength
    solution, sensitivity) and compares the spread of output calibrated
    flux to the analytically propagated uncertainty.

    Parameters
    ----------
    wavelength : np.ndarray
        Wavelength grid in Angstroms.
    flux_counts : np.ndarray
        Observed flux in counts.
    uncertainty_counts : np.ndarray
        Per-pixel 1-sigma uncertainty on the counts.
    sensitivity : np.ndarray
        Sensitivity function S(λ).
    sensitivity_unc : np.ndarray
        Uncertainty on the sensitivity function.
    exptime : float
        Exposure time in seconds.
    wavelength_unc : float or np.ndarray
        Wavelength calibration uncertainty in Angstroms.
    n_realizations : int
        Number of MC trials.
    random_state : int, optional
        Random seed.

    Returns
    -------
    MonteCarloResult
    """
    if random_state is None:
        rng = np.random.RandomState(42)
    elif isinstance(random_state, int):
        rng = np.random.RandomState(random_state)
    else:
        rng = random_state

    wavelength = np.asarray(wavelength, dtype=np.float64)
    flux_counts = np.asarray(flux_counts, dtype=np.float64)
    uncertainty_counts = np.asarray(uncertainty_counts, dtype=np.float64)
    sensitivity = np.asarray(sensitivity, dtype=np.float64)
    sensitivity_unc = np.asarray(sensitivity_unc, dtype=np.float64)
    wavelength_unc = np.asarray(wavelength_unc, dtype=np.float64)

    n_pix = len(wavelength)
    flux_realizations = np.full((n_realizations, n_pix), np.nan)

    for i in range(n_realizations):
        # Perturb counts: Gaussian noise with the estimated uncertainty
        perturbed_counts = flux_counts + rng.normal(0, uncertainty_counts)

        # Perturb wavelength solution
        if np.any(wavelength_unc > 0):
            dw = rng.normal(0, wavelength_unc)
            perturbed_wave = wavelength + dw
        else:
            perturbed_wave = wavelength

        # Perturb sensitivity function
        perturbed_sens = sensitivity + rng.normal(0, sensitivity_unc)
        # Ensure sensitivity stays positive
        perturbed_sens = np.maximum(perturbed_sens, 1e-30)

        # Apply calibration: F = counts / (exptime * S)
        count_rate = perturbed_counts / exptime
        flux_cal = count_rate / perturbed_sens

        # If wavelength was perturbed, interpolate back to the
        # nominal grid to enable pixel-wise comparison
        if np.any(wavelength_unc > 0):
            try:
                interp_func = interp1d(perturbed_wave, flux_cal,
                                       kind='linear', bounds_error=False,
                                       fill_value=np.nan)
                flux_cal = interp_func(wavelength)
            except ValueError:
                pass  # degenerate case, use un-interpolated

        flux_realizations[i, :] = flux_cal

    # Compute empirical statistics
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        flux_mean = np.nanmean(flux_realizations, axis=0)
        flux_std = np.nanstd(flux_realizations, axis=0)

    # Compute the analytic uncertainty for comparison
    # F = counts / (t * S)
    # (σ_F/F)^2 = (σ_counts/counts)^2 + (σ_S/S)^2
    count_rate = flux_counts / exptime
    flux_nominal = count_rate / sensitivity

    frac_counts = np.zeros_like(flux_nominal)
    good_c = np.abs(flux_counts) > 0
    frac_counts[good_c] = (uncertainty_counts[good_c] / flux_counts[good_c]) ** 2

    frac_sens = np.zeros_like(flux_nominal)
    good_s = sensitivity > 0
    frac_sens[good_s] = (sensitivity_unc[good_s] / sensitivity[good_s]) ** 2

    analytic_unc = np.abs(flux_nominal) * np.sqrt(frac_counts + frac_sens)

    # Add wavelength contribution to analytic uncertainty
    if np.any(wavelength_unc > 0):
        wave_flux_unc = wavelength_to_flux_uncertainty(
            wavelength, flux_nominal, wavelength_unc
        )
        analytic_unc = np.sqrt(analytic_unc ** 2 + wave_flux_unc ** 2)

    # Ratio of MC to analytic
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(analytic_unc > 0, flux_std / analytic_unc, np.nan)

    return MonteCarloResult(
        wavelength=wavelength,
        flux_mean=flux_mean,
        flux_std=flux_std,
        analytic_unc=analytic_unc,
        ratio=ratio,
        n_realizations=n_realizations,
    )
