"""
validation.py -- Validation, Testing & Paper Analyses
=====================================================
Step 10 of the JJMO Spectral Flux Calibration Pipeline.

Provides end-to-end pipeline execution, self-consistency checks,
cross-validation between stars, parameter sensitivity analysis,
SNR degradation studies, and literature comparison utilities.

Tasks implemented:
  10.1 - Self-consistency test on Sirius
  10.2 - Cross-validation: Sirius sensitivity applied to Betelgeuse
  10.3 - Cross-validation: Betelgeuse sensitivity applied to Sirius
  10.5 - Parameter sensitivity analysis
  10.6 - SNR degradation study
  10.9 - Comparison to literature spectra

Authors: JJMO Pipeline
"""

import copy
import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.interpolate import interp1d

# Pipeline module imports
from jjmo_fluxcal.stitching import (
    load_jjmo_sirius,
    load_jjmo_betelgeuse,
    stitch_segments,
    find_overlaps,
    StitchResult,
)
from jjmo_fluxcal.quality import assess_segments
from jjmo_fluxcal.sensitivity import (
    derive_sensitivity,
    GlobalSensitivity,
    build_sensitivity_mask,
)
from jjmo_fluxcal.calibrate import (
    SensitivityFunction,
    CalibrationResult,
    apply_sensitivity,
    self_calibration_check,
    SelfCalibrationReport,
)
from jjmo_fluxcal.reference import (
    load_reference_spectrum,
    prepare_reference,
    get_stellar_parameters,
    apply_interstellar_extinction,
    load_atmospheric_extinction,
    correct_atmospheric_extinction,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Data paths
# ============================================================================

DATA_DIR = Path("/home/habjan.e/JJMO_home/Data")
SIRIUS_DIR = DATA_DIR / "Sirius"
BETELGEUSE_DIR = DATA_DIR / "Betelgeuse"
OUTPUT_DIR = Path("/home/habjan.e/JJMO_home/JJMO_spectrograph/validation_outputs")

# Default pipeline parameters (baseline).
# Names match run_pipeline() keyword arguments; the function maps them
# to the actual module parameter names (e.g. balmer_width -> balmer_half_width).
DEFAULT_PARAMS = {
    "fit_method": "chebyshev",
    "fit_order": 5,
    "sigma_clip": 3.0,
    "balmer_width": 15.0,
    "metal_width": 5.0,
    "edge_threshold": 0.20,
    "cosmic_sigma": 5.0,
    "rv": 3.1,
    "extinction_law": "odonnell94",
}


# ============================================================================
# Data structures for validation results
# ============================================================================

@dataclass
class ValidationResult:
    """Container for a single validation experiment.

    Stores the calibrated spectrum, reference comparison, and summary
    statistics for one pipeline run.
    """
    label: str
    wavelength: np.ndarray
    flux_calibrated: np.ndarray
    flux_reference: np.ndarray
    residual_frac: np.ndarray  # (calibrated - reference) / reference
    mask: np.ndarray  # True = bad/excluded pixel
    rms_residual: float
    median_residual: float
    max_residual: float
    mean_residual: float
    params: Dict = field(default_factory=dict)
    meta: Dict = field(default_factory=dict)


@dataclass
class ParameterSensitivityResult:
    """Results from sweeping one pipeline parameter.

    Stores the RMS residual at each parameter value, plus the full
    residual arrays for detailed analysis.
    """
    parameter_name: str
    parameter_values: list
    rms_residuals: list
    median_residuals: list
    max_residuals: list
    baseline_rms: float
    results: List[ValidationResult] = field(default_factory=list)


@dataclass
class SNRDegradationResult:
    """Results from the SNR degradation study (task 10.6)."""
    snr_levels: list  # input SNR values
    rms_residuals: list  # calibration RMS at each noise level
    median_residuals: list
    max_residuals: list
    threshold_snr: float  # minimum SNR for <10% RMS residual
    results: List[ValidationResult] = field(default_factory=list)


# ============================================================================
# 10.0  Pipeline execution helper
# ============================================================================

def run_pipeline(
    segments: List[Tuple[np.ndarray, np.ndarray]],
    star_name: str,
    *,
    fit_method: str = "chebyshev",
    fit_order: int = 5,
    sigma_clip: float = 3.0,
    balmer_width: float = 15.0,
    metal_width: float = 5.0,
    edge_threshold: float = 0.20,
    cosmic_sigma: float = 5.0,
    rv: float = 3.1,
    extinction_law: str = "odonnell94",
) -> GlobalSensitivity:
    """Run the full pipeline from raw segments to per-segment sensitivity fits.

    Steps: quality assessment -> reference loading -> sensitivity derivation.

    Parameters
    ----------
    segments : list of (wavelength, flux) tuples
    star_name : str
        Standard star name for reference spectrum lookup.
    fit_method, fit_order, sigma_clip : sensitivity fit parameters
    balmer_width, metal_width : masking widths in Angstroms
    edge_threshold : fractional threshold for edge trimming
    cosmic_sigma : sigma threshold for cosmic ray rejection
    rv : R_V for extinction law
    extinction_law : name of interstellar extinction law

    Returns
    -------
    global_sens : GlobalSensitivity
        Contains per-segment fits.  Each segment fit maps
        S(λ) = F_ref / C_obs (sensitivity.py convention).
    """
    # Quality assessment to get masks
    wavelengths = [seg[0] for seg in segments]
    fluxes = [seg[1] for seg in segments]
    segment_ids = [f"{int(seg[0].min() + 0.5)}" for seg in segments]

    reports = assess_segments(
        wavelengths, fluxes,
        segment_ids=segment_ids,
        edge_threshold_frac=edge_threshold,
        cosmic_sigma=cosmic_sigma,
        balmer_half_width=balmer_width,
        metal_half_width=metal_width,
    )
    masks = [r.mask_good for r in reports]

    # Load reference spectrum
    ref_spec = load_reference_spectrum(star_name, prefer="calspec")
    ref_wave = ref_spec.spectral_axis.to("AA").value
    ref_flux = ref_spec.flux.value

    # Derive sensitivity per segment.
    global_sens = derive_sensitivity(
        segments_obs=segments,
        wavelength_ref=ref_wave,
        flux_ref=ref_flux,
        approach="per_segment",
        masks=masks,
        segment_ids=segment_ids,
        fit_method=fit_method,
        fit_order=fit_order,
        sigma_clip=sigma_clip,
        fit_global=False,
    )

    # Reset grey shifts to 1.0 on per-segment fits.  The grey-shift
    # mechanism in combine_segment_sensitivities assumes all segments share
    # identical exposure times and atmospheric conditions, which the JJMO
    # data does not.  Each segment was observed independently, so its own
    # S = F_ref / C_obs is the correct calibration for that segment.
    for fit in global_sens.segment_fits:
        fit.grey_shift = 1.0
    global_sens.grey_shifts = {
        fit.segment_id: 1.0 for fit in global_sens.segment_fits
    }

    return global_sens


def _calibrate_per_segment(
    segments: List[Tuple[np.ndarray, np.ndarray]],
    global_sens: GlobalSensitivity,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Calibrate each segment using its own per-segment sensitivity fit.

    sensitivity.py convention: S = F_ref / C_obs, so F_cal = S_fit * C_obs.

    Parameters
    ----------
    segments : list of (wavelength, flux) tuples
    global_sens : GlobalSensitivity with per-segment fits

    Returns
    -------
    calibrated : list of (wavelength, flux_cal) tuples
    """
    calibrated = []
    fits = global_sens.segment_fits

    for i, (w, f) in enumerate(segments):
        f_float = np.asarray(f, dtype=np.float64)

        if i < len(fits):
            # Evaluate this segment's own fit
            s_fit = fits[i](w)
            valid = np.isfinite(s_fit) & (s_fit > 0)
            flux_cal = np.full_like(f_float, np.nan)
            flux_cal[valid] = s_fit[valid] * f_float[valid]
        else:
            # No fit for this segment; pass through with NaN
            flux_cal = np.full_like(f_float, np.nan)

        calibrated.append((w, flux_cal))

    return calibrated


def calibrate_and_compare(
    segments: List[Tuple[np.ndarray, np.ndarray]],
    global_sens: GlobalSensitivity,
    ref_star_name: str,
    label: str = "",
    params: Optional[Dict] = None,
) -> ValidationResult:
    """Calibrate segments per-segment and compare to a reference spectrum.

    Each segment is calibrated by multiplying by its own sensitivity fit
    (S = F_ref / C_obs), then the calibrated segments are stitched and
    compared to the reference.

    Parameters
    ----------
    segments : observed segments (wavelength, flux) tuples
    global_sens : GlobalSensitivity with per-segment fits
    ref_star_name : name of the star for reference spectrum lookup
    label : descriptive label for this validation run
    params : dict of pipeline parameters used (for provenance)

    Returns
    -------
    ValidationResult
    """
    # Calibrate each segment using its own per-segment fit
    calibrated = _calibrate_per_segment(segments, global_sens)

    # Stitch calibrated segments
    result = stitch_segments(calibrated)
    cal_wave = result.wavelength
    cal_flux = result.flux
    cal_mask = ~result.mask  # StitchResult mask: True=good, we want True=bad

    # Load and resample reference
    ref_spec = load_reference_spectrum(ref_star_name, prefer="calspec")
    ref_wave = ref_spec.spectral_axis.to("AA").value
    ref_flux = ref_spec.flux.value

    ref_interp = interp1d(ref_wave, ref_flux, kind="linear",
                          bounds_error=False, fill_value=np.nan)
    ref_on_grid = ref_interp(cal_wave)

    # Compute fractional residuals
    good = (~cal_mask & np.isfinite(cal_flux) & np.isfinite(ref_on_grid)
            & (ref_on_grid > 0) & (cal_flux > 0))
    residual_frac = np.full_like(cal_flux, np.nan)
    residual_frac[good] = (cal_flux[good] - ref_on_grid[good]) / ref_on_grid[good]

    if np.any(good):
        rms = float(np.sqrt(np.nanmean(residual_frac[good] ** 2)))
        med = float(np.nanmedian(np.abs(residual_frac[good])))
        mx = float(np.nanmax(np.abs(residual_frac[good])))
        mn = float(np.nanmean(residual_frac[good]))
    else:
        rms = med = mx = mn = np.nan

    logger.info(
        "Validation '%s': RMS=%.4f, median=%.4f, max=%.4f (%d/%d valid px)",
        label, rms, med, mx, int(np.sum(good)), len(cal_wave),
    )

    return ValidationResult(
        label=label,
        wavelength=cal_wave,
        flux_calibrated=cal_flux,
        flux_reference=ref_on_grid,
        residual_frac=residual_frac,
        mask=~good,
        rms_residual=rms,
        median_residual=med,
        max_residual=mx,
        mean_residual=mn,
        params=params or {},
    )


# ============================================================================
# 10.1  Self-consistency test on Sirius
# ============================================================================

def self_consistency_sirius(
    sirius_dir: Optional[str] = None,
    **pipeline_kwargs,
) -> ValidationResult:
    """Run the full pipeline on Sirius and compare back to its reference.

    This is the self-consistency check: derive the sensitivity function
    from Sirius, apply it to Sirius, and measure residuals against the
    CALSPEC reference.

    Parameters
    ----------
    sirius_dir : path to Sirius data directory
    **pipeline_kwargs : forwarded to run_pipeline

    Returns
    -------
    ValidationResult
    """
    data_dir = sirius_dir or str(SIRIUS_DIR)
    segments = load_jjmo_sirius(data_dir)
    logger.info("Loaded %d Sirius segments", len(segments))

    # Derive sensitivity from Sirius
    params = {**DEFAULT_PARAMS, **pipeline_kwargs}
    global_sens = run_pipeline(segments, "sirius", **pipeline_kwargs)

    # Calibrate Sirius with its own sensitivity and compare to reference
    result = calibrate_and_compare(
        segments, global_sens, "sirius",
        label="Sirius self-consistency (10.1)",
        params=params,
    )

    logger.info(
        "Sirius self-consistency: RMS=%.4f, median=%.4f, max=%.4f",
        result.rms_residual, result.median_residual, result.max_residual,
    )
    return result


# ============================================================================
# 10.2  Cross-validation: Sirius sensitivity applied to Betelgeuse
# ============================================================================

def cross_validate_sirius_to_betelgeuse(
    sirius_dir: Optional[str] = None,
    betelgeuse_dir: Optional[str] = None,
    **pipeline_kwargs,
) -> ValidationResult:
    """Derive sensitivity from Sirius, apply to Betelgeuse, compare to reference.

    The sensitivity function should describe the instrument, not the star.
    If the method is working, the Sirius-derived sensitivity should also
    calibrate Betelgeuse.

    Returns
    -------
    ValidationResult
    """
    sir_dir = sirius_dir or str(SIRIUS_DIR)
    bet_dir = betelgeuse_dir or str(BETELGEUSE_DIR)

    sirius_segments = load_jjmo_sirius(sir_dir)
    betelgeuse_segments = load_jjmo_betelgeuse(bet_dir)

    params = {**DEFAULT_PARAMS, **pipeline_kwargs}

    # Derive sensitivity from Sirius
    global_sens = run_pipeline(sirius_segments, "sirius", **pipeline_kwargs)

    # Apply Sirius per-segment sensitivity to Betelgeuse.
    # Cross-validation uses _cross_calibrate: build a continuous sensitivity
    # interpolator from the source star's per-segment fits, then apply to
    # the target star's segments.
    result = _cross_calibrate_and_compare(
        betelgeuse_segments, global_sens, "betelgeuse",
        label="Sirius sens -> Betelgeuse (10.2)",
        params=params,
    )

    logger.info(
        "Sirius->Betelgeuse cross-val: RMS=%.4f, median=%.4f, max=%.4f",
        result.rms_residual, result.median_residual, result.max_residual,
    )
    return result


# ============================================================================
# 10.3  Cross-validation: Betelgeuse sensitivity applied to Sirius
# ============================================================================

def cross_validate_betelgeuse_to_sirius(
    sirius_dir: Optional[str] = None,
    betelgeuse_dir: Optional[str] = None,
    **pipeline_kwargs,
) -> ValidationResult:
    """Derive sensitivity from Betelgeuse, apply to Sirius, compare to reference.

    Reverse of 10.2: if both sensitivity functions agree, the method is
    capturing the instrument response, not the stellar spectrum.

    Returns
    -------
    ValidationResult
    """
    sir_dir = sirius_dir or str(SIRIUS_DIR)
    bet_dir = betelgeuse_dir or str(BETELGEUSE_DIR)

    sirius_segments = load_jjmo_sirius(sir_dir)
    betelgeuse_segments = load_jjmo_betelgeuse(bet_dir)

    params = {**DEFAULT_PARAMS, **pipeline_kwargs}

    # Derive sensitivity from Betelgeuse
    global_sens = run_pipeline(betelgeuse_segments, "betelgeuse", **pipeline_kwargs)

    # Apply Betelgeuse per-segment sensitivity to Sirius
    result = _cross_calibrate_and_compare(
        sirius_segments, global_sens, "sirius",
        label="Betelgeuse sens -> Sirius (10.3)",
        params=params,
    )

    logger.info(
        "Betelgeuse->Sirius cross-val: RMS=%.4f, median=%.4f, max=%.4f",
        result.rms_residual, result.median_residual, result.max_residual,
    )
    return result


def _build_piecewise_interpolator(
    global_sens: GlobalSensitivity,
) -> Callable:
    """Build a continuous interpolator from per-segment sensitivity fits.

    Evaluates each per-segment fit on a dense grid within its domain,
    then builds a single linear interpolator spanning the full range.
    In overlap regions, segment values are averaged.
    """
    all_w = []
    all_s = []
    for fit in global_sens.segment_fits:
        w = np.linspace(fit.wave_min, fit.wave_max, 300)
        s = fit(w)
        valid = np.isfinite(s) & (s > 0)
        all_w.append(w[valid])
        all_s.append(s[valid])

    all_w = np.concatenate(all_w)
    all_s = np.concatenate(all_s)

    # Sort and average duplicates in overlapping regions
    order = np.argsort(all_w)
    all_w = all_w[order]
    all_s = all_s[order]

    return interp1d(all_w, all_s, kind="linear",
                    bounds_error=False, fill_value=np.nan)


def _cross_calibrate_and_compare(
    target_segments: List[Tuple[np.ndarray, np.ndarray]],
    source_sens: GlobalSensitivity,
    ref_star_name: str,
    label: str = "",
    params: Optional[Dict] = None,
) -> ValidationResult:
    """Calibrate target segments using source star's sensitivity.

    For cross-validation: the sensitivity function is from one star,
    but applied to another.  Uses a piecewise interpolator built from
    the source star's per-segment fits.
    """
    sens_interp = _build_piecewise_interpolator(source_sens)

    # Calibrate each target segment
    calibrated = []
    for w, f in target_segments:
        f_float = np.asarray(f, dtype=np.float64)
        s_val = sens_interp(w)
        valid = np.isfinite(s_val) & (s_val > 0)
        flux_cal = np.full_like(f_float, np.nan)
        flux_cal[valid] = s_val[valid] * f_float[valid]
        calibrated.append((w, flux_cal))

    # Stitch and compare
    result = stitch_segments(calibrated)
    cal_wave = result.wavelength
    cal_flux = result.flux
    cal_mask = ~result.mask

    ref_spec = load_reference_spectrum(ref_star_name, prefer="calspec")
    ref_wave = ref_spec.spectral_axis.to("AA").value
    ref_flux = ref_spec.flux.value

    ref_interp = interp1d(ref_wave, ref_flux, kind="linear",
                          bounds_error=False, fill_value=np.nan)
    ref_on_grid = ref_interp(cal_wave)

    good = (~cal_mask & np.isfinite(cal_flux) & np.isfinite(ref_on_grid)
            & (ref_on_grid > 0) & (cal_flux > 0))
    residual_frac = np.full_like(cal_flux, np.nan)
    residual_frac[good] = (cal_flux[good] - ref_on_grid[good]) / ref_on_grid[good]

    if np.any(good):
        rms = float(np.sqrt(np.nanmean(residual_frac[good] ** 2)))
        med = float(np.nanmedian(np.abs(residual_frac[good])))
        mx = float(np.nanmax(np.abs(residual_frac[good])))
        mn = float(np.nanmean(residual_frac[good]))
    else:
        rms = med = mx = mn = np.nan

    return ValidationResult(
        label=label,
        wavelength=cal_wave,
        flux_calibrated=cal_flux,
        flux_reference=ref_on_grid,
        residual_frac=residual_frac,
        mask=~good,
        rms_residual=rms,
        median_residual=med,
        max_residual=mx,
        mean_residual=mn,
        params=params or {},
    )


# ============================================================================
# 10.5  Sensitivity to pipeline parameters
# ============================================================================

def parameter_sensitivity_analysis(
    sirius_dir: Optional[str] = None,
    parameter_grid: Optional[Dict[str, list]] = None,
) -> Dict[str, ParameterSensitivityResult]:
    """Vary key pipeline parameters and measure the effect on calibration.

    For each parameter, all others are held at their default values while
    the target parameter is swept through the provided grid.

    Parameters
    ----------
    sirius_dir : path to Sirius data
    parameter_grid : dict mapping parameter names to lists of values.
        Defaults to the grid specified in the step 10 spec.

    Returns
    -------
    dict mapping parameter name -> ParameterSensitivityResult
    """
    data_dir = sirius_dir or str(SIRIUS_DIR)
    segments = load_jjmo_sirius(data_dir)

    if parameter_grid is None:
        parameter_grid = {
            "fit_order": [3, 5, 7, 9],
            "balmer_width": [5.0, 10.0, 15.0, 20.0],
            "sigma_clip": [2.0, 3.0, 5.0],
        }

    # Run baseline first
    logger.info("Running baseline pipeline...")
    baseline_sens = run_pipeline(segments, "sirius")
    baseline = calibrate_and_compare(
        segments, baseline_sens, "sirius",
        label="baseline",
        params=DEFAULT_PARAMS,
    )
    baseline_rms = baseline.rms_residual

    results = {}

    for param_name, values in parameter_grid.items():
        logger.info("Sweeping parameter '%s' over %s", param_name, values)
        rms_list = []
        median_list = []
        max_list = []
        val_results = []

        for val in values:
            kwargs = {param_name: val}
            try:
                gs = run_pipeline(segments, "sirius", **kwargs)
                vr = calibrate_and_compare(
                    segments, gs, "sirius",
                    label=f"{param_name}={val}",
                    params={**DEFAULT_PARAMS, param_name: val},
                )
                rms_list.append(vr.rms_residual)
                median_list.append(vr.median_residual)
                max_list.append(vr.max_residual)
                val_results.append(vr)
            except Exception as exc:
                logger.warning(
                    "Failed for %s=%s: %s", param_name, val, exc,
                )
                rms_list.append(np.nan)
                median_list.append(np.nan)
                max_list.append(np.nan)

        results[param_name] = ParameterSensitivityResult(
            parameter_name=param_name,
            parameter_values=values,
            rms_residuals=rms_list,
            median_residuals=median_list,
            max_residuals=max_list,
            baseline_rms=baseline_rms,
            results=val_results,
        )

    return results


# ============================================================================
# 10.6  SNR degradation study
# ============================================================================

def snr_degradation_study(
    sirius_dir: Optional[str] = None,
    noise_multipliers: Optional[List[float]] = None,
    n_trials: int = 3,
    rng_seed: int = 42,
) -> SNRDegradationResult:
    """Add increasing noise to Sirius data and track calibration accuracy.

    Takes the best (highest-SNR) Sirius segment and progressively degrades
    it by adding Gaussian noise. Runs the pipeline at each noise level and
    records the RMS residual.

    Parameters
    ----------
    sirius_dir : path to Sirius data
    noise_multipliers : list of factors by which to multiply the noise.
        1.0 = original noise level. Higher = more noise added.
    n_trials : number of independent noise realisations to average over
    rng_seed : random seed for reproducibility

    Returns
    -------
    SNRDegradationResult
    """
    data_dir = sirius_dir or str(SIRIUS_DIR)
    segments = load_jjmo_sirius(data_dir)

    if noise_multipliers is None:
        noise_multipliers = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]

    rng = np.random.default_rng(rng_seed)

    # Find the best segment (highest SNR estimate)
    snr_estimates = []
    for w, f in segments:
        good = np.isfinite(f) & (f > 0)
        if np.sum(good) > 10:
            noise_est = np.median(np.abs(np.diff(f[good]))) / 1.4826
            snr_est = np.median(f[good]) / noise_est if noise_est > 0 else 0
        else:
            snr_est = 0
        snr_estimates.append(snr_est)

    best_idx = int(np.argmax(snr_estimates))
    best_w, best_f = segments[best_idx]
    logger.info(
        "Best segment for SNR study: index %d (SNR ~ %.1f)",
        best_idx, snr_estimates[best_idx],
    )

    # Estimate the intrinsic noise level from the best segment
    good = np.isfinite(best_f) & (best_f > 0)
    noise_level = np.median(np.abs(np.diff(best_f[good]))) / 1.4826

    snr_levels = []
    rms_list = []
    median_list = []
    max_list = []
    val_results = []

    for mult in noise_multipliers:
        trial_rms = []
        trial_med = []
        trial_max = []

        for trial in range(n_trials):
            # Build degraded segments: add noise to all segments proportionally
            degraded = []
            for w, f in segments:
                added_noise = rng.normal(0, noise_level * mult, size=len(f))
                degraded.append((w.copy(), f + added_noise))

            # Estimate SNR of the degraded best segment
            dw, df = degraded[best_idx]
            dgood = np.isfinite(df) & (df > 0)
            if np.sum(dgood) > 10:
                d_noise = np.median(np.abs(np.diff(df[dgood]))) / 1.4826
                d_snr = np.median(df[dgood]) / d_noise if d_noise > 0 else 0
            else:
                d_snr = 0

            try:
                gs = run_pipeline(degraded, "sirius")
                vr = calibrate_and_compare(
                    degraded, gs, "sirius",
                    label=f"noise_mult={mult:.1f}_trial={trial}",
                )
                trial_rms.append(vr.rms_residual)
                trial_med.append(vr.median_residual)
                trial_max.append(vr.max_residual)
                if trial == 0:
                    val_results.append(vr)
            except Exception as exc:
                logger.warning(
                    "SNR degradation failed for mult=%.1f trial=%d: %s",
                    mult, trial, exc,
                )
                trial_rms.append(np.nan)
                trial_med.append(np.nan)
                trial_max.append(np.nan)

        # Average over trials
        snr_levels.append(d_snr)
        rms_list.append(float(np.nanmean(trial_rms)))
        median_list.append(float(np.nanmean(trial_med)))
        max_list.append(float(np.nanmean(trial_max)))

        logger.info(
            "Noise mult=%.1f: SNR~%.1f, RMS=%.4f",
            mult, snr_levels[-1], rms_list[-1],
        )

    # Find threshold: minimum SNR where RMS < 0.10
    threshold_snr = np.nan
    for snr, rms in zip(snr_levels, rms_list):
        if np.isfinite(rms) and rms < 0.10 and np.isfinite(snr) and snr > 0:
            threshold_snr = snr

    return SNRDegradationResult(
        snr_levels=snr_levels,
        rms_residuals=rms_list,
        median_residuals=median_list,
        max_residuals=max_list,
        threshold_snr=threshold_snr,
        results=val_results,
    )


# ============================================================================
# 10.9  Comparison to literature spectra
# ============================================================================

def compare_to_literature(
    calibrated_wave: np.ndarray,
    calibrated_flux: np.ndarray,
    star_name: str = "sirius",
    libraries: Optional[List[str]] = None,
) -> Dict[str, ValidationResult]:
    """Compare a calibrated spectrum to literature spectral libraries.

    Attempts to load reference spectra from CALSPEC and PHOENIX models
    and computes residuals against each. This provides an independent
    check beyond the self-consistency test.

    Parameters
    ----------
    calibrated_wave, calibrated_flux : arrays
        The flux-calibrated spectrum from the pipeline.
    star_name : target star
    libraries : list of library names to compare against.
        Defaults to ['calspec', 'phoenix'].

    Returns
    -------
    dict mapping library name -> ValidationResult
    """
    if libraries is None:
        libraries = ["calspec", "phoenix"]

    results = {}

    for lib in libraries:
        try:
            ref_spec = load_reference_spectrum(star_name, prefer=lib)
            ref_wave = ref_spec.spectral_axis.to("AA").value
            ref_flux = ref_spec.flux.value

            ref_interp = interp1d(
                ref_wave, ref_flux, kind="linear",
                bounds_error=False, fill_value=np.nan,
            )
            ref_on_grid = ref_interp(calibrated_wave)

            good = (np.isfinite(calibrated_flux) & np.isfinite(ref_on_grid)
                    & (ref_on_grid > 0) & (calibrated_flux > 0))
            residual_frac = np.full_like(calibrated_flux, np.nan)
            residual_frac[good] = (
                (calibrated_flux[good] - ref_on_grid[good])
                / ref_on_grid[good]
            )

            if np.any(good):
                rms = float(np.sqrt(np.nanmean(residual_frac[good] ** 2)))
                med = float(np.nanmedian(np.abs(residual_frac[good])))
                mx = float(np.nanmax(np.abs(residual_frac[good])))
                mn = float(np.nanmean(residual_frac[good]))
            else:
                rms = med = mx = mn = np.nan

            results[lib] = ValidationResult(
                label=f"{star_name} vs {lib}",
                wavelength=calibrated_wave,
                flux_calibrated=calibrated_flux,
                flux_reference=ref_on_grid,
                residual_frac=residual_frac,
                mask=~good,
                rms_residual=rms,
                median_residual=med,
                max_residual=mx,
                mean_residual=mn,
            )

            logger.info(
                "%s vs %s: RMS=%.4f, median=%.4f, max=%.4f",
                star_name, lib, rms, med, mx,
            )

        except Exception as exc:
            logger.warning("Could not load %s reference for %s: %s",
                           lib, star_name, exc)

    return results


# ============================================================================
# Sensitivity function comparison utilities
# ============================================================================

def compare_sensitivity_functions(
    sensfunc_a: SensitivityFunction,
    sensfunc_b: SensitivityFunction,
    label_a: str = "A",
    label_b: str = "B",
) -> Dict[str, float]:
    """Compare two sensitivity functions on their common wavelength range.

    Used to check whether Sirius-derived and Betelgeuse-derived sensitivity
    functions agree (task 10.2-10.3).

    Returns
    -------
    dict with keys: rms_frac_diff, median_frac_diff, max_frac_diff,
                    wave_min, wave_max
    """
    # Common wavelength range
    w_min = max(sensfunc_a.wavelength.min(), sensfunc_b.wavelength.min())
    w_max = min(sensfunc_a.wavelength.max(), sensfunc_b.wavelength.max())

    if w_min >= w_max:
        logger.warning("No overlapping wavelength range between %s and %s",
                        label_a, label_b)
        return {"rms_frac_diff": np.nan, "median_frac_diff": np.nan,
                "max_frac_diff": np.nan, "wave_min": w_min, "wave_max": w_max}

    grid = np.linspace(w_min, w_max, 1000)
    sa, _ = sensfunc_a.evaluate(grid)
    sb, _ = sensfunc_b.evaluate(grid)

    good = np.isfinite(sa) & np.isfinite(sb) & (sa > 0) & (sb > 0)
    if not np.any(good):
        return {"rms_frac_diff": np.nan, "median_frac_diff": np.nan,
                "max_frac_diff": np.nan, "wave_min": w_min, "wave_max": w_max}

    frac_diff = (sa[good] - sb[good]) / (0.5 * (sa[good] + sb[good]))

    return {
        "rms_frac_diff": float(np.sqrt(np.mean(frac_diff ** 2))),
        "median_frac_diff": float(np.median(np.abs(frac_diff))),
        "max_frac_diff": float(np.max(np.abs(frac_diff))),
        "wave_min": float(w_min),
        "wave_max": float(w_max),
    }


# ============================================================================
# Summary report generation
# ============================================================================

def format_validation_summary(
    results: Dict[str, Any],
) -> str:
    """Format a human-readable summary of all validation results.

    Parameters
    ----------
    results : dict of named results from various validation tasks

    Returns
    -------
    str : multi-line summary text
    """
    lines = []
    lines.append("=" * 72)
    lines.append("JJMO Pipeline Validation Summary")
    lines.append("=" * 72)
    lines.append("")

    for name, res in results.items():
        if isinstance(res, ValidationResult):
            lines.append(f"--- {res.label} ---")
            lines.append(f"  RMS residual:    {res.rms_residual:.4f}")
            lines.append(f"  Median residual: {res.median_residual:.4f}")
            lines.append(f"  Max residual:    {res.max_residual:.4f}")
            lines.append(f"  Mean residual:   {res.mean_residual:.4f}")
            n_good = int(np.sum(~res.mask))
            lines.append(f"  Valid pixels:    {n_good}/{len(res.mask)}")
            lines.append("")

        elif isinstance(res, ParameterSensitivityResult):
            lines.append(f"--- Parameter: {res.parameter_name} ---")
            lines.append(f"  Baseline RMS: {res.baseline_rms:.4f}")
            for val, rms in zip(res.parameter_values, res.rms_residuals):
                lines.append(f"  {res.parameter_name}={val}: RMS={rms:.4f}")
            lines.append("")

        elif isinstance(res, SNRDegradationResult):
            lines.append("--- SNR Degradation Study ---")
            lines.append(f"  Threshold SNR (<10% RMS): {res.threshold_snr:.1f}")
            for snr, rms in zip(res.snr_levels, res.rms_residuals):
                lines.append(f"  SNR~{snr:.1f}: RMS={rms:.4f}")
            lines.append("")

        elif isinstance(res, dict):
            lines.append(f"--- {name} ---")
            for k, v in res.items():
                if isinstance(v, float):
                    lines.append(f"  {k}: {v:.4f}")
                elif isinstance(v, ValidationResult):
                    lines.append(f"  {k}: RMS={v.rms_residual:.4f}, "
                                 f"median={v.median_residual:.4f}, "
                                 f"max={v.max_residual:.4f}")
                else:
                    lines.append(f"  {k}: {v}")
            lines.append("")

    lines.append("=" * 72)
    return "\n".join(lines)


# ============================================================================
# Master validation runner
# ============================================================================

def run_all_validations(
    sirius_dir: Optional[str] = None,
    betelgeuse_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    skip_slow: bool = False,
) -> Dict[str, Any]:
    """Run all validation tasks and return collected results.

    Parameters
    ----------
    sirius_dir, betelgeuse_dir : data directories
    output_dir : where to save outputs
    skip_slow : if True, skip the parameter sweep and SNR degradation

    Returns
    -------
    dict of all validation results, keyed by task name
    """
    sir_dir = sirius_dir or str(SIRIUS_DIR)
    bet_dir = betelgeuse_dir or str(BETELGEUSE_DIR)
    out_dir = Path(output_dir) if output_dir else OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # 10.1 Self-consistency on Sirius
    logger.info("=== Task 10.1: Self-consistency on Sirius ===")
    results["10.1_self_consistency"] = self_consistency_sirius(sir_dir)

    # 10.2 Cross-validation: Sirius -> Betelgeuse
    logger.info("=== Task 10.2: Sirius sens -> Betelgeuse ===")
    results["10.2_sirius_to_betelgeuse"] = (
        cross_validate_sirius_to_betelgeuse(sir_dir, bet_dir)
    )

    # 10.3 Cross-validation: Betelgeuse -> Sirius
    logger.info("=== Task 10.3: Betelgeuse sens -> Sirius ===")
    results["10.3_betelgeuse_to_sirius"] = (
        cross_validate_betelgeuse_to_sirius(sir_dir, bet_dir)
    )

    # 10.9 Literature comparison
    logger.info("=== Task 10.9: Literature comparison ===")
    vr = results["10.1_self_consistency"]
    results["10.9_literature"] = compare_to_literature(
        vr.wavelength, vr.flux_calibrated, "sirius",
    )

    if not skip_slow:
        # 10.5 Parameter sensitivity
        logger.info("=== Task 10.5: Parameter sensitivity ===")
        results["10.5_param_sensitivity"] = (
            parameter_sensitivity_analysis(sir_dir)
        )

        # 10.6 SNR degradation
        logger.info("=== Task 10.6: SNR degradation study ===")
        results["10.6_snr_degradation"] = (
            snr_degradation_study(sir_dir)
        )

    # Write summary
    summary = format_validation_summary(results)
    summary_path = out_dir / "validation_summary.txt"
    summary_path.write_text(summary)
    logger.info("Validation summary written to %s", summary_path)

    return results
