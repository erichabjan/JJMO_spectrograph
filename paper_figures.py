"""
paper_figures.py -- Publication-Quality Figures
===============================================
Step 10.7 of the JJMO Spectral Flux Calibration Pipeline.

Generates all nine paper figures described in the step 10 specification:
  Fig 1: Raw data overview (all Sirius segments)
  Fig 2: Wavelength calibration (detected lines with IDs)
  Fig 3: Quality masks (segments with masked regions shaded)
  Fig 4: Sensitivity function (per-segment raw ratios + global curve)
  Fig 5: Self-calibration residuals
  Fig 6: Cross-calibration (Sirius-derived applied to Betelgeuse)
  Fig 7: Error budget (uncertainty contributions vs wavelength)
  Fig 8: SNR degradation curve
  Fig 9: Parameter sensitivity

All figures use matplotlib with a publication-quality style. Output formats
are PDF and PNG at 300 DPI.

Authors: JJMO Pipeline
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import AutoMinorLocator

logger = logging.getLogger(__name__)

# Publication style
FIGSIZE_SINGLE = (8, 5)
FIGSIZE_WIDE = (10, 5)
FIGSIZE_TALL = (8, 8)
DPI = 300
FONTSIZE = 11

# Colour palette (colourblind-friendly)
COLORS = [
    "#4477AA",  # blue
    "#EE6677",  # red
    "#228833",  # green
    "#CCBB44",  # yellow
    "#66CCEE",  # cyan
    "#AA3377",  # purple
    "#BBBBBB",  # grey
    "#EE8866",  # orange
]


def _apply_style():
    """Set matplotlib RC parameters for publication-quality plots."""
    plt.rcParams.update({
        "font.size": FONTSIZE,
        "axes.labelsize": FONTSIZE + 1,
        "axes.titlesize": FONTSIZE + 2,
        "xtick.labelsize": FONTSIZE - 1,
        "ytick.labelsize": FONTSIZE - 1,
        "legend.fontsize": FONTSIZE - 1,
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.width": 0.5,
        "ytick.minor.width": 0.5,
        "lines.linewidth": 1.2,
        "axes.grid": False,
        "figure.constrained_layout.use": True,
    })


def _save_fig(fig, path, formats=("pdf", "png")):
    """Save figure in multiple formats."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        out = path.with_suffix(f".{fmt}")
        fig.savefig(out, format=fmt)
        logger.info("Saved figure: %s", out)
    plt.close(fig)


# ============================================================================
# Figure 1: Raw data overview
# ============================================================================

def fig_raw_data_overview(
    segments: List[Tuple[np.ndarray, np.ndarray]],
    star_name: str = "Sirius",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Fig 1: All raw spectral segments plotted together.

    Shows the segmented nature of the data and the noise level.
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)

    for i, (w, f) in enumerate(segments):
        color = COLORS[i % len(COLORS)]
        label = f"{int(w.min() + 0.5)}-{int(w.max() + 0.5)} A"
        ax.plot(w, f, color=color, alpha=0.8, lw=0.8, label=label)

    ax.set_xlabel(r"Wavelength ($\AA$)")
    ax.set_ylabel("Counts")
    ax.set_title(f"{star_name} -- Raw Spectral Segments")
    ax.legend(ncol=2, loc="upper right", fontsize=FONTSIZE - 2)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    if save_path:
        _save_fig(fig, save_path)
    return fig


# ============================================================================
# Figure 2: Wavelength calibration
# ============================================================================

def fig_wavelength_calibration(
    segments: List[Tuple[np.ndarray, np.ndarray]],
    solutions: list,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Fig 2: Detected absorption lines with matched identifications.

    Uses wavelength.WavelengthSolution objects from step 2.
    """
    _apply_style()
    n_seg = len(segments)
    fig, axes = plt.subplots(n_seg, 1, figsize=(10, 2 * n_seg), sharex=False)
    if n_seg == 1:
        axes = [axes]

    for i, ((w, f), sol) in enumerate(zip(segments, solutions)):
        ax = axes[i]
        ax.plot(w, f, color=COLORS[0], lw=0.6, alpha=0.8)

        # Mark detected lines
        if hasattr(sol, "centroids") and sol.centroids is not None:
            for cent in sol.centroids:
                ax.axvline(cent, color=COLORS[1], alpha=0.4, lw=0.5)

        # Mark matched lines with labels
        if hasattr(sol, "matches") and sol.matches is not None:
            for m in sol.matches:
                obs_w = m.get("obs_wave", m.get("observed", None))
                name = m.get("name", m.get("rest_name", ""))
                if obs_w is not None:
                    ax.axvline(obs_w, color=COLORS[2], alpha=0.6, lw=0.8)
                    # Place label at top
                    ymin, ymax = ax.get_ylim()
                    ax.text(obs_w, ymax * 0.95, name,
                            fontsize=6, rotation=90, ha="right", va="top")

        seg_label = f"{int(w.min()+0.5)}-{int(w.max()+0.5)} A"
        ax.set_ylabel("Counts")
        ax.text(0.02, 0.85, seg_label, transform=ax.transAxes,
                fontsize=FONTSIZE - 2, bbox=dict(facecolor="white", alpha=0.7))

    axes[-1].set_xlabel(r"Wavelength ($\AA$)")
    fig.suptitle("Wavelength Calibration: Detected Lines", y=1.01)

    if save_path:
        _save_fig(fig, save_path)
    return fig


# ============================================================================
# Figure 3: Quality masks
# ============================================================================

def fig_quality_masks(
    segments: List[Tuple[np.ndarray, np.ndarray]],
    reports: list,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Fig 3: Segments with masked regions shaded.

    Shows edge masks, telluric, stellar, and cosmic ray masks.
    """
    _apply_style()
    n_seg = len(segments)
    fig, axes = plt.subplots(n_seg, 1, figsize=(10, 2.2 * n_seg), sharex=False)
    if n_seg == 1:
        axes = [axes]

    mask_colors = {
        "mask_edges": ("#CCCCCC", "Edges"),
        "mask_telluric": ("#66CCEE", "Telluric"),
        "mask_stellar": ("#EE6677", "Stellar lines"),
        "mask_cosmic": ("#AA3377", "Cosmics"),
    }

    for i, ((w, f), report) in enumerate(zip(segments, reports)):
        ax = axes[i]
        ax.plot(w, f, color="k", lw=0.5, alpha=0.6)

        # Shade masked regions
        for attr, (color, label) in mask_colors.items():
            mask_arr = getattr(report, attr, None)
            if mask_arr is not None:
                bad = ~mask_arr if attr == "mask_edges" else (
                    ~mask_arr if hasattr(mask_arr, '__len__') else None
                )
                # mask arrays: True = good in quality.py
                if bad is not None and hasattr(bad, '__len__'):
                    ax.fill_between(
                        w, 0, f, where=~getattr(report, attr),
                        color=color, alpha=0.3,
                        label=label if i == 0 else None,
                    )

        seg_label = f"{int(w.min()+0.5)}-{int(w.max()+0.5)} A"
        snr_label = f"SNR={report.snr_median:.0f}" if hasattr(report, 'snr_median') else ""
        ax.text(0.02, 0.85, f"{seg_label}  {snr_label}",
                transform=ax.transAxes, fontsize=FONTSIZE - 2,
                bbox=dict(facecolor="white", alpha=0.7))
        ax.set_ylabel("Counts")

    axes[-1].set_xlabel(r"Wavelength ($\AA$)")
    axes[0].legend(ncol=4, loc="upper right", fontsize=FONTSIZE - 2)
    fig.suptitle("Quality Masks by Segment", y=1.01)

    if save_path:
        _save_fig(fig, save_path)
    return fig


# ============================================================================
# Figure 4: Sensitivity function
# ============================================================================

def fig_sensitivity_function(
    global_sens,
    segments: List[Tuple[np.ndarray, np.ndarray]] = None,
    ref_wave: np.ndarray = None,
    ref_flux: np.ndarray = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Fig 4: Per-segment raw ratios and combined global sensitivity curve.

    Parameters
    ----------
    global_sens : GlobalSensitivity from sensitivity.py
    segments : raw observed segments (for raw ratio overlay)
    ref_wave, ref_flux : reference spectrum for raw ratio computation
    """
    _apply_style()
    fig, axes = plt.subplots(2, 1, figsize=FIGSIZE_TALL)

    ax_top = axes[0]
    ax_bot = axes[1]

    # Top panel: per-segment fits
    for i, fit in enumerate(global_sens.segment_fits):
        color = COLORS[i % len(COLORS)]
        # Evaluate the segment fit on its own domain
        w_grid = np.linspace(fit.wave_min, fit.wave_max, 200)
        s_vals = fit(w_grid)
        valid = np.isfinite(s_vals)
        ax_top.plot(w_grid[valid], s_vals[valid], color=color,
                    lw=1.5, label=fit.segment_id)

    # Global curve
    all_wmin = min(f.wave_min for f in global_sens.segment_fits)
    all_wmax = max(f.wave_max for f in global_sens.segment_fits)
    w_global = np.linspace(all_wmin, all_wmax, 2000)
    s_global = global_sens(w_global)
    valid = np.isfinite(s_global)
    ax_top.plot(w_global[valid], s_global[valid], "k--", lw=2.0,
                label="Global", alpha=0.7)

    ax_top.set_xlabel(r"Wavelength ($\AA$)")
    ax_top.set_ylabel(r"Sensitivity (counts/s per erg/s/cm$^2$/$\AA$)")
    ax_top.set_title("Sensitivity Function: Per-Segment Fits")
    ax_top.legend(ncol=3, fontsize=FONTSIZE - 2)
    ax_top.xaxis.set_minor_locator(AutoMinorLocator())

    # Bottom panel: raw sensitivity ratios (if segments and ref provided)
    if segments is not None and ref_wave is not None and ref_flux is not None:
        from sensitivity import compute_sensitivity_ratio
        for i, (w, f) in enumerate(segments):
            color = COLORS[i % len(COLORS)]
            try:
                wr, ratio, _ = compute_sensitivity_ratio(w, f, ref_wave, ref_flux)
                ax_bot.scatter(wr, ratio, s=1, alpha=0.3, color=color,
                               label=f"{int(w.min()+0.5)} A")
            except Exception:
                pass

        ax_bot.set_xlabel(r"Wavelength ($\AA$)")
        ax_bot.set_ylabel(r"Raw Sensitivity Ratio")
        ax_bot.set_title("Raw Sensitivity Ratios (before fitting)")
        ax_bot.legend(ncol=3, fontsize=FONTSIZE - 2)
        ax_bot.xaxis.set_minor_locator(AutoMinorLocator())
    else:
        ax_bot.text(0.5, 0.5, "Raw ratios not available\n(segments/reference not provided)",
                    transform=ax_bot.transAxes, ha="center", va="center")

    if save_path:
        _save_fig(fig, save_path)
    return fig


# ============================================================================
# Figure 5: Self-calibration residuals
# ============================================================================

def fig_self_calibration_residuals(
    result,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Fig 5: (calibrated - reference) / reference for self-calibration.

    Parameters
    ----------
    result : ValidationResult from self_consistency_sirius()
    """
    _apply_style()
    fig, axes = plt.subplots(2, 1, figsize=FIGSIZE_TALL, height_ratios=[2, 1])

    ax_spec = axes[0]
    ax_res = axes[1]

    w = result.wavelength
    good = ~result.mask

    # Top: calibrated vs reference
    ax_spec.plot(w[good], result.flux_calibrated[good],
                 color=COLORS[0], lw=0.8, label="Calibrated", alpha=0.8)
    ax_spec.plot(w[good], result.flux_reference[good],
                 color=COLORS[1], lw=0.8, label="Reference", alpha=0.8)
    ax_spec.set_xlabel(r"Wavelength ($\AA$)")
    ax_spec.set_ylabel(r"Flux (erg/s/cm$^2$/$\AA$)")
    ax_spec.set_title(f"Sirius Self-Calibration (RMS={result.rms_residual:.4f})")
    ax_spec.legend()
    ax_spec.xaxis.set_minor_locator(AutoMinorLocator())

    # Bottom: fractional residuals
    ax_res.axhline(0, color="k", lw=0.5, ls="--")
    ax_res.scatter(w[good], result.residual_frac[good], s=1,
                   color=COLORS[0], alpha=0.5)

    # Running median for trend
    if np.sum(good) > 50:
        from scipy.ndimage import median_filter
        sorted_idx = np.argsort(w[good])
        w_sorted = w[good][sorted_idx]
        r_sorted = result.residual_frac[good][sorted_idx]
        window = max(51, len(r_sorted) // 20)
        if window % 2 == 0:
            window += 1
        r_smooth = median_filter(r_sorted, size=window)
        ax_res.plot(w_sorted, r_smooth, color=COLORS[1], lw=1.5,
                    label="Running median")
        ax_res.legend(fontsize=FONTSIZE - 2)

    ax_res.set_xlabel(r"Wavelength ($\AA$)")
    ax_res.set_ylabel("Fractional Residual")
    ax_res.set_ylim(-0.5, 0.5)
    ax_res.xaxis.set_minor_locator(AutoMinorLocator())

    if save_path:
        _save_fig(fig, save_path)
    return fig


# ============================================================================
# Figure 6: Cross-calibration
# ============================================================================

def fig_cross_calibration(
    result_sir_to_bet,
    result_bet_to_sir=None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Fig 6: Cross-calibration results.

    Shows Sirius-derived calibration applied to Betelgeuse (and optionally
    the reverse) compared to their respective references.
    """
    _apply_style()
    n_panels = 2 if result_bet_to_sir is not None else 1
    fig, axes = plt.subplots(n_panels, 1, figsize=(10, 5 * n_panels))
    if n_panels == 1:
        axes = [axes]

    for ax, result, title in zip(
        axes,
        [result_sir_to_bet, result_bet_to_sir] if result_bet_to_sir else [result_sir_to_bet],
        ["Sirius Sensitivity -> Betelgeuse",
         "Betelgeuse Sensitivity -> Sirius"] if result_bet_to_sir else
        ["Sirius Sensitivity -> Betelgeuse"],
    ):
        good = ~result.mask
        ax.plot(result.wavelength[good], result.flux_calibrated[good],
                color=COLORS[0], lw=0.8, label="Calibrated", alpha=0.8)
        ax.plot(result.wavelength[good], result.flux_reference[good],
                color=COLORS[1], lw=0.8, label="Reference", alpha=0.8)
        ax.set_xlabel(r"Wavelength ($\AA$)")
        ax.set_ylabel(r"Flux (erg/s/cm$^2$/$\AA$)")
        ax.set_title(f"{title}  (RMS={result.rms_residual:.4f})")
        ax.legend()
        ax.xaxis.set_minor_locator(AutoMinorLocator())

    if save_path:
        _save_fig(fig, save_path)
    return fig


# ============================================================================
# Figure 7: Error budget
# ============================================================================

def fig_error_budget(
    result,
    sensfunc=None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Fig 7: Stacked uncertainty contributions vs wavelength.

    Decomposes the total uncertainty into photon noise, sensitivity
    function uncertainty, and calibration residual contributions.

    Parameters
    ----------
    result : ValidationResult from self-consistency test
    sensfunc : SensitivityFunction (for sensitivity uncertainty)
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)

    w = result.wavelength
    good = ~result.mask

    # Residual-based uncertainty estimate
    residual_unc = np.abs(result.residual_frac)
    ax.fill_between(w[good], 0, np.abs(residual_unc[good]),
                    color=COLORS[0], alpha=0.5, label="Calibration residual")

    # Sensitivity function uncertainty (if available)
    if sensfunc is not None and sensfunc.uncertainty is not None:
        sens_val, sens_unc = sensfunc.evaluate(w[good])
        frac_sens_unc = np.where(
            (sens_val > 0) & np.isfinite(sens_unc),
            sens_unc / sens_val,
            0.0,
        )
        ax.fill_between(
            w[good], np.abs(residual_unc[good]),
            np.abs(residual_unc[good]) + frac_sens_unc,
            color=COLORS[1], alpha=0.5, label="Sensitivity function unc.",
        )

    ax.set_xlabel(r"Wavelength ($\AA$)")
    ax.set_ylabel("Fractional Uncertainty")
    ax.set_title("Error Budget")
    ax.legend()
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.set_ylim(0, None)

    if save_path:
        _save_fig(fig, save_path)
    return fig


# ============================================================================
# Figure 8: SNR degradation curve
# ============================================================================

def fig_snr_degradation(
    snr_result,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Fig 8: Calibration accuracy vs input SNR.

    Parameters
    ----------
    snr_result : SNRDegradationResult
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)

    snr = np.array(snr_result.snr_levels)
    rms = np.array(snr_result.rms_residuals)
    med = np.array(snr_result.median_residuals)

    valid = np.isfinite(snr) & np.isfinite(rms) & (snr > 0)

    ax.semilogy(snr[valid], rms[valid], "o-", color=COLORS[0],
                label="RMS residual", markersize=6)
    ax.semilogy(snr[valid], med[valid], "s--", color=COLORS[1],
                label="Median |residual|", markersize=5)

    # 10% threshold line
    ax.axhline(0.10, color="grey", ls=":", lw=1.0, label="10% threshold")

    # Mark threshold SNR
    if np.isfinite(snr_result.threshold_snr):
        ax.axvline(snr_result.threshold_snr, color=COLORS[2], ls="--", lw=1.0,
                   label=f"Min SNR = {snr_result.threshold_snr:.0f}")

    ax.set_xlabel("Estimated Input SNR")
    ax.set_ylabel("Fractional Residual")
    ax.set_title("Calibration Accuracy vs. Input SNR")
    ax.legend()
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.invert_xaxis()  # high SNR on left

    if save_path:
        _save_fig(fig, save_path)
    return fig


# ============================================================================
# Figure 9: Parameter sensitivity
# ============================================================================

def fig_parameter_sensitivity(
    param_results: Dict[str, "ParameterSensitivityResult"],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Fig 9: Effect of pipeline parameters on calibration residuals.

    Shows one subplot per parameter, with RMS residual vs parameter value.
    """
    _apply_style()
    n_params = len(param_results)
    if n_params == 0:
        fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
        ax.text(0.5, 0.5, "No parameter results available",
                transform=ax.transAxes, ha="center", va="center")
        if save_path:
            _save_fig(fig, save_path)
        return fig

    fig, axes = plt.subplots(1, n_params, figsize=(4 * n_params, 4))
    if n_params == 1:
        axes = [axes]

    for ax, (name, res) in zip(axes, param_results.items()):
        vals = np.array(res.parameter_values, dtype=float)
        rms = np.array(res.rms_residuals, dtype=float)
        valid = np.isfinite(rms)

        ax.plot(vals[valid], rms[valid], "o-", color=COLORS[0], markersize=6)
        ax.axhline(res.baseline_rms, color="grey", ls=":", lw=1.0,
                   label=f"Baseline={res.baseline_rms:.4f}")
        ax.set_xlabel(name.replace("_", " ").title())
        ax.set_ylabel("RMS Residual")
        ax.legend(fontsize=FONTSIZE - 2)
        ax.xaxis.set_minor_locator(AutoMinorLocator())

    fig.suptitle("Parameter Sensitivity Analysis")

    if save_path:
        _save_fig(fig, save_path)
    return fig


# ============================================================================
# Master figure generator
# ============================================================================

def generate_all_figures(
    validation_results: Dict[str, Any],
    segments_sirius: List[Tuple[np.ndarray, np.ndarray]] = None,
    segments_betelgeuse: List[Tuple[np.ndarray, np.ndarray]] = None,
    global_sens=None,
    sensfunc=None,
    quality_reports: list = None,
    wavelength_solutions: list = None,
    ref_wave: np.ndarray = None,
    ref_flux: np.ndarray = None,
    output_dir: Optional[str] = None,
) -> Dict[str, plt.Figure]:
    """Generate all nine paper figures.

    Parameters
    ----------
    validation_results : dict from run_all_validations()
    segments_sirius, segments_betelgeuse : raw data
    global_sens : GlobalSensitivity from sensitivity derivation
    sensfunc : SensitivityFunction
    quality_reports : QualityReport list from assess_segments
    wavelength_solutions : WavelengthSolution list from calibrate_segments
    ref_wave, ref_flux : reference spectrum arrays
    output_dir : directory for saved figures

    Returns
    -------
    dict of figure name -> matplotlib Figure
    """
    out = Path(output_dir) if output_dir else Path("validation_outputs/figures")
    out.mkdir(parents=True, exist_ok=True)

    figures = {}

    # Fig 1: Raw data overview
    if segments_sirius is not None:
        figures["fig1_raw_data"] = fig_raw_data_overview(
            segments_sirius, "Sirius", save_path=str(out / "fig1_raw_data"),
        )
        logger.info("Generated Fig 1: Raw data overview")

    # Fig 2: Wavelength calibration
    if segments_sirius is not None and wavelength_solutions is not None:
        figures["fig2_wavelength_cal"] = fig_wavelength_calibration(
            segments_sirius, wavelength_solutions,
            save_path=str(out / "fig2_wavelength_cal"),
        )
        logger.info("Generated Fig 2: Wavelength calibration")

    # Fig 3: Quality masks
    if segments_sirius is not None and quality_reports is not None:
        figures["fig3_quality_masks"] = fig_quality_masks(
            segments_sirius, quality_reports,
            save_path=str(out / "fig3_quality_masks"),
        )
        logger.info("Generated Fig 3: Quality masks")

    # Fig 4: Sensitivity function
    if global_sens is not None:
        figures["fig4_sensitivity"] = fig_sensitivity_function(
            global_sens, segments_sirius, ref_wave, ref_flux,
            save_path=str(out / "fig4_sensitivity"),
        )
        logger.info("Generated Fig 4: Sensitivity function")

    # Fig 5: Self-calibration residuals
    self_cal = validation_results.get("10.1_self_consistency")
    if self_cal is not None:
        figures["fig5_self_cal_residuals"] = fig_self_calibration_residuals(
            self_cal, save_path=str(out / "fig5_self_cal_residuals"),
        )
        logger.info("Generated Fig 5: Self-calibration residuals")

    # Fig 6: Cross-calibration
    sir_to_bet = validation_results.get("10.2_sirius_to_betelgeuse")
    bet_to_sir = validation_results.get("10.3_betelgeuse_to_sirius")
    if sir_to_bet is not None:
        figures["fig6_cross_calibration"] = fig_cross_calibration(
            sir_to_bet, bet_to_sir,
            save_path=str(out / "fig6_cross_calibration"),
        )
        logger.info("Generated Fig 6: Cross-calibration")

    # Fig 7: Error budget
    if self_cal is not None:
        figures["fig7_error_budget"] = fig_error_budget(
            self_cal, sensfunc,
            save_path=str(out / "fig7_error_budget"),
        )
        logger.info("Generated Fig 7: Error budget")

    # Fig 8: SNR degradation
    snr_res = validation_results.get("10.6_snr_degradation")
    if snr_res is not None:
        figures["fig8_snr_degradation"] = fig_snr_degradation(
            snr_res, save_path=str(out / "fig8_snr_degradation"),
        )
        logger.info("Generated Fig 8: SNR degradation")

    # Fig 9: Parameter sensitivity
    param_res = validation_results.get("10.5_param_sensitivity")
    if param_res is not None:
        figures["fig9_param_sensitivity"] = fig_parameter_sensitivity(
            param_res, save_path=str(out / "fig9_param_sensitivity"),
        )
        logger.info("Generated Fig 9: Parameter sensitivity")

    logger.info("Generated %d/%d figures", len(figures), 9)
    return figures
