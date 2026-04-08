"""
plotting.py — Consolidated Diagnostic Plotting
===============================================

Re-exports diagnostic plot functions from individual pipeline modules
and provides additional high-level convenience plots for the full
pipeline.  This module is optional — it is not imported unless
explicitly requested.

Usage
-----
    from jjmo_fluxcal.plotting import plot_pipeline_summary
    plot_pipeline_summary(result)  # result from fluxcal()

Individual module plots are also available directly:

    from jjmo_fluxcal.wavelength import plot_all_segments
    from jjmo_fluxcal.quality import plot_quality_overview
    from jjmo_fluxcal.stitching import plot_stitched_spectrum
    from jjmo_fluxcal.sensitivity import plot_sensitivity_diagnostic
    from jjmo_fluxcal.calibrate import plot_calibrated_spectrum
"""

import logging
import warnings

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Re-exports from individual modules (lazy to avoid import overhead)
# ---------------------------------------------------------------------------

def __getattr__(name):
    """Lazy re-export of plot functions from pipeline modules."""
    _plot_functions = {
        # wavelength.py
        "plot_segment_diagnostics": "jjmo_fluxcal.wavelength",
        "plot_all_segments": "jjmo_fluxcal.wavelength",
        "print_calibration_table": "jjmo_fluxcal.wavelength",
        # quality.py
        "plot_segment_quality": "jjmo_fluxcal.quality",
        "plot_quality_overview": "jjmo_fluxcal.quality",
        "print_quality_table": "jjmo_fluxcal.quality",
        # stitching.py
        "plot_stitched_spectrum": "jjmo_fluxcal.stitching",
        "plot_normalization_factors": "jjmo_fluxcal.stitching",
        "plot_overlap_quality": "jjmo_fluxcal.stitching",
        # reference.py
        "plot_reference_comparison": "jjmo_fluxcal.reference",
        # sensitivity.py
        "plot_sensitivity_ratio": "jjmo_fluxcal.sensitivity",
        "plot_sensitivity_residuals": "jjmo_fluxcal.sensitivity",
        "plot_global_sensitivity": "jjmo_fluxcal.sensitivity",
        "plot_sensitivity_diagnostic": "jjmo_fluxcal.sensitivity",
        "plot_method_comparison": "jjmo_fluxcal.sensitivity",
        # calibrate.py
        "plot_calibrated_spectrum": "jjmo_fluxcal.calibrate",
        "plot_self_calibration": "jjmo_fluxcal.calibrate",
    }
    if name in _plot_functions:
        import importlib
        mod = importlib.import_module(_plot_functions[name])
        return getattr(mod, name)
    raise AttributeError(f"module 'jjmo_fluxcal.plotting' has no attribute {name!r}")


# ---------------------------------------------------------------------------
# High-level pipeline summary plot
# ---------------------------------------------------------------------------

def plot_pipeline_summary(result, *, save_path=None, show=True):
    """Generate a multi-panel summary of the full pipeline run.

    Parameters
    ----------
    result : dict
        Output from ``jjmo_fluxcal.fluxcal()``.
    save_path : str or Path, optional
        If provided, save the figure to this path.
    show : bool
        Whether to call ``plt.show()``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for plotting")

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    cal = result["calibrated"]
    stitch = result["stitched"]
    config = result["config"]

    # Panel 1: Raw stitched spectrum
    ax1 = axes[0]
    ax1.plot(stitch.wavelength, stitch.flux, "k-", lw=0.5, alpha=0.8)
    ax1.set_ylabel("Counts")
    ax1.set_title(f"JJMO Pipeline: {config.star_name}")

    # Shade masked regions
    if stitch.mask is not None:
        bad = ~stitch.mask
        if np.any(bad):
            for start, end in _contiguous_regions(stitch.wavelength, bad):
                ax1.axvspan(start, end, color="red", alpha=0.1)

    # Panel 2: Sensitivity function
    ax2 = axes[1]
    sens = result["sensitivity"]
    if hasattr(sens, "_wavelength_data") and sens._wavelength_data is not None:
        w = sens._wavelength_data
        s_vals = sens(w)
        ax2.plot(w, s_vals, "b-", lw=1.0)
    elif hasattr(sens, "wavelength"):
        ax2.plot(sens.wavelength, sens.sensitivity, "b-", lw=1.0)
    ax2.set_ylabel("Sensitivity")
    ax2.set_yscale("log")

    # Panel 3: Calibrated spectrum
    ax3 = axes[2]
    ax3.plot(cal.wavelength, cal.flux, "k-", lw=0.5)
    if cal.uncertainty is not None:
        ax3.fill_between(
            cal.wavelength,
            cal.flux - cal.uncertainty,
            cal.flux + cal.uncertainty,
            alpha=0.2, color="gray",
        )
    ax3.set_ylabel(r"Flux (erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$)")
    ax3.set_xlabel(r"Wavelength ($\AA$)")

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        logger.info("Pipeline summary saved to %s", save_path)

    if show:
        plt.show()

    return fig


def _contiguous_regions(wavelength, mask):
    """Yield (start_wave, end_wave) for contiguous True regions in mask."""
    if not np.any(mask):
        return
    diff = np.diff(mask.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1
    if mask[0]:
        starts = np.concatenate([[0], starts])
    if mask[-1]:
        ends = np.concatenate([ends, [len(mask)]])
    for s, e in zip(starts, ends):
        yield wavelength[s], wavelength[min(e, len(wavelength) - 1)]
