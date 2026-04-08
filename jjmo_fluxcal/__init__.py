"""
jjmo_fluxcal — Flux Calibration for Small Observatory Spectrographs
====================================================================

A lightweight Python package for flux-calibrating noisy, segmented spectra
from educational observatory spectrographs.  Primary target is the JJMO
(John J. McCarthy Observatory) spectrograph, but the tools generalise to
any instrument that produces ~500 A segments across the optical range.

Quick start
-----------
    from jjmo_fluxcal import fluxcal

    result = fluxcal("./data/Sirius", star_name="sirius",
                     output_dir="./results")

Step-by-step API
----------------
Each pipeline module is available as a sub-import:

    from jjmo_fluxcal import io, wavelength, quality, stitching
    from jjmo_fluxcal import reference, sensitivity, calibrate
    from jjmo_fluxcal import uncertainties, config

See individual module docstrings for function-level documentation.
"""

__version__ = "0.1.0"

from jjmo_fluxcal.config import PipelineConfig
from jjmo_fluxcal._logging import setup_logging, is_configured

# Lazy imports for step modules — avoids heavyweight imports at package load
# Users access them as  jjmo_fluxcal.io, jjmo_fluxcal.wavelength, etc.

__all__ = [
    "__version__",
    "PipelineConfig",
    "setup_logging",
    "fluxcal",
    # Sub-modules (importable but not eagerly loaded)
    "io",
    "wavelength",
    "quality",
    "stitching",
    "reference",
    "sensitivity",
    "calibrate",
    "uncertainties",
    "config",
]


def fluxcal(
    input_dir,
    star_name="sirius",
    output_dir=None,
    *,
    config=None,
    config_file=None,
    **overrides,
):
    """Run the full flux calibration pipeline with sensible defaults.

    This is the primary one-call entry point.  It executes Steps 1-7
    in sequence, returning the calibrated spectrum and writing outputs
    to ``output_dir`` if provided.

    Parameters
    ----------
    input_dir : str or Path
        Directory containing raw spectral data files (.fit/.txt, .csv,
        or 1D FITS).
    star_name : str
        Target star name (looked up in the standard star database).
    output_dir : str or Path, optional
        If provided, write calibrated spectra and diagnostics here.
    config : PipelineConfig, optional
        Full configuration object.  If None, one is built from defaults
        plus any ``overrides``.
    config_file : str or Path, optional
        Path to a YAML configuration file.  Applied before ``overrides``.
    **overrides
        Keyword arguments that override individual config parameters
        (e.g., ``fit_order=5``, ``sigma_clip=2.5``).

    Returns
    -------
    dict
        Keys:
        - ``calibrated`` : CalibrationResult — the flux-calibrated spectrum
        - ``sensitivity`` : SensitivityFit — the derived sensitivity function
        - ``stitched`` : StitchResult — the stitched observed spectrum
        - ``quality_reports`` : list[QualityReport] — per-segment quality
        - ``wavelength_solutions`` : list[WavelengthSolution] — per-segment
        - ``config`` : PipelineConfig — the configuration that was used
    """
    import logging
    from pathlib import Path

    from jjmo_fluxcal import (
        io as jio,
        wavelength as jwav,
        quality as jqual,
        stitching as jstitch,
        reference as jref,
        sensitivity as jsens,
        calibrate as jcal,
    )
    from jjmo_fluxcal._logging import setup_logging, is_configured
    from jjmo_fluxcal.config import PipelineConfig

    # --- Build configuration ---
    if config is None:
        if config_file is not None:
            config = PipelineConfig.from_yaml(str(config_file))
        else:
            config = PipelineConfig()

    # Apply explicit overrides
    config.star_name = star_name
    config.input_dir = str(input_dir)
    if output_dir is not None:
        config.output_dir = str(output_dir)
    for k, v in overrides.items():
        if hasattr(config, k):
            setattr(config, k, v)

    # --- Set up logging ---
    if not is_configured():
        setup_logging(config.log_level)
    logger = logging.getLogger(__name__)
    logger.info("Starting JJMO flux calibration pipeline")
    logger.info("  Star: %s", config.star_name)
    logger.info("  Input: %s", config.input_dir)

    # --- Step 1: Data ingestion ---
    logger.info("Step 1: Reading spectral data")
    segments = jio.read_directory(config.input_dir, format=config.file_format)
    logger.info("  Loaded %d segments", len(segments))

    if len(segments) == 0:
        raise ValueError(f"No spectra found in {config.input_dir}")

    # Extract numpy arrays from Spectrum1D objects for subsequent steps
    import numpy as np
    import astropy.units as u

    wavelengths = []
    fluxes = []
    segment_ids = []
    for i, seg in enumerate(segments):
        w = seg.spectral_axis.to(u.AA).value
        f = seg.flux.value
        wavelengths.append(w)
        fluxes.append(f)
        seg_id = seg.meta.get("segment_id", f"seg_{i:02d}")
        segment_ids.append(seg_id)

    # --- Step 2: Wavelength calibration ---
    logger.info("Step 2: Wavelength calibration")
    solutions = jwav.calibrate_segments(
        wavelengths,
        fluxes,
        segment_ids=segment_ids,
        min_depth_sigma=config.min_depth_sigma,
        tolerance=config.line_match_tolerance,
    )
    # Apply corrections
    corrected_wavelengths = []
    for sol in solutions:
        corrected_wavelengths.append(sol.wavelength_corrected)
    logger.info("  Calibrated %d segments", len(solutions))

    # --- Step 3: Quality assessment ---
    logger.info("Step 3: Quality assessment")
    quality_reports = jqual.assess_segments(
        corrected_wavelengths,
        fluxes,
        segment_ids=segment_ids,
        edge_threshold_frac=config.edge_trim_frac,
        cosmic_sigma=config.cosmic_sigma,
    )
    logger.info("  Assessed %d segments", len(quality_reports))

    # Build Spectrum1D list with corrected wavelengths and quality masks
    from astropy.nddata import StdDevUncertainty
    try:
        from specutils import Spectrum as Spectrum1D
    except ImportError:
        from specutils import Spectrum1D

    corrected_segments = []
    for i, (w, f, qr) in enumerate(
        zip(corrected_wavelengths, fluxes, quality_reports)
    ):
        mask = qr.mask_good
        meta = segments[i].meta.copy() if hasattr(segments[i], "meta") else {}
        meta["segment_id"] = segment_ids[i]
        sp = Spectrum1D(
            spectral_axis=w * u.AA,
            flux=f * u.ct,
            mask=~mask,  # specutils convention: True = bad
            meta=meta,
        )
        corrected_segments.append(sp)

    # --- Step 4: Stitching ---
    logger.info("Step 4: Stitching segments")
    stitch_result = jstitch.stitch_segments(
        corrected_segments,
        normalize=config.normalize_segments,
        norm_method=config.norm_method,
    )
    logger.info(
        "  Stitched spectrum: %.1f - %.1f A (%d pixels)",
        stitch_result.wavelength[0],
        stitch_result.wavelength[-1],
        len(stitch_result.wavelength),
    )

    # --- Step 5: Load reference spectrum ---
    logger.info("Step 5: Loading reference spectrum for %s", config.star_name)
    ref_spec = jref.load_reference_spectrum(
        config.star_name,
        prefer=config.prefer_reference,
    )
    # Prepare reference: resample, convolve, mask features
    ref_wave, ref_flux, ref_mask = jref.prepare_reference(
        ref_spec,
        stitch_result.wavelength,
        ebv=config.ebv,
        rv=config.rv,
    )
    logger.info("  Reference spectrum prepared")

    # --- Step 6: Derive sensitivity function ---
    logger.info("Step 6: Deriving sensitivity function")
    sens_fit = jsens.derive_sensitivity(
        corrected_segments,
        ref_spec,
        method=config.fit_method,
        order=config.fit_order,
        sigma_clip_threshold=config.sigma_clip,
        n_iterations=config.n_clip_iterations,
    )
    # Extract the fit (derive_sensitivity may return dict or SensitivityFit)
    if isinstance(sens_fit, dict):
        sensitivity_result = sens_fit.get("global", sens_fit.get("stitched"))
    else:
        sensitivity_result = sens_fit
    logger.info("  Sensitivity function derived")

    # --- Step 7: Apply calibration ---
    logger.info("Step 7: Applying flux calibration")
    cal_result = jcal.apply_sensitivity(
        stitch_result,
        sensitivity_result,
        exptime=config.exptime,
        airmass_obs=config.airmass,
    )
    logger.info("  Calibration complete")

    # --- Write outputs ---
    if config.output_dir is not None:
        out = Path(config.output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # Save calibrated spectrum
        if config.output_format == "fits":
            outfile = out / f"{config.star_name}_calibrated.fits"
            jcal.write_calibrated_fits(cal_result, str(outfile))
        else:
            outfile = out / f"{config.star_name}_calibrated.csv"
            jcal.write_calibrated_csv(cal_result, str(outfile))
        logger.info("  Wrote calibrated spectrum to %s", outfile)

        # Save sensitivity function
        sens_file = out / f"{config.star_name}_sensitivity.json"
        jsens.save_sensitivity(sensitivity_result, str(sens_file))
        logger.info("  Wrote sensitivity function to %s", sens_file)

        # Save config
        cfg_file = out / "pipeline_config.yaml"
        config.save_yaml(str(cfg_file))

    logger.info("Pipeline complete")

    return {
        "calibrated": cal_result,
        "sensitivity": sensitivity_result,
        "stitched": stitch_result,
        "quality_reports": quality_reports,
        "wavelength_solutions": solutions,
        "config": config,
    }
