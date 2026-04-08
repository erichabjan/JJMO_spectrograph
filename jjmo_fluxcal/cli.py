"""
cli.py — Command-Line Interface
================================

Provides the ``jjmo-fluxcal`` console entry point with three sub-commands:

  jjmo-fluxcal run       — full end-to-end pipeline
  jjmo-fluxcal sensfunc  — derive and save sensitivity function only
  jjmo-fluxcal apply     — apply a saved sensitivity function to spectra
"""

import argparse
import logging
import sys

logger = logging.getLogger(__name__)


def _build_parser():
    """Build the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="jjmo-fluxcal",
        description=(
            "JJMO Flux Calibration Pipeline — flux-calibrate noisy, "
            "segmented spectra from small observatory spectrographs."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {_get_version()}",
    )
    sub = parser.add_subparsers(dest="command", help="Pipeline sub-commands")

    # ---- run ----
    p_run = sub.add_parser(
        "run",
        help="Run the full flux calibration pipeline",
    )
    p_run.add_argument(
        "--input-dir", "-i",
        required=True,
        help="Directory containing raw spectral data files",
    )
    p_run.add_argument(
        "--star", "-s",
        default="sirius",
        help="Standard star name (default: sirius)",
    )
    p_run.add_argument(
        "--output-dir", "-o",
        default=None,
        help="Output directory for calibrated spectra and diagnostics",
    )
    p_run.add_argument(
        "--config", "-c",
        default=None,
        help="Path to YAML configuration file",
    )
    p_run.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO)",
    )
    _add_common_options(p_run)

    # ---- sensfunc ----
    p_sens = sub.add_parser(
        "sensfunc",
        help="Derive sensitivity function from standard star data",
    )
    p_sens.add_argument(
        "--input-dir", "-i",
        required=True,
        help="Directory containing standard star spectral data",
    )
    p_sens.add_argument(
        "--star", "-s",
        default="sirius",
        help="Standard star name (default: sirius)",
    )
    p_sens.add_argument(
        "--output", "-o",
        required=True,
        help="Output path for sensitivity function (JSON)",
    )
    p_sens.add_argument(
        "--config", "-c",
        default=None,
        help="Path to YAML configuration file",
    )
    p_sens.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO)",
    )
    _add_common_options(p_sens)

    # ---- apply ----
    p_apply = sub.add_parser(
        "apply",
        help="Apply a saved sensitivity function to science spectra",
    )
    p_apply.add_argument(
        "--sensfunc",
        required=True,
        help="Path to sensitivity function file (JSON)",
    )
    p_apply.add_argument(
        "--input", "-i",
        required=True,
        help="Input spectrum (FITS or CSV)",
    )
    p_apply.add_argument(
        "--output", "-o",
        required=True,
        help="Output path for calibrated spectrum",
    )
    p_apply.add_argument(
        "--exptime",
        type=float,
        default=None,
        help="Exposure time in seconds (if not in FITS header)",
    )
    p_apply.add_argument(
        "--airmass",
        type=float,
        default=None,
        help="Observation airmass (if not in FITS header)",
    )
    p_apply.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO)",
    )

    return parser


def _add_common_options(parser):
    """Add options shared between 'run' and 'sensfunc' sub-commands."""
    parser.add_argument(
        "--fit-method",
        default=None,
        choices=["chebyshev", "legendre", "spline", "savgol"],
        help="Sensitivity fit method",
    )
    parser.add_argument(
        "--fit-order",
        type=int,
        default=None,
        help="Polynomial/spline fit order",
    )
    parser.add_argument(
        "--sigma-clip",
        type=float,
        default=None,
        help="Sigma-clipping threshold for fitting",
    )
    parser.add_argument(
        "--output-format",
        default=None,
        choices=["fits", "csv"],
        help="Output file format (default: fits)",
    )


def _get_version():
    """Return package version string."""
    try:
        from jjmo_fluxcal import __version__
        return __version__
    except ImportError:
        return "unknown"


def cmd_run(args):
    """Execute the full pipeline."""
    from jjmo_fluxcal import fluxcal, setup_logging
    from jjmo_fluxcal.config import PipelineConfig

    setup_logging(args.log_level)

    overrides = {}
    for key in ("fit_method", "fit_order", "sigma_clip", "output_format"):
        val = getattr(args, key.replace("-", "_"), None)
        if val is not None:
            overrides[key] = val

    result = fluxcal(
        args.input_dir,
        star_name=args.star,
        output_dir=args.output_dir,
        config_file=args.config,
        **overrides,
    )

    cal = result["calibrated"]
    logger.info(
        "Done. Calibrated spectrum covers %.1f - %.1f A",
        cal.wavelength[0], cal.wavelength[-1],
    )


def cmd_sensfunc(args):
    """Derive and save the sensitivity function."""
    from jjmo_fluxcal import setup_logging
    from jjmo_fluxcal.config import PipelineConfig
    from jjmo_fluxcal import (
        io as jio,
        wavelength as jwav,
        quality as jqual,
        stitching as jstitch,
        reference as jref,
        sensitivity as jsens,
    )
    import numpy as np
    import astropy.units as u

    setup_logging(args.log_level)

    # Build config
    cfg = PipelineConfig()
    if args.config:
        cfg = PipelineConfig.from_yaml(args.config)
    cfg.star_name = args.star
    for key in ("fit_method", "fit_order", "sigma_clip"):
        val = getattr(args, key.replace("-", "_"), None)
        if val is not None:
            setattr(cfg, key, val)

    # Steps 1-3: ingest, calibrate wavelengths, assess quality
    segments = jio.read_directory(args.input_dir)
    wavelengths = [s.spectral_axis.to(u.AA).value for s in segments]
    fluxes = [s.flux.value for s in segments]
    seg_ids = [s.meta.get("segment_id", f"seg_{i:02d}")
               for i, s in enumerate(segments)]

    solutions = jwav.calibrate_segments(wavelengths, fluxes,
                                        segment_ids=seg_ids)
    corrected_w = [sol.wavelength_corrected for sol in solutions]

    # Step 5: reference spectrum
    ref_spec = jref.load_reference_spectrum(cfg.star_name,
                                            prefer=cfg.prefer_reference)

    # Step 6: derive sensitivity
    try:
        from specutils import Spectrum as Spectrum1D
    except ImportError:
        from specutils import Spectrum1D

    corrected_segments = []
    for i, (w, f) in enumerate(zip(corrected_w, fluxes)):
        sp = Spectrum1D(
            spectral_axis=w * u.AA,
            flux=f * u.ct,
            meta={"segment_id": seg_ids[i]},
        )
        corrected_segments.append(sp)

    sens_fit = jsens.derive_sensitivity(
        corrected_segments, ref_spec,
        method=cfg.fit_method, order=cfg.fit_order,
        sigma_clip_threshold=cfg.sigma_clip,
    )

    if isinstance(sens_fit, dict):
        result = sens_fit.get("global", sens_fit.get("stitched"))
    else:
        result = sens_fit

    jsens.save_sensitivity(result, args.output)
    logger.info("Sensitivity function saved to %s", args.output)


def cmd_apply(args):
    """Apply a saved sensitivity function to a spectrum."""
    from jjmo_fluxcal import setup_logging
    from jjmo_fluxcal import io as jio, sensitivity as jsens, calibrate as jcal

    setup_logging(args.log_level)

    # Load sensitivity function
    sens = jsens.load_sensitivity(args.sensfunc)

    # Read input spectrum
    spectrum = jio.read_spectrum(args.input)

    # Apply calibration
    result = jcal.apply_sensitivity(
        spectrum,
        sens,
        exptime=args.exptime,
        airmass_obs=args.airmass,
    )

    # Write output
    if args.output.endswith(".csv"):
        jcal.write_calibrated_csv(result, args.output)
    else:
        jcal.write_calibrated_fits(result, args.output)

    logger.info("Calibrated spectrum written to %s", args.output)


def main(argv=None):
    """Main entry point for the jjmo-fluxcal CLI."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    commands = {
        "run": cmd_run,
        "sensfunc": cmd_sensfunc,
        "apply": cmd_apply,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
