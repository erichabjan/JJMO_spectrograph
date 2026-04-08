"""
test_package.py — Tests for the package-level API, config, CLI, and logging.

Verifies that the jjmo_fluxcal package is correctly structured, importable,
and that the new components (config, CLI, logging, plotting) work as expected.
"""

import os
import tempfile

import numpy as np
import pytest


# ===================================================================
# Package import and version
# ===================================================================

class TestPackageImport:
    def test_version(self):
        import jjmo_fluxcal
        assert jjmo_fluxcal.__version__ == "0.1.0"

    def test_all_submodules_importable(self):
        from jjmo_fluxcal import io
        from jjmo_fluxcal import wavelength
        from jjmo_fluxcal import quality
        from jjmo_fluxcal import stitching
        from jjmo_fluxcal import reference
        from jjmo_fluxcal import sensitivity
        from jjmo_fluxcal import calibrate
        from jjmo_fluxcal import uncertainties
        from jjmo_fluxcal import config
        from jjmo_fluxcal import plotting
        from jjmo_fluxcal import cli

    def test_fluxcal_callable(self):
        from jjmo_fluxcal import fluxcal
        assert callable(fluxcal)


# ===================================================================
# Config
# ===================================================================

class TestConfig:
    def test_default_creation(self):
        from jjmo_fluxcal.config import PipelineConfig
        cfg = PipelineConfig()
        assert cfg.star_name == "sirius"
        assert cfg.fit_order == 4
        assert cfg.sigma_clip == 3.0
        assert cfg.fit_method == "chebyshev"

    def test_override(self):
        from jjmo_fluxcal.config import PipelineConfig
        cfg = PipelineConfig(star_name="betelgeuse", fit_order=3)
        assert cfg.star_name == "betelgeuse"
        assert cfg.fit_order == 3

    def test_yaml_roundtrip(self):
        from jjmo_fluxcal.config import PipelineConfig
        cfg = PipelineConfig(
            star_name="vega",
            fit_order=5,
            ebv=0.01,
            fit_method="legendre",
        )
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            path = f.name
        try:
            cfg.save_yaml(path)
            cfg2 = PipelineConfig.from_yaml(path)
            assert cfg2.star_name == "vega"
            assert cfg2.fit_order == 5
            assert cfg2.ebv == 0.01
            assert cfg2.fit_method == "legendre"
        finally:
            os.unlink(path)

    def test_from_dict_ignores_unknown(self):
        from jjmo_fluxcal.config import PipelineConfig
        cfg = PipelineConfig.from_dict({"star_name": "vega", "bogus": 42})
        assert cfg.star_name == "vega"

    def test_to_dict(self):
        from jjmo_fluxcal.config import PipelineConfig
        cfg = PipelineConfig(star_name="sirius")
        d = cfg.to_dict()
        assert d["star_name"] == "sirius"
        assert isinstance(d, dict)


# ===================================================================
# Config data
# ===================================================================

class TestConfigData:
    def test_balmer_lines(self):
        from jjmo_fluxcal.config import BALMER_LINES
        assert "H-alpha" in BALMER_LINES
        assert BALMER_LINES["H-alpha"] == pytest.approx(6562.8)

    def test_telluric_bands(self):
        from jjmo_fluxcal.config import TELLURIC_BANDS
        assert len(TELLURIC_BANDS) == 5
        assert all(isinstance(b, tuple) and len(b) == 2 for b in TELLURIC_BANDS)

    def test_standard_star_db(self):
        from jjmo_fluxcal.config import STANDARD_STAR_DB
        assert "sirius" in STANDARD_STAR_DB
        assert "vega" in STANDARD_STAR_DB
        assert "betelgeuse" in STANDARD_STAR_DB
        assert STANDARD_STAR_DB["sirius"]["teff"] == 9940

    def test_all_stellar_lines(self):
        from jjmo_fluxcal.config import ALL_STELLAR_LINES, BALMER_LINES, METAL_LINES
        assert len(ALL_STELLAR_LINES) == len(BALMER_LINES) + len(METAL_LINES)


# ===================================================================
# Logging
# ===================================================================

class TestLogging:
    def test_setup_logging(self):
        from jjmo_fluxcal._logging import setup_logging, is_configured
        setup_logging("WARNING")
        assert is_configured()

    def test_setup_logging_repeated(self):
        """Repeated calls don't add duplicate handlers."""
        import logging
        from jjmo_fluxcal._logging import setup_logging
        setup_logging("INFO")
        setup_logging("DEBUG")
        pkg_logger = logging.getLogger("jjmo_fluxcal")
        assert len(pkg_logger.handlers) == 1


# ===================================================================
# CLI
# ===================================================================

class TestCLI:
    def test_parser_help(self, capsys):
        from jjmo_fluxcal.cli import _build_parser
        parser = _build_parser()
        with pytest.raises(SystemExit) as exc:
            parser.parse_args(["--help"])
        assert exc.value.code == 0

    def test_parser_version(self, capsys):
        from jjmo_fluxcal.cli import _build_parser
        parser = _build_parser()
        with pytest.raises(SystemExit) as exc:
            parser.parse_args(["--version"])
        assert exc.value.code == 0

    def test_run_subcommand_parses(self):
        from jjmo_fluxcal.cli import _build_parser
        parser = _build_parser()
        args = parser.parse_args([
            "run", "--input-dir", "/tmp/data", "--star", "sirius",
            "--output-dir", "/tmp/out",
        ])
        assert args.command == "run"
        assert args.input_dir == "/tmp/data"
        assert args.star == "sirius"
        assert args.output_dir == "/tmp/out"

    def test_sensfunc_subcommand_parses(self):
        from jjmo_fluxcal.cli import _build_parser
        parser = _build_parser()
        args = parser.parse_args([
            "sensfunc", "--input-dir", "/tmp/data", "--star", "vega",
            "--output", "/tmp/sens.json",
        ])
        assert args.command == "sensfunc"
        assert args.star == "vega"

    def test_apply_subcommand_parses(self):
        from jjmo_fluxcal.cli import _build_parser
        parser = _build_parser()
        args = parser.parse_args([
            "apply", "--sensfunc", "/tmp/sens.json",
            "--input", "/tmp/spec.fits", "--output", "/tmp/cal.fits",
        ])
        assert args.command == "apply"
        assert args.sensfunc == "/tmp/sens.json"

    def test_no_command_exits(self):
        from jjmo_fluxcal.cli import main
        with pytest.raises(SystemExit) as exc:
            main([])
        assert exc.value.code == 1


# ===================================================================
# Plotting module lazy re-exports
# ===================================================================

class TestPlottingReexports:
    def test_unknown_attribute_raises(self):
        from jjmo_fluxcal import plotting
        with pytest.raises(AttributeError):
            plotting.nonexistent_function

    def test_known_reexport_available(self):
        """Verify that a known re-export resolves without error."""
        from jjmo_fluxcal.plotting import plot_stitched_spectrum
        assert callable(plot_stitched_spectrum)


# ===================================================================
# Example config files
# ===================================================================

class TestExampleConfigs:
    def test_sirius_config_loads(self):
        from jjmo_fluxcal.config import PipelineConfig
        from pathlib import Path
        cfg_path = Path(__file__).parent.parent / "jjmo_fluxcal" / "examples" / "sirius_config.yaml"
        cfg = PipelineConfig.from_yaml(str(cfg_path))
        assert cfg.star_name == "sirius"
        assert cfg.ebv == 0.0

    def test_betelgeuse_config_loads(self):
        from jjmo_fluxcal.config import PipelineConfig
        from pathlib import Path
        cfg_path = Path(__file__).parent.parent / "jjmo_fluxcal" / "examples" / "betelgeuse_config.yaml"
        cfg = PipelineConfig.from_yaml(str(cfg_path))
        assert cfg.star_name == "betelgeuse"
        assert cfg.ebv == 0.15
        assert cfg.prefer_reference == "phoenix"
