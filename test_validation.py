"""
Tests for validation.py and paper_figures.py — Step 10
======================================================

Tests cover:
- Synthetic end-to-end pipeline (unit-level, no network)
- ValidationResult and summary formatting
- Self-consistency check with synthetic data
- Cross-validation logic with synthetic data
- Parameter sensitivity sweep (synthetic)
- SNR degradation study (synthetic)
- Literature comparison (synthetic)
- Paper figure generation (smoke tests)
- Integration tests on real JJMO data (marked @network)
"""

import os
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from calibrate import SensitivityFunction, CalibrationResult
from sensitivity import GlobalSensitivity, SensitivityFit


# ============================================================================
# Synthetic data fixtures
# ============================================================================

def _make_synthetic_segments(
    n_segments=4,
    n_pixels=200,
    wave_start=4000,
    segment_width=600,
    overlap=100,
    snr=50.0,
    rng_seed=42,
):
    """Create synthetic spectral segments that mimic JJMO data.

    Returns list of (wavelength, flux) tuples.  The 'true flux' follows
    a smooth blackbody-like curve.  Counts = true_flux * sensitivity + noise,
    where sensitivity is a smooth polynomial.
    """
    rng = np.random.default_rng(rng_seed)
    segments = []

    for i in range(n_segments):
        w0 = wave_start + i * (segment_width - overlap)
        w = np.linspace(w0, w0 + segment_width, n_pixels)

        # Smooth 'true flux' (blackbody-like)
        true_flux = 1e-12 * (5000.0 / w) ** 3

        # Smooth sensitivity (instrument throughput): polynomial shape
        w_norm = (w - w.min()) / (w.max() - w.min()) * 2 - 1
        sensitivity = 1e10 * (1.0 + 0.3 * w_norm - 0.1 * w_norm ** 2)

        # Observed counts = true_flux * sensitivity (+ noise)
        counts = true_flux * sensitivity
        noise_level = np.median(counts) / snr
        counts += rng.normal(0, noise_level, size=len(counts))

        segments.append((w, counts))

    return segments


def _make_synthetic_reference(wave_min=3800, wave_max=8000, n_points=5000):
    """Create a smooth reference spectrum (like CALSPEC).

    Returns (wavelength, flux) arrays in physical units.
    """
    w = np.linspace(wave_min, wave_max, n_points)
    # Smooth blackbody-like reference (Sirius-ish A-star)
    flux = 1e-12 * (5000.0 / w) ** 3
    return w, flux


def _make_synthetic_sensfunc(wave_min=3800, wave_max=8000, n_points=1000):
    """Create a smooth sensitivity function for testing.

    Returns a SensitivityFunction object.
    """
    w = np.linspace(wave_min, wave_max, n_points)
    w_norm = (w - w.min()) / (w.max() - w.min()) * 2 - 1
    sensitivity = 1e10 * (1.0 + 0.3 * w_norm - 0.1 * w_norm ** 2)
    uncertainty = sensitivity * 0.01

    return SensitivityFunction(
        wavelength=w,
        sensitivity=sensitivity,
        uncertainty=uncertainty,
        meta={"standard_star": "synthetic", "fit_method": "chebyshev"},
    )


@pytest.fixture
def synthetic_segments():
    """Four overlapping synthetic segments."""
    return _make_synthetic_segments()


@pytest.fixture
def synthetic_reference():
    """Smooth reference spectrum."""
    return _make_synthetic_reference()


@pytest.fixture
def synthetic_sensfunc():
    """Smooth synthetic sensitivity function."""
    return _make_synthetic_sensfunc()


# ============================================================================
# Unit tests: ValidationResult
# ============================================================================

class TestValidationResult:
    """Tests for the ValidationResult dataclass."""

    def test_creation(self):
        from validation import ValidationResult
        n = 100
        vr = ValidationResult(
            label="test",
            wavelength=np.linspace(4000, 5000, n),
            flux_calibrated=np.ones(n),
            flux_reference=np.ones(n) * 1.05,
            residual_frac=np.full(n, -0.05),
            mask=np.zeros(n, dtype=bool),
            rms_residual=0.05,
            median_residual=0.05,
            max_residual=0.05,
            mean_residual=-0.05,
        )
        assert vr.label == "test"
        assert vr.rms_residual == 0.05

    def test_with_params(self):
        from validation import ValidationResult
        n = 50
        vr = ValidationResult(
            label="with_params",
            wavelength=np.linspace(4000, 5000, n),
            flux_calibrated=np.ones(n),
            flux_reference=np.ones(n),
            residual_frac=np.zeros(n),
            mask=np.zeros(n, dtype=bool),
            rms_residual=0.0,
            median_residual=0.0,
            max_residual=0.0,
            mean_residual=0.0,
            params={"fit_order": 5},
        )
        assert vr.params["fit_order"] == 5


# ============================================================================
# Unit tests: calibrate_and_compare with synthetic data
# ============================================================================

def _make_synthetic_global_sens(
    segments=None,
    wave_min=4000,
    wave_max=6400,
    n_segments=4,
    segment_width=600,
    overlap=100,
):
    """Create a GlobalSensitivity with per-segment SensitivityFit objects.

    Mimics the output of derive_sensitivity with per-segment fits.
    """
    if segments is None:
        segments = _make_synthetic_segments(
            n_segments=n_segments, wave_start=wave_min,
            segment_width=segment_width, overlap=overlap,
        )

    fits = []
    for i, (w, f) in enumerate(segments):
        # sensitivity = F_ref / C_obs -- create a smooth function
        w_norm = (w - w.min()) / (w.max() - w.min()) * 2 - 1
        ref_flux = 1e-12 * (5000.0 / w) ** 3
        sensitivity_vals = 1e10 * (1.0 + 0.3 * w_norm - 0.1 * w_norm ** 2)
        sens_ratio = ref_flux / (f / sensitivity_vals)  # F_ref / C_obs ≈ F_ref / (F_ref * S) = 1/S

        fit = SensitivityFit(
            method="chebyshev",
            order=3,
            wave_min=float(w.min()),
            wave_max=float(w.max()),
            sigma_clip_threshold=3.0,
            n_iterations=3,
            n_rejected=0,
            n_points_used=len(w),
            segment_id=f"seg_{i:02d}",
        )
        # Fit a simple chebyshev to the ratio
        w_norm_fit = (w - w.min()) / (w.max() - w.min()) * 2 - 1
        coeffs = np.polynomial.chebyshev.chebfit(w_norm_fit, sens_ratio, 3)
        fit.coefficients = coeffs.tolist()
        fits.append(fit)

    gs = GlobalSensitivity(
        segment_fits=fits,
        wave_min=float(min(w.min() for w, _ in segments)),
        wave_max=float(max(w.max() for w, _ in segments)),
        approach="per_segment",
    )
    return gs


class TestCalibrateAndCompare:
    """Test the calibrate_and_compare helper with synthetic inputs."""

    def test_calibration_produces_output(self, synthetic_segments):
        """The function should produce a valid ValidationResult."""
        from validation import calibrate_and_compare

        gs = _make_synthetic_global_sens(synthetic_segments)

        result = calibrate_and_compare(
            synthetic_segments, gs, "sirius",
            label="synthetic_test",
        )

        assert result.label == "synthetic_test"
        assert result.wavelength is not None
        assert len(result.wavelength) > 0
        # The calibrated flux should be finite where not masked
        good = ~result.mask
        assert np.sum(good) > 0
        assert np.all(np.isfinite(result.flux_calibrated[good]))

    def test_returns_statistics(self, synthetic_segments):
        """Verify that summary statistics are computed."""
        from validation import calibrate_and_compare

        gs = _make_synthetic_global_sens(synthetic_segments)

        result = calibrate_and_compare(
            synthetic_segments, gs, "sirius",
            label="stats_test",
        )
        # Statistics should be finite
        assert np.isfinite(result.rms_residual)
        assert np.isfinite(result.median_residual)
        assert result.rms_residual >= 0
        assert result.median_residual >= 0


# ============================================================================
# Unit tests: compare_sensitivity_functions
# ============================================================================

class TestCompareSensitivityFunctions:
    """Test sensitivity function comparison utility."""

    def test_identical_functions(self, synthetic_sensfunc):
        from validation import compare_sensitivity_functions
        result = compare_sensitivity_functions(
            synthetic_sensfunc, synthetic_sensfunc,
            label_a="A", label_b="B",
        )
        assert result["rms_frac_diff"] < 1e-10
        assert result["median_frac_diff"] < 1e-10

    def test_different_functions(self):
        from validation import compare_sensitivity_functions

        w = np.linspace(4000, 7000, 500)
        sf_a = SensitivityFunction(
            wavelength=w,
            sensitivity=np.ones_like(w) * 1e10,
            meta={},
        )
        sf_b = SensitivityFunction(
            wavelength=w,
            sensitivity=np.ones_like(w) * 1.1e10,  # 10% different
            meta={},
        )
        result = compare_sensitivity_functions(sf_a, sf_b)
        # ~9.5% fractional difference expected
        assert 0.05 < result["rms_frac_diff"] < 0.15

    def test_no_overlap(self):
        from validation import compare_sensitivity_functions

        w1 = np.linspace(4000, 5000, 100)
        w2 = np.linspace(6000, 7000, 100)
        sf_a = SensitivityFunction(wavelength=w1, sensitivity=np.ones(100) * 1e10, meta={})
        sf_b = SensitivityFunction(wavelength=w2, sensitivity=np.ones(100) * 1e10, meta={})
        result = compare_sensitivity_functions(sf_a, sf_b)
        assert np.isnan(result["rms_frac_diff"])


# ============================================================================
# Unit tests: format_validation_summary
# ============================================================================

class TestFormatSummary:

    def test_with_validation_result(self):
        from validation import ValidationResult, format_validation_summary

        n = 50
        vr = ValidationResult(
            label="Test result",
            wavelength=np.linspace(4000, 5000, n),
            flux_calibrated=np.ones(n),
            flux_reference=np.ones(n),
            residual_frac=np.zeros(n),
            mask=np.zeros(n, dtype=bool),
            rms_residual=0.05,
            median_residual=0.03,
            max_residual=0.12,
            mean_residual=-0.01,
        )
        text = format_validation_summary({"test": vr})
        assert "Test result" in text
        assert "RMS residual" in text
        assert "0.0500" in text

    def test_empty_results(self):
        from validation import format_validation_summary
        text = format_validation_summary({})
        assert "Validation Summary" in text


# ============================================================================
# Smoke tests: paper figures (no assertions on visual quality)
# ============================================================================

class TestPaperFigures:
    """Smoke tests that figure functions run without errors."""

    def test_fig_raw_data_overview(self, synthetic_segments):
        from paper_figures import fig_raw_data_overview
        fig = fig_raw_data_overview(synthetic_segments, "Synthetic")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_fig_self_calibration_residuals(self):
        from paper_figures import fig_self_calibration_residuals
        from validation import ValidationResult

        n = 200
        w = np.linspace(4000, 7000, n)
        rng = np.random.default_rng(42)
        residual = rng.normal(0, 0.05, n)

        vr = ValidationResult(
            label="synthetic",
            wavelength=w,
            flux_calibrated=np.ones(n) * 1e-12,
            flux_reference=np.ones(n) * 1e-12,
            residual_frac=residual,
            mask=np.zeros(n, dtype=bool),
            rms_residual=0.05,
            median_residual=0.04,
            max_residual=0.15,
            mean_residual=0.001,
        )
        fig = fig_self_calibration_residuals(vr)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_fig_error_budget(self, synthetic_sensfunc):
        from paper_figures import fig_error_budget
        from validation import ValidationResult

        n = 200
        rng = np.random.default_rng(42)
        vr = ValidationResult(
            label="test",
            wavelength=np.linspace(4000, 7000, n),
            flux_calibrated=np.ones(n),
            flux_reference=np.ones(n),
            residual_frac=rng.normal(0, 0.02, n),
            mask=np.zeros(n, dtype=bool),
            rms_residual=0.02,
            median_residual=0.015,
            max_residual=0.06,
            mean_residual=0.001,
        )
        fig = fig_error_budget(vr, synthetic_sensfunc)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_fig_sensitivity_function(self):
        from paper_figures import fig_sensitivity_function

        # Build a minimal GlobalSensitivity-like object
        w = np.linspace(4000, 7000, 500)
        w_norm = (w - 4000) / 3000 * 2 - 1
        sens_vals = 1e10 * (1.0 + 0.3 * w_norm)

        # Create a SensitivityFit that is callable
        fit = SensitivityFit(
            method="chebyshev",
            order=3,
            wave_min=4000.0,
            wave_max=7000.0,
            sigma_clip_threshold=3.0,
            n_iterations=3,
            n_rejected=5,
            n_points_used=450,
            segment_id="test",
        )
        # Set coefficients so the fit is callable
        fit.coefficients = [1e10, 0.3e10, -0.1e10, 0.0]

        gs = GlobalSensitivity(
            segment_fits=[fit],
            wave_min=4000.0,
            wave_max=7000.0,
        )

        fig = fig_sensitivity_function(gs)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_fig_snr_degradation(self):
        from paper_figures import fig_snr_degradation
        from validation import SNRDegradationResult

        result = SNRDegradationResult(
            snr_levels=[100, 50, 25, 10, 5],
            rms_residuals=[0.02, 0.04, 0.08, 0.15, 0.35],
            median_residuals=[0.015, 0.03, 0.06, 0.12, 0.28],
            max_residuals=[0.05, 0.10, 0.20, 0.40, 0.80],
            threshold_snr=15.0,
        )
        fig = fig_snr_degradation(result)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_fig_parameter_sensitivity(self):
        from paper_figures import fig_parameter_sensitivity
        from validation import ParameterSensitivityResult

        results = {
            "fit_order": ParameterSensitivityResult(
                parameter_name="fit_order",
                parameter_values=[3, 5, 7, 9],
                rms_residuals=[0.06, 0.04, 0.05, 0.08],
                median_residuals=[0.04, 0.03, 0.035, 0.06],
                max_residuals=[0.15, 0.10, 0.12, 0.20],
                baseline_rms=0.04,
            ),
        }
        fig = fig_parameter_sensitivity(results)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_fig_cross_calibration(self):
        from paper_figures import fig_cross_calibration
        from validation import ValidationResult

        n = 200
        vr = ValidationResult(
            label="cross test",
            wavelength=np.linspace(4000, 7000, n),
            flux_calibrated=np.ones(n) * 1e-12,
            flux_reference=np.ones(n) * 1e-12 * 1.05,
            residual_frac=np.full(n, -0.05),
            mask=np.zeros(n, dtype=bool),
            rms_residual=0.05,
            median_residual=0.05,
            max_residual=0.05,
            mean_residual=-0.05,
        )
        fig = fig_cross_calibration(vr)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_fig_quality_masks(self, synthetic_segments):
        from paper_figures import fig_quality_masks
        from types import SimpleNamespace

        # Mock quality reports
        reports = []
        for w, f in synthetic_segments:
            report = SimpleNamespace(
                mask_edges=np.ones(len(w), dtype=bool),
                mask_telluric=np.ones(len(w), dtype=bool),
                mask_stellar=np.ones(len(w), dtype=bool),
                mask_cosmic=np.ones(len(w), dtype=bool),
                snr_median=50.0,
            )
            # Mask some edges
            report.mask_edges[:10] = False
            report.mask_edges[-10:] = False
            reports.append(report)

        fig = fig_quality_masks(synthetic_segments, reports)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_save_figure(self, synthetic_segments, tmp_path):
        from paper_figures import fig_raw_data_overview
        save_path = str(tmp_path / "test_fig")
        fig = fig_raw_data_overview(synthetic_segments, "Test", save_path=save_path)
        assert (tmp_path / "test_fig.pdf").exists()
        assert (tmp_path / "test_fig.png").exists()
        plt.close(fig)


# ============================================================================
# Integration tests on real JJMO data (require network + data)
# ============================================================================

SIRIUS_DIR = "/home/habjan.e/JJMO_home/Data/Sirius"
BETELGEUSE_DIR = "/home/habjan.e/JJMO_home/Data/Betelgeuse"

_has_sirius_data = Path(SIRIUS_DIR).exists() and any(
    Path(SIRIUS_DIR).glob("*.fit")
)
_has_betelgeuse_data = Path(BETELGEUSE_DIR).exists() and any(
    Path(BETELGEUSE_DIR).glob("*.csv")
)


@pytest.mark.network
@pytest.mark.skipif(not _has_sirius_data, reason="Sirius data not found")
class TestSiriusSelfConsistency:
    """Integration test: full pipeline on Sirius (task 10.1)."""

    def test_self_consistency(self):
        from validation import self_consistency_sirius

        result = self_consistency_sirius(SIRIUS_DIR)

        # The self-consistency RMS should be below 50% for this noisy data
        assert np.isfinite(result.rms_residual)
        assert result.rms_residual < 0.50, (
            f"Self-consistency RMS too high: {result.rms_residual:.4f}"
        )
        # Should have valid pixels
        n_good = np.sum(~result.mask)
        assert n_good > 100, f"Only {n_good} valid pixels"


@pytest.mark.network
@pytest.mark.skipif(
    not (_has_sirius_data and _has_betelgeuse_data),
    reason="Data not found",
)
class TestCrossValidation:
    """Integration tests: cross-validation between stars (tasks 10.2-10.3)."""

    def test_sirius_to_betelgeuse(self):
        from validation import cross_validate_sirius_to_betelgeuse

        result = cross_validate_sirius_to_betelgeuse(SIRIUS_DIR, BETELGEUSE_DIR)
        assert np.isfinite(result.rms_residual)
        n_good = np.sum(~result.mask)
        assert n_good > 50

    def test_betelgeuse_to_sirius(self):
        from validation import cross_validate_betelgeuse_to_sirius

        result = cross_validate_betelgeuse_to_sirius(SIRIUS_DIR, BETELGEUSE_DIR)
        assert np.isfinite(result.rms_residual)
        n_good = np.sum(~result.mask)
        assert n_good > 50


@pytest.mark.network
@pytest.mark.skipif(not _has_sirius_data, reason="Sirius data not found")
class TestParameterSensitivity:
    """Integration test: parameter sensitivity analysis (task 10.5)."""

    def test_fit_order_sweep(self):
        from validation import parameter_sensitivity_analysis

        # Use a small grid to keep the test fast
        results = parameter_sensitivity_analysis(
            SIRIUS_DIR,
            parameter_grid={"fit_order": [3, 5]},
        )
        assert "fit_order" in results
        assert len(results["fit_order"].rms_residuals) == 2
        assert all(np.isfinite(r) for r in results["fit_order"].rms_residuals)


@pytest.mark.network
@pytest.mark.skipif(not _has_sirius_data, reason="Sirius data not found")
class TestSNRDegradation:
    """Integration test: SNR degradation study (task 10.6)."""

    def test_degradation(self):
        from validation import snr_degradation_study

        result = snr_degradation_study(
            SIRIUS_DIR,
            noise_multipliers=[0.0, 2.0],
            n_trials=1,
        )
        assert len(result.snr_levels) == 2
        assert len(result.rms_residuals) == 2
        # The no-added-noise case should have lower RMS than the noisy one
        assert result.rms_residuals[0] <= result.rms_residuals[1] + 0.1


@pytest.mark.network
@pytest.mark.skipif(not _has_sirius_data, reason="Sirius data not found")
class TestLiteratureComparison:
    """Integration test: comparison to literature (task 10.9)."""

    def test_calspec_comparison(self):
        from validation import self_consistency_sirius, compare_to_literature

        # First get a calibrated spectrum
        vr = self_consistency_sirius(SIRIUS_DIR)
        good = ~vr.mask
        results = compare_to_literature(
            vr.wavelength[good], vr.flux_calibrated[good], "sirius",
            libraries=["calspec"],
        )
        assert "calspec" in results
        assert np.isfinite(results["calspec"].rms_residual)


@pytest.mark.network
@pytest.mark.skipif(not _has_sirius_data, reason="Sirius data not found")
class TestFigureGeneration:
    """Integration test: generate paper figures from real data."""

    def test_generate_key_figures(self, tmp_path):
        from validation import self_consistency_sirius, run_pipeline
        from paper_figures import (
            fig_raw_data_overview,
            fig_self_calibration_residuals,
        )
        from stitching import load_jjmo_sirius

        segments = load_jjmo_sirius(SIRIUS_DIR)
        fig1 = fig_raw_data_overview(
            segments, "Sirius",
            save_path=str(tmp_path / "fig1_test"),
        )
        assert (tmp_path / "fig1_test.png").exists()
        plt.close(fig1)

        # Self-consistency
        result = self_consistency_sirius(SIRIUS_DIR)
        fig5 = fig_self_calibration_residuals(
            result,
            save_path=str(tmp_path / "fig5_test"),
        )
        assert (tmp_path / "fig5_test.png").exists()
        plt.close(fig5)


# ============================================================================
# Regression test: save and compare known-good outputs
# ============================================================================

class TestRegression:
    """Regression tests using synthetic data with fixed seeds."""

    def test_synthetic_pipeline_deterministic(self):
        """Running the same synthetic data twice gives identical results."""
        from validation import calibrate_and_compare

        segs1 = _make_synthetic_segments(rng_seed=123)
        segs2 = _make_synthetic_segments(rng_seed=123)
        gs = _make_synthetic_global_sens(segs1)

        r1 = calibrate_and_compare(segs1, gs, "sirius", label="run1")
        r2 = calibrate_and_compare(segs2, gs, "sirius", label="run2")

        assert_allclose(r1.rms_residual, r2.rms_residual, rtol=1e-10)
        good1 = ~r1.mask
        good2 = ~r2.mask
        assert_allclose(
            r1.flux_calibrated[good1],
            r2.flux_calibrated[good2],
            rtol=1e-10,
        )

    def test_noise_increases_scatter(self):
        """Adding noise should increase the scatter in calibrated flux."""
        from validation import calibrate_and_compare

        # Measure the pixel-to-pixel scatter (high-freq noise) in the
        # calibrated spectrum, which should increase with added noise
        # even though the reference mismatch dominates the absolute RMS.
        scatter_values = []
        for snr in [200, 50, 10]:
            segs = _make_synthetic_segments(snr=snr, rng_seed=42)
            gs = _make_synthetic_global_sens(segs)
            r = calibrate_and_compare(segs, gs, "sirius", label=f"snr={snr}")
            good = ~r.mask
            if np.sum(good) > 10:
                # Pixel-to-pixel scatter from first differences
                diffs = np.diff(r.flux_calibrated[good])
                scatter = np.median(np.abs(diffs)) / 1.4826
                scatter_values.append(scatter)

        assert len(scatter_values) == 3
        # High-freq scatter should increase monotonically as SNR drops
        assert scatter_values[-1] > scatter_values[0], (
            f"Expected higher scatter at lower SNR: {scatter_values}"
        )
