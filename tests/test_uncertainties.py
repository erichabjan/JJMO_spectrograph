"""
test_uncertainties.py — Tests for the Uncertainty Propagation & Error Budget module.

Tests cover all subsections:
  8.1  Photon noise and read noise propagation
  8.2  Wavelength calibration uncertainty
  8.3  Sensitivity function fit uncertainty
  8.4  Stitching / cross-normalization uncertainty
  8.5  Systematic error sources
  8.6  Total error budget table
  8.7  Monte Carlo end-to-end validation
"""

import numpy as np
import pytest

from jjmo_fluxcal.uncertainties import (
    # 8.1 - Photon/read noise
    estimate_pixel_uncertainty,
    propagate_spatial_collapse,
    estimate_1d_uncertainty,
    propagate_division,
    propagate_multiplication,
    DEFAULT_GAIN,
    DEFAULT_READ_NOISE_E,
    DEFAULT_DARK_CURRENT,
    # 8.2 - Wavelength calibration
    wavelength_to_flux_uncertainty,
    wavelength_calibration_uncertainty,
    # 8.3 - Sensitivity fit
    sensitivity_fit_covariance,
    propagate_sensitivity_fit_uncertainty,
    bootstrap_sensitivity_uncertainty,
    # 8.4 - Stitching
    normalization_factor_uncertainty,
    stitching_uncertainty,
    # 8.5 - Systematic errors
    extinction_law_uncertainty,
    reference_spectrum_uncertainty,
    airmass_uncertainty,
    telluric_residual_uncertainty,
    slit_loss_uncertainty,
    TELLURIC_BANDS,
    # 8.6 - Error budget
    ErrorBudget,
    build_error_budget,
    # 8.7 - Monte Carlo
    monte_carlo_validation,
    MonteCarloResult,
)


# ============================================================================
# Helper: generate synthetic data for tests
# ============================================================================

def _synthetic_spectrum(n=500, wave_min=4000.0, wave_max=4500.0,
                        peak_counts=10000.0, noise_frac=0.01):
    """Generate a synthetic smooth spectrum with Poisson-like noise."""
    rng = np.random.RandomState(123)
    wavelength = np.linspace(wave_min, wave_max, n)
    # Smooth continuum: a parabola peaking in the center
    w_mid = 0.5 * (wave_min + wave_max)
    flux = peak_counts * (1.0 - 0.5 * ((wavelength - w_mid) /
                                         (0.5 * (wave_max - wave_min))) ** 2)
    # Add a Gaussian absorption line at the center
    flux -= 2000.0 * np.exp(-0.5 * ((wavelength - w_mid) / 3.0) ** 2)
    # Add noise
    noise = rng.normal(0, noise_frac * flux)
    flux_noisy = flux + noise
    return wavelength, flux, flux_noisy


# ============================================================================
# 8.1  Photon noise and read noise propagation
# ============================================================================

class TestPhotonReadNoise:
    """Tests for §8.1: pixel-level uncertainty estimation."""

    def test_estimate_pixel_uncertainty_shape(self):
        """Output shape matches input 2D image."""
        image = np.ones((50, 100)) * 1000.0
        unc = estimate_pixel_uncertainty(image)
        assert unc.shape == image.shape

    def test_estimate_pixel_uncertainty_poisson_dominant(self):
        """For high counts, photon noise dominates and unc ~ sqrt(counts/gain)."""
        counts = 100000.0  # high counts
        image = np.full((10, 100), counts)
        unc = estimate_pixel_uncertainty(image, gain=DEFAULT_GAIN,
                                         read_noise=DEFAULT_READ_NOISE_E)
        expected_photon = np.sqrt(counts / DEFAULT_GAIN)
        # Photon noise should dominate; unc should be close to photon-only value
        assert np.allclose(unc, expected_photon, rtol=0.05)

    def test_estimate_pixel_uncertainty_read_noise_dominant(self):
        """For very low counts, read noise dominates."""
        counts = 0.0  # zero counts
        image = np.full((10, 100), counts)
        unc = estimate_pixel_uncertainty(image, gain=DEFAULT_GAIN,
                                         read_noise=DEFAULT_READ_NOISE_E)
        expected_read = DEFAULT_READ_NOISE_E / DEFAULT_GAIN
        assert np.allclose(unc, expected_read, rtol=1e-10)

    def test_estimate_pixel_uncertainty_negative_counts(self):
        """Negative counts (e.g., from sky subtraction) are clamped to 0 for photon noise."""
        image = np.full((5, 100), -50.0)
        unc = estimate_pixel_uncertainty(image)
        # Only read noise + dark current (photon noise = 0 because clamped)
        assert np.all(unc > 0)
        assert np.all(np.isfinite(unc))

    def test_propagate_spatial_collapse(self):
        """Quadrature sum along spatial axis is correct."""
        # 5 rows, each with uncertainty = 3.0 -> collapsed unc = 3*sqrt(5)
        unc_2d = np.full((5, 100), 3.0)
        unc_1d = propagate_spatial_collapse(unc_2d, collapse_axis=0)
        assert unc_1d.shape == (100,)
        expected = 3.0 * np.sqrt(5)
        assert np.allclose(unc_1d, expected, rtol=1e-10)

    def test_estimate_1d_uncertainty_scaling(self):
        """1D uncertainty scales with sqrt(counts) for high counts."""
        flux_lo = np.full(100, 100.0)
        flux_hi = np.full(100, 10000.0)
        unc_lo = estimate_1d_uncertainty(flux_lo, n_rows=1)
        unc_hi = estimate_1d_uncertainty(flux_hi, n_rows=1)
        # For Poisson-dominated case, unc ratio ~ sqrt(flux ratio) = 10
        ratio = np.median(unc_hi / unc_lo)
        assert 5 < ratio < 15  # approximately sqrt(100) = 10

    def test_estimate_1d_uncertainty_n_rows(self):
        """More spatial rows increases read noise contribution."""
        flux = np.full(100, 1000.0)
        unc_1row = estimate_1d_uncertainty(flux, n_rows=1)
        unc_10rows = estimate_1d_uncertainty(flux, n_rows=10)
        # 10 rows adds more read noise, so unc should be slightly larger
        assert np.all(unc_10rows >= unc_1row)

    def test_propagate_division_no_divisor_unc(self):
        """Division by a constant: only numerator uncertainty matters."""
        flux = np.array([100.0, 200.0, 300.0])
        flux_unc = np.array([10.0, 20.0, 30.0])
        divisor = 2.0
        result, result_unc = propagate_division(flux, flux_unc, divisor)
        np.testing.assert_allclose(result, flux / 2.0)
        np.testing.assert_allclose(result_unc, flux_unc / 2.0)

    def test_propagate_division_with_divisor_unc(self):
        """Division with uncertainty on both numerator and denominator."""
        flux = np.array([100.0])
        flux_unc = np.array([10.0])  # 10% fractional
        divisor = np.array([50.0])
        divisor_unc = np.array([5.0])  # 10% fractional
        result, result_unc = propagate_division(flux, flux_unc, divisor,
                                                 divisor_unc)
        expected_result = 100.0 / 50.0
        expected_frac = np.sqrt(0.1**2 + 0.1**2)
        np.testing.assert_allclose(result, expected_result)
        np.testing.assert_allclose(result_unc, expected_result * expected_frac,
                                    rtol=1e-10)

    def test_propagate_multiplication(self):
        """Multiplication propagation with both factors uncertain."""
        a = np.array([100.0])
        a_unc = np.array([10.0])  # 10%
        b = np.array([5.0])
        b_unc = np.array([0.5])  # 10%
        result, result_unc = propagate_multiplication(a, a_unc, b, b_unc)
        np.testing.assert_allclose(result, 500.0)
        expected_frac = np.sqrt(0.1**2 + 0.1**2)
        np.testing.assert_allclose(result_unc, 500.0 * expected_frac,
                                    rtol=1e-10)


# ============================================================================
# 8.2  Wavelength calibration uncertainty
# ============================================================================

class TestWavelengthCalibrationUncertainty:
    """Tests for §8.2: wavelength-to-flux uncertainty conversion."""

    def test_flat_spectrum_no_uncertainty(self):
        """Flat spectrum has zero dF/dlambda, so wavelength error contributes nothing."""
        wave = np.linspace(4000, 4500, 100)
        flux = np.ones(100) * 1000.0
        unc = wavelength_to_flux_uncertainty(wave, flux, delta_lambda=1.0)
        # dF/dlambda ~ 0 for a flat spectrum
        assert np.all(unc < 1.0)  # should be ~0

    def test_steep_spectrum_large_uncertainty(self):
        """Steep spectral feature produces large wavelength-induced flux error."""
        wave = np.linspace(4000, 4500, 500)
        flux = np.ones(500) * 1000.0
        # Add a narrow, deep absorption line
        flux -= 800.0 * np.exp(-0.5 * ((wave - 4250) / 2.0) ** 2)
        unc = wavelength_to_flux_uncertainty(wave, flux, delta_lambda=1.0)
        # Uncertainty should be large near the line wings
        line_region = (wave > 4245) & (wave < 4255)
        continuum_region = (wave < 4200) | (wave > 4300)
        assert np.mean(unc[line_region]) > 5 * np.mean(unc[continuum_region])

    def test_wavelength_calibration_uncertainty_report(self):
        """Full function returns both uncertainty array and report dict."""
        wave = np.linspace(4000, 4500, 100)
        flux = np.linspace(1000, 2000, 100)  # linearly increasing
        flux_unc, report = wavelength_calibration_uncertainty(
            wave, flux, rms_residual=0.5
        )
        assert len(flux_unc) == 100
        assert "rms_residual_angstrom" in report
        assert report["rms_residual_angstrom"] == 0.5

    def test_with_line_info(self):
        """Report includes calibration line info when provided."""
        wave = np.linspace(4000, 4500, 100)
        flux = np.ones(100) * 1000.0
        centroids = np.array([4100, 4200, 4300])
        line_unc = np.array([0.1, 0.2, 0.15])
        _, report = wavelength_calibration_uncertainty(
            wave, flux, rms_residual=0.3,
            line_centroids=centroids, line_uncertainties=line_unc
        )
        assert report["n_calibration_lines"] == 3
        assert np.isclose(report["mean_line_centroid_uncertainty"], 0.15)


# ============================================================================
# 8.3  Sensitivity function fit uncertainty
# ============================================================================

class TestSensitivityFitUncertainty:
    """Tests for §8.3: covariance-based and bootstrap uncertainty."""

    def _make_sensitivity_data(self, n=200, order=3):
        """Create synthetic sensitivity ratio data with known polynomial."""
        rng = np.random.RandomState(42)
        wave = np.linspace(4000, 5000, n)
        # True sensitivity: cubic polynomial
        w_norm = 2 * (wave - 4000) / 1000 - 1
        true_coeffs = np.array([1.0, 0.1, -0.05, 0.02])
        true_sens = np.polynomial.chebyshev.chebval(w_norm, true_coeffs)
        noise = rng.normal(0, 0.02 * np.mean(true_sens), n)
        observed = true_sens + noise
        mask = np.ones(n, dtype=bool)
        return wave, observed, mask, true_coeffs

    def test_covariance_shape(self):
        """Covariance matrix has correct shape."""
        wave, ratio, mask, _ = self._make_sensitivity_data(order=3)
        coeffs, cov, rms = sensitivity_fit_covariance(
            wave, ratio, mask, method='chebyshev', order=3
        )
        assert coeffs.shape == (4,)
        assert cov.shape == (4, 4)
        assert rms > 0

    def test_covariance_positive_diagonal(self):
        """Diagonal of covariance must be positive (variances)."""
        wave, ratio, mask, _ = self._make_sensitivity_data()
        _, cov, _ = sensitivity_fit_covariance(
            wave, ratio, mask, method='chebyshev', order=3
        )
        assert np.all(np.diag(cov) > 0)

    def test_covariance_symmetric(self):
        """Covariance matrix must be symmetric."""
        wave, ratio, mask, _ = self._make_sensitivity_data()
        _, cov, _ = sensitivity_fit_covariance(
            wave, ratio, mask, method='chebyshev', order=3
        )
        np.testing.assert_allclose(cov, cov.T, atol=1e-12)

    def test_propagated_uncertainty_shape(self):
        """Propagated sensitivity uncertainty has correct shape."""
        wave, ratio, mask, _ = self._make_sensitivity_data()
        coeffs, cov, _ = sensitivity_fit_covariance(
            wave, ratio, mask, method='chebyshev', order=3
        )
        sens, sens_unc = propagate_sensitivity_fit_uncertainty(
            wave, coeffs, cov, method='chebyshev',
            wave_min=4000, wave_max=5000
        )
        assert sens.shape == wave.shape
        assert sens_unc.shape == wave.shape
        assert np.all(sens_unc >= 0)

    def test_more_data_reduces_uncertainty(self):
        """More data points should reduce the fit uncertainty."""
        rng = np.random.RandomState(42)
        # Small dataset
        wave_small = np.linspace(4000, 5000, 50)
        w_norm_s = 2 * (wave_small - 4000) / 1000 - 1
        true = np.polynomial.chebyshev.chebval(w_norm_s, [1, 0.1])
        obs_small = true + rng.normal(0, 0.02, 50)
        mask_small = np.ones(50, dtype=bool)
        _, cov_small, _ = sensitivity_fit_covariance(
            wave_small, obs_small, mask_small, method='chebyshev', order=1
        )
        # Large dataset
        wave_large = np.linspace(4000, 5000, 500)
        w_norm_l = 2 * (wave_large - 4000) / 1000 - 1
        true_l = np.polynomial.chebyshev.chebval(w_norm_l, [1, 0.1])
        obs_large = true_l + rng.normal(0, 0.02, 500)
        mask_large = np.ones(500, dtype=bool)
        _, cov_large, _ = sensitivity_fit_covariance(
            wave_large, obs_large, mask_large, method='chebyshev', order=1
        )
        # Larger dataset should have smaller diagonal covariance
        assert np.all(np.diag(cov_large) < np.diag(cov_small))

    def test_legendre_method(self):
        """Covariance works for Legendre polynomials too."""
        wave, ratio, mask, _ = self._make_sensitivity_data()
        coeffs, cov, rms = sensitivity_fit_covariance(
            wave, ratio, mask, method='legendre', order=3
        )
        assert cov.shape == (4, 4)
        assert np.all(np.diag(cov) > 0)

    def test_bootstrap_returns_plausible_spread(self):
        """Bootstrap std should be comparable to analytic uncertainty."""
        wave, ratio, mask, _ = self._make_sensitivity_data(n=100)

        def fit_func(w, r, m):
            good = m & np.isfinite(r)
            if np.sum(good) < 5:
                return np.full_like(w, np.nan)
            w_norm = 2 * (w - 4000) / 1000 - 1
            coeffs = np.polynomial.chebyshev.chebfit(w_norm[good], r[good], 3)
            return np.polynomial.chebyshev.chebval(w_norm, coeffs)

        mean, std, samples = bootstrap_sensitivity_uncertainty(
            wave, ratio, mask, fit_func, n_bootstrap=50, random_state=42
        )
        assert mean.shape == wave.shape
        assert std.shape == wave.shape
        # Standard deviation should be positive and reasonable
        good = np.isfinite(std) & (std > 0)
        assert np.sum(good) > 50  # most pixels should have valid bootstrap std
        # std should be much smaller than the signal
        assert np.nanmedian(std[good]) < 0.5 * np.nanmedian(np.abs(mean[good]))


# ============================================================================
# 8.4  Stitching / cross-normalization uncertainty
# ============================================================================

class TestStitchingUncertainty:
    """Tests for §8.4: normalization factor uncertainty propagation."""

    def test_reference_segment_zero_uncertainty(self):
        """Reference segment should have zero normalization uncertainty."""
        unc = normalization_factor_uncertainty(factor=1.0, factor_unc=0.01,
                                               n_hops=0)
        assert unc == 0.0

    def test_uncertainty_grows_with_hops(self):
        """Uncertainty should increase with distance from reference."""
        unc_1 = normalization_factor_uncertainty(1.0, 0.02, n_hops=1)
        unc_3 = normalization_factor_uncertainty(1.0, 0.02, n_hops=3)
        unc_5 = normalization_factor_uncertainty(1.0, 0.02, n_hops=5)
        assert unc_1 < unc_3 < unc_5

    def test_uncertainty_sqrt_n_scaling(self):
        """Uncertainty should scale as sqrt(n) hops."""
        unc_1 = normalization_factor_uncertainty(1.0, 0.02, n_hops=1)
        unc_4 = normalization_factor_uncertainty(1.0, 0.02, n_hops=4)
        # unc_4 / unc_1 should be ~ sqrt(4) = 2
        ratio = unc_4 / unc_1
        np.testing.assert_allclose(ratio, 2.0, rtol=1e-10)

    def test_stitching_uncertainty_output(self):
        """Full stitching uncertainty returns correct shapes."""
        wave = np.linspace(4000, 7000, 1000)
        flux = np.ones(1000) * 5000.0
        norm_factors = [
            {'segment_idx': 0, 'factor': 1.0, 'factor_uncertainty': 0.0},
            {'segment_idx': 1, 'factor': 1.02, 'factor_uncertainty': 0.01},
            {'segment_idx': 2, 'factor': 0.98, 'factor_uncertainty': 0.015},
        ]
        stitch_unc, seg_map = stitching_uncertainty(
            wave, flux, norm_factors, reference_idx=0
        )
        assert stitch_unc.shape == wave.shape
        assert 0 in seg_map
        assert seg_map[0][1] == 0.0  # reference has zero uncertainty


# ============================================================================
# 8.5  Systematic error sources
# ============================================================================

class TestSystematicErrors:
    """Tests for §8.5: extinction, reference, airmass, telluric, slit loss."""

    def test_extinction_law_uncertainty_shape(self):
        """Extinction uncertainty has correct shape."""
        wave = np.linspace(3900, 7900, 500)
        flux = np.ones(500) * 1e-14
        unc, report = extinction_law_uncertainty(wave, flux, airmass=1.5)
        assert unc.shape == wave.shape
        assert "rv_values_tested" in report

    def test_extinction_larger_at_blue(self):
        """Extinction uncertainty should be larger at shorter wavelengths."""
        wave = np.linspace(3900, 7900, 500)
        flux = np.ones(500) * 1e-14  # flat for clean comparison
        unc, _ = extinction_law_uncertainty(wave, flux, airmass=1.5)
        blue = wave < 4500
        red = wave > 7000
        assert np.nanmean(unc[blue]) > np.nanmean(unc[red])

    def test_reference_spectrum_uncertainty(self):
        """Reference comparison returns sensible fractional difference."""
        wave = np.linspace(4000, 5000, 200)
        s1 = np.ones(200) * 1.0
        s2 = np.ones(200) * 1.05  # 5% difference
        frac_diff, report = reference_spectrum_uncertainty(wave, s1, s2)
        # Should be ~ 0.05 / 1.025 ≈ 0.0488
        np.testing.assert_allclose(
            report["mean_fractional_difference"], 0.05 / 1.025, rtol=0.01
        )

    def test_airmass_uncertainty_proportional_to_delta(self):
        """Airmass uncertainty scales linearly with delta_airmass."""
        wave = np.linspace(4000, 5000, 100)
        flux = np.ones(100) * 1e-14
        unc_small = airmass_uncertainty(wave, flux, airmass=1.5,
                                         delta_airmass=0.05)
        unc_large = airmass_uncertainty(wave, flux, airmass=1.5,
                                         delta_airmass=0.10)
        # Should scale linearly
        np.testing.assert_allclose(unc_large, 2.0 * unc_small, rtol=1e-10)

    def test_telluric_residual_inflation(self):
        """Telluric regions get inflated uncertainty."""
        wave = np.linspace(6200, 6400, 200)
        base_unc = np.ones(200) * 0.01
        inflated, in_telluric = telluric_residual_uncertainty(
            wave, base_unc, inflation_factor=3.0
        )
        # The O2 6270-6290 band should be flagged
        assert np.any(in_telluric)
        # Inflated values should be 3x the base
        np.testing.assert_allclose(inflated[in_telluric],
                                    3.0 * base_unc[in_telluric])
        # Non-telluric regions unchanged
        np.testing.assert_allclose(inflated[~in_telluric],
                                    base_unc[~in_telluric])

    def test_slit_loss_uncertainty_shape(self):
        """Slit loss uncertainty returns correct shape."""
        wave = np.linspace(4000, 7000, 300)
        slit_unc = slit_loss_uncertainty(wave)
        assert slit_unc.shape == wave.shape
        assert np.all(slit_unc > 0)

    def test_slit_loss_chromatic_blue_larger(self):
        """Chromatic slit loss should be larger at blue wavelengths."""
        wave = np.linspace(4000, 7000, 300)
        slit_unc = slit_loss_uncertainty(wave, fractional_grey=0.0,
                                          fractional_chromatic=0.05)
        # Blue end should have larger uncertainty
        blue = wave < 4500
        red = wave > 6500
        assert np.mean(slit_unc[blue]) > np.mean(slit_unc[red])


# ============================================================================
# 8.6  Total error budget table
# ============================================================================

class TestErrorBudget:
    """Tests for §8.6: error budget construction and table output."""

    def _make_budget(self):
        """Create a simple test error budget."""
        n = 200
        wave = np.linspace(4000, 5000, n)
        flux = np.ones(n) * 1000.0
        return build_error_budget(
            wave, flux,
            photon_noise=np.ones(n) * 30.0,      # dominant
            read_noise=np.ones(n) * 5.0,
            wavelength_cal_unc=np.ones(n) * 2.0,
            sensitivity_fit_unc=np.ones(n) * 10.0,
            stitching_unc=np.ones(n) * 3.0,
            extinction_unc=np.ones(n) * 1.0,
        )

    def test_build_error_budget_total(self):
        """Total uncertainty is quadrature sum of components."""
        budget = self._make_budget()
        expected_total = np.sqrt(30**2 + 5**2 + 2**2 + 10**2 + 3**2 + 1**2)
        np.testing.assert_allclose(budget.total, expected_total, rtol=1e-10)

    def test_dominant_source_is_photon_noise(self):
        """Photon noise (30) should dominate over all other terms."""
        budget = self._make_budget()
        # Most pixels should have photon_noise as dominant
        from collections import Counter
        counts = Counter(budget.dominant_source)
        assert counts.get("photon_noise", 0) == len(budget.wavelength)

    def test_error_budget_table(self):
        """Table output has expected structure."""
        budget = self._make_budget()
        table = budget.to_table(n_bins=5)
        assert len(table) == 5
        row = table[0]
        assert "wave_center" in row
        assert "snr" in row
        assert "dominant_source" in row
        assert "frac_photon_noise" in row

    def test_error_budget_summary(self):
        """Summary string is generated without errors."""
        budget = self._make_budget()
        summary = budget.summary()
        assert "Error Budget Summary" in summary
        assert "photon_noise" in summary

    def test_none_components_treated_as_zero(self):
        """Components passed as None should not affect the total."""
        n = 100
        wave = np.linspace(4000, 5000, n)
        flux = np.ones(n) * 1000.0
        budget = build_error_budget(
            wave, flux,
            photon_noise=np.ones(n) * 10.0,
            # all others are None
        )
        np.testing.assert_allclose(budget.total, 10.0, rtol=1e-10)


# ============================================================================
# 8.7  Monte Carlo end-to-end validation
# ============================================================================

class TestMonteCarlo:
    """Tests for §8.7: MC end-to-end validation."""

    def _setup_mc(self):
        """Create simple inputs for MC validation."""
        n = 100
        wave = np.linspace(4000, 5000, n)
        counts = np.ones(n) * 10000.0
        counts_unc = np.ones(n) * 100.0  # 1% noise
        sensitivity = np.ones(n) * 2.0
        sensitivity_unc = np.ones(n) * 0.05  # 2.5%
        return wave, counts, counts_unc, sensitivity, sensitivity_unc

    def test_mc_result_shapes(self):
        """MC result arrays have correct shapes."""
        wave, counts, counts_unc, sens, sens_unc = self._setup_mc()
        result = monte_carlo_validation(
            wave, counts, counts_unc, sens, sens_unc,
            n_realizations=50, random_state=42
        )
        assert result.wavelength.shape == wave.shape
        assert result.flux_mean.shape == wave.shape
        assert result.flux_std.shape == wave.shape
        assert result.analytic_unc.shape == wave.shape
        assert result.ratio.shape == wave.shape
        assert result.n_realizations == 50

    def test_mc_consistency(self):
        """MC and analytic uncertainties should roughly agree."""
        wave, counts, counts_unc, sens, sens_unc = self._setup_mc()
        result = monte_carlo_validation(
            wave, counts, counts_unc, sens, sens_unc,
            n_realizations=500, random_state=42
        )
        consistent, med_ratio = result.is_consistent(tolerance=0.3)
        assert consistent, (
            f"MC/analytic ratio median = {med_ratio:.3f}, expected ~1.0"
        )

    def test_mc_with_wavelength_uncertainty(self):
        """MC validation works with wavelength perturbation."""
        wave, counts, counts_unc, sens, sens_unc = self._setup_mc()
        result = monte_carlo_validation(
            wave, counts, counts_unc, sens, sens_unc,
            wavelength_unc=0.5, n_realizations=50, random_state=42
        )
        assert np.all(np.isfinite(result.flux_mean))

    def test_mc_summary(self):
        """Summary string is generated without errors."""
        wave, counts, counts_unc, sens, sens_unc = self._setup_mc()
        result = monte_carlo_validation(
            wave, counts, counts_unc, sens, sens_unc,
            n_realizations=50, random_state=42
        )
        summary = result.summary()
        assert "Monte Carlo Validation" in summary
        assert "Realizations: 50" in summary

    def test_mc_zero_sensitivity_unc(self):
        """MC works when sensitivity has zero uncertainty."""
        wave, counts, counts_unc, sens, _ = self._setup_mc()
        sens_unc = np.zeros_like(sens)
        result = monte_carlo_validation(
            wave, counts, counts_unc, sens, sens_unc,
            n_realizations=50, random_state=42
        )
        # MC spread should come only from count noise
        # Analytic: sigma_F = (sigma_counts / counts) * F
        flux_nominal = counts / sens
        analytic_from_counts = (counts_unc / counts) * flux_nominal
        # MC std should approximate this
        good = np.isfinite(result.flux_std) & (result.flux_std > 0)
        ratio = np.nanmedian(result.flux_std[good] / analytic_from_counts[good])
        assert 0.7 < ratio < 1.5


# ============================================================================
# Edge cases and integration
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and unusual inputs."""

    def test_zero_flux_division(self):
        """Division by zero flux is handled gracefully."""
        flux = np.array([0.0, 100.0, 0.0])
        flux_unc = np.array([10.0, 10.0, 10.0])
        divisor = np.array([2.0, 2.0, 2.0])
        result, result_unc = propagate_division(flux, flux_unc, divisor)
        assert np.isfinite(result[1])
        assert result[0] == 0.0  # 0/2 = 0

    def test_zero_divisor(self):
        """Division by zero produces NaN, not crash."""
        flux = np.array([100.0])
        flux_unc = np.array([10.0])
        divisor = np.array([0.0])
        result, result_unc = propagate_division(flux, flux_unc, divisor)
        assert np.isnan(result[0])

    def test_single_pixel_spectrum(self):
        """Functions handle single-pixel arrays."""
        wave = np.array([4500.0])
        flux = np.array([1000.0])
        unc = wavelength_to_flux_uncertainty(wave, flux, delta_lambda=0.5)
        assert len(unc) == 1

    def test_slit_loss_single_wavelength(self):
        """Slit loss handles a single-element wavelength array."""
        wave = np.array([5000.0])
        slit_unc = slit_loss_uncertainty(wave)
        assert len(slit_unc) == 1
        assert slit_unc[0] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
