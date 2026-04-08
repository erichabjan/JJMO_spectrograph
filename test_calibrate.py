"""
Tests for calibrate.py — Step 7: Flux Calibration Application
==============================================================

Tests cover:
- SensitivityFunction creation, evaluation, serialisation (FITS round-trip)
- Core calibration (apply_sensitivity) with known analytical inputs
- Differential atmospheric extinction correction
- Self-calibration check (residual recovery)
- FITS and CSV output/read round-trip
- Batch mode
- Edge cases: missing metadata, bad pixels, non-uniform grids
"""

import os
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import astropy.units as u
from astropy.nddata import StdDevUncertainty

try:
    from specutils import Spectrum as Spectrum1D
except ImportError:
    from specutils import Spectrum1D

from calibrate import (
    SensitivityFunction,
    CalibrationResult,
    apply_sensitivity,
    apply_sensitivity_per_segment,
    default_extinction_curve,
    self_calibration_check,
    SelfCalibrationReport,
    write_calibrated_fits,
    write_calibrated_csv,
    read_calibrated_fits,
    calibrate_batch,
    get_slit_loss_documentation,
    flag_slit_loss_systematic,
    _unpack_observed,
    _compute_extinction_factor,
    _resolve_extinction_curve,
    FLUX_UNIT,
    SLIT_LOSS_DOCUMENTATION,
)


# ---------------------------------------------------------------------------
# Fixtures: synthetic data
# ---------------------------------------------------------------------------

@pytest.fixture
def wavelength_grid():
    """Simple uniform wavelength grid 4000-5000 Å."""
    return np.linspace(4000, 5000, 500)


@pytest.fixture
def flat_sensitivity(wavelength_grid):
    """Sensitivity function that is a constant (no colour term)."""
    sens = np.full_like(wavelength_grid, 1e10)  # counts/s per erg/s/cm^2/A
    return SensitivityFunction(
        wavelength=wavelength_grid,
        sensitivity=sens,
        uncertainty=sens * 0.01,
        meta={"standard_star": "test_flat", "airmass_std": 1.0},
    )


@pytest.fixture
def linear_sensitivity(wavelength_grid):
    """Sensitivity with a linear colour term: more sensitive in the red."""
    sens = 1e10 * (1.0 + (wavelength_grid - 4000) / 2000)
    unc = sens * 0.02
    return SensitivityFunction(
        wavelength=wavelength_grid,
        sensitivity=sens,
        uncertainty=unc,
        meta={"standard_star": "test_linear", "airmass_std": 1.2},
    )


@pytest.fixture
def known_flux(wavelength_grid):
    """A synthetic 'true' flux: smooth blackbody-like curve."""
    # Simple power-law: F ∝ λ^{-2}
    flux = 1e-12 * (5000 / wavelength_grid) ** 2
    return flux


@pytest.fixture
def observed_counts(wavelength_grid, flat_sensitivity, known_flux):
    """Observed counts = true_flux × sensitivity × exptime.

    With a flat sensitivity of 1e10, exptime=100s:
      counts = flux * 1e10 * 100
    """
    exptime = 100.0
    counts = known_flux * flat_sensitivity.sensitivity * exptime
    return counts, exptime


# ---------------------------------------------------------------------------
# SensitivityFunction tests
# ---------------------------------------------------------------------------

class TestSensitivityFunction:

    def test_creation_and_evaluate(self, wavelength_grid):
        """Basic creation and evaluation at the same wavelength grid."""
        sens_vals = np.ones(len(wavelength_grid)) * 5e9
        sf = SensitivityFunction(wavelength=wavelength_grid,
                                 sensitivity=sens_vals)
        result, unc = sf.evaluate(wavelength_grid)
        assert_allclose(result, 5e9)
        assert unc is None

    def test_evaluate_with_uncertainty(self, wavelength_grid):
        sens_vals = np.ones(len(wavelength_grid)) * 5e9
        unc_vals = np.ones(len(wavelength_grid)) * 1e8
        sf = SensitivityFunction(wavelength=wavelength_grid,
                                 sensitivity=sens_vals,
                                 uncertainty=unc_vals)
        result, unc = sf.evaluate(wavelength_grid)
        assert_allclose(result, 5e9)
        assert_allclose(unc, 1e8)

    def test_interpolation(self, wavelength_grid):
        """Evaluate at wavelengths between the grid points."""
        sens_vals = wavelength_grid * 1e6  # linear function
        sf = SensitivityFunction(wavelength=wavelength_grid,
                                 sensitivity=sens_vals)
        test_wave = np.array([4250.0, 4750.0])
        result, _ = sf.evaluate(test_wave)
        expected = test_wave * 1e6
        assert_allclose(result, expected, rtol=1e-3)

    def test_extrapolation_returns_nan(self, wavelength_grid):
        """Evaluating outside the wavelength range returns NaN."""
        sens_vals = np.ones(len(wavelength_grid)) * 1e10
        sf = SensitivityFunction(wavelength=wavelength_grid,
                                 sensitivity=sens_vals)
        result, _ = sf.evaluate(np.array([3000.0, 6000.0]))
        assert np.isnan(result[0])
        assert np.isnan(result[1])

    def test_fits_roundtrip(self, wavelength_grid, tmp_path):
        """Write to FITS and read back; values should match."""
        sens_vals = np.linspace(1e9, 2e9, len(wavelength_grid))
        unc_vals = sens_vals * 0.05
        sf = SensitivityFunction(
            wavelength=wavelength_grid,
            sensitivity=sens_vals,
            uncertainty=unc_vals,
            meta={"standard_star": "roundtrip_test"},
        )
        fpath = tmp_path / "test_sens.fits"
        sf.to_fits(fpath)

        sf2 = SensitivityFunction.from_fits(fpath)
        assert_allclose(sf2.wavelength, wavelength_grid)
        assert_allclose(sf2.sensitivity, sens_vals)
        assert_allclose(sf2.uncertainty, unc_vals)

    def test_from_callable(self, wavelength_grid):
        """Create from a callable function."""
        func = lambda w: 1e10 * np.ones_like(w)
        sf = SensitivityFunction.from_callable(func, wavelength_grid)
        result, _ = sf.evaluate(wavelength_grid)
        assert_allclose(result, 1e10)

    def test_too_few_valid_points_raises(self):
        """Sensitivity with < 2 valid points should raise."""
        with pytest.raises(ValueError, match="fewer than 2"):
            SensitivityFunction(wavelength=np.array([4000.0]),
                                sensitivity=np.array([1e10]))


# ---------------------------------------------------------------------------
# _unpack_observed tests
# ---------------------------------------------------------------------------

class TestUnpackObserved:

    def test_from_spectrum1d(self, wavelength_grid, known_flux):
        """Unpack a Spectrum1D object."""
        spec = Spectrum1D(
            spectral_axis=wavelength_grid * u.AA,
            flux=known_flux * u.ct,
            uncertainty=StdDevUncertainty(known_flux * 0.1),
            mask=np.zeros(len(wavelength_grid), dtype=bool),
            meta={"exptime": 10.0, "airmass": 1.5},
        )
        wave, flux, unc, mask, meta = _unpack_observed(spec)
        assert_allclose(wave, wavelength_grid)
        assert_allclose(flux, known_flux)
        assert meta["exptime"] == 10.0

    def test_from_dict(self, wavelength_grid, known_flux):
        """Unpack from a dictionary."""
        d = {
            "wavelength": wavelength_grid,
            "flux": known_flux,
            "uncertainty": known_flux * 0.05,
            "mask": np.zeros(len(wavelength_grid), dtype=bool),
            "meta": {"exptime": 50.0},
        }
        wave, flux, unc, mask, meta = _unpack_observed(d)
        assert_allclose(wave, wavelength_grid)
        assert meta["exptime"] == 50.0

    def test_from_tuple(self, wavelength_grid, known_flux):
        """Unpack from a (wave, flux) tuple."""
        wave, flux, unc, mask, meta = _unpack_observed(
            (wavelength_grid, known_flux))
        assert_allclose(wave, wavelength_grid)
        assert_allclose(flux, known_flux)
        # Default uncertainty: sqrt(max(|flux|, 1.0))
        expected_unc = np.sqrt(np.maximum(np.abs(known_flux), 1.0))
        assert_allclose(unc, expected_unc)

    def test_spectrum1d_no_uncertainty(self, wavelength_grid, known_flux):
        """Spectrum1D without uncertainty gets Poisson estimate."""
        spec = Spectrum1D(
            spectral_axis=wavelength_grid * u.AA,
            flux=known_flux * u.ct,
        )
        wave, flux, unc, mask, meta = _unpack_observed(spec)
        expected_unc = np.sqrt(np.maximum(np.abs(known_flux), 1.0))
        assert_allclose(unc, expected_unc)


# ---------------------------------------------------------------------------
# Core calibration tests
# ---------------------------------------------------------------------------

class TestApplySensitivity:

    def test_flat_sensitivity_recovers_flux(self, wavelength_grid,
                                            flat_sensitivity, known_flux,
                                            observed_counts):
        """With a flat S(λ), dividing counts by S×t should recover flux."""
        counts, exptime = observed_counts
        spec = (wavelength_grid, counts)
        result = apply_sensitivity(spec, flat_sensitivity, exptime=exptime)

        good = ~result.mask
        assert_allclose(result.flux[good], known_flux[good], rtol=1e-10)

    def test_linear_sensitivity(self, wavelength_grid, linear_sensitivity,
                                known_flux):
        """With a linear S(λ), verify correct colour-dependent calibration."""
        exptime = 60.0
        counts = known_flux * linear_sensitivity.sensitivity * exptime
        spec = (wavelength_grid, counts)

        result = apply_sensitivity(spec, linear_sensitivity, exptime=exptime,
                                   airmass_obs=1.2, airmass_std=1.2)
        good = ~result.mask
        assert_allclose(result.flux[good], known_flux[good], rtol=1e-10)

    def test_exptime_from_metadata(self, wavelength_grid, flat_sensitivity,
                                   known_flux):
        """Exposure time read from Spectrum1D.meta if not provided."""
        exptime = 30.0
        counts = known_flux * flat_sensitivity.sensitivity * exptime
        spec = Spectrum1D(
            spectral_axis=wavelength_grid * u.AA,
            flux=counts * u.ct,
            meta={"exptime": exptime},
        )
        result = apply_sensitivity(spec, flat_sensitivity)
        good = ~result.mask
        assert_allclose(result.flux[good], known_flux[good], rtol=1e-10)

    def test_missing_exptime_warns(self, wavelength_grid, flat_sensitivity):
        """Missing exptime triggers a warning and defaults to 1."""
        counts = np.ones(len(wavelength_grid)) * 100
        spec = (wavelength_grid, counts)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = apply_sensitivity(spec, flat_sensitivity)
            assert any("Exposure time" in str(warn.message) for warn in w)

    def test_zero_exptime_raises(self, wavelength_grid, flat_sensitivity):
        """Exposure time of zero should raise ValueError."""
        counts = np.ones(len(wavelength_grid)) * 100
        spec = (wavelength_grid, counts)
        with pytest.raises(ValueError, match="positive"):
            apply_sensitivity(spec, flat_sensitivity, exptime=0.0)

    def test_masked_pixels(self, wavelength_grid, flat_sensitivity, known_flux):
        """Masked pixels in the input should be NaN in the output."""
        exptime = 10.0
        counts = known_flux * flat_sensitivity.sensitivity * exptime
        mask = np.zeros(len(wavelength_grid), dtype=bool)
        mask[0:10] = True  # mask first 10 pixels

        spec = Spectrum1D(
            spectral_axis=wavelength_grid * u.AA,
            flux=counts * u.ct,
            mask=mask,
            meta={"exptime": exptime},
        )
        result = apply_sensitivity(spec, flat_sensitivity)
        assert np.all(np.isnan(result.flux[:10]))
        assert np.all(np.isfinite(result.flux[10:]))

    def test_out_of_range_sensitivity_masked(self, known_flux):
        """Pixels outside the sensitivity wavelength range are flagged."""
        # Sensitivity covers 4200-4800, but observation is 4000-5000
        sens_wave = np.linspace(4200, 4800, 300)
        sens_vals = np.ones(300) * 1e10
        sf = SensitivityFunction(wavelength=sens_wave, sensitivity=sens_vals)

        obs_wave = np.linspace(4000, 5000, 500)
        counts = known_flux[:500] * 1e10 * 10
        spec = (obs_wave, counts)

        result = apply_sensitivity(spec, sf, exptime=10.0)
        # Pixels outside [4200, 4800] should be masked
        outside = (obs_wave < 4200) | (obs_wave > 4800)
        assert np.all(result.mask[outside])

    def test_uncertainty_propagation(self, wavelength_grid, flat_sensitivity):
        """Verify uncertainty is propagated through the division."""
        exptime = 100.0
        counts = np.ones(len(wavelength_grid)) * 1e6
        counts_unc = counts * 0.1  # 10% photon noise
        sens_unc = flat_sensitivity.uncertainty  # 1% sensitivity error

        spec = Spectrum1D(
            spectral_axis=wavelength_grid * u.AA,
            flux=counts * u.ct,
            uncertainty=StdDevUncertainty(counts_unc),
            meta={"exptime": exptime},
        )
        result = apply_sensitivity(spec, flat_sensitivity)

        # Expected fractional uncertainty: sqrt(0.1^2 + 0.01^2) ≈ 0.1005
        expected_frac = np.sqrt(0.1**2 + 0.01**2)
        actual_frac = result.uncertainty / np.abs(result.flux)
        good = ~result.mask
        assert_allclose(actual_frac[good], expected_frac, rtol=0.01)

    def test_provenance_metadata(self, wavelength_grid, flat_sensitivity):
        """Calibration result carries provenance metadata."""
        counts = np.ones(len(wavelength_grid)) * 100
        spec = (wavelength_grid, counts)
        result = apply_sensitivity(spec, flat_sensitivity, exptime=10.0)

        assert result.meta["flux_unit"] == "erg/s/cm2/A"
        assert result.meta["calibration_step"] == "step_07_flux_calibration"
        assert result.meta["slit_loss_corrected"] is False

    def test_to_spectrum1d(self, wavelength_grid, flat_sensitivity, known_flux,
                           observed_counts):
        """CalibrationResult.to_spectrum1d() produces valid Spectrum1D."""
        counts, exptime = observed_counts
        result = apply_sensitivity(
            (wavelength_grid, counts), flat_sensitivity, exptime=exptime)
        spec = result.to_spectrum1d()
        assert spec.flux.unit == FLUX_UNIT
        assert spec.spectral_axis.unit == u.AA


# ---------------------------------------------------------------------------
# Atmospheric extinction correction tests
# ---------------------------------------------------------------------------

class TestExtinctionCorrection:

    def test_equal_airmass_no_correction(self, wavelength_grid,
                                         flat_sensitivity, known_flux):
        """If science and standard airmass are equal, no correction."""
        exptime = 10.0
        counts = known_flux * flat_sensitivity.sensitivity * exptime
        spec = (wavelength_grid, counts)

        result = apply_sensitivity(
            spec, flat_sensitivity, exptime=exptime,
            airmass_obs=1.5, airmass_std=1.5,
        )
        good = ~result.mask
        assert_allclose(result.flux[good], known_flux[good], rtol=1e-10)

    def test_different_airmass_applies_correction(self, wavelength_grid):
        """Higher science airmass → correction dims the blue more."""
        sens_vals = np.ones(len(wavelength_grid)) * 1e10
        sf = SensitivityFunction(
            wavelength=wavelength_grid, sensitivity=sens_vals,
            meta={"airmass_std": 1.0},
        )

        # True flux is flat = 1e-13 everywhere
        true_flux = np.ones(len(wavelength_grid)) * 1e-13
        exptime = 1.0

        # The standard was observed at airmass=1.0, science at airmass=2.0.
        # The extinction dims the science counts; the correction should
        # restore the true flux.
        k = default_extinction_curve(wavelength_grid)
        # Science counts are dimmed by extinction relative to std
        atm_dimming = 10.0 ** (-0.4 * (2.0 - 1.0) * k)
        counts = true_flux * sens_vals * exptime * atm_dimming

        result = apply_sensitivity(
            (wavelength_grid, counts), sf, exptime=exptime,
            airmass_obs=2.0, airmass_std=1.0,
        )
        good = ~result.mask
        assert_allclose(result.flux[good], true_flux[good], rtol=1e-6)

    def test_missing_airmass_warns(self, wavelength_grid, flat_sensitivity):
        """Missing airmass triggers a warning."""
        counts = np.ones(len(wavelength_grid)) * 100
        spec = {"wavelength": wavelength_grid, "flux": counts}

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # airmass_std is set in flat_sensitivity, obs is not
            apply_sensitivity(spec, flat_sensitivity, exptime=1.0,
                              airmass_std=1.0)
            airmass_warns = [
                x for x in w if "airmass" in str(x.message).lower()
            ]
            assert len(airmass_warns) >= 1

    def test_custom_extinction_curve_callable(self, wavelength_grid):
        """Custom extinction curve as a callable."""
        # Constant extinction: 0.5 mag/airmass everywhere
        k_const = lambda w: np.full_like(w, 0.5)
        sens_vals = np.ones(len(wavelength_grid)) * 1e10
        sf = SensitivityFunction(wavelength=wavelength_grid,
                                 sensitivity=sens_vals,
                                 meta={"airmass_std": 1.0})
        true_flux = np.ones(len(wavelength_grid)) * 1e-13
        # Science at airmass 1.5, std at 1.0 → Δ = 0.5
        # Correction factor = 10^(0.4 * 0.5 * 0.5) = 10^0.1
        exptime = 1.0
        atm_dimming = 10.0 ** (-0.4 * 0.5 * 0.5)
        counts = true_flux * sens_vals * exptime * atm_dimming

        result = apply_sensitivity(
            (wavelength_grid, counts), sf, exptime=exptime,
            airmass_obs=1.5, airmass_std=1.0,
            extinction_curve=k_const,
        )
        good = ~result.mask
        assert_allclose(result.flux[good], true_flux[good], rtol=1e-8)

    def test_custom_extinction_curve_tuple(self, wavelength_grid):
        """Custom extinction curve as (wave, k) tuple."""
        ext_wave = np.array([3000., 5000., 8000.])
        ext_k = np.array([0.5, 0.5, 0.5])  # flat 0.5

        resolved = _resolve_extinction_curve((ext_wave, ext_k))
        k_at_4500 = resolved(np.array([4500.0]))
        assert_allclose(k_at_4500, 0.5, atol=0.01)

    def test_default_extinction_curve_shape(self):
        """Default extinction curve has reasonable values."""
        wave = np.linspace(3500, 8000, 100)
        k = default_extinction_curve(wave)
        # Blue end should have higher extinction than red
        assert k[0] > k[-1]
        # Values should be positive
        assert np.all(k > 0)
        # Reasonable range for optical
        assert np.all(k < 2.0)


# ---------------------------------------------------------------------------
# Self-calibration check tests
# ---------------------------------------------------------------------------

class TestSelfCalibration:

    def test_perfect_recovery(self, wavelength_grid, flat_sensitivity,
                              known_flux, observed_counts):
        """Applying S to the standard should recover the reference exactly."""
        counts, exptime = observed_counts
        report = self_calibration_check(
            (wavelength_grid, counts),
            flat_sensitivity,
            known_flux,
            reference_wavelength=wavelength_grid,
            exptime=exptime,
            airmass_obs=1.0,
            airmass_std=1.0,
        )
        assert isinstance(report, SelfCalibrationReport)
        good = ~report.mask
        assert report.rms_residual < 1e-8
        assert report.median_residual < 1e-8

    def test_noisy_data_residuals(self, wavelength_grid):
        """With noise, residuals should be on the order of the noise level."""
        # Use high counts so noise is a small fraction of signal
        sens_vals = np.ones(len(wavelength_grid)) * 1e10
        sf = SensitivityFunction(
            wavelength=wavelength_grid, sensitivity=sens_vals,
            meta={"airmass_std": 1.0},
        )
        true_flux = np.ones(len(wavelength_grid)) * 1e-10  # bright source
        exptime = 100.0
        counts = true_flux * sens_vals * exptime  # = 1e2, decent SNR

        rng = np.random.default_rng(42)
        noisy_counts = counts + rng.normal(0, np.sqrt(counts))

        report = self_calibration_check(
            (wavelength_grid, noisy_counts),
            sf,
            true_flux,
            reference_wavelength=wavelength_grid,
            exptime=exptime,
            airmass_obs=1.0,
            airmass_std=1.0,
        )
        # Residuals should be small (~10% for SNR~10) but not zero
        assert report.rms_residual < 0.2  # < 20%
        assert report.rms_residual > 1e-10  # not perfect

    def test_reference_as_spectrum1d(self, wavelength_grid, flat_sensitivity,
                                    known_flux, observed_counts):
        """Reference flux can be passed as a Spectrum1D."""
        counts, exptime = observed_counts
        ref_spec = Spectrum1D(
            spectral_axis=wavelength_grid * u.AA,
            flux=known_flux * FLUX_UNIT,
        )
        report = self_calibration_check(
            (wavelength_grid, counts),
            flat_sensitivity,
            ref_spec,
            exptime=exptime,
            airmass_obs=1.0,
            airmass_std=1.0,
        )
        good = ~report.mask
        assert report.rms_residual < 1e-8

    def test_missing_reference_wavelength_raises(self, wavelength_grid,
                                                  flat_sensitivity, known_flux,
                                                  observed_counts):
        """Passing a plain array without wavelength should raise."""
        counts, exptime = observed_counts
        with pytest.raises(ValueError, match="reference_wavelength"):
            self_calibration_check(
                (wavelength_grid, counts), flat_sensitivity,
                known_flux,  # no reference_wavelength
                exptime=exptime,
            )


# ---------------------------------------------------------------------------
# FITS and CSV output round-trip tests
# ---------------------------------------------------------------------------

class TestOutputRoundTrip:

    @pytest.fixture
    def sample_result(self, wavelength_grid, known_flux):
        """A sample CalibrationResult for testing I/O."""
        return CalibrationResult(
            wavelength=wavelength_grid,
            flux=known_flux,
            uncertainty=known_flux * 0.05,
            mask=np.zeros(len(wavelength_grid), dtype=bool),
            meta={
                "flux_unit": "erg/s/cm2/A",
                "standard_star": "test_star",
                "exptime_applied": 100.0,
                "calibration_step": "step_07",
            }
        )

    def test_fits_roundtrip(self, sample_result, tmp_path):
        """Write and read back a FITS file; data should match."""
        fpath = tmp_path / "test_cal.fits"
        write_calibrated_fits(sample_result, fpath)

        loaded = read_calibrated_fits(fpath)
        assert_allclose(loaded.wavelength, sample_result.wavelength)
        assert_allclose(loaded.flux, sample_result.flux)
        assert_allclose(loaded.uncertainty, sample_result.uncertainty)
        assert_array_equal(loaded.mask, sample_result.mask)

    def test_fits_nonuniform_grid(self, tmp_path):
        """Non-uniform wavelength grid round-trips via WAVELENGTH extension."""
        wave = np.array([4000, 4100, 4300, 4700, 5000], dtype=np.float64)
        flux = np.array([1e-13, 2e-13, 1.5e-13, 1e-13, 0.8e-13])
        result = CalibrationResult(
            wavelength=wave, flux=flux,
            uncertainty=flux * 0.1,
            mask=np.zeros(5, dtype=bool),
            meta={"flux_unit": "erg/s/cm2/A"},
        )
        fpath = tmp_path / "nonuniform.fits"
        write_calibrated_fits(result, fpath)
        loaded = read_calibrated_fits(fpath)
        assert_allclose(loaded.wavelength, wave)

    def test_csv_output(self, sample_result, tmp_path):
        """CSV output has correct number of rows and columns."""
        fpath = tmp_path / "test_cal.csv"
        write_calibrated_csv(sample_result, fpath)

        # Read back
        lines = fpath.read_text().strip().split("\n")
        # Header lines start with '#'
        data_lines = [l for l in lines if not l.startswith("#")]
        assert len(data_lines) == len(sample_result.wavelength)

        # Parse first data line
        vals = data_lines[0].split(",")
        assert len(vals) == 4  # wavelength, flux, uncertainty, mask

    def test_fits_mask_preserved(self, wavelength_grid, known_flux, tmp_path):
        """Mask is preserved through FITS round-trip."""
        mask = np.zeros(len(wavelength_grid), dtype=bool)
        mask[100:110] = True
        result = CalibrationResult(
            wavelength=wavelength_grid, flux=known_flux,
            uncertainty=known_flux * 0.1, mask=mask, meta={},
        )
        fpath = tmp_path / "masked.fits"
        write_calibrated_fits(result, fpath)
        loaded = read_calibrated_fits(fpath)
        assert_array_equal(loaded.mask, mask)


# ---------------------------------------------------------------------------
# Batch mode tests
# ---------------------------------------------------------------------------

class TestBatchMode:

    def test_basic_batch(self, wavelength_grid, flat_sensitivity, known_flux):
        """Batch calibration of multiple spectra."""
        exptime = 10.0
        counts = known_flux * flat_sensitivity.sensitivity * exptime
        spectra = [(wavelength_grid, counts)] * 3

        results = calibrate_batch(
            spectra, flat_sensitivity,
            exptimes=[exptime, exptime, exptime],
            airmasses=[1.0, 1.0, 1.0],
            airmass_std=1.0,
        )
        assert len(results) == 3
        for r in results:
            good = ~r.mask
            assert_allclose(r.flux[good], known_flux[good], rtol=1e-10)

    def test_batch_different_airmasses(self, wavelength_grid):
        """Batch with per-spectrum airmasses."""
        sens_vals = np.ones(len(wavelength_grid)) * 1e10
        sf = SensitivityFunction(
            wavelength=wavelength_grid, sensitivity=sens_vals,
            meta={"airmass_std": 1.0},
        )

        true_flux = np.ones(len(wavelength_grid)) * 1e-13
        airmasses = [1.0, 1.5, 2.0]
        k = default_extinction_curve(wavelength_grid)

        spectra = []
        for am in airmasses:
            dimming = 10.0 ** (-0.4 * (am - 1.0) * k)
            counts = true_flux * sens_vals * 1.0 * dimming
            spectra.append((wavelength_grid, counts))

        results = calibrate_batch(
            spectra, sf,
            exptimes=[1.0, 1.0, 1.0],
            airmasses=airmasses,
            airmass_std=1.0,
        )
        for r in results:
            good = ~r.mask
            assert_allclose(r.flux[good], true_flux[good], rtol=1e-5)

    def test_batch_with_output(self, wavelength_grid, flat_sensitivity,
                               known_flux, tmp_path):
        """Batch mode writes output files."""
        exptime = 10.0
        counts = known_flux * flat_sensitivity.sensitivity * exptime
        spectra = [(wavelength_grid, counts)] * 2

        calibrate_batch(
            spectra, flat_sensitivity,
            exptimes=[exptime, exptime],
            output_dir=tmp_path / "batch_out",
            output_format="both",
        )
        assert (tmp_path / "batch_out" / "calibrated_000.fits").exists()
        assert (tmp_path / "batch_out" / "calibrated_000.csv").exists()
        assert (tmp_path / "batch_out" / "calibrated_001.fits").exists()

    def test_batch_length_mismatch_raises(self, wavelength_grid,
                                           flat_sensitivity, known_flux):
        """Mismatched exptimes/airmasses length should raise."""
        counts = known_flux * flat_sensitivity.sensitivity * 10
        spectra = [(wavelength_grid, counts)] * 3

        with pytest.raises(ValueError, match="Length of exptimes"):
            calibrate_batch(spectra, flat_sensitivity, exptimes=[10.0, 10.0])

        with pytest.raises(ValueError, match="Length of airmasses"):
            calibrate_batch(spectra, flat_sensitivity,
                            airmasses=[1.0, 1.0])


# ---------------------------------------------------------------------------
# Per-segment calibration tests
# ---------------------------------------------------------------------------

class TestPerSegment:

    def test_per_segment_basic(self, wavelength_grid, flat_sensitivity,
                               known_flux):
        """Per-segment calibration processes each segment independently."""
        exptime = 10.0
        # Create 3 sub-segments
        n = len(wavelength_grid) // 3
        segments = []
        for i in range(3):
            sl = slice(i * n, (i + 1) * n)
            counts = known_flux[sl] * flat_sensitivity.sensitivity[sl] * exptime
            segments.append((wavelength_grid[sl], counts))

        results = apply_sensitivity_per_segment(
            segments, flat_sensitivity, exptime=exptime,
            airmass_obs=1.0, airmass_std=1.0,
        )
        assert len(results) == 3
        for i, r in enumerate(results):
            assert r.meta["segment_index"] == i


# ---------------------------------------------------------------------------
# Slit-loss documentation tests
# ---------------------------------------------------------------------------

class TestSlitLoss:

    def test_documentation_exists(self):
        """Slit-loss documentation string is non-empty and informative."""
        doc = get_slit_loss_documentation()
        assert len(doc) > 100
        assert "slit" in doc.lower()
        assert "JJMO" in doc

    def test_flag_slit_loss(self, wavelength_grid, known_flux):
        """flag_slit_loss_systematic sets metadata correctly."""
        result = CalibrationResult(
            wavelength=wavelength_grid, flux=known_flux,
            uncertainty=known_flux * 0.1,
            mask=np.zeros(len(wavelength_grid), dtype=bool),
            meta={},
        )
        flag_slit_loss_systematic(result)
        assert result.meta["slit_loss_corrected"] is False
        assert "slit" in result.meta["slit_loss_warning"].lower()


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_all_masked_input(self, wavelength_grid, flat_sensitivity):
        """All pixels masked → all output NaN, no crash."""
        counts = np.ones(len(wavelength_grid)) * 100
        mask = np.ones(len(wavelength_grid), dtype=bool)
        spec = {"wavelength": wavelength_grid, "flux": counts, "mask": mask}
        result = apply_sensitivity(spec, flat_sensitivity, exptime=1.0)
        assert np.all(np.isnan(result.flux))

    def test_negative_counts(self, wavelength_grid, flat_sensitivity):
        """Negative counts (sky-subtracted) should still calibrate."""
        counts = np.ones(len(wavelength_grid)) * -50
        spec = (wavelength_grid, counts)
        result = apply_sensitivity(spec, flat_sensitivity, exptime=1.0)
        good = ~result.mask
        # Should be negative flux (valid in sky-subtracted spectra)
        assert np.all(result.flux[good] < 0)

    def test_single_pixel_spectrum(self):
        """Single-pixel spectrum should work (edge case)."""
        # Need at least 2 points for sensitivity interpolation
        sf = SensitivityFunction(
            wavelength=np.array([4000.0, 5000.0]),
            sensitivity=np.array([1e10, 1e10]),
        )
        spec = (np.array([4500.0]), np.array([1000.0]))
        result = apply_sensitivity(spec, sf, exptime=1.0)
        assert len(result.flux) == 1
        assert np.isfinite(result.flux[0])
