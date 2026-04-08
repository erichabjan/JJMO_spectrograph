"""
test_reference.py -- Tests for Step 5: Reference Spectrum Selection & Loading

Tests are organized into:
  - Unit tests (fast, no network): parameter lookup, extinction math,
    resampling, convolution, masking, plotting
  - Integration tests (require network): CALSPEC download, expecto model
    download, atmospheric extinction loading

Integration tests are marked with @pytest.mark.network and skipped by default.
Run them explicitly: pytest test_reference.py -m network
"""

import numpy as np
import pytest
import warnings

import astropy.units as u

from jjmo_fluxcal import reference


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def simple_wavelength():
    """A simple optical wavelength grid (4000-7000 A, 1 A spacing)."""
    return np.arange(4000.0, 7001.0, 1.0)


@pytest.fixture
def simple_flux(simple_wavelength):
    """A simple smooth test flux (blackbody-like curve)."""
    wave = simple_wavelength
    # Simple Planck-like function (arbitrary units)
    T = 9000.0  # K
    h, c, k = 6.626e-27, 3.0e10, 1.381e-16  # cgs
    wave_cm = wave * 1e-8
    flux = 2 * h * c**2 / wave_cm**5 / (np.exp(h * c / (wave_cm * k * T)) - 1)
    return flux


@pytest.fixture
def high_res_wavelength():
    """High-resolution wavelength grid (0.01 A spacing over 100 A)."""
    return np.arange(5000.0, 5100.0, 0.01)


@pytest.fixture
def high_res_flux(high_res_wavelength):
    """High-res flux with a narrow absorption line at 5050 A."""
    wave = high_res_wavelength
    continuum = np.ones_like(wave) * 1000.0
    # Gaussian absorption line at 5050 A, sigma = 0.5 A
    line = 500.0 * np.exp(-0.5 * ((wave - 5050.0) / 0.5)**2)
    return continuum - line


# ============================================================================
# 5.3 -- Stellar parameter database tests
# ============================================================================

class TestStellarParameterDB:
    """Tests for the stellar parameter lookup system."""

    def test_get_sirius_by_name(self):
        params = reference.get_stellar_parameters("sirius")
        assert params["teff"] == 9940
        assert params["logg"] == 4.33
        assert params["feh"] == 0.5
        assert params["calspec_file"] == "sirius_stis_001.fits"

    def test_get_sirius_by_alias(self):
        params = reference.get_stellar_parameters("alpha_cma")
        assert params["teff"] == 9940

    def test_get_sirius_case_insensitive(self):
        params = reference.get_stellar_parameters("SIRIUS")
        assert params["teff"] == 9940

    def test_get_betelgeuse_by_alias(self):
        params = reference.get_stellar_parameters("alpha ori")
        assert params["teff"] == 3600
        assert params["calspec_file"] is None

    def test_get_vega(self):
        params = reference.get_stellar_parameters("vega")
        assert params["teff"] == 9550
        assert params["calspec_file"] == "alpha_lyr_stis_011.fits"

    def test_all_standard_stars_have_required_keys(self):
        required = {"teff", "logg", "feh", "ebv", "calspec_file",
                     "spectral_type", "notes"}
        for name, entry in reference.STANDARD_STAR_DB.items():
            for key in required:
                assert key in entry, f"Star '{name}' missing key '{key}'"

    def test_parameter_override(self):
        params = reference.get_stellar_parameters("sirius", teff=10000, logg=4.5)
        assert params["teff"] == 10000
        assert params["logg"] == 4.5
        # Non-overridden values preserved
        assert params["feh"] == 0.5

    def test_unknown_star_raises(self):
        with pytest.raises(ValueError, match="not found"):
            reference.get_stellar_parameters("proxima_centauri")

    def test_list_standard_stars(self):
        stars = reference.list_standard_stars()
        assert "sirius" in stars
        assert "vega" in stars
        assert "betelgeuse" in stars
        assert "teff" in stars["sirius"]
        assert "aliases" in stars["sirius"]

    def test_resolve_star_name_returns_none_for_unknown(self):
        assert reference._resolve_star_name("not_a_star") is None

    def test_all_aliases_resolve(self):
        """Every alias in the DB should resolve to its parent key."""
        for key, entry in reference.STANDARD_STAR_DB.items():
            for alias in entry.get("aliases", []):
                resolved = reference._resolve_star_name(alias)
                assert resolved == key, (
                    f"Alias '{alias}' resolved to '{resolved}' "
                    f"instead of '{key}'"
                )


# ============================================================================
# 5.4 -- Extinction correction tests
# ============================================================================

class TestInterstellarExtinction:
    """Tests for interstellar extinction correction."""

    def test_dereddening_increases_blue_flux(self, simple_wavelength, simple_flux):
        """Dereddening should increase flux, especially in the blue."""
        corrected = reference.apply_interstellar_extinction(
            simple_wavelength, simple_flux, ebv=0.1
        )
        # Dereddened flux should be >= original everywhere
        assert np.all(corrected >= simple_flux * 0.999)  # small tolerance
        # Blue end should be boosted more than red end
        blue_ratio = corrected[0] / simple_flux[0]
        red_ratio = corrected[-1] / simple_flux[-1]
        assert blue_ratio > red_ratio

    def test_zero_ebv_is_identity(self, simple_wavelength, simple_flux):
        corrected = reference.apply_interstellar_extinction(
            simple_wavelength, simple_flux, ebv=0.0
        )
        np.testing.assert_allclose(corrected, simple_flux, rtol=1e-10)

    def test_reddening_is_inverse_of_dereddening(self, simple_wavelength,
                                                  simple_flux):
        reddened = reference.apply_interstellar_reddening(
            simple_wavelength, simple_flux, ebv=0.2
        )
        recovered = reference.apply_interstellar_extinction(
            simple_wavelength, reddened, ebv=0.2
        )
        np.testing.assert_allclose(recovered, simple_flux, rtol=1e-6)

    def test_fitzpatrick99_law(self, simple_wavelength, simple_flux):
        corrected = reference.apply_interstellar_extinction(
            simple_wavelength, simple_flux, ebv=0.1, law="fitzpatrick99"
        )
        assert np.all(corrected >= simple_flux * 0.999)

    def test_ccm89_law(self, simple_wavelength, simple_flux):
        corrected = reference.apply_interstellar_extinction(
            simple_wavelength, simple_flux, ebv=0.1, law="ccm89"
        )
        assert np.all(corrected >= simple_flux * 0.999)

    def test_unknown_law_raises(self, simple_wavelength, simple_flux):
        with pytest.raises(ValueError, match="Unknown extinction law"):
            reference.apply_interstellar_extinction(
                simple_wavelength, simple_flux, ebv=0.1, law="nonexistent"
            )

    def test_rv_affects_correction(self, simple_wavelength, simple_flux):
        """Different R_V should give different corrections."""
        corr_31 = reference.apply_interstellar_extinction(
            simple_wavelength, simple_flux, ebv=0.1, rv=3.1
        )
        corr_50 = reference.apply_interstellar_extinction(
            simple_wavelength, simple_flux, ebv=0.1, rv=5.0
        )
        assert not np.allclose(corr_31, corr_50, rtol=1e-3)


class TestAtmosphericExtinction:
    """Tests for atmospheric extinction correction."""

    def test_load_kpno_extinction(self):
        wave, ext = reference.load_atmospheric_extinction("kpno")
        assert len(wave) > 10
        assert len(ext) == len(wave)
        # Extinction should be positive (magnitudes per airmass)
        assert np.all(ext >= 0)
        # Wavelength range should cover optical
        assert wave.min() <= 4000.0
        assert wave.max() >= 7000.0

    def test_unknown_observatory_raises(self):
        with pytest.raises(ValueError, match="Unknown observatory"):
            reference.load_atmospheric_extinction("nonexistent_obs")

    def test_correction_increases_flux(self, simple_wavelength, simple_flux):
        corrected = reference.correct_atmospheric_extinction(
            simple_wavelength, simple_flux, airmass=1.5, observatory="kpno"
        )
        # Atmospheric correction should increase flux (atmosphere dims)
        assert np.all(corrected >= simple_flux * 0.999)

    def test_airmass_one_minimal_correction(self, simple_wavelength, simple_flux):
        """Airmass=1 should give smaller corrections than airmass=2."""
        corr_1 = reference.correct_atmospheric_extinction(
            simple_wavelength, simple_flux, airmass=1.0
        )
        corr_2 = reference.correct_atmospheric_extinction(
            simple_wavelength, simple_flux, airmass=2.0
        )
        # Higher airmass -> larger correction
        ratio_1 = np.mean(corr_1 / simple_flux)
        ratio_2 = np.mean(corr_2 / simple_flux)
        assert ratio_2 > ratio_1

    def test_invalid_airmass_raises(self, simple_wavelength, simple_flux):
        with pytest.raises(ValueError, match="positive"):
            reference.correct_atmospheric_extinction(
                simple_wavelength, simple_flux, airmass=-1.0
            )


class TestCombinedCorrection:
    """Tests for the unified correction pipeline."""

    def test_no_corrections(self, simple_wavelength, simple_flux):
        corrected, applied = reference.correct_observed_spectrum(
            simple_wavelength, simple_flux
        )
        np.testing.assert_allclose(corrected, simple_flux)
        assert "No extinction corrections" in applied[0]

    def test_atmospheric_only(self, simple_wavelength, simple_flux):
        corrected, applied = reference.correct_observed_spectrum(
            simple_wavelength, simple_flux, airmass=1.5
        )
        assert len(applied) == 1
        assert "Atmospheric" in applied[0]

    def test_interstellar_only(self, simple_wavelength, simple_flux):
        corrected, applied = reference.correct_observed_spectrum(
            simple_wavelength, simple_flux, ebv=0.1
        )
        assert len(applied) == 1
        assert "Interstellar" in applied[0]

    def test_both_corrections(self, simple_wavelength, simple_flux):
        corrected, applied = reference.correct_observed_spectrum(
            simple_wavelength, simple_flux, airmass=1.5, ebv=0.1
        )
        assert len(applied) == 2
        assert "Atmospheric" in applied[0]
        assert "Interstellar" in applied[1]


# ============================================================================
# 5.5 -- Reference spectrum preparation tests
# ============================================================================

class TestResampling:
    """Tests for flux-conserving resampling."""

    def test_resample_preserves_total_flux_approx(self):
        """Resampling a flat spectrum should preserve the mean flux."""
        old_wave = np.linspace(4000.0, 7000.0, 3000)
        old_flux = np.ones(3000) * 100.0
        new_wave = np.linspace(4500.0, 6500.0, 1000)

        resampled, unc = reference.resample_to_observed(
            old_wave, old_flux, new_wave
        )
        assert unc is None
        np.testing.assert_allclose(resampled, 100.0, rtol=0.01)

    def test_resample_with_uncertainty(self):
        old_wave = np.linspace(4000.0, 7000.0, 1000)
        old_flux = np.ones(1000) * 100.0
        old_unc = np.ones(1000) * 5.0
        new_wave = np.linspace(4500.0, 6500.0, 500)

        resampled, unc = reference.resample_to_observed(
            old_wave, old_flux, new_wave, ref_uncertainty=old_unc
        )
        assert unc is not None
        assert len(unc) == len(new_wave)

    def test_resample_handles_unsorted_input(self):
        """spectres requires sorted input; our wrapper should handle unsorted."""
        old_wave = np.linspace(7000.0, 4000.0, 1000)  # descending
        old_flux = np.ones(1000) * 100.0
        new_wave = np.linspace(4500.0, 6500.0, 500)

        resampled, _ = reference.resample_to_observed(
            old_wave, old_flux, new_wave
        )
        np.testing.assert_allclose(resampled, 100.0, rtol=0.01)


class TestConvolution:
    """Tests for resolution degradation via convolution."""

    def test_convolution_broadens_line(self, high_res_wavelength, high_res_flux):
        """Convolution should broaden a narrow absorption line."""
        convolved = reference.convolve_to_resolution(
            high_res_wavelength, high_res_flux, target_fwhm_aa=5.0
        )
        # The absorption line minimum should be shallower after convolution
        orig_min = np.min(high_res_flux)
        conv_min = np.min(convolved)
        assert conv_min > orig_min

    def test_convolution_preserves_continuum(self, high_res_wavelength,
                                             high_res_flux):
        """Convolution of a flat spectrum should return the flat spectrum."""
        flat = np.ones_like(high_res_wavelength) * 1000.0
        convolved = reference.convolve_to_resolution(
            high_res_wavelength, flat, target_fwhm_aa=5.0
        )
        # Interior points should be ~unchanged (edges have boundary effects)
        interior = slice(50, -50)
        np.testing.assert_allclose(
            convolved[interior], flat[interior], rtol=1e-6
        )

    def test_zero_fwhm_raises(self, high_res_wavelength, high_res_flux):
        with pytest.raises(ValueError, match="positive"):
            reference.convolve_to_resolution(
                high_res_wavelength, high_res_flux, target_fwhm_aa=-1.0
            )

    def test_small_fwhm_no_convolution(self, high_res_wavelength,
                                        high_res_flux):
        """If FWHM < pixel scale, convolution should be skipped."""
        convolved = reference.convolve_to_resolution(
            high_res_wavelength, high_res_flux, target_fwhm_aa=0.001
        )
        np.testing.assert_allclose(convolved, high_res_flux)


class TestFeatureMask:
    """Tests for the feature masking system."""

    def test_balmer_lines_masked(self, simple_wavelength):
        mask = reference.build_feature_mask(
            simple_wavelength, mask_balmer=True, mask_telluric=False
        )
        # H-alpha (6562.8 A) should be masked
        idx_ha = np.argmin(np.abs(simple_wavelength - 6562.8))
        assert mask[idx_ha] == True

        # H-beta (4861.3 A) should be masked
        idx_hb = np.argmin(np.abs(simple_wavelength - 4861.3))
        assert mask[idx_hb] == True

    def test_telluric_bands_masked(self, simple_wavelength):
        mask = reference.build_feature_mask(
            simple_wavelength, mask_balmer=False, mask_telluric=True
        )
        # O2 B-band (6860-6880 A) should be masked
        idx_o2 = np.argmin(np.abs(simple_wavelength - 6870.0))
        assert mask[idx_o2] == True

    def test_no_masking(self, simple_wavelength):
        mask = reference.build_feature_mask(
            simple_wavelength, mask_balmer=False, mask_telluric=False
        )
        assert not np.any(mask)

    def test_extra_mask_regions(self, simple_wavelength):
        mask = reference.build_feature_mask(
            simple_wavelength,
            mask_balmer=False, mask_telluric=False,
            extra_mask_regions=[(5000.0, 5100.0)],
        )
        idx_in = np.argmin(np.abs(simple_wavelength - 5050.0))
        assert mask[idx_in] == True
        idx_out = np.argmin(np.abs(simple_wavelength - 4500.0))
        assert mask[idx_out] == False

    def test_custom_balmer_regions(self, simple_wavelength):
        custom = [(5000.0, 50.0)]  # center=5000, half_width=50
        mask = reference.build_feature_mask(
            simple_wavelength,
            mask_balmer=True, mask_telluric=False,
            balmer_regions=custom,
        )
        idx_in = np.argmin(np.abs(simple_wavelength - 5000.0))
        assert mask[idx_in] == True
        # Default H-alpha should NOT be masked (custom overrides default)
        idx_ha = np.argmin(np.abs(simple_wavelength - 6562.8))
        assert mask[idx_ha] == False


class TestPrepareReference:
    """Tests for the full preparation pipeline."""

    def test_basic_preparation(self, simple_wavelength, simple_flux):
        """prepare_reference should work with a synthetic Spectrum1D."""
        try:
            from specutils import Spectrum as Spectrum1D
        except ImportError:
            from specutils import Spectrum1D

        # Create a mock reference spectrum at higher resolution
        ref_wave = np.arange(3800.0, 7200.0, 0.5)
        ref_flux_vals = np.ones(len(ref_wave)) * 1e-10  # arbitrary
        ref_spec = Spectrum1D(
            flux=ref_flux_vals * u.erg / u.s / u.cm**2 / u.AA,
            spectral_axis=ref_wave * u.AA,
        )

        obs_wave = simple_wavelength
        wave, flux, mask = reference.prepare_reference(
            ref_spec, obs_wave, mask_balmer=True, mask_telluric=True
        )

        assert len(wave) == len(obs_wave)
        assert len(flux) == len(obs_wave)
        assert len(mask) == len(obs_wave)
        assert mask.dtype == bool

    def test_preparation_with_convolution(self, simple_wavelength):
        try:
            from specutils import Spectrum as Spectrum1D
        except ImportError:
            from specutils import Spectrum1D

        ref_wave = np.arange(3800.0, 7200.0, 0.5)
        ref_flux_vals = np.ones(len(ref_wave)) * 1e-10
        ref_spec = Spectrum1D(
            flux=ref_flux_vals * u.erg / u.s / u.cm**2 / u.AA,
            spectral_axis=ref_wave * u.AA,
        )

        wave, flux, mask = reference.prepare_reference(
            ref_spec, simple_wavelength, target_fwhm_aa=10.0
        )
        assert len(flux) == len(simple_wavelength)


# ============================================================================
# 5.6 -- Diagnostic plot tests
# ============================================================================

class TestDiagnosticPlot:
    """Tests for the comparison plot function."""

    def test_plot_runs_without_error(self, simple_wavelength, simple_flux):
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend for testing

        fig = reference.plot_reference_comparison(
            simple_wavelength, simple_flux,
            simple_wavelength, simple_flux * 1.1,
            star_name="Test Star",
        )
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_with_mask(self, simple_wavelength, simple_flux):
        import matplotlib
        matplotlib.use("Agg")

        mask = reference.build_feature_mask(simple_wavelength)

        fig = reference.plot_reference_comparison(
            simple_wavelength, simple_flux,
            simple_wavelength, simple_flux * 0.9,
            feature_mask=mask,
            normalize=True,
        )
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_saves_to_file(self, simple_wavelength, simple_flux, tmp_path):
        import matplotlib
        matplotlib.use("Agg")

        save_path = tmp_path / "test_comparison.png"
        fig = reference.plot_reference_comparison(
            simple_wavelength, simple_flux,
            simple_wavelength, simple_flux,
            save_path=str(save_path),
        )
        assert save_path.exists()
        import matplotlib.pyplot as plt
        plt.close(fig)


# ============================================================================
# 5.1 -- CALSPEC access tests
# ============================================================================

class TestCALSPECUnit:
    """Unit tests for CALSPEC-related functions (no network)."""

    def test_calspec_requires_star_or_file(self):
        with pytest.raises(ValueError, match="Must specify"):
            reference.load_calspec()

    def test_calspec_star_without_calspec_file_raises(self):
        with pytest.raises(ValueError, match="does not have a CALSPEC"):
            reference.load_calspec(star_name="betelgeuse")

    def test_convert_to_flambda(self):
        """Test mJy -> erg/s/cm^2/A conversion."""
        try:
            from specutils import Spectrum as Spectrum1D
        except ImportError:
            from specutils import Spectrum1D

        # Create a fake mJy spectrum at 5000 A
        wave = np.array([5000.0]) * u.AA
        # 1 mJy at 5000 A
        flux = np.array([1.0]) * u.mJy

        spec = Spectrum1D(flux=flux, spectral_axis=wave)
        converted = reference._convert_to_flambda(spec)

        # Check the conversion: F_lambda = F_nu * c / lambda^2
        # F_nu = 1e-26 erg/s/cm^2/Hz (from 1 mJy)
        # c = 2.998e18 A/s
        # lambda = 5000 A
        # F_lambda = 1e-26 * 2.998e18 / 5000^2 = 1.1992e-15
        expected = 1.0e-26 * 2.99792458e18 / 5000.0**2
        actual = converted.flux.value[0]
        np.testing.assert_allclose(actual, expected, rtol=1e-4)


# ============================================================================
# 5.2 -- Model spectrum tests (no network)
# ============================================================================

class TestModelSpectrumUnit:
    """Unit tests for model loading functions (no network)."""

    def test_model_requires_params(self):
        with pytest.raises(ValueError, match="Must specify teff"):
            reference.load_model_spectrum()

    def test_model_star_name_lookup(self):
        """Verify that star_name lookup produces correct parameters."""
        # We can't actually download, but we can test the parameter resolution
        params = reference.get_stellar_parameters("sirius")
        assert params["teff"] == 9940
        assert params["logg"] == 4.33


# ============================================================================
# Integration tests (require network access)
# ============================================================================

@pytest.mark.network
class TestCALSPECIntegration:
    """Integration tests that download CALSPEC spectra from MAST."""

    def test_load_sirius_calspec(self):
        spec = reference.load_calspec(star_name="sirius")
        assert spec is not None
        wave = spec.spectral_axis.to(u.AA).value
        # Should cover optical range
        assert wave.min() < 4000.0
        assert wave.max() > 8000.0
        # Flux should be in F_lambda units
        assert spec.flux.unit == u.erg / u.s / u.cm**2 / u.AA

    def test_load_vega_calspec(self):
        spec = reference.load_calspec(star_name="vega")
        assert spec is not None
        wave = spec.spectral_axis.to(u.AA).value
        assert wave.min() < 4000.0

    def test_load_by_filename(self):
        spec = reference.load_calspec(calspec_file="sirius_stis_001.fits")
        assert spec is not None

    def test_load_reference_prefers_calspec(self):
        spec = reference.load_reference_spectrum("sirius", prefer="calspec")
        assert spec is not None


@pytest.mark.network
class TestPhoenixIntegration:
    """Integration tests that download PHOENIX models via expecto."""

    def test_load_phoenix_sirius(self):
        spec = reference.load_phoenix_model(
            teff=9900, logg=4.3, feh=0.0, cache=True
        )
        assert spec is not None
        wave = spec.spectral_axis.to(u.AA).value
        assert wave.min() < 4000.0
        assert wave.max() > 8000.0
        # Flux should be in erg/s/cm^2/A
        assert spec.flux.unit == u.erg / u.s / u.cm**2 / u.AA

    def test_load_model_by_star_name(self):
        spec = reference.load_model_spectrum("betelgeuse", cache=True)
        assert spec is not None
        assert spec.meta["teff"] == 3600

    def test_load_reference_for_betelgeuse_uses_model(self):
        """Betelgeuse has no CALSPEC; should fall back to model."""
        spec = reference.load_reference_spectrum("betelgeuse")
        assert spec is not None
        assert spec.meta.get("source") == "PHOENIX/expecto"


@pytest.mark.network
class TestFullPipelineIntegration:
    """End-to-end integration test: load, prepare, compare."""

    def test_sirius_calspec_preparation(self):
        """Load Sirius CALSPEC, prepare for comparison with fake obs data."""
        ref = reference.load_calspec(star_name="sirius")
        obs_wave = np.arange(4000.0, 7500.0, 2.0)

        wave, flux, mask = reference.prepare_reference(
            ref, obs_wave,
            target_fwhm_aa=10.0,
            mask_balmer=True,
            mask_telluric=True,
        )
        assert len(wave) == len(obs_wave)
        assert np.all(np.isfinite(flux))
        assert mask.dtype == bool

    def test_atmospheric_extinction_all_models(self):
        """Verify all observatory extinction models load successfully."""
        for model in reference.SUPPORTED_OBSERVATORY_MODELS:
            wave, ext = reference.load_atmospheric_extinction(model)
            assert len(wave) > 0
            assert np.all(ext >= 0)
