"""
Unit tests for the JJMO data ingestion module (io.py).

Tests cover:
- Each reader against actual Sirius and Betelgeuse data
- The dispatcher with a mixed directory
- Edge cases: missing wavelength file, empty spectrum, single-pixel segment
- Metadata extraction, YAML config loading
- Validation: NaN/Inf masking, monotonicity enforcement, minimum pixel check
- Generic 1D FITS reader
"""

import os
import shutil
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
import astropy.units as u

# ---------------------------------------------------------------------------
# Module import -- use the package namespace
# ---------------------------------------------------------------------------

from jjmo_fluxcal import io as jio

# Suppress specutils Spectrum1D deprecation warning globally for tests
warnings.filterwarnings("ignore", message=".*Spectrum1D.*deprecated.*")

# ---------------------------------------------------------------------------
# Paths to real data
# ---------------------------------------------------------------------------

DATA_ROOT = Path("/home/habjan.e/JJMO_home/Data")
SIRIUS_DIR = DATA_ROOT / "Sirius"
BETELGEUSE_DIR = DATA_ROOT / "Betelgeuse"

SIRIUS_FIT = SIRIUS_DIR / "3900.fit"
SIRIUS_TXT = SIRIUS_DIR / "3900.txt"
BETELGEUSE_CSV = BETELGEUSE_DIR / "Betelgeuse_4400.csv"

# Skip data-dependent tests if data is not available
HAS_DATA = SIRIUS_FIT.exists() and BETELGEUSE_CSV.exists()
requires_data = pytest.mark.skipif(not HAS_DATA, reason="Data not found")


# ===================================================================
# Tests against real data
# ===================================================================

class TestReadFitTxtPair:
    """Tests for read_fit_txt_pair against Sirius data."""

    @requires_data
    def test_basic_read(self):
        spec = jio.read_fit_txt_pair(SIRIUS_FIT)
        assert spec.flux.shape == (765,)
        assert spec.spectral_axis.shape == (765,)
        assert spec.flux.unit == u.ct
        assert spec.spectral_axis.unit == u.AA

    @requires_data
    def test_wavelength_range(self):
        spec = jio.read_fit_txt_pair(SIRIUS_FIT)
        wmin = spec.spectral_axis.value.min()
        wmax = spec.spectral_axis.value.max()
        assert 3200 < wmin < 3400  # ~3280
        assert 4100 < wmax < 4300  # ~4244

    @requires_data
    def test_wavelength_monotonic(self):
        spec = jio.read_fit_txt_pair(SIRIUS_FIT)
        diffs = np.diff(spec.spectral_axis.value)
        assert np.all(diffs >= 0), "Wavelength must be monotonically increasing"

    @requires_data
    def test_metadata_extraction(self):
        spec = jio.read_fit_txt_pair(SIRIUS_FIT)
        assert spec.meta["exptime"] == 30.0
        assert spec.meta["date_obs"] is not None
        assert "2004" in spec.meta["date_obs"]
        assert spec.meta["segment_id"] == "3900"
        assert spec.meta["wavelength_unit"] == "Angstrom"
        assert spec.meta["flux_unit"] == "counts"
        assert spec.meta["source_file"] == str(SIRIUS_FIT)

    @requires_data
    def test_all_sirius_segments(self):
        """Every Sirius segment should load successfully."""
        for seg in ["3900", "4400", "4900", "5400", "5900", "6400", "6900", "7400"]:
            spec = jio.read_fit_txt_pair(SIRIUS_DIR / f"{seg}.fit")
            assert spec.flux.shape[0] > 100
            assert spec.meta["segment_id"] == seg

    @requires_data
    def test_explicit_txt_path(self):
        spec = jio.read_fit_txt_pair(SIRIUS_FIT, txt_path=SIRIUS_TXT)
        assert spec.flux.shape == (765,)

    @requires_data
    def test_metadata_overrides(self):
        spec = jio.read_fit_txt_pair(
            SIRIUS_FIT,
            metadata_overrides={"airmass": 1.23, "custom_key": "test"}
        )
        assert spec.meta["airmass"] == 1.23
        assert spec.meta["custom_key"] == "test"

    @requires_data
    def test_header_variant_keywords(self):
        """Segments 4400+ use different header keywords (EXPOSURE vs EXPTIME)."""
        spec = jio.read_fit_txt_pair(SIRIUS_DIR / "4400.fit")
        assert spec.meta["exptime"] == 30.0
        assert spec.meta["date_obs"] is not None


class TestReadCSV:
    """Tests for read_csv against Betelgeuse data."""

    @requires_data
    def test_basic_read(self):
        spec = jio.read_csv(BETELGEUSE_CSV)
        assert spec.flux.shape == (765,)
        assert spec.spectral_axis.shape == (765,)

    @requires_data
    def test_wavelength_range(self):
        spec = jio.read_csv(BETELGEUSE_CSV)
        wmin = spec.spectral_axis.value.min()
        wmax = spec.spectral_axis.value.max()
        assert 3800 < wmin < 3950  # ~3882
        assert 4600 < wmax < 4750  # ~4698

    @requires_data
    def test_wavelength_monotonic(self):
        spec = jio.read_csv(BETELGEUSE_CSV)
        diffs = np.diff(spec.spectral_axis.value)
        assert np.all(diffs >= 0)

    @requires_data
    def test_segment_id_parsed(self):
        spec = jio.read_csv(BETELGEUSE_CSV)
        assert spec.meta["segment_id"] == "4400"

    @requires_data
    def test_all_betelgeuse_segments(self):
        for seg in ["4400", "4900", "5400", "5900", "6400", "6900", "7400"]:
            spec = jio.read_csv(BETELGEUSE_DIR / f"Betelgeuse_{seg}.csv")
            assert spec.flux.shape[0] > 100
            assert spec.meta["segment_id"] == seg

    @requires_data
    def test_no_header_metadata(self):
        """CSV files have no FITS headers; metadata should be None for those fields."""
        spec = jio.read_csv(BETELGEUSE_CSV)
        assert spec.meta["exptime"] is None
        assert spec.meta["airmass"] is None
        assert spec.meta["instrument"] is None


# ===================================================================
# Tests for generic 1D FITS reader
# ===================================================================

class TestReadFits1D:
    """Tests for read_fits_1d using synthetic FITS files."""

    def _make_1d_fits(self, tmp_path, n_pix=100, crval1=5000.0, cdelt1=1.5):
        """Create a minimal 1D FITS file with WCS keywords."""
        flux = np.random.RandomState(42).normal(1000, 50, n_pix).astype(np.float64)
        hdu = fits.PrimaryHDU(flux)
        hdu.header["CRVAL1"] = crval1
        hdu.header["CDELT1"] = cdelt1
        hdu.header["CRPIX1"] = 1.0
        hdu.header["CTYPE1"] = "WAVE"
        hdu.header["EXPTIME"] = 60.0
        hdu.header["DATE-OBS"] = "2024-06-15"
        fpath = tmp_path / "test_1d.fits"
        hdu.writeto(fpath, overwrite=True)
        return fpath

    def test_basic_read(self, tmp_path):
        fpath = self._make_1d_fits(tmp_path)
        spec = jio.read_fits_1d(fpath)
        assert spec.flux.shape == (100,)
        assert np.isclose(spec.spectral_axis.value[0], 5000.0)
        assert np.isclose(spec.spectral_axis.value[1] - spec.spectral_axis.value[0], 1.5)

    def test_metadata(self, tmp_path):
        fpath = self._make_1d_fits(tmp_path)
        spec = jio.read_fits_1d(fpath)
        assert spec.meta["exptime"] == 60.0
        assert "2024" in spec.meta["date_obs"]

    def test_cd1_1_fallback(self, tmp_path):
        """Test that CD1_1 is used when CDELT1 is missing."""
        flux = np.ones(50)
        hdu = fits.PrimaryHDU(flux)
        hdu.header["CRVAL1"] = 4000.0
        hdu.header["CD1_1"] = 2.0
        hdu.header["CRPIX1"] = 1.0
        fpath = tmp_path / "cd1_1.fits"
        hdu.writeto(fpath, overwrite=True)
        spec = jio.read_fits_1d(fpath)
        assert np.isclose(spec.spectral_axis.value[1] - spec.spectral_axis.value[0], 2.0)

    def test_missing_wcs_raises(self, tmp_path):
        flux = np.ones(50)
        hdu = fits.PrimaryHDU(flux)
        fpath = tmp_path / "no_wcs.fits"
        hdu.writeto(fpath, overwrite=True)
        with pytest.raises(ValueError, match="missing CRVAL1"):
            jio.read_fits_1d(fpath)


# ===================================================================
# Dispatcher tests
# ===================================================================

class TestDispatcher:

    @requires_data
    def test_auto_detect_fit(self):
        spec = jio.read_spectrum(SIRIUS_FIT)
        assert spec.flux.shape == (765,)

    @requires_data
    def test_auto_detect_csv(self):
        spec = jio.read_spectrum(BETELGEUSE_CSV)
        assert spec.flux.shape == (765,)

    @requires_data
    def test_forced_format(self):
        spec = jio.read_spectrum(SIRIUS_FIT, format="fit_txt")
        assert spec.flux.shape == (765,)

    def test_unknown_extension_raises(self, tmp_path):
        p = tmp_path / "data.xyz"
        p.write_text("junk")
        with pytest.raises(ValueError, match="Unrecognised file extension"):
            jio.read_spectrum(p)


# ===================================================================
# Directory reader tests
# ===================================================================

class TestReadDirectory:

    @requires_data
    def test_sirius_directory(self):
        specs = jio.read_directory(SIRIUS_DIR)
        assert len(specs) == 8
        # Verify sorted by wavelength
        for i in range(len(specs) - 1):
            assert (specs[i].spectral_axis.value.min()
                    <= specs[i + 1].spectral_axis.value.min())

    @requires_data
    def test_betelgeuse_directory(self):
        specs = jio.read_directory(BETELGEUSE_DIR)
        assert len(specs) == 7
        for i in range(len(specs) - 1):
            assert (specs[i].spectral_axis.value.min()
                    <= specs[i + 1].spectral_axis.value.min())

    @requires_data
    def test_mixed_directory(self):
        """Create a temporary directory with both .fit and .csv files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            # Copy one Sirius pair and one Betelgeuse CSV
            shutil.copy(SIRIUS_FIT, tmpdir / "3900.fit")
            shutil.copy(SIRIUS_TXT, tmpdir / "3900.txt")
            shutil.copy(BETELGEUSE_CSV, tmpdir / "Betelgeuse_4400.csv")

            specs = jio.read_directory(tmpdir)
            assert len(specs) == 2
            # Both should be valid Spectrum1D objects
            for s in specs:
                assert s.flux.shape[0] > 100

    def test_empty_directory(self, tmp_path):
        with pytest.warns(UserWarning, match="No spectral data files"):
            specs = jio.read_directory(tmp_path)
        assert specs == []

    def test_not_a_directory_raises(self, tmp_path):
        f = tmp_path / "file.txt"
        f.write_text("not a dir")
        with pytest.raises(NotADirectoryError):
            jio.read_directory(f)


# ===================================================================
# Validation tests
# ===================================================================

class TestValidation:

    def test_nan_masking(self, tmp_path):
        """NaN values in flux should be masked."""
        n = 15
        flux = np.arange(n, dtype=float) * 100 + 500
        flux[3] = np.nan
        wave = np.linspace(5000, 5014, n)
        data = np.column_stack([np.arange(n), wave, flux])
        fpath = tmp_path / "nan_test.csv"
        np.savetxt(fpath, data, delimiter=",")
        spec = jio.read_csv(fpath, wavelength_col=1, flux_col=2)
        assert spec.mask[3] == True  # NaN pixel masked
        assert spec.mask[0] == False  # good pixel not masked

    def test_inf_masking(self, tmp_path):
        """Inf values should be masked."""
        n = 15
        flux = np.arange(n, dtype=float) * 100 + 500
        flux[2] = np.inf
        flux[5] = -np.inf
        wave = np.linspace(5000, 5014, n)
        data = np.column_stack([np.arange(n), wave, flux])
        fpath = tmp_path / "inf_test.csv"
        np.savetxt(fpath, data, delimiter=",")
        spec = jio.read_csv(fpath, wavelength_col=1, flux_col=2)
        assert spec.mask[2] == True
        assert spec.mask[5] == True

    def test_too_few_pixels_raises(self, tmp_path):
        """Segment with fewer than MIN_VALID_PIXELS should be rejected."""
        flux = np.array([1.0, 2.0, 3.0])
        wave = np.array([5000, 5001, 5002], dtype=float)
        data = np.column_stack([np.arange(3), wave, flux])
        fpath = tmp_path / "tiny.csv"
        np.savetxt(fpath, data, delimiter=",")
        with pytest.raises(ValueError, match="valid pixels"):
            jio.read_csv(fpath, wavelength_col=1, flux_col=2)

    def test_non_monotonic_sorted(self, tmp_path):
        """Non-monotonic wavelengths should be sorted."""
        wave = np.array([5002, 5000, 5003, 5001, 5004,
                         5005, 5006, 5007, 5008, 5009,
                         5010, 5011], dtype=float)
        flux = np.arange(12, dtype=float) * 100
        data = np.column_stack([np.arange(12), wave, flux])
        fpath = tmp_path / "unsorted.csv"
        np.savetxt(fpath, data, delimiter=",")
        spec = jio.read_csv(fpath, wavelength_col=1, flux_col=2)
        diffs = np.diff(spec.spectral_axis.value)
        assert np.all(diffs >= 0), "Should be sorted after validation"

    def test_non_positive_wavelength_masked(self, tmp_path):
        """Zero or negative wavelength should be masked."""
        n = 15
        wave = np.linspace(5000, 5014, n)
        wave[0] = 0
        wave[1] = -1
        flux = np.ones(n) * 100
        data = np.column_stack([np.arange(n), wave, flux])
        fpath = tmp_path / "badwave.csv"
        np.savetxt(fpath, data, delimiter=",")
        spec = jio.read_csv(fpath, wavelength_col=1, flux_col=2)
        # The two bad-wavelength pixels should be masked
        assert np.sum(spec.mask) == 2


# ===================================================================
# Edge case tests
# ===================================================================

class TestEdgeCases:

    def test_missing_txt_companion(self):
        """Reading a .fit file with no companion .txt should raise."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a dummy .fit file
            hdu = fits.PrimaryHDU(np.ones((5, 50)))
            fpath = Path(tmpdir) / "test.fit"
            hdu.writeto(fpath, overwrite=True)
            with pytest.raises(FileNotFoundError, match="Wavelength text file"):
                jio.read_fit_txt_pair(fpath)

    def test_nonexistent_file_raises(self):
        with pytest.raises(FileNotFoundError):
            jio.read_fit_txt_pair("/no/such/file.fit")

    def test_nonexistent_csv_raises(self):
        with pytest.raises(FileNotFoundError):
            jio.read_csv("/no/such/file.csv")

    def test_nonexistent_fits_raises(self):
        with pytest.raises(FileNotFoundError):
            jio.read_fits_1d("/no/such/file.fits")

    def test_2d_fits_rejected_by_1d_reader(self, tmp_path):
        """read_fits_1d should reject 2D data."""
        hdu = fits.PrimaryHDU(np.ones((5, 50)))
        fpath = tmp_path / "twod.fits"
        hdu.writeto(fpath, overwrite=True)
        with pytest.raises(ValueError, match="1D"):
            jio.read_fits_1d(fpath)

    def test_single_column_csv_raises(self, tmp_path):
        fpath = tmp_path / "onecolumn.csv"
        np.savetxt(fpath, np.arange(20), delimiter=",")
        with pytest.raises(ValueError, match="one column"):
            jio.read_csv(fpath)


# ===================================================================
# Metadata / config tests
# ===================================================================

class TestMetadataConfig:

    def test_normalise_date_iso(self):
        assert jio._normalise_date("2004-01-29") == "2004-01-29"

    def test_normalise_date_mmddyy(self):
        assert jio._normalise_date("01/29/04") == "2004-01-29"

    def test_normalise_date_mmddyy_1900s(self):
        assert jio._normalise_date("06/15/98") == "1998-06-15"

    def test_yaml_config_roundtrip(self, tmp_path):
        cfg_content = """
global:
    instrument: "SBIG ST-7"
    airmass: 1.05
segments:
    3900:
        exptime: 30.0
    4400:
        exptime: 25.0
"""
        cfg_path = tmp_path / "meta.yaml"
        cfg_path.write_text(cfg_content)
        cfg = jio.load_metadata_config(cfg_path)
        assert cfg["global"]["instrument"] == "SBIG ST-7"
        assert cfg["segments"]["3900"]["exptime"] == 30.0
        assert cfg["segments"]["4400"]["exptime"] == 25.0

    @requires_data
    def test_config_applied_to_directory(self, tmp_path):
        cfg_content = """
global:
    airmass: 1.23
"""
        cfg_path = tmp_path / "meta.yaml"
        cfg_path.write_text(cfg_content)
        specs = jio.read_directory(BETELGEUSE_DIR, config_path=cfg_path)
        for s in specs:
            assert s.meta["airmass"] == 1.23

    def test_config_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            jio.load_metadata_config("/no/such/config.yaml")


# ===================================================================
# Required metadata keys test
# ===================================================================

class TestRequiredMetaKeys:

    @requires_data
    def test_all_keys_present_sirius(self):
        spec = jio.read_fit_txt_pair(SIRIUS_FIT)
        for key in jio.REQUIRED_META_KEYS:
            assert key in spec.meta, f"Missing required meta key: {key}"

    @requires_data
    def test_all_keys_present_betelgeuse(self):
        spec = jio.read_csv(BETELGEUSE_CSV)
        for key in jio.REQUIRED_META_KEYS:
            assert key in spec.meta, f"Missing required meta key: {key}"
