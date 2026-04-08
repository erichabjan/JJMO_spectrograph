"""
test_sensitivity.py -- Tests for Step 6: Sensitivity Function Derivation
=========================================================================

Tests the sensitivity module with synthetic data that mimics the characteristics
of JJMO spectrograph observations: noisy counts, short ~500 A segments, telluric
and stellar features, and known reference spectra.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from jjmo_fluxcal.sensitivity import (
    # Core functions
    compute_sensitivity_ratio,
    fit_sensitivity,
    combine_segment_sensitivities,
    derive_sensitivity_per_segment,
    derive_sensitivity_stitched,
    derive_sensitivity,
    build_sensitivity_mask,
    estimate_grey_shift_multi_obs,
    save_sensitivity,
    load_sensitivity,
    # Classes
    SensitivityFit,
    GlobalSensitivity,
    # Helpers
    _normalize_wavelength,
    _denormalize_wavelength,
    _resample_reference,
    # Constants
    TELLURIC_BANDS,
    BALMER_LINES_AA,
    METAL_LINES_AA,
    MIN_FIT_PIXELS,
)


# ============================================================================
# Fixtures: synthetic data generators
# ============================================================================

def make_smooth_sensitivity(wavelength, scale=1e-15, slope=2e-19):
    """A simple smooth sensitivity function: linear trend + shallow curve.

    Mimics the smooth wavelength-dependent throughput of optics + detector.
    """
    w_center = np.mean(wavelength)
    return scale + slope * (wavelength - w_center) + \
        1e-20 * (wavelength - w_center) ** 2


def make_reference_flux(wavelength, teff=9940.0):
    """Fake blackbody-like reference flux for testing.

    Returns flux in erg/s/cm^2/A units, roughly matching an A-star SED.
    """
    # Simplified Planck-like shape (not physically accurate, but smooth)
    h, c, k = 6.626e-27, 3e10, 1.381e-16  # cgs
    lam_cm = wavelength * 1e-8
    # Avoid overflow by computing in log space
    x = h * c / (lam_cm * k * teff)
    x = np.clip(x, 0, 500)
    flux = 2 * h * c**2 / lam_cm**5 / (np.exp(x) - 1)
    # Scale to realistic erg/s/cm^2/A values
    flux *= 1e-8  # per Angstrom
    flux /= flux.max()
    flux *= 5e-9   # roughly Sirius-level flux
    return flux


def make_observed_counts(wavelength, ref_flux, sensitivity, exptime=60.0,
                         noise_level=0.05, rng=None):
    """Generate synthetic observed counts from reference flux and sensitivity.

    C_obs = F_ref / S(lambda) * exptime + noise
    """
    if rng is None:
        rng = np.random.default_rng(42)

    count_rate = ref_flux / sensitivity  # counts/s/A
    counts = count_rate * exptime

    # Add Poisson-like noise (approximated as Gaussian for simplicity)
    noise = rng.normal(0, noise_level * counts)
    counts += noise
    counts = np.maximum(counts, 1.0)  # no zero/negative counts

    return counts


def make_segment(center_wave, width=500, n_pixels=500, exptime=60.0,
                 noise_level=0.05, rng=None):
    """Create a complete synthetic segment: wavelength, observed flux,
    reference flux, true sensitivity, and mask.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    wavelength = np.linspace(center_wave - width / 2,
                             center_wave + width / 2, n_pixels)
    sensitivity = make_smooth_sensitivity(wavelength)
    ref_flux = make_reference_flux(wavelength)
    obs_flux = make_observed_counts(
        wavelength, ref_flux, sensitivity, exptime=exptime,
        noise_level=noise_level, rng=rng,
    )
    mask = np.ones(n_pixels, dtype=bool)

    return wavelength, obs_flux, ref_flux, sensitivity, mask


def make_multi_segments(centers=None, overlap=50, **kwargs):
    """Create multiple overlapping segments."""
    if centers is None:
        centers = [4150, 4650, 5150, 5650, 6150, 6650, 7150]

    segments = []
    rng = np.random.default_rng(12345)
    for center in centers:
        w, f_obs, f_ref, true_s, mask = make_segment(
            center, rng=rng, **kwargs
        )
        segments.append({
            'wavelength': w,
            'flux_obs': f_obs,
            'flux_ref': f_ref,
            'true_sensitivity': true_s,
            'mask': mask,
            'center': center,
        })

    # Also build the combined reference spectrum
    all_w = np.concatenate([s['wavelength'] for s in segments])
    all_ref = np.concatenate([s['flux_ref'] for s in segments])
    order = np.argsort(all_w)
    # Remove duplicates (from overlaps)
    w_unique, idx = np.unique(all_w[order], return_index=True)
    ref_unique = all_ref[order][idx]

    return segments, w_unique, ref_unique


# ============================================================================
# Tests: 6.1 Raw sensitivity ratio
# ============================================================================

class TestComputeSensitivityRatio:
    """Tests for compute_sensitivity_ratio()."""

    def test_basic_ratio(self):
        """The ratio F_ref/C_obs should recover the sensitivity."""
        exptime = 60.0
        w, f_obs, f_ref, true_s, mask = make_segment(
            5000, noise_level=0.0, exptime=exptime
        )
        _, ratio, mask_out = compute_sensitivity_ratio(
            w, f_obs, w, f_ref, exptime=exptime
        )
        # With no noise, the ratio should closely match the true sensitivity
        valid = mask_out & np.isfinite(ratio)
        np.testing.assert_allclose(
            ratio[valid], true_s[valid], rtol=0.01,
            err_msg="Noiseless ratio should recover true sensitivity"
        )

    def test_exposure_time_normalization(self):
        """Passing exptime should divide observed flux by that value."""
        w = np.linspace(4000, 4500, 100)
        ref = np.ones(100) * 1e-10
        # If counts = 100 and exptime = 10, count_rate = 10
        # ratio = 1e-10 / 10 = 1e-11
        obs = np.ones(100) * 100.0
        _, ratio, _ = compute_sensitivity_ratio(
            w, obs, w, ref, exptime=10.0
        )
        expected = 1e-10 / 10.0
        np.testing.assert_allclose(ratio, expected, rtol=1e-10)

    def test_extinction_correction(self):
        """Atmospheric extinction correction should boost observed flux."""
        w = np.linspace(4000, 4500, 100)
        ref = np.ones(100) * 1e-10
        obs = np.ones(100) * 100.0
        airmass = 1.5

        # Simple extinction: k(lambda) = 0.2 mag/airmass (constant)
        k_func = lambda w: np.full_like(w, 0.2)

        _, ratio_no_ext, _ = compute_sensitivity_ratio(w, obs, w, ref)
        _, ratio_ext, _ = compute_sensitivity_ratio(
            w, obs, w, ref, airmass=airmass, extinction_curve=k_func
        )

        # Extinction correction boosts observed flux, so ratio should decrease
        ext_factor = 10.0 ** (0.4 * airmass * 0.2)
        expected_ratio = ratio_no_ext / ext_factor
        np.testing.assert_allclose(ratio_ext, expected_ratio, rtol=1e-10)

    def test_mask_propagation(self):
        """Input mask should propagate to output."""
        w = np.linspace(4000, 4500, 100)
        ref = np.ones(100) * 1e-10
        obs = np.ones(100) * 100.0

        mask_in = np.ones(100, dtype=bool)
        mask_in[10:20] = False

        _, _, mask_out = compute_sensitivity_ratio(
            w, obs, w, ref, mask=mask_in
        )
        assert not np.any(mask_out[10:20]), "Masked pixels should stay masked"
        assert np.all(mask_out[:10]), "Unmasked pixels should remain good"

    def test_non_positive_flux_masked(self):
        """Pixels with zero or negative observed flux should be masked."""
        w = np.linspace(4000, 4500, 100)
        ref = np.ones(100) * 1e-10
        obs = np.ones(100) * 100.0
        obs[50] = 0.0
        obs[51] = -10.0

        _, ratio, mask_out = compute_sensitivity_ratio(w, obs, w, ref)
        assert not mask_out[50]
        assert not mask_out[51]

    def test_reference_resampling(self):
        """Reference spectrum on a different grid should be resampled."""
        w_obs = np.linspace(4000, 4500, 100)
        w_ref = np.linspace(3800, 4800, 500)  # wider, denser grid
        ref = np.ones(500) * 1e-10
        obs = np.ones(100) * 100.0

        _, ratio, mask_out = compute_sensitivity_ratio(
            w_obs, obs, w_ref, ref
        )
        expected = 1e-10 / 100.0
        np.testing.assert_allclose(
            ratio[mask_out], expected, rtol=1e-5
        )

    def test_outside_reference_range(self):
        """Pixels outside reference coverage should be masked."""
        w_obs = np.linspace(3800, 5000, 200)
        w_ref = np.linspace(4000, 4500, 100)
        ref = np.ones(100) * 1e-10
        obs = np.ones(200) * 100.0

        _, _, mask_out = compute_sensitivity_ratio(
            w_obs, obs, w_ref, ref
        )
        # Pixels below 4000 A and above 4500 A should be masked
        below = w_obs < 4000
        above = w_obs > 4500
        assert not np.any(mask_out[below])
        assert not np.any(mask_out[above])


# ============================================================================
# Tests: 6.2 Smooth fitting
# ============================================================================

class TestFitSensitivity:
    """Tests for fit_sensitivity()."""

    def test_chebyshev_fit(self):
        """Chebyshev polynomial should fit smooth data well."""
        w, f_obs, f_ref, true_s, mask = make_segment(
            5000, noise_level=0.01
        )
        _, ratio, mask_out = compute_sensitivity_ratio(
            w, f_obs, w, f_ref
        )

        fit = fit_sensitivity(
            w, ratio, mask=mask_out,
            method='chebyshev', order=5,
            segment_id='test_cheb'
        )

        assert fit.method == 'chebyshev'
        assert fit.order == 5
        assert fit.n_points_used > 0
        assert fit.rms_residual > 0
        assert fit.wave_min < fit.wave_max

        # Evaluate the fit
        s_fitted = fit(w)
        valid = mask_out & np.isfinite(s_fitted)
        assert np.any(valid)

    def test_legendre_fit(self):
        """Legendre polynomial should also work."""
        w, f_obs, f_ref, true_s, mask = make_segment(
            5000, noise_level=0.01
        )
        _, ratio, mask_out = compute_sensitivity_ratio(w, f_obs, w, f_ref)

        fit = fit_sensitivity(
            w, ratio, mask=mask_out,
            method='legendre', order=5,
            segment_id='test_leg'
        )
        assert fit.method == 'legendre'
        assert fit.rms_residual > 0

    def test_spline_fit(self):
        """Cubic spline with few knots should fit smooth data."""
        w, f_obs, f_ref, true_s, mask = make_segment(
            5000, noise_level=0.01
        )
        _, ratio, mask_out = compute_sensitivity_ratio(w, f_obs, w, f_ref)

        fit = fit_sensitivity(
            w, ratio, mask=mask_out,
            method='spline', order=5,
            segment_id='test_spline'
        )
        assert fit.method == 'spline'
        assert len(fit._wavelength_data) > 0
        assert len(fit._flux_data) > 0

    def test_savgol_fit(self):
        """Savitzky-Golay smoothing should produce a smooth result."""
        w, f_obs, f_ref, true_s, mask = make_segment(
            5000, noise_level=0.05
        )
        _, ratio, mask_out = compute_sensitivity_ratio(w, f_obs, w, f_ref)

        fit = fit_sensitivity(
            w, ratio, mask=mask_out,
            method='savgol', order=51,
            segment_id='test_savgol'
        )
        assert fit.method == 'savgol'

    def test_sigma_clipping(self):
        """Sigma clipping should reject outliers and improve the fit."""
        rng = np.random.default_rng(42)
        w, f_obs, f_ref, true_s, mask = make_segment(
            5000, noise_level=0.01, rng=rng
        )
        _, ratio, mask_out = compute_sensitivity_ratio(w, f_obs, w, f_ref)

        # Inject outliers
        outlier_idx = rng.choice(np.where(mask_out)[0], size=10, replace=False)
        ratio[outlier_idx] *= 5.0

        fit_no_clip = fit_sensitivity(
            w, ratio, mask=mask_out,
            method='chebyshev', order=5,
            sigma_clip=3.0, max_iter=0,
            segment_id='no_clip'
        )
        fit_clipped = fit_sensitivity(
            w, ratio, mask=mask_out,
            method='chebyshev', order=5,
            sigma_clip=3.0, max_iter=5,
            segment_id='clipped'
        )

        assert fit_clipped.n_rejected > 0
        assert fit_clipped.rms_residual <= fit_no_clip.rms_residual

    def test_invalid_method_raises(self):
        """Invalid fit method should raise ValueError."""
        w = np.linspace(4000, 4500, 100)
        r = np.ones(100)
        with pytest.raises(ValueError, match="Unknown fit method"):
            fit_sensitivity(w, r, method='invalid')

    def test_too_few_points_raises(self):
        """Fitting with fewer than MIN_FIT_PIXELS should raise."""
        w = np.linspace(4000, 4500, 5)
        r = np.ones(5)
        with pytest.raises(ValueError, match="unmasked pixels"):
            fit_sensitivity(w, r, method='chebyshev', order=3)

    def test_callable_evaluation(self):
        """The fit object should be callable at arbitrary wavelengths."""
        w = np.linspace(4000, 4500, 200)
        r = 1e-15 + 1e-19 * (w - 4250)
        fit = fit_sensitivity(
            w, r, method='chebyshev', order=3, segment_id='callable_test'
        )

        # Evaluate at data points
        s = fit(w)
        assert s.shape == w.shape
        assert np.all(np.isfinite(s))

        # Evaluate outside domain -> NaN
        w_outside = np.array([3000.0, 8000.0])
        s_outside = fit(w_outside)
        assert np.all(np.isnan(s_outside))

    def test_fit_recovers_smooth_sensitivity(self):
        """With low noise, the fit should closely match the true sensitivity."""
        exptime = 60.0
        w, f_obs, f_ref, true_s, mask = make_segment(
            5000, noise_level=0.001, exptime=exptime
        )
        _, ratio, mask_out = compute_sensitivity_ratio(
            w, f_obs, w, f_ref, exptime=exptime
        )

        fit = fit_sensitivity(
            w, ratio, mask=mask_out,
            method='chebyshev', order=5,
        )

        s_fitted = fit(w)
        valid = mask_out & np.isfinite(s_fitted)

        # The fit should be within ~5% of the true sensitivity
        rel_error = np.abs(s_fitted[valid] - true_s[valid]) / true_s[valid]
        assert np.median(rel_error) < 0.05, \
            f"Median relative error {np.median(rel_error):.4f} too large"


# ============================================================================
# Tests: 6.3 Global combination
# ============================================================================

class TestCombineSegments:
    """Tests for combine_segment_sensitivities()."""

    def test_single_segment(self):
        """A single segment should produce a valid GlobalSensitivity."""
        w = np.linspace(4000, 4500, 200)
        r = make_smooth_sensitivity(w)
        fit = fit_sensitivity(w, r, method='chebyshev', order=3,
                              segment_id='seg_0')

        gs = combine_segment_sensitivities([fit], fit_global=False)
        assert len(gs.segment_fits) == 1
        assert gs.global_fit is None

        # Evaluate
        s = gs(w)
        assert np.all(np.isfinite(s))

    def test_multiple_segments_grey_shift(self):
        """Grey shifts should bring segments to a common scale."""
        segments, w_ref, f_ref = make_multi_segments(
            centers=[4500, 5000], noise_level=0.001
        )

        fits = []
        for seg in segments:
            w = seg['wavelength']
            _, ratio, mask = compute_sensitivity_ratio(
                w, seg['flux_obs'], w_ref, f_ref
            )
            f = fit_sensitivity(
                w, ratio, mask=mask,
                method='chebyshev', order=3,
                segment_id=f"seg_{seg['center']}"
            )
            fits.append(f)

        gs = combine_segment_sensitivities(fits, fit_global=True)
        assert len(gs.segment_fits) == 2
        assert gs.global_fit is not None
        assert len(gs.grey_shifts) == 2

    def test_global_fit_evaluates(self):
        """Global sensitivity should be evaluable across the full range."""
        segments, w_ref, f_ref = make_multi_segments(
            centers=[4500, 5000, 5500], noise_level=0.001
        )

        fits = []
        for seg in segments:
            w = seg['wavelength']
            _, ratio, mask = compute_sensitivity_ratio(
                w, seg['flux_obs'], w_ref, f_ref
            )
            f = fit_sensitivity(
                w, ratio, mask=mask,
                method='chebyshev', order=3,
                segment_id=f"seg_{seg['center']}"
            )
            fits.append(f)

        gs = combine_segment_sensitivities(fits, fit_global=True)

        w_test = np.linspace(4300, 5700, 500)
        s = gs(w_test)
        valid = np.isfinite(s)
        assert np.sum(valid) > 400, "Most of the evaluation range should be valid"


# ============================================================================
# Tests: 6.4 Per-segment vs. stitched approaches
# ============================================================================

class TestDualApproach:
    """Tests for per-segment and stitched-first approaches."""

    def test_per_segment_approach(self):
        """Per-segment approach should produce a valid result."""
        segments, w_ref, f_ref = make_multi_segments(
            centers=[4500, 5000, 5500], noise_level=0.01
        )

        segs_obs = [(s['wavelength'], s['flux_obs']) for s in segments]
        masks = [s['mask'] for s in segments]
        seg_ids = [f"seg_{s['center']}" for s in segments]

        gs = derive_sensitivity_per_segment(
            segs_obs, w_ref, f_ref,
            masks=masks,
            segment_ids=seg_ids,
            fit_method='chebyshev',
            fit_order=5,
            fit_global=True,
        )

        assert gs.approach == 'per_segment'
        assert len(gs.segment_fits) == 3
        assert gs.global_fit is not None

    def test_stitched_approach(self):
        """Stitched approach should produce a valid result."""
        segments, w_ref, f_ref = make_multi_segments(
            centers=[4500, 5000, 5500], noise_level=0.01
        )

        # Concatenate into a stitched spectrum
        all_w = np.concatenate([s['wavelength'] for s in segments])
        all_f = np.concatenate([s['flux_obs'] for s in segments])
        order = np.argsort(all_w)

        gs = derive_sensitivity_stitched(
            all_w[order], all_f[order],
            w_ref, f_ref,
            fit_method='chebyshev',
            fit_order=6,
        )

        assert gs.approach == 'stitched'
        assert gs.global_fit is not None

    def test_derive_sensitivity_dispatcher(self):
        """The derive_sensitivity() dispatcher should work for both approaches."""
        segments, w_ref, f_ref = make_multi_segments(
            centers=[4500, 5000], noise_level=0.01
        )
        segs_obs = [(s['wavelength'], s['flux_obs']) for s in segments]

        gs_ps = derive_sensitivity(
            segs_obs, w_ref, f_ref,
            approach='per_segment',
        )
        assert gs_ps.approach == 'per_segment'

        gs_st = derive_sensitivity(
            segs_obs, w_ref, f_ref,
            approach='stitched',
        )
        assert gs_st.approach == 'stitched'

    def test_invalid_approach_raises(self):
        """Invalid approach should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown approach"):
            derive_sensitivity(
                [(np.array([1, 2]), np.array([1, 2]))],
                np.array([1, 2]), np.array([1, 2]),
                approach='invalid'
            )


# ============================================================================
# Tests: 6.5 Telluric and stellar masking
# ============================================================================

class TestBuildMask:
    """Tests for build_sensitivity_mask()."""

    def test_telluric_masking(self):
        """Telluric bands should be masked."""
        # Wavelength range covering the O2 B-band region
        w = np.linspace(6800, 6950, 300)
        mask = build_sensitivity_mask(w, mask_telluric=True, mask_stellar=False,
                                      edge_fraction=0)

        # Pixels in the 6860-6880 band should be masked
        in_band = (w >= 6860) & (w <= 6880)
        assert not np.any(mask[in_band]), "Telluric band should be masked"

        # Pixels outside should be unmasked
        outside = (w < 6860) | (w > 6880)
        assert np.all(mask[outside]), "Non-telluric pixels should be good"

    def test_stellar_masking(self):
        """Stellar absorption lines should be masked."""
        # Range covering H-beta (4861.3 A)
        w = np.linspace(4800, 4920, 200)
        mask = build_sensitivity_mask(w, mask_telluric=False, mask_stellar=True,
                                      edge_fraction=0)

        # H-beta +/- 15 A should be masked
        near_hbeta = np.abs(w - 4861.3) <= 15.0
        assert not np.any(mask[near_hbeta]), "H-beta region should be masked"

    def test_edge_masking(self):
        """Segment edges should be trimmed."""
        w = np.linspace(4000, 4500, 200)
        mask = build_sensitivity_mask(
            w, mask_telluric=False, mask_stellar=False,
            edge_fraction=0.05
        )
        n_edge = int(200 * 0.05)
        assert not np.any(mask[:n_edge])
        assert not np.any(mask[-n_edge:])
        assert np.all(mask[n_edge:-n_edge])

    def test_quality_mask_propagation(self):
        """Pre-existing quality mask should be combined."""
        w = np.linspace(4000, 4500, 100)
        qmask = np.ones(100, dtype=bool)
        qmask[40:60] = False

        mask = build_sensitivity_mask(
            w, quality_mask=qmask,
            mask_telluric=False, mask_stellar=False,
            edge_fraction=0,
        )
        assert not np.any(mask[40:60])

    def test_custom_lines(self):
        """Custom stellar lines should be maskable."""
        w = np.linspace(5000, 5500, 200)
        custom_lines = np.array([5200.0, 5300.0])
        custom_widths = np.array([10.0, 8.0])

        mask = build_sensitivity_mask(
            w, mask_telluric=False, mask_stellar=True,
            stellar_lines=custom_lines,
            stellar_half_widths=custom_widths,
            edge_fraction=0,
        )

        near_5200 = np.abs(w - 5200) <= 10.0
        near_5300 = np.abs(w - 5300) <= 8.0
        assert not np.any(mask[near_5200])
        assert not np.any(mask[near_5300])


# ============================================================================
# Tests: 6.6 Non-photometric condition handling
# ============================================================================

class TestGreyShift:
    """Tests for estimate_grey_shift_multi_obs()."""

    def test_single_observation(self):
        """Single observation should return unity shift."""
        w = np.linspace(4000, 4500, 200)
        r = make_smooth_sensitivity(w)
        fit = fit_sensitivity(w, r, method='chebyshev', order=3)

        shifts, rms = estimate_grey_shift_multi_obs([fit])
        assert shifts == [1.0]
        assert rms == 0.0

    def test_multiple_observations_detect_offset(self):
        """Multiple observations with a known offset should be detected."""
        w = np.linspace(4000, 4500, 200)
        r1 = make_smooth_sensitivity(w)
        r2 = r1 * 1.15  # 15% offset

        fit1 = fit_sensitivity(w, r1, method='chebyshev', order=3,
                               segment_id='obs1')
        fit2 = fit_sensitivity(w, r2, method='chebyshev', order=3,
                               segment_id='obs2')

        shifts, rms = estimate_grey_shift_multi_obs([fit1, fit2])
        assert len(shifts) == 2
        assert shifts[0] == 1.0
        # The second shift should be close to 1/1.15 ~ 0.87
        np.testing.assert_allclose(shifts[1], 1.0 / 1.15, rtol=0.05)


# ============================================================================
# Tests: 6.7 Serialization
# ============================================================================

class TestSerialization:
    """Tests for save/load sensitivity functions."""

    def test_save_load_sensitivity_fit(self, tmp_path):
        """Round-trip save and load of a SensitivityFit."""
        w = np.linspace(4000, 4500, 200)
        r = make_smooth_sensitivity(w)
        fit = fit_sensitivity(
            w, r, method='chebyshev', order=5,
            segment_id='test_seg',
            metadata={'star': 'Sirius', 'source': 'CALSPEC'},
        )

        filepath = tmp_path / 'test_sensitivity.json'
        save_sensitivity(fit, filepath)
        assert filepath.exists()

        loaded = load_sensitivity(filepath)
        assert isinstance(loaded, SensitivityFit)
        assert loaded.method == 'chebyshev'
        assert loaded.order == 5
        assert loaded.segment_id == 'test_seg'
        assert loaded.metadata['star'] == 'Sirius'

        # Evaluate and compare
        s_orig = fit(w)
        s_loaded = loaded(w)
        np.testing.assert_allclose(s_orig, s_loaded, rtol=1e-10)

    def test_save_load_global_sensitivity(self, tmp_path):
        """Round-trip save and load of a GlobalSensitivity."""
        segments, w_ref, f_ref = make_multi_segments(
            centers=[4500, 5000], noise_level=0.001
        )

        segs_obs = [(s['wavelength'], s['flux_obs']) for s in segments]
        gs = derive_sensitivity_per_segment(
            segs_obs, w_ref, f_ref,
            segment_ids=['seg_4500', 'seg_5000'],
            fit_method='chebyshev',
            fit_order=3,
            fit_global=True,
        )

        filepath = tmp_path / 'global_sensitivity.json'
        save_sensitivity(gs, filepath)
        assert filepath.exists()

        loaded = load_sensitivity(filepath)
        assert isinstance(loaded, GlobalSensitivity)
        assert len(loaded.segment_fits) == 2
        assert loaded.global_fit is not None

        # Evaluate and compare
        w_test = np.linspace(4300, 5200, 300)
        s_orig = gs(w_test)
        s_loaded = loaded(w_test)
        valid = np.isfinite(s_orig) & np.isfinite(s_loaded)
        np.testing.assert_allclose(s_orig[valid], s_loaded[valid], rtol=1e-10)

    def test_load_nonexistent_raises(self):
        """Loading a nonexistent file should raise."""
        with pytest.raises(FileNotFoundError):
            load_sensitivity('/nonexistent/path.json')

    def test_to_dict_from_dict_roundtrip(self):
        """SensitivityFit.to_dict / from_dict should be lossless."""
        w = np.linspace(4000, 4500, 200)
        r = make_smooth_sensitivity(w)
        fit = fit_sensitivity(w, r, method='legendre', order=4,
                              segment_id='rt_test')

        d = fit.to_dict()
        restored = SensitivityFit.from_dict(d)

        assert restored.method == fit.method
        assert restored.order == fit.order
        assert restored.segment_id == fit.segment_id
        np.testing.assert_allclose(restored.coefficients, fit.coefficients)

        s_orig = fit(w)
        s_restored = restored(w)
        np.testing.assert_allclose(s_orig, s_restored, rtol=1e-12)


# ============================================================================
# Tests: SensitivityFit object
# ============================================================================

class TestSensitivityFitObject:
    """Tests for the SensitivityFit dataclass methods."""

    def test_to_magnitude(self):
        """to_magnitude should convert to -2.5*log10(S)."""
        w = np.linspace(4000, 4500, 200)
        r = make_smooth_sensitivity(w)
        fit = fit_sensitivity(w, r, method='chebyshev', order=3)

        mag = fit.to_magnitude(w)
        s = fit(w)
        expected = -2.5 * np.log10(s)
        np.testing.assert_allclose(mag, expected, rtol=1e-10)

    def test_grey_shift_applied(self):
        """Grey shift should multiply the evaluated sensitivity."""
        w = np.linspace(4000, 4500, 200)
        r = make_smooth_sensitivity(w)
        fit = fit_sensitivity(w, r, method='chebyshev', order=3)

        s_no_shift = fit(w).copy()
        fit.grey_shift = 1.5
        s_shifted = fit(w)

        np.testing.assert_allclose(s_shifted, 1.5 * s_no_shift, rtol=1e-10)


# ============================================================================
# Tests: Helper functions
# ============================================================================

class TestHelpers:
    """Tests for internal helper functions."""

    def test_normalize_denormalize_roundtrip(self):
        """Normalization should map to [-1,1] and back."""
        w = np.linspace(4000, 5000, 100)
        w_norm = _normalize_wavelength(w, 4000, 5000)
        assert w_norm[0] == pytest.approx(-1.0)
        assert w_norm[-1] == pytest.approx(1.0)

        w_back = _denormalize_wavelength(w_norm, 4000, 5000)
        np.testing.assert_allclose(w_back, w, atol=1e-10)

    def test_resample_reference(self):
        """Reference resampling should interpolate correctly."""
        w_ref = np.linspace(4000, 5000, 1000)
        f_ref = np.sin(w_ref / 100)

        w_target = np.linspace(4100, 4900, 200)
        f_resampled = _resample_reference(w_ref, f_ref, w_target)

        expected = np.sin(w_target / 100)
        np.testing.assert_allclose(f_resampled, expected, atol=1e-3)


# ============================================================================
# Tests: GlobalSensitivity piecewise evaluation
# ============================================================================

class TestGlobalSensitivityPiecewise:
    """Test piecewise evaluation fallback when no global fit is present."""

    def test_piecewise_evaluation(self):
        """Piecewise evaluation should use segment fits."""
        w1 = np.linspace(4000, 4500, 200)
        r1 = make_smooth_sensitivity(w1)
        fit1 = fit_sensitivity(w1, r1, method='chebyshev', order=3,
                               segment_id='seg1')

        w2 = np.linspace(4400, 4900, 200)
        r2 = make_smooth_sensitivity(w2)
        fit2 = fit_sensitivity(w2, r2, method='chebyshev', order=3,
                               segment_id='seg2')

        gs = GlobalSensitivity(
            segment_fits=[fit1, fit2],
            wave_min=4000, wave_max=4900,
        )

        # In the overlap region (4400-4500), both fits contribute
        w_test = np.array([4200, 4450, 4700])
        s = gs(w_test)
        assert np.isfinite(s[0])  # only fit1
        assert np.isfinite(s[1])  # overlap: average of fit1 and fit2
        assert np.isfinite(s[2])  # only fit2


# ============================================================================
# Integration test: end-to-end with synthetic data
# ============================================================================

class TestEndToEnd:
    """Integration tests running the full sensitivity derivation pipeline."""

    def test_full_pipeline_per_segment(self):
        """Full pipeline with per-segment approach on synthetic data."""
        segments, w_ref, f_ref = make_multi_segments(
            centers=[4500, 5000, 5500, 6000],
            noise_level=0.02,
        )

        segs_obs = [(s['wavelength'], s['flux_obs']) for s in segments]
        masks = [s['mask'] for s in segments]
        seg_ids = [f"seg_{s['center']}" for s in segments]

        gs = derive_sensitivity(
            segs_obs, w_ref, f_ref,
            approach='per_segment',
            masks=masks,
            segment_ids=seg_ids,
            fit_method='chebyshev',
            fit_order=5,
            sigma_clip=3.0,
            max_iter=5,
            fit_global=True,
            global_order=6,
        )

        assert gs.approach == 'per_segment'
        assert len(gs.segment_fits) == 4
        assert gs.global_fit is not None

        # Evaluate across the full range
        w_eval = np.linspace(4300, 6200, 500)
        s_eval = gs(w_eval)
        valid = np.isfinite(s_eval)
        assert np.sum(valid) > 400

    def test_full_pipeline_stitched(self):
        """Full pipeline with stitched approach on synthetic data."""
        segments, w_ref, f_ref = make_multi_segments(
            centers=[4500, 5000, 5500],
            noise_level=0.02,
        )

        segs_obs = [(s['wavelength'], s['flux_obs']) for s in segments]

        gs = derive_sensitivity(
            segs_obs, w_ref, f_ref,
            approach='stitched',
            fit_method='chebyshev',
            fit_order=6,
        )

        assert gs.approach == 'stitched'
        assert gs.global_fit is not None

    def test_save_load_evaluate_consistency(self, tmp_path):
        """Save, load, and re-evaluate should give identical results."""
        segments, w_ref, f_ref = make_multi_segments(
            centers=[5000, 5500],
            noise_level=0.01,
        )

        segs_obs = [(s['wavelength'], s['flux_obs']) for s in segments]
        gs = derive_sensitivity(
            segs_obs, w_ref, f_ref,
            approach='per_segment',
            segment_ids=['a', 'b'],
            fit_global=True,
        )

        fp = tmp_path / 'roundtrip.json'
        save_sensitivity(gs, fp)
        gs_loaded = load_sensitivity(fp)

        w_eval = np.linspace(4800, 5700, 300)
        s1 = gs(w_eval)
        s2 = gs_loaded(w_eval)
        valid = np.isfinite(s1) & np.isfinite(s2)
        np.testing.assert_allclose(s1[valid], s2[valid], rtol=1e-10)

    def test_with_masking(self):
        """Pipeline should work correctly when masks exclude stellar/telluric features."""
        segments, w_ref, f_ref = make_multi_segments(
            centers=[4850, 5350],  # includes H-beta at 4861
            noise_level=0.02,
        )

        segs_obs = []
        masks = []
        for seg in segments:
            w = seg['wavelength']
            segs_obs.append((w, seg['flux_obs']))
            # Build mask that excludes stellar lines
            m = build_sensitivity_mask(
                w, mask_telluric=True, mask_stellar=True,
                edge_fraction=0.05,
            )
            masks.append(m)

        gs = derive_sensitivity(
            segs_obs, w_ref, f_ref,
            approach='per_segment',
            masks=masks,
            fit_method='chebyshev',
            fit_order=5,
        )

        assert len(gs.segment_fits) >= 1
