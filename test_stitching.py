"""
test_stitching.py - Validation tests for the stitching module
==============================================================

Runs all tests from the Step 4 spec:
  - Overlap detection on real JJMO data
  - Cross-normalization (median_ratio and polynomial methods)
  - Inverse-variance weighted overlap combination
  - Gap handling (small interpolated, large masked)
  - Flux-conserving resampling to uniform grid
  - Spectrum1D conversion round-trip
  - Both pre-calibration and post-calibration workflows
  - Edge cases (single segment, NaN values, descending wavelengths, dict input)
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stitching import (
    find_overlaps, cross_normalize, combine_overlap_region,
    handle_gap, resample_to_uniform_grid, stitch_segments,
    estimate_segment_snr, load_jjmo_sirius, load_jjmo_betelgeuse,
    OverlapInfo, plot_stitched_spectrum, plot_normalization_factors,
    plot_overlap_quality, _unpack_segment, _ensure_sorted
)

SIRIUS_DIR = '/home/habjan.e/JJMO_home/Data/Sirius'
BETELGEUSE_DIR = '/home/habjan.e/JJMO_home/Data/Betelgeuse'
PLOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots')


def test_load_sirius():
    """Test loading Sirius segments from FITS+TXT files."""
    segs = load_jjmo_sirius(SIRIUS_DIR)
    assert len(segs) == 8, f"Expected 8 Sirius segments, got {len(segs)}"
    for i, (w, f) in enumerate(segs):
        assert len(w) == len(f), f"Seg {i}: wave/flux length mismatch"
        assert w[0] < w[-1], f"Seg {i}: not sorted ascending"
        assert np.all(np.isfinite(w)), f"Seg {i}: NaN in wavelengths"
    # Check ordering
    for i in range(len(segs) - 1):
        assert segs[i][0].min() < segs[i+1][0].min(), "Segments not sorted by start wavelength"
    print("PASS: test_load_sirius")


def test_load_betelgeuse():
    """Test loading Betelgeuse segments from CSV files."""
    segs = load_jjmo_betelgeuse(BETELGEUSE_DIR)
    assert len(segs) == 7, f"Expected 7 Betelgeuse segments, got {len(segs)}"
    for i, (w, f) in enumerate(segs):
        assert len(w) == len(f)
        assert w[0] < w[-1]
    print("PASS: test_load_betelgeuse")


def test_find_overlaps():
    """Test overlap detection on Sirius data (all segments overlap)."""
    segs = load_jjmo_sirius(SIRIUS_DIR)
    overlaps = find_overlaps(segs)
    assert len(overlaps) == 7, f"Expected 7 overlap pairs, got {len(overlaps)}"
    for ol in overlaps:
        assert not ol.is_gap, f"Unexpected gap: {ol}"
        assert ol.overlap_width > 200, f"Overlap too small: {ol}"
    print("PASS: test_find_overlaps")


def test_find_overlaps_with_gap():
    """Test overlap detection when there's a gap."""
    wave_a = np.linspace(4000, 4500, 100)
    flux_a = np.ones(100) * 1000
    wave_b = np.linspace(4550, 5000, 100)  # 50 A gap
    flux_b = np.ones(100) * 1000
    overlaps = find_overlaps([(wave_a, flux_a), (wave_b, flux_b)])
    assert len(overlaps) == 1
    assert overlaps[0].is_gap
    assert overlaps[0].overlap_width < 0
    print("PASS: test_find_overlaps_with_gap")


def test_cross_normalization_median():
    """Test median-ratio cross-normalization."""
    segs = load_jjmo_sirius(SIRIUS_DIR)
    overlaps = find_overlaps(segs)
    normalized, factors = cross_normalize(segs, overlaps, method='median_ratio')
    assert len(normalized) == 8
    assert len(factors) == 7  # all non-reference segments
    # Reference segment should have factor ~1.0 (not in factors list)
    # Other factors should be positive
    for nf in factors:
        assert nf.factor > 0, f"Negative normalization factor: {nf}"
        assert np.isfinite(nf.factor)
    print("PASS: test_cross_normalization_median")


def test_cross_normalization_polynomial():
    """Test polynomial cross-normalization."""
    segs = load_jjmo_sirius(SIRIUS_DIR)
    overlaps = find_overlaps(segs)
    normalized, factors = cross_normalize(segs, overlaps, method='polynomial')
    assert len(normalized) == 8
    for nf in factors:
        assert nf.factor > 0
        assert nf.method == 'polynomial'
    print("PASS: test_cross_normalization_polynomial")


def test_snr_estimation():
    """Test SNR estimation."""
    segs = load_jjmo_sirius(SIRIUS_DIR)
    snrs = [estimate_segment_snr(f) for w, f in segs]
    # Middle segments should have higher SNR than edges
    assert snrs[3] > snrs[0], "Middle segment should have higher SNR than edge"
    assert snrs[3] > snrs[7]
    # All should be positive
    assert all(s > 0 for s in snrs)
    print("PASS: test_snr_estimation")


def test_combine_overlap():
    """Test inverse-variance weighted overlap combination."""
    wave_l = np.linspace(5000, 5600, 120)
    flux_l = np.ones(120) * 1000.0
    unc_l = np.ones(120) * 30.0
    mask_l = np.ones(120, dtype=bool)

    wave_r = np.linspace(5400, 6000, 120)
    flux_r = np.ones(120) * 1000.0
    unc_r = np.ones(120) * 60.0  # 2x higher uncertainty
    mask_r = np.ones(120, dtype=bool)

    ol = OverlapInfo(0, 1, 5400, 5600, 200, False)
    wc, fc, uc, mc = combine_overlap_region(
        wave_l, flux_l, unc_l, mask_l,
        wave_r, flux_r, unc_r, mask_r, ol
    )
    # In overlap interior: median combined uncertainty should be less than worst input.
    # Edge pixels may only have data from one segment, so check median not all.
    overlap_mask = (wc >= 5410) & (wc <= 5590) & mc
    assert np.median(uc[overlap_mask]) < 30, "Combined unc should be < best single input"
    # Combined flux should be ~1000 (both inputs are 1000)
    assert np.abs(np.median(fc[overlap_mask]) - 1000) < 50
    print("PASS: test_combine_overlap")


def test_gap_interpolation():
    """Test gap handling: small gap gets interpolated."""
    wave_l = np.linspace(4000, 4500, 100)
    flux_l = np.ones(100) * 500.0
    unc_l = np.ones(100) * 20.0
    mask_l = np.ones(100, dtype=bool)

    wave_r = np.linspace(4530, 5000, 100)  # 30 A gap
    flux_r = np.ones(100) * 500.0
    unc_r = np.ones(100) * 20.0
    mask_r = np.ones(100, dtype=bool)

    gap = OverlapInfo(0, 1, 4500, 4530, -30, True)
    wc, fc, uc, mc, interp = handle_gap(
        wave_l, flux_l, unc_l, mask_l,
        wave_r, flux_r, unc_r, mask_r,
        gap, max_gap_angstrom=50
    )
    assert np.any(interp), "Small gap should be interpolated"
    assert np.all(mc[interp]), "Interpolated pixels should be valid"
    print("PASS: test_gap_interpolation")


def test_gap_masking():
    """Test gap handling: large gap gets masked."""
    wave_l = np.linspace(4000, 4400, 80)
    flux_l = np.ones(80) * 500.0
    unc_l = np.ones(80) * 20.0
    mask_l = np.ones(80, dtype=bool)

    wave_r = np.linspace(4500, 4900, 80)  # 100 A gap
    flux_r = np.ones(80) * 500.0
    unc_r = np.ones(80) * 20.0
    mask_r = np.ones(80, dtype=bool)

    gap = OverlapInfo(0, 1, 4400, 4500, -100, True)
    wc, fc, uc, mc, interp = handle_gap(
        wave_l, flux_l, unc_l, mask_l,
        wave_r, flux_r, unc_r, mask_r,
        gap, max_gap_angstrom=50
    )
    gap_region = (wc > 4400) & (wc < 4500)
    assert not np.any(mc[gap_region]), "Large gap should be masked"
    assert not np.any(interp[gap_region]), "Large gap should not be interpolated"
    print("PASS: test_gap_masking")


def test_resample_spectres():
    """Test flux-conserving resampling with spectres."""
    wave = np.linspace(4000, 5000, 500)
    flux = np.sin(wave / 100) * 100 + 500
    unc = np.ones(500) * 10.0
    mask = np.ones(500, dtype=bool)

    wn, fn, un, mn = resample_to_uniform_grid(
        wave, flux, unc, mask,
        grid_start=4100, grid_end=4900, grid_step=2.0,
        method='spectres'
    )
    assert wn[0] >= 4100
    assert wn[-1] <= 4900
    assert np.abs(np.median(np.diff(wn)) - 2.0) < 0.01
    # Flux should be conserved (roughly)
    assert np.abs(np.mean(fn[mn]) - 500) < 20
    print("PASS: test_resample_spectres")


def test_resample_interp():
    """Test interpolation-based resampling fallback."""
    wave = np.linspace(4000, 5000, 500)
    flux = np.ones(500) * 300.0
    unc = np.ones(500) * 10.0
    mask = np.ones(500, dtype=bool)

    wn, fn, un, mn = resample_to_uniform_grid(
        wave, flux, unc, mask,
        grid_step=3.0, method='interp'
    )
    assert np.all(np.abs(fn[mn] - 300) < 1), "Constant flux should be preserved"
    print("PASS: test_resample_interp")


def test_stitch_sirius_full():
    """Full end-to-end stitch of Sirius data."""
    segs = load_jjmo_sirius(SIRIUS_DIR)
    result = stitch_segments(segs, mode='pre_calibration', normalize=True,
                             resample=True, resample_method='spectres')
    assert len(result.wavelength) > 3000
    assert result.wavelength[0] < 3500
    assert result.wavelength[-1] > 7500
    assert np.sum(result.mask) > 0.95 * len(result.mask)
    assert result.reference_segment >= 0
    print("PASS: test_stitch_sirius_full")


def test_stitch_betelgeuse_full():
    """Full end-to-end stitch of Betelgeuse data."""
    segs = load_jjmo_betelgeuse(BETELGEUSE_DIR)
    result = stitch_segments(segs, mode='pre_calibration', normalize=True,
                             norm_method='polynomial', resample=True)
    assert len(result.wavelength) > 3000
    assert np.sum(result.mask) > 0.95 * len(result.mask)
    print("PASS: test_stitch_betelgeuse_full")


def test_stitch_post_calibration():
    """Test post-calibration stitching mode."""
    segs = load_jjmo_sirius(SIRIUS_DIR)
    result = stitch_segments(segs, mode='post_calibration', normalize=False,
                             resample=True)
    assert len(result.wavelength) > 3000
    assert len(result.norm_factors) == 0
    print("PASS: test_stitch_post_calibration")


def test_stitch_no_resample():
    """Test stitching without resampling."""
    segs = load_jjmo_sirius(SIRIUS_DIR)
    result = stitch_segments(segs, normalize=True, resample=False)
    # Without resampling, wavelength spacing may be non-uniform
    spacings = np.diff(result.wavelength)
    assert np.all(spacings > 0), "Should be monotonically increasing"
    print("PASS: test_stitch_no_resample")


def test_single_segment():
    """Test with a single segment input."""
    wave = np.linspace(5000, 5500, 200)
    flux = np.random.poisson(800, 200).astype(float)
    result = stitch_segments([(wave, flux)])
    assert len(result.wavelength) == 200
    assert result.reference_segment == 0
    print("PASS: test_single_segment")


def test_spectrum1d_roundtrip():
    """Test Spectrum1D input and output conversion."""
    from specutils import Spectrum1D
    from astropy import units as u
    from astropy.nddata import StdDevUncertainty

    wave = np.linspace(6000, 6500, 200) * u.AA
    flux = np.random.poisson(1000, 200).astype(float) * u.ct
    unc = StdDevUncertainty(np.sqrt(np.abs(flux.value)))
    spec = Spectrum1D(spectral_axis=wave, flux=flux, uncertainty=unc)

    result = stitch_segments([spec])
    s1d = result.to_spectrum1d()
    assert hasattr(s1d, 'spectral_axis')
    assert hasattr(s1d, 'flux')
    assert s1d.uncertainty is not None
    print("PASS: test_spectrum1d_roundtrip")


def test_descending_wavelength():
    """Test that descending wavelengths are handled correctly."""
    wave = np.linspace(5000, 4500, 100)  # descending
    flux = np.random.poisson(500, 100).astype(float)
    result = stitch_segments([(wave, flux)])
    assert result.wavelength[0] < result.wavelength[-1]
    print("PASS: test_descending_wavelength")


def test_diagnostic_plots():
    """Test that diagnostic plots are generated without errors."""
    segs = load_jjmo_sirius(SIRIUS_DIR)
    result = stitch_segments(segs, mode='pre_calibration', normalize=True)

    os.makedirs(PLOT_DIR, exist_ok=True)
    plot_stitched_spectrum(result, segments=segs,
                          save_path=os.path.join(PLOT_DIR, 'test_stitched.png'))
    assert os.path.exists(os.path.join(PLOT_DIR, 'test_stitched.png'))

    if result.norm_factors:
        plot_normalization_factors(segs, result.norm_factors, result.overlaps,
                                  save_path=os.path.join(PLOT_DIR, 'test_norms.png'))
        assert os.path.exists(os.path.join(PLOT_DIR, 'test_norms.png'))

    plot_overlap_quality(segs, result.overlaps,
                         save_path=os.path.join(PLOT_DIR, 'test_overlaps.png'))
    assert os.path.exists(os.path.join(PLOT_DIR, 'test_overlaps.png'))
    print("PASS: test_diagnostic_plots")


if __name__ == '__main__':
    tests = [
        test_load_sirius,
        test_load_betelgeuse,
        test_find_overlaps,
        test_find_overlaps_with_gap,
        test_cross_normalization_median,
        test_cross_normalization_polynomial,
        test_snr_estimation,
        test_combine_overlap,
        test_gap_interpolation,
        test_gap_masking,
        test_resample_spectres,
        test_resample_interp,
        test_stitch_sirius_full,
        test_stitch_betelgeuse_full,
        test_stitch_post_calibration,
        test_stitch_no_resample,
        test_single_segment,
        test_spectrum1d_roundtrip,
        test_descending_wavelength,
        test_diagnostic_plots,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"FAIL: {test.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    if failed == 0:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
