"""
test_quality.py — Test script for the quality assessment module (Step 3).

Tests the quality.py module against real JJMO Sirius and Betelgeuse data,
validates each component, and produces diagnostic plots.
"""

import sys
import os
import numpy as np
from astropy.io import fits

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from quality import (
    trim_edges, detect_cosmics_1d, detect_cosmics_2d,
    mask_telluric, mask_stellar_lines, estimate_snr,
    assess_segment, assess_segments, print_quality_table,
    plot_segment_quality, plot_quality_overview,
    mask_custom_regions, QualityReport,
    BALMER_LINES, METAL_LINES, TELLURIC_BANDS,
)


# ---- Data loading helpers ----

SIRIUS_DIR = "/home/habjan.e/JJMO_home/Data/Sirius/"
BETELGEUSE_DIR = "/home/habjan.e/JJMO_home/Data/Betelgeuse/"
SIRIUS_SEGMENTS = [3900, 4400, 4900, 5400, 5900, 6400, 6900, 7400]
BETELGEUSE_SEGMENTS = [4400, 4900, 5400, 5900, 6400, 6900, 7400]

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "quality_diagnostics")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_sirius_segment(seg_id):
    """Load a Sirius segment, returning (wavelength, flux_1d, image_2d)."""
    wave = np.genfromtxt(
        f"{SIRIUS_DIR}{seg_id}.txt", delimiter='\t',
        usecols=(1,), invalid_raise=False)
    hdu = fits.open(f"{SIRIUS_DIR}{seg_id}.fit")[0]
    image_2d = hdu.data.astype(float)
    flux_1d = np.flip(np.nansum(image_2d, axis=0))
    # Wavelength is descending in .txt files — sort both to ascending
    sort_idx = np.argsort(wave)
    wave = wave[sort_idx]
    flux_1d = flux_1d[sort_idx]
    return wave, flux_1d, image_2d


def load_betelgeuse_segment(seg_id):
    """Load a Betelgeuse segment, returning (wavelength, flux)."""
    data = np.genfromtxt(
        f"{BETELGEUSE_DIR}Betelgeuse_{seg_id}.csv",
        delimiter=',', skip_header=0)
    wave = data[:, 1]
    flux = data[:, 2]
    return wave, flux


# ---- Unit tests ----

def test_trim_edges():
    """Test edge trimming on synthetic and real data."""
    print("\n--- Test: Edge Trimming ---")

    # Synthetic: spectrum with low-flux edges
    n = 500
    wave = np.linspace(4000, 4500, n)
    flux = np.ones(n) * 1000.0
    flux[:30] = 50.0   # low blue edge
    flux[-20:] = 30.0  # low red edge
    flux[200] = 5000.0  # a spike — should not affect trimming

    mask, trim_b, trim_r = trim_edges(wave, flux, threshold_frac=0.20)
    assert trim_b > 4000.0, f"Blue trim should be inside segment, got {trim_b}"
    assert trim_r < 4500.0, f"Red trim should be inside segment, got {trim_r}"
    assert mask[250], "Interior pixel should be unmasked"
    assert not mask[0], "First pixel (low flux) should be masked"
    print(f"  Synthetic: trim [{trim_b:.1f}, {trim_r:.1f}] — "
          f"{np.sum(~mask)} pixels trimmed. PASS")

    # Real data: Sirius 3900 (likely has significant edge effects)
    wave_r, flux_r, _ = load_sirius_segment(3900)
    mask_r, tb, tr = trim_edges(wave_r, flux_r)
    print(f"  Sirius 3900: trim [{tb:.1f}, {tr:.1f}] — "
          f"{np.sum(~mask_r)}/{len(mask_r)} pixels trimmed")

    return True


def test_cosmic_detection():
    """Test cosmic ray detection on synthetic and real data."""
    print("\n--- Test: Cosmic Ray Detection ---")

    # Synthetic: smooth spectrum with cosmic ray spikes
    n = 500
    flux = np.random.normal(1000, 30, n)
    # Inject 5 cosmic rays
    cr_indices = [50, 120, 200, 350, 450]
    for idx in cr_indices:
        flux[idx] += 2000

    mask, n_cr = detect_cosmics_1d(flux, sigma_thresh=5.0, window_size=21)
    print(f"  Synthetic: injected {len(cr_indices)} CRs, detected {n_cr}. "
          f"{'PASS' if n_cr >= 4 else 'WARN (may have missed some)'}")

    # Real data: Sirius 4400 (check we don't flag too many)
    wave, flux_r, _ = load_sirius_segment(4400)
    mask_r, n_cr_r = detect_cosmics_1d(flux_r, sigma_thresh=5.0)
    print(f"  Sirius 4400: {n_cr_r} cosmic rays detected "
          f"({n_cr_r/len(flux_r)*100:.1f}% of pixels)")
    assert n_cr_r < len(flux_r) * 0.05, "Should not flag > 5% as cosmics"

    return True


def test_cosmic_2d():
    """Test 2D cosmic ray detection."""
    print("\n--- Test: 2D Cosmic Ray Detection ---")

    wave, flux_1d, image_2d = load_sirius_segment(4400)
    cr_mask, n_cr = detect_cosmics_2d(image_2d, sigma_thresh=5.0)
    total_pixels = image_2d.shape[0] * image_2d.shape[1]
    print(f"  Sirius 4400 2D: shape={image_2d.shape}, "
          f"{n_cr} CRs detected ({n_cr/total_pixels*100:.2f}% of pixels)")

    return True


def test_telluric_masking():
    """Test telluric region masking."""
    print("\n--- Test: Telluric Masking ---")

    # Segment that should have telluric bands (6900)
    wave, flux = load_sirius_segment(6900)[:2]
    mask, bands = mask_telluric(wave)
    print(f"  Sirius 6900: {len(bands)} band(s) found, "
          f"{np.sum(~mask)} pixels masked")
    assert len(bands) > 0, "6900 segment should contain telluric bands"

    # Segment that should NOT have telluric bands (4400)
    wave2, flux2 = load_sirius_segment(4400)[:2]
    mask2, bands2 = mask_telluric(wave2)
    print(f"  Sirius 4400: {len(bands2)} band(s) found, "
          f"{np.sum(~mask2)} pixels masked")

    return True


def test_stellar_masking():
    """Test stellar absorption line masking."""
    print("\n--- Test: Stellar Line Masking ---")

    # Segment with Hβ (4861 A) — should be in 4400 segment
    wave, flux = load_sirius_segment(4400)[:2]
    mask, lines = mask_stellar_lines(wave, velocity_shift=0.0)
    line_names = [f"{l['type']}@{l['rest_wave']:.0f}" for l in lines]
    print(f"  Sirius 4400: masked {len(lines)} lines: {', '.join(line_names)}")
    print(f"    {np.sum(~mask)} pixels masked")

    # Test with velocity shift
    mask_v, lines_v = mask_stellar_lines(wave, velocity_shift=50.0)
    if len(lines_v) > 0:
        shift = lines_v[0]['obs_wave'] - lines_v[0]['rest_wave']
        print(f"    With v=50 km/s: line shift = {shift:.3f} A")

    return True


def test_snr_estimation():
    """Test SNR estimation."""
    print("\n--- Test: SNR Estimation ---")

    # Synthetic Poisson-like data
    true_flux = 10000.0
    flux = np.random.poisson(true_flux, 500).astype(float)
    snr_pix, snr_med, method = estimate_snr(flux, method='poisson')
    expected_snr = np.sqrt(true_flux)
    print(f"  Synthetic (Poisson, counts={true_flux:.0f}): "
          f"median SNR={snr_med:.1f}, expected~{expected_snr:.1f}, "
          f"method={method}")

    # Real data
    for seg_id in [3900, 5400, 7400]:
        wave, flux_r, _ = load_sirius_segment(seg_id)
        mask = np.ones(len(flux_r), dtype=bool)
        snr_pix, snr_med, method = estimate_snr(flux_r, mask_good=mask)
        print(f"  Sirius {seg_id}: median SNR={snr_med:.1f} ({method})")

    return True


def test_assess_segment():
    """Test the full assessment pipeline on one segment."""
    print("\n--- Test: Full Pipeline (assess_segment) ---")

    wave, flux, _ = load_sirius_segment(6900)
    report = assess_segment(wave, flux, segment_id='Sirius_6900')
    print(report.summary())

    assert report.n_pixels_total == 765
    assert report.mask_good is not None
    assert report.snr_median > 0
    assert len(report.telluric_bands_found) > 0, "6900 should have telluric bands"

    return True


def test_full_sirius():
    """Run the full pipeline on all Sirius segments."""
    print("\n\n========== FULL SIRIUS ASSESSMENT ==========")

    wavelengths = []
    fluxes = []
    seg_ids = []

    for seg_id in SIRIUS_SEGMENTS:
        wave, flux, _ = load_sirius_segment(seg_id)
        wavelengths.append(wave)
        fluxes.append(flux)
        seg_ids.append(f'Sir_{seg_id}')

    reports = assess_segments(wavelengths, fluxes, segment_ids=seg_ids)

    print()
    print_quality_table(reports)

    # Print detailed summary for each
    for r in reports:
        print()
        print(r.summary())

    # Generate individual diagnostic plots
    for wave, flux, report in zip(wavelengths, fluxes, reports):
        path = os.path.join(OUTPUT_DIR, f"sirius_{report.segment_id}.png")
        plot_segment_quality(wave, flux, report, save_path=path)
        print(f"  Saved: {path}")

    # Overview plot
    overview_path = os.path.join(OUTPUT_DIR, "sirius_overview.png")
    plot_quality_overview(wavelengths, fluxes, reports, save_path=overview_path)
    print(f"  Saved: {overview_path}")

    return reports


def test_full_betelgeuse():
    """Run the full pipeline on all Betelgeuse segments."""
    print("\n\n========== FULL BETELGEUSE ASSESSMENT ==========")

    wavelengths = []
    fluxes = []
    seg_ids = []

    for seg_id in BETELGEUSE_SEGMENTS:
        wave, flux = load_betelgeuse_segment(seg_id)
        wavelengths.append(wave)
        fluxes.append(flux)
        seg_ids.append(f'Bet_{seg_id}')

    reports = assess_segments(wavelengths, fluxes, segment_ids=seg_ids)

    print()
    print_quality_table(reports)

    for r in reports:
        print()
        print(r.summary())

    # Generate diagnostic plots
    for wave, flux, report in zip(wavelengths, fluxes, reports):
        path = os.path.join(OUTPUT_DIR, f"betelgeuse_{report.segment_id}.png")
        plot_segment_quality(wave, flux, report, save_path=path)

    overview_path = os.path.join(OUTPUT_DIR, "betelgeuse_overview.png")
    plot_quality_overview(wavelengths, fluxes, reports, save_path=overview_path)
    print(f"  Saved: {overview_path}")

    return reports


# ---- Edge case tests ----

def test_edge_cases():
    """Test edge cases: constant flux, single pixel, all NaN, etc."""
    print("\n--- Test: Edge Cases ---")

    # Constant flux (no features)
    wave = np.linspace(5000, 5500, 100)
    flux = np.ones(100) * 5000.0
    report = assess_segment(wave, flux, segment_id='constant')
    print(f"  Constant flux: SNR={report.snr_median:.1f}, "
          f"usable={report.usable}")

    # Very low flux (near zero)
    flux_low = np.ones(100) * 5.0
    report_low = assess_segment(wave, flux_low, segment_id='low_flux')
    print(f"  Low flux (5 counts): SNR={report_low.snr_median:.1f}, "
          f"usable={report_low.usable}")

    # Spectrum with NaN/Inf
    flux_nan = np.ones(100) * 5000.0
    flux_nan[10:15] = np.nan
    flux_nan[50] = np.inf
    # Replace NaN/Inf before assessment (this is what Step 1 should do,
    # but we handle it gracefully here)
    flux_clean = np.where(np.isfinite(flux_nan), flux_nan, 0.0)
    report_nan = assess_segment(wave, flux_clean, segment_id='nan_cleaned')
    print(f"  NaN-cleaned flux: SNR={report_nan.snr_median:.1f}, "
          f"usable={report_nan.usable}")

    # Very short spectrum (fewer than 20 good pixels)
    wave_short = np.linspace(5000, 5010, 15)
    flux_short = np.ones(15) * 1000.0
    report_short = assess_segment(wave_short, flux_short, segment_id='short')
    print(f"  Short (15px): usable={report_short.usable}, "
          f"reason='{report_short.unusable_reason}'")

    return True


def test_custom_mask_regions():
    """Test custom mask region support."""
    print("\n--- Test: Custom Mask Regions ---")

    wave = np.linspace(5000, 5500, 500)
    flux = np.ones(500) * 5000.0
    regions = [(5100, 5150), (5300, 5350)]

    mask = mask_custom_regions(wave, regions)
    n_masked = np.sum(~mask)
    print(f"  Custom regions {regions}: {n_masked} pixels masked")
    assert n_masked > 0

    # Through the pipeline
    report = assess_segment(wave, flux, segment_id='custom_test',
                           custom_mask_regions=regions)
    print(f"  Pipeline with custom: {report.n_pixels_good}/{report.n_pixels_total} good")

    return True


def test_manual_trim_override():
    """Test manual trim override."""
    print("\n--- Test: Manual Trim Override ---")

    wave, flux, _ = load_sirius_segment(4400)
    report = assess_segment(wave, flux, segment_id='manual_trim',
                           manual_trim=(4000, 4700))
    print(f"  Manual trim [4000, 4700]: "
          f"range [{report.wave_min:.0f}, {report.wave_max:.0f}], "
          f"{report.n_pixels_edge_trimmed} trimmed")
    return True


# ---- Main ----

if __name__ == '__main__':
    print("=" * 60)
    print("JJMO Quality Assessment Module — Test Suite")
    print("=" * 60)

    # Unit tests
    tests = [
        test_trim_edges,
        test_cosmic_detection,
        test_cosmic_2d,
        test_telluric_masking,
        test_stellar_masking,
        test_snr_estimation,
        test_assess_segment,
        test_edge_cases,
        test_custom_mask_regions,
        test_manual_trim_override,
    ]

    passed = 0
    failed = 0
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*60}")
    print(f"Unit tests: {passed} passed, {failed} failed")
    print(f"{'='*60}")

    # Full dataset runs
    sirius_reports = test_full_sirius()
    betelgeuse_reports = test_full_betelgeuse()

    print(f"\nDiagnostic plots saved to: {OUTPUT_DIR}")
    print("Done.")
