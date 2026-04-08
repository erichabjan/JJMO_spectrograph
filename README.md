# jjmo-fluxcal

Flux calibration pipeline for noisy, segmented spectra from small observatory spectrographs.

This package was developed for the [John J. McCarthy Observatory (JJMO)](https://www.mccarthyobservatory.org/) spectrograph in New Milford, Connecticut.  It processes ~500 A spectral segments across the optical range (3900--7900 A) and produces flux-calibrated spectra in physical units (erg/s/cm^2/A).

## Installation

```bash
# Basic install
pip install -e .

# Full install (includes optional dependencies for plotting, CALSPEC access, etc.)
pip install -e ".[full]"

# Development install (includes testing tools)
pip install -e ".[dev]"
```

## Quick Start

### Python API

```python
from jjmo_fluxcal import fluxcal

result = fluxcal(
    "./data/Sirius",
    star_name="sirius",
    output_dir="./results",
)
```

### Command Line

```bash
# Full pipeline
jjmo-fluxcal run --input-dir ./data/Sirius --star sirius --output-dir ./results

# Derive sensitivity function only
jjmo-fluxcal sensfunc --input-dir ./data/Sirius --star sirius --output sensfunc.json

# Apply saved sensitivity to new data
jjmo-fluxcal apply --sensfunc sensfunc.json --input science.fits --output calibrated.fits
```

### Configuration

All parameters can be set via YAML configuration file:

```bash
jjmo-fluxcal run --input-dir ./data/Sirius --config jjmo_fluxcal/examples/sirius_config.yaml
```

See `jjmo_fluxcal/examples/` for example configurations.

## Pipeline Steps

| Step | Module | Description |
|------|--------|-------------|
| 1 | `io.py` | Data ingestion & format auto-detection |
| 2 | `wavelength.py` | Absorption-line detection, wavelength calibration, velocity correction |
| 3 | `quality.py` | Edge trimming, cosmic ray rejection, SNR estimation, masking |
| 4 | `stitching.py` | Segment cross-normalization & combination |
| 5 | `reference.py` | CALSPEC/PHOENIX reference spectra, extinction correction |
| 6 | `sensitivity.py` | Sensitivity function derivation (Chebyshev/Legendre/spline fits) |
| 7 | `calibrate.py` | Flux calibration application, FITS/CSV output |
| 8 | `uncertainties.py` | Error propagation & Monte Carlo validation |

## Package Structure

```
jjmo_fluxcal/
    __init__.py          # High-level API (fluxcal one-call pipeline)
    io.py                # Step 1: readers, writers, format dispatcher
    wavelength.py        # Step 2: line finding, matching, velocity, correction
    quality.py           # Step 3: masking, cosmic rays, SNR, edge trimming
    stitching.py         # Step 4: overlap, cross-norm, combination
    reference.py         # Step 5: CALSPEC, PHOENIX models, extinction
    sensitivity.py       # Step 6: ratio, fitting, global combination
    calibrate.py         # Step 7: apply sensitivity, output calibrated spectra
    uncertainties.py     # Step 8: error propagation, Monte Carlo
    config.py            # Configuration dataclass with defaults
    cli.py               # Command-line interface
    plotting.py          # Diagnostic plot re-exports
    _logging.py          # Unified logging setup
    data/
        line_lists.yaml       # Balmer, metal, telluric line lists
        stellar_params.yaml   # Standard star parameters
    examples/
        sirius_config.yaml    # Example config for Sirius
        betelgeuse_config.yaml # Example config for Betelgeuse
tests/                   # Test suite (pytest)
notebooks/
    quickstart.ipynb     # Tutorial notebook
```

## Testing

```bash
pytest tests/ -v -m "not network"
```

## Dependencies

**Required:** numpy, scipy, astropy, specutils, pyyaml

**Optional (recommended):** spectres, specreduce, synphot, expecto, extinction, dust_extinction, matplotlib
