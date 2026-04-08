"""
Shared fixtures and configuration for the JJMO flux calibration test suite.
"""

import numpy as np
import pytest
from pathlib import Path


# ---------------------------------------------------------------------------
# Common data paths
# ---------------------------------------------------------------------------

DATA_ROOT = Path("/home/habjan.e/JJMO_home/Data")
SIRIUS_DIR = DATA_ROOT / "Sirius"
BETELGEUSE_DIR = DATA_ROOT / "Betelgeuse"

HAS_DATA = SIRIUS_DIR.exists() and BETELGEUSE_DIR.exists()

requires_data = pytest.mark.skipif(not HAS_DATA, reason="Data not found")


# ---------------------------------------------------------------------------
# Common fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_wavelength():
    """A simple optical wavelength grid (4000-7000 A, 1 A spacing)."""
    return np.arange(4000.0, 7001.0, 1.0)


@pytest.fixture
def simple_flux(simple_wavelength):
    """A simple smooth test flux (blackbody-like curve)."""
    wave = simple_wavelength
    T = 9000.0  # K
    h, c, k = 6.626e-27, 3.0e10, 1.381e-16  # cgs
    wave_cm = wave * 1e-8
    flux = 2 * h * c**2 / wave_cm**5 / (np.exp(h * c / (wave_cm * k * T)) - 1)
    return flux
