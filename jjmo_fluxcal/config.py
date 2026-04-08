"""
config.py — Pipeline Configuration & Default Parameters
========================================================

Centralises all tunable parameters for the JJMO flux calibration pipeline
into a single configuration dataclass with documented defaults.  Parameters
can be overridden programmatically or via a YAML configuration file.

Usage
-----
    from jjmo_fluxcal.config import PipelineConfig

    # Use defaults
    cfg = PipelineConfig()

    # Override specific parameters
    cfg = PipelineConfig(fit_order=5, sigma_clip=2.5)

    # Load from YAML
    cfg = PipelineConfig.from_yaml("my_config.yaml")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Package data directory
# ---------------------------------------------------------------------------

_DATA_DIR = Path(__file__).parent / "data"


def _load_yaml(name: str) -> dict:
    """Load a YAML file from the package data directory."""
    path = _DATA_DIR / name
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Line lists (loaded once from bundled YAML)
# ---------------------------------------------------------------------------

_LINES = _load_yaml("line_lists.yaml")

BALMER_LINES: Dict[str, float] = _LINES["balmer_lines"]
METAL_LINES: Dict[str, float] = _LINES["metal_lines"]
TELLURIC_LINES: Dict[str, float] = _LINES["telluric_lines"]
ALL_STELLAR_LINES: Dict[str, float] = {**BALMER_LINES, **METAL_LINES}
ALL_KNOWN_LINES: Dict[str, float] = {**ALL_STELLAR_LINES, **TELLURIC_LINES}
TELLURIC_BANDS: List[Tuple[float, float]] = [
    tuple(b) for b in _LINES["telluric_bands"]
]

# Stellar parameter database
STANDARD_STAR_DB: dict = _load_yaml("stellar_params.yaml")

# Physical constants
C_KMS = 299792.458  # speed of light in km/s

# Default CCD parameters (SBIG ST-7 at JJMO)
DEFAULT_GAIN = 2.3             # e-/ADU
DEFAULT_READ_NOISE_E = 15.0    # electrons RMS per pixel
DEFAULT_DARK_CURRENT = 1.0     # e-/s/pixel at -10 C


# ---------------------------------------------------------------------------
# Pipeline configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """All tunable parameters for the JJMO flux calibration pipeline.

    Attributes
    ----------
    star_name : str
        Target star name (must be in STANDARD_STAR_DB or have overrides).
    input_dir : str or None
        Directory containing raw spectral data files.
    output_dir : str or None
        Directory for calibrated output and diagnostics.

    file_format : str or None
        Force file format ('fit_txt', 'csv', 'fits_1d') or None for auto.
    extensions : list of str or None
        File extensions to search for in input_dir.

    # Wavelength calibration (Step 2)
    smoothing_window : int or None
        Savitzky-Golay smoothing window for line finding (auto if None).
    smoothing_polyorder : int or None
        Savitzky-Golay polynomial order (auto if None).
    min_depth_sigma : float
        Minimum absorption line depth in sigma units.
    line_match_tolerance : float
        Maximum wavelength offset (A) for line matching.

    # Quality assessment (Step 3)
    edge_trim_frac : float
        Fractional threshold for edge vignetting detection.
    cosmic_sigma : float
        Sigma threshold for cosmic ray detection.
    telluric_bands : list of tuple
        Wavelength ranges to mask as telluric.

    # Stitching (Step 4)
    normalize_segments : bool
        Whether to cross-normalize segments before stitching.
    norm_method : str
        Cross-normalization method ('median_ratio' or 'polynomial').

    # Reference (Step 5)
    prefer_reference : str
        Preferred reference type ('calspec' or 'phoenix').
    ebv : float or None
        E(B-V) reddening override (uses database value if None).
    rv : float
        R_V for extinction law (default 3.1).
    extinction_law : str
        Interstellar extinction law ('odonnell', 'fitzpatrick', 'ccm').
    atmospheric_extinction : str
        Atmospheric extinction model ('kpno').

    # Sensitivity function (Step 6)
    fit_method : str
        Sensitivity fit method ('chebyshev', 'legendre', 'spline', 'savgol').
    fit_order : int
        Polynomial/spline order for sensitivity fit.
    sigma_clip : float
        Sigma-clipping threshold during sensitivity fitting.
    n_clip_iterations : int
        Number of sigma-clipping iterations.
    mask_balmer : bool
        Mask Balmer lines during sensitivity fitting.
    mask_telluric : bool
        Mask telluric bands during sensitivity fitting.
    mask_metals : bool
        Mask metal lines during sensitivity fitting.
    balmer_half_width : float
        Half-width (A) for Balmer line masks.
    metal_half_width : float
        Half-width (A) for metal line masks.

    # Calibration (Step 7)
    output_format : str
        Output format for calibrated spectra ('fits' or 'csv').

    # Uncertainty (Step 8)
    gain : float
        CCD gain in e-/ADU.
    read_noise : float
        Read noise in electrons RMS.
    dark_current : float
        Dark current in e-/s/pixel.
    n_monte_carlo : int
        Number of Monte Carlo realisations for uncertainty validation.

    # Logging
    log_level : str
        Logging verbosity ('DEBUG', 'INFO', 'WARNING', 'ERROR').
    """

    # Target
    star_name: str = "sirius"
    input_dir: Optional[str] = None
    output_dir: Optional[str] = None

    # File format
    file_format: Optional[str] = None
    extensions: Optional[List[str]] = None

    # Step 2: Wavelength calibration
    smoothing_window: Optional[int] = None
    smoothing_polyorder: Optional[int] = None
    min_depth_sigma: float = 3.0
    line_match_tolerance: float = 50.0

    # Step 3: Quality assessment
    edge_trim_frac: float = 0.20
    cosmic_sigma: float = 5.0
    telluric_bands: List[Tuple[float, float]] = field(
        default_factory=lambda: list(TELLURIC_BANDS)
    )

    # Step 4: Stitching
    normalize_segments: bool = True
    norm_method: str = "median_ratio"

    # Step 5: Reference
    prefer_reference: str = "calspec"
    ebv: Optional[float] = None
    rv: float = 3.1
    extinction_law: str = "odonnell"
    atmospheric_extinction: str = "kpno"

    # Step 6: Sensitivity function
    fit_method: str = "chebyshev"
    fit_order: int = 4
    sigma_clip: float = 3.0
    n_clip_iterations: int = 3
    mask_balmer: bool = True
    mask_telluric: bool = True
    mask_metals: bool = True
    balmer_half_width: float = 15.0
    metal_half_width: float = 5.0

    # Step 7: Calibration output
    output_format: str = "fits"

    # Step 8: Uncertainty propagation
    gain: float = DEFAULT_GAIN
    read_noise: float = DEFAULT_READ_NOISE_E
    dark_current: float = DEFAULT_DARK_CURRENT
    n_monte_carlo: int = 100

    # Logging
    log_level: str = "INFO"

    # Airmass (observing conditions)
    airmass: Optional[float] = None
    exptime: Optional[float] = None

    def to_dict(self) -> dict:
        """Serialise configuration to a plain dictionary."""
        from dataclasses import asdict
        return asdict(self)

    def save_yaml(self, path: str) -> None:
        """Write configuration to a YAML file."""
        d = self.to_dict()
        # Convert tuples to lists for YAML
        if "telluric_bands" in d:
            d["telluric_bands"] = [list(b) for b in d["telluric_bands"]]
        with open(path, "w") as f:
            yaml.dump(d, f, default_flow_style=False, sort_keys=False)
        logger.info("Configuration saved to %s", path)

    @classmethod
    def from_yaml(cls, path: str) -> "PipelineConfig":
        """Load configuration from a YAML file.

        Only keys that match PipelineConfig fields are used; unknown keys
        are logged as warnings and ignored.
        """
        with open(path, "r") as f:
            raw = yaml.safe_load(f) or {}

        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        overrides = {}
        for k, v in raw.items():
            if k in valid_fields:
                overrides[k] = v
            else:
                logger.warning("Unknown config key '%s' — ignored", k)

        # Convert telluric bands from lists to tuples
        if "telluric_bands" in overrides:
            overrides["telluric_bands"] = [
                tuple(b) for b in overrides["telluric_bands"]
            ]

        return cls(**overrides)

    @classmethod
    def from_dict(cls, d: dict) -> "PipelineConfig":
        """Create configuration from a dictionary, ignoring unknown keys."""
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in valid_fields}
        if "telluric_bands" in filtered:
            filtered["telluric_bands"] = [
                tuple(b) for b in filtered["telluric_bands"]
            ]
        return cls(**filtered)
