"""
reference.py -- Reference Spectrum Selection & Loading
======================================================

Step 5 of the JJMO Spectral Flux Calibration Pipeline.

Provides access to trusted flux-calibrated reference spectra for known
standard stars from CALSPEC (via MAST), PHOENIX model atmospheres (via
expecto), and tabulated spectrophotometric standard databases (via
specreduce).  Also handles interstellar and atmospheric extinction
correction, flux-conserving resampling, resolution matching, and feature
masking to prepare reference spectra for sensitivity-function derivation.

The module works with specutils Spectrum/Spectrum1D objects as the canonical
container, consistent with earlier pipeline steps.

Dependencies
------------
- specreduce (CALSPEC access, atmospheric extinction curves)
- synphot (required by specreduce for CALSPEC FITS parsing)
- expecto (PHOENIX model atmosphere download)
- extinction (interstellar extinction laws: O'Donnell 94, Fitzpatrick 99, CCM89)
- dust_extinction (astropy-affiliated extinction models, used as fallback)
- spectres (flux-conserving resampling)
- scipy (Gaussian convolution for resolution matching)
"""

import logging
import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.ndimage import gaussian_filter1d

import astropy.units as u

try:
    from specutils import Spectrum as Spectrum1D
except ImportError:
    from specutils import Spectrum1D

logger = logging.getLogger(__name__)


# ============================================================================
# 5.3 -- Stellar parameter database for known standard stars
# ============================================================================

# Each entry: {name: {teff, logg, feh, ebv, calspec_file, spectral_type, notes}}
# Parameters sourced from the literature:
#   Sirius: Liebert et al. 2005 (Teff=9940, logg=4.33), Bohlin 2014 CALSPEC
#   Vega: Bohlin 2014, Castelli & Kurucz 2004
#   Betelgeuse: Levesque et al. 2005, Harper et al. 2008
#   WD standards: Bohlin et al. 2014, 2020 CALSPEC documentation
STANDARD_STAR_DB = {
    "sirius": {
        "teff": 9940,
        "logg": 4.33,
        "feh": 0.5,
        "ebv": 0.0,          # negligible for Sirius at 2.6 pc
        "calspec_file": "sirius_stis_001.fits",
        "spectral_type": "A1V",
        "aliases": ["alpha_cma", "alpha cma", "hd 48915", "hr 2491"],
        "notes": "Primary CALSPEC standard, Teff/logg from Liebert+2005",
    },
    "vega": {
        "teff": 9550,
        "logg": 3.95,
        "feh": -0.5,
        "ebv": 0.0,
        "calspec_file": "alpha_lyr_stis_011.fits",
        "spectral_type": "A0V",
        "aliases": ["alpha_lyr", "alpha lyr", "hd 172167", "hr 7001"],
        "notes": "Fundamental CALSPEC standard, Bohlin+2014",
    },
    "betelgeuse": {
        "teff": 3600,
        "logg": 0.0,
        "feh": 0.0,
        "ebv": 0.15,         # Harper et al. 2008
        "calspec_file": None,  # Not in CALSPEC (variable red supergiant)
        "spectral_type": "M1Iab",
        "aliases": ["alpha_ori", "alpha ori", "hd 39801", "hr 2061"],
        "notes": "Not a CALSPEC standard; use PHOENIX model. E(B-V) from Harper+2008",
    },
    "g191b2b": {
        "teff": 57000,
        "logg": 7.5,
        "feh": 0.0,
        "ebv": 0.0,
        "calspec_file": "g191b2b_mod_011.fits",
        "spectral_type": "DA0",
        "aliases": ["g191-b2b", "wd 0501+527"],
        "notes": "Hot WD primary standard, Bohlin+2014/2020",
    },
    "gd153": {
        "teff": 40000,
        "logg": 7.8,
        "feh": 0.0,
        "ebv": 0.0,
        "calspec_file": "gd153_mod_012.fits",
        "spectral_type": "DA1",
        "aliases": ["gd 153", "wd 1254+223"],
        "notes": "WD primary standard, Bohlin+2014/2020",
    },
    "bd28d4211": {
        "teff": 82000,
        "logg": 6.2,
        "feh": 0.0,
        "ebv": 0.0,
        "calspec_file": "bd_28d4211_stis_003.fits",
        "spectral_type": "Op",
        "aliases": ["bd+28 4211", "bd+28d4211"],
        "notes": "Hot subdwarf standard",
    },
}


def _resolve_star_name(name):
    """Resolve a star name (or alias) to its canonical key in STANDARD_STAR_DB.

    Case-insensitive matching against both keys and aliases.

    Returns
    -------
    str or None
        Canonical key if found, else None.
    """
    name_lower = name.strip().lower()

    # Direct match on key
    if name_lower in STANDARD_STAR_DB:
        return name_lower

    # Search aliases
    for key, entry in STANDARD_STAR_DB.items():
        for alias in entry.get("aliases", []):
            if alias.lower() == name_lower:
                return key

    return None


def get_stellar_parameters(star_name, **overrides):
    """Look up stellar parameters for a known standard star.

    Parameters
    ----------
    star_name : str
        Star name or alias (case-insensitive).
    **overrides
        Any parameter to override (teff, logg, feh, ebv).

    Returns
    -------
    dict
        Dictionary with keys: teff, logg, feh, ebv, calspec_file,
        spectral_type, notes.

    Raises
    ------
    ValueError
        If star_name is not found in the database.
    """
    key = _resolve_star_name(star_name)
    if key is None:
        known = list(STANDARD_STAR_DB.keys())
        raise ValueError(
            f"Star '{star_name}' not found in standard star database. "
            f"Known stars: {known}"
        )

    params = dict(STANDARD_STAR_DB[key])  # shallow copy
    # Apply user overrides
    for k, v in overrides.items():
        if k in params:
            params[k] = v
        else:
            logger.warning("Unknown parameter override '%s' ignored", k)

    return params


def list_standard_stars():
    """Return a summary of all stars in the standard star database.

    Returns
    -------
    dict
        {canonical_name: {spectral_type, teff, calspec_file, aliases}}
    """
    summary = {}
    for key, entry in STANDARD_STAR_DB.items():
        summary[key] = {
            "spectral_type": entry["spectral_type"],
            "teff": entry["teff"],
            "calspec_file": entry["calspec_file"],
            "aliases": entry.get("aliases", []),
        }
    return summary


# ============================================================================
# 5.1 -- CALSPEC standard star access
# ============================================================================

def load_calspec(star_name=None, *, calspec_file=None, cache=True):
    """Load a CALSPEC flux-calibrated reference spectrum from MAST.

    The spectrum is returned as a Spectrum1D with physical flux units.
    CALSPEC spectra are delivered in milli-Jansky by specreduce; this
    function converts them to erg/s/cm^2/A (F_lambda) for consistency
    with the pipeline convention.

    Parameters
    ----------
    star_name : str, optional
        Star name to look up in the database for its CALSPEC filename.
    calspec_file : str, optional
        Direct CALSPEC FITS filename (overrides star_name lookup).
    cache : bool
        Cache downloaded files locally (default True).

    Returns
    -------
    Spectrum1D
        Reference spectrum with spectral_axis in Angstrom and flux in
        erg / (s cm^2 Angstrom).

    Raises
    ------
    ValueError
        If the star has no CALSPEC spectrum or loading fails.
    """
    from specreduce.calibration_data import load_MAST_calspec

    if calspec_file is None:
        if star_name is None:
            raise ValueError("Must specify either star_name or calspec_file")
        params = get_stellar_parameters(star_name)
        calspec_file = params.get("calspec_file")
        if calspec_file is None:
            raise ValueError(
                f"Star '{star_name}' does not have a CALSPEC spectrum. "
                f"Use load_phoenix_model() or load_model_spectrum() instead."
            )

    logger.info("Loading CALSPEC spectrum: %s", calspec_file)
    spec = load_MAST_calspec(calspec_file, cache=cache, show_progress=False)

    if spec is None:
        raise ValueError(
            f"Failed to download CALSPEC spectrum '{calspec_file}' from MAST. "
            f"Check your internet connection and the filename."
        )

    # Convert from mJy to F_lambda (erg/s/cm^2/A)
    # F_lambda = F_nu * c / lambda^2, where F_nu is in cgs from mJy
    spec = _convert_to_flambda(spec)

    logger.info(
        "Loaded CALSPEC %s: %.1f - %.1f A, %d pixels",
        calspec_file,
        spec.spectral_axis.value.min(),
        spec.spectral_axis.value.max(),
        len(spec.flux),
    )
    return spec


def _convert_to_flambda(spec):
    """Convert a spectrum from mJy (F_nu) to erg/s/cm^2/A (F_lambda).

    Uses the relation: F_lambda = F_nu * c / lambda^2
    where lambda is in cm and F_nu is in erg/s/cm^2/Hz.
    1 mJy = 1e-26 erg/s/cm^2/Hz.
    """
    wave_aa = spec.spectral_axis.to(u.AA).value
    flux_mJy = spec.flux.value

    # Check if already in F_lambda-like units; mJy has no /A dimension
    # mJy -> erg/s/cm^2/Hz -> erg/s/cm^2/A
    c_aa_per_s = 2.99792458e18  # speed of light in Angstrom/s
    flux_fnu_cgs = flux_mJy * 1.0e-26  # mJy -> erg/s/cm^2/Hz
    flux_flambda = flux_fnu_cgs * c_aa_per_s / wave_aa**2  # erg/s/cm^2/A

    return Spectrum1D(
        flux=flux_flambda * u.erg / u.s / u.cm**2 / u.AA,
        spectral_axis=wave_aa * u.AA,
        meta=dict(getattr(spec, 'meta', {})),
    )


# ============================================================================
# 5.2 -- Synthetic model grid loading (PHOENIX via expecto)
# ============================================================================

def load_phoenix_model(teff, logg, feh=0.0, *, cache=True, air=True):
    """Load a PHOENIX model atmosphere spectrum via expecto.

    Parameters
    ----------
    teff : float
        Effective temperature in Kelvin. Rounded to the nearest grid point
        by expecto internally.
    logg : float
        Surface gravity (log10, cgs). Rounded to nearest grid point.
    feh : float
        Metallicity [Fe/H]. Default 0.0. Rounded to nearest grid point.
    cache : bool
        Cache the downloaded model (default True).
    air : bool
        If True, return air wavelengths (default). JJMO data is in air.

    Returns
    -------
    Spectrum1D
        Model spectrum in native PHOENIX units (erg/s/cm^2/cm i.e. per cm),
        converted to erg/s/cm^2/A for pipeline consistency.
    """
    from expecto import get_spectrum

    logger.info(
        "Loading PHOENIX model: Teff=%d, logg=%.2f, [Fe/H]=%.2f",
        teff, logg, feh,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        spec = get_spectrum(
            T_eff=teff, log_g=logg, Z=feh,
            cache=cache, vacuum=(not air),
        )

    # expecto returns flux in erg/s/cm^3 (i.e. erg/s/cm^2/cm).
    # Convert to erg/s/cm^2/A by dividing by 1e8 (1 cm = 1e8 A).
    wave_aa = spec.spectral_axis.to(u.AA).value
    flux_cgs = spec.flux.value  # erg/s/cm^3

    # erg/s/cm^2/cm -> erg/s/cm^2/A
    flux_flambda = flux_cgs / 1.0e8

    result = Spectrum1D(
        flux=flux_flambda * u.erg / u.s / u.cm**2 / u.AA,
        spectral_axis=wave_aa * u.AA,
        meta={"source": "PHOENIX/expecto", "teff": teff, "logg": logg, "feh": feh},
    )

    logger.info(
        "Loaded PHOENIX model: %.1f - %.1f A, %d pixels",
        wave_aa.min(), wave_aa.max(), len(wave_aa),
    )
    return result


def load_model_spectrum(star_name=None, *, teff=None, logg=None, feh=None,
                        cache=True, air=True):
    """Load a PHOENIX model spectrum, optionally looking up parameters by name.

    If star_name is given, parameters are looked up in the standard star
    database.  Any explicit teff/logg/feh values override the database.

    Parameters
    ----------
    star_name : str, optional
        Star name for parameter lookup.
    teff, logg, feh : float, optional
        Override stellar parameters.
    cache, air : bool
        Passed to load_phoenix_model().

    Returns
    -------
    Spectrum1D
    """
    if star_name is not None:
        params = get_stellar_parameters(star_name)
        if teff is None:
            teff = params["teff"]
        if logg is None:
            logg = params["logg"]
        if feh is None:
            feh = params["feh"]

    if teff is None or logg is None:
        raise ValueError(
            "Must specify teff and logg, either directly or via star_name"
        )
    if feh is None:
        feh = 0.0

    return load_phoenix_model(teff, logg, feh, cache=cache, air=air)


# ============================================================================
# 5.1 continued -- Unified reference spectrum loader
# ============================================================================

def load_reference_spectrum(star_name, *, prefer="calspec", teff=None,
                            logg=None, feh=None, calspec_file=None,
                            cache=True):
    """Load the best available reference spectrum for a star.

    Tries CALSPEC first (if available and preferred), then falls back to
    PHOENIX model.  The user can force a specific source via the `prefer`
    parameter.

    Parameters
    ----------
    star_name : str
        Star name or alias.
    prefer : str
        "calspec" (default) or "model". Controls which source is tried first.
    teff, logg, feh : float, optional
        Override stellar parameters for model loading.
    calspec_file : str, optional
        Override CALSPEC filename.
    cache : bool
        Cache downloads.

    Returns
    -------
    Spectrum1D
        Reference spectrum in erg/s/cm^2/A.
    """
    params = get_stellar_parameters(star_name)
    has_calspec = calspec_file or params.get("calspec_file")

    if prefer == "calspec" and has_calspec:
        try:
            return load_calspec(
                star_name=star_name, calspec_file=calspec_file, cache=cache
            )
        except (ValueError, Exception) as exc:
            logger.warning(
                "CALSPEC loading failed for '%s': %s. Falling back to model.",
                star_name, exc,
            )

    # Fall back (or primary choice) to PHOENIX model
    return load_model_spectrum(
        star_name=star_name, teff=teff, logg=logg, feh=feh,
        cache=cache, air=True,
    )


# ============================================================================
# 5.4 -- Extinction correction
# ============================================================================

# --- 5.4a: Interstellar extinction ---

EXTINCTION_LAWS = {
    "odonnell94": "O'Donnell 1994 (recommended for optical)",
    "fitzpatrick99": "Fitzpatrick 1999",
    "ccm89": "Cardelli, Clayton & Mathis 1989",
}


def apply_interstellar_extinction(wavelength, flux, ebv, *, rv=3.1,
                                  law="odonnell94"):
    """Correct (deredden) observed flux for interstellar extinction.

    Removes the effect of dust reddening: returns the flux the source
    would have if there were no intervening dust.

    Parameters
    ----------
    wavelength : array-like
        Wavelength in Angstroms.
    flux : array-like
        Observed flux (any units; returned in same units).
    ebv : float
        Colour excess E(B-V) in magnitudes.
    rv : float
        Total-to-selective extinction ratio R_V (default 3.1 for diffuse ISM).
    law : str
        Extinction law: "odonnell94", "fitzpatrick99", or "ccm89".

    Returns
    -------
    flux_corrected : np.ndarray
        Dereddened flux.
    """
    import extinction as ext_pkg

    wavelength = np.asarray(wavelength, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64)
    av = ebv * rv

    law_map = {
        "odonnell94": ext_pkg.odonnell94,
        "fitzpatrick99": ext_pkg.fitzpatrick99,
        "ccm89": ext_pkg.ccm89,
    }

    if law not in law_map:
        raise ValueError(
            f"Unknown extinction law '{law}'. Choose from: {list(law_map.keys())}"
        )

    ext_curve = law_map[law](wavelength, av, rv)
    # ext_pkg.remove divides out the extinction (dereddens)
    flux_corrected = ext_pkg.remove(ext_curve, flux)

    logger.info(
        "Applied %s dereddening: E(B-V)=%.4f, R_V=%.2f, "
        "A_V=%.4f, range=%.1f-%.1f A",
        law, ebv, rv, av, wavelength.min(), wavelength.max(),
    )
    return flux_corrected


def apply_interstellar_reddening(wavelength, flux, ebv, *, rv=3.1,
                                 law="odonnell94"):
    """Apply interstellar extinction (redden) to a model/reference flux.

    This is the inverse of dereddening: dims and reddens the flux to
    simulate the effect of dust.  Useful for reddening a model to
    match reddened observed data, rather than dereddening the observed
    data (which amplifies noise in the blue).

    Parameters
    ----------
    wavelength, flux, ebv, rv, law
        Same as apply_interstellar_extinction.

    Returns
    -------
    flux_reddened : np.ndarray
    """
    import extinction as ext_pkg

    wavelength = np.asarray(wavelength, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64)
    av = ebv * rv

    law_map = {
        "odonnell94": ext_pkg.odonnell94,
        "fitzpatrick99": ext_pkg.fitzpatrick99,
        "ccm89": ext_pkg.ccm89,
    }

    if law not in law_map:
        raise ValueError(
            f"Unknown extinction law '{law}'. Choose from: {list(law_map.keys())}"
        )

    ext_curve = law_map[law](wavelength, av, rv)
    flux_reddened = ext_pkg.apply(ext_curve, flux)
    return flux_reddened


# --- 5.4b: Atmospheric extinction ---

SUPPORTED_OBSERVATORY_MODELS = [
    "kpno", "ctio", "apo", "lapalma", "mko", "mtham", "paranal",
]


def load_atmospheric_extinction(model="kpno"):
    """Load a tabulated atmospheric extinction curve for an observatory.

    Returns extinction in magnitudes per airmass as a function of wavelength.

    Parameters
    ----------
    model : str
        Observatory model name. Supported: kpno, ctio, apo, lapalma,
        mko, mtham, paranal.

    Returns
    -------
    wavelength : np.ndarray
        Wavelength in Angstroms.
    ext_per_airmass : np.ndarray
        Extinction in magnitudes per airmass.
    """
    from specreduce.calibration_data import AtmosphericExtinction

    if model not in SUPPORTED_OBSERVATORY_MODELS:
        raise ValueError(
            f"Unknown observatory model '{model}'. "
            f"Supported: {SUPPORTED_OBSERVATORY_MODELS}"
        )

    atm = AtmosphericExtinction(model=model)
    wavelength = atm.spectral_axis.to(u.AA).value
    ext_per_airmass = atm.flux.value

    return wavelength, ext_per_airmass


def correct_atmospheric_extinction(wavelength, flux, airmass, *,
                                   observatory="kpno"):
    """Correct observed flux for atmospheric extinction.

    Divides out the atmospheric dimming: flux_corrected = flux * 10^(0.4 * k * X)
    where k is the extinction in mag/airmass and X is the airmass.

    Parameters
    ----------
    wavelength : array-like
        Observed wavelengths in Angstrom.
    flux : array-like
        Observed flux.
    airmass : float
        Airmass of the observation.
    observatory : str
        Observatory extinction model name (default "kpno").

    Returns
    -------
    flux_corrected : np.ndarray
        Flux corrected for atmospheric extinction.
    """
    wavelength = np.asarray(wavelength, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64)

    if airmass <= 0:
        raise ValueError(f"Airmass must be positive, got {airmass}")

    ext_wave, ext_mag = load_atmospheric_extinction(observatory)

    # Interpolate the extinction curve onto the observed wavelength grid
    from scipy.interpolate import interp1d
    ext_interp = interp1d(
        ext_wave, ext_mag, kind="linear",
        bounds_error=False, fill_value="extrapolate",
    )
    k_lambda = ext_interp(wavelength)

    # Correct: observed = true * 10^(-0.4 * k * X)
    #   => true = observed * 10^(+0.4 * k * X)
    correction = 10.0 ** (0.4 * k_lambda * airmass)
    flux_corrected = flux * correction

    logger.info(
        "Corrected atmospheric extinction: airmass=%.3f, observatory=%s, "
        "mean correction factor=%.4f",
        airmass, observatory, np.mean(correction),
    )
    return flux_corrected


def correct_observed_spectrum(wavelength, flux, *, airmass=None,
                              observatory="kpno", ebv=0.0, rv=3.1,
                              extinction_law="odonnell94"):
    """Apply all extinction corrections to an observed spectrum.

    Order of corrections:
    1. Atmospheric extinction (if airmass provided)
    2. Interstellar extinction (if ebv > 0)

    Parameters
    ----------
    wavelength : array-like
        Wavelength in Angstrom.
    flux : array-like
        Observed flux (counts or counts/s).
    airmass : float, optional
        Airmass. If None, atmospheric correction is skipped.
    observatory : str
        Observatory for atmospheric extinction (default "kpno").
    ebv : float
        E(B-V) for interstellar extinction (default 0.0 = no correction).
    rv : float
        R_V for interstellar extinction (default 3.1).
    extinction_law : str
        Interstellar extinction law name.

    Returns
    -------
    flux_corrected : np.ndarray
        Corrected flux.
    corrections_applied : list of str
        Human-readable list of corrections that were applied.
    """
    wavelength = np.asarray(wavelength, dtype=np.float64)
    flux_out = np.asarray(flux, dtype=np.float64).copy()
    corrections = []

    # Step 1: Atmospheric extinction
    if airmass is not None and airmass > 0:
        flux_out = correct_atmospheric_extinction(
            wavelength, flux_out, airmass, observatory=observatory
        )
        corrections.append(
            f"Atmospheric extinction: airmass={airmass:.3f}, model={observatory}"
        )

    # Step 2: Interstellar extinction
    if ebv > 0:
        flux_out = apply_interstellar_extinction(
            wavelength, flux_out, ebv, rv=rv, law=extinction_law
        )
        corrections.append(
            f"Interstellar dereddening: E(B-V)={ebv:.4f}, R_V={rv:.2f}, "
            f"law={extinction_law}"
        )

    if not corrections:
        corrections.append("No extinction corrections applied")

    return flux_out, corrections


# ============================================================================
# 5.5 -- Reference spectrum preparation
# ============================================================================

def resample_to_observed(ref_wavelength, ref_flux, obs_wavelength,
                         *, ref_uncertainty=None):
    """Resample a reference spectrum onto the observed wavelength grid.

    Uses flux-conserving resampling (spectres) which properly handles
    the bin-integration needed for spectra of different resolutions.

    Parameters
    ----------
    ref_wavelength : array-like
        Reference wavelength grid (Angstrom).
    ref_flux : array-like
        Reference flux.
    obs_wavelength : array-like
        Target (observed) wavelength grid (Angstrom).
    ref_uncertainty : array-like, optional
        Uncertainties on the reference flux.

    Returns
    -------
    resampled_flux : np.ndarray
        Reference flux resampled onto obs_wavelength.
    resampled_unc : np.ndarray or None
        Resampled uncertainties (if ref_uncertainty was provided).
    """
    from spectres import spectres

    ref_wavelength = np.asarray(ref_wavelength, dtype=np.float64)
    ref_flux = np.asarray(ref_flux, dtype=np.float64)
    obs_wavelength = np.asarray(obs_wavelength, dtype=np.float64)

    # spectres requires monotonically increasing wavelengths
    if not np.all(np.diff(ref_wavelength) > 0):
        order = np.argsort(ref_wavelength)
        ref_wavelength = ref_wavelength[order]
        ref_flux = ref_flux[order]
        if ref_uncertainty is not None:
            ref_uncertainty = np.asarray(ref_uncertainty)[order]

    if ref_uncertainty is not None:
        ref_uncertainty = np.asarray(ref_uncertainty, dtype=np.float64)
        resampled_flux, resampled_unc = spectres(
            obs_wavelength, ref_wavelength, ref_flux,
            spec_errs=ref_uncertainty,
        )
        return resampled_flux, resampled_unc
    else:
        resampled_flux = spectres(obs_wavelength, ref_wavelength, ref_flux)
        return resampled_flux, None


def convolve_to_resolution(wavelength, flux, target_fwhm_aa):
    """Convolve a spectrum to match a lower spectral resolution.

    Applies a wavelength-dependent Gaussian kernel whose FWHM matches
    the target instrumental resolution.

    Parameters
    ----------
    wavelength : array-like
        Wavelength in Angstrom (must be uniformly spaced or nearly so).
    flux : array-like
        Flux values.
    target_fwhm_aa : float
        Target FWHM resolution in Angstrom.  The spectrum is convolved
        with a Gaussian of this width.

    Returns
    -------
    convolved_flux : np.ndarray
        Resolution-degraded flux.
    """
    wavelength = np.asarray(wavelength, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64)

    if target_fwhm_aa <= 0:
        raise ValueError(f"target_fwhm_aa must be positive, got {target_fwhm_aa}")

    # Estimate pixel scale (Angstrom/pixel) from median spacing
    dpix = np.median(np.diff(wavelength))
    if dpix <= 0:
        raise ValueError("Wavelength grid must be monotonically increasing")

    # Convert FWHM to sigma in pixels: FWHM = 2.355 * sigma
    sigma_pix = target_fwhm_aa / (2.3548200 * dpix)

    if sigma_pix < 0.5:
        logger.warning(
            "Target FWHM (%.2f A) is smaller than pixel scale (%.2f A); "
            "no convolution applied.",
            target_fwhm_aa, dpix,
        )
        return flux.copy()

    convolved = gaussian_filter1d(flux, sigma_pix)
    return convolved


# Feature masking regions for sensitivity function fitting
# These are regions where models are known to be unreliable or where
# strong features complicate the sensitivity derivation.

BALMER_MASK_REGIONS = [
    # (center_wavelength, half_width) in Angstrom
    # Broad wings mean we need wide masks for Balmer lines
    (6562.8, 20.0),   # H-alpha
    (4861.3, 15.0),   # H-beta
    (4340.5, 12.0),   # H-gamma
    (4101.7, 10.0),   # H-delta
    (3970.1, 8.0),    # H-epsilon
    (3889.1, 8.0),    # H-zeta
    (3835.4, 8.0),    # H-eta
]

TELLURIC_MASK_REGIONS = [
    # (start_wavelength, end_wavelength) in Angstrom
    (6270.0, 6290.0),   # O2 weak
    (6860.0, 6880.0),   # O2 B-band
    (7150.0, 7300.0),   # H2O band
    (7590.0, 7650.0),   # O2 A-band (deep)
]


def build_feature_mask(wavelength, *, mask_balmer=True, mask_telluric=True,
                       balmer_regions=None, telluric_regions=None,
                       extra_mask_regions=None):
    """Build a boolean mask that flags wavelengths to exclude.

    Masked regions include strong stellar absorption lines (Balmer series)
    and telluric absorption bands.  These regions should be excluded when
    fitting the sensitivity function.

    Parameters
    ----------
    wavelength : array-like
        Wavelength grid in Angstrom.
    mask_balmer : bool
        Mask Balmer line cores (default True).
    mask_telluric : bool
        Mask telluric absorption bands (default True).
    balmer_regions : list of (center, half_width), optional
        Override default Balmer mask regions.
    telluric_regions : list of (start, end), optional
        Override default telluric mask regions.
    extra_mask_regions : list of (start, end), optional
        Additional wavelength ranges to mask.

    Returns
    -------
    mask : np.ndarray of bool
        True = pixel should be EXCLUDED (masked).  False = good pixel.
    """
    wavelength = np.asarray(wavelength, dtype=np.float64)
    mask = np.zeros(len(wavelength), dtype=bool)

    if mask_balmer:
        regions = balmer_regions or BALMER_MASK_REGIONS
        for center, hw in regions:
            mask |= (wavelength >= center - hw) & (wavelength <= center + hw)

    if mask_telluric:
        regions = telluric_regions or TELLURIC_MASK_REGIONS
        for wmin, wmax in regions:
            mask |= (wavelength >= wmin) & (wavelength <= wmax)

    if extra_mask_regions:
        for wmin, wmax in extra_mask_regions:
            mask |= (wavelength >= wmin) & (wavelength <= wmax)

    n_masked = int(np.sum(mask))
    logger.info(
        "Feature mask: %d of %d pixels masked (%.1f%%)",
        n_masked, len(wavelength), 100.0 * n_masked / max(len(wavelength), 1),
    )
    return mask


def prepare_reference(ref_spec, obs_wavelength, *,
                      target_fwhm_aa=None, mask_balmer=True,
                      mask_telluric=True, extra_mask_regions=None):
    """Full preparation pipeline for a reference spectrum.

    Steps:
    1. Optionally convolve to match observed resolution.
    2. Resample onto the observed wavelength grid.
    3. Build feature mask for sensitivity function fitting.

    Parameters
    ----------
    ref_spec : Spectrum1D
        Reference spectrum (from load_calspec, load_phoenix_model, etc.).
    obs_wavelength : array-like
        Target wavelength grid (Angstrom).
    target_fwhm_aa : float, optional
        If set, convolve the reference to this spectral resolution (Angstrom
        FWHM) before resampling.  Important when the reference has much higher
        resolution than the data.
    mask_balmer : bool
        Build Balmer-line mask (default True).
    mask_telluric : bool
        Build telluric mask (default True).
    extra_mask_regions : list of (start, end), optional
        Additional mask regions.

    Returns
    -------
    ref_wavelength : np.ndarray
        The observed wavelength grid (same as obs_wavelength).
    ref_flux : np.ndarray
        Resampled (and optionally convolved) reference flux.
    feature_mask : np.ndarray of bool
        True = masked pixel (exclude from sensitivity fitting).
    """
    obs_wavelength = np.asarray(obs_wavelength, dtype=np.float64)

    ref_wave = ref_spec.spectral_axis.to(u.AA).value
    ref_flux = ref_spec.flux.value

    # Step 1: Optional resolution degradation
    if target_fwhm_aa is not None:
        ref_flux = convolve_to_resolution(ref_wave, ref_flux, target_fwhm_aa)

    # Step 2: Resample to observed grid
    ref_flux_resampled, _ = resample_to_observed(
        ref_wave, ref_flux, obs_wavelength
    )

    # Step 3: Feature mask
    feature_mask = build_feature_mask(
        obs_wavelength,
        mask_balmer=mask_balmer,
        mask_telluric=mask_telluric,
        extra_mask_regions=extra_mask_regions,
    )

    return obs_wavelength, ref_flux_resampled, feature_mask


# ============================================================================
# 5.6 -- Diagnostic comparison plot
# ============================================================================

def plot_reference_comparison(obs_wavelength, obs_flux, ref_wavelength,
                              ref_flux, *, feature_mask=None,
                              star_name=None, normalize=True,
                              save_path=None, figsize=(14, 8)):
    """Plot observed vs. reference spectrum for visual quality check.

    Overlays the two spectra (optionally normalized) and highlights masked
    regions where the agreement is expected to be poor (Balmer lines,
    telluric bands).

    Parameters
    ----------
    obs_wavelength : array-like
        Observed wavelength (Angstrom).
    obs_flux : array-like
        Observed flux.
    ref_wavelength : array-like
        Reference wavelength (Angstrom). If same as obs_wavelength, the
        reference was already resampled.
    ref_flux : array-like
        Reference flux.
    feature_mask : array-like of bool, optional
        True = masked pixel (highlighted on plot).
    star_name : str, optional
        Star name for the plot title.
    normalize : bool
        If True, normalize both spectra by their median (default True).
    save_path : str or Path, optional
        If given, save the figure to this path.
    figsize : tuple
        Figure size (default (14, 8)).

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    obs_wavelength = np.asarray(obs_wavelength)
    obs_flux = np.asarray(obs_flux)
    ref_wavelength = np.asarray(ref_wavelength)
    ref_flux = np.asarray(ref_flux)

    # Normalize by median of unmasked pixels if requested
    if normalize:
        if feature_mask is not None and len(feature_mask) == len(obs_flux):
            good = ~feature_mask
        else:
            good = np.isfinite(obs_flux)

        obs_median = np.median(obs_flux[good]) if np.any(good) else 1.0
        ref_good = np.isfinite(ref_flux)
        ref_median = np.median(ref_flux[ref_good]) if np.any(ref_good) else 1.0

        if obs_median > 0:
            obs_flux = obs_flux / obs_median
        if ref_median > 0:
            ref_flux = ref_flux / ref_median

    fig, axes = plt.subplots(2, 1, figsize=figsize, height_ratios=[3, 1],
                             sharex=True, gridspec_kw={"hspace": 0.05})

    # --- Top panel: spectra overlay ---
    ax1 = axes[0]
    ax1.plot(obs_wavelength, obs_flux, color="royalblue", alpha=0.8,
             linewidth=0.8, label="Observed")
    ax1.plot(ref_wavelength, ref_flux, color="crimson", alpha=0.8,
             linewidth=0.8, label="Reference")

    # Highlight masked regions
    if feature_mask is not None:
        _shade_masked_regions(ax1, obs_wavelength, feature_mask)

    ylabel = "Normalized flux" if normalize else "Flux"
    ax1.set_ylabel(ylabel)
    ax1.legend(loc="upper right")
    title = "Reference vs. Observed Spectrum"
    if star_name:
        title = f"{star_name}: {title}"
    ax1.set_title(title)

    # --- Bottom panel: residuals ---
    ax2 = axes[1]
    # Residual requires same grid; interpolate ref if needed
    if len(ref_wavelength) == len(obs_wavelength) and np.allclose(
        ref_wavelength, obs_wavelength, atol=0.1
    ):
        residual = obs_flux - ref_flux
    else:
        from scipy.interpolate import interp1d
        interp_ref = interp1d(
            ref_wavelength, ref_flux,
            bounds_error=False, fill_value=np.nan,
        )
        residual = obs_flux - interp_ref(obs_wavelength)

    ax2.plot(obs_wavelength, residual, color="gray", linewidth=0.6)
    ax2.axhline(0, color="black", linewidth=0.5, linestyle="--")

    if feature_mask is not None:
        _shade_masked_regions(ax2, obs_wavelength, feature_mask)

    ax2.set_ylabel("Residual (Obs - Ref)")
    ax2.set_xlabel("Wavelength (A)")

    fig.tight_layout(h_pad=1.0)

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved comparison plot to %s", save_path)

    return fig


def _shade_masked_regions(ax, wavelength, mask):
    """Shade contiguous masked regions on a matplotlib axis."""
    if mask is None or not np.any(mask):
        return

    # Find contiguous masked blocks
    mask = np.asarray(mask, dtype=bool)
    diff = np.diff(mask.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1

    # Handle edge cases where mask starts or ends at the array boundary
    if mask[0]:
        starts = np.concatenate([[0], starts])
    if mask[-1]:
        ends = np.concatenate([ends, [len(mask)]])

    ymin, ymax = ax.get_ylim()
    for s, e in zip(starts, ends):
        ax.axvspan(
            wavelength[s], wavelength[min(e, len(wavelength) - 1)],
            alpha=0.15, color="orange", zorder=0,
        )

    # Add a single legend entry
    if len(starts) > 0:
        ax.axvspan(0, 0, alpha=0.15, color="orange", label="Masked region")
