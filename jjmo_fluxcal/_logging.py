"""
_logging.py — Unified Logging Configuration
============================================

Provides a single ``setup_logging`` function that configures the Python
logging hierarchy for the ``jjmo_fluxcal`` package.  Called automatically
by the CLI and by ``fluxcal()`` if logging has not been configured.
"""

import logging
import sys


_CONFIGURED = False


def setup_logging(level: str = "INFO", stream=None) -> None:
    """Configure logging for the jjmo_fluxcal package.

    Parameters
    ----------
    level : str
        One of 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'.
    stream : file-like, optional
        Where to send log output.  Defaults to ``sys.stderr``.
    """
    global _CONFIGURED

    numeric = getattr(logging, level.upper(), logging.INFO)
    pkg_logger = logging.getLogger("jjmo_fluxcal")
    pkg_logger.setLevel(numeric)

    # Avoid adding duplicate handlers on repeated calls
    if not pkg_logger.handlers:
        handler = logging.StreamHandler(stream or sys.stderr)
        handler.setLevel(numeric)
        fmt = logging.Formatter(
            "[%(levelname)s] %(name)s: %(message)s"
        )
        handler.setFormatter(fmt)
        pkg_logger.addHandler(handler)
    else:
        for h in pkg_logger.handlers:
            h.setLevel(numeric)

    _CONFIGURED = True


def is_configured() -> bool:
    """Return True if setup_logging has been called."""
    return _CONFIGURED
