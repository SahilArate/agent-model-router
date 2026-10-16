"""Logging setup. Call configure_logging() once at application start."""

import logging
import sys

from app.core.config import get_settings

_ROOT_LOGGER_NAME = "app"
_HANDLER_NAME = "app-stdout"
_LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"


def configure_logging(level: str | None = None) -> None:
    """Configure the 'app' logger. Safe to call more than once."""
    resolved_level = level or get_settings().log_level

    logger = logging.getLogger(_ROOT_LOGGER_NAME)
    logger.setLevel(resolved_level)
    logger.propagate = False

    # Look for our own handler by name, so handlers added by other tools
    # (pytest, uvicorn) are never mistaken for ours.
    if not any(h.get_name() == _HANDLER_NAME for h in logger.handlers):
        handler = logging.StreamHandler(sys.stdout)
        handler.set_name(_HANDLER_NAME)
        handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        logger.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Return a logger. Pass __name__ so it nests under the 'app' logger."""
    return logging.getLogger(name)