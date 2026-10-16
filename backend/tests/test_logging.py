import logging

import pytest

from app.core.logging import configure_logging, get_logger


def _our_handlers() -> list[logging.Handler]:
    return [h for h in logging.getLogger("app").handlers if h.get_name() == "app-stdout"]


@pytest.fixture(autouse=True)
def clean_app_logger():
    def _remove_ours() -> None:
        logger = logging.getLogger("app")
        for handler in _our_handlers():
            logger.removeHandler(handler)

    _remove_ours()
    yield
    _remove_ours()


def test_configure_is_idempotent():
    configure_logging("INFO")
    configure_logging("INFO")

    assert len(_our_handlers()) == 1


def test_level_is_applied():
    configure_logging("DEBUG")

    assert logging.getLogger("app").level == logging.DEBUG


def test_messages_are_written_to_stdout(capsys):
    configure_logging("INFO")

    get_logger("app.router").info("routing decision made")

    assert "routing decision made" in capsys.readouterr().out