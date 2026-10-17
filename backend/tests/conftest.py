"""Shared fixtures."""

import pytest
from routing_helpers import ScriptedLLM, make_model

from app.core.types import Tier


@pytest.fixture
def tier_models():
    return [
        make_model("small", Tier.SMALL, 0.1, 0.5),
        make_model("mid", Tier.MID, 2.0, 10.0),
        make_model("frontier", Tier.FRONTIER, 10.0, 50.0),
    ]


@pytest.fixture
def script():
    return ScriptedLLM()