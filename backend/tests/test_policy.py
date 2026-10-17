import pytest

from app.core.types import RiskLevel
from app.router.policy import RoutingPolicy


def test_defaults_get_stricter_with_risk():
    policy = RoutingPolicy()

    assert (
        policy.floor_for(RiskLevel.LOW)
        < policy.floor_for(RiskLevel.MEDIUM)
        < policy.floor_for(RiskLevel.HIGH)
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"quality_floors": {RiskLevel.LOW: 0.8}},
        {"quality_floors": {RiskLevel.LOW: 0.8, RiskLevel.MEDIUM: 0.9, RiskLevel.HIGH: 1.5}},
        {"expected_output_fraction": 0},
        {"verify_cost_usd": -1},
        {"max_escalations": -1},
    ],
)
def test_invalid_policy_is_rejected(kwargs):
    with pytest.raises(ValueError):
        RoutingPolicy(**kwargs)