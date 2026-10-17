import pytest

from app.router.cascade import cascade_beats_direct, cascade_expected_cost, direct_expected_cost


def test_expected_cost_formula():
    # 1 + 0.5 + 0.2 * (10 + 0.5) = 3.6
    assert cascade_expected_cost(1.0, 10.0, 0.5, 0.2) == pytest.approx(3.6)


def test_direct_cost_is_strong_plus_one_verification():
    assert direct_expected_cost(10.0, 0.5) == pytest.approx(10.5)


def test_cascade_wins_when_escalation_is_rare():
    assert cascade_beats_direct(1.0, 10.0, 0.0, escalation_prob=0.2)


def test_cascade_loses_when_escalation_is_common():
    assert not cascade_beats_direct(1.0, 10.0, 0.0, escalation_prob=0.95)


def test_break_even_point_without_verification_cost():
    # cascade < direct  <=>  p < 1 - cheap/strong = 0.9
    assert cascade_beats_direct(1.0, 10.0, 0.0, escalation_prob=0.89)
    assert not cascade_beats_direct(1.0, 10.0, 0.0, escalation_prob=0.91)


def test_invalid_probability_is_rejected():
    with pytest.raises(ValueError):
        cascade_expected_cost(1.0, 10.0, 0.0, 1.5)