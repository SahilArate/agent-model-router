"""Break-even maths for cheap-first cascades.

A cascade tries a cheap model, verifies, and escalates on failure. It only saves money when
the escalation probability is low enough, otherwise going straight to the stronger model is
cheaper. This module answers that question with numbers rather than intuition.
"""


def cascade_expected_cost(
    cheap_cost: float, strong_cost: float, verify_cost: float, escalation_prob: float
) -> float:
    """E[cost] = c_cheap + c_verify + P(escalate) * (c_strong + c_verify)."""
    if not 0.0 <= escalation_prob <= 1.0:
        raise ValueError("escalation_prob must be between 0 and 1")
    return cheap_cost + verify_cost + escalation_prob * (strong_cost + verify_cost)


def direct_expected_cost(strong_cost: float, verify_cost: float) -> float:
    """Going straight to the stronger model, still verified once."""
    return strong_cost + verify_cost


def cascade_beats_direct(
    cheap_cost: float, strong_cost: float, verify_cost: float, escalation_prob: float
) -> bool:
    """True when trying the cheap model first has a lower expected cost than skipping it."""
    cascade = cascade_expected_cost(cheap_cost, strong_cost, verify_cost, escalation_prob)
    return cascade < direct_expected_cost(strong_cost, verify_cost)