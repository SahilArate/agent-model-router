from app.core.types import RiskLevel, Tier


def test_tiers_are_ordered_cheapest_first():
    assert [t.rank for t in Tier] == [0, 1, 2]
    assert Tier.SMALL.rank < Tier.MID.rank < Tier.FRONTIER.rank


def test_escalation_walks_up_and_stops_at_top():
    assert Tier.SMALL.escalate() is Tier.MID
    assert Tier.MID.escalate() is Tier.FRONTIER
    assert Tier.FRONTIER.escalate() is None


def test_only_low_risk_allows_cheap_first():
    assert RiskLevel.LOW.cheap_first_allowed is True
    assert RiskLevel.MEDIUM.cheap_first_allowed is False
    assert RiskLevel.HIGH.cheap_first_allowed is False