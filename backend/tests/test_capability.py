import pytest

from app.core.types import AgentRole, TaskCategory
from app.registry.capability import CapabilityRegistry

PLAN = (AgentRole.PLANNER, TaskCategory.REASONING)
EXTRACT = (AgentRole.WORKER, TaskCategory.EXTRACTION)


def record(registry, model_id, key, successes, failures):
    for _ in range(successes):
        registry.record(model_id, *key, success=True)
    for _ in range(failures):
        registry.record(model_id, *key, success=False)


def test_no_data_means_wide_uncertainty_and_insufficient_evidence():
    estimate = CapabilityRegistry().estimate("m", *PLAN)

    assert estimate.sample_size == 0
    assert estimate.mean == pytest.approx(0.5)
    assert estimate.lower_bound == pytest.approx(0.05, abs=1e-6)
    assert estimate.sufficient_evidence is False


def test_mean_is_the_posterior_mean():
    registry = CapabilityRegistry()
    record(registry, "m", PLAN, successes=8, failures=2)

    estimate = registry.estimate("m", *PLAN)

    assert estimate.mean == pytest.approx(9 / 12)  # (1+8) / (2+10)
    assert (estimate.successes, estimate.failures) == (8, 2)


def test_lower_bound_is_below_the_mean():
    registry = CapabilityRegistry()
    record(registry, "m", PLAN, successes=9, failures=1)

    estimate = registry.estimate("m", *PLAN)

    assert estimate.lower_bound < estimate.mean


def test_solid_evidence_beats_lucky_streak():
    registry = CapabilityRegistry()
    record(registry, "lucky", PLAN, successes=5, failures=0)
    record(registry, "solid", PLAN, successes=170, failures=30)

    lucky = registry.estimate("lucky", *PLAN)
    solid = registry.estimate("solid", *PLAN)

    assert lucky.mean > solid.mean
    assert solid.lower_bound > lucky.lower_bound


def test_more_evidence_at_same_rate_raises_the_lower_bound():
    registry = CapabilityRegistry()
    record(registry, "small_n", PLAN, successes=9, failures=1)
    record(registry, "large_n", PLAN, successes=90, failures=10)

    assert (
        registry.estimate("large_n", *PLAN).lower_bound
        > registry.estimate("small_n", *PLAN).lower_bound
    )


def test_sufficient_evidence_flips_at_min_samples():
    registry = CapabilityRegistry(min_samples=10)
    record(registry, "m", PLAN, successes=9, failures=0)
    assert registry.estimate("m", *PLAN).sufficient_evidence is False

    registry.record("m", *PLAN, success=True)
    assert registry.estimate("m", *PLAN).sufficient_evidence is True


def test_keys_are_independent():
    registry = CapabilityRegistry()
    record(registry, "a", PLAN, successes=10, failures=0)

    assert registry.estimate("a", *EXTRACT).sample_size == 0
    assert registry.estimate("b", *PLAN).sample_size == 0


def test_save_and_load_roundtrip(tmp_path):
    registry = CapabilityRegistry()
    record(registry, "m", PLAN, successes=7, failures=3)
    record(registry, "m", EXTRACT, successes=2, failures=0)
    path = tmp_path / "profiles" / "capabilities.json"

    registry.save(path)
    restored = CapabilityRegistry()
    restored.load(path)

    assert restored.estimate("m", *PLAN) == registry.estimate("m", *PLAN)
    assert restored.estimate("m", *EXTRACT) == registry.estimate("m", *EXTRACT)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"prior_alpha": 0},
        {"prior_beta": -1},
        {"lower_quantile": 0.6},
        {"lower_quantile": 0},
        {"min_samples": 0},
    ],
)
def test_invalid_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        CapabilityRegistry(**kwargs)