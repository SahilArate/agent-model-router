import pytest
from pydantic import ValidationError

from app.core.types import Tier
from app.registry.catalog import ModelCatalog
from app.registry.models import ModelProfile, Provider


def make_model(**overrides) -> ModelProfile:
    fields = {
        "id": "test-model",
        "provider": Provider.MOCK,
        "litellm_model": "mock/test-model",
        "tier": Tier.SMALL,
        "input_price_per_mtok": 1.0,
        "output_price_per_mtok": 4.0,
        "context_window": 8000,
    }
    return ModelProfile(**{**fields, **overrides})


def test_cost_without_cache():
    model = make_model()

    assert model.cost(input_tokens=1_000_000, output_tokens=500_000) == pytest.approx(3.0)


def test_cost_uses_cached_price_for_cached_tokens():
    model = make_model(cached_input_price_per_mtok=0.1)

    cost = model.cost(input_tokens=1_000_000, output_tokens=0, cached_tokens=500_000)

    assert cost == pytest.approx(0.5 * 1.0 + 0.5 * 0.1)


def test_cached_tokens_billed_at_input_price_when_no_cache_price():
    model = make_model()

    cost = model.cost(input_tokens=1_000_000, output_tokens=0, cached_tokens=400_000)

    assert cost == pytest.approx(1.0)


def test_cost_rejects_invalid_token_counts():
    model = make_model()

    with pytest.raises(ValueError):
        model.cost(input_tokens=100, output_tokens=0, cached_tokens=200)
    with pytest.raises(ValueError):
        model.cost(input_tokens=-1, output_tokens=0)


def test_cached_price_above_input_price_is_rejected():
    with pytest.raises(ValidationError):
        make_model(cached_input_price_per_mtok=2.0)


def test_negative_price_is_rejected():
    with pytest.raises(ValidationError):
        make_model(input_price_per_mtok=-1.0)


def test_duplicate_ids_are_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        ModelCatalog([make_model(), make_model()])


def test_unknown_model_raises_helpful_error():
    catalog = ModelCatalog([make_model()])

    with pytest.raises(KeyError, match="unknown model id"):
        catalog.get("nope")


def test_disabled_models_are_hidden_by_default():
    catalog = ModelCatalog([make_model(), make_model(id="off", enabled=False)])

    assert [m.id for m in catalog.all()] == ["test-model"]
    assert len(catalog.all(enabled_only=False)) == 2


def test_default_catalog_loads_and_has_every_tier_enabled():
    catalog = ModelCatalog.from_file()

    assert len(catalog) >= 3
    for tier in Tier:
        assert catalog.by_tier(tier), f"no enabled model in tier {tier}"


def test_default_catalog_enabled_models_are_mocks_until_pricing_is_verified():
    catalog = ModelCatalog.from_file()

    assert all(m.provider is Provider.MOCK for m in catalog.all())