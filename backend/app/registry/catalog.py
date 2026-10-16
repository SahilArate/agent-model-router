"""The set of candidate models, loaded from a JSON file."""

import json
from collections.abc import Iterable
from pathlib import Path

from app.core.types import Tier
from app.registry.models import ModelProfile

DEFAULT_CATALOG_PATH = Path(__file__).with_name("models.json")


class ModelCatalog:
    """Immutable lookup of model profiles by id or tier."""

    def __init__(self, models: Iterable[ModelProfile]) -> None:
        by_id: dict[str, ModelProfile] = {}
        for model in models:
            if model.id in by_id:
                raise ValueError(f"duplicate model id: {model.id}")
            by_id[model.id] = model
        self._models = by_id

    @classmethod
    def from_file(cls, path: Path = DEFAULT_CATALOG_PATH) -> "ModelCatalog":
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls(ModelProfile.model_validate(item) for item in raw["models"])

    def __len__(self) -> int:
        return len(self._models)

    def get(self, model_id: str) -> ModelProfile:
        try:
            return self._models[model_id]
        except KeyError:
            raise KeyError(f"unknown model id: {model_id}") from None

    def all(self, *, enabled_only: bool = True) -> list[ModelProfile]:
        return [m for m in self._models.values() if m.enabled or not enabled_only]

    def by_tier(self, tier: Tier, *, enabled_only: bool = True) -> list[ModelProfile]:
        return [m for m in self.all(enabled_only=enabled_only) if m.tier is tier]