"""Everything the API needs to build and run a workflow, assembled in one place."""

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from app.agents.common import VerifierFactory
from app.core.config import Settings, get_settings
from app.llm.backends import MockBackend
from app.llm.client import LLMClient, build_default_client
from app.llm.simulated import SimulatedLLM
from app.registry.capability import CapabilityRegistry
from app.registry.catalog import ModelCatalog
from app.registry.models import ModelProfile, Provider
from app.router.engine import ModelRouter
from app.router.policy import RoutingPolicy
from app.workflow.executor import RoutedExecutor
from app.workflow.runner import MultiAgentWorkflow

# backend/app/api/services.py -> repository root is three levels above "app"
_DATA_DIR = Path(__file__).resolve().parents[3] / "data"
DEFAULT_PROFILES_PATH = _DATA_DIR / "capability_profiles.json"
DEFAULT_BENCHMARK_PATH = _DATA_DIR / "benchmark_results.json"


class Mode(StrEnum):
    MOCK = "mock"  # simulated models, free and offline
    LIVE = "live"  # real providers through LiteLLM


class NoModelsError(Exception):
    """No enabled models are available for the requested mode."""


@dataclass
class Services:
    catalog: ModelCatalog
    registry: CapabilityRegistry
    settings: Settings
    mock_client: LLMClient
    live_client: LLMClient
    policy: RoutingPolicy = field(default_factory=RoutingPolicy)
    benchmark_path: Path = DEFAULT_BENCHMARK_PATH

    def models_for(self, mode: Mode) -> list[ModelProfile]:
        is_mock = mode is Mode.MOCK
        return [m for m in self.catalog.all() if (m.provider is Provider.MOCK) == is_mock]

    def workflow_for(self, mode: Mode, *, max_steps: int) -> MultiAgentWorkflow:
        models = self.models_for(mode)
        if not models:
            raise NoModelsError(
                f"no enabled {mode.value} models in the catalog"
                + (
                    " (verify pricing in models.json, set enabled to true and add API keys)"
                    if mode is Mode.LIVE
                    else ""
                )
            )
        client = self.mock_client if mode is Mode.MOCK else self.live_client
        router = ModelRouter(models, self.registry, self.policy)
        verifiers = VerifierFactory(client=client, judge_candidates=tuple(models))
        return MultiAgentWorkflow(RoutedExecutor(router, client), verifiers, max_steps=max_steps)


def build_services(settings: Settings | None = None) -> Services:
    settings = settings or get_settings()

    registry = CapabilityRegistry()
    if DEFAULT_PROFILES_PATH.exists():
        registry.load(DEFAULT_PROFILES_PATH)

    return Services(
        catalog=ModelCatalog.from_file(),
        registry=registry,
        settings=settings,
        mock_client=LLMClient(MockBackend(responder=SimulatedLLM())),
        live_client=build_default_client(settings),
    )
