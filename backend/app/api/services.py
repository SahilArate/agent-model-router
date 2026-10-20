"""Everything the API needs to build and run a workflow, assembled in one place."""

from dataclasses import dataclass, field, replace
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
from app.verification.judge import estimated_judge_cost
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


class InvalidJudgeError(Exception):
    """The requested or configured judge model cannot be used."""


@dataclass(frozen=True)
class RunPlan:
    workflow: MultiAgentWorkflow
    judge_model_id: str | None  # None means the judge is picked automatically for each call


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

    def judge_for(self, mode: Mode, requested: str | None = None) -> ModelProfile | None:
        """The dedicated judge for a run, or None to let the verifier pick one per call.

        A judge named in the request must be usable. The JUDGE_MODEL setting is applied when it
        belongs to this mode's models, and ignored in the other mode so one setting can serve
        both the simulated and the live demo.
        """
        pool = self.models_for(mode)
        wanted = requested or self.settings.judge_model
        if not wanted:
            return None

        match = next((m for m in pool if m.id == wanted), None)
        if match is not None:
            return match

        if requested is None:
            if wanted not in {m.id for m in self.catalog.all(enabled_only=False)}:
                raise InvalidJudgeError(f"JUDGE_MODEL '{wanted}' is not in the model catalog")
            if mode is Mode.MOCK:
                return None
        available = ", ".join(m.id for m in pool) or "none"
        raise InvalidJudgeError(
            f"judge '{wanted}' is not an enabled {mode.value} model (available: {available})"
        )

    def policy_for(self, models: list[ModelProfile], judge: ModelProfile | None) -> RoutingPolicy:
        """Routing policy whose verification cost is what one judge call really costs."""
        priced = judge or min(models, key=lambda m: estimated_judge_cost(m))
        return replace(self.policy, verify_cost_usd=estimated_judge_cost(priced))

    def plan_run(self, mode: Mode, *, max_steps: int, judge_model: str | None = None) -> RunPlan:
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
        judge = self.judge_for(mode, judge_model)

        # A dedicated judge never writes answers, so it can never grade its own work.
        workers = [m for m in models if judge is None or m.id != judge.id]
        if not workers:
            raise NoModelsError(
                f"no model is left for routing after reserving '{judge.id}' as the judge"
                if judge
                else "no models available"
            )

        client = self.mock_client if mode is Mode.MOCK else self.live_client
        router = ModelRouter(workers, self.registry, self.policy_for(models, judge))
        verifiers = VerifierFactory(
            client=client,
            judge_candidates=tuple(models),
            judge_model_id=judge.id if judge else None,
        )
        workflow = MultiAgentWorkflow(
            RoutedExecutor(router, client), verifiers, max_steps=max_steps
        )
        return RunPlan(workflow=workflow, judge_model_id=judge.id if judge else None)


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
