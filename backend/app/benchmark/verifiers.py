"""Verification checks used by the benchmark's router strategy."""

from app.benchmark.grading import is_correct, is_well_formed
from app.benchmark.tasks import Task
from app.benchmark.world import SimulatedWorld, uniform
from app.llm.backends import estimate_tokens
from app.registry.models import ModelProfile
from app.verification.pipeline import VerificationPipeline
from app.verification.types import CheckResult, CheckStatus, VerificationContext

_JUDGE_OUTPUT_TOKENS = 20


class FormatCheck:
    """Free and deterministic. Catches malformed answers; cannot see a plausible wrong one."""

    name = "format"

    def __init__(self, task: Task) -> None:
        self._kind = task.kind

    async def run(self, ctx: VerificationContext) -> CheckResult:
        if is_well_formed(self._kind, ctx.output):
            return CheckResult(self.name, CheckStatus.PASS, "well formed")
        return CheckResult(self.name, CheckStatus.FAIL, "answer is not in the requested format")


class NoisyJudge:
    """A simulated LLM judge that is deliberately imperfect.

    It catches a wrong answer with probability `detect_rate` and wrongly rejects a correct
    answer with probability `false_reject_rate`. Both are assumptions stated in the report,
    and the CLI can sweep them. It costs one call of the judge model's price.
    """

    name = "judge"

    def __init__(
        self,
        world: SimulatedWorld,
        judge_model: ModelProfile,
        *,
        detect_rate: float = 0.85,
        false_reject_rate: float = 0.03,
    ) -> None:
        self._world = world
        self._judge_model = judge_model
        self._detect_rate = detect_rate
        self._false_reject_rate = false_reject_rate

    async def run(self, ctx: VerificationContext) -> CheckResult:
        task = self._world.task_for(ctx.task_prompt)
        cost = self._judge_model.cost(
            estimate_tokens(ctx.task_prompt) + estimate_tokens(ctx.output), _JUDGE_OUTPUT_TOKENS
        )
        roll = uniform(self._world.seed, "judge", task.id, ctx.output, ctx.worker_model.id)
        if is_correct(task, ctx.output):
            rejected = roll < self._false_reject_rate
        else:
            rejected = roll < self._detect_rate
        if rejected:
            return CheckResult(self.name, CheckStatus.FAIL, "judge rejected the answer", cost)
        return CheckResult(self.name, CheckStatus.PASS, "judge accepted the answer", cost)


def build_pipeline(
    task: Task,
    world: SimulatedWorld,
    judge_model: ModelProfile,
    *,
    detect_rate: float = 0.85,
    false_reject_rate: float = 0.03,
) -> VerificationPipeline:
    """Free format check first, so the paid judge only sees well-formed answers."""
    return VerificationPipeline(
        [
            FormatCheck(task),
            NoisyJudge(
                world,
                judge_model,
                detect_rate=detect_rate,
                false_reject_rate=false_reject_rate,
            ),
        ]
    )
