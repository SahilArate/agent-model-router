"""Runs checks cheapest-first and stops at the first one that does not pass."""

from collections.abc import Sequence

from app.verification.types import Check, CheckStatus, VerificationContext, VerificationReport


class VerificationPipeline:
    """Order checks from cheapest to most expensive: a failed free check saves the judge call."""

    def __init__(self, checks: Sequence[Check]) -> None:
        if not checks:
            raise ValueError("a verification pipeline needs at least one check")
        self._checks = tuple(checks)

    async def verify(self, ctx: VerificationContext) -> VerificationReport:
        results = []
        for check in self._checks:
            result = await check.run(ctx)
            results.append(result)
            if result.status is not CheckStatus.PASS:
                break
        return VerificationReport(results=tuple(results))