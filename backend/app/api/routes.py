"""HTTP endpoints."""

import json
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request

from app import __version__
from app.api.schemas import HealthOut, ModelOut, RunRequest, RunResponse
from app.api.serializers import model_out, run_response
from app.api.services import NoModelsError, Services
from app.workflow.budget import Budget
from app.workflow.spec import RunOptions

router = APIRouter()


def get_services(request: Request) -> Services:
    return request.app.state.services


ServicesDep = Annotated[Services, Depends(get_services)]


@router.get("/health", response_model=HealthOut)
def health(services: ServicesDep) -> HealthOut:
    return HealthOut(
        status="ok",
        version=__version__,
        configured_providers=services.settings.configured_providers,
    )


@router.get("/models", response_model=list[ModelOut])
def list_models(services: ServicesDep) -> list[ModelOut]:
    """Every model in the catalog, including disabled ones, so the UI can show why."""
    return [model_out(m) for m in services.catalog.all(enabled_only=False)]


@router.post("/run", response_model=RunResponse)
async def run_workflow(body: RunRequest, services: ServicesDep) -> RunResponse:
    try:
        workflow = services.workflow_for(body.mode, max_steps=body.max_steps)
    except NoModelsError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    options = RunOptions(
        category=body.category,
        risk=body.risk,
        contains_sensitive_data=body.contains_sensitive_data,
    )
    budget = Budget(max_cost_usd=body.max_cost_usd, max_calls=body.max_calls)
    result = await workflow.run(body.task, options, budget)
    return run_response(result, budget, body.mode)


@router.get("/benchmark")
def benchmark_results(services: ServicesDep) -> dict[str, Any]:
    """The latest saved benchmark run (python -m app.benchmark.cli)."""
    path = services.benchmark_path
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail="no benchmark results yet; run: python -m app.benchmark.cli",
        )
    return json.loads(path.read_text(encoding="utf-8"))
