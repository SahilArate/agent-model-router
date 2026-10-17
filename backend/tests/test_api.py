import pytest
from fastapi.testclient import TestClient
from routing_helpers import ScriptedLLM

from app.api.factory import create_app
from app.api.services import Services
from app.core.config import Settings
from app.llm.backends import MockBackend
from app.llm.client import LLMClient
from app.llm.simulated import SimulatedLLM
from app.registry.capability import CapabilityRegistry
from app.registry.catalog import ModelCatalog


def make_services(responder=None) -> Services:
    mock_client = LLMClient(MockBackend(responder=responder or SimulatedLLM({})))
    return Services(
        catalog=ModelCatalog.from_file(),
        registry=CapabilityRegistry(),
        settings=Settings(_env_file=None),
        mock_client=mock_client,
        live_client=LLMClient(MockBackend()),
    )


@pytest.fixture
def client():
    return TestClient(create_app(make_services()))


def test_health(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert "configured_providers" in response.json()


def test_models_lists_enabled_and_disabled_models(client):
    models = {m["id"]: m for m in client.get("/models").json()}

    assert models["mock-small"]["enabled"] is True
    assert models["claude-sonnet-5-5"]["enabled"] is False
    assert models["mock-small"]["tier"] == "small"


def test_run_returns_a_full_trace(client):
    response = client.post("/run", json={"task": "Summarise the contract risks."})

    body = response.json()
    assert response.status_code == 200
    assert body["completed"] is True and body["final_answer"]
    assert [t["agent"] for t in body["traces"]] == [
        "planner",
        "worker",
        "worker",
        "worker",
        "synthesizer",
    ]
    first = body["traces"][0]
    assert first["routing"]["model_id"] == "mock-small"
    assert first["routing"]["mode"] == "cheap_first"
    assert first["routing"]["candidates"] and first["routing"]["fallbacks"]
    assert body["cost_usd"] > 0
    assert body["budget"]["spent_usd"] == pytest.approx(body["cost_usd"])


def test_run_shows_escalation_when_the_cheap_model_is_flawed():
    services = make_services(SimulatedLLM({"mock-small": 1.0}))
    client = TestClient(create_app(services))

    body = client.post("/run", json={"task": "Summarise the contract risks."}).json()

    assert body["completed"] is True
    plan = body["traces"][0]
    assert [a["model_id"] for a in plan["attempts"]] == ["mock-small", "mock-mid"]
    assert plan["escalations"] == 1
    assert plan["attempts"][0]["action"] == "escalate"
    assert plan["attempts"][1]["action"] == "accept"


def test_medium_risk_runs_the_judge_and_uses_the_strongest_model(client):
    body = client.post("/run", json={"task": "Review the clause.", "risk": "medium"}).json()

    first = body["traces"][0]
    assert first["routing"]["mode"] == "strongest"
    assert first["routing"]["requires_independent_verifier"] is True
    check_names = [c["name"] for c in first["attempts"][0]["checks"]]
    assert "judge" in check_names


def test_budget_limit_makes_the_workflow_abstain(client):
    body = client.post("/run", json={"task": "Do the thing.", "max_calls": 2}).json()

    assert body["completed"] is False
    assert body["abstained_at"] == "worker"
    assert body["final_answer"] is None


def test_live_mode_without_enabled_models_is_a_clear_409(client):
    response = client.post("/run", json={"task": "Do the thing.", "mode": "live"})

    assert response.status_code == 409
    assert "no enabled live models" in response.json()["detail"]


@pytest.mark.parametrize(
    "payload",
    [
        {"task": ""},
        {"task": "   "},
        {"task": "ok", "max_cost_usd": 0},
        {"task": "ok", "max_calls": 0},
        {"task": "ok", "risk": "extreme"},
    ],
)
def test_invalid_requests_are_rejected(client, payload):
    assert client.post("/run", json=payload).status_code == 422


def test_cors_allows_the_local_dashboard(client):
    response = client.options(
        "/run",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
        },
    )

    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_scripted_responses_flow_through_the_api():
    services = make_services(ScriptedLLM())
    client = TestClient(create_app(services))

    body = client.post("/run", json={"task": "Anything."}).json()

    assert body["completed"] is True
    assert body["final_answer"] == "final answer"