import pytest
from fastapi.testclient import TestClient
from routing_helpers import ScriptedLLM, make_model

from app.api.factory import create_app
from app.api.services import Mode, Services
from app.core.config import Settings
from app.core.types import AgentRole, TaskCategory, Tier
from app.llm.backends import MockBackend
from app.llm.client import LLMClient
from app.llm.simulated import SimulatedLLM
from app.registry.capability import CapabilityRegistry
from app.registry.catalog import ModelCatalog
from app.registry.models import Provider
from app.verification.judge import estimated_judge_cost


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


def test_benchmark_is_404_until_a_run_exists(tmp_path):
    services = make_services()
    services.benchmark_path = tmp_path / "missing.json"
    response = TestClient(create_app(services)).get("/benchmark")

    assert response.status_code == 404
    assert "app.benchmark.cli" in response.json()["detail"]


def test_benchmark_returns_the_saved_results(tmp_path):
    path = tmp_path / "benchmark_results.json"
    path.write_text('{"chosen_low_floor": 0.6}', encoding="utf-8")
    services = make_services()
    services.benchmark_path = path

    response = TestClient(create_app(services)).get("/benchmark")

    assert response.status_code == 200
    assert response.json() == {"chosen_low_floor": 0.6}


def test_deployed_dashboard_origin_is_allowed_through_settings():
    services = make_services()
    services.settings = Settings(
        _env_file=None, cors_origins="https://demo.vercel.app/, https://x.dev"
    )
    client = TestClient(create_app(services))

    allowed = client.get("/health", headers={"Origin": "https://demo.vercel.app"})
    blocked = client.get("/health", headers={"Origin": "https://evil.example"})

    assert allowed.headers["access-control-allow-origin"] == "https://demo.vercel.app"
    assert "access-control-allow-origin" not in blocked.headers
    assert services.settings.cors_origin_list == ["https://demo.vercel.app", "https://x.dev"]


# ---- dedicated judge ----


def test_run_reports_the_dedicated_judge_and_never_routes_work_to_it(client):
    body = client.post(
        "/run", json={"task": "Review the clause.", "risk": "medium", "judge_model": "mock-mid"}
    ).json()

    assert body["judge_model"] == "mock-mid"
    models_used = {a["model_id"] for t in body["traces"] for a in t["attempts"]}
    assert "mock-mid" not in models_used
    judged = [c for t in body["traces"] for a in t["attempts"] for c in a["checks"]]
    assert any(c["name"] == "judge" for c in judged)


def test_run_without_a_judge_leaves_the_choice_automatic(client):
    body = client.post("/run", json={"task": "Summarise the contract risks."}).json()

    assert body["judge_model"] is None


def test_unknown_judge_is_a_clear_422(client):
    response = client.post("/run", json={"task": "Do it.", "judge_model": "no-such-model"})

    assert response.status_code == 422
    assert "no-such-model" in response.json()["detail"]


def test_judge_from_the_other_mode_is_rejected(client):
    response = client.post("/run", json={"task": "Do it.", "judge_model": "gpt-6-luna"})

    assert response.status_code == 422


def test_judge_model_setting_applies_to_runs():
    services = make_services()
    services.settings = Settings(_env_file=None, judge_model="mock-frontier")
    client = TestClient(create_app(services))

    body = client.post("/run", json={"task": "Summarise the contract risks."}).json()

    assert body["judge_model"] == "mock-frontier"


def test_judge_setting_naming_a_live_model_is_ignored_in_mock_mode():
    services = make_services()
    services.settings = Settings(_env_file=None, judge_model="groq-gpt-oss-120b")
    client = TestClient(create_app(services))

    body = client.post("/run", json={"task": "Summarise the contract risks."}).json()

    assert body["completed"] is True and body["judge_model"] is None


def test_policy_prices_verification_at_the_judge_cost():
    services = make_services()
    models = services.models_for(Mode.MOCK)
    judge = next(m for m in models if m.id == "mock-mid")

    policy = services.policy_for(models, judge)

    assert policy.verify_cost_usd == pytest.approx(estimated_judge_cost(judge))
    assert policy.verify_cost_usd > 0


# ---- live mode: keys, measured profiles, escalation explanations ----

PLAN = '{"steps": ["a", "b"]}'


def live_services(responder, *, keys=True) -> Services:
    models = [
        make_model("cheap", Tier.SMALL, 0.01, 0.01, provider=Provider.GEMINI),
        make_model("mid", Tier.MID, 2.0, 10.0, provider=Provider.GEMINI),
        make_model("grader", Tier.MID, 0.15, 0.6, provider=Provider.GROQ),
    ]
    settings = (
        Settings(_env_file=None, gemini_api_key="k", groq_api_key="k")
        if keys
        else Settings(_env_file=None)
    )
    return Services(
        catalog=ModelCatalog(models),
        registry=CapabilityRegistry(),
        settings=settings,
        mock_client=LLMClient(MockBackend()),
        live_client=LLMClient(MockBackend(responder=responder)),
    )


def grader_rejecting_cheap(model, messages):
    system, last = messages[0].content, messages[-1].content
    if "strict evaluator" in system:
        if "[cheap]" in last:
            return '{"pass": false, "reason": "omits the amounts"}'
        return '{"pass": true, "reason": "ok"}'
    if "planning agent" in system:
        return PLAN
    return f"[{model.id}] answer"


LIVE_BODY = {"task": "Summarise the supplier contract.", "mode": "live", "judge_model": "grader"}


def test_live_mode_without_provider_keys_is_a_clear_409():
    client = TestClient(create_app(live_services(grader_rejecting_cheap, keys=False)))

    response = client.post("/run", json=LIVE_BODY)

    assert response.status_code == 409
    detail = response.json()["detail"]
    assert "GEMINI_API_KEY" in detail and "GROQ_API_KEY" in detail


def test_live_run_explains_why_the_cheap_model_was_escalated():
    client = TestClient(create_app(live_services(grader_rejecting_cheap)))

    body = client.post("/run", json=LIVE_BODY).json()

    assert body["completed"] is True and body["judge_model"] == "grader"
    first = next(t for t in body["traces"] if t["agent"] == "worker")
    assert first["routing"]["mode"] == "cheap_first"
    step = first["escalation_steps"][0]
    assert (step["from_model"], step["to_model"]) == ("cheap", "mid")
    assert step["check"] == "judge" and step["judge_model"] == "grader"
    assert step["independent"] is True
    assert "omits the amounts" in step["reason"]
    assert step["wasted_cost_usd"] > 0
    judge_check = next(c for c in first["attempts"][0]["checks"] if c["name"] == "judge")
    assert judge_check["judge_model"] == "grader"


def test_a_provider_error_is_explained_as_the_reason_for_escalation():
    def cheap_is_down(model, messages):
        if model.id == "cheap" and "strict evaluator" not in messages[0].content:
            raise RuntimeError("503 overloaded")
        return grader_rejecting_cheap(model, messages)

    client = TestClient(create_app(live_services(cheap_is_down)))

    body = client.post("/run", json=LIVE_BODY).json()

    step = body["traces"][0]["escalation_steps"][0]
    assert step["check"] == "provider" and "503" in step["reason"]
    assert step["judge_model"] is None


def test_a_failed_format_check_is_explained_without_a_judge(client):
    services = make_services(SimulatedLLM({"mock-small": 1.0}))
    client = TestClient(create_app(services))

    body = client.post("/run", json={"task": "Summarise the contract risks."}).json()

    step = body["traces"][0]["escalation_steps"][0]
    assert (step["from_model"], step["to_model"]) == ("mock-small", "mock-mid")
    assert step["check"] != "judge" and step["reason"]


def test_a_run_with_no_escalation_has_no_steps(client):
    body = client.post("/run", json={"task": "Summarise the contract risks."}).json()

    assert all(t["escalation_steps"] == [] for t in body["traces"])


def test_live_runs_use_the_measured_profiles_not_the_simulated_ones():
    services = live_services(grader_rejecting_cheap)
    for _ in range(40):
        services.live_registry.record(
            "cheap", AgentRole.PLANNER, TaskCategory.REASONING, success=True
        )
        services.registry.record("mid", AgentRole.PLANNER, TaskCategory.REASONING, success=True)
    client = TestClient(create_app(services))

    body = client.post("/run", json=LIVE_BODY).json()

    candidates = {c["model_id"]: c for c in body["traces"][0]["routing"]["candidates"]}
    assert candidates["cheap"]["estimate"]["sample_size"] == 40
    assert candidates["mid"]["estimate"]["sample_size"] == 0
