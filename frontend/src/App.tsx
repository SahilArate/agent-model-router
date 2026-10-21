import { useEffect, useState } from "react";
import { api, ApiError } from "./api";
import AgentGraph from "./components/AgentGraph";
import BenchmarkView from "./components/BenchmarkView";
import RunForm from "./components/RunForm";
import Summary from "./components/Summary";
import TraceDetail from "./components/TraceDetail";
import type { Health, ModelInfo, RunRequest, RunResponse } from "./types";

type Tab = "run" | "benchmark";

export default function App() {
  const [tab, setTab] = useState<Tab>("run");
  const [health, setHealth] = useState<Health | null>(null);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [run, setRun] = useState<RunResponse | null>(null);
  const [selected, setSelected] = useState(0);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Free hosting puts the API to sleep when idle, so keep asking until it answers.
  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout>;
    const check = () => {
      api
        .health()
        .then((result) => !cancelled && setHealth(result))
        .catch(() => {
          if (cancelled) return;
          setHealth(null);
          timer = setTimeout(check, 4000);
        });
    };
    check();
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, []);

  useEffect(() => {
    if (health) api.models().then(setModels).catch(() => setModels([]));
  }, [health]);

  const execute = async (request: RunRequest) => {
    setBusy(true);
    setError(null);
    try {
      const result = await api.run(request);
      setRun(result);
      setSelected(0);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Something went wrong.");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="app">
      <header>
        <div>
          <h1>Agent Model Router</h1>
          <p className="muted">Every agent call goes to the cheapest model that is measured to be good enough.</p>
        </div>
        <div className="header-right">
          <nav role="tablist">
            <button role="tab" aria-selected={tab === "run"} onClick={() => setTab("run")}>Live run</button>
            <button role="tab" aria-selected={tab === "benchmark"} onClick={() => setTab("benchmark")}>Benchmark</button>
          </nav>
          <span className={`status ${health ? "ok" : "bad"}`}>
            {health ? `API v${health.version}` : "Connecting to API…"}
          </span>
        </div>
      </header>

      {tab === "benchmark" ? (
        <BenchmarkView />
      ) : (
        <main className="layout">
          <RunForm busy={busy} models={models} onRun={execute} />
          <div className="stack">
            {!health && (
              <div className="panel muted">
                The API is waking up. Free hosting sleeps when idle, so the first request can take
                up to a minute.
              </div>
            )}
            {error && <div className="panel error-text">{error}</div>}
            {!run && !error && (
              <div className="panel muted">
                Run a workflow to see each agent, the model it was given, and why.
              </div>
            )}
            {run && (
              <>
                <Summary run={run} />
                <div className="panel">
                  <h2>Agents</h2>
                  <p className="muted">Select an agent to see how its model was chosen.</p>
                  <AgentGraph traces={run.traces} selected={selected} onSelect={setSelected} />
                </div>
                {run.traces[selected] && <TraceDetail trace={run.traces[selected]} />}
                {run.final_answer && (
                  <div className="panel">
                    <h2>Final answer</h2>
                    <pre className="output">{run.final_answer}</pre>
                  </div>
                )}
                {run.abstained_at && (
                  <div className="panel error-text">
                    Stopped at {run.abstained_at}: the system abstained rather than guess.
                  </div>
                )}
              </>
            )}
          </div>
        </main>
      )}
    </div>
  );
}
