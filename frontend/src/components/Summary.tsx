import { pct, seconds, usd } from "../format";
import type { RunResponse } from "../types";

export default function Summary({ run }: { run: RunResponse }) {
  const escalations = run.traces.reduce((total, t) => total + t.escalations, 0);
  const tiles = [
    { label: "Outcome", value: run.completed ? "Completed" : "Abstained", tone: run.completed ? "ok" : "bad" },
    { label: "Total cost", value: usd(run.cost_usd) },
    { label: "Budget used", value: pct(run.budget.spent_usd / run.budget.max_cost_usd) },
    { label: "Model calls", value: `${run.budget.calls} / ${run.budget.max_calls}` },
    { label: "Escalations", value: String(escalations) },
    { label: "Judge", value: run.judge_model ?? "automatic" },
    { label: "Latency", value: seconds(run.latency_s) },
  ];
  return (
    <section className="tiles" aria-label="Run summary">
      {tiles.map((tile) => (
        <div className="tile" key={tile.label}>
          <span className="tile-label">{tile.label}</span>
          <span className={`tile-value ${tile.tone ?? ""}`}>{tile.value}</span>
        </div>
      ))}
    </section>
  );
}
