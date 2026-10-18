import { useEffect, useState } from "react";
import { api, ApiError } from "../api";
import { pct, seconds, signedPct, usd } from "../format";
import type { BenchmarkResults, Scorecard } from "../types";

const ORDER = [
  "fixed_cheap",
  "complexity_only",
  "fixed_strongest",
  "router_cold_start",
  "router_learned",
];
const MODEL_COLORS: Record<string, string> = {
  "mock-small": "var(--c1)",
  "mock-mid": "var(--c2)",
  "mock-frontier": "var(--c3)",
};

// Three strategies land almost on top of each other, so their labels are offset by hand.
const LABEL_OFFSETS: Record<string, { dx: number; dy: number; anchor: "start" | "middle" | "end" }> = {
  fixed_strongest: { dx: 0, dy: -14, anchor: "middle" },
  router_cold_start: { dx: 12, dy: 22, anchor: "start" },
  router_learned: { dx: -12, dy: 22, anchor: "end" },
};
const DEFAULT_OFFSET = { dx: 0, dy: -12, anchor: "middle" as const };

function CostQualityChart({ cards }: { cards: Scorecard[] }) {
  const width = 520;
  const height = 280;
  const pad = { left: 56, right: 90, top: 16, bottom: 44 };
  const costs = cards.map((c) => c.cost_per_success_usd ?? 0).filter((v) => v > 0);
  const lo = Math.log10(Math.min(...costs)) - 0.15;
  const hi = Math.log10(Math.max(...costs)) + 0.15;
  const rates = cards.map((c) => c.success_rate);
  const yMin = Math.floor((Math.min(...rates) - 0.05) * 20) / 20;
  const x = (cost: number) =>
    pad.left + ((Math.log10(cost) - lo) / (hi - lo)) * (width - pad.left - pad.right);
  const y = (rate: number) =>
    pad.top + (1 - (rate - yMin) / (1 - yMin)) * (height - pad.top - pad.bottom);

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="chart" role="img"
      aria-label="Success rate against cost per successful task">
      {[yMin, (yMin + 1) / 2, 1].map((tick) => (
        <g key={tick}>
          <line x1={pad.left} x2={width - pad.right} y1={y(tick)} y2={y(tick)} className="grid" />
          <text x={pad.left - 8} y={y(tick) + 4} textAnchor="end" className="axis">
            {pct(tick, 0)}
          </text>
        </g>
      ))}
      <text x={(width + pad.left) / 2} y={height - 8} textAnchor="middle" className="axis">
        cost per successful task (log scale, cheaper is left)
      </text>
      {cards.map((card) => {
        const cost = card.cost_per_success_usd;
        if (!cost) return null;
        const isRouter = card.strategy === "router_learned";
        const offset = LABEL_OFFSETS[card.strategy] ?? DEFAULT_OFFSET;
        return (
          <g key={card.strategy}>
            <circle cx={x(cost)} cy={y(card.success_rate)} r={isRouter ? 8 : 6}
              className={isRouter ? "dot router" : "dot"} />
            <text
              x={x(cost) + offset.dx}
              y={y(card.success_rate) + offset.dy}
              textAnchor={offset.anchor}
              className={isRouter ? "axis strong" : "axis"}
            >
              {card.strategy.replace(/_/g, " ")}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

export default function BenchmarkView() {
  const [data, setData] = useState<BenchmarkResults | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.benchmark().then(setData).catch((e: ApiError) => setError(e.message));
  }, []);

  if (error) return <div className="panel error-text">{error}</div>;
  if (!data) return <div className="panel muted">Loading benchmark…</div>;

  const cards = ORDER.map((name) => data.scorecards[name]).filter(Boolean);
  const routerVsFrontier = data.comparisons.find((c) => c.b === "fixed_strongest");

  return (
    <div className="stack">
      <div className="banner">
        Results use <strong>simulated models</strong> whose skill and the judge's accuracy are
        assumptions, so they show that the method works, not how real models perform.{" "}
        {data.config.n_tasks} tasks, {data.config.repeats} repeats, scored on the held-out test split
        ({data.split_sizes.test} tasks) against ground truth.
      </div>

      <div className="panel">
        <h2>Scorecard</h2>
        <table>
          <thead>
            <tr>
              <th>Strategy</th><th>Success (95% CI)</th><th>Cost / success</th>
              <th>Latency</th><th>Escalated</th><th>Abstained</th><th>Wrong but accepted</th>
            </tr>
          </thead>
          <tbody>
            {cards.map((c) => (
              <tr key={c.strategy} className={c.strategy === "router_learned" ? "chosen" : ""}>
                <td>{c.strategy}</td>
                <td>{pct(c.success_rate)} <span className="muted">[{pct(c.success_ci[0])}, {pct(c.success_ci[1])}]</span></td>
                <td>{usd(c.cost_per_success_usd)}</td>
                <td>{seconds(c.mean_latency_s)}</td>
                <td>{pct(c.escalation_rate)}</td>
                <td>{pct(c.abstain_rate)}</td>
                <td>{pct(c.false_acceptance_rate)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="grid-2">
        <div className="panel">
          <h2>Cost against quality</h2>
          <CostQualityChart cards={cards} />
        </div>
        <div className="panel">
          <h2>Which model answered</h2>
          {Object.entries(data.routing_mix).map(([category, counts]) => {
            const total = Object.values(counts).reduce((a, b) => a + b, 0);
            return (
              <div key={category} className="mix-row">
                <span className="mix-label">{category}</span>
                <div className="mix-bar">
                  {["mock-small", "mock-mid", "mock-frontier"].map((model) => (
                    <div key={model} title={`${model}: ${counts[model] ?? 0}`}
                      style={{ width: `${((counts[model] ?? 0) / total) * 100}%`, background: MODEL_COLORS[model] }} />
                  ))}
                </div>
              </div>
            );
          })}
          <div className="legend">
            {Object.entries(MODEL_COLORS).map(([model, color]) => (
              <span key={model}><i style={{ background: color }} />{model}</span>
            ))}
          </div>
        </div>
      </div>

      <div className="panel">
        <h2>Is the router really different? (paired, same tasks)</h2>
        <table>
          <thead>
            <tr><th>router_learned vs</th><th>Success difference</th><th>Cost / success difference (negative = router cheaper)</th><th>McNemar p</th></tr>
          </thead>
          <tbody>
            {data.comparisons.map((c) => (
              <tr key={c.b}>
                <td>{c.b}</td>
                <td>{signedPct(c.success_diff)} <span className="muted">[{signedPct(c.success_diff_ci[0])}, {signedPct(c.success_diff_ci[1])}]</span></td>
                <td>{c.cost_per_success_diff >= 0 ? "+" : "−"}{usd(Math.abs(c.cost_per_success_diff))}</td>
                <td>{c.mcnemar_p < 0.001 ? "< 0.001" : c.mcnemar_p.toFixed(3)}</td>
              </tr>
            ))}
          </tbody>
        </table>
        {routerVsFrontier && (
          <p className="muted">
            Against always using the strongest model, success is statistically indistinguishable
            (p = {routerVsFrontier.mcnemar_p.toFixed(2)}) while cost per success is lower.
          </p>
        )}
      </div>

      {data.sensitivity.length > 0 && (
        <div className="panel">
          <h2>How much the saving depends on the judge</h2>
          <table>
            <thead>
              <tr><th>Judge</th><th>Router success</th><th>Strongest model</th><th>Wrong but accepted</th><th>Cost saving</th></tr>
            </thead>
            <tbody>
              {data.sensitivity.map((row) => (
                <tr key={row.judge}>
                  <td>{row.judge}</td>
                  <td>{pct(row.router_success)}</td>
                  <td>{pct(row.frontier_success)}</td>
                  <td>{pct(row.router_false_acceptance)}</td>
                  <td className={row.saving_vs_frontier < 0 ? "bad" : ""}>{signedPct(row.saving_vs_frontier)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="muted">
            Verification is not free. A weak or expensive judge can erase the saving, so the judge
            is part of the system to be measured.
          </p>
        </div>
      )}
    </div>
  );
}
