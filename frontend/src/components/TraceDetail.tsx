import { pct, seconds, usd } from "../format";
import type { Attempt, CallTrace, EscalationStep } from "../types";

function AttemptRow({ attempt, index }: { attempt: Attempt; index: number }) {
  return (
    <li className="attempt">
      <div className="attempt-head">
        <strong>
          Attempt {index + 1}: {attempt.model_id}
        </strong>
        <span className={`badge ${attempt.action === "accept" ? "ok" : "warn"}`}>
          {attempt.error ? "provider error" : (attempt.action ?? "n/a")}
        </span>
      </div>
      <div className="muted">
        {attempt.input_tokens} in / {attempt.output_tokens} out tokens · generation{" "}
        {usd(attempt.generation_cost_usd)} · verification {usd(attempt.verification_cost_usd)} ·{" "}
        {seconds(attempt.latency_s)}
      </div>
      {attempt.error && <div className="error-text">{attempt.error}</div>}
      <div className="checks">
        {attempt.checks.map((check) => (
          <span key={check.name} className={`badge ${check.status}`} title={check.detail}>
            {check.name}
            {check.judge_model ? ` (${check.judge_model})` : ""}: {check.status}
          </span>
        ))}
      </div>
    </li>
  );
}

function EscalationCard({ step }: { step: EscalationStep }) {
  const verdict =
    step.check === "provider"
      ? "the provider call failed"
      : step.judge_model
        ? `judge ${step.judge_model}${step.independent ? " (a different provider)" : " (same provider: weaker independence)"} rejected the answer`
        : `the ${step.check} check failed`;
  return (
    <div className="escalation">
      <div className="escalation-head">
        <strong>{step.from_model}</strong>
        <span aria-hidden="true">→</span>
        <strong>{step.to_model}</strong>
        <span className="muted">wasted {usd(step.wasted_cost_usd)}</span>
      </div>
      <div>Because {verdict}:</div>
      <blockquote>{step.reason}</blockquote>
    </div>
  );
}

export default function TraceDetail({ trace }: { trace: CallTrace }) {
  const routing = trace.routing;
  return (
    <div className="panel detail">
      <h2>
        {trace.agent} <span className="muted">· {trace.category} · {trace.risk} risk</span>
      </h2>

      {routing ? (
        <>
          <p className="reason">
            <span className="badge info">{routing.mode}</span> {routing.reason}
          </p>
          <table>
            <thead>
              <tr>
                <th>Model</th>
                <th>Tier</th>
                <th>Expected cost</th>
                <th>Measured rate</th>
                <th>Lower bound</th>
                <th>Samples</th>
                <th>Floor {routing.quality_floor.toFixed(2)}</th>
              </tr>
            </thead>
            <tbody>
              {routing.candidates.map((c) => (
                <tr key={c.model_id} className={c.model_id === routing.model_id ? "chosen" : ""}>
                  <td>{c.model_id}</td>
                  <td>{c.tier}</td>
                  <td>{usd(c.expected_cost_usd)}</td>
                  <td>{c.estimate.sample_size ? pct(c.estimate.mean) : "–"}</td>
                  <td>{c.estimate.sample_size ? pct(c.estimate.lower_bound) : "–"}</td>
                  <td>{c.estimate.sample_size}</td>
                  <td>{c.qualified ? "clears" : "no"}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {Object.keys(routing.rejected).length > 0 && (
            <p className="muted">
              Excluded by hard constraints:{" "}
              {Object.entries(routing.rejected)
                .map(([id, reasons]) => `${id} (${reasons.join(", ")})`)
                .join("; ")}
            </p>
          )}
        </>
      ) : (
        <p className="muted">No model could be routed for this call.</p>
      )}

      {trace.escalation_steps.length > 0 && (
        <>
          <h3>Why it was escalated</h3>
          {trace.escalation_steps.map((step, i) => (
            <EscalationCard key={i} step={step} />
          ))}
        </>
      )}

      <h3>Attempts</h3>
      <ol className="attempts">
        {trace.attempts.map((attempt, i) => (
          <AttemptRow key={i} attempt={attempt} index={i} />
        ))}
      </ol>

      <p className="muted">{trace.reason}</p>
      {trace.output && (
        <>
          <h3>Output</h3>
          <pre className="output">{trace.output}</pre>
        </>
      )}
    </div>
  );
}
