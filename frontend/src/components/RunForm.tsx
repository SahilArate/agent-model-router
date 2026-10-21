import { useState } from "react";
import type { Category, Mode, ModelInfo, Risk, RunRequest } from "../types";

const RISKS: Risk[] = ["low", "medium", "high"];
const CATEGORIES: Category[] = ["extraction", "summarization", "reasoning", "coding", "tool_use"];

const EXAMPLES = [
  { label: "Summarise a contract", task: "Summarise the key obligations in a 12-page supplier contract." },
  { label: "Debug a function", task: "Find why the invoice total is off by one cent and propose a fix." },
  { label: "Plan a migration", task: "Plan the steps to migrate our billing service to a new database." },
];

interface Props {
  busy: boolean;
  models: ModelInfo[];
  onRun: (request: RunRequest) => void;
}

export default function RunForm({ busy, models, onRun }: Props) {
  const [task, setTask] = useState(EXAMPLES[0].task);
  const [risk, setRisk] = useState<Risk>("low");
  const [category, setCategory] = useState<Category>("reasoning");
  const [sensitive, setSensitive] = useState(false);
  const [maxCost, setMaxCost] = useState(0.5);
  const [mode, setMode] = useState<Mode>("mock");
  const [judge, setJudge] = useState("");

  // Only models of the chosen mode that are switched on can act as the judge.
  const judges = models.filter((m) => m.enabled && (m.provider === "mock") === (mode === "mock"));

  const changeMode = (next: Mode) => {
    setMode(next);
    setJudge("");
  };

  const submit = (event: React.FormEvent) => {
    event.preventDefault();
    onRun({
      task,
      mode,
      risk,
      category,
      contains_sensitive_data: sensitive,
      max_cost_usd: maxCost,
      max_calls: 20,
      max_steps: 3,
      judge_model: judge || null,
    });
  };

  return (
    <form className="panel form" onSubmit={submit}>
      <h2>Run a workflow</h2>

      <div className="row">
        <div>
          <label htmlFor="mode">Models</label>
          <select id="mode" value={mode} onChange={(e) => changeMode(e.target.value as Mode)}>
            <option value="mock">Simulated (free, offline)</option>
            <option value="live">Live (real providers)</option>
          </select>
        </div>
        <div>
          <label htmlFor="judge">Judge</label>
          <select id="judge" value={judge} onChange={(e) => setJudge(e.target.value)}>
            <option value="">Automatic</option>
            {judges.map((m) => (
              <option key={m.id} value={m.id}>
                {m.id}
              </option>
            ))}
          </select>
        </div>
      </div>

      <label htmlFor="task">Task</label>
      <textarea
        id="task"
        value={task}
        rows={4}
        maxLength={4000}
        onChange={(e) => setTask(e.target.value)}
      />
      <div className="chips">
        {EXAMPLES.map((example) => (
          <button
            type="button"
            key={example.label}
            className="chip"
            onClick={() => setTask(example.task)}
          >
            {example.label}
          </button>
        ))}
      </div>

      <div className="row">
        <div>
          <label htmlFor="risk">Risk</label>
          <select id="risk" value={risk} onChange={(e) => setRisk(e.target.value as Risk)}>
            {RISKS.map((r) => (
              <option key={r}>{r}</option>
            ))}
          </select>
        </div>
        <div>
          <label htmlFor="category">Task type</label>
          <select
            id="category"
            value={category}
            onChange={(e) => setCategory(e.target.value as Category)}
          >
            {CATEGORIES.map((c) => (
              <option key={c}>{c}</option>
            ))}
          </select>
        </div>
      </div>

      <label htmlFor="budget">Budget cap (USD): {maxCost.toFixed(2)}</label>
      <input
        id="budget"
        type="range"
        min={0.01}
        max={2}
        step={0.01}
        value={maxCost}
        onChange={(e) => setMaxCost(Number(e.target.value))}
      />

      <label className="check">
        <input
          type="checkbox"
          checked={sensitive}
          onChange={(e) => setSensitive(e.target.checked)}
        />
        Contains sensitive data (only approved models are eligible)
      </label>

      <button className="primary" type="submit" disabled={busy || task.trim().length === 0}>
        {busy ? "Running…" : "Run workflow"}
      </button>
      <p className="hint">
        {mode === "mock"
          ? "Uses simulated models, so it is free and works offline."
          : "Calls real models with the server's API keys. A dedicated judge never writes answers, so it cannot grade its own work."}
      </p>
    </form>
  );
}
