// Mirrors backend/app/api/schemas.py. Keep the two in sync.

export type Risk = "low" | "medium" | "high";
export type Category = "extraction" | "summarization" | "reasoning" | "coding" | "tool_use";
export type Mode = "mock" | "live";

export interface RunRequest {
  task: string;
  mode: Mode;
  risk: Risk;
  category: Category;
  contains_sensitive_data: boolean;
  max_cost_usd: number;
  max_calls: number;
  max_steps: number;
}

export interface Estimate {
  mean: number;
  lower_bound: number;
  sample_size: number;
  sufficient_evidence: boolean;
}

export interface Candidate {
  model_id: string;
  tier: string;
  expected_cost_usd: number;
  qualified: boolean;
  estimate: Estimate;
}

export interface Routing {
  model_id: string;
  mode: string;
  reason: string;
  quality_floor: number;
  fallbacks: string[];
  requires_independent_verifier: boolean;
  candidates: Candidate[];
  rejected: Record<string, string[]>;
}

export interface Check {
  name: string;
  status: "pass" | "fail" | "error";
  detail: string;
  cost_usd: number;
}

export interface Attempt {
  model_id: string;
  input_tokens: number;
  output_tokens: number;
  latency_s: number;
  generation_cost_usd: number;
  verification_cost_usd: number;
  action: "accept" | "escalate" | "abstain" | null;
  error: string;
  checks: Check[];
}

export interface CallTrace {
  agent: "planner" | "worker" | "verifier" | "synthesizer";
  category: string;
  risk: string;
  final_action: "accept" | "escalate" | "abstain";
  reason: string;
  output: string | null;
  cost_usd: number;
  latency_s: number;
  escalations: number;
  routing: Routing | null;
  attempts: Attempt[];
}

export interface RunResponse {
  mode: Mode;
  task: string;
  completed: boolean;
  final_answer: string | null;
  abstained_at: string | null;
  cost_usd: number;
  latency_s: number;
  budget: { spent_usd: number; calls: number; max_cost_usd: number; max_calls: number };
  traces: CallTrace[];
}

export interface Health {
  status: string;
  version: string;
  configured_providers: string[];
}

export interface Scorecard {
  strategy: string;
  n: number;
  success_rate: number;
  success_ci: [number, number];
  total_cost_usd: number;
  cost_per_success_usd: number | null;
  mean_latency_s: number;
  escalation_rate: number;
  abstain_rate: number;
  false_acceptance_rate: number;
}

export interface PairedComparison {
  a: string;
  b: string;
  a_only: number;
  b_only: number;
  mcnemar_p: number;
  success_diff: number;
  success_diff_ci: [number, number];
  cost_per_success_diff: number;
  cost_per_success_diff_ci: [number, number];
}

export interface SensitivityRow {
  judge: string;
  router_success: number;
  router_false_acceptance: number;
  frontier_success: number;
  saving_vs_frontier: number;
}

export interface BenchmarkResults {
  config: { n_tasks: number; repeats: number; seed: number };
  split_sizes: Record<string, number>;
  chosen_low_floor: number;
  scorecards: Record<string, Scorecard>;
  comparisons: PairedComparison[];
  routing_mix: Record<string, Record<string, number>>;
  sensitivity: SensitivityRow[];
}
