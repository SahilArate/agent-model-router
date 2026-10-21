import type { BenchmarkResults, Health, ModelInfo, RunRequest, RunResponse } from "./types";

const BASE = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${BASE}${path}`, {
      headers: { "Content-Type": "application/json" },
      ...init,
    });
  } catch {
    throw new ApiError(`Cannot reach the API at ${BASE}. Is the backend running?`, 0);
  }
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    const detail = typeof body?.detail === "string" ? body.detail : response.statusText;
    throw new ApiError(detail, response.status);
  }
  return response.json() as Promise<T>;
}

export const api = {
  health: () => request<Health>("/health"),
  run: (body: RunRequest) =>
    request<RunResponse>("/run", { method: "POST", body: JSON.stringify(body) }),
  models: () => request<ModelInfo[]>("/models"),
  benchmark: () => request<BenchmarkResults>("/benchmark"),
};
