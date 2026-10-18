export const usd = (value: number | null): string => {
  if (value === null) return "n/a";
  if (value === 0) return "$0";
  if (value < 0.01) return `$${value.toPrecision(3)}`;
  return `$${value.toFixed(4)}`;
};

export const pct = (value: number, digits = 1): string => `${(value * 100).toFixed(digits)}%`;

export const seconds = (value: number): string => `${value.toFixed(2)}s`;

export const signedPct = (value: number): string =>
  `${value >= 0 ? "+" : ""}${(value * 100).toFixed(1)}%`;
