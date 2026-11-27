import type { EpistemicLabel } from "../types";

/** Format ISO YYYY-MM-DD (or DD-MM-YYYY) into Indian display format DD-MM-YYYY. */
export function formatDateIN(raw: string | null | undefined): string {
  if (!raw) return "—";
  const s = String(raw).trim();
  if (/^\d{2}-\d{2}-\d{4}/.test(s)) return s.slice(0, 10);
  if (/^\d{4}-\d{2}-\d{2}/.test(s)) {
    return `${s.slice(8, 10)}-${s.slice(5, 7)}-${s.slice(0, 4)}`;
  }
  return s;
}

/** Format minutes into `Xh Ym` (returns `—` for null/NaN; never zero-fills). */
export function formatSleepHm(minutes: number | string | null | undefined): string {
  if (minutes === null || minutes === undefined || minutes === "") return "—";
  const n = Number(minutes);
  if (!Number.isFinite(n)) return "—";
  const h = Math.floor(n / 60);
  const m = Math.round(n % 60);
  return `${h}h ${m}m`;
}

/** Format numeric value with fixed decimals or `—` when null/NaN. */
export function formatNum(
  val: number | string | null | undefined,
  decimals = 1,
  suffix = ""
): string {
  if (val === null || val === undefined || val === "") return "—";
  const n = Number(val);
  if (!Number.isFinite(n)) return "—";
  return `${n.toFixed(decimals)}${suffix}`;
}

/** Format signed delta or Z-score (`+0.85` / `-1.20`). */
export function formatSigned(
  val: number | string | null | undefined,
  decimals = 2,
  suffix = ""
): string {
  if (val === null || val === undefined || val === "") return "—";
  const n = Number(val);
  if (!Number.isFinite(n)) return "—";
  const sign = n > 0 ? "+" : "";
  return `${sign}${n.toFixed(decimals)}${suffix}`;
}

export function formatPct(val: number | string | null | undefined, isRatio = false): string {
  if (val === null || val === undefined || val === "") return "—";
  const n = Number(val);
  if (!Number.isFinite(n)) return "—";
  return `${(isRatio ? n * 100 : n).toFixed(1)}%`;
}

export function safeJsonParse<T>(raw: unknown, fallback: T): T {
  if (raw === null || raw === undefined) return fallback;
  if (typeof raw !== "string") return raw as T;
  try {
    return JSON.parse(raw) as T;
  } catch {
    return fallback;
  }
}

export function normalizeEpistemicLabel(raw: string | null | undefined): EpistemicLabel {
  const s = String(raw || "Descriptive").trim();
  if (
    s === "Measured" ||
    s === "Descriptive" ||
    s === "Correlation" ||
    s === "Prediction" ||
    s === "Causal_Hypothesis" ||
    s === "Screening_Context"
  ) {
    return s;
  }
  return "Descriptive";
}

export function epistemicDisplayText(label: EpistemicLabel): string {
  return label.replace(/_/g, " ");
}
