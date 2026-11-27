"""Core V2.0 Statistical Standards, Autocorrelation Effective-N, & Maturity Gates.

Enforced on EVERY V2.0 statistical output before persisting to Health_Intelligence_Store_v2:
1. Mandatory multi-dimensional audit columns on every row:
   - `raw_n`, `valid_n`, `effective_n`, `missingness_pct`, `predictor_count`,
     `stability_flag`, `epistemic_label`, `maturity_level`, `caveat_note`
2. Two explicit maturity gate scales:
   - Daily / Event Series Gate (`<15` Level 0 Baseline, `15..29` Level 1 Exploratory, `>=30` Level 2 Modeling)
   - Weigh-In Interval Gate (`<12` Level 0 Descriptive Ledger, `12..23` Level 1 Early Signal, `>=24` Level 2 Model)
   plus Treatment Scorecard Gate (`<14` Level 0 Collecting Baseline).
3. Unified reproducibility log (`Stats_Run_Audit`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Callable

import numpy as np
import pandas as pd

from health_engine.config_loader import (
    EPISTEMIC_CLASSES,
    now_ist_str,
    to_iso_date_str,
    validate_epistemic_class,
)

MANDATORY_AUDIT_COLUMNS: tuple[str, ...] = (
    "raw_n",
    "valid_n",
    "effective_n",
    "missingness_pct",
    "predictor_count",
    "stability_flag",
    "epistemic_label",
    "maturity_level",
    "caveat_note",
)

STATS_RUN_AUDIT_COLUMNS: list[str] = [
    "analysis_id",
    "module",
    "run_timestamp_ist",
    "epistemic_label",
    "maturity_level",
    "raw_n",
    "valid_n",
    "effective_n",
    "missingness_pct",
    "cv_or_ci_summary",
    "stability_flag",
    "caveat_note",
    "epistemic_class",
]


def compute_effective_n_with_r1(
    series: pd.Series | np.ndarray | list[float],
    dates: pd.Series | np.ndarray | list[str] | None = None,
) -> tuple[int, float]:
    """Compute autocorrelation-adjusted independent sample size n_eff and lag-1 r1.

    Formula:
        n_eff = clip(round(n * (1 - r1) / (1 + r1)), 1, n)
    Falls back to valid_n if n < 5 or if r1 cannot be estimated (zero variance / insufficient pairs).
    """
    vals = pd.to_numeric(pd.Series(series), errors="coerce").to_numpy(dtype=float)
    valid_mask = np.isfinite(vals)
    n = int(valid_mask.sum())
    if n == 0:
        return 0, np.nan
    if n < 5:
        return n, np.nan

    y_prev: np.ndarray | None = None
    y_curr: np.ndarray | None = None

    if dates is not None and len(dates) == len(vals):
        iso_list = [to_iso_date_str(d) for d in dates]
        paired: list[tuple[date, float]] = []
        for ds, v in zip(iso_list, vals):
            if ds is not None and np.isfinite(v):
                try:
                    paired.append((date.fromisoformat(ds), float(v)))
                except ValueError:
                    continue
        paired.sort(key=lambda item: item[0])
        adj_prev: list[float] = []
        adj_curr: list[float] = []
        for idx in range(1, len(paired)):
            if (paired[idx][0] - paired[idx - 1][0]).days == 1:
                adj_prev.append(paired[idx - 1][1])
                adj_curr.append(paired[idx][1])
        if len(adj_prev) >= 4:
            y_prev = np.asarray(adj_prev, dtype=float)
            y_curr = np.asarray(adj_curr, dtype=float)
        elif len(paired) >= 5:
            ordered = np.asarray([p[1] for p in paired], dtype=float)
            y_prev = ordered[:-1]
            y_curr = ordered[1:]

    if y_prev is None or y_curr is None:
        clean = vals[valid_mask]
        y_prev = clean[:-1]
        y_curr = clean[1:]

    if len(y_prev) < 4:
        return n, np.nan

    sd_prev = float(np.std(y_prev, ddof=1))
    sd_curr = float(np.std(y_curr, ddof=1))
    if sd_prev <= 1e-12 or sd_curr <= 1e-12:
        return n, np.nan

    r1 = float(np.corrcoef(y_prev, y_curr)[0, 1])
    if not np.isfinite(r1) or (1.0 + r1) <= 1e-9:
        return n, np.nan

    raw_neff = float(n) * (1.0 - r1) / (1.0 + r1)
    neff = int(np.clip(np.round(raw_neff), 1, n))
    return neff, round(r1, 4)


def compute_effective_n(
    series: pd.Series | np.ndarray | list[float],
    dates: pd.Series | np.ndarray | list[str] | None = None,
) -> int:
    """Return integer autocorrelation-adjusted independent sample size n_eff."""
    neff, _ = compute_effective_n_with_r1(series, dates=dates)
    return neff


def compute_missingness_pct(raw_n: int, valid_n: int) -> float:
    """Return missingness percentage in [0.0, 100.0]."""
    if raw_n <= 0:
        return 100.0
    pct = (1.0 - (float(valid_n) / float(raw_n))) * 100.0
    return round(float(np.clip(pct, 0.0, 100.0)), 2)


def classify_daily_maturity(
    valid_n: int,
    level1_min: int = 15,
    level2_min: int = 30,
) -> tuple[int, str, str]:
    """Classify Daily / Event Series maturity gate -> (level_int, maturity_level, badge)."""
    n = int(max(0, valid_n))
    if n < level1_min:
        return 0, "Level 0 (Baseline)", "Baseline"
    if n < level2_min:
        return 1, "Level 1 (Exploratory)", "Exploratory"
    return 2, "Level 2 (Statistical Modeling)", "Statistical_Modeling"


def classify_interval_maturity(
    valid_intervals: int,
    level1_min: int = 12,
    level2_min: int = 24,
) -> tuple[int, str, str]:
    """Classify Weigh-In Interval maturity gate -> (level_int, maturity_level, badge)."""
    n = int(max(0, valid_intervals))
    if n < level1_min:
        return 0, "Level 0 (Descriptive Ledger)", "Descriptive_Ledger"
    if n < level2_min:
        return 1, "Level 1 (Early Signal)", "Early_Signal"
    return 2, "Level 2 (Statistical Model)", "Statistical_Model"


def classify_scorecard_maturity(
    valid_n: int,
    min_baseline_n: int = 14,
    level2_min: int = 30,
) -> tuple[int, str, str]:
    """Classify Treatment Scorecard maturity -> (level_int, maturity_level, badge)."""
    n = int(max(0, valid_n))
    if n < min_baseline_n:
        return 0, "Level 0 (Collecting Baseline)", "Collecting_Baseline"
    if n < level2_min:
        return 1, "Level 1 (Exploratory)", "Exploratory"
    return 2, "Level 2 (Statistical Modeling)", "Statistical_Modeling"


def classify_stability_from_sign_agreement(
    sign_agreement: float,
    valid_n: int,
    min_required_n: int = 15,
    stable_thresh: float = 0.90,
    moderate_thresh: float = 0.75,
) -> str:
    """Map bootstrap/subsample sign agreement [0, 1] to stability_flag."""
    if valid_n < min_required_n or not np.isfinite(sign_agreement):
        return "Insufficient_N"
    if sign_agreement >= stable_thresh:
        return "Stable"
    if sign_agreement >= moderate_thresh:
        return "Moderate"
    return "Unstable"


def bootstrap_bivariate_ci(
    x: np.ndarray,
    y: np.ndarray,
    stat_fn: Callable[[np.ndarray, np.ndarray], float],
    point_est: float,
    n_resamples: int = 1000,
    seed: int = 42,
    min_required_n: int = 12,
) -> tuple[float, float, float, str]:
    """Compute 95% percentile bootstrap CI and sign-agreement stability_flag."""
    n = len(x)
    if n < min_required_n or not np.isfinite(point_est):
        return np.nan, np.nan, np.nan, "Insufficient_N"

    rng = np.random.default_rng(seed)
    boot_vals = np.empty(n_resamples, dtype=float)
    boot_vals.fill(np.nan)

    for b in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        xb = x[idx]
        yb = y[idx]
        if np.std(xb) <= 1e-12 or np.std(yb) <= 1e-12:
            continue
        boot_vals[b] = stat_fn(xb, yb)

    finite_boot = boot_vals[np.isfinite(boot_vals)]
    if len(finite_boot) < max(50, n_resamples // 4):
        return np.nan, np.nan, np.nan, "Insufficient_N"

    ci_low, ci_high = np.percentile(finite_boot, [2.5, 97.5])
    if abs(point_est) <= 1e-12:
        sign_agree = 0.5
    else:
        target_sign = np.sign(point_est)
        sign_agree = float(np.mean(np.sign(finite_boot) == target_sign))

    stab = classify_stability_from_sign_agreement(
        sign_agree, valid_n=n, min_required_n=min_required_n
    )
    return round(float(ci_low), 4), round(float(ci_high), 4), round(sign_agree, 4), stab


def subsample_univariate_stability(
    values: np.ndarray,
    reference_center: float,
    min_required_n: int = 14,
    n_resamples: int = 500,
    seed: int = 42,
) -> str:
    """Assess bootstrap sign stability of (sample_mean - reference_center) for descriptive deltas."""
    clean = values[np.isfinite(values)]
    n = len(clean)
    if n < min_required_n or not np.isfinite(reference_center):
        return "Insufficient_N"
    diff = float(np.mean(clean) - reference_center)
    if abs(diff) <= 1e-12:
        return "Moderate"
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    boot_means = np.mean(clean[idx], axis=1) - reference_center
    sign_agree = float(np.mean(np.sign(boot_means) == np.sign(diff)))
    return classify_stability_from_sign_agreement(
        sign_agree, valid_n=n, min_required_n=min_required_n
    )


@dataclass
class StatsAuditLogger:
    """Collects standardized V2.0 reproducibility records for `Stats_Run_Audit`."""

    run_timestamp_ist: str = field(default_factory=now_ist_str)
    records: list[dict[str, Any]] = field(default_factory=list)

    def log_analysis(
        self,
        analysis_id: str,
        module: str,
        epistemic_label: str,
        maturity_level: str,
        raw_n: int,
        valid_n: int,
        effective_n: int,
        missingness_pct: float,
        cv_or_ci_summary: str,
        stability_flag: str,
        caveat_note: str,
    ) -> None:
        validate_epistemic_class(epistemic_label)
        self.records.append(
            {
                "analysis_id": str(analysis_id),
                "module": str(module),
                "run_timestamp_ist": self.run_timestamp_ist,
                "epistemic_label": epistemic_label,
                "maturity_level": str(maturity_level),
                "raw_n": int(raw_n),
                "valid_n": int(valid_n),
                "effective_n": int(effective_n),
                "missingness_pct": round(float(missingness_pct), 2),
                "cv_or_ci_summary": str(cv_or_ci_summary),
                "stability_flag": str(stability_flag),
                "caveat_note": str(caveat_note),
                "epistemic_class": epistemic_label,
            }
        )

    def to_dataframe(self) -> pd.DataFrame:
        if not self.records:
            return pd.DataFrame(columns=STATS_RUN_AUDIT_COLUMNS)
        return pd.DataFrame(self.records, columns=STATS_RUN_AUDIT_COLUMNS)


def enforce_maturity_audit(
    df: pd.DataFrame,
    *,
    module_name: str,
    analysis_id: str,
    audit_logger: StatsAuditLogger | None = None,
    suppressed_at_level0: tuple[str, ...] = (),
    default_epistemic_label: str = "Descriptive",
    summary_cv_or_ci: str = "",
    summary_caveat: str = "",
) -> pd.DataFrame:
    """Validate mandatory audit columns, enforce Level 0 suppression, and log to Stats_Run_Audit."""
    validate_epistemic_class(default_epistemic_label)
    out = df.copy()

    if out.empty:
        for col in MANDATORY_AUDIT_COLUMNS:
            if col not in out.columns:
                out[col] = pd.Series(dtype="object")
        if "epistemic_class" not in out.columns:
            out["epistemic_class"] = pd.Series(dtype="object")
        if audit_logger is not None:
            audit_logger.log_analysis(
                analysis_id=analysis_id,
                module=module_name,
                epistemic_label=default_epistemic_label,
                maturity_level="Level 0 (Empty_Scope)",
                raw_n=0,
                valid_n=0,
                effective_n=0,
                missingness_pct=100.0,
                cv_or_ci_summary=summary_cv_or_ci or "n=0 (no rows in scope)",
                stability_flag="Insufficient_N",
                caveat_note=summary_caveat or "No valid records available in scope.",
            )
        return out

    missing_cols = [c for c in MANDATORY_AUDIT_COLUMNS if c not in out.columns]
    if missing_cols:
        raise ValueError(
            f"[{module_name}:{analysis_id}] Missing mandatory V2.0 maturity audit columns: {missing_cols}"
        )

    # Enforce Level 0 suppression of correlation coefficients, p-values, and regression weights
    lvl_str = out["maturity_level"].fillna("").astype(str)
    is_level0 = lvl_str.str.startswith("Level 0", na=False)
    for scol in suppressed_at_level0:
        if scol in out.columns:
            out.loc[is_level0, scol] = np.nan

    # Ensure Level 0 rows never overclaim Correlation or Prediction epistemic class
    if is_level0.any():
        overclaim_mask = is_level0 & out["epistemic_label"].isin(["Correlation", "Prediction"])
        if overclaim_mask.any():
            out.loc[overclaim_mask, "epistemic_label"] = "Descriptive"

    # Validate every epistemic_label against AGENTS.md Rule 3 enum
    invalid_labels = set(out["epistemic_label"].dropna().astype(str).unique()) - EPISTEMIC_CLASSES
    if invalid_labels:
        raise ValueError(
            f"[{module_name}:{analysis_id}] Invalid epistemic_label values: {sorted(invalid_labels)}"
        )

    # Mirror epistemic_class for unified downstream compatibility
    out["epistemic_class"] = out["epistemic_label"]

    if audit_logger is not None:
        raw_max = int(pd.to_numeric(out["raw_n"], errors="coerce").fillna(0).max())
        val_max = int(pd.to_numeric(out["valid_n"], errors="coerce").fillna(0).max())
        eff_max = int(pd.to_numeric(out["effective_n"], errors="coerce").fillna(0).max())
        miss_med = float(pd.to_numeric(out["missingness_pct"], errors="coerce").fillna(100.0).median())
        top_mat = str(out["maturity_level"].mode().iloc[0]) if not out["maturity_level"].dropna().empty else "Level 0"
        top_ep = str(out["epistemic_label"].mode().iloc[0]) if not out["epistemic_label"].dropna().empty else default_epistemic_label
        top_stab = str(out["stability_flag"].mode().iloc[0]) if not out["stability_flag"].dropna().empty else "Insufficient_N"
        note = summary_caveat or str(out["caveat_note"].dropna().iloc[0] if not out["caveat_note"].dropna().empty else "")
        audit_logger.log_analysis(
            analysis_id=analysis_id,
            module=module_name,
            epistemic_label=top_ep,
            maturity_level=top_mat,
            raw_n=raw_max,
            valid_n=val_max,
            effective_n=eff_max,
            missingness_pct=miss_med,
            cv_or_ci_summary=summary_cv_or_ci or f"rows={len(out)}, max_valid_n={val_max}",
            stability_flag=top_stab,
            caveat_note=note,
        )

    return out
