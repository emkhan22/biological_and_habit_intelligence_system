"""Gold Table 2: Gold_Personal_Baselines & Rolling Trends.

Rules (Prompt Section 4.2):
- Explicit phase baselines:
  - `Pre_Oct_Baseline`: `2026-05-01` to `2026-09-30`
  - `Active_GLP1_Phase`: `2026-10-21` onward
- Target metrics: HRV (`hrv_rmssd_ms`), Sleep Avg HR (`sleep_hr_avg`), Daytime Avg HR
  (`daytime_avg_hr`), Sleep Duration (`asleep_min`), Deep Sleep (`deep_min`),
  Bedtime (`bedtime_hr_rel_mid`), and Steps (`steps`).
- Compute BOTH classical Z-score `(x - mean) / std` AND robust Z-score
  `(x - median) / (1.4826 * MAD)` relative to each baseline, plus 7d, 14d, and 28d
  rolling means, SDs, and linear slopes.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from health_engine.config_loader import add_display_date_col


BASELINE_METRICS: dict[str, str] = {
    "hrv": "hrv_rmssd_ms",
    "sleep_hr": "sleep_hr_avg",
    "daytime_hr": "daytime_avg_hr",
    "sleep_dur": "asleep_min",
    "deep_sleep": "deep_min",
    "bedtime": "bedtime_hr_rel_mid",
    "steps": "steps",
}


def _ols_slope_window(values: np.ndarray, min_pts: int) -> float:
    """Compute OLS linear slope (units/day) over a trailing calendar window ignoring NaNs."""
    valid_mask = ~np.isnan(values)
    if int(valid_mask.sum()) < max(2, min_pts):
        return np.nan
    x = np.nonzero(valid_mask)[0].astype(float)
    y = values[valid_mask].astype(float)
    x_centered = x - x.mean()
    denom = float(np.dot(x_centered, x_centered))
    if denom <= 0.0:
        return np.nan
    return float(np.dot(x_centered, y - y.mean()) / denom)


def _compute_phase_stats(
    series: pd.Series,
    mask: pd.Series,
    mad_scale: float = 1.4826,
) -> dict[str, float]:
    vals = pd.to_numeric(series.loc[mask], errors="coerce").dropna().to_numpy(dtype=float)
    n = len(vals)
    if n == 0:
        return {
            "n_valid": 0,
            "mean": np.nan,
            "std": np.nan,
            "median": np.nan,
            "mad": np.nan,
            "scaled_mad": np.nan,
        }
    mean_v = float(np.mean(vals))
    std_v = float(np.std(vals, ddof=1)) if n >= 2 else np.nan
    med_v = float(np.median(vals))
    mad_v = float(np.median(np.abs(vals - med_v)))
    scaled_mad = mad_scale * mad_v
    return {
        "n_valid": n,
        "mean": mean_v,
        "std": std_v,
        "median": med_v,
        "mad": mad_v,
        "scaled_mad": scaled_mad,
    }


def _safe_zscore(x: np.ndarray, center: float, scale: float) -> np.ndarray:
    if np.isnan(center) or np.isnan(scale) or scale <= 1e-12:
        return np.full(len(x), np.nan, dtype=float)
    return np.round((x - center) / scale, 4)


def build_gold_personal_baselines(
    gold_daily: pd.DataFrame,
    thresholds_cfg: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (Gold_Personal_Baselines, Gold_Baseline_Summary)."""
    if gold_daily.empty:
        return pd.DataFrame(), pd.DataFrame()

    anchors = thresholds_cfg["phase_anchors"]
    pre_start = str(anchors["pre_oct_baseline"]["start_date"])
    pre_end = str(anchors["pre_oct_baseline"]["end_date"])
    glp1_start = str(anchors["active_glp1_phase"]["start_date"])

    roll_cfg = thresholds_cfg["rolling_maturity_gates"]
    windows: list[int] = [int(w) for w in roll_cfg["windows_days"]]
    min_periods_map: dict[int, int] = {
        int(k): int(v) for k, v in roll_cfg["min_periods"].items()
    }
    mad_scale = float(roll_cfg.get("robust_mad_scale", 1.4826))

    df = gold_daily.sort_values("date").reset_index(drop=True)
    pre_mask = df["date"].between(pre_start, pre_end)
    glp1_mask = df["date"] >= glp1_start

    phase_defs = [
        ("Pre_Oct_Baseline", pre_mask, f"{pre_start}..{pre_end}"),
        ("Active_GLP1_Phase", glp1_mask, f"{glp1_start}..present"),
    ]

    summary_rows: list[dict[str, Any]] = []
    out_cols: dict[str, Any] = {
        "date": df["date"].to_numpy(),
        "phase_membership": np.where(
            pre_mask,
            "Pre_Oct_Baseline",
            np.where(glp1_mask, "Active_GLP1_Phase", "Other_Historical"),
        ),
    }

    for short_name, col_name in BASELINE_METRICS.items():
        s = (
            pd.to_numeric(df[col_name], errors="coerce")
            if col_name in df.columns
            else pd.Series(np.nan, index=df.index, dtype=float)
        )
        arr = s.to_numpy(dtype=float)
        out_cols[col_name] = arr

        for phase_label, pmask, span_str in phase_defs:
            st = _compute_phase_stats(s, pmask, mad_scale=mad_scale)
            summary_rows.append(
                {
                    "phase_name": phase_label,
                    "phase_window": span_str,
                    "metric_key": short_name,
                    "source_column": col_name,
                    "n_valid": int(st["n_valid"]),
                    "mean": round(st["mean"], 4) if not np.isnan(st["mean"]) else np.nan,
                    "std": round(st["std"], 4) if not np.isnan(st["std"]) else np.nan,
                    "median": round(st["median"], 4) if not np.isnan(st["median"]) else np.nan,
                    "mad": round(st["mad"], 4) if not np.isnan(st["mad"]) else np.nan,
                    "scaled_mad_14826": (
                        round(st["scaled_mad"], 4) if not np.isnan(st["scaled_mad"]) else np.nan
                    ),
                    "epistemic_class": "Descriptive",
                }
            )
            tag = "pre_oct" if phase_label == "Pre_Oct_Baseline" else "glp1"
            out_cols[f"{short_name}_z_classical_{tag}"] = _safe_zscore(
                arr, st["mean"], st["std"]
            )
            out_cols[f"{short_name}_z_robust_{tag}"] = _safe_zscore(
                arr, st["median"], st["scaled_mad"]
            )

        # Rolling 7d, 14d, 28d mean, SD, and OLS linear slope
        for w in windows:
            mp = min_periods_map.get(w, max(2, w // 2))
            roll = s.rolling(window=w, min_periods=mp)
            out_cols[f"{short_name}_roll{w}d_mean"] = np.round(roll.mean().to_numpy(dtype=float), 4)
            out_cols[f"{short_name}_roll{w}d_sd"] = np.round(roll.std(ddof=1).to_numpy(dtype=float), 4)

            slopes = np.full(len(arr), np.nan, dtype=float)
            for i in range(len(arr)):
                lo = max(0, i - w + 1)
                win_slice = arr[lo : i + 1]
                slopes[i] = _ols_slope_window(win_slice, min_pts=mp)
            out_cols[f"{short_name}_roll{w}d_slope"] = np.round(slopes, 4)

    out_cols["epistemic_class"] = "Descriptive"
    baselines_df = add_display_date_col(pd.DataFrame(out_cols), "date")
    summary_df = pd.DataFrame(summary_rows)
    return baselines_df, summary_df
