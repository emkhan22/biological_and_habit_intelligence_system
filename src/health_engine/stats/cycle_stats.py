"""Cycle-Phase Biometrics & Longitudinal Regularity (`cycle_stats.py` -> `Gold_Cycle_Phase_Stats`).

Rules (Prompt Section 4.5 & AGENTS.md Rules 1, 3, 4):
- Aggregate `Gold_Daily_Unified` by `cycle_phase`
  (`Menstrual`, `Follicular_D1_14`, `Mid_Extended`, `Late_Luteal_PreMenst`):
  medians, IQRs, `valid_n`, and Kruskal-Wallis test (gated at `>= 10` days per phase) for
  `steps`, `avg_hr`, `sleep_hr_avg`, `hrv_rmssd_ms`, `asleep_min`, `hunger`, `food_noise`.
- Summarize multi-year cycle regularity (`cycle_length_days`, `span_days`) alongside
  historical `Ferritin` / `MCV` / `TSH` lab dates.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from health_engine.stats.maturity import (
    StatsAuditLogger,
    classify_daily_maturity,
    compute_effective_n,
    compute_missingness_pct,
    enforce_maturity_audit,
)

CYCLE_PHASES_ORDER: tuple[str, ...] = (
    "Menstrual",
    "Follicular_D1_14",
    "Mid_Extended",
    "Late_Luteal_PreMenst",
)

CYCLE_BIOMETRIC_METRICS: tuple[str, ...] = (
    "steps",
    "avg_hr",
    "sleep_hr_avg",
    "hrv_rmssd_ms",
    "asleep_min",
    "hunger",
    "food_noise",
)


def _historical_lab_context_str(silver_labs: pd.DataFrame) -> str:
    if silver_labs.empty or "Collection_Date" not in silver_labs.columns:
        return "No Ferritin/MCV/TSH labs in scope."
    targets = {
        "ferritin": "Ferritin",
        "serum ferritin": "Ferritin",
        "mcv": "MCV",
        "tsh": "TSH",
        "thyroid stimulating hormone (tsh)": "TSH",
    }
    valid = silver_labs.loc[
        (~silver_labs.get("data_corruption_flag", pd.Series(False, index=silver_labs.index)).fillna(False))
        & silver_labs["Normalized_Value"].notna()
        & silver_labs["Collection_Date"].notna()
    ]
    snippets: list[str] = []
    for r in valid.sort_values("Collection_Date").to_dict(orient="records"):
        nm = str(r.get("Standardized_Test_Name") or "").strip().lower()
        if nm in targets:
            snippets.append(
                f"{targets[nm]}={round(float(r['Normalized_Value']), 2)} {r.get('Normalized_Unit', '')} ({r['Collection_Date']})"
            )
    return "; ".join(snippets) if snippets else "No Ferritin/MCV/TSH labs recorded in current dataset."


def build_gold_cycle_phase_stats(
    gold_daily: pd.DataFrame,
    silver_cycles: pd.DataFrame,
    silver_cycle_dim: pd.DataFrame,
    silver_labs: pd.DataFrame,
    thresholds_cfg: dict[str, Any],
    audit_logger: StatsAuditLogger | None = None,
) -> pd.DataFrame:
    """Build Gold_Cycle_Phase_Stats (Phase Biometrics + Multi-Year Regularity Summary)."""
    cols = [
        "summary_section",
        "metric_name",
        "cycle_phase_or_stat",
        "median",
        "iqr",
        "mean",
        "sd",
        "min_val",
        "max_val",
        "kruskal_gate_status",
        "kruskal_h_stat",
        "kruskal_p_value",
        "historical_hematology_thyroid_context",
        "raw_n",
        "valid_n",
        "effective_n",
        "missingness_pct",
        "predictor_count",
        "stability_flag",
        "maturity_level",
        "epistemic_label",
        "caveat_note",
    ]
    if gold_daily.empty:
        return enforce_maturity_audit(
            pd.DataFrame(columns=cols),
            module_name="cycle_stats",
            analysis_id="CYCLE_PHASE_BIOMETRICS",
            audit_logger=audit_logger,
        )

    min_kw_days = int(
        thresholds_cfg.get("v20_maturity_gates", {})
        .get("cycle_stats", {})
        .get("kruskal_min_days_per_phase", 10)
    )
    lab_ctx = _historical_lab_context_str(silver_labs)
    df = gold_daily.sort_values("date").copy()
    phase_col = df.get("cycle_phase", pd.Series("Unknown", index=df.index)).astype(str)

    out_rows: list[dict[str, Any]] = []

    # 1. Phase-by-Phase Biometrics + Gated Kruskal-Wallis Test
    for m in CYCLE_BIOMETRIC_METRICS:
        s_all = (
            pd.to_numeric(df[m], errors="coerce")
            if m in df.columns
            else pd.Series(np.nan, index=df.index, dtype=float)
        )
        phase_arrays: dict[str, np.ndarray] = {}
        phase_raw_counts: dict[str, int] = {}
        for ph in CYCLE_PHASES_ORDER:
            pmask = phase_col == ph
            phase_raw_counts[ph] = int(pmask.sum())
            phase_arrays[ph] = s_all.loc[pmask].dropna().to_numpy(dtype=float)

        all_ge_min = all(len(phase_arrays[ph]) >= min_kw_days for ph in CYCLE_PHASES_ORDER)
        kw_h = np.nan
        kw_p = np.nan
        if all_ge_min:
            # Guard identical constant values across all groups
            concat_all = np.concatenate([phase_arrays[ph] for ph in CYCLE_PHASES_ORDER])
            if np.std(concat_all) > 1e-12:
                kw_res = sp_stats.kruskal(*(phase_arrays[ph] for ph in CYCLE_PHASES_ORDER))
                kw_h = round(float(kw_res.statistic), 4)
                kw_p = round(float(kw_res.pvalue), 6)
                kw_status = f"Unlocked_All_Phases_GE_{min_kw_days}"
            else:
                kw_status = "Constant_Outcome_No_Variance"
        else:
            kw_status = f"Gated_Phase_N_Under_{min_kw_days}"

        for ph in CYCLE_PHASES_ORDER:
            vals = phase_arrays[ph]
            r_n = phase_raw_counts[ph]
            v_n = len(vals)
            if v_n > 0:
                med_v = round(float(np.median(vals)), 4)
                q75, q25 = np.percentile(vals, [75, 25])
                iqr_v = round(float(q75 - q25), 4)
                mean_v = round(float(np.mean(vals)), 4)
                sd_v = round(float(np.std(vals, ddof=1)), 4) if v_n >= 2 else np.nan
                min_v = round(float(np.min(vals)), 4)
                max_v = round(float(np.max(vals)), 4)
                eff_n = compute_effective_n(
                    s_all.loc[phase_col == ph],
                    dates=df.loc[phase_col == ph, "date"].tolist(),
                )
            else:
                med_v = iqr_v = mean_v = sd_v = min_v = max_v = np.nan
                eff_n = 0

            _, mat_lbl, _ = classify_daily_maturity(v_n)
            out_rows.append(
                {
                    "summary_section": "Cycle_Phase_Biometrics",
                    "metric_name": m,
                    "cycle_phase_or_stat": ph,
                    "median": med_v,
                    "iqr": iqr_v,
                    "mean": mean_v,
                    "sd": sd_v,
                    "min_val": min_v,
                    "max_val": max_v,
                    "kruskal_gate_status": kw_status,
                    "kruskal_h_stat": kw_h,
                    "kruskal_p_value": kw_p,
                    "historical_hematology_thyroid_context": lab_ctx,
                    "raw_n": r_n,
                    "valid_n": v_n,
                    "effective_n": eff_n,
                    "missingness_pct": compute_missingness_pct(r_n, v_n),
                    "predictor_count": 1,
                    "stability_flag": "Descriptive_Only" if v_n >= min_kw_days else "Insufficient_N",
                    "maturity_level": mat_lbl,
                    "epistemic_label": "Descriptive",
                    "caveat_note": f"Cycle-phase distribution ({kw_status}); irregular cycle lengths use stitched Flo+Fitbit_HC anchors.",
                }
            )

    # 2. Multi-Year Cycle Regularity Summary (`cycle_length_days`, `span_days`)
    cyc_lengths: list[float] = []
    span_days_vals: list[float] = []
    raw_cyc_rows = 0

    if not silver_cycles.empty:
        raw_cyc_rows = len(silver_cycles)
        if "cycle_length_days" in silver_cycles.columns:
            cyc_lengths = (
                pd.to_numeric(silver_cycles["cycle_length_days"].replace("", np.nan), errors="coerce")
                .dropna()
                .to_numpy(dtype=float)
                .tolist()
            )
        if "span_days" in silver_cycles.columns:
            span_days_vals = (
                pd.to_numeric(silver_cycles["span_days"].replace("", np.nan), errors="coerce")
                .dropna()
                .to_numpy(dtype=float)
                .tolist()
            )

    # Supplement cycle lengths from stitched Silver_Cycle_Dim starts if Silver_Cycles had < 2 lengths
    if len(cyc_lengths) < 2 and not silver_cycle_dim.empty and "cycle_start" in silver_cycle_dim.columns:
        starts = sorted(set(silver_cycle_dim["cycle_start"].dropna().astype(str)) - {""})
        raw_cyc_rows = max(raw_cyc_rows, len(starts))
        for i in range(1, len(starts)):
            d_prev = pd.Timestamp(starts[i - 1])
            d_curr = pd.Timestamp(starts[i])
            diff_d = float((d_curr - d_prev).days)
            if diff_d >= 18.0:
                cyc_lengths.append(diff_d)

    for reg_metric, vals_list in (
        ("cycle_length_days", cyc_lengths),
        ("span_days", span_days_vals),
    ):
        arr = np.asarray(vals_list, dtype=float)
        v_n = len(arr)
        r_n = max(raw_cyc_rows, v_n)
        if v_n > 0:
            med_v = round(float(np.median(arr)), 3)
            q75, q25 = np.percentile(arr, [75, 25])
            iqr_v = round(float(q75 - q25), 3)
            mean_v = round(float(np.mean(arr)), 3)
            sd_v = round(float(np.std(arr, ddof=1)), 3) if v_n >= 2 else np.nan
            min_v = round(float(np.min(arr)), 3)
            max_v = round(float(np.max(arr)), 3)
        else:
            med_v = iqr_v = mean_v = sd_v = min_v = max_v = np.nan

        _, mat_lbl, _ = classify_daily_maturity(v_n)
        out_rows.append(
            {
                "summary_section": "Multi_Year_Cycle_Regularity",
                "metric_name": reg_metric,
                "cycle_phase_or_stat": "All_Recorded_Cycles",
                "median": med_v,
                "iqr": iqr_v,
                "mean": mean_v,
                "sd": sd_v,
                "min_val": min_v,
                "max_val": max_v,
                "kruskal_gate_status": "Not_Applicable_Regularity_Ledger",
                "kruskal_h_stat": np.nan,
                "kruskal_p_value": np.nan,
                "historical_hematology_thyroid_context": lab_ctx,
                "raw_n": r_n,
                "valid_n": v_n,
                "effective_n": compute_effective_n(arr) if v_n > 0 else 0,
                "missingness_pct": compute_missingness_pct(r_n, v_n),
                "predictor_count": 0,
                "stability_flag": "Descriptive_Only" if v_n >= 5 else "Insufficient_N",
                "maturity_level": mat_lbl,
                "epistemic_label": "Descriptive",
                "caveat_note": "Longitudinal menstrual regularity summarized alongside Ferritin, MCV, and TSH dates.",
            }
        )

    out_df = pd.DataFrame(out_rows, columns=cols)
    return enforce_maturity_audit(
        out_df,
        module_name="cycle_stats",
        analysis_id="CYCLE_PHASE_BIOMETRICS",
        audit_logger=audit_logger,
    )
