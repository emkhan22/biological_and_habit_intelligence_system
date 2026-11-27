"""7-Day Folded Tirzepatide Descriptive Wave (`tirzepatide_wave.py` -> `Gold_Tirz_Folded_Wave`).

Rules (Prompt Section 4.2 & AGENTS.md Rules 1, 3, 4):
- Fold active GLP-1 daily records by `shot_day_int` (`0..6`), producing:
  1. `Overall` (`All_Doses`)
  2. `Dose_Stratified` (`2.5mg`, `5.0mg`, ...)
  3. `Week_By_Week` (`Week_1`, `Week_2`, ...)
  across `calories_kcal`, `protein_g`, `hunger`, `food_noise`, `gi_severity`,
  `sleep_hr_avg`, `resting_hr`, `hrv_rmssd_ms`, `asleep_min`, `spikes_per_asleep_hr`,
  and `meds_skipped_cnt`.
- Strictly NO hardcoded phase labels (like "nadir" or "rebound").
- Gate dose + accumulation-adjusted regression
  (`outcome ~ shot_day + active_dose + cumulative_pk + week_num`) behind `>= 12` treatment weeks.
"""

from __future__ import annotations

import math
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

WAVE_METRICS: tuple[str, ...] = (
    "calories_kcal",
    "protein_g",
    "hunger",
    "food_noise",
    "gi_severity",
    "sleep_hr_avg",
    "resting_hr",
    "hrv_rmssd_ms",
    "asleep_min",
    "spikes_per_asleep_hr",
    "meds_skipped_cnt",
)


def _fit_folded_wave_regression(
    df_glp1: pd.DataFrame,
    metric: str,
) -> dict[int, tuple[float, float]]:
    """Fit OLS `metric ~ C(shot_day_int) + active_tirz_dose_mg + tirz_pk_est + treatment_week_num`.

    Returns `{shot_day_int: (coef_vs_day0, p_value)}`.
    """
    req = [metric, "shot_day_int", "active_tirz_dose_mg", "tirz_pk_est", "treatment_week_num"]
    sub = df_glp1[req].apply(pd.to_numeric, errors="coerce").dropna()
    if len(sub) < 20:
        return {}

    y = sub[metric].to_numpy(dtype=float)
    if np.std(y) <= 1e-12:
        return {}

    sdays = sub["shot_day_int"].astype(int).to_numpy()
    cols_list = [np.ones(len(sub), dtype=float)]
    day_col_map: dict[int, int] = {}
    for d in range(1, 7):
        indicator = (sdays == d).astype(float)
        if np.std(indicator) > 1e-10:
            day_col_map[d] = len(cols_list)
            cols_list.append(indicator)

    for extra_c in ("active_tirz_dose_mg", "tirz_pk_est", "treatment_week_num"):
        arr = sub[extra_c].to_numpy(dtype=float)
        if np.std(arr) > 1e-10:
            cols_list.append(arr)

    x_mat = np.column_stack(cols_list)
    n, k = x_mat.shape
    if n <= k + 2:
        return {}

    beta, _, _, _ = np.linalg.lstsq(x_mat, y, rcond=None)
    resid = y - x_mat @ beta
    dof = max(1, n - k)
    sigma2 = float(np.dot(resid, resid)) / float(dof)
    xtx_inv = np.linalg.pinv(x_mat.T @ x_mat)
    se = np.sqrt(np.maximum(1e-18, np.diag(xtx_inv) * sigma2))

    out: dict[int, tuple[float, float]] = {0: (0.0, np.nan)}
    for d, c_idx in day_col_map.items():
        b_val = float(beta[c_idx])
        t_val = b_val / float(se[c_idx]) if se[c_idx] > 1e-12 else 0.0
        p_val = float(2.0 * sp_stats.t.sf(abs(t_val), df=dof))
        out[d] = (round(b_val, 4), round(p_val, 6))
    return out


def build_gold_tirz_folded_wave(
    gold_daily: pd.DataFrame,
    gold_sleep_spikes: pd.DataFrame,
    gold_event_windows: pd.DataFrame,
    thresholds_cfg: dict[str, Any],
    audit_logger: StatsAuditLogger | None = None,
) -> pd.DataFrame:
    """Build 7-day folded Tirzepatide wave summaries across Overall, Dose, and Week strata."""
    cols = [
        "stratum_type",
        "stratum_value",
        "metric_name",
        "shot_day_int",
        "mean",
        "median",
        "iqr",
        "q25",
        "q75",
        "distinct_treatment_weeks",
        "adj_model_status",
        "adj_effect_vs_day0",
        "adj_p_value",
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
    if gold_daily.empty or "shot_day_int" not in gold_daily.columns:
        return enforce_maturity_audit(
            pd.DataFrame(columns=cols),
            module_name="tirzepatide_wave",
            analysis_id="TIRZ_7DAY_FOLDED_WAVE",
            audit_logger=audit_logger,
        )

    min_reg_wks = int(
        thresholds_cfg.get("v20_maturity_gates", {})
        .get("tirzepatide_wave", {})
        .get("adj_regression_min_weeks", 12)
    )

    df = gold_daily.sort_values("date").copy()
    if not gold_sleep_spikes.empty and "spikes_per_asleep_hr" in gold_sleep_spikes.columns:
        df = df.merge(
            gold_sleep_spikes[["date", "spikes_per_asleep_hr"]].drop_duplicates("date"),
            on="date",
            how="left",
        )
    else:
        df["spikes_per_asleep_hr"] = np.nan

    if not gold_event_windows.empty and "out_day_t_gi_symptom_score" in gold_event_windows.columns:
        df = df.merge(
            gold_event_windows[["date", "out_day_t_gi_symptom_score"]]
            .drop_duplicates("date")
            .rename(columns={"out_day_t_gi_symptom_score": "gi_severity"}),
            on="date",
            how="left",
        )
    else:
        df["gi_severity"] = pd.to_numeric(df.get("glapp_gi_score", np.nan), errors="coerce")

    sday_num = pd.to_numeric(df["shot_day_int"], errors="coerce")
    glp1_df = df.loc[sday_num.between(0, 6)].copy()
    if glp1_df.empty:
        return enforce_maturity_audit(
            pd.DataFrame(columns=cols),
            module_name="tirzepatide_wave",
            analysis_id="TIRZ_7DAY_FOLDED_WAVE",
            audit_logger=audit_logger,
        )

    glp1_df["shot_day_int"] = pd.to_numeric(glp1_df["shot_day_int"], errors="coerce").astype(int)
    glp1_df["treatment_week_num_int"] = (
        pd.to_numeric(glp1_df["treatment_week_num"], errors="coerce").fillna(1).astype(int)
    )
    glp1_df["dose_label"] = [
        f"{float(d):.1f}mg" if pd.notna(d) else "Unknown_Dose"
        for d in pd.to_numeric(glp1_df["active_tirz_dose_mg"], errors="coerce")
    ]

    total_distinct_wks = int(glp1_df["treatment_week_num_int"].nunique())
    reg_unlocked = total_distinct_wks >= min_reg_wks
    adj_status = (
        "Unlocked_Dose_PK_Adjusted"
        if reg_unlocked
        else f"Gated_Under_{min_reg_wks}_Treatment_Weeks"
    )

    reg_by_metric: dict[str, dict[int, tuple[float, float]]] = {}
    if reg_unlocked:
        for m in WAVE_METRICS:
            if m in glp1_df.columns:
                reg_by_metric[m] = _fit_folded_wave_regression(glp1_df, m)

    # Assemble strata groups: Overall, Dose_Stratified, Week_By_Week
    strata_defs: list[tuple[str, str, pd.DataFrame]] = [("Overall", "All_Doses", glp1_df)]
    for dlbl, dgrp in glp1_df.groupby("dose_label", sort=True):
        strata_defs.append(("Dose_Stratified", str(dlbl), dgrp))
    for wnum, wgrp in glp1_df.groupby("treatment_week_num_int", sort=True):
        strata_defs.append(("Week_By_Week", f"Week_{int(wnum)}", wgrp))

    out_rows: list[dict[str, Any]] = []

    for st_type, st_val, s_df in strata_defs:
        wks_in_stratum = int(s_df["treatment_week_num_int"].nunique())
        for m in WAVE_METRICS:
            m_series_all = (
                pd.to_numeric(s_df[m], errors="coerce")
                if m in s_df.columns
                else pd.Series(np.nan, index=s_df.index, dtype=float)
            )
            for d_int in range(7):
                day_mask = s_df["shot_day_int"] == d_int
                raw_n = int(day_mask.sum())
                if st_type == "Week_By_Week" and raw_n == 0:
                    continue
                vals = m_series_all.loc[day_mask].dropna().to_numpy(dtype=float)
                v_n = len(vals)
                if v_n > 0:
                    mean_v = round(float(np.mean(vals)), 4)
                    med_v = round(float(np.median(vals)), 4)
                    q75, q25 = np.percentile(vals, [75, 25])
                    q25_v = round(float(q25), 4)
                    q75_v = round(float(q75), 4)
                    iqr_v = round(float(q75 - q25), 4)
                    eff_n = compute_effective_n(vals)
                else:
                    mean_v = med_v = iqr_v = q25_v = q75_v = np.nan
                    eff_n = 0

                adj_eff = np.nan
                adj_p = np.nan
                if st_type == "Overall" and reg_unlocked and m in reg_by_metric and d_int in reg_by_metric[m]:
                    adj_eff, adj_p = reg_by_metric[m][d_int]

                _, mat_lbl, _ = classify_daily_maturity(v_n)
                out_rows.append(
                    {
                        "stratum_type": st_type,
                        "stratum_value": st_val,
                        "metric_name": m,
                        "shot_day_int": d_int,
                        "mean": mean_v,
                        "median": med_v,
                        "iqr": iqr_v,
                        "q25": q25_v,
                        "q75": q75_v,
                        "distinct_treatment_weeks": wks_in_stratum,
                        "adj_model_status": adj_status if st_type == "Overall" else "Descriptive_Stratum_Only",
                        "adj_effect_vs_day0": adj_eff,
                        "adj_p_value": adj_p,
                        "raw_n": raw_n,
                        "valid_n": v_n,
                        "effective_n": eff_n,
                        "missingness_pct": compute_missingness_pct(raw_n, v_n),
                        "predictor_count": 4 if (st_type == "Overall" and reg_unlocked) else 1,
                        "stability_flag": "Descriptive_Only" if v_n >= 4 else "Insufficient_N",
                        "maturity_level": mat_lbl,
                        "epistemic_label": "Descriptive",
                        "caveat_note": (
                            f"7-day post-injection descriptive profile ({st_type}={st_val}, weeks={total_distinct_wks}); "
                            f"regression status={adj_status}."
                        ),
                    }
                )

    out_df = pd.DataFrame(out_rows, columns=cols)
    return enforce_maturity_audit(
        out_df,
        module_name="tirzepatide_wave",
        analysis_id="TIRZ_7DAY_FOLDED_WAVE",
        audit_logger=audit_logger,
        summary_cv_or_ci=f"treatment_weeks={total_distinct_wks}, adj_status={adj_status}",
    )
