"""3-Tier Intervention Event Study (`event_study.py` -> `Gold_Intervention_Events`).

Rules (Prompt Section 4.3 & AGENTS.md Rules 1, 3, 4):
- Anchors from `config/thresholds.yaml`:
  `2026-10-02` (`Vit_D_600K_IM`), `2026-10-10` (`IV_Iron_FCM_1g`),
  `2026-10-21` (`Tirzepatide_2.5mg`), `2026-11-19` (`Tirzepatide_5.0mg`).
- Tier 1 (Descriptive Windows): 7d, 14d, and 28d pre vs. post windows on coverage-filtered
  vitals, sleep, steps, and cardiac efficiency. Flag `overlap_warning = True` whenever
  another anchor falls within ±28 days.
- Tier 2 (Confounder-Adjusted Phase Comparison): OLS + Huber regression comparing
  `Pre_Oct_Baseline` vs `Active_Post_Interv` adjusting for `steps`, `work_mode`,
  `cycle_phase`, and `hr_hours_covered`, outputting adjusted delta, 95% CI, and
  `epistemic_label = 'Causal_Hypothesis'`.
- Tier 3 (Segmented Interrupted Time Series): Run level + slope segmented regression ONLY if
  `pre_valid_n >= 28` AND `post_valid_n >= 28` AND `overlap_warning == False`; otherwise
  output `its_status = 'Gated_Overlap_Or_Insufficient_Window'`.
"""

from __future__ import annotations

from datetime import date, timedelta
import math
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.linear_model import HuberRegressor

from health_engine.config_loader import iso_to_display_date
from health_engine.stats.maturity import (
    StatsAuditLogger,
    classify_daily_maturity,
    compute_effective_n,
    compute_missingness_pct,
    enforce_maturity_audit,
)

EVENT_STUDY_METRICS: tuple[str, ...] = (
    "avg_hr",
    "daytime_avg_hr",
    "resting_hr",
    "p95_hr",
    "hrv_rmssd_ms",
    "sleep_hr_avg",
    "asleep_min",
    "deep_min",
    "rem_min",
    "steps",
    "cardiac_efficiency_residual",
)


def _fit_tier2_adjusted_phase_model(
    df: pd.DataFrame,
    metric: str,
    pre_start: str,
    pre_end: str,
    post_start: str,
    post_end: str | None,
) -> dict[str, Any]:
    """Fit OLS + Huber regression for `metric ~ post_flag + steps + work_mode + cycle_phase + hr_hours_covered`."""
    pre_mask = (df["date"] >= pre_start) & (df["date"] <= pre_end)
    post_mask = (df["date"] >= post_start) & (
        (df["date"] <= post_end) if post_end else pd.Series(True, index=df.index)
    )
    sub = df.loc[pre_mask | post_mask].copy()
    sub["_post"] = (sub["date"] >= post_start).astype(float)
    sub["_y"] = pd.to_numeric(sub[metric], errors="coerce")
    sub = sub.loc[sub["_y"].notna()].copy()

    n_pre = int((sub["_post"] == 0.0).sum())
    n_post = int((sub["_post"] == 1.0).sum())
    n_tot = len(sub)
    if n_pre < 3 or n_post < 3 or n_tot < 10:
        return {
            "ols_delta": np.nan,
            "huber_delta": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "p_value": np.nan,
            "n_model": n_tot,
            "pred_cnt": 1,
            "stability": "Insufficient_N",
        }

    design_cols: list[np.ndarray] = [
        np.ones(n_tot, dtype=float),
        sub["_post"].to_numpy(dtype=float),
    ]

    # Numeric covariates (only include if non-missing across >=80% of valid rows, imputing median of pre/overall only for covariate control when needed — Wait: AGENTS.md Rule 4 no silent zero-filling: filter complete cases on available covariates!)
    num_cov_candidates: list[str] = []
    if metric != "steps" and "steps" in sub.columns:
        num_cov_candidates.append("steps")
    if metric in ("avg_hr", "daytime_avg_hr", "resting_hr", "p95_hr", "hrv_rmssd_ms", "cardiac_efficiency_residual"):
        if "hr_hours_covered" in sub.columns:
            num_cov_candidates.append("hr_hours_covered")

    for cc in num_cov_candidates:
        s_num = pd.to_numeric(sub[cc], errors="coerce")
        if int(s_num.notna().sum()) == n_tot and float(s_num.std()) > 1e-9:
            # Standardize covariate for numerical stability in HuberRegressor
            design_cols.append(((s_num - s_num.mean()) / s_num.std()).to_numpy(dtype=float))

    # Work mode indicators (drop Weekend reference)
    if "work_mode" in sub.columns:
        for wm in ("Office_Mon_Wed", "WFH_Tue_Thu_Fri"):
            ind = sub["work_mode"].astype(str).eq(wm).astype(float).to_numpy()
            if np.std(ind) > 1e-9:
                design_cols.append(ind)

    # Cycle phase indicators (drop Follicular_D1_14 / Unknown reference)
    if "cycle_phase" in sub.columns:
        for cp in ("Menstrual", "Mid_Extended", "Late_Luteal_PreMenst"):
            ind = sub["cycle_phase"].astype(str).eq(cp).astype(float).to_numpy()
            if np.std(ind) > 1e-9:
                design_cols.append(ind)

    x_mat = np.column_stack(design_cols)
    y_vec = sub["_y"].to_numpy(dtype=float)

    # Guard degrees of freedom; trim covariates if sample is tiny
    while x_mat.shape[1] >= n_tot - 2 and x_mat.shape[1] > 2:
        x_mat = x_mat[:, :-1]

    k = x_mat.shape[1]
    dof = max(1, n_tot - k)
    beta, _, _, _ = np.linalg.lstsq(x_mat, y_vec, rcond=None)
    resid = y_vec - x_mat @ beta
    sigma2 = float(np.dot(resid, resid)) / float(dof)
    xtx_inv = np.linalg.pinv(x_mat.T @ x_mat)
    se_post = math.sqrt(max(1e-18, float(xtx_inv[1, 1]) * sigma2))

    ols_delta = float(beta[1])
    t_crit = float(sp_stats.t.ppf(0.975, df=dof))
    ci_lo = ols_delta - t_crit * se_post
    ci_hi = ols_delta + t_crit * se_post
    t_stat = ols_delta / se_post if se_post > 1e-12 else 0.0
    pval = float(2.0 * sp_stats.t.sf(abs(t_stat), df=dof))

    # Robust HuberRegressor comparison
    huber_delta = np.nan
    if np.std(y_vec) > 1e-12:
        try:
            hub = HuberRegressor(max_iter=300, fit_intercept=True)
            hub.fit(x_mat[:, 1:], y_vec)
            huber_delta = float(hub.coef_[0])
        except Exception:
            huber_delta = np.nan

    same_sign = (
        np.isfinite(huber_delta)
        and (np.sign(ols_delta) == np.sign(huber_delta) or abs(ols_delta) < 1e-9)
    )
    if n_tot < 15:
        stab = "Insufficient_N"
    elif same_sign and pval < 0.10:
        stab = "Stable"
    elif same_sign:
        stab = "Moderate"
    else:
        stab = "Unstable"

    return {
        "ols_delta": round(ols_delta, 4),
        "huber_delta": round(huber_delta, 4) if np.isfinite(huber_delta) else np.nan,
        "ci_low": round(ci_lo, 4),
        "ci_high": round(ci_hi, 4),
        "p_value": round(pval, 6),
        "n_model": n_tot,
        "pred_cnt": k - 1,
        "stability": stab,
    }


def _fit_tier3_segmented_its(
    df: pd.DataFrame,
    metric: str,
    anchor_ds: str,
    post_end: str | None,
    window_days: int = 56,
) -> dict[str, float]:
    """Fit Segmented Interrupted Time Series: `Y_t = b0 + b1*t + b2*I(t>=0) + b3*t*I(t>=0)`."""
    adt = date.fromisoformat(anchor_ds)
    w_start = (adt - timedelta(days=window_days)).isoformat()
    w_end_cand = (adt + timedelta(days=window_days - 1)).isoformat()
    w_end = min(w_end_cand, post_end) if post_end else w_end_cand

    sub = df.loc[(df["date"] >= w_start) & (df["date"] <= w_end)].copy()
    sub["_y"] = pd.to_numeric(sub[metric], errors="coerce")
    sub = sub.loc[sub["_y"].notna()].copy()
    if len(sub) < 56:
        return {"level_change": np.nan, "slope_change_wk": np.nan, "its_p_value": np.nan}

    t_rel = np.asarray([(date.fromisoformat(str(d)) - adt).days for d in sub["date"]], dtype=float)
    post_ind = (t_rel >= 0.0).astype(float)
    t_after = t_rel * post_ind
    y = sub["_y"].to_numpy(dtype=float)

    x_mat = np.column_stack([np.ones(len(sub), dtype=float), t_rel, post_ind, t_after])
    beta, _, _, _ = np.linalg.lstsq(x_mat, y, rcond=None)
    resid = y - x_mat @ beta
    dof = max(1, len(sub) - 4)
    sigma2 = float(np.dot(resid, resid)) / float(dof)
    xtx_inv = np.linalg.pinv(x_mat.T @ x_mat)
    se_lvl = math.sqrt(max(1e-18, float(xtx_inv[2, 2]) * sigma2))
    t_lvl = float(beta[2]) / se_lvl if se_lvl > 1e-12 else 0.0
    p_lvl = float(2.0 * sp_stats.t.sf(abs(t_lvl), df=dof))

    return {
        "level_change": round(float(beta[2]), 4),
        "slope_change_wk": round(float(beta[3]) * 7.0, 4),
        "its_p_value": round(p_lvl, 6),
    }


def build_gold_intervention_events(
    gold_daily: pd.DataFrame,
    cardiac_daily: pd.DataFrame,
    thresholds_cfg: dict[str, Any],
    audit_logger: StatsAuditLogger | None = None,
) -> pd.DataFrame:
    """Build 3-Tier Intervention Event Study (`Gold_Intervention_Events`)."""
    cols = [
        "anchor_id",
        "anchor_date",
        "display_anchor_date",
        "anchor_label",
        "metric_name",
        "overlapping_anchors",
        "overlap_warning",
        "tier1_pre7d_mean",
        "tier1_post7d_mean",
        "tier1_delta7d",
        "tier1_pre14d_mean",
        "tier1_post14d_mean",
        "tier1_delta14d",
        "tier1_pre28d_mean",
        "tier1_post28d_mean",
        "tier1_delta28d",
        "tier1_pre28d_valid_n",
        "tier1_post28d_valid_n",
        "tier2_adj_delta_ols",
        "tier2_adj_delta_huber",
        "tier2_ci_95_low",
        "tier2_ci_95_high",
        "tier2_p_value",
        "its_status",
        "tier3_its_level_change",
        "tier3_its_slope_change_per_wk",
        "tier3_its_p_value",
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
            module_name="event_study",
            analysis_id="INTERVENTION_3TIER_EVENT_STUDY",
            audit_logger=audit_logger,
            default_epistemic_label="Causal_Hypothesis",
        )

    es_cfg = thresholds_cfg.get("v20_maturity_gates", {}).get("event_study", {})
    overlap_days = int(es_cfg.get("overlap_window_days", 28))
    its_min_pre = int(es_cfg.get("its_min_pre_valid_n", 28))
    its_min_post = int(es_cfg.get("its_min_post_valid_n", 28))
    dgates = thresholds_cfg.get("v20_maturity_gates", {}).get("daily_series", {})
    lvl1_min = int(dgates.get("level1_exploratory_min_n", 15))
    lvl2_min = int(dgates.get("level2_modeling_min_n", 30))

    pre_oct_start = str(thresholds_cfg["phase_anchors"]["pre_oct_baseline"]["start_date"])
    pre_oct_end = str(thresholds_cfg["phase_anchors"]["pre_oct_baseline"]["end_date"])

    anchors = list(thresholds_cfg.get("intervention_anchors", []))
    df = gold_daily.sort_values("date").reset_index(drop=True).copy()
    if not cardiac_daily.empty and "cardiac_efficiency_residual" in cardiac_daily.columns:
        df = df.merge(
            cardiac_daily[["date", "cardiac_efficiency_residual"]].drop_duplicates("date"),
            on="date",
            how="left",
        )
    else:
        df["cardiac_efficiency_residual"] = np.nan

    out_rows: list[dict[str, Any]] = []

    for anc in anchors:
        aid = str(anc["anchor_id"])
        ads = str(anc["anchor_date"])
        albl = str(anc.get("label") or aid)
        aend = str(anc["end_date"]) if anc.get("end_date") else None
        adt = date.fromisoformat(ads)

        overlaps: list[str] = []
        for other in anchors:
            oid = str(other["anchor_id"])
            if oid == aid:
                continue
            odt = date.fromisoformat(str(other["anchor_date"]))
            if abs((odt - adt).days) <= overlap_days:
                overlaps.append(oid)
        overlap_str = ", ".join(overlaps)
        overlap_warn = len(overlaps) > 0

        for m in EVENT_STUDY_METRICS:
            s = (
                pd.to_numeric(df[m], errors="coerce")
                if m in df.columns
                else pd.Series(np.nan, index=df.index, dtype=float)
            )

            # Tier 1: 7d, 14d, 28d pre vs post windows
            win_stats: dict[int, tuple[float, float, float, int, int]] = {}
            for w in (7, 14, 28):
                pre_lo = (adt - timedelta(days=w)).isoformat()
                pre_hi = (adt - timedelta(days=1)).isoformat()
                post_lo = ads
                post_hi_cand = (adt + timedelta(days=w - 1)).isoformat()
                post_hi = min(post_hi_cand, aend) if aend else post_hi_cand

                v_pre = s.loc[(df["date"] >= pre_lo) & (df["date"] <= pre_hi)].dropna()
                v_post = s.loc[(df["date"] >= post_lo) & (df["date"] <= post_hi)].dropna()
                m_pre = round(float(v_pre.mean()), 4) if not v_pre.empty else np.nan
                m_post = round(float(v_post.mean()), 4) if not v_post.empty else np.nan
                delt = round(m_post - m_pre, 4) if (np.isfinite(m_pre) and np.isfinite(m_post)) else np.nan
                win_stats[w] = (m_pre, m_post, delt, len(v_pre), len(v_post))

            pre28_n = win_stats[28][3]
            post28_n = win_stats[28][4]

            # Check wider 56d pre/post counts for Tier 3 ITS eligibility
            pre56_n = int(
                s.loc[(df["date"] >= (adt - timedelta(days=56)).isoformat()) & (df["date"] < ads)].notna().sum()
            )
            post56_end = min((adt + timedelta(days=55)).isoformat(), aend) if aend else (adt + timedelta(days=55)).isoformat()
            post56_n = int(
                s.loc[(df["date"] >= ads) & (df["date"] <= post56_end)].notna().sum()
            )

            # Tier 2: Confounder-adjusted OLS + Huber vs Pre_Oct_Baseline
            t2 = _fit_tier2_adjusted_phase_model(
                df=df,
                metric=m,
                pre_start=pre_oct_start,
                pre_end=pre_oct_end,
                post_start=ads,
                post_end=aend,
            )

            valid_n = int(t2["n_model"])
            lvl_num, mat_lbl, _ = classify_daily_maturity(valid_n, lvl1_min, lvl2_min)

            # Tier 3: Segmented Interrupted Time Series (STRICT GATE!)
            if (pre56_n >= its_min_pre) and (post56_n >= its_min_post) and (not overlap_warn):
                its_res = _fit_tier3_segmented_its(df, m, ads, aend, window_days=56)
                its_status = "Unlocked_Segmented_ITS"
                its_lvl = its_res["level_change"]
                its_slp = its_res["slope_change_wk"]
                its_p = its_res["its_p_value"]
            else:
                its_status = "Gated_Overlap_Or_Insufficient_Window"
                its_lvl = np.nan
                its_slp = np.nan
                its_p = np.nan

            raw_scope_mask = ((df["date"] >= pre_oct_start) & (df["date"] <= pre_oct_end)) | (
                (df["date"] >= ads) & ((df["date"] <= aend) if aend else True)
            )
            raw_n = int(raw_scope_mask.sum())
            eff_n = compute_effective_n(
                s.loc[raw_scope_mask],
                dates=df.loc[raw_scope_mask, "date"].tolist(),
            )

            caveat = (
                f"Causal hypothesis only: concurrent interventions within +/-{overlap_days}d ({overlap_str}) confound single-therapy attribution; Tier 3 ITS gated."
                if overlap_warn
                else f"Confounder-adjusted pre/post comparison for {aid} (ITS status={its_status})."
            )

            out_rows.append(
                {
                    "anchor_id": aid,
                    "anchor_date": ads,
                    "display_anchor_date": iso_to_display_date(ads),
                    "anchor_label": albl,
                    "metric_name": m,
                    "overlapping_anchors": overlap_str,
                    "overlap_warning": overlap_warn,
                    "tier1_pre7d_mean": win_stats[7][0],
                    "tier1_post7d_mean": win_stats[7][1],
                    "tier1_delta7d": win_stats[7][2],
                    "tier1_pre14d_mean": win_stats[14][0],
                    "tier1_post14d_mean": win_stats[14][1],
                    "tier1_delta14d": win_stats[14][2],
                    "tier1_pre28d_mean": win_stats[28][0],
                    "tier1_post28d_mean": win_stats[28][1],
                    "tier1_delta28d": win_stats[28][2],
                    "tier1_pre28d_valid_n": pre28_n,
                    "tier1_post28d_valid_n": post28_n,
                    "tier2_adj_delta_ols": t2["ols_delta"] if lvl_num >= 1 else np.nan,
                    "tier2_adj_delta_huber": t2["huber_delta"] if lvl_num >= 1 else np.nan,
                    "tier2_ci_95_low": t2["ci_low"] if lvl_num >= 1 else np.nan,
                    "tier2_ci_95_high": t2["ci_high"] if lvl_num >= 1 else np.nan,
                    "tier2_p_value": t2["p_value"] if lvl_num >= 1 else np.nan,
                    "its_status": its_status,
                    "tier3_its_level_change": its_lvl,
                    "tier3_its_slope_change_per_wk": its_slp,
                    "tier3_its_p_value": its_p,
                    "raw_n": raw_n,
                    "valid_n": valid_n,
                    "effective_n": eff_n,
                    "missingness_pct": compute_missingness_pct(raw_n, valid_n),
                    "predictor_count": int(t2["pred_cnt"]),
                    "stability_flag": str(t2["stability"]),
                    "maturity_level": mat_lbl,
                    "epistemic_label": "Causal_Hypothesis",
                    "caveat_note": caveat,
                }
            )

    out_df = pd.DataFrame(out_rows, columns=cols)
    return enforce_maturity_audit(
        out_df,
        module_name="event_study",
        analysis_id="INTERVENTION_3TIER_EVENT_STUDY",
        audit_logger=audit_logger,
        suppressed_at_level0=(
            "tier2_adj_delta_ols",
            "tier2_adj_delta_huber",
            "tier2_ci_95_low",
            "tier2_ci_95_high",
            "tier2_p_value",
            "tier3_its_level_change",
            "tier3_its_slope_change_per_wk",
            "tier3_its_p_value",
        ),
        default_epistemic_label="Causal_Hypothesis",
        summary_caveat="3-Tier intervention event study; Tier 3 ITS strictly gated on overlapping Oct-Nov 2026 anchors.",
    )
