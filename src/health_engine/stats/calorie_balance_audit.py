"""BMR/TDEE Calorie Deficit vs. KGs Lost Audit (`calorie_balance_audit.py` -> `Gold_Calorie_Balance_Audit`).

Compares actual weight lost (`delta_kg` / `loss_rate_kg_per_wk`) on fasted/morning weigh-ins
against expected energy deficit derived from Fitdays BIA BMR (Katch-McArdle on FFM with
Mifflin-St Jeor fallback) + Fitbit activity/TDEE estimates across:
- Fasted weigh-in intervals (`Gold_Weight_Intervals`)
- Multi-week rolling windows (`14d`, `28d`)
- Weekly snapshots (`7d`)

Classifies each window into:
- `Within_Expected_Band` (`|discrepancy_kcal_per_day| <= 250`)
- `Potential_Under_Logging` (logged deficit predicts >250 kcal/day more loss than actual on clean multi-week window)
- `Potential_Over_Logging` (actual loss exceeds predicted deficit by >250 kcal/day on clean multi-week window)
- `Unclear_Water_Cycle_Or_Log_Gap` (low nutrition coverage `< 0.70`, `< 7` days, non-fasted anchor, or short `< 14d` window with period/constipation noise)
"""

from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from health_engine.config_loader import iso_to_display_date
from health_engine.stats.maturity import (
    StatsAuditLogger,
    classify_interval_maturity,
    compute_missingness_pct,
    enforce_maturity_audit,
)

CALORIE_AUDIT_COLUMNS: list[str] = [
    "window_type",
    "window_id",
    "start_date",
    "display_start_date",
    "end_date",
    "display_end_date",
    "window_days",
    "is_fasted_anchor_pair",
    "bmr_method",
    "bmr_kcal_mean",
    "tdee_est_kcal_mean",
    "logged_kcal_mean",
    "nutrition_logged_days",
    "nutrition_coverage_pct",
    "expected_deficit_kcal_day",
    "expected_delta_kg",
    "actual_delta_kg",
    "actual_loss_rate_kg_wk",
    "expected_loss_rate_kg_wk",
    "discrepancy_kcal_per_day",
    "bia_tissue_energy_kcal_day",
    "audit_classification",
    "confounder_note",
    "raw_n",
    "valid_n",
    "effective_n",
    "missingness_pct",
    "predictor_count",
    "stability_flag",
    "maturity_level",
    "epistemic_label",
    "caveat_note",
    "epistemic_class",
]


def _estimate_bmr_kcal(
    weight_kg: float | None,
    ffm_kg: float | None,
    *,
    height_cm: float = 163.0,
    age_years: float = 32.1,
) -> tuple[float, str]:
    """Return `(bmr_kcal, method)` prioritizing Fitdays Katch-McArdle FFM over Mifflin-St Jeor."""
    if ffm_kg is not None and np.isfinite(ffm_kg) and ffm_kg > 25.0:
        return round(370.0 + 21.6 * float(ffm_kg), 1), "Fitdays_Katch_McArdle_FFM"
    if weight_kg is not None and np.isfinite(weight_kg) and weight_kg > 30.0:
        # Mifflin-St Jeor (female baseline)
        return round(10.0 * float(weight_kg) + 6.25 * height_cm - 5.0 * age_years - 161.0, 1), "Mifflin_St_Jeor_Fallback"
    return np.nan, "Unavailable"


def _classify_balance_window(
    *,
    window_days: int,
    nutrition_coverage_pct: float,
    is_fasted_pair: bool,
    period_or_constip: bool,
    discrepancy_kcal_day: float,
    band_kcal_day: float = 250.0,
) -> tuple[str, str]:
    """Classify discrepancy into one of the 4 audit states with explicit noise notes."""
    if not np.isfinite(discrepancy_kcal_day):
        return "Unclear_Water_Cycle_Or_Log_Gap", "Missing logged calories, TDEE, or start/end weight."
    if nutrition_coverage_pct < 0.70:
        return (
            "Unclear_Water_Cycle_Or_Log_Gap",
            f"Nutrition coverage {nutrition_coverage_pct*100:.0f}% < 70% threshold; unlogged days cannot be audited.",
        )
    if window_days < 7:
        return (
            "Unclear_Water_Cycle_Or_Log_Gap",
            f"Short {window_days}-day window is dominated by water/glycogen/GI transit fluctuations.",
        )
    if not is_fasted_pair:
        return (
            "Unclear_Water_Cycle_Or_Log_Gap",
            "Non-fasted weigh-in anchor introduces meal-weight/hydration scale noise.",
        )
    if window_days < 14 and period_or_constip:
        return (
            "Unclear_Water_Cycle_Or_Log_Gap",
            "Short (<14d) window coincided with menstrual water retention or 48h pre-weigh-in constipation.",
        )
    if abs(discrepancy_kcal_day) <= band_kcal_day:
        return (
            "Within_Expected_Band",
            f"Observed weight change aligns within +/-{band_kcal_day:.0f} kcal/day of BMR/TDEE energy model.",
        )
    if discrepancy_kcal_day > band_kcal_day:
        # expected_deficit - implied_actual_deficit > 250 -> logged deficit looks larger than actual loss
        return (
            "Potential_Under_Logging",
            f"Logged deficit predicts ~{discrepancy_kcal_day:.0f} kcal/day more weight loss than observed (possible under-logged bites/oils, TDEE overestimate, or fluid retention).",
        )
    return (
        "Potential_Over_Logging",
        f"Actual weight loss exceeds logged deficit by ~{abs(discrepancy_kcal_day):.0f} kcal/day (possible over-logged portions, higher NEAT/TDEE, or glycogen/water drop).",
    )


def build_gold_calorie_balance_audit(
    gold_daily: pd.DataFrame,
    gold_weight_intervals: pd.DataFrame | None,
    canonical_weighins: pd.DataFrame | None = None,
    silver_body_comp: pd.DataFrame | None = None,
    thresholds_cfg: dict[str, Any] | None = None,
    audit_logger: StatsAuditLogger | None = None,
) -> pd.DataFrame:
    """Build `Gold_Calorie_Balance_Audit` across weigh-in intervals and 14d/28d/7d rolling windows."""
    if gold_daily is None or gold_daily.empty or "date" not in gold_daily.columns:
        return pd.DataFrame(columns=CALORIE_AUDIT_COLUMNS)

    profile = (thresholds_cfg or {}).get("patient_profile", {})
    height_cm = float(profile.get("height_cm", 163.0))
    dob_str = str(profile.get("dob", "1994-09-22"))
    dob_d = date.fromisoformat(dob_str[:10])

    daily = gold_daily.sort_values("date").reset_index(drop=True).copy()
    daily["_kcal"] = pd.to_numeric(daily.get("calories_kcal", np.nan), errors="coerce")
    daily["_tdee_hc"] = pd.to_numeric(daily.get("activity_total_kcal", np.nan), errors="coerce")
    daily["_steps"] = pd.to_numeric(daily.get("steps", np.nan), errors="coerce")
    daily["_nutr_ok"] = (
        daily["nutrition_status"].astype(str) == "Present"
        if "nutrition_status" in daily.columns
        else daily["_kcal"].notna()
    )

    # Map latest Fitdays FFM & weight onto daily spine via forward-fill (strictly past/concurrent)
    ffm_map: dict[str, float] = {}
    if silver_body_comp is not None and not silver_body_comp.empty and "date" in silver_body_comp.columns:
        bc = silver_body_comp.copy()
        bc["_ffm"] = pd.to_numeric(bc.get("fat_free_body_weight_kg", np.nan), errors="coerce")
        for r in bc.dropna(subset=["_ffm"]).to_dict(orient="records"):
            ffm_map[str(r["date"])[:10]] = float(r["_ffm"])

    daily["_ffm"] = daily["date"].astype(str).str.slice(0, 10).map(ffm_map).ffill().bfill()
    daily["_wt_ref"] = pd.to_numeric(
        daily.get("latest_weight_kg", daily.get("weight_kg", np.nan)), errors="coerce"
    ).ffill().bfill()

    bmr_vals: list[float] = []
    bmr_methods: list[str] = []
    tdee_vals: list[float] = []
    for idx in range(len(daily)):
        d_iso = str(daily.loc[idx, "date"])[:10]
        age_y = (date.fromisoformat(d_iso) - dob_d).days / 365.25
        bmr_v, bmr_m = _estimate_bmr_kcal(
            float(daily.loc[idx, "_wt_ref"]) if pd.notna(daily.loc[idx, "_wt_ref"]) else None,
            float(daily.loc[idx, "_ffm"]) if pd.notna(daily.loc[idx, "_ffm"]) else None,
            height_cm=height_cm,
            age_years=age_y,
        )
        bmr_vals.append(bmr_v)
        bmr_methods.append(bmr_m)
        tdee_hc = float(daily.loc[idx, "_tdee_hc"]) if pd.notna(daily.loc[idx, "_tdee_hc"]) else np.nan
        steps_v = float(daily.loc[idx, "_steps"]) if pd.notna(daily.loc[idx, "_steps"]) else np.nan
        if np.isfinite(tdee_hc) and tdee_hc >= 1200.0:
            tdee_vals.append(tdee_hc)
        elif np.isfinite(bmr_v):
            step_burn = (steps_v * 0.045) if np.isfinite(steps_v) else 220.0
            tdee_vals.append(round(bmr_v * 1.15 + step_burn, 1))
        else:
            tdee_vals.append(np.nan)

    daily["_bmr"] = bmr_vals
    daily["_bmr_method"] = bmr_methods
    daily["_tdee_est"] = tdee_vals

    rows: list[dict[str, Any]] = []

    def _emit_row(
        *,
        w_type: str,
        w_id: str,
        s_iso: str,
        e_iso: str,
        actual_delta_kg: float,
        is_fasted: bool,
        period_or_constip: bool,
        fat_delta_kg: float = np.nan,
        ffm_delta_kg: float = np.nan,
    ) -> None:
        w_days = max(1, (date.fromisoformat(e_iso) - date.fromisoformat(s_iso)).days)
        sub = daily.loc[(daily["date"].astype(str) > s_iso) & (daily["date"].astype(str) <= e_iso)]
        if sub.empty:
            return
        n_logged = int(sub["_nutr_ok"].sum())
        cov = round(n_logged / w_days, 4)

        bmr_m_val = round(float(sub["_bmr"].mean()), 1) if sub["_bmr"].notna().any() else np.nan
        bmr_meth = str(sub["_bmr_method"].mode().iloc[0]) if not sub["_bmr_method"].empty else "Unavailable"
        tdee_m_val = round(float(sub["_tdee_est"].mean()), 1) if sub["_tdee_est"].notna().any() else np.nan
        kcal_m_val = round(float(sub.loc[sub["_nutr_ok"], "_kcal"].mean()), 1) if n_logged > 0 else np.nan

        exp_def = (
            round(tdee_m_val - kcal_m_val, 1)
            if np.isfinite(tdee_m_val) and np.isfinite(kcal_m_val)
            else np.nan
        )
        exp_delta_kg = (
            round(-(exp_def * w_days) / 7700.0, 3) if np.isfinite(exp_def) else np.nan
        )

        # Optional BIA tissue energy (9400 kcal/kg fat + 1800 kcal/kg FFM)
        bia_kcal_day = np.nan
        if np.isfinite(fat_delta_kg) and np.isfinite(ffm_delta_kg):
            bia_kcal_day = round(-((fat_delta_kg * 9400.0) + (ffm_delta_kg * 1800.0)) / w_days, 1)

        implied_def_actual = (
            bia_kcal_day
            if np.isfinite(bia_kcal_day)
            else (round((-actual_delta_kg * 7700.0) / w_days, 1) if np.isfinite(actual_delta_kg) else np.nan)
        )
        discrepancy = (
            round(exp_def - implied_def_actual, 1)
            if np.isfinite(exp_def) and np.isfinite(implied_def_actual)
            else np.nan
        )

        cls_tag, conf_note = _classify_balance_window(
            window_days=w_days,
            nutrition_coverage_pct=cov,
            is_fasted_pair=is_fasted,
            period_or_constip=period_or_constip,
            discrepancy_kcal_day=discrepancy,
        )
        actual_rate = round((-actual_delta_kg) / (w_days / 7.0), 3) if np.isfinite(actual_delta_kg) else np.nan
        exp_rate = round((-exp_delta_kg) / (w_days / 7.0), 3) if np.isfinite(exp_delta_kg) else np.nan

        rows.append(
            {
                "window_type": w_type,
                "window_id": w_id,
                "start_date": s_iso,
                "display_start_date": iso_to_display_date(s_iso),
                "end_date": e_iso,
                "display_end_date": iso_to_display_date(e_iso),
                "window_days": w_days,
                "is_fasted_anchor_pair": bool(is_fasted),
                "bmr_method": bmr_meth,
                "bmr_kcal_mean": bmr_m_val,
                "tdee_est_kcal_mean": tdee_m_val,
                "logged_kcal_mean": kcal_m_val,
                "nutrition_logged_days": n_logged,
                "nutrition_coverage_pct": cov,
                "expected_deficit_kcal_day": exp_def,
                "expected_delta_kg": exp_delta_kg,
                "actual_delta_kg": round(actual_delta_kg, 3) if np.isfinite(actual_delta_kg) else np.nan,
                "actual_loss_rate_kg_wk": actual_rate,
                "expected_loss_rate_kg_wk": exp_rate,
                "discrepancy_kcal_per_day": discrepancy,
                "bia_tissue_energy_kcal_day": bia_kcal_day,
                "audit_classification": cls_tag,
                "confounder_note": conf_note,
                "raw_n": w_days,
                "valid_n": n_logged,
                "effective_n": n_logged,
                "missingness_pct": compute_missingness_pct(w_days, n_logged),
                "predictor_count": 2,
                "stability_flag": "Descriptive_Only" if w_days >= 14 and cov >= 0.70 else "Insufficient_N",
                "maturity_level": "Level 1 (Early Signal)" if w_days >= 14 and cov >= 0.70 else "Level 0 (Descriptive Ledger)",
                "epistemic_label": "Descriptive",
                "caveat_note": "7700 kcal/kg energy-balance audit (or BIA tissue energy when paired); short-window scale noise reflects hydration, glycogen, cycle, and GI transit.",
            }
        )

    # 1. Interval Audit from Gold_Weight_Intervals
    if gold_weight_intervals is not None and not gold_weight_intervals.empty:
        for r in gold_weight_intervals.to_dict(orient="records"):
            s_iso = str(r.get("start_date", ""))[:10]
            e_iso = str(r.get("end_date", ""))[:10]
            if not s_iso or not e_iso:
                continue
            p_flag = str(r.get("period_at_weighin_flag", "")).lower() == "true"
            c_flag = str(r.get("constipation_48h_before_weighin_flag", "")).lower() == "true"
            _emit_row(
                w_type="WeighIn_Interval",
                w_id=str(r.get("interval_id", f"INT-{e_iso}")),
                s_iso=s_iso,
                e_iso=e_iso,
                actual_delta_kg=float(pd.to_numeric(r.get("delta_kg"), errors="coerce")),
                is_fasted=str(r.get("is_fasted_interval", "")).lower() == "true",
                period_or_constip=(p_flag or c_flag),
                fat_delta_kg=float(pd.to_numeric(r.get("fat_delta_kg"), errors="coerce")),
                ffm_delta_kg=float(pd.to_numeric(r.get("ffm_delta_kg"), errors="coerce")),
            )

    # 2. Multi-Week Rolling Windows (14d, 28d) & Weekly Snapshot (7d) anchored to fasted/canonical weigh-ins
    weighins = canonical_weighins.copy() if canonical_weighins is not None else pd.DataFrame()
    if weighins.empty and gold_weight_intervals is not None and not gold_weight_intervals.empty:
        # Reconstruct anchor points from Gold_Weight_Intervals if canonical_weighins not passed directly
        pts: list[dict[str, Any]] = []
        for r in gold_weight_intervals.to_dict(orient="records"):
            pts.append(
                {
                    "date": str(r.get("start_date", ""))[:10],
                    "weight_kg": pd.to_numeric(r.get("start_kg"), errors="coerce"),
                    "is_fasted_anchor": str(r.get("start_weighin_status", "")).startswith("Fasted"),
                }
            )
            pts.append(
                {
                    "date": str(r.get("end_date", ""))[:10],
                    "weight_kg": pd.to_numeric(r.get("end_kg"), errors="coerce"),
                    "is_fasted_anchor": str(r.get("end_weighin_status", "")).startswith("Fasted"),
                }
            )
        weighins = pd.DataFrame(pts).dropna(subset=["weight_kg"]).drop_duplicates(subset=["date"]).sort_values("date")

    if not weighins.empty and "date" in weighins.columns and "weight_kg" in weighins.columns:
        wi = weighins.sort_values("date").reset_index(drop=True)
        wi["_d"] = wi["date"].astype(str).str.slice(0, 10).map(date.fromisoformat)
        wi["_wt"] = pd.to_numeric(wi["weight_kg"], errors="coerce")
        wi["_fasted"] = (
            wi["is_fasted_anchor"].fillna(False).astype(bool)
            if "is_fasted_anchor" in wi.columns
            else True
        )
        wi = wi.dropna(subset=["_wt"]).reset_index(drop=True)

        if len(wi) >= 2:
            latest = wi.iloc[-1]
            for target_days, label in [(7, "Rolling_7d_Snapshot"), (14, "Rolling_14d_Audit"), (28, "Rolling_28d_Audit")]:
                # Pick anchor closest to target_days prior (preferring fasted anchors within +/-4 days)
                diffs = (latest["_d"] - wi["_d"]).map(lambda td: td.days)
                cand = wi.loc[(diffs >= max(5, target_days - 5)) & (diffs <= target_days + 7)].copy()
                if cand.empty:
                    continue
                cand["_err"] = (diffs.loc[cand.index] - target_days).abs()
                cand = cand.sort_values(["_fasted", "_err"], ascending=[False, True])
                start_pt = cand.iloc[0]
                s_iso = start_pt["_d"].isoformat()
                e_iso = latest["_d"].isoformat()
                _emit_row(
                    w_type=label,
                    w_id=f"AUD-{target_days}D-{e_iso.replace('-', '')}",
                    s_iso=s_iso,
                    e_iso=e_iso,
                    actual_delta_kg=float(latest["_wt"] - start_pt["_wt"]),
                    is_fasted=bool(latest["_fasted"] and start_pt["_fasted"]),
                    period_or_constip=False,
                )

    raw_out = pd.DataFrame(rows)
    clean_cnt = (
        int((raw_out["audit_classification"] != "Unclear_Water_Cycle_Or_Log_Gap").sum())
        if not raw_out.empty
        else 0
    )
    out_df = enforce_maturity_audit(
        raw_out,
        module_name="calorie_balance_audit",
        analysis_id="CALORIE_BALANCE_BMR_TDEE_AUDIT",
        audit_logger=audit_logger,
        default_epistemic_label="Descriptive",
        summary_cv_or_ci=f"total_windows={len(raw_out)}, auditable_clean_windows={clean_cnt}",
        summary_caveat="Compares Fitdays BMR/TDEE energy deficit vs fasted weight change across 14d/28d/interval windows.",
    )
    for col in CALORIE_AUDIT_COLUMNS:
        if col not in out_df.columns:
            out_df[col] = np.nan
    return out_df[CALORIE_AUDIT_COLUMNS]
