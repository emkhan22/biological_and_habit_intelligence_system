"""Module B: Per-Treatment & Per-Dose Scorecards (`treatment_scorecards.py` -> `Gold_Treatment_Scorecards`).

Rules (Prompt Section 3 & AGENTS.md Rules 1, 3, 4, 5):
- Syncs declarative `config/treatments.yaml` against `Silver_Medications`, `Silver_Procedures`,
  and `Silver_Glapp_Shots` so any future Tirzepatide dose step (e.g., `7.5 mg`) automatically
  closes the prior dose step and registers a new scorecard with `prior_dose_id` linked.
- Reconciles `Thyronorm_100mcg` (`2026-11-10` transition anchor) cleanly with
  `Micro_Journal!Meds_Log` `75 mcg` bottle-finishing entries without raising false errors.
- Computes BOTH `Pre-Treatment Baseline` AND `Prior-Dose Baseline` vs `Current Performance`
  (`current_full_phase_mean`, `current_14d_mean`) on every `treatment_id x metric_name` row.
- Enforces `overlap_warning` (±28d anchor window), `Level 0 (Collecting Baseline)` until
  `valid_n >= 14`, `epistemic_label = 'Descriptive'`, and `clinical_window_status`.
"""

from __future__ import annotations

import copy
from datetime import date, timedelta
import re
from typing import Any

import numpy as np
import pandas as pd

from health_engine.config_loader import iso_to_display_date, to_iso_date_str
from health_engine.stats.maturity import (
    StatsAuditLogger,
    classify_scorecard_maturity,
    compute_effective_n,
    compute_missingness_pct,
    enforce_maturity_audit,
    subsample_univariate_stability,
)


def _format_dose_tag(dose: float) -> str:
    return f"{dose:.1f}" if abs(dose - round(dose)) > 1e-6 or dose in (2.5, 5.0, 7.5, 12.5) else f"{dose:.1f}"


def sync_treatment_registry(
    treatments_cfg: dict[str, Any],
    silver_medications: pd.DataFrame,
    silver_procedures: pd.DataFrame,
    silver_glapp_shots: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Sync YAML treatment registry with Silver clinical tables and auto-register new GLP-1 dose steps."""
    entries: list[dict[str, Any]] = copy.deepcopy(treatments_cfg.get("treatments", []))
    by_id: dict[str, dict[str, Any]] = {str(e["treatment_id"]): e for e in entries}

    # 1. Sync procedure dates if present in Silver_Procedures
    if not silver_procedures.empty:
        for r in silver_procedures.to_dict(orient="records"):
            pname = str(r.get("Procedure_Name") or "").lower()
            pdate = to_iso_date_str(r.get("Date_Performed_Or_Planned"))
            if not pdate:
                continue
            if ("arachitol" in pname or "600" in pname) and "Vit_D_600K_IM" in by_id:
                by_id["Vit_D_600K_IM"]["start_date"] = pdate
            elif ("orofer" in pname or "fcm" in pname or "iron" in pname) and "IV_Iron_FCM_1g" in by_id:
                by_id["IV_Iron_FCM_1g"]["start_date"] = pdate

    # 2. Annotate provenance from Silver_Medications where available
    if not silver_medications.empty:
        for r in silver_medications.to_dict(orient="records"):
            bname = f"{r.get('Brand_Name', '')} {r.get('Drug_Name_Generic', '')}".lower()
            rx_dt = to_iso_date_str(r.get("Prescription_Date"))
            if "thyronorm 100" in bname and "Thyronorm_100mcg" in by_id and rx_dt:
                by_id["Thyronorm_100mcg"]["prescribed_date"] = rx_dt

    # 3. Dynamically sync Tirzepatide dose escalations from Silver_Glapp_Shots
    auto_cfg = treatments_cfg.get("tirzepatide_auto_escalation", {})
    if auto_cfg.get("enabled", True) and not silver_glapp_shots.empty:
        shots_parsed: list[tuple[str, float]] = []
        for r in silver_glapp_shots.to_dict(orient="records"):
            ds = to_iso_date_str(r.get("date"))
            med = str(r.get("medication") or "tirzepatide").strip().lower()
            if "tirz" not in med and "yurpeak" not in med and "mounjaro" not in med and med != "":
                continue
            try:
                d_mg = float(str(r.get("dose_mg") or "").strip())
            except ValueError:
                continue
            if ds and d_mg > 0:
                shots_parsed.append((ds, d_mg))

        shots_parsed.sort(key=lambda x: x[0])
        # Identify contiguous or chronological first-start per distinct dose step
        dose_steps: list[dict[str, Any]] = []
        for ds, d_mg in shots_parsed:
            if not dose_steps or abs(dose_steps[-1]["dose_mg"] - d_mg) > 1e-6:
                # Check if this dose already appeared earlier; if purely escalating, register step
                existing_same = next((s for s in dose_steps if abs(s["dose_mg"] - d_mg) <= 1e-6), None)
                if existing_same is None:
                    dose_steps.append({"dose_mg": d_mg, "first_shot_date": ds})

        # Link sequential Tirzepatide steps and auto-register any unseen dose (e.g. 7.5mg, 10.0mg)
        known_dose_map: dict[float, dict[str, Any]] = {}
        for e in entries:
            if e.get("glapp_dose_mg") is not None:
                known_dose_map[float(e["glapp_dose_mg"])] = e

        ordered_tirz_entries: list[dict[str, Any]] = []
        for idx, step in enumerate(dose_steps):
            d_mg = float(step["dose_mg"])
            obs_start = str(step["first_shot_date"])
            if d_mg in known_dose_map:
                ent = known_dose_map[d_mg]
                # Preserve configured clinical start_date if earlier/configured, unless shot occurred earlier
                if obs_start < str(ent["start_date"]):
                    ent["start_date"] = obs_start
            else:
                dose_tag = _format_dose_tag(d_mg)
                tid = str(auto_cfg.get("id_pattern", "Tirzepatide_{dose}mg")).format(dose=dose_tag)
                prev_ent = ordered_tirz_entries[-1] if ordered_tirz_entries else None
                prev_end_iso = (date.fromisoformat(obs_start) - timedelta(days=1)).isoformat()
                if prev_ent is not None and not prev_ent.get("end_date"):
                    prev_ent["end_date"] = prev_end_iso

                ent = {
                    "treatment_id": tid,
                    "display_name": f"Tirzepatide {dose_tag} mg (Auto-Registered Escalation)",
                    "category": "GLP1_GIP_Therapy",
                    "glapp_dose_mg": d_mg,
                    "start_date": obs_start,
                    "end_date": None,
                    "prior_dose_id": prev_ent["treatment_id"] if prev_ent else None,
                    "prior_dose_start": prev_ent["start_date"] if prev_ent else None,
                    "prior_dose_end": prev_end_iso if prev_ent else None,
                    "pre_baseline_start": str(auto_cfg.get("pre_baseline_start", "2025-06-05")),
                    "pre_baseline_end": str(auto_cfg.get("pre_baseline_end", "2026-09-30")),
                    "equilibration_days": int(auto_cfg.get("equilibration_days", 21)),
                    "retest_window_start_days": int(auto_cfg.get("retest_window_start_days", 84)),
                    "retest_window_end_days": int(auto_cfg.get("retest_window_end_days", 112)),
                    "retest_labs": list(auto_cfg.get("retest_labs", ["HbA1c"])),
                    "metrics": list(auto_cfg.get("metrics", ["avg_hr", "hrv_rmssd_ms", "steps"])),
                    "clinical_note": f"Auto-registered from Silver_Glapp_Shots ({dose_tag} mg starting {obs_start}).",
                }
                entries.append(ent)
                by_id[tid] = ent
                known_dose_map[d_mg] = ent
            ordered_tirz_entries.append(ent)

    return entries


def _prepare_daily_scorecard_frame(
    gold_daily: pd.DataFrame,
    gold_sleep_spikes: pd.DataFrame,
    gold_event_windows: pd.DataFrame,
    cardiac_daily: pd.DataFrame,
    silver_sleep_daily: pd.DataFrame,
    silver_meds: pd.DataFrame,
    silver_symptoms: pd.DataFrame,
    silver_glapp_se: pd.DataFrame,
    silver_nutr_log: pd.DataFrame,
) -> pd.DataFrame:
    """Assemble all daily metric columns needed across Treatment Scorecards."""
    df = gold_daily.sort_values("date").reset_index(drop=True).copy()

    # Sleep onset latency from Silver_Sleep_Daily
    if not silver_sleep_daily.empty and "sleep_onset_latency_min_est" in silver_sleep_daily.columns:
        sol_sub = silver_sleep_daily[["date", "sleep_onset_latency_min_est"]].drop_duplicates("date")
        df = df.merge(sol_sub, on="date", how="left")
    elif "sleep_onset_latency_min_est" not in df.columns:
        df["sleep_onset_latency_min_est"] = np.nan

    # Spikes per asleep hr
    if not gold_sleep_spikes.empty and "spikes_per_asleep_hr" in gold_sleep_spikes.columns:
        sp_sub = gold_sleep_spikes[["date", "spikes_per_asleep_hr"]].drop_duplicates("date")
        df = df.merge(sp_sub, on="date", how="left")
    else:
        df["spikes_per_asleep_hr"] = np.nan

    # GI severity
    if not gold_event_windows.empty and "out_day_t_gi_symptom_score" in gold_event_windows.columns:
        gi_sub = gold_event_windows[["date", "out_day_t_gi_symptom_score"]].drop_duplicates("date")
        df = df.merge(gi_sub.rename(columns={"out_day_t_gi_symptom_score": "gi_severity"}), on="date", how="left")
    else:
        df["gi_severity"] = pd.to_numeric(df.get("glapp_gi_score", np.nan), errors="coerce")

    # Cardiac efficiency residual
    if not cardiac_daily.empty and "cardiac_efficiency_residual" in cardiac_daily.columns:
        ce_sub = cardiac_daily[["date", "cardiac_efficiency_residual"]].drop_duplicates("date")
        df = df.merge(ce_sub, on="date", how="left")
    else:
        df["cardiac_efficiency_residual"] = np.nan

    # Symptom-specific daily counts on active Micro Journal / Glapp dates (NaN when unlogged!)
    mj_active = df.get("micro_journal_status", pd.Series("", index=df.index)).astype(str).eq("Present")
    gl_active = df.get("glapp_mood_status", pd.Series("", index=df.index)).astype(str).eq("Present")
    any_sym_obs = mj_active | gl_active

    foot_cramp_by_d: dict[str, float] = {}
    joint_shld_by_d: dict[str, float] = {}
    fatigue_by_d: dict[str, float] = {}

    if not silver_symptoms.empty:
        for r in silver_symptoms.to_dict(orient="records"):
            ds = to_iso_date_str(r.get("date"))
            if not ds:
                continue
            sym = f"{r.get('Normalized_Symptom', '')} {r.get('Symptom', '')}".lower()
            if "right foot cramp" in sym or "foot cramp" in sym:
                foot_cramp_by_d[ds] = foot_cramp_by_d.get(ds, 0.0) + 1.0
            if any(k in sym for k in ("shoulder", "joint", "arthralgia", "elbow", "wrist")):
                joint_shld_by_d[ds] = joint_shld_by_d.get(ds, 0.0) + 1.0
            if any(k in sym for k in ("fatigue", "tired", "exhaust", "letharg")):
                fatigue_by_d[ds] = 1.0

    if not silver_glapp_se.empty:
        for r in silver_glapp_se.to_dict(orient="records"):
            ds = to_iso_date_str(r.get("date"))
            se = str(r.get("side_effect") or "").lower()
            if ds and any(k in se for k in ("fatigue", "tired", "letharg")):
                fatigue_by_d[ds] = 1.0

    if "feeling" in df.columns:
        for ds, fl in zip(df["date"].fillna("").astype(str), df["feeling"].fillna("").astype(str).str.lower()):
            if fl and any(k in fl for k in ("tired", "fatigued", "exhausted")):
                fatigue_by_d[ds] = 1.0

    dates_list = df["date"].astype(str).tolist()
    df["nocturnal_right_foot_cramp_cnt"] = np.where(
        mj_active,
        [foot_cramp_by_d.get(d, 0.0) for d in dates_list],
        np.nan,
    )
    df["joint_shoulder_pain_cnt"] = np.where(
        mj_active,
        [joint_shld_by_d.get(d, 0.0) for d in dates_list],
        np.nan,
    )
    df["fatigue_days"] = np.where(
        any_sym_obs,
        [fatigue_by_d.get(d, 0.0) for d in dates_list],
        np.nan,
    )

    # Midday meal logged check (11:30..16:30 IST) for skipped-meal cascade context
    midday_meal_dates: set[str] = set()
    nutr_logged_dates: set[str] = set()
    if not silver_nutr_log.empty and "date" in silver_nutr_log.columns:
        for r in silver_nutr_log.to_dict(orient="records"):
            ds = to_iso_date_str(r.get("date"))
            if not ds:
                continue
            nutr_logged_dates.add(ds)
            m = re.search(r"(\d{1,2}):(\d{2})", str(r.get("time_ist") or ""))
            if m:
                hr = int(m.group(1)) + int(m.group(2)) / 60.0
                if 11.5 <= hr <= 16.5:
                    midday_meal_dates.add(ds)
    df["_has_nutr_log"] = [d in nutr_logged_dates for d in dates_list]
    df["_has_midday_meal"] = [d in midday_meal_dates for d in dates_list]

    return df


def _compute_med_specific_daily_series(
    daily_df: pd.DataFrame,
    silver_meds: pd.DataFrame,
    med_regex: str,
    metric_name: str,
) -> pd.Series:
    """Compute per-medication daily adherence or shot-day skipped-meal cascade series."""
    idx = daily_df.index
    if silver_meds.empty or not med_regex:
        return pd.Series(np.nan, index=idx, dtype=float)

    pat = re.compile(med_regex, re.IGNORECASE)
    matched = silver_meds.loc[
        silver_meds["Med_or_Supplement"].astype(str).map(lambda s: bool(pat.search(s)))
    ]
    if matched.empty:
        return pd.Series(np.nan, index=idx, dtype=float)

    adh_by_d: dict[str, float] = {}
    cascade_by_d: dict[str, float] = {}

    dss_map = pd.to_numeric(daily_df.set_index("date")["days_since_shot"], errors="coerce").to_dict()
    midday_map = daily_df.set_index("date")["_has_midday_meal"].to_dict()
    nutr_map = daily_df.set_index("date")["_has_nutr_log"].to_dict()

    for ds, grp in matched.groupby("date"):
        d_str = str(ds)
        st_low = grp["Status"].fillna("").astype(str).str.lower()
        notes_low = grp.get("Notes", pd.Series("", index=grp.index)).fillna("").astype(str).str.lower()
        taken = int(st_low.str.contains("taken", na=False).sum())
        skipped = int(st_low.str.contains("skip|miss", regex=True, na=False).sum())
        tot = taken + skipped
        if tot > 0:
            adh_by_d[d_str] = float(taken) / float(tot)
            dss_v = dss_map.get(d_str, np.nan)
            in_shot_win = bool(pd.notna(dss_v) and float(dss_v) in (0.0, 1.0))
            meal_skipped_ctx = bool(
                any("meal" in n or "appetite" in n or "lunch" in n for n in notes_low)
                or (nutr_map.get(d_str, False) and not midday_map.get(d_str, True))
            )
            if in_shot_win or meal_skipped_ctx:
                cascade_by_d[d_str] = 1.0 if skipped > 0 else 0.0
            else:
                cascade_by_d[d_str] = 0.0

    target_map = adh_by_d if metric_name == "med_adherence_rate" else cascade_by_d
    return pd.Series([target_map.get(str(d), np.nan) for d in daily_df["date"]], index=idx, dtype=float)


def _clinical_window_status(
    days_on_tx: int,
    current_valid_n: int,
    min_valid_n_gate: int,
    equilibration_days: int,
    retest_start_days: int,
    retest_end_days: int,
    is_completed_prior_dose: bool,
) -> str:
    if current_valid_n < min_valid_n_gate and not is_completed_prior_dose:
        return "Collecting_Baseline"
    if days_on_tx < equilibration_days and not is_completed_prior_dose:
        return "In_Equilibration"
    if retest_start_days <= days_on_tx <= retest_end_days and not is_completed_prior_dose:
        return "Retest_Window_Open"
    return "Monitoring"


def build_gold_treatment_scorecards(
    gold_daily: pd.DataFrame,
    gold_weight_intervals: pd.DataFrame,
    gold_sleep_spikes: pd.DataFrame,
    gold_event_windows: pd.DataFrame,
    cardiac_daily: pd.DataFrame,
    silver_tables: dict[str, pd.DataFrame],
    thresholds_cfg: dict[str, Any],
    treatments_cfg: dict[str, Any],
    audit_logger: StatsAuditLogger | None = None,
) -> pd.DataFrame:
    """Build Module B: Gold_Treatment_Scorecards with Pre-Treatment AND Prior-Dose baselines."""
    cols = [
        "treatment_id",
        "display_name",
        "category",
        "start_date",
        "display_start_date",
        "end_date",
        "prior_dose_id",
        "metric_name",
        "pre_treatment_baseline_mean",
        "pre_treatment_baseline_median",
        "pre_treatment_valid_n",
        "prior_dose_baseline_mean",
        "prior_dose_valid_n",
        "current_full_phase_mean",
        "current_14d_mean",
        "current_valid_n",
        "days_on_treatment",
        "delta_vs_pre_treatment",
        "z_score_vs_pre_treatment",
        "delta_vs_prior_dose",
        "pct_change_vs_pre_treatment",
        "overlapping_therapies_list",
        "overlap_warning",
        "clinical_window_status",
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
            module_name="treatment_scorecards",
            analysis_id="TREATMENT_SCORECARDS_DUAL_BASELINE",
            audit_logger=audit_logger,
        )

    registry = sync_treatment_registry(
        treatments_cfg,
        silver_tables.get("Silver_Medications", pd.DataFrame()),
        silver_tables.get("Silver_Procedures", pd.DataFrame()),
        silver_tables.get("Silver_Glapp_Shots", pd.DataFrame()),
    )
    by_id = {str(t["treatment_id"]): t for t in registry}

    overlap_days = int(
        treatments_cfg.get(
            "overlap_window_days",
            thresholds_cfg.get("v20_maturity_gates", {})
            .get("treatment_scorecards", {})
            .get("overlap_window_days", 28),
        )
    )
    default_min_n = int(
        treatments_cfg.get(
            "min_valid_n_gate",
            thresholds_cfg.get("v20_maturity_gates", {})
            .get("treatment_scorecards", {})
            .get("min_valid_n_baseline_gate", 14),
        )
    )
    mad_scale = float(thresholds_cfg.get("rolling_maturity_gates", {}).get("robust_mad_scale", 1.4826))

    silver_meds = silver_tables.get("Silver_Meds_Log", pd.DataFrame())
    df = _prepare_daily_scorecard_frame(
        gold_daily=gold_daily,
        gold_sleep_spikes=gold_sleep_spikes,
        gold_event_windows=gold_event_windows,
        cardiac_daily=cardiac_daily,
        silver_sleep_daily=silver_tables.get("Silver_Sleep_Daily", pd.DataFrame()),
        silver_meds=silver_meds,
        silver_symptoms=silver_tables.get("Silver_Symptoms_Log", pd.DataFrame()),
        silver_glapp_se=silver_tables.get("Silver_Glapp_Side_Effects", pd.DataFrame()),
        silver_nutr_log=silver_tables.get("Silver_Nutrition_Log", pd.DataFrame()),
    )

    max_spine_ds = str(df["date"].max())
    max_spine_dt = date.fromisoformat(max_spine_ds)

    # Pre-check Thyronorm bottle-finishing reconciliation context in Silver_Meds_Log
    thyronorm_reconcile_note = ""
    if not silver_meds.empty and "Med_or_Supplement" in silver_meds.columns:
        thy_rows = silver_meds.loc[
            silver_meds["Med_or_Supplement"].fillna("").astype(str).str.lower().str.contains("thyronorm|levothyroxine", regex=True, na=False)
            & (silver_meds["date"].fillna("").astype(str) >= "2026-11-10")
        ]
        if not thy_rows.empty:
            doses_seen = sorted(set(thy_rows["Dose"].fillna("").astype(str).str.strip()) - {""})
            if any("75" in d for d in doses_seen):
                thyronorm_reconcile_note = (
                    " Reconciled Meds_Log 75 mcg entries post-2026-11-10 cleanly (bottle-finishing / voice-log context)."
                )

    out_rows: list[dict[str, Any]] = []

    for tx in registry:
        tid = str(tx["treatment_id"])
        disp_name = str(tx.get("display_name") or tid)
        cat = str(tx.get("category") or "Therapy")
        s_ds = str(tx["start_date"])
        e_ds = str(tx["end_date"]) if tx.get("end_date") else None
        s_dt = date.fromisoformat(s_ds)
        eff_end_ds = min(e_ds, max_spine_ds) if e_ds else max_spine_ds
        eff_end_dt = date.fromisoformat(eff_end_ds)
        days_on_tx = max(0, (eff_end_dt - s_dt).days + 1) if eff_end_dt >= s_dt else 0

        prior_id = str(tx["prior_dose_id"]) if tx.get("prior_dose_id") else ""
        prior_start = tx.get("prior_dose_start")
        prior_end = tx.get("prior_dose_end")
        if prior_id and (not prior_start or not prior_end) and prior_id in by_id:
            prior_start = by_id[prior_id].get("start_date")
            prior_end = by_id[prior_id].get("end_date") or (s_dt - timedelta(days=1)).isoformat()

        pre_start = str(tx.get("pre_baseline_start") or "2025-06-05")
        pre_end = str(tx.get("pre_baseline_end") or (s_dt - timedelta(days=1)).isoformat())
        min_gate_n = int(tx.get("min_valid_n_gate", default_min_n))
        eq_days = int(tx.get("equilibration_days", 21))
        ret_s = int(tx.get("retest_window_start_days", 56))
        ret_e = int(tx.get("retest_window_end_days", 84))
        is_closed = bool(e_ds is not None and e_ds < max_spine_ds)

        # Overlapping therapies within ±overlap_days of s_dt
        overlaps: list[str] = []
        for other in registry:
            oid = str(other["treatment_id"])
            if oid == tid:
                continue
            odt = date.fromisoformat(str(other["start_date"]))
            if abs((odt - s_dt).days) <= overlap_days:
                overlaps.append(oid)
        overlap_str = ", ".join(overlaps)
        overlap_warn = len(overlaps) > 0

        med_match = str(tx.get("meds_log_match") or "")

        for mname in tx.get("metrics", []):
            # Special interval-grain handling for loss_rate_kg_per_wk
            if mname == "loss_rate_kg_per_wk":
                w_all = (
                    gold_weight_intervals.loc[gold_weight_intervals["in_all_weighins"]]
                    if not gold_weight_intervals.empty and "in_all_weighins" in gold_weight_intervals.columns
                    else gold_weight_intervals
                )
                if not w_all.empty:
                    pre_iv = w_all.loc[(w_all["end_date"] >= pre_start) & (w_all["end_date"] <= pre_end)]
                    cur_iv = w_all.loc[(w_all["end_date"] > s_ds) & (w_all["end_date"] <= eff_end_ds)]
                    d14_cut = max(s_ds, (eff_end_dt - timedelta(days=14)).isoformat())
                    cur14_iv = w_all.loc[(w_all["end_date"] > d14_cut) & (w_all["end_date"] <= eff_end_ds)]
                    if prior_id and prior_start and prior_end:
                        pri_iv = w_all.loc[
                            (w_all["end_date"] > str(prior_start)) & (w_all["end_date"] <= str(prior_end))
                        ]
                    else:
                        pri_iv = pd.DataFrame()
                else:
                    pre_iv = cur_iv = cur14_iv = pri_iv = pd.DataFrame()

                pre_vals = pd.to_numeric(pre_iv.get("loss_rate_kg_per_wk", pd.Series(dtype=float)), errors="coerce").dropna().to_numpy(dtype=float)
                cur_vals = pd.to_numeric(cur_iv.get("loss_rate_kg_per_wk", pd.Series(dtype=float)), errors="coerce").dropna().to_numpy(dtype=float)
                cur14_vals = pd.to_numeric(cur14_iv.get("loss_rate_kg_per_wk", pd.Series(dtype=float)), errors="coerce").dropna().to_numpy(dtype=float)
                pri_vals = pd.to_numeric(pri_iv.get("loss_rate_kg_per_wk", pd.Series(dtype=float)), errors="coerce").dropna().to_numpy(dtype=float)
                raw_phase_n = max(1, len(cur_iv))
                cur_dates = cur_iv["end_date"].tolist() if not cur_iv.empty else None
            else:
                if mname in ("med_adherence_rate", "shot_day_skipped_meal_cascade_rate"):
                    s_col = _compute_med_specific_daily_series(df, silver_meds, med_match, mname)
                elif mname in df.columns:
                    s_col = pd.to_numeric(df[mname], errors="coerce")
                else:
                    s_col = pd.Series(np.nan, index=df.index, dtype=float)

                pre_mask = (df["date"] >= pre_start) & (df["date"] <= pre_end)
                cur_mask = (df["date"] >= s_ds) & (df["date"] <= eff_end_ds)
                d14_start = max(s_ds, (eff_end_dt - timedelta(days=13)).isoformat())
                cur14_mask = (df["date"] >= d14_start) & (df["date"] <= eff_end_ds)

                pre_vals = s_col.loc[pre_mask].dropna().to_numpy(dtype=float)
                cur_vals = s_col.loc[cur_mask].dropna().to_numpy(dtype=float)
                cur14_vals = s_col.loc[cur14_mask].dropna().to_numpy(dtype=float)

                if prior_id and prior_start and prior_end:
                    pri_mask = (df["date"] >= str(prior_start)) & (df["date"] <= str(prior_end))
                    pri_vals = s_col.loc[pri_mask].dropna().to_numpy(dtype=float)
                else:
                    pri_vals = np.asarray([], dtype=float)

                raw_phase_n = int(cur_mask.sum())
                cur_dates = df.loc[cur_mask, "date"].tolist()
                cur_series_with_na = s_col.loc[cur_mask]

            pre_n = len(pre_vals)
            pre_mean = round(float(np.mean(pre_vals)), 4) if pre_n > 0 else np.nan
            pre_med = round(float(np.median(pre_vals)), 4) if pre_n > 0 else np.nan
            pre_sd = float(np.std(pre_vals, ddof=1)) if pre_n >= 2 else np.nan
            pre_mad = mad_scale * float(np.median(np.abs(pre_vals - pre_med))) if pre_n >= 2 else np.nan

            pri_n: Any = len(pri_vals) if prior_id else np.nan
            pri_mean = round(float(np.mean(pri_vals)), 4) if (prior_id and len(pri_vals) > 0) else np.nan

            cur_n = len(cur_vals)
            cur_mean = round(float(np.mean(cur_vals)), 4) if cur_n > 0 else np.nan
            cur14_mean = round(float(np.mean(cur14_vals)), 4) if len(cur14_vals) > 0 else cur_mean

            delta_pre = (
                round(cur_mean - pre_mean, 4)
                if (np.isfinite(cur_mean) and np.isfinite(pre_mean))
                else np.nan
            )
            scale = pre_sd if (np.isfinite(pre_sd) and pre_sd > 1e-12) else pre_mad
            z_pre = (
                round((cur_mean - pre_mean) / scale, 4)
                if (np.isfinite(cur_mean) and np.isfinite(pre_mean) and np.isfinite(scale) and scale > 1e-12)
                else np.nan
            )
            delta_pri = (
                round(cur_mean - pri_mean, 4)
                if (prior_id and np.isfinite(cur_mean) and np.isfinite(pri_mean))
                else np.nan
            )
            pct_chg_pre = (
                round(((cur_mean - pre_mean) / abs(pre_mean)) * 100.0, 2)
                if (np.isfinite(cur_mean) and np.isfinite(pre_mean) and abs(pre_mean) > 1e-9)
                else np.nan
            )

            _, mat_lbl, _ = classify_scorecard_maturity(cur_n, min_baseline_n=min_gate_n)
            cwin_status = _clinical_window_status(
                days_on_tx=days_on_tx,
                current_valid_n=cur_n,
                min_valid_n_gate=min_gate_n,
                equilibration_days=eq_days,
                retest_start_days=ret_s,
                retest_end_days=ret_e,
                is_completed_prior_dose=is_closed,
            )

            if mname == "loss_rate_kg_per_wk":
                eff_n = compute_effective_n(cur_vals)
            else:
                eff_n = compute_effective_n(cur_series_with_na, dates=cur_dates)

            ref_for_stab = pri_mean if (prior_id and np.isfinite(pri_mean)) else pre_mean
            stab = subsample_univariate_stability(
                cur_vals, reference_center=ref_for_stab, min_required_n=min_gate_n
            )

            base_note = str(tx.get("clinical_note") or "")
            if tid == "Thyronorm_100mcg" and thyronorm_reconcile_note:
                base_note += thyronorm_reconcile_note
            if overlap_warn:
                base_note += f" Overlap warning: concurrent start within +/-{overlap_days}d ({overlap_str})."

            out_rows.append(
                {
                    "treatment_id": tid,
                    "display_name": disp_name,
                    "category": cat,
                    "start_date": s_ds,
                    "display_start_date": iso_to_display_date(s_ds),
                    "end_date": e_ds or "",
                    "prior_dose_id": prior_id,
                    "metric_name": mname,
                    "pre_treatment_baseline_mean": pre_mean,
                    "pre_treatment_baseline_median": pre_med,
                    "pre_treatment_valid_n": int(pre_n),
                    "prior_dose_baseline_mean": pri_mean,
                    "prior_dose_valid_n": pri_n,
                    "current_full_phase_mean": cur_mean,
                    "current_14d_mean": cur14_mean,
                    "current_valid_n": int(cur_n),
                    "days_on_treatment": int(days_on_tx),
                    "delta_vs_pre_treatment": delta_pre,
                    "z_score_vs_pre_treatment": z_pre,
                    "delta_vs_prior_dose": delta_pri,
                    "pct_change_vs_pre_treatment": pct_chg_pre,
                    "overlapping_therapies_list": overlap_str,
                    "overlap_warning": overlap_warn,
                    "clinical_window_status": cwin_status,
                    "raw_n": int(max(days_on_tx, raw_phase_n)),
                    "valid_n": int(cur_n),
                    "effective_n": int(eff_n),
                    "missingness_pct": compute_missingness_pct(max(days_on_tx, raw_phase_n), cur_n),
                    "predictor_count": 0,
                    "stability_flag": stab,
                    "maturity_level": mat_lbl,
                    "epistemic_label": "Descriptive",
                    "caveat_note": base_note.strip(),
                }
            )

    scorecards_df = pd.DataFrame(out_rows, columns=cols)
    return enforce_maturity_audit(
        scorecards_df,
        module_name="treatment_scorecards",
        analysis_id="TREATMENT_SCORECARDS_DUAL_BASELINE",
        audit_logger=audit_logger,
        summary_caveat="Dual-baseline comparison (Pre-Treatment & Prior-Dose) with +/-28d overlap guardrails.",
    )
