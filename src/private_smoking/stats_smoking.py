"""Isolated V2.0 Statistical Summaries for `private_smoking` (`Smoking_Private_Store` ONLY).

Security & Privacy Guarantees (AGENTS.md Rule 2 & Prompt Section 4.7):
- Completely isolated from `health_engine` (zero `health_engine` imports).
- Strictly separates pre-decision **onset context** (`onset_hour_ist`, `trigger_onset`,
  `mood_onset`, `shot_day_int`, `prior_night_sleep_tier`, `craving_intensity_onset`)
  from **post-event outcomes** (`event_type`, `count_smoked_post`, `delay_minutes_post`,
  `is_resisted_outcome`).
- Builds:
  1. `Private_Smoking_Descriptive`
  2. `Private_Weight_Interval_Smoking`
  3. `Private_Treatment_Smoking_Scorecard`
- Never logs raw rows, triggers, timestamps, or counts to stdout/CI logs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _sleep_tier_from_asleep_min(val: float) -> str:
    if not np.isfinite(val):
        return "Unmeasured"
    if val < 360.0:  # < 6h
        return "<6h"
    if val >= 420.0:  # >= 7h
        return ">=7h"
    return "6h_to_7h"


def _classify_private_maturity(valid_n: int, min_l1: int = 14, min_l2: int = 30) -> str:
    if valid_n < min_l1:
        return "Level 0 (Collecting Baseline)"
    if valid_n < min_l2:
        return "Level 1 (Exploratory)"
    return "Level 2 (Statistical Modeling)"


def _classify_private_interval_maturity(valid_intervals: int) -> str:
    if valid_intervals < 12:
        return "Level 0 (Descriptive Ledger)"
    if valid_intervals < 24:
        return "Level 1 (Early Signal)"
    return "Level 2 (Statistical Model)"


def build_private_smoking_stats(
    silver_smoking_tables: dict[str, pd.DataFrame],
    daily_context_df: pd.DataFrame | None = None,
    weight_intervals_df: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """Compute isolated V2.0 private smoking tables from Silver smoking outputs."""
    onset_df = silver_smoking_tables.get("Silver_Smoking_Onset_Context", pd.DataFrame()).copy()
    post_df = silver_smoking_tables.get("Silver_Smoking_Post_Outcomes", pd.DataFrame()).copy()
    miss_df = silver_smoking_tables.get("Silver_Smoking_Daily_Missingness", pd.DataFrame()).copy()

    desc_cols = [
        # Strictly pre-decision ONSET context columns
        "onset_dimension",
        "onset_stratum",
        "onset_event_cnt",
        "onset_mean_craving_intensity",
        # Strictly POST-EVENT outcome columns
        "post_smoked_event_cnt",
        "post_resisted_event_cnt",
        "post_total_count",
        "post_mean_delay_minutes",
        # Audit & schema partition columns
        "schema_partition_guard",
        "raw_n",
        "valid_n",
        "maturity_level",
        "epistemic_label",
        "epistemic_class",
        "caveat_note",
    ]

    interval_cols = [
        "interval_id",
        "start_date",
        "end_date",
        "interval_days",
        "logged_days_in_interval",
        "logging_coverage_pct",
        "onset_craving_events",
        "post_total_count",
        "post_resisted_events",
        "post_mean_delay_minutes",
        "loss_rate_kg_per_wk_ref",
        "raw_n",
        "valid_n",
        "maturity_level",
        "epistemic_label",
        "epistemic_class",
        "caveat_note",
    ]

    scorecard_cols = [
        "treatment_step_id",
        "phase_start_date",
        "phase_end_date",
        "logged_days_n",
        "onset_events_per_logged_day",
        "onset_mean_craving_intensity",
        "post_count_per_logged_day",
        "post_resist_rate",
        "post_mean_delay_minutes",
        "delta_count_vs_prior_step",
        "raw_n",
        "valid_n",
        "maturity_level",
        "epistemic_label",
        "epistemic_class",
        "caveat_note",
    ]

    # Map optional daily non-sensitive context (date -> shot_day_int, active_tirz_dose_mg, prior-night asleep_min)
    shot_day_by_date: dict[str, str] = {}
    dose_by_date: dict[str, float] = {}
    sleep_tier_by_date: dict[str, str] = {}

    if daily_context_df is not None and not daily_context_df.empty and "date" in daily_context_df.columns:
        dc = daily_context_df.sort_values("date").copy()
        for r in dc.to_dict(orient="records"):
            ds = str(r.get("date") or "")
            if not ds:
                continue
            sd_raw = pd.to_numeric(pd.Series([r.get("shot_day_int")]), errors="coerce").iloc[0]
            shot_day_by_date[ds] = f"Day_{int(sd_raw)}" if np.isfinite(sd_raw) else "Pre_Or_Unlinked_GLP1"

            dm_raw = pd.to_numeric(pd.Series([r.get("active_tirz_dose_mg")]), errors="coerce").iloc[0]
            if np.isfinite(dm_raw):
                dose_by_date[ds] = float(dm_raw)

            # In Gold_Daily_Unified, asleep_min on `date` IS the primary nocturnal sleep ending that morning
            asl_raw = pd.to_numeric(pd.Series([r.get("asleep_min")]), errors="coerce").iloc[0]
            sleep_tier_by_date[ds] = _sleep_tier_from_asleep_min(float(asl_raw) if np.isfinite(asl_raw) else np.nan)

    # 1. Build Private_Smoking_Descriptive (strictly separating onset vs post columns)
    desc_rows: list[dict[str, object]] = []
    if not onset_df.empty and not post_df.empty:
        # Join strictly by event_key inside aggregation builder while keeping output columns partitioned
        o_work = onset_df.copy()
        o_work["hour_bucket"] = o_work["onset_hour_ist"].map(
            lambda v: f"{int(float(v)):02d}:00" if pd.notna(v) and str(v).strip() != "" and np.isfinite(float(v)) else "Unlogged_Hour"
        )
        o_work["trigger_clean"] = o_work["trigger_onset"].fillna("").astype(str).str.strip().replace("", "Unlogged_Trigger")
        o_work["mood_clean"] = o_work["mood_onset"].fillna("").astype(str).str.strip().replace("", "Unlogged_Mood")
        o_work["shot_day_stratum"] = o_work["date"].astype(str).map(
            lambda d: shot_day_by_date.get(d, "Unlinked")
        )
        o_work["prior_night_sleep_tier"] = o_work["date"].astype(str).map(
            lambda d: sleep_tier_by_date.get(d, "Unmeasured")
        )

        merged_ev = o_work.merge(
            post_df[
                [
                    "event_key",
                    "event_type",
                    "is_resisted_outcome",
                    "count_smoked_post",
                    "delay_minutes_post",
                ]
            ],
            on="event_key",
            how="left",
        )
        total_events = len(merged_ev)

        dim_specs = (
            ("hour_ist", "hour_bucket"),
            ("Trigger", "trigger_clean"),
            ("Mood", "mood_clean"),
            ("shot_day_int", "shot_day_stratum"),
            ("prior_night_sleep_tier", "prior_night_sleep_tier"),
        )
        for dim_name, col_key in dim_specs:
            for s_val, grp in merged_ev.groupby(col_key, sort=True):
                ev_n = len(grp)
                crav_s = pd.to_numeric(grp["craving_intensity_onset"], errors="coerce").dropna()
                cnt_s = pd.to_numeric(grp["count_smoked_post"], errors="coerce")
                dly_s = pd.to_numeric(grp["delay_minutes_post"], errors="coerce").dropna()
                res_s = grp["is_resisted_outcome"].fillna(False).astype(bool)
                smk_ev_n = int((~res_s & (cnt_s.fillna(0.0) > 0.0)).sum())
                res_ev_n = int(res_s.sum())

                desc_rows.append(
                    {
                        "onset_dimension": dim_name,
                        "onset_stratum": str(s_val),
                        "onset_event_cnt": int(ev_n),
                        "onset_mean_craving_intensity": (
                            round(float(crav_s.mean()), 3) if not crav_s.empty else np.nan
                        ),
                        "post_smoked_event_cnt": smk_ev_n,
                        "post_resisted_event_cnt": res_ev_n,
                        "post_total_count": (
                            round(float(cnt_s.dropna().sum()), 2) if cnt_s.notna().any() else np.nan
                        ),
                        "post_mean_delay_minutes": (
                            round(float(dly_s.mean()), 2) if not dly_s.empty else np.nan
                        ),
                        "schema_partition_guard": "Onset_PreDecision_Strict_Split_From_PostOutcome",
                        "raw_n": int(total_events),
                        "valid_n": int(ev_n),
                        "maturity_level": _classify_private_maturity(ev_n),
                        "epistemic_label": "Descriptive",
                        "epistemic_class": "Descriptive",
                        "caveat_note": "Strictly isolated in Smoking_Private_Store; onset features never mix post-event outcomes.",
                    }
                )

    desc_out = pd.DataFrame(desc_rows, columns=desc_cols)

    # 2. Build Private_Weight_Interval_Smoking
    int_rows: list[dict[str, object]] = []
    if (
        weight_intervals_df is not None
        and not weight_intervals_df.empty
        and {"start_date", "end_date"}.issubset(weight_intervals_df.columns)
    ):
        w_sub = (
            weight_intervals_df.loc[weight_intervals_df["in_all_weighins"]].copy()
            if "in_all_weighins" in weight_intervals_df.columns
            else weight_intervals_df.copy()
        )
        logged_dates_set = (
            set(
                miss_df.loc[
                    miss_df["smoking_log_status"].isin(["Present", "True_Zero"]),
                    "date",
                ].astype(str)
            )
            if not miss_df.empty
            else set()
        )
        n_iv = len(w_sub)
        iv_mat = _classify_private_interval_maturity(n_iv)

        for idx, r in enumerate(w_sub.to_dict(orient="records"), start=1):
            s_ds = str(r["start_date"])
            e_ds = str(r["end_date"])
            iv_days = int(pd.to_numeric(pd.Series([r.get("interval_days", 1)]), errors="coerce").fillna(1).iloc[0])
            o_iv = onset_df.loc[(onset_df["date"] > s_ds) & (onset_df["date"] <= e_ds)] if not onset_df.empty else pd.DataFrame()
            p_iv = post_df.loc[(post_df["date"] > s_ds) & (post_df["date"] <= e_ds)] if not post_df.empty else pd.DataFrame()

            cal_d = pd.date_range(start=s_ds, end=e_ds, freq="D").strftime("%Y-%m-%d").tolist()[1:]
            log_d_cnt = sum(1 for d in cal_d if d in logged_dates_set)
            cov_pct = round(float(log_d_cnt) / float(max(1, iv_days)), 4)

            cnt_iv = pd.to_numeric(p_iv["count_smoked_post"], errors="coerce") if not p_iv.empty else pd.Series(dtype=float)
            dly_iv = pd.to_numeric(p_iv["delay_minutes_post"], errors="coerce").dropna() if not p_iv.empty else pd.Series(dtype=float)
            res_iv = int(p_iv["is_resisted_outcome"].fillna(False).astype(bool).sum()) if not p_iv.empty else 0
            loss_ref = pd.to_numeric(pd.Series([r.get("loss_rate_kg_per_wk")]), errors="coerce").iloc[0]

            int_rows.append(
                {
                    "interval_id": str(r.get("interval_id") or f"PINT-{idx:03d}"),
                    "start_date": s_ds,
                    "end_date": e_ds,
                    "interval_days": iv_days,
                    "logged_days_in_interval": log_d_cnt,
                    "logging_coverage_pct": cov_pct,
                    "onset_craving_events": int(len(o_iv)),
                    "post_total_count": round(float(cnt_iv.dropna().sum()), 2) if cnt_iv.notna().any() else np.nan,
                    "post_resisted_events": res_iv,
                    "post_mean_delay_minutes": round(float(dly_iv.mean()), 2) if not dly_iv.empty else np.nan,
                    "loss_rate_kg_per_wk_ref": round(float(loss_ref), 4) if np.isfinite(loss_ref) else np.nan,
                    "raw_n": n_iv,
                    "valid_n": n_iv,
                    "maturity_level": iv_mat,
                    "epistemic_label": "Descriptive",
                    "epistemic_class": "Descriptive",
                    "caveat_note": "Weight-interval smoking ledger; correlations locked until >= 12 intervals accrue.",
                }
            )

    int_out = pd.DataFrame(int_rows, columns=interval_cols)

    # 3. Build Private_Treatment_Smoking_Scorecard across Tirzepatide dose steps
    sc_rows: list[dict[str, object]] = []
    if not miss_df.empty:
        valid_logged_days = miss_df.loc[
            miss_df["smoking_log_status"].isin(["Present", "True_Zero"]), "date"
        ].astype(str).tolist()

        # Build dose step windows (default anchors + any higher auto-detected doses in dose_by_date)
        step_defs: list[tuple[str, str, str | None]] = [
            ("Pre_Tirzepatide", "2025-06-05", "2026-10-20"),
            ("Tirzepatide_2.5mg", "2026-10-21", "2026-11-18"),
            ("Tirzepatide_5.0mg", "2026-11-19", None),
        ]
        extra_doses = sorted({d for d in dose_by_date.values() if d > 5.01})
        for ed in extra_doses:
            ed_dates = sorted(dt for dt, dv in dose_by_date.items() if abs(dv - ed) <= 1e-4)
            if ed_dates:
                # Close previous open step
                prev_id, prev_s, _ = step_defs[-1]
                prev_end = (pd.Timestamp(ed_dates[0]) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                step_defs[-1] = (prev_id, prev_s, prev_end)
                step_defs.append((f"Tirzepatide_{ed:.1f}mg", ed_dates[0], None))

        prev_cnt_per_day: float = np.nan
        for step_id, p_start, p_end in step_defs:
            days_in_step = [
                d for d in valid_logged_days if d >= p_start and (p_end is None or d <= p_end)
            ]
            n_days = len(days_in_step)
            dset = set(days_in_step)

            o_step = onset_df.loc[onset_df["date"].isin(dset)] if not onset_df.empty else pd.DataFrame()
            p_step = post_df.loc[post_df["date"].isin(dset)] if not post_df.empty else pd.DataFrame()

            if n_days > 0:
                onset_rate = round(float(len(o_step)) / float(n_days), 4)
                crav_m = (
                    round(float(pd.to_numeric(o_step["craving_intensity_onset"], errors="coerce").dropna().mean()), 3)
                    if not o_step.empty and pd.to_numeric(o_step["craving_intensity_onset"], errors="coerce").notna().any()
                    else np.nan
                )
                tot_smk = (
                    float(pd.to_numeric(p_step["count_smoked_post"], errors="coerce").fillna(0.0).sum())
                    if not p_step.empty
                    else 0.0
                )
                cnt_per_d = round(tot_smk / float(n_days), 4)
                res_rate = (
                    round(float(p_step["is_resisted_outcome"].fillna(False).astype(bool).mean()), 4)
                    if not p_step.empty
                    else np.nan
                )
                dly_s = pd.to_numeric(p_step["delay_minutes_post"], errors="coerce").dropna() if not p_step.empty else pd.Series(dtype=float)
                dly_m = round(float(dly_s.mean()), 2) if not dly_s.empty else np.nan
            else:
                onset_rate = crav_m = cnt_per_d = res_rate = dly_m = np.nan

            delta_prior = (
                round(cnt_per_d - prev_cnt_per_day, 4)
                if (np.isfinite(cnt_per_d) and np.isfinite(prev_cnt_per_day) and n_days >= 14)
                else np.nan
            )
            if np.isfinite(cnt_per_d):
                prev_cnt_per_day = cnt_per_d

            sc_rows.append(
                {
                    "treatment_step_id": step_id,
                    "phase_start_date": p_start,
                    "phase_end_date": p_end or "",
                    "logged_days_n": n_days,
                    "onset_events_per_logged_day": onset_rate,
                    "onset_mean_craving_intensity": crav_m,
                    "post_count_per_logged_day": cnt_per_d,
                    "post_resist_rate": res_rate,
                    "post_mean_delay_minutes": dly_m,
                    "delta_count_vs_prior_step": delta_prior,
                    "raw_n": len(miss_df),
                    "valid_n": n_days,
                    "maturity_level": _classify_private_maturity(n_days, min_l1=14),
                    "epistemic_label": "Descriptive",
                    "epistemic_class": "Descriptive",
                    "caveat_note": (
                        "Gated at Level 0 (Collecting Baseline) until >= 14 logged days accrue in dose step."
                        if n_days < 14
                        else "Dose-step comparison across verified logged days in Smoking_Private_Store."
                    ),
                }
            )

    sc_out = pd.DataFrame(sc_rows, columns=scorecard_cols)

    return {
        "Private_Smoking_Descriptive": desc_out,
        "Private_Weight_Interval_Smoking": int_out,
        "Private_Treatment_Smoking_Scorecard": sc_out,
    }
