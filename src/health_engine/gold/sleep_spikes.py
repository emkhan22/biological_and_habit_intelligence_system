"""Gold Table 3: Gold_Sleep_HR_Spikes (Stage-Specific Denominators).

Rules (Prompt Section 4.3):
- Join `Sleep_HR_Events` with valid Fitbit `Silver_Sleep_Daily` (`Sleep_Sessions`) and
  `Silver_Sleep_Stage_Timeline`.
- Compute per night:
  - `spikes_per_asleep_hr = total_spikes / (asleep_min / 60)`
  - Stage-specific rates using THAT stage's duration (in hours) as denominator:
    `light_spikes_per_light_hr`, `rem_spikes_per_rem_hr`, `deep_spikes_per_deep_hr`
  - `wake_transition_spike_cnt` (for `Awake`, `Light / Awake`, `Unstaged`, etc.)
  - `mean_delta_bpm`, `max_peak_bpm`, `total_elevated_sec`, `pct_sleep_elevated`, `hr_coverage_valid`
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from health_engine.config_loader import add_display_date_col, to_iso_date_str


def _num_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[col].replace("", np.nan), errors="coerce")


def _infer_stage_from_timeline(
    ev_start: pd.Timestamp,
    ev_end: pd.Timestamp,
    sess_timeline: pd.DataFrame,
) -> str:
    if sess_timeline.empty or pd.isna(ev_start) or pd.isna(ev_end):
        return "Unstaged"
    overlaps = sess_timeline.loc[
        (sess_timeline["_t0"] < ev_end) & (sess_timeline["_t1"] > ev_start),
        "stage_name",
    ].astype(str).str.strip().unique().tolist()
    if not overlaps:
        return "Unstaged"
    if len(overlaps) == 1:
        return overlaps[0]
    return " / ".join(overlaps)


def _classify_spike_stage(stage_ctx: str) -> str:
    s = str(stage_ctx or "").strip().lower()
    if s == "light":
        return "light"
    if s == "rem":
        return "rem"
    if s == "deep":
        return "deep"
    return "wake_transition"


def build_gold_sleep_hr_spikes(
    silver_sleep_daily: pd.DataFrame,
    silver_stage_timeline: pd.DataFrame,
    sleep_hr_events_silver: pd.DataFrame,
) -> pd.DataFrame:
    """Build Gold_Sleep_HR_Spikes with exact per-stage hourly denominators."""
    cols = [
        "date",
        "display_date",
        "session_id",
        "asleep_min",
        "light_min",
        "rem_min",
        "deep_min",
        "hr_coverage_valid",
        "total_spikes",
        "light_spike_cnt",
        "rem_spike_cnt",
        "deep_spike_cnt",
        "wake_transition_spike_cnt",
        "spikes_per_asleep_hr",
        "light_spikes_per_light_hr",
        "rem_spikes_per_rem_hr",
        "deep_spikes_per_deep_hr",
        "mean_delta_bpm",
        "max_peak_bpm",
        "total_elevated_sec",
        "pct_sleep_elevated",
        "epistemic_class",
    ]
    if silver_sleep_daily.empty:
        return pd.DataFrame(columns=cols)

    # Index stage timeline by (date, session_id) for fallback stage lookup & verification
    tl_by_sess: dict[tuple[str, str], pd.DataFrame] = {}
    if not silver_stage_timeline.empty:
        tl = silver_stage_timeline.copy()
        tl["_t0"] = pd.to_datetime(tl["stage_start_ist"], errors="coerce")
        tl["_t1"] = pd.to_datetime(tl["stage_end_ist"], errors="coerce")
        for (d_k, s_k), grp in tl.groupby(["date", "session_id"]):
            tl_by_sess[(str(d_k), str(s_k))] = grp

    # Index HR spike events by (date, session_id)
    ev_by_sess: dict[tuple[str, str], pd.DataFrame] = {}
    if not sleep_hr_events_silver.empty:
        ev = sleep_hr_events_silver.copy()
        ev["date"] = ev["date"].map(to_iso_date_str)
        ev["_dur_sec"] = _num_series(ev, "duration_sec").fillna(0.0)
        ev["_peak_bpm"] = _num_series(ev, "peak_hr_bpm")
        ev["_delta_bpm"] = _num_series(ev, "delta_bpm")
        ev["_ev_t0"] = pd.to_datetime(ev["event_start_ist"], errors="coerce")
        ev["_ev_t1"] = pd.to_datetime(ev["event_end_ist"], errors="coerce")
        for (d_k, s_k), grp in ev.groupby(["date", "session_id"]):
            ev_by_sess[(str(d_k), str(s_k))] = grp

    out_rows: list[dict[str, Any]] = []
    for srow in silver_sleep_daily.sort_values("date").to_dict(orient="records"):
        ds = str(srow["date"])
        sid = str(srow["session_id"])
        asleep_m = float(srow["asleep_min"]) if pd.notna(srow.get("asleep_min")) else np.nan
        light_m = float(srow["light_min"]) if pd.notna(srow.get("light_min")) else 0.0
        rem_m = float(srow["rem_min"]) if pd.notna(srow.get("rem_min")) else 0.0
        deep_m = float(srow["deep_min"]) if pd.notna(srow.get("deep_min")) else 0.0

        hr_samples = srow.get("sleep_hr_sample_count")
        hr_avg = srow.get("sleep_hr_avg")
        hr_cov_valid = bool(
            pd.notna(hr_avg) and (pd.isna(hr_samples) or float(hr_samples) > 0.0)
        )

        sess_ev = ev_by_sess.get((ds, sid), pd.DataFrame())
        sess_tl = tl_by_sess.get((ds, sid), pd.DataFrame())

        if not hr_cov_valid or pd.isna(asleep_m) or asleep_m <= 0.0:
            out_rows.append(
                {
                    "date": ds,
                    "session_id": sid,
                    "asleep_min": asleep_m,
                    "light_min": light_m,
                    "rem_min": rem_m,
                    "deep_min": deep_m,
                    "hr_coverage_valid": False,
                    "total_spikes": np.nan,
                    "light_spike_cnt": np.nan,
                    "rem_spike_cnt": np.nan,
                    "deep_spike_cnt": np.nan,
                    "wake_transition_spike_cnt": np.nan,
                    "spikes_per_asleep_hr": np.nan,
                    "light_spikes_per_light_hr": np.nan,
                    "rem_spikes_per_rem_hr": np.nan,
                    "deep_spikes_per_deep_hr": np.nan,
                    "mean_delta_bpm": np.nan,
                    "max_peak_bpm": np.nan,
                    "total_elevated_sec": np.nan,
                    "pct_sleep_elevated": np.nan,
                    "epistemic_class": "Descriptive",
                }
            )
            continue

        if sess_ev.empty:
            total_spikes = 0
            light_cnt = 0
            rem_cnt = 0
            deep_cnt = 0
            wake_trans_cnt = 0
            mean_delta = np.nan
            max_peak = np.nan
            elev_sec = 0.0
        else:
            buckets = {"light": 0, "rem": 0, "deep": 0, "wake_transition": 0}
            for er in sess_ev.to_dict(orient="records"):
                ctx = str(er.get("stage_context") or "").strip()
                if not ctx:
                    ctx = _infer_stage_from_timeline(er["_ev_t0"], er["_ev_t1"], sess_tl)
                buckets[_classify_spike_stage(ctx)] += 1

            total_spikes = len(sess_ev)
            light_cnt = buckets["light"]
            rem_cnt = buckets["rem"]
            deep_cnt = buckets["deep"]
            wake_trans_cnt = buckets["wake_transition"]
            mean_delta = (
                round(float(sess_ev["_delta_bpm"].mean()), 4)
                if sess_ev["_delta_bpm"].notna().any()
                else np.nan
            )
            max_peak = (
                round(float(sess_ev["_peak_bpm"].max()), 2)
                if sess_ev["_peak_bpm"].notna().any()
                else np.nan
            )
            elev_sec = round(float(sess_ev["_dur_sec"].sum()), 2)

        asleep_hr = asleep_m / 60.0
        light_rate = round(light_cnt / (light_m / 60.0), 4) if light_m > 0.0 else np.nan
        rem_rate = round(rem_cnt / (rem_m / 60.0), 4) if rem_m > 0.0 else np.nan
        deep_rate = round(deep_cnt / (deep_m / 60.0), 4) if deep_m > 0.0 else np.nan

        out_rows.append(
            {
                "date": ds,
                "session_id": sid,
                "asleep_min": asleep_m,
                "light_min": light_m,
                "rem_min": rem_m,
                "deep_min": deep_m,
                "hr_coverage_valid": True,
                "total_spikes": total_spikes,
                "light_spike_cnt": light_cnt,
                "rem_spike_cnt": rem_cnt,
                "deep_spike_cnt": deep_cnt,
                "wake_transition_spike_cnt": wake_trans_cnt,
                "spikes_per_asleep_hr": round(total_spikes / asleep_hr, 4),
                "light_spikes_per_light_hr": light_rate,
                "rem_spikes_per_rem_hr": rem_rate,
                "deep_spikes_per_deep_hr": deep_rate,
                "mean_delta_bpm": mean_delta,
                "max_peak_bpm": max_peak,
                "total_elevated_sec": elev_sec,
                "pct_sleep_elevated": round((elev_sec / (asleep_m * 60.0)) * 100.0, 4),
                "epistemic_class": "Descriptive",
            }
        )

    out_df = add_display_date_col(pd.DataFrame(out_rows), "date")
    return out_df[cols]
