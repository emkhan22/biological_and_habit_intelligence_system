"""1-Compartment Bateman Pharmacokinetic Superposition for Tirzepatide.

Evaluates:
- Daily trajectory at 12:00:00 IST for `Gold_Daily_Unified` (`tirz_pk_est`)
- Continuous hourly curve (`Gold_Tirzepatide_PK_Hourly`) for UI overlay
- Strictly tagged with `tirz_pk_provenance = 'Model_Estimate'` and `epistemic_class = 'Prediction'`
"""

from __future__ import annotations

from datetime import date
import math
import re
from typing import Any

import numpy as np
import pandas as pd

from health_engine.config_loader import IST, add_display_date_col, to_iso_date_str


def _parse_shot_records(shots_silver: pd.DataFrame, default_time: str = "12:00") -> list[dict[str, Any]]:
    if shots_silver.empty:
        return []
    parsed: list[dict[str, Any]] = []
    for r in shots_silver.to_dict(orient="records"):
        ds = to_iso_date_str(r.get("date"))
        dose_raw = r.get("dose_mg")
        if not ds or dose_raw is None or str(dose_raw).strip() == "":
            continue
        try:
            dose = float(dose_raw)
        except ValueError:
            continue
        t_raw = str(r.get("time") or "").strip()
        m = re.search(r"(\d{1,2}):(\d{2})", t_raw)
        hhmm = f"{int(m.group(1)):02d}:{m.group(2)}" if m else default_time
        ts = pd.Timestamp(f"{ds} {hhmm}:00", tz=IST)
        parsed.append({"date": ds, "time": hhmm, "ts": ts, "dose_mg": dose})
    parsed.sort(key=lambda x: x["ts"])
    return parsed


def _bateman_superposition_mg(
    eval_timestamps: pd.DatetimeIndex,
    shots: list[dict[str, Any]],
    ka_per_day: float,
    ke_per_day: float,
) -> np.ndarray:
    """Vectorized 1-compartment Bateman sum across all prior doses."""
    n = len(eval_timestamps)
    if n == 0 or not shots:
        return np.full(n, np.nan, dtype=float)

    eval_sec = eval_timestamps.as_unit("ns").asi8.astype(np.float64) / 1e9  # epoch seconds
    total = np.zeros(n, dtype=float)
    has_prior = np.zeros(n, dtype=bool)
    scale = ka_per_day / (ka_per_day - ke_per_day)

    for s in shots:
        shot_sec = s["ts"].timestamp()
        dt_days = (eval_sec - shot_sec) / 86400.0
        mask = dt_days > 0.0
        if np.any(mask):
            has_prior |= mask
            dt_pos = dt_days[mask]
            conc = s["dose_mg"] * scale * (
                np.exp(-ke_per_day * dt_pos) - np.exp(-ka_per_day * dt_pos)
            )
            total[mask] += np.maximum(0.0, conc)

    return np.where(has_prior, np.round(total, 4), np.nan)


def build_tirzepatide_pk_tables(
    spine_dates: list[str],
    shots_silver: pd.DataFrame,
    thresholds_cfg: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (daily_pk_df, Gold_Tirzepatide_PK_Hourly)."""
    pk_cfg = thresholds_cfg["pharmacokinetics"]["tirzepatide"]
    half_life_days = float(pk_cfg["half_life_days"])
    ka_per_day = float(pk_cfg["ka_per_day"])
    ke_per_day = math.log(2.0) / half_life_days
    eval_time = str(pk_cfg.get("daily_eval_time_ist", "12:00:00"))
    default_shot_time = str(pk_cfg.get("default_shot_time_ist", "12:00"))
    glp1_start = str(thresholds_cfg["phase_anchors"]["active_glp1_phase"]["start_date"])

    shots = _parse_shot_records(shots_silver, default_shot_time)

    # 1. Daily PK evaluated at 12:00:00 IST
    if not spine_dates:
        return pd.DataFrame(), pd.DataFrame()

    daily_ts = pd.DatetimeIndex(
        [pd.Timestamp(f"{d} {eval_time}", tz=IST) for d in spine_dates]
    )
    pk_daily_vals = _bateman_superposition_mg(daily_ts, shots, ka_per_day, ke_per_day)

    glp1_start_dt = date.fromisoformat(glp1_start)
    rows: list[dict[str, Any]] = []
    shot_idx = -1

    for idx, ds in enumerate(spine_dates):
        d_obj = date.fromisoformat(ds)
        while shot_idx + 1 < len(shots) and shots[shot_idx + 1]["date"] <= ds:
            shot_idx += 1

        if shot_idx >= 0:
            last_shot = shots[shot_idx]
            last_dt = date.fromisoformat(last_shot["date"])
            days_since = (d_obj - last_dt).days
            shot_day_int = min(days_since, 6)
            active_dose = float(last_shot["dose_mg"])
            wk_num = max(1, ((d_obj - glp1_start_dt).days // 7) + 1)
            pk_val = float(pk_daily_vals[idx]) if not np.isnan(pk_daily_vals[idx]) else 0.0
            prov = "Model_Estimate"
        else:
            days_since = np.nan
            shot_day_int = np.nan
            active_dose = np.nan
            wk_num = np.nan
            pk_val = np.nan
            prov = ""

        rows.append(
            {
                "date": ds,
                "days_since_shot": days_since,
                "shot_day_int": shot_day_int,
                "active_tirz_dose_mg": active_dose,
                "treatment_week_num": wk_num,
                "tirz_pk_est": pk_val,
                "tirz_pk_provenance": prov,
                "tirz_pk_epistemic_class": "Prediction" if prov else "Descriptive",
            }
        )

    daily_pk_df = pd.DataFrame(rows)

    # 2. Hourly PK curve from first shot date through max spine date for UI overlay
    if not shots:
        hourly_cols = [
            "timestamp_ist",
            "date",
            "display_date",
            "hour_ist",
            "active_tirz_dose_mg",
            "tirz_pk_est",
            "provenance_tag",
            "epistemic_class",
        ]
        return daily_pk_df, pd.DataFrame(columns=hourly_cols)

    first_shot_date = shots[0]["date"]
    last_spine_date = max(spine_dates[-1], shots[-1]["date"])
    hourly_ts = pd.date_range(
        start=pd.Timestamp(f"{first_shot_date} 00:00:00", tz=IST),
        end=pd.Timestamp(f"{last_spine_date} 23:00:00", tz=IST),
        freq="h",
    )
    hourly_vals = _bateman_superposition_mg(hourly_ts, shots, ka_per_day, ke_per_day)
    hourly_dates = hourly_ts.strftime("%Y-%m-%d")

    dose_by_date = daily_pk_df.set_index("date")["active_tirz_dose_mg"].to_dict()
    hourly_df = pd.DataFrame(
        {
            "timestamp_ist": hourly_ts.strftime("%Y-%m-%d %H:%M:%S"),
            "date": hourly_dates,
            "hour_ist": hourly_ts.hour,
            "active_tirz_dose_mg": [dose_by_date.get(d, np.nan) for d in hourly_dates],
            "tirz_pk_est": np.nan_to_num(hourly_vals, nan=0.0),
            "provenance_tag": "Model_Estimate",
            "epistemic_class": "Prediction",
        }
    )
    hourly_df = add_display_date_col(hourly_df, "date")
    return daily_pk_df, hourly_df
