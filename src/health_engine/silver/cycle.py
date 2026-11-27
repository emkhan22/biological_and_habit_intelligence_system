"""Silver Gate 5: Stitched Menstrual Cycle Dimension (Silver_Cycle_Dim).

Rules:
- Unify `Cycles` (Flo ownership <= 2026-11-05), `Menstrual_Cycle` (Fitbit/HC ownership >= 2026-11-06),
  `Period_Flow_Daily`, and `Archived_Menstrual_Cycle` (preserving `Spotting` events).
- Output daily:
  - `cycle_day`
  - `is_period_day`
  - `is_spotting_day`
  - `flow_level`
  - `cycle_phase` (`Menstrual`, `Follicular_D1_14`, `Mid_Extended`, `Late_Luteal_PreMenst`)
  - `source_owner` (`Flo` vs `Fitbit_HC`)
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd

from health_engine.config_loader import add_display_date_col, to_iso_date_str
from health_engine.silver.qa_logger import SilverQALogger


def _expand_date_span(start_iso: str | None, end_iso: str | None) -> list[str]:
    if not start_iso:
        return []
    d0 = date.fromisoformat(start_iso)
    d1 = date.fromisoformat(end_iso) if end_iso else d0
    if d1 < d0:
        d1 = d0
    days = (d1 - d0).days
    return [(d0 + timedelta(days=i)).isoformat() for i in range(days + 1)]


def build_silver_cycle_dim(
    cycles_raw: pd.DataFrame,
    menstrual_cycle_raw: pd.DataFrame,
    period_flow_raw: pd.DataFrame,
    archived_mc_raw: pd.DataFrame,
    thresholds_cfg: dict[str, Any],
    qa: SilverQALogger,
    max_calendar_date: str | None = None,
    min_new_cycle_gap_days: int = 18,
) -> pd.DataFrame:
    """Build stitched daily Silver_Cycle_Dim across Flo and Fitbit/HealthConnect ownership eras."""
    own_cfg = thresholds_cfg["phase_anchors"]["menstrual_ownership"]
    flo_end = str(own_cfg["flo_end_date"])
    hc_start = str(own_cfg["fitbit_hc_start_date"])
    spine_start = str(thresholds_cfg["calendar_spine"]["start_date"])

    period_days: dict[str, str] = {}     # date -> flow_level
    spotting_days: set[str] = set()
    flo_cycle_starts: set[str] = set()
    hc_period_starts: list[str] = []

    # 1. Flo ownership era (<= flo_end): Cycles + Period_Flow_Daily + Flo Archived_Menstrual_Cycle
    if not cycles_raw.empty:
        for r in cycles_raw.to_dict(orient="records"):
            cs = to_iso_date_str(r.get("cycle_start"))
            ce = to_iso_date_str(r.get("cycle_end"))
            if not cs or cs > flo_end:
                continue
            flo_cycle_starts.add(cs)
            ce_clipped = min(ce, flo_end) if ce else cs
            for d in _expand_date_span(cs, ce_clipped):
                if d <= flo_end:
                    period_days.setdefault(d, "Period_Span")

    if not period_flow_raw.empty:
        for r in period_flow_raw.to_dict(orient="records"):
            d = to_iso_date_str(r.get("date"))
            cs = to_iso_date_str(r.get("cycle_start"))
            fl = str(r.get("flow_level") or "").strip()
            if not d or d > flo_end:
                continue
            if cs and cs <= flo_end:
                flo_cycle_starts.add(cs)
            if fl.lower() == "spotting":
                spotting_days.add(d)
            else:
                period_days[d] = fl or "Flow Unmarked"

    if not archived_mc_raw.empty:
        for r in archived_mc_raw.to_dict(orient="records"):
            d = to_iso_date_str(r.get("date"))
            ed = to_iso_date_str(r.get("end_date"))
            rtype = str(r.get("record_type") or "").strip().lower()
            fl = str(r.get("flow_level") or "").strip()
            src = str(r.get("source_app") or "").strip().lower()
            if not d:
                continue
            if d <= flo_end and ("flo" in src or src == ""):
                if rtype == "spotting" or fl.lower() == "spotting":
                    for x in _expand_date_span(d, min(ed, flo_end) if ed else d):
                        spotting_days.add(x)
                elif rtype in {"period", "flow"}:
                    if rtype == "period":
                        flo_cycle_starts.add(d)
                    for x in _expand_date_span(d, min(ed, flo_end) if ed else d):
                        period_days.setdefault(x, fl or "Period_Span")
            elif d >= hc_start and ("fitbit" in src or "health" in src):
                if rtype == "spotting" or fl.lower() == "spotting":
                    for x in _expand_date_span(max(d, hc_start), ed):
                        spotting_days.add(x)
                elif rtype in {"period", "flow"}:
                    if rtype == "period":
                        hc_period_starts.append(d)
                    for x in _expand_date_span(max(d, hc_start), ed):
                        period_days.setdefault(x, fl or "Period_Span")

    # 2. Fitbit/HC ownership era (>= hc_start): Menstrual_Cycle
    if not menstrual_cycle_raw.empty:
        for r in menstrual_cycle_raw.to_dict(orient="records"):
            d = to_iso_date_str(r.get("date"))
            ed = to_iso_date_str(r.get("end_date"))
            rtype = str(r.get("record_type") or "").strip().lower()
            fl = str(r.get("flow_level") or "").strip()
            src = str(r.get("source_app") or "").strip().lower()
            if not d:
                continue
            span_end = ed or d
            if span_end < hc_start:
                continue
            if "flo" in src and d <= flo_end:
                continue
            eff_start = max(d, hc_start)
            if rtype == "spotting" or fl.lower() == "spotting":
                for x in _expand_date_span(eff_start, span_end):
                    spotting_days.add(x)
            else:
                if rtype == "period" and d >= hc_start:
                    hc_period_starts.append(d)
                for x in _expand_date_span(eff_start, span_end):
                    period_days[x] = fl or period_days.get(x, "Period_Span")

    # If a day has both period flow and spotting, menses takes precedence for is_period_day,
    # while pure spotting days remain non-period spotting events.
    pure_spotting_days = spotting_days - set(period_days.keys())

    # 3. Stitch cycle_start anchors chronologically
    sorted_starts = sorted(flo_cycle_starts)
    for cand in sorted(set(hc_period_starts)):
        if not sorted_starts:
            sorted_starts.append(cand)
            continue
        prev = date.fromisoformat(sorted_starts[-1])
        curr = date.fromisoformat(cand)
        if (curr - prev).days >= min_new_cycle_gap_days:
            sorted_starts.append(cand)

    all_known_dates = list(period_days.keys()) + list(spotting_days) + sorted_starts
    if max_calendar_date:
        all_known_dates.append(max_calendar_date)

    if not all_known_dates:
        qa.log("Menstrual", "HealthConnectDaily!Cycles+Menstrual_Cycle", "CYCLE_DIM_STITCHED", 0, 0, 0, "No cycle inputs")
        return pd.DataFrame(
            columns=[
                "date",
                "display_date",
                "cycle_start",
                "cycle_day",
                "is_period_day",
                "is_spotting_day",
                "flow_level",
                "cycle_phase",
                "source_owner",
                "epistemic_class",
            ]
        )

    min_dt = min(spine_start, min(sorted_starts) if sorted_starts else spine_start)
    max_dt = max(all_known_dates)

    # Pre-index cycle intervals [start_i, next_start_i)
    cycle_intervals: list[tuple[date, date | None]] = []
    start_dates_obj = [date.fromisoformat(s) for s in sorted_starts]
    for idx, s_dt in enumerate(start_dates_obj):
        nxt_dt = start_dates_obj[idx + 1] if idx + 1 < len(start_dates_obj) else None
        cycle_intervals.append((s_dt, nxt_dt))

    cal_dates = pd.date_range(start=min_dt, end=max_dt, freq="D").strftime("%Y-%m-%d").tolist()

    out_rows: list[dict[str, Any]] = []
    int_idx = 0
    for ds in cal_dates:
        d_obj = date.fromisoformat(ds)
        while (
            int_idx + 1 < len(cycle_intervals)
            and cycle_intervals[int_idx + 1][0] <= d_obj
        ):
            int_idx += 1

        active_start: date | None = None
        next_start: date | None = None
        if cycle_intervals and cycle_intervals[int_idx][0] <= d_obj:
            active_start, next_start = cycle_intervals[int_idx]

        cycle_day: float | int = (
            int((d_obj - active_start).days + 1) if active_start is not None else np.nan
        )
        is_period = ds in period_days
        is_spotting = ds in pure_spotting_days
        flow_lvl = period_days.get(ds, "Spotting" if is_spotting else "")

        if is_period:
            phase = "Menstrual"
        elif active_start is None or pd.isna(cycle_day):
            phase = "Unknown"
        elif next_start is not None and 1 <= (next_start - d_obj).days <= 5:
            phase = "Late_Luteal_PreMenst"
        elif int(cycle_day) <= 14:
            phase = "Follicular_D1_14"
        else:
            phase = "Mid_Extended"

        owner = "Flo" if ds <= flo_end else "Fitbit_HC"
        epistemic = "Measured" if (is_period or is_spotting) else "Descriptive"

        if ds >= spine_start:
            out_rows.append(
                {
                    "date": ds,
                    "cycle_start": active_start.isoformat() if active_start else "",
                    "cycle_day": cycle_day,
                    "is_period_day": bool(is_period),
                    "is_spotting_day": bool(is_spotting),
                    "flow_level": flow_lvl,
                    "cycle_phase": phase,
                    "source_owner": owner,
                    "epistemic_class": epistemic,
                }
            )

    out_df = add_display_date_col(pd.DataFrame(out_rows), "date")
    qa.log(
        domain="Menstrual",
        source_tab="Cycles+Menstrual_Cycle+Period_Flow_Daily+Archived_Menstrual_Cycle",
        rule_id="CYCLE_DIM_STITCHED",
        total_input_rows=len(cycles_raw) + len(menstrual_cycle_raw) + len(period_flow_raw) + len(archived_mc_raw),
        flagged_or_excluded_rows=0,
        valid_output_rows=len(out_df),
        detail_note=(
            f"Stitched {len(sorted_starts)} cycle starts (Flo<={flo_end}, Fitbit_HC>={hc_start}); "
            f"period_days={int(out_df['is_period_day'].sum())}, spotting_days={int(out_df['is_spotting_day'].sum())}"
        ),
    )
    return out_df.reset_index(drop=True)
