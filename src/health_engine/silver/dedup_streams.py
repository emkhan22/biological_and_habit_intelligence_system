"""Silver Gate 4 & Auxiliary Stream Cleaners: Deduplication & Typed Bronze->Silver Ingestion.

Rules:
- `Micro Journal!Meds_Log`: deduplicate identical `(Date, Time, Med_or_Supplement, Dose, Status)`.
- `Micro Journal!Symptoms_Log`: normalize singular/plural symptom names into `Normalized_Symptom`
  (never overwriting `Symptom`, e.g., `Right Foot Cramps` -> `Right Foot Cramp`) and deduplicate
  identical `(Date, Time, Normalized_Symptom)`.
- `HealthConnectDaily!Workouts`: deduplicate identical `(date, start_time_ist, end_time_ist, source_app)`.
- Clean & type auxiliary non-smoking tables (`Journal_Mood`, `Steps`, `Activity_Daily`, `Nutrition`,
  `Nutrition_Log`, `Body_Composition`, `Body_Measurements`, `Weight`, `Sleep_HR_Events`, `Glapp`).
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from health_engine.config_loader import add_display_date_col, to_iso_date_str
from health_engine.silver.qa_logger import SilverQALogger


def _norm_time_hhmm(val: Any) -> str:
    if val is None or pd.isna(val):
        return ""
    s = str(val).strip()
    if not s:
        return ""
    m = re.search(r"(\d{1,2}):(\d{2})", s)
    if m:
        return f"{int(m.group(1)):02d}:{m.group(2)}"
    return s


def canonicalize_symptom_name(raw_symptom: Any, overrides: dict[str, str] | None = None) -> str:
    """Normalize symptom casing & singular/plural (e.g. 'Right Foot Cramps' -> 'Right Foot Cramp')."""
    s = re.sub(r"\s+", " ", str(raw_symptom or "").strip())
    if not s:
        return ""
    low = s.lower()
    if overrides and low in overrides:
        return overrides[low]

    words = s.split(" ")
    last = words[-1]
    last_low = last.lower()

    # Preserve words naturally ending in 'ss', 'us', 'is', or invariant symptom nouns
    invariant_endings = ("ness", "itis", "osis", "sis", "us", "ss", "gas", "chills", "hives", "aches_keep")
    if not any(last_low.endswith(sfx) for sfx in invariant_endings) and len(last) > 3:
        if last_low.endswith("ies") and len(last) > 4:
            last = last[:-3] + "y"
        elif last_low.endswith(("cramps", "aches", "pains", "spasms", "twitches", "headaches", "migraines", "burps")):
            last = last[:-1]
        elif last_low.endswith("s") and not last_low.endswith(("es", "ss")):
            last = last[:-1]

    words[-1] = last
    titled = " ".join(w.capitalize() if w.islower() else w for w in words)
    if overrides and titled.lower() in overrides:
        return overrides[titled.lower()]
    return titled


def clean_meds_log(meds_raw: pd.DataFrame, qa: SilverQALogger) -> pd.DataFrame:
    """Deduplicate identical (Date, Time, Med_or_Supplement, Dose, Status) rows."""
    total_in = len(meds_raw)
    if meds_raw.empty:
        qa.log("Micro_Journal", "Micro Journal!Meds_Log", "MEDS_DUPLICATE_REMOVED", 0, 0, 0, "Empty Meds_Log")
        return pd.DataFrame()

    df = meds_raw.copy()
    df["date"] = df["Date"].map(to_iso_date_str)
    df["time_ist"] = df["Time"].map(_norm_time_hhmm)
    df["_dedup_med"] = df["Med_or_Supplement"].astype(str).str.strip().str.lower()
    df["_dedup_dose"] = df["Dose"].astype(str).str.strip().str.lower()
    df["_dedup_status"] = df["Status"].astype(str).str.strip().str.lower()

    dedup_cols = ["date", "time_ist", "_dedup_med", "_dedup_dose", "_dedup_status"]
    dup_mask = df.duplicated(subset=dedup_cols, keep="first")
    dup_cnt = int(dup_mask.sum())

    out = df.loc[~dup_mask].drop(columns=["_dedup_med", "_dedup_dose", "_dedup_status"]).copy()
    out["epistemic_class"] = "Measured"
    out = add_display_date_col(out, "date")

    qa.log(
        domain="Micro_Journal",
        source_tab="Micro Journal!Meds_Log",
        rule_id="MEDS_DUPLICATE_REMOVED",
        total_input_rows=total_in,
        flagged_or_excluded_rows=dup_cnt,
        valid_output_rows=len(out),
        detail_note="Deduplicated identical (Date, Time, Med_or_Supplement, Dose, Status)",
    )
    return out.reset_index(drop=True)


def clean_symptoms_log(
    symptoms_raw: pd.DataFrame,
    thresholds_cfg: dict[str, Any],
    qa: SilverQALogger,
) -> pd.DataFrame:
    """Add Normalized_Symptom (singular/plural) without overwriting Symptom, and deduplicate."""
    total_in = len(symptoms_raw)
    if symptoms_raw.empty:
        qa.log(
            "Micro_Journal",
            "Micro Journal!Symptoms_Log",
            "SYMPTOMS_DUPLICATE_REMOVED",
            0,
            0,
            0,
            "Empty Symptoms_Log",
        )
        return pd.DataFrame()

    overrides_raw = thresholds_cfg.get("silver_gates", {}).get("symptom_canonical_overrides", {})
    overrides = {str(k).strip().lower(): str(v).strip() for k, v in overrides_raw.items()}

    df = symptoms_raw.copy()
    orig_symptom_snapshot = df["Symptom"].copy()

    df["date"] = df["Date"].map(to_iso_date_str)
    df["time_ist"] = df["Time"].map(_norm_time_hhmm)
    df["Normalized_Symptom"] = [
        canonicalize_symptom_name(v, overrides) for v in df["Symptom"].to_numpy()
    ]
    df["_dedup_sym"] = df["Normalized_Symptom"].str.lower()

    dup_mask = df.duplicated(subset=["date", "time_ist", "_dedup_sym"], keep="first")
    dup_cnt = int(dup_mask.sum())

    if not df["Symptom"].equals(orig_symptom_snapshot):
        raise RuntimeError("Original 'Symptom' column was mutated!")

    out = df.loc[~dup_mask].drop(columns=["_dedup_sym"]).copy()
    out["epistemic_class"] = "Measured"
    out = add_display_date_col(out, "date")

    qa.log(
        domain="Micro_Journal",
        source_tab="Micro Journal!Symptoms_Log",
        rule_id="SYMPTOMS_DUPLICATE_REMOVED",
        total_input_rows=total_in,
        flagged_or_excluded_rows=dup_cnt,
        valid_output_rows=len(out),
        detail_note="Canonicalized singular/plural Normalized_Symptom and deduplicated (Date, Time, Normalized_Symptom)",
    )
    return out.reset_index(drop=True)


def clean_journal_mood(mood_raw: pd.DataFrame, qa: SilverQALogger) -> pd.DataFrame:
    """Clean Micro Journal!Journal_Mood preserving raw fields."""
    if mood_raw.empty:
        qa.log("Micro_Journal", "Micro Journal!Journal_Mood", "JOURNAL_MOOD_INGESTED", 0, 0, 0, "Empty Journal_Mood")
        return pd.DataFrame()
    df = mood_raw.copy()
    df["date"] = df["Date"].map(to_iso_date_str)
    df["time_ist"] = df["Time"].map(_norm_time_hhmm)
    dup_mask = df.duplicated(subset=["date", "time_ist", "Mood_Tags", "Summary"], keep="first")
    out = df.loc[~dup_mask].copy()
    out["epistemic_class"] = "Measured"
    out = add_display_date_col(out, "date")
    qa.log(
        domain="Micro_Journal",
        source_tab="Micro Journal!Journal_Mood",
        rule_id="JOURNAL_MOOD_INGESTED",
        total_input_rows=len(mood_raw),
        flagged_or_excluded_rows=int(dup_mask.sum()),
        valid_output_rows=len(out),
        detail_note="Cleaned non-smoking Journal_Mood entries",
    )
    return out.reset_index(drop=True)


def clean_workouts(workouts_raw: pd.DataFrame, qa: SilverQALogger) -> pd.DataFrame:
    """Deduplicate identical (date, start_time_ist, end_time_ist, source_app) workout sessions."""
    total_in = len(workouts_raw)
    if workouts_raw.empty:
        qa.log("Workouts", "HealthConnectDaily!Workouts", "WORKOUTS_DUPLICATE_REMOVED", 0, 0, 0, "Empty Workouts")
        return pd.DataFrame()

    df = workouts_raw.copy()
    df["date"] = df["date"].map(to_iso_date_str)
    df["_k_start"] = df["start_time_ist"].astype(str).str.strip()
    df["_k_end"] = df["end_time_ist"].astype(str).str.strip()
    df["_k_src"] = df["source_app"].astype(str).str.strip().str.lower()

    dup_mask = df.duplicated(subset=["date", "_k_start", "_k_end", "_k_src"], keep="first")
    dup_cnt = int(dup_mask.sum())

    out = df.loc[~dup_mask].drop(columns=["_k_start", "_k_end", "_k_src"]).copy()
    for num_c in ("duration_mins", "avg_hr", "max_hr", "moderate_mins", "vigorous_mins", "peak_mins", "zone_mins"):
        if num_c in out.columns:
            out[f"{num_c}_val"] = pd.to_numeric(out[num_c].replace("", np.nan), errors="coerce")
    out["epistemic_class"] = "Measured"
    out = add_display_date_col(out, "date")

    qa.log(
        domain="Workouts",
        source_tab="HealthConnectDaily!Workouts",
        rule_id="WORKOUTS_DUPLICATE_REMOVED",
        total_input_rows=total_in,
        flagged_or_excluded_rows=dup_cnt,
        valid_output_rows=len(out),
        detail_note="Deduplicated identical (date, start_time_ist, end_time_ist, source_app)",
    )
    return out.reset_index(drop=True)


def clean_simple_dated_tab(
    raw_df: pd.DataFrame,
    date_col: str = "date",
    dedup_subset: list[str] | None = None,
) -> pd.DataFrame:
    """Normalize ISO date and add display_date without mutating raw source columns."""
    if raw_df.empty:
        return pd.DataFrame()
    df = raw_df.copy()
    if date_col in df.columns:
        df["date"] = df[date_col].map(to_iso_date_str)
        df = df.loc[df["date"].notna()].copy()
    if dedup_subset:
        present_sub = [c for c in dedup_subset if c in df.columns]
        if present_sub:
            df = df.drop_duplicates(subset=present_sub, keep="last").copy()
    df["epistemic_class"] = "Measured"
    return add_display_date_col(df, "date").reset_index(drop=True)
