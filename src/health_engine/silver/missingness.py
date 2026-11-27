"""Silver Gate 6: Missingness Provenance Tags (AGENTS.md Rule 4 & Prompt Section 3.6).

Every daily stream (`sleep`, `vitals`, `nutrition`, `glapp_mood`, `micro_journal`) is tagged:
- `<stream>_status`: one of `Present`, `Not_Worn`, `Not_Logged`, `Source_Gap`, `True_Zero`
- `<stream>_missing_reason`: explicit `AGENTS.md` Rule 4 tag (`Not_Worn`, `Not_Logged`,
  `Source_Gap`, `True_Zero`) when missing/zero, or empty string `""` when `Present`.

Missing wearables, unlogged nutrition days, and unlogged journal days ALWAYS remain `NaN`.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from health_engine.config_loader import add_display_date_col
from health_engine.silver.qa_logger import SilverQALogger


def build_missingness_provenance(
    spine_dates: list[str],
    silver_sleep_daily: pd.DataFrame,
    silver_sleep_sessions: pd.DataFrame,
    silver_vitals: pd.DataFrame,
    silver_nutrition: pd.DataFrame,
    silver_glapp_mood: pd.DataFrame,
    silver_meds: pd.DataFrame,
    silver_symptoms: pd.DataFrame,
    silver_journal_mood: pd.DataFrame,
    thresholds_cfg: dict[str, Any],
    qa: SilverQALogger,
) -> pd.DataFrame:
    """Compute daily missingness status & missing_reason tags across all 5 core streams."""
    glp1_start = str(thresholds_cfg["phase_anchors"]["active_glp1_phase"]["start_date"])

    valid_sleep_dates = (
        set(silver_sleep_daily["date"].dropna().astype(str))
        if not silver_sleep_daily.empty
        else set()
    )
    unstaged_or_partial_sleep_dates = set()
    if not silver_sleep_sessions.empty:
        sub_invalid = silver_sleep_sessions.loc[~silver_sleep_sessions["nocturnal_valid"]]
        unstaged_or_partial_sleep_dates = set(sub_invalid["date"].dropna().astype(str)) - valid_sleep_dates

    vitals_valid_dates: set[str] = set()
    vitals_partial_dates: set[str] = set()
    if not silver_vitals.empty:
        vitals_valid_dates = set(
            silver_vitals.loc[silver_vitals["vitals_valid_coverage"], "date"].astype(str)
        )
        vitals_partial_dates = (
            set(
                silver_vitals.loc[
                    (~silver_vitals["vitals_valid_coverage"]) & (silver_vitals["hr_hours_covered"] > 0),
                    "date",
                ].astype(str)
            )
            - vitals_valid_dates
        )

    nutr_present_dates: set[str] = set()
    if not silver_nutrition.empty and "calories_kcal" in silver_nutrition.columns:
        cal_num = pd.to_numeric(silver_nutrition["calories_kcal"].replace("", pd.NA), errors="coerce")
        nutr_present_dates = set(silver_nutrition.loc[cal_num.notna() & (cal_num > 0), "date"].astype(str))

    glapp_present_dates = (
        set(silver_glapp_mood["date"].dropna().astype(str))
        if not silver_glapp_mood.empty
        else set()
    )

    mj_dates: set[str] = set()
    for mj_df in (silver_meds, silver_symptoms, silver_journal_mood):
        if not mj_df.empty and "date" in mj_df.columns:
            mj_dates |= set(mj_df["date"].dropna().astype(str))
    mj_first_date = min(mj_dates) if mj_dates else None

    rows: list[dict[str, Any]] = []
    for ds in spine_dates:
        # 1. Sleep stream
        if ds in valid_sleep_dates:
            sl_stat, sl_reason = "Present", ""
        elif ds in unstaged_or_partial_sleep_dates:
            sl_stat, sl_reason = "Source_Gap", "Source_Gap"
        else:
            sl_stat, sl_reason = "Not_Worn", "Not_Worn"

        # 2. Vitals stream
        if ds in vitals_valid_dates:
            vi_stat, vi_reason = "Present", ""
        elif ds in vitals_partial_dates:
            vi_stat, vi_reason = "Source_Gap", "Source_Gap"
        else:
            vi_stat, vi_reason = "Not_Worn", "Not_Worn"

        # 3. Nutrition stream
        if ds in nutr_present_dates:
            nu_stat, nu_reason = "Present", ""
        else:
            nu_stat, nu_reason = "Not_Logged", "Not_Logged"

        # 4. Glapp mood stream (app started at GLP-1 initiation)
        if ds in glapp_present_dates:
            gl_stat, gl_reason = "Present", ""
        elif ds < glp1_start:
            gl_stat, gl_reason = "Source_Gap", "Source_Gap"
        else:
            gl_stat, gl_reason = "Not_Logged", "Not_Logged"

        # 5. Micro Journal stream
        if ds in mj_dates:
            mj_stat, mj_reason = "Present", ""
        elif mj_first_date is not None and ds < mj_first_date:
            mj_stat, mj_reason = "Source_Gap", "Source_Gap"
        else:
            mj_stat, mj_reason = "Not_Logged", "Not_Logged"

        rows.append(
            {
                "date": ds,
                "sleep_status": sl_stat,
                "sleep_missing_reason": sl_reason,
                "vitals_status": vi_stat,
                "vitals_missing_reason": vi_reason,
                "nutrition_status": nu_stat,
                "nutrition_missing_reason": nu_reason,
                "glapp_mood_status": gl_stat,
                "glapp_mood_missing_reason": gl_reason,
                "micro_journal_status": mj_stat,
                "micro_journal_missing_reason": mj_reason,
                "epistemic_class": "Descriptive",
            }
        )

    out = add_display_date_col(pd.DataFrame(rows), "date")
    qa.log(
        domain="Missingness",
        source_tab="All_Daily_Streams",
        rule_id="MISSINGNESS_PROVENANCE_TAGGED",
        total_input_rows=len(out),
        flagged_or_excluded_rows=int((out["sleep_status"] != "Present").sum()),
        valid_output_rows=len(out),
        detail_note=(
            f"Present days: sleep={len(valid_sleep_dates)}, vitals={len(vitals_valid_dates)}, "
            f"nutrition={len(nutr_present_dates)}, glapp={len(glapp_present_dates)}, journal={len(mj_dates)}"
        ),
    )
    return out
