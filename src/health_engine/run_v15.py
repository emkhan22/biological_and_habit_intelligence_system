"""CLI Entrypoint & Pipeline Orchestrator for V1.5 Health Intelligence Engine.

Supports:
  python -m health_engine.run_v15 --source csv --data-dir ./sample_data --output-dir ./build_out
  python -m health_engine.run_v15 --source sheets
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from health_engine.config_loader import load_engine_configs
from health_engine.gold.baselines import build_gold_personal_baselines
from health_engine.gold.clinical_indices import build_gold_clinical_indices
from health_engine.gold.daily_unified import build_gold_daily_unified
from health_engine.gold.event_windows import (
    build_gold_event_windows,
    build_med_bioavailability_table,
)
from health_engine.gold.sleep_spikes import build_gold_sleep_hr_spikes
from health_engine.gold.tirzepatide_pk import build_tirzepatide_pk_tables
from health_engine.io_adapter import GoogleSheetsAdapter, HealthIOAdapter, LocalCSVAdapter
from health_engine.silver.cycle import build_silver_cycle_dim
from health_engine.silver.dedup_streams import (
    clean_journal_mood,
    clean_meds_log,
    clean_simple_dated_tab,
    clean_symptoms_log,
    clean_workouts,
)
from health_engine.silver.labs import clean_lab_results
from health_engine.silver.missingness import build_missingness_provenance
from health_engine.silver.qa_logger import SilverQALogger
from health_engine.silver.sleep import clean_sleep_tables
from health_engine.silver.vitals import clean_vitals_tables


def _max_date_across(tables: list[pd.DataFrame], date_cols: tuple[str, ...] = ("date", "Collection_Date")) -> str:
    candidates: list[str] = []
    for tbl in tables:
        if tbl is None or tbl.empty:
            continue
        for dc in date_cols:
            if dc in tbl.columns:
                vals = tbl[dc].dropna().astype(str)
                vals = vals.loc[vals.str.len() >= 10]
                if not vals.empty:
                    candidates.append(str(vals.max())[:10])
    return max(candidates) if candidates else "2026-11-27"


def run_v15_pipeline(
    adapter: HealthIOAdapter,
    config_dir: Path | None = None,
    persist: bool = True,
) -> dict[str, pd.DataFrame]:
    """Execute full Bronze -> Silver -> Gold -> QA pipeline via the supplied I/O adapter."""
    thresholds_cfg, lab_norm_cfg = load_engine_configs(config_dir)
    qa = SilverQALogger()

    # ==================== 1. READ NON-SMOKING BRONZE SOURCES ====================
    hc_sleep_sessions = adapter.read_source_tab("HealthConnectDaily", "Sleep_Sessions")
    hc_sleep_daily = adapter.read_source_tab("HealthConnectDaily", "Sleep")
    hc_stage_tl = adapter.read_source_tab("HealthConnectDaily", "Sleep_Stage_Timeline")
    hc_sleep_hr_ev = adapter.read_source_tab("HealthConnectDaily", "Sleep_HR_Events")
    hc_vitals = adapter.read_source_tab("HealthConnectDaily", "Vitals")
    hc_hr_hourly = adapter.read_source_tab("HealthConnectDaily", "Heart_Rate_Hourly")
    hc_steps = adapter.read_source_tab("HealthConnectDaily", "Steps")
    hc_activity = adapter.read_source_tab("HealthConnectDaily", "Activity_Daily")
    hc_workouts = adapter.read_source_tab("HealthConnectDaily", "Workouts")
    hc_nutrition = adapter.read_source_tab("HealthConnectDaily", "Nutrition")
    hc_nutr_log = adapter.read_source_tab("HealthConnectDaily", "Nutrition_Log")
    hc_weight = adapter.read_source_tab("HealthConnectDaily", "Weight")
    hc_body_comp = adapter.read_source_tab("HealthConnectDaily", "Body_Composition")
    hc_body_meas = adapter.read_source_tab("HealthConnectDaily", "Body_Measurements")
    hc_cycles = adapter.read_source_tab("HealthConnectDaily", "Cycles")
    hc_menstrual = adapter.read_source_tab("HealthConnectDaily", "Menstrual_Cycle")
    hc_flow_daily = adapter.read_source_tab("HealthConnectDaily", "Period_Flow_Daily")
    hc_arch_mc = adapter.read_source_tab("HealthConnectDaily", "Archived_Menstrual_Cycle")

    gl_shots = adapter.read_source_tab("Glapp Data", "shots")
    gl_weight = adapter.read_source_tab("Glapp Data", "weight")
    gl_mood = adapter.read_source_tab("Glapp Data", "mood")
    gl_se = adapter.read_source_tab("Glapp Data", "side_effects")

    med_labs = adapter.read_source_tab("AggregatedMedicalRecords", "Lab_Results")

    # Micro Journal strictly reads ONLY the 3 non-smoking tabs
    mj_meds = adapter.read_source_tab("Micro Journal", "Meds_Log")
    mj_symptoms = adapter.read_source_tab("Micro Journal", "Symptoms_Log")
    mj_mood = adapter.read_source_tab("Micro Journal", "Journal_Mood")

    # ==================== 2. SILVER CLEANING GATE ====================
    sleep_silver_dict = clean_sleep_tables(
        hc_sleep_sessions, hc_sleep_daily, hc_stage_tl, thresholds_cfg, qa
    )
    silver_vitals = clean_vitals_tables(hc_vitals, hc_hr_hourly, thresholds_cfg, qa)
    silver_labs = clean_lab_results(med_labs, lab_norm_cfg, qa)

    silver_meds = clean_meds_log(mj_meds, qa)
    silver_symptoms = clean_symptoms_log(mj_symptoms, thresholds_cfg, qa)
    silver_journal_mood = clean_journal_mood(mj_mood, qa)
    silver_workouts = clean_workouts(hc_workouts, qa)

    silver_steps = clean_simple_dated_tab(hc_steps, "date", ["date"])
    silver_activity = clean_simple_dated_tab(hc_activity, "date", ["date"])
    silver_nutrition = clean_simple_dated_tab(hc_nutrition, "date", ["date"])
    silver_nutr_log = clean_simple_dated_tab(hc_nutr_log, "date")
    silver_hc_weight = clean_simple_dated_tab(hc_weight, "date", ["date"])
    silver_body_comp = clean_simple_dated_tab(hc_body_comp, "date", ["date"])
    silver_body_meas = clean_simple_dated_tab(hc_body_meas, "date", ["date"])
    silver_sleep_hr_ev = clean_simple_dated_tab(hc_sleep_hr_ev, "date")

    silver_glapp_shots = clean_simple_dated_tab(gl_shots, "date")
    silver_glapp_weight = clean_simple_dated_tab(gl_weight, "date", ["date"])
    silver_glapp_mood = clean_simple_dated_tab(gl_mood, "date", ["date"])
    silver_glapp_se = clean_simple_dated_tab(gl_se, "date")

    spine_start = str(thresholds_cfg["calendar_spine"]["start_date"])
    max_dt = _max_date_across(
        [
            sleep_silver_dict["Silver_Sleep_Daily"],
            silver_vitals,
            silver_steps,
            silver_nutrition,
            silver_body_comp,
            silver_glapp_shots,
            silver_glapp_mood,
            silver_meds,
            silver_symptoms,
            silver_labs,
        ]
    )
    if max_dt < spine_start:
        max_dt = spine_start
    spine_dates = pd.date_range(start=spine_start, end=max_dt, freq="D").strftime("%Y-%m-%d").tolist()

    silver_cycle_dim = build_silver_cycle_dim(
        hc_cycles,
        hc_menstrual,
        hc_flow_daily,
        hc_arch_mc,
        thresholds_cfg,
        qa,
        max_calendar_date=max_dt,
    )

    silver_missingness = build_missingness_provenance(
        spine_dates,
        sleep_silver_dict["Silver_Sleep_Daily"],
        sleep_silver_dict["Silver_Sleep_Sessions"],
        silver_vitals,
        silver_nutrition,
        silver_glapp_mood,
        silver_meds,
        silver_symptoms,
        silver_journal_mood,
        thresholds_cfg,
        qa,
    )

    silver_tables: dict[str, pd.DataFrame] = {
        "Silver_Sleep_Sessions": sleep_silver_dict["Silver_Sleep_Sessions"],
        "Silver_Sleep_Daily": sleep_silver_dict["Silver_Sleep_Daily"],
        "Silver_Sleep_Stage_Timeline": sleep_silver_dict["Silver_Sleep_Stage_Timeline"],
        "Silver_Sleep_HR_Events": silver_sleep_hr_ev,
        "Silver_Vitals": silver_vitals,
        "Silver_Lab_Results": silver_labs,
        "Silver_Meds_Log": silver_meds,
        "Silver_Symptoms_Log": silver_symptoms,
        "Silver_Journal_Mood": silver_journal_mood,
        "Silver_Workouts": silver_workouts,
        "Silver_Steps": silver_steps,
        "Silver_Activity_Daily": silver_activity,
        "Silver_Nutrition": silver_nutrition,
        "Silver_Nutrition_Log": silver_nutr_log,
        "Silver_HC_Weight": silver_hc_weight,
        "Silver_Body_Composition": silver_body_comp,
        "Silver_Body_Measurements": silver_body_meas,
        "Silver_Glapp_Shots": silver_glapp_shots,
        "Silver_Glapp_Weight": silver_glapp_weight,
        "Silver_Glapp_Mood": silver_glapp_mood,
        "Silver_Glapp_Side_Effects": silver_glapp_se,
        "Silver_Cycle_Dim": silver_cycle_dim,
        "Silver_Missingness": silver_missingness,
    }

    # ==================== 3. V1.5 GOLD STATISTICAL FOUNDATIONS ====================
    daily_pk_df, gold_pk_hourly = build_tirzepatide_pk_tables(
        spine_dates, silver_glapp_shots, thresholds_cfg
    )
    gold_daily_unified = build_gold_daily_unified(
        spine_dates, silver_tables, daily_pk_df, silver_missingness
    )
    gold_personal_baselines, gold_baseline_summary = build_gold_personal_baselines(
        gold_daily_unified, thresholds_cfg
    )
    gold_sleep_spikes = build_gold_sleep_hr_spikes(
        sleep_silver_dict["Silver_Sleep_Daily"],
        sleep_silver_dict["Silver_Sleep_Stage_Timeline"],
        silver_sleep_hr_ev,
    )
    gold_med_bio = build_med_bioavailability_table(
        silver_meds, silver_nutr_log, daily_pk_df, thresholds_cfg
    )
    gold_event_windows = build_gold_event_windows(
        gold_daily_unified,
        gold_sleep_spikes,
        silver_glapp_se,
        silver_symptoms,
        gold_med_bio,
    )
    gold_clinical_indices = build_gold_clinical_indices(
        silver_labs, silver_body_meas, thresholds_cfg
    )

    silver_qa_log = qa.to_dataframe()

    # Materialized Store Tables (written to Health_Intelligence_Store_v2 / Parquet+CSV)
    materialized: dict[str, pd.DataFrame] = {
        "Silver_QA_Log": silver_qa_log,
        "Silver_Sleep_Sessions": sleep_silver_dict["Silver_Sleep_Sessions"],
        "Silver_Sleep_Daily": sleep_silver_dict["Silver_Sleep_Daily"],
        "Silver_Vitals": silver_vitals,
        "Silver_Lab_Results": silver_labs,
        "Silver_Cycle_Dim": silver_cycle_dim,
        "Silver_Meds_Log": silver_meds,
        "Silver_Symptoms_Log": silver_symptoms,
        "Silver_Workouts": silver_workouts,
        "Gold_Daily_Unified": gold_daily_unified,
        "Gold_Tirzepatide_PK_Hourly": gold_pk_hourly,
        "Gold_Personal_Baselines": gold_personal_baselines,
        "Gold_Baseline_Summary": gold_baseline_summary,
        "Gold_Sleep_HR_Spikes": gold_sleep_spikes,
        "Gold_Event_Windows": gold_event_windows,
        "Gold_Med_Bioavailability": gold_med_bio,
        "Gold_Clinical_Indices": gold_clinical_indices,
    }

    if persist:
        for tab_name, table_df in materialized.items():
            adapter.write_derived_table(tab_name, table_df)

    return materialized


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run V1.5 Silver Cleaning Gate + Gold Statistical Foundations Engine."
    )
    parser.add_argument(
        "--source",
        choices=("csv", "sheets"),
        required=True,
        help="Input/output mode: offline 'csv' or live Google 'sheets'.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("./sample_data"),
        help="Directory containing source CSV files when --source csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./build_out"),
        help="Directory for exported CSV + Parquet tables.",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=None,
        help="Optional override path for config/ YAML files.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.source == "csv":
        adapter: HealthIOAdapter = LocalCSVAdapter(
            data_dir=args.data_dir, output_dir=args.output_dir
        )
    else:
        adapter = GoogleSheetsAdapter(local_mirror_dir=args.output_dir)

    outputs = run_v15_pipeline(adapter=adapter, config_dir=args.config_dir, persist=True)
    summary = ", ".join(f"{k}={len(v)}" for k, v in outputs.items())
    print(f"[health_engine v1.5] Completed successfully | {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
