"""CLI Entrypoint & Pipeline Orchestrator for V2.0 Statistical Inference Engine.

Supports:
  python -m health_engine.run_v20 --source csv --data-dir ./sample_data --output-dir ./build_out
  python -m health_engine.run_v20 --source sheets --output-dir ./build_out
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from health_engine.config_loader import load_engine_configs, load_treatments_config
from health_engine.io_adapter import GoogleSheetsAdapter, HealthIOAdapter, LocalCSVAdapter
from health_engine.run_v15 import run_v15_with_silver_bundle
from health_engine.stats.cardiac_efficiency import build_gold_cardiac_efficiency
from health_engine.stats.correlations import build_gold_stats_correlations
from health_engine.stats.cycle_stats import build_gold_cycle_phase_stats
from health_engine.stats.episode_chains import build_gold_episode_chains
from health_engine.stats.event_study import build_gold_intervention_events
from health_engine.stats.maturity import StatsAuditLogger
from health_engine.stats.tirzepatide_wave import build_gold_tirz_folded_wave
from health_engine.stats.treatment_scorecards import build_gold_treatment_scorecards
from health_engine.stats.weight_alignment import (
    build_canonical_weighins,
    build_gold_weight_clinical_link,
    build_gold_weight_intervals,
    build_gold_weight_trend,
)


def run_v20_pipeline(
    adapter: HealthIOAdapter,
    config_dir: Path | None = None,
    persist: bool = True,
    persist_v15: bool = True,
) -> dict[str, pd.DataFrame]:
    """Execute V1.5 Silver+Gold foundations followed by all V2.0 Statistical Inference modules."""
    thresholds_cfg, _ = load_engine_configs(config_dir)
    treatments_cfg = load_treatments_config(config_dir)
    audit_logger = StatsAuditLogger()

    # 1. Obtain validated Silver & V1.5 Gold tables strictly via Silver Cleaning Gate
    v15_tables, silver_tables, qa = run_v15_with_silver_bundle(
        adapter=adapter, config_dir=config_dir, persist=persist_v15
    )
    qa_count_before_v20 = len(qa.records)

    gold_daily = v15_tables["Gold_Daily_Unified"]
    gold_sleep_spikes = v15_tables["Gold_Sleep_HR_Spikes"]
    gold_event_windows = v15_tables["Gold_Event_Windows"]
    gold_clinical_indices = v15_tables["Gold_Clinical_Indices"]

    # 2. Exertion-Adjusted Heart Rate (runs first to supply daily cardiac_efficiency_residual)
    gold_cardiac_eff, gold_cardiac_buckets = build_gold_cardiac_efficiency(
        gold_daily=gold_daily,
        silver_hr_hourly=silver_tables["Silver_Heart_Rate_Hourly"],
        silver_steps_hourly=silver_tables["Silver_Steps_Hourly"],
        thresholds_cfg=thresholds_cfg,
        audit_logger=audit_logger,
    )

    # 3. Module A: 3-Grain Weight-Loss Alignment Engine
    canonical_weighins = build_canonical_weighins(
        silver_body_comp=silver_tables["Silver_Body_Composition"],
        silver_weight=silver_tables["Silver_Weight"],
        silver_glapp_weight=silver_tables["Silver_Glapp_Weight"],
        silver_nutr_log=silver_tables["Silver_Nutrition_Log"],
        thresholds_cfg=thresholds_cfg,
        qa=qa,
    )
    gold_weight_intervals, gold_weight_interval_stats = build_gold_weight_intervals(
        canonical_weighins=canonical_weighins,
        gold_daily=gold_daily,
        gold_sleep_spikes=gold_sleep_spikes,
        gold_event_windows=gold_event_windows,
        silver_symptoms=silver_tables["Silver_Symptoms_Log"],
        silver_glapp_se=silver_tables["Silver_Glapp_Side_Effects"],
        thresholds_cfg=thresholds_cfg,
        audit_logger=audit_logger,
    )
    gold_weight_trend = build_gold_weight_trend(
        canonical_weighins=canonical_weighins,
        gold_daily=gold_daily,
        cardiac_daily=gold_cardiac_eff,
        silver_cycles=silver_tables["Silver_Cycles"],
        thresholds_cfg=thresholds_cfg,
        audit_logger=audit_logger,
    )
    gold_weight_clinical_link = build_gold_weight_clinical_link(
        canonical_weighins=canonical_weighins,
        silver_labs=silver_tables["Silver_Lab_Results"],
        gold_clinical_indices=gold_clinical_indices,
        thresholds_cfg=thresholds_cfg,
        audit_logger=audit_logger,
    )

    # 4. Module B: Dual-Baseline Per-Treatment & Per-Dose Scorecards
    gold_treatment_scorecards = build_gold_treatment_scorecards(
        gold_daily=gold_daily,
        gold_weight_intervals=gold_weight_intervals,
        gold_sleep_spikes=gold_sleep_spikes,
        gold_event_windows=gold_event_windows,
        cardiac_daily=gold_cardiac_eff,
        silver_tables=silver_tables,
        thresholds_cfg=thresholds_cfg,
        treatments_cfg=treatments_cfg,
        audit_logger=audit_logger,
    )

    # 5. Event-Window & Confounder-Adjusted Correlations
    gold_stats_correlations = build_gold_stats_correlations(
        gold_daily=gold_daily,
        gold_event_windows=gold_event_windows,
        thresholds_cfg=thresholds_cfg,
        audit_logger=audit_logger,
    )

    # 6. 7-Day Folded Tirzepatide Descriptive Wave
    gold_tirz_wave = build_gold_tirz_folded_wave(
        gold_daily=gold_daily,
        gold_sleep_spikes=gold_sleep_spikes,
        gold_event_windows=gold_event_windows,
        thresholds_cfg=thresholds_cfg,
        audit_logger=audit_logger,
    )

    # 7. 3-Tier Intervention Event Study
    gold_intervention_events = build_gold_intervention_events(
        gold_daily=gold_daily,
        cardiac_daily=gold_cardiac_eff,
        thresholds_cfg=thresholds_cfg,
        audit_logger=audit_logger,
    )

    # 8. Cycle-Phase Biometrics & Regularity
    gold_cycle_phase_stats = build_gold_cycle_phase_stats(
        gold_daily=gold_daily,
        silver_cycles=silver_tables["Silver_Cycles"],
        silver_cycle_dim=silver_tables["Silver_Cycle_Dim"],
        silver_labs=silver_tables["Silver_Lab_Results"],
        thresholds_cfg=thresholds_cfg,
        audit_logger=audit_logger,
    )

    # 9. Micro Journal Episode Chains (Non-Smoking)
    gold_episode_chains = build_gold_episode_chains(
        gold_daily=gold_daily,
        silver_meds=silver_tables["Silver_Meds_Log"],
        silver_symptoms=silver_tables["Silver_Symptoms_Log"],
        silver_journal_mood=silver_tables["Silver_Journal_Mood"],
        silver_nutr_log=silver_tables["Silver_Nutrition_Log"],
        thresholds_cfg=thresholds_cfg,
        audit_logger=audit_logger,
    )

    stats_run_audit = audit_logger.to_dataframe()
    updated_qa_log = qa.to_dataframe()

    v20_only_tables: dict[str, pd.DataFrame] = {
        "Gold_Weight_Intervals": gold_weight_intervals,
        "Gold_Weight_Interval_Stats": gold_weight_interval_stats,
        "Gold_Weight_Trend": gold_weight_trend,
        "Gold_Weight_Clinical_Link": gold_weight_clinical_link,
        "Gold_Treatment_Scorecards": gold_treatment_scorecards,
        "Gold_Stats_Correlations": gold_stats_correlations,
        "Gold_Tirz_Folded_Wave": gold_tirz_wave,
        "Gold_Intervention_Events": gold_intervention_events,
        "Gold_Cardiac_Efficiency": gold_cardiac_eff,
        "Gold_Cardiac_Efficiency_Buckets": gold_cardiac_buckets,
        "Gold_Cycle_Phase_Stats": gold_cycle_phase_stats,
        "Gold_Episode_Chains": gold_episode_chains,
        "Stats_Run_Audit": stats_run_audit,
    }

    if persist:
        if len(qa.records) > qa_count_before_v20:
            adapter.write_derived_table("Silver_QA_Log", updated_qa_log)
        for tab_name, table_df in v20_only_tables.items():
            adapter.write_derived_table(tab_name, table_df)

    merged_outputs = {
        **v15_tables,
        "Silver_QA_Log": updated_qa_log,
        "Gold_Canonical_WeighIns": canonical_weighins,
        **v20_only_tables,
    }
    return merged_outputs


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run V2.0 Statistical Inference Engine on top of V1.5 Silver & Gold."
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
    parser.add_argument(
        "--skip-v15-persist",
        action="store_true",
        help="Skip re-uploading V1.5 tabs when run immediately after run_v15.",
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

    outputs = run_v20_pipeline(
        adapter=adapter,
        config_dir=args.config_dir,
        persist=True,
        persist_v15=not args.skip_v15_persist,
    )
    v20_keys = [
        "Gold_Weight_Intervals",
        "Gold_Weight_Interval_Stats",
        "Gold_Weight_Trend",
        "Gold_Weight_Clinical_Link",
        "Gold_Treatment_Scorecards",
        "Gold_Stats_Correlations",
        "Gold_Tirz_Folded_Wave",
        "Gold_Intervention_Events",
        "Gold_Cardiac_Efficiency",
        "Gold_Cycle_Phase_Stats",
        "Gold_Episode_Chains",
        "Stats_Run_Audit",
    ]
    summary = ", ".join(f"{k}={len(outputs[k])}" for k in v20_keys if k in outputs)
    print(f"[health_engine v2.0] Completed successfully | {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
