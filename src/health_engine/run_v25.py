"""CLI Entrypoint & Pipeline Orchestrator for V2.5 Progressive ML Engine.

Supports:
  python -m health_engine.run_v25 --source csv --data-dir ./sample_data --output-dir ./build_out
  python -m health_engine.run_v25 --source sheets --output-dir ./build_out --reuse-local-mirror
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from health_engine.config_loader import load_engine_configs, load_journal_themes_config
from health_engine.io_adapter import GoogleSheetsAdapter, HealthIOAdapter, LocalCSVAdapter
from health_engine.ml.clustering import build_gold_phenotypes
from health_engine.ml.evaluator import ModelAuditLogger
from health_engine.ml.journal_nlp import build_gold_journal_nlp
from health_engine.ml.regressors import build_gold_ml_regressors
from health_engine.run_v15 import load_cached_v15_bundle
from health_engine.run_v20 import run_v20_pipeline


def _load_cached_v20_weight_intervals(output_dir: Path | None) -> pd.DataFrame | None:
    if output_dir is None:
        return None
    pq = Path(output_dir) / "Gold_Weight_Intervals.parquet"
    if pq.exists():
        return pd.read_parquet(pq)
    return None


def run_v25_pipeline(
    adapter: HealthIOAdapter,
    config_dir: Path | None = None,
    persist: bool = True,
    persist_upstream: bool = True,
    reuse_mirror_dir: Path | None = None,
    cache_output_dir: Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Execute V2.5 Progressive ML Engine on top of V1.5 Silver/Gold and V2.0 Weight Intervals."""
    thresholds_cfg, _ = load_engine_configs(config_dir)
    themes_cfg = load_journal_themes_config(config_dir)
    audit_logger = ModelAuditLogger(
        data_cleaning_version="v1.5_silver_gold",
        model_version=str(thresholds_cfg.get("engine_version", "v2.5.0")),
    )

    cached_v15 = load_cached_v15_bundle(reuse_mirror_dir) if reuse_mirror_dir is not None else None
    cached_wi = _load_cached_v20_weight_intervals(reuse_mirror_dir)

    if cached_v15 is not None and cached_wi is not None:
        v15_tables, silver_tables, _ = cached_v15
        upstream_tables = {**v15_tables, "Gold_Weight_Intervals": cached_wi}
    else:
        upstream_tables = run_v20_pipeline(
            adapter=adapter,
            config_dir=config_dir,
            persist=persist_upstream,
            persist_v15=persist_upstream,
            cache_output_dir=cache_output_dir,
        )
        # Reload Silver bundle via cached or in-memory V1.5 run for Micro Journal NLP
        check_dir = reuse_mirror_dir or cache_output_dir
        cached_after = load_cached_v15_bundle(check_dir) if check_dir is not None else None
        if cached_after is not None:
            _, silver_tables, _ = cached_after
        else:
            from health_engine.run_v15 import run_v15_with_silver_bundle

            _, silver_tables, _ = run_v15_with_silver_bundle(
                adapter=adapter, config_dir=config_dir, persist=False
            )

    gold_daily = upstream_tables["Gold_Daily_Unified"]
    gold_event_windows = upstream_tables["Gold_Event_Windows"]
    gold_weight_intervals = upstream_tables["Gold_Weight_Intervals"]

    # 1. Micro Journal Structured NLP Themes & Episode Co-Occurrence Rules (+ enrich Gold_Daily_Unified)
    gold_journal_themes, gold_episode_rules, enriched_gold_daily = build_gold_journal_nlp(
        gold_daily=gold_daily,
        silver_journal_mood=silver_tables.get("Silver_Journal_Mood", pd.DataFrame()),
        silver_meds=silver_tables.get("Silver_Meds_Log", pd.DataFrame()),
        silver_symptoms=silver_tables.get("Silver_Symptoms_Log", pd.DataFrame()),
        silver_nutr_log=silver_tables.get("Silver_Nutrition_Log", pd.DataFrame()),
        thresholds_cfg=thresholds_cfg,
        themes_cfg=themes_cfg,
        audit_logger=audit_logger,
    )

    # 2. Supervised Predictive Regression Suite (Models 1A, 1B, 2A, 2B, 3, 4)
    gold_ml_summary, gold_ml_drivers = build_gold_ml_regressors(
        gold_daily=enriched_gold_daily,
        gold_event_windows=gold_event_windows,
        gold_weight_intervals=gold_weight_intervals,
        thresholds_cfg=thresholds_cfg,
        audit_logger=audit_logger,
    )

    # 3. Unsupervised Physiological State Phenotyping (K-Means k=3 + 25x ARI + Post-Hoc Labels)
    gold_daily_phenotypes, gold_phenotype_profiles = build_gold_phenotypes(
        gold_daily=enriched_gold_daily,
        thresholds_cfg=thresholds_cfg,
        audit_logger=audit_logger,
    )

    model_run_audit = audit_logger.to_dataframe()

    v25_tables: dict[str, pd.DataFrame] = {
        "Gold_Daily_Unified": enriched_gold_daily,
        "Gold_ML_Model_Summary": gold_ml_summary,
        "Gold_ML_Feature_Drivers": gold_ml_drivers,
        "Gold_Daily_Phenotypes": gold_daily_phenotypes,
        "Gold_Phenotype_Profiles": gold_phenotype_profiles,
        "Gold_Journal_Themes": gold_journal_themes,
        "Gold_Episode_Rules": gold_episode_rules,
        "Model_Run_Audit": model_run_audit,
    }

    if persist:
        for tab_name, table_df in v25_tables.items():
            adapter.write_derived_table(tab_name, table_df)

    return {**upstream_tables, **v25_tables}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run V2.5 Progressive ML Engine on top of V1.5 Silver/Gold and V2.0 Inference."
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
        "--skip-upstream-persist",
        action="store_true",
        help="Skip re-uploading V1.5 & V2.0 tabs when run immediately after run_v20.",
    )
    parser.add_argument(
        "--reuse-local-mirror",
        action="store_true",
        help="Reuse cached V1.5 & V2.0 Parquet tables in --output-dir without re-reading Bronze Sheets.",
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

    outputs = run_v25_pipeline(
        adapter=adapter,
        config_dir=args.config_dir,
        persist=True,
        persist_upstream=not (args.skip_upstream_persist or args.reuse_local_mirror),
        reuse_mirror_dir=args.output_dir if args.reuse_local_mirror else None,
        cache_output_dir=args.output_dir,
    )
    v25_keys = [
        "Gold_ML_Model_Summary",
        "Gold_ML_Feature_Drivers",
        "Gold_Daily_Phenotypes",
        "Gold_Phenotype_Profiles",
        "Gold_Journal_Themes",
        "Gold_Episode_Rules",
        "Model_Run_Audit",
    ]
    summary = ", ".join(f"{k}={len(outputs[k])}" for k in v25_keys if k in outputs)
    print(f"[health_engine v2.5] Completed successfully | {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
