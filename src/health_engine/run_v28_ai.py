"""CLI Entrypoint & Orchestrator for V2.8 Model-Routed AI Synthesis Layer (`run_v28_ai.py`).

Supports:
  python -m health_engine.run_v28_ai --source csv --data-dir ./sample_data --output-dir ./build_out --daily --dry-run
  python -m health_engine.run_v28_ai --source sheets --output-dir ./build_out --reuse-local-mirror --daily --weekly
  python -m health_engine.run_v28_ai --source sheets --output-dir ./build_out --reuse-local-mirror --doctor-prep Endocrinology
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from health_engine.ai.enricher_sweets import run_module_1d_sweet_classifier
from health_engine.ai.modules import (
    VALID_SPECIALTIES,
    run_module_1b_journal_enricher,
    run_module_1c_meal_auditor,
    run_module_2a_daily_brief,
    run_module_2b_weekly_review,
    run_module_2c_monthly_review,
    run_module_3a_peer_expectedness,
    run_module_3b_doctor_prep,
    run_module_3c_research_pulse,
)
from health_engine.ai.packet_builder import build_ai_briefing_packet, serialize_briefing_packet
from health_engine.ai.router import ModelRouter
from health_engine.config_loader import load_ai_routing_config
from health_engine.io_adapter import GoogleSheetsAdapter, HealthIOAdapter, LocalCSVAdapter
from health_engine.run_v15 import load_cached_v15_bundle, run_v15_with_silver_bundle
from health_engine.run_v25 import run_v25_pipeline

UPSTREAM_TABLE_NAMES: tuple[str, ...] = (
    "Gold_Daily_Unified",
    "Gold_Personal_Baselines",
    "Gold_Baseline_Summary",
    "Gold_Sleep_HR_Spikes",
    "Gold_Event_Windows",
    "Gold_Med_Bioavailability",
    "Gold_Clinical_Indices",
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
    "Gold_Habit_Durability",
    "Gold_Sweet_Hunger_Analysis",
    "Gold_Calorie_Balance_Audit",
    "Stats_Run_Audit",
    "Gold_ML_Model_Summary",
    "Gold_ML_Feature_Drivers",
    "Gold_Daily_Phenotypes",
    "Gold_Phenotype_Profiles",
    "Gold_Journal_Themes",
    "Gold_Episode_Rules",
    "Model_Run_Audit",
)


def _load_upstream_from_mirror(mirror_dir: Path | None) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]] | None:
    if mirror_dir is None or not mirror_dir.exists():
        return None
    cached_v15 = load_cached_v15_bundle(mirror_dir)
    if cached_v15 is None:
        return None
    _, silver_tables, _ = cached_v15
    tables: dict[str, pd.DataFrame] = {}
    for tname in UPSTREAM_TABLE_NAMES:
        pq = mirror_dir / f"{tname}.parquet"
        csv = mirror_dir / f"{tname}.csv"
        if pq.exists():
            tables[tname] = pd.read_parquet(pq)
        elif csv.exists():
            tables[tname] = pd.read_csv(csv)
    if "Gold_Daily_Unified" not in tables:
        return None
    return tables, silver_tables


def run_v28_ai_pipeline(
    adapter: HealthIOAdapter,
    *,
    config_dir: Path | None = None,
    reuse_mirror_dir: Path | None = None,
    packet_output_dir: Path | None = None,
    run_daily: bool = True,
    run_weekly: bool = False,
    backfill_weeks: bool = False,
    run_monthly: bool = False,
    doctor_prep_specialty: str | None = None,
    dry_run: bool = False,
    persist: bool = True,
    persist_upstream: bool = False,
    injected_client: Any | None = None,
) -> dict[str, Any]:
    """Execute requested V2.8 AI synthesis cadences and persist outputs + `AI_Cost_Ledger`."""
    routing_cfg = load_ai_routing_config(config_dir)

    loaded = _load_upstream_from_mirror(reuse_mirror_dir)
    if loaded is not None:
        upstream_tables, silver_tables = loaded
    else:
        upstream_tables = run_v25_pipeline(
            adapter=adapter,
            config_dir=config_dir,
            persist=persist_upstream,
            persist_upstream=persist_upstream,
            reuse_mirror_dir=reuse_mirror_dir,
        )
        cached_v15 = load_cached_v15_bundle(reuse_mirror_dir) if reuse_mirror_dir is not None else None
        if cached_v15 is not None:
            _, silver_tables, _ = cached_v15
        else:
            _, silver_tables, _ = run_v15_with_silver_bundle(
                adapter=adapter, config_dir=config_dir, persist=False
            )

    existing_ledger = adapter.read_derived_table("AI_Cost_Ledger")
    router = ModelRouter(
        routing_cfg=routing_cfg,
        ledger_df=existing_ledger,
        client=injected_client,
        dry_run=dry_run,
    )

    # Assemble & validate strictly smoking-free AI_Briefing_Packet
    packet = build_ai_briefing_packet(upstream_tables, silver_tables)
    packet_json_str = serialize_briefing_packet(packet)
    if packet_output_dir is not None:
        packet_output_dir.mkdir(parents=True, exist_ok=True)
        (packet_output_dir / "AI_Briefing_Packet.json").write_text(packet_json_str, encoding="utf-8")

    materialized_ai_tabs: dict[str, pd.DataFrame] = {}

    if run_daily:
        # 1B: Incremental Micro Journal Hinglish/English Semantic Enricher
        existing_1b = adapter.read_derived_table("Gold_Journal_Themes_AI")
        df_1b = run_module_1b_journal_enricher(
            silver_journal_mood=silver_tables.get("Silver_Journal_Mood"),
            silver_symptoms=silver_tables.get("Silver_Symptoms_Log"),
            det_themes_df=upstream_tables.get("Gold_Journal_Themes"),
            existing_ai_df=existing_1b,
            router=router,
        )
        materialized_ai_tabs["Gold_Journal_Themes_AI"] = df_1b

        # 1C: Incremental Home-Cooked Meal & Bioavailability Auditor
        existing_1c = adapter.read_derived_table("Gold_Meal_Bioavailability_AI")
        df_1c = run_module_1c_meal_auditor(
            silver_nutr_log=silver_tables.get("Silver_Nutrition_Log"),
            silver_meds_log=silver_tables.get("Silver_Meds_Log"),
            gold_daily=upstream_tables.get("Gold_Daily_Unified"),
            gold_event_windows=upstream_tables.get("Gold_Event_Windows"),
            existing_ai_df=existing_1c,
            router=router,
        )
        materialized_ai_tabs["Gold_Meal_Bioavailability_AI"] = df_1c

        # 1D: Minimal-cost ambiguous sweet-meal classifier (cached by normalized_meal_name)
        existing_1d = adapter.read_derived_table("Gold_Sweet_Meal_Tags_AI")
        df_1d = run_module_1d_sweet_classifier(
            silver_nutr_log=silver_tables.get("Silver_Nutrition_Log"),
            existing_tags_df=existing_1d,
            router=router,
        )
        materialized_ai_tabs["Gold_Sweet_Meal_Tags_AI"] = df_1d

        # 2A: Daily Holistic Brief & Coaching Nudges
        existing_2a = adapter.read_derived_table("AI_Insights_Cache")
        df_2a = run_module_2a_daily_brief(
            packet=packet,
            existing_cache_df=existing_2a,
            router=router,
        )
        materialized_ai_tabs["AI_Insights_Cache"] = df_2a

    if run_weekly or backfill_weeks:
        # 2B: Friday Weekly Deep Review (Thu-Wed)
        existing_2b = adapter.read_derived_table("AI_Weekly_Briefs")
        df_2b = run_module_2b_weekly_review(
            tables=upstream_tables,
            silver_tables=silver_tables,
            existing_weekly_df=existing_2b,
            router=router,
            backfill_weeks=backfill_weeks,
        )
        materialized_ai_tabs["AI_Weekly_Briefs"] = df_2b

    if run_weekly:
        # 3A: Weekly Peer-Expectedness & Regimen Safety Check
        existing_3a = adapter.read_derived_table("AI_Peer_Expectedness")
        df_3a = run_module_3a_peer_expectedness(
            packet=packet,
            existing_peer_df=existing_3a,
            router=router,
        )
        materialized_ai_tabs["AI_Peer_Expectedness"] = df_3a

        # 3C: Weekly GLP-1 & Metabolic Research Pulse
        existing_3c = adapter.read_derived_table("AI_Research_Pulse")
        df_3c = run_module_3c_research_pulse(
            packet=packet,
            existing_pulse_df=existing_3c,
            router=router,
        )
        materialized_ai_tabs["AI_Research_Pulse"] = df_3c

    if run_monthly:
        # 2C: Monthly Whole-Body Clinical Review
        existing_2c = adapter.read_derived_table("AI_Monthly_Briefs")
        df_2c = run_module_2c_monthly_review(
            packet=packet,
            existing_monthly_df=existing_2c,
            router=router,
        )
        materialized_ai_tabs["AI_Monthly_Briefs"] = df_2c

    if doctor_prep_specialty:
        # 3B: On-Demand Specialist Doctor Visit Prep
        existing_3b = adapter.read_derived_table("AI_Doctor_Briefs")
        df_3b = run_module_3b_doctor_prep(
            specialty=doctor_prep_specialty,
            packet=packet,
            existing_doc_df=existing_3b,
            router=router,
        )
        materialized_ai_tabs["AI_Doctor_Briefs"] = df_3b

    updated_ledger = router.breaker.to_dataframe()
    materialized_ai_tabs["AI_Cost_Ledger"] = updated_ledger

    if persist:
        for tab_name, tbl_df in materialized_ai_tabs.items():
            adapter.write_derived_table(tab_name, tbl_df)

    return {
        "packet": packet,
        "tables": materialized_ai_tabs,
        "mtd_spend_inr": router.breaker.month_to_date_spend_inr(),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run V2.8 Model-Routed AI Synthesis Layer on top of V1.5/V2.0/V2.5 Store."
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
        help="Directory for local CSV/Parquet mirrors and AI_Briefing_Packet.json.",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=None,
        help="Optional override directory for config/ YAML files.",
    )
    parser.add_argument(
        "--reuse-local-mirror",
        action="store_true",
        help="Reuse cached V1.5/V2.0/V2.5 Parquet tables in --output-dir without re-reading Bronze Sheets.",
    )
    parser.add_argument("--daily", action="store_true", help="Run daily AI modules (1B, 1C, 2A).")
    parser.add_argument("--weekly", action="store_true", help="Run Friday weekly AI modules (2B, 3A, 3C).")
    parser.add_argument(
        "--backfill-weeks",
        action="store_true",
        help="Backfill historical Thu-Wed weekly reviews since 2026-10-21 (Module 2B).",
    )
    parser.add_argument("--monthly", action="store_true", help="Run monthly whole-body clinical review (2C).")
    parser.add_argument(
        "--doctor-prep",
        choices=VALID_SPECIALTIES,
        default=None,
        help="Generate on-demand specialist doctor visit brief (Module 3B).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Execute full packet assembly, privacy checks, and deterministic Python fallbacks with zero API charges.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    # Default to --daily if no cadence flag was explicitly supplied
    any_cadence = any(
        [args.daily, args.weekly, args.backfill_weeks, args.monthly, bool(args.doctor_prep)]
    )
    run_daily = args.daily or (not any_cadence)

    if args.source == "csv":
        adapter: HealthIOAdapter = LocalCSVAdapter(
            data_dir=args.data_dir, output_dir=args.output_dir
        )
    else:
        adapter = GoogleSheetsAdapter(local_mirror_dir=args.output_dir)

    result = run_v28_ai_pipeline(
        adapter=adapter,
        config_dir=args.config_dir,
        reuse_mirror_dir=args.output_dir if args.reuse_local_mirror else None,
        packet_output_dir=args.output_dir,
        run_daily=run_daily,
        run_weekly=args.weekly,
        backfill_weeks=args.backfill_weeks,
        run_monthly=args.monthly,
        doctor_prep_specialty=args.doctor_prep,
        dry_run=args.dry_run,
        persist=True,
        persist_upstream=False,
    )
    tabs = result["tables"]
    summary = ", ".join(f"{k}={len(v)}" for k, v in tabs.items())
    print(
        f"[health_engine v2.8 ai] Completed successfully | mtd_spend_inr={result['mtd_spend_inr']:.2f} | {summary}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
