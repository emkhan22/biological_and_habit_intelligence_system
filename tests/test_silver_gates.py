"""Unit tests for all 6 Silver Cleaning Gate rules."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest

from health_engine.io_adapter import LocalCSVAdapter
from health_engine.run_v15 import run_v15_pipeline


def test_sleep_unstaged_google_fit_and_short_sessions_excluded(synthetic_data_dir: Path, tmp_path: Path) -> None:
    adapter = LocalCSVAdapter(synthetic_data_dir, tmp_path / "out")
    tables = run_v15_pipeline(adapter, persist=False)

    sleep_daily = tables["Silver_Sleep_Daily"]
    sleep_sessions = tables["Silver_Sleep_Sessions"]
    qa_log = tables["Silver_QA_Log"]

    # Google Fit unstaged session on 2026-11-22 (session 700) MUST NOT appear in Silver_Sleep_Daily
    assert "2026-11-22" not in set(sleep_daily["date"])
    assert "Fit" not in set(sleep_daily["source_app"])

    # Afternoon 90m nap (session 701) flagged nap_or_partial_wear and excluded from primary daily
    nap_row = sleep_sessions.loc[sleep_sessions["session_id"] == "701"].iloc[0]
    assert bool(nap_row["nap_or_partial_wear"]) is True
    assert bool(nap_row["nocturnal_valid"]) is False

    # Primary nocturnal session on 2026-11-23 is session 702 (bedtime 23:00 -> -1.0, asleep=360m)
    s_nov23 = sleep_daily.loc[sleep_daily["date"] == "2026-11-23"].iloc[0]
    assert s_nov23["session_id"] == "702"
    assert s_nov23["asleep_min"] == pytest.approx(360.0)
    assert s_nov23["bedtime_hr_rel_mid"] == pytest.approx(-1.0)
    assert s_nov23["wake_hr"] == pytest.approx(5.5)
    assert s_nov23["sleep_eff_pct"] == pytest.approx((360.0 / 390.0) * 100.0)
    assert s_nov23["sleep_fragmentation_per_hr"] == pytest.approx(3.0 / 6.0)

    # Late bedtime on 2026-11-24 (02:00 IST -> +2.0)
    s_nov24 = sleep_daily.loc[sleep_daily["date"] == "2026-11-24"].iloc[0]
    assert s_nov24["bedtime_hr_rel_mid"] == pytest.approx(2.0)

    # QA log records excluded counts
    unstaged_qa = qa_log.loc[qa_log["rule_id"] == "SLEEP_UNSTAGED_EXCLUDED"].iloc[0]
    assert int(unstaged_qa["flagged_or_excluded_rows"]) == 1
    short_qa = qa_log.loc[qa_log["rule_id"] == "SLEEP_SHORT_SESSION_EXCLUDED"].iloc[0]
    assert int(short_qa["flagged_or_excluded_rows"]) == 1


def test_vitals_hour_coverage_consistent_across_pre_and_post_may2026(
    synthetic_data_dir: Path, tmp_path: Path
) -> None:
    adapter = LocalCSVAdapter(synthetic_data_dir, tmp_path / "out")
    tables = run_v15_pipeline(adapter, persist=False)

    vit = tables["Silver_Vitals"].set_index("date")
    gold = tables["Gold_Daily_Unified"].set_index("date")

    # Pre-May 2026 (only 1,180 samples, 20 hours covered) -> MUST PASS
    assert bool(vit.loc["2026-04-15", "vitals_valid_coverage"]) is True
    assert vit.loc["2026-04-15", "sampling_era"] == "pre_may2026_1min"
    assert int(vit.loc["2026-04-15", "hr_hours_covered"]) == 20
    assert gold.loc["2026-04-15", "resting_hr"] == pytest.approx(66.0)

    # Post-May 2026 (29,800 samples, 19 hours covered) -> MUST PASS
    assert bool(vit.loc["2026-11-23", "vitals_valid_coverage"]) is True
    assert vit.loc["2026-11-23", "sampling_era"] == "may2026_high_freq"
    assert gold.loc["2026-11-23", "resting_hr"] == pytest.approx(69.0)

    # Post-May 2026 high sample count (24,000 samples) in only 10 hours -> MUST FAIL & mask Gold vitals
    assert bool(vit.loc["2026-11-24", "vitals_valid_coverage"]) is False
    assert int(vit.loc["2026-11-24", "hr_hours_covered"]) == 10
    assert np.isnan(gold.loc["2026-11-24", "resting_hr"])
    assert np.isnan(gold.loc["2026-11-24", "hrv_rmssd_ms"])


def test_lab_unit_normalization_immutability_and_autodate_quarantine(
    synthetic_data_dir: Path, tmp_path: Path
) -> None:
    adapter = LocalCSVAdapter(synthetic_data_dir, tmp_path / "out")
    tables = run_v15_pipeline(adapter, persist=False)

    labs = tables["Silver_Lab_Results"].set_index("Record_ID")

    # Platelet Count: 250000 cells/cumm -> 250.0 10^3/µL; raw columns strictly untouched
    plt = labs.loc["LAB-001"]
    assert plt["Original_Value"] == "250000"
    assert plt["Numeric_Value"] == "250000"
    assert plt["Unit"] == "cells/cumm"
    assert plt["Standardized_Unit"] == "cells/µL"
    assert plt["Normalized_Value"] == pytest.approx(250.0)
    assert plt["Normalized_Unit"] == "10^3/µL"
    assert plt["Normalization_Rule_ID"] == "PLT_CELLS_UL_TO_10E3_UL"
    assert bool(plt["Normalization_Valid_Flag"]) is True

    # WBC Count: 7400 cells/µL -> 7.4 10^3/µL
    wbc = labs.loc["LAB-002"]
    assert wbc["Original_Value"] == "7400"
    assert wbc["Normalized_Value"] == pytest.approx(7.4)
    assert wbc["Normalized_Unit"] == "10^3/µL"

    # Total T3: 115 ng/dL -> 1.15 ng/mL
    tt3 = labs.loc["LAB-003"]
    assert tt3["Original_Value"] == "115"
    assert tt3["Normalized_Value"] == pytest.approx(1.15)
    assert tt3["Normalized_Unit"] == "ng/mL"

    # Auto-date corrupted urinalysis ("2026-05-02") quarantined
    corrupt = labs.loc["LAB-020"]
    assert bool(corrupt["data_corruption_flag"]) is True
    assert bool(corrupt["Normalization_Valid_Flag"]) is False
    assert np.isnan(corrupt["Normalized_Value"])


def test_deduplication_and_stitched_cycle_dim(synthetic_data_dir: Path, tmp_path: Path) -> None:
    adapter = LocalCSVAdapter(synthetic_data_dir, tmp_path / "out")
    tables = run_v15_pipeline(adapter, persist=False)

    # Meds_Log duplicate Thyronorm removed (5 -> 4)
    assert len(tables["Silver_Meds_Log"]) == 4

    # Symptoms_Log singular/plural normalized without mutating original Symptom (5 -> 4)
    sym = tables["Silver_Symptoms_Log"]
    assert len(sym) == 4
    foot_rows = sym.loc[sym["Normalized_Symptom"] == "Right Foot Cramp"]
    assert len(foot_rows) == 1
    assert foot_rows.iloc[0]["Symptom"] == "Right Foot Cramps"

    # Workouts duplicate removed (2 -> 1)
    assert len(tables["Silver_Workouts"]) == 1

    # Stitched cycle dimension: Flo <= 2026-11-05, Fitbit_HC >= 2026-11-06, Spotting preserved
    cyc = tables["Silver_Cycle_Dim"].set_index("date")
    assert cyc.loc["2026-10-20", "source_owner"] == "Flo"
    assert bool(cyc.loc["2026-10-20", "is_spotting_day"]) is True
    assert bool(cyc.loc["2026-10-20", "is_period_day"]) is False

    assert cyc.loc["2026-11-05", "source_owner"] == "Flo"
    assert bool(cyc.loc["2026-11-05", "is_period_day"]) is True
    assert cyc.loc["2026-11-06", "source_owner"] == "Fitbit_HC"
    assert bool(cyc.loc["2026-11-06", "is_period_day"]) is True
    assert cyc.loc["2026-11-06", "cycle_phase"] == "Menstrual"
    assert int(cyc.loc["2026-11-06", "cycle_day"]) == 3
