"""Unit tests for V1.5 Gold Statistical Foundations & Clinical Screening Indices."""

from __future__ import annotations

import math
from pathlib import Path
import numpy as np
import pytest

from health_engine.config_loader import EPISTEMIC_CLASSES
from health_engine.io_adapter import LocalCSVAdapter
from health_engine.run_v15 import run_v15_pipeline


def test_sleep_hr_spikes_divide_by_stage_specific_duration(
    synthetic_data_dir: Path, tmp_path: Path
) -> None:
    adapter = LocalCSVAdapter(synthetic_data_dir, tmp_path / "out")
    tables = run_v15_pipeline(adapter, persist=False)

    spikes = tables["Gold_Sleep_HR_Spikes"].set_index("date")
    row = spikes.loc["2026-11-23"]

    # Session 702: asleep=360m (6h), light=180m (3h), deep=60m (1h), rem=120m (2h)
    # Events: 3 Light, 1 Deep, 2 REM, 2 Wake/Transition (Awake + Light / Awake) = 8 total
    assert int(row["total_spikes"]) == 8
    assert int(row["wake_transition_spike_cnt"]) == 2
    assert row["spikes_per_asleep_hr"] == pytest.approx(8.0 / 6.0, rel=1e-3)

    # Stage-specific denominators: 3/3h = 1.0, 2/2h = 1.0, 1/1h = 1.0 (NOT divided by 6h!)
    assert row["light_spikes_per_light_hr"] == pytest.approx(3.0 / 3.0)
    assert row["rem_spikes_per_rem_hr"] == pytest.approx(2.0 / 2.0)
    assert row["deep_spikes_per_deep_hr"] == pytest.approx(1.0 / 1.0)
    assert row["max_peak_bpm"] == pytest.approx(96.0)
    assert row["total_elevated_sec"] == pytest.approx(360.0)


def test_homa_ir_and_tyg_strict_fasting_and_same_date_enforcement(
    synthetic_data_dir: Path, tmp_path: Path
) -> None:
    adapter = LocalCSVAdapter(synthetic_data_dir, tmp_path / "out")
    tables = run_v15_pipeline(adapter, persist=False)

    clin = tables["Gold_Clinical_Indices"]

    # 2026-10-10 valid same-day fasting draw MUST produce HOMA_IR and TyG_Index
    homa_rows = clin.loc[clin["index_name"] == "HOMA_IR"]
    tyg_rows = clin.loc[clin["index_name"] == "TyG_Index"]

    assert list(homa_rows["date"]) == ["2026-10-10"]
    assert list(tyg_rows["date"]) == ["2026-10-10"]
    # Non-fasting 2026-10-15 and mismatched 2026-10-18/19 strictly rejected!
    assert "2026-10-15" not in set(homa_rows["date"])
    assert "2026-10-18" not in set(homa_rows["date"])
    assert "2026-10-19" not in set(homa_rows["date"])

    assert homa_rows.iloc[0]["index_value"] == pytest.approx((95.0 * 16.2) / 405.0, rel=1e-3)
    assert tyg_rows.iloc[0]["index_value"] == pytest.approx(math.log((150.0 * 95.0) / 2.0), rel=1e-3)

    # Verify FIB-4 & APRI used normalized Platelets (250.0 10^3/µL, NOT raw 250000) & carry <35y caveat
    apri_row = clin.loc[(clin["date"] == "2026-10-10") & (clin["index_name"] == "APRI")].iloc[0]
    assert apri_row["index_value"] == pytest.approx(((28.0 / 40.0) / 250.0) * 100.0, rel=1e-3)
    assert "Reduced diagnostic accuracy under age 35" in apri_row["clinical_caveat"]

    fib4_row = clin.loc[(clin["date"] == "2026-10-10") & (clin["index_name"] == "FIB4")].iloc[0]
    assert 0.1 < float(fib4_row["index_value"]) < 2.0
    assert "Reduced diagnostic accuracy under age 35" in fib4_row["clinical_caveat"]

    mentzer_row = clin.loc[(clin["date"] == "2026-10-10") & (clin["index_name"] == "Mentzer_Index")].iloc[0]
    assert mentzer_row["index_value"] == pytest.approx(84.0 / 4.8, rel=1e-3)
    assert "Microcytosis screening context only" in mentzer_row["clinical_caveat"]

    # Anthropometric WHtR uses height = 163.0 cm
    whtr_row = clin.loc[(clin["date"] == "2026-11-22") & (clin["index_name"] == "WHtR")].iloc[0]
    assert whtr_row["index_value"] == pytest.approx(105.95 / 163.0, rel=1e-3)


def test_daily_unified_pk_baselines_event_windows_and_epistemic_labels(
    synthetic_data_dir: Path, tmp_path: Path
) -> None:
    adapter = LocalCSVAdapter(synthetic_data_dir, tmp_path / "out")
    tables = run_v15_pipeline(adapter, persist=True)

    daily = tables["Gold_Daily_Unified"].set_index("date")

    # Localization & Spine check
    assert daily.index.min() == "2025-06-05"
    assert daily.loc["2026-11-23", "display_date"] == "23-11-2026"
    assert daily.loc["2026-11-23", "work_mode"] == "Office_Mon_Wed"

    # Body comp: unweighed day 2026-11-23 keeps weight_kg as NaN while latest_weight_kg ffills 104.1
    assert np.isnan(daily.loc["2026-11-23", "weight_kg"])
    assert daily.loc["2026-11-23", "latest_weight_kg"] == pytest.approx(104.1)

    # Unlogged nutrition day 2026-11-24 stays strictly NaN (No Silent Zero-Filling!)
    assert np.isnan(daily.loc["2026-11-24", "calories_kcal"])
    assert daily.loc["2026-11-24", "nutrition_missing_reason"] == "Not_Logged"

    # Meal timing on 2026-11-22: last meal 21:30 -> bedtime 23:00 (session 702 waking 2026-11-23) = 90 min
    assert daily.loc["2026-11-22", "late_kcal_after_21"] == pytest.approx(600.0)
    assert daily.loc["2026-11-22", "late_fat_after_21"] == pytest.approx(30.0)
    assert daily.loc["2026-11-22", "last_meal_to_bed_min"] == pytest.approx(90.0)

    # Late work flag (03:30 IST Work entry on 2026-11-24)
    assert bool(daily.loc["2026-11-24", "late_work_00_05_flag"]) is True

    # Tirzepatide Bateman PK superposition tagged Model_Estimate + hourly table generated
    assert daily.loc["2026-11-22", "tirz_pk_provenance"] == "Model_Estimate"
    assert float(daily.loc["2026-11-23", "tirz_pk_est"]) > 2.0
    assert len(tables["Gold_Tirzepatide_PK_Hourly"]) > 24

    # Personal Baselines Z-scores (2026-06-10..12 HRV = [42, 40, 44], mean=42, median=42, MAD=2.0)
    base_df = tables["Gold_Personal_Baselines"].set_index("date")
    assert base_df.loc["2026-06-12", "hrv_z_classical_pre_oct"] == pytest.approx(1.0, rel=1e-3)
    assert base_df.loc["2026-06-12", "hrv_z_robust_pre_oct"] == pytest.approx(
        (44.0 - 42.0) / (1.4826 * 2.0), rel=1e-3
    )

    # Event Windows anti-reverse-causality alignment
    ev_win = tables["Gold_Event_Windows"].set_index("date")
    # Day 2026-11-22 meal timing -> night waking 2026-11-23 sleep HR (69.0) & spikes (8/6 = 1.3333)
    assert ev_win.loc["2026-11-22", "out_following_night_sleep_hr_avg"] == pytest.approx(69.0)
    assert ev_win.loc["2026-11-22", "out_following_night_spikes_per_asleep_hr"] == pytest.approx(
        8.0 / 6.0, rel=1e-3
    )
    # Day 2026-11-23 receives t-1 (2026-11-22) late fat (30g) -> day 2026-11-23 GI score (nausea=2 + reflux=1 = 3)
    assert ev_win.loc["2026-11-23", "exp_t_minus_1_late_fat_after_21"] == pytest.approx(30.0)
    assert ev_win.loc["2026-11-23", "out_day_t_gi_symptom_score"] == pytest.approx(3.0)
    # Thyronorm 09:00 vs first meal 09:50 on 2026-11-22 -> 50 min gap; Vit D 14:00 vs 13:30 20g fat lunch -> valid
    assert ev_win.loc["2026-11-22", "thyronorm_fasting_gap_min"] == pytest.approx(50.0)
    assert bool(ev_win.loc["2026-11-22", "fat_soluble_cofactor_valid_10g"]) is True
    assert bool(ev_win.loc["2026-11-22", "shot_day_skip_cascade_flag"]) is True

    # Epistemic Honesty Labels check across ALL materialized tables
    for tbl_name, tbl in tables.items():
        assert "epistemic_class" in tbl.columns, f"{tbl_name} missing epistemic_class"
        if not tbl.empty:
            invalid = set(tbl["epistemic_class"].dropna().unique()) - EPISTEMIC_CLASSES
            assert not invalid, f"{tbl_name} has invalid epistemic_class values: {invalid}"
