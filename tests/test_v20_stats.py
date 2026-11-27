"""Comprehensive unit & integration tests for the V2.0 Statistical Inference Engine."""

from __future__ import annotations

import ast
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from health_engine.config_loader import EPISTEMIC_CLASSES, load_engine_configs, load_treatments_config
from health_engine.io_adapter import LocalCSVAdapter
from health_engine.run_v20 import run_v20_pipeline
from health_engine.silver.qa_logger import SilverQALogger
from health_engine.stats.event_study import build_gold_intervention_events
from health_engine.stats.maturity import (
    MANDATORY_AUDIT_COLUMNS,
    compute_effective_n,
    compute_effective_n_with_r1,
)
from health_engine.stats.treatment_scorecards import build_gold_treatment_scorecards
from health_engine.stats.weight_alignment import (
    build_canonical_weighins,
    build_gold_weight_intervals,
    classify_fasted_weighin,
)
from private_smoking.run_private import main as run_private_main
from private_smoking.silver_smoking import process_silver_smoking
from private_smoking.stats_smoking import build_private_smoking_stats


def test_fasted_weighin_dynamic_classification_and_cross_source_qa() -> None:
    # 10:54 weigh-in before 12:30 breakfast -> Fasted
    st_fast, anchor_fast = classify_fasted_weighin("10:54", "12:30", has_logged_meals=True)
    assert st_fast == "Fasted"
    assert anchor_fast is True

    # 22:41 weigh-in after 4 logged meals (first meal 09:30) -> Post_Meal
    st_post, anchor_post = classify_fasted_weighin("22:41", "09:30", has_logged_meals=True)
    assert st_post == "Post_Meal"
    assert anchor_post is False

    # Fallback rules when no nutrition is logged on date
    assert classify_fasted_weighin("08:40", "", has_logged_meals=False) == ("Fasted_Assumed_Morning", True)
    assert classify_fasted_weighin("", "", has_logged_meals=False) == ("Fasted_Assumed_Morning", True)
    assert classify_fasted_weighin("15:30", "", has_logged_meals=False) == ("Midday_Unverified", False)
    assert classify_fasted_weighin("20:28", "", has_logged_meals=False) == ("Evening_Unverified", False)

    # End-to-end canonical weigh-in deduplication + >0.2 kg cross-source QA logging
    thresholds_cfg, _ = load_engine_configs()
    qa = SilverQALogger()

    silver_bc = pd.DataFrame(
        [
            {
                "date": "2026-11-20",
                "time_ist": "22:41",
                "weight_kg": "103.1",
                "bmi": "38.8",
                "body_fat_pct": "50.3",
                "fat_mass_kg": "51.9",
                "fat_free_body_weight_kg": "51.2",
                "muscle_mass_kg": "48.0",
                "skeletal_muscle_pct": "34.2",
                "visceral_fat": "14.2",
                "bone_mass_kg": "3.1",
            },
            {
                "date": "2026-11-26",
                "time_ist": "10:54",
                "weight_kg": "102.7",
                "bmi": "38.7",
                "body_fat_pct": "50.2",
                "fat_mass_kg": "51.6",
                "fat_free_body_weight_kg": "51.1",
                "muscle_mass_kg": "48.0",
                "skeletal_muscle_pct": "34.3",
                "visceral_fat": "14.1",
                "bone_mass_kg": "3.1",
            },
        ]
    )
    silver_w = pd.DataFrame(
        [
            {"date": "2026-11-20", "weight_kg": "103.5"},  # 0.4 kg discrepancy (>0.2 kg) -> QA logged, BIA wins
            {"date": "2026-11-26", "weight_kg": "102.7"},
        ]
    )
    silver_gl_w = pd.DataFrame([{"date": "2026-11-26", "weight_kg": "102.7"}])
    silver_nl = pd.DataFrame(
        [
            # 4 meals on 2026-11-20 before 22:41 weigh-in
            {"date": "2026-11-20", "time_ist": "09:15", "calories_kcal": "380"},
            {"date": "2026-11-20", "time_ist": "13:30", "calories_kcal": "520"},
            {"date": "2026-11-20", "time_ist": "17:00", "calories_kcal": "180"},
            {"date": "2026-11-20", "time_ist": "20:30", "calories_kcal": "450"},
            # First meal at 12:30 on 2026-11-26 after 10:54 weigh-in
            {"date": "2026-11-26", "time_ist": "12:30", "calories_kcal": "510"},
        ]
    )

    canon = build_canonical_weighins(silver_bc, silver_w, silver_gl_w, silver_nl, thresholds_cfg, qa=qa).set_index("date")
    assert canon.loc["2026-11-26", "weighin_status"] == "Fasted"
    assert bool(canon.loc["2026-11-26", "is_fasted_anchor"]) is True
    assert canon.loc["2026-11-20", "weighin_status"] == "Post_Meal"
    assert bool(canon.loc["2026-11-20", "is_fasted_anchor"]) is False
    assert canon.loc["2026-11-20", "canonical_source"] == "Fitdays_BIA"
    assert canon.loc["2026-11-20", "weight_kg"] == pytest.approx(103.1)
    assert bool(canon.loc["2026-11-20", "cross_source_discrepancy_flag"]) is True

    qa_df = qa.to_dataframe()
    disc_row = qa_df.loc[qa_df["rule_id"] == "WEIGHT_CROSS_SOURCE_DISCREPANCY_GT_0_2KG"].iloc[0]
    assert int(disc_row["flagged_or_excluded_rows"]) == 1
    assert "2026-11-20" in str(disc_row["detail_note"])


def test_weight_intervals_exact_math_and_maturity_gate_locking_and_unlocking() -> None:
    thresholds_cfg, _ = load_engine_configs()

    # Case A: 2 weigh-ins -> 1 interval (<12) -> MUST lock correlations at Level 0
    cw_small = pd.DataFrame(
        [
            {
                "date": "2026-11-08",
                "time_ist": "08:30",
                "canonical_source": "Fitdays_BIA",
                "weight_kg": 106.0,
                "has_bia": True,
                "fat_mass_kg": 54.0,
                "ffm_kg": 52.0,
                "muscle_mass_kg": 48.6,
                "visceral_fat": 15.0,
                "weighin_status": "Fasted",
                "is_fasted_anchor": True,
            },
            {
                "date": "2026-11-22",
                "time_ist": "09:00",
                "canonical_source": "Fitdays_BIA",
                "weight_kg": 104.0,
                "has_bia": True,
                "fat_mass_kg": 52.5,
                "ffm_kg": 51.5,
                "muscle_mass_kg": 48.2,
                "visceral_fat": 14.4,
                "weighin_status": "Fasted",
                "is_fasted_anchor": True,
            },
        ]
    )
    # 14-day interval (2026-11-08, 2026-11-22]: log nutrition on 7 of 14 days -> coverage = 0.50
    spine_14 = [
        (date(2026, 11, 8) + timedelta(days=i)).isoformat()
        for i in range(15)
    ]
    daily_small = pd.DataFrame(
        {
            "date": spine_14,
            "calories_kcal": [np.nan] + [1500.0 if i <= 7 else np.nan for i in range(1, 15)],
            "activity_total_kcal": [2300.0] * 15,
            "protein_g": [np.nan] + [100.0 if i <= 7 else np.nan for i in range(1, 15)],
            "steps": [7000.0] * 15,
            "is_period_day": [False] * 15,
        }
    )

    iv_df, iv_stats = build_gold_weight_intervals(
        cw_small,
        daily_small,
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        thresholds_cfg,
    )
    assert len(iv_df) == 1
    r0 = iv_df.iloc[0]
    assert int(r0["interval_days"]) == 14
    assert r0["delta_kg"] == pytest.approx(-2.0)
    assert r0["loss_rate_kg_per_wk"] == pytest.approx(1.0)  # -(-2.0) / (14/7) = 1.0 kg/wk
    assert r0["fat_share_of_loss"] == pytest.approx((-1.5) / (-2.0))  # 0.75
    assert r0["nutrition_coverage_pct"] == pytest.approx(7.0 / 14.0)  # 0.50
    assert bool(r0["is_fasted_interval"]) is True

    # Level 0 Gate enforcement when valid_intervals < 12
    assert (iv_stats["maturity_level"] == "Level 0 (Descriptive Ledger)").all()
    assert iv_stats["spearman_rho"].isna().all()
    assert iv_stats["p_value"].isna().all()
    assert iv_stats["partial_rho"].isna().all()
    assert iv_stats["ridge_std_coef"].isna().all()
    assert (iv_stats["epistemic_label"] == "Descriptive").all()

    # Case B: 14 weekly intervals (>=12) -> Unlocks Level 1 correlations, while enforcing nutrition_coverage_pct >= 0.60
    start_d = date(2026, 7, 1)
    cw_14_rows = []
    daily_14_rows = []
    curr_w = 112.0
    for wk in range(15):
        w_dt = start_d + timedelta(days=7 * wk)
        loss_step = 0.4 + 0.08 * (wk % 5)
        if wk > 0:
            curr_w = round(curr_w - loss_step, 3)
        cw_14_rows.append(
            {
                "date": w_dt.isoformat(),
                "time_ist": "08:30",
                "canonical_source": "Fitdays_BIA",
                "weight_kg": curr_w,
                "has_bia": True,
                "fat_mass_kg": round(curr_w * 0.48, 3),
                "ffm_kg": round(curr_w * 0.52, 3),
                "muscle_mass_kg": round(curr_w * 0.45, 3),
                "visceral_fat": 14.0,
                "weighin_status": "Fasted",
                "is_fasted_anchor": True,
            }
        )
        if wk > 0:
            prev_dt = start_d + timedelta(days=7 * (wk - 1))
            for d_off in range(1, 8):
                dd = (prev_dt + timedelta(days=d_off)).isoformat()
                # Log 6/7 days (coverage 0.857 >= 0.60) on 13 intervals, and only 2/7 days on 1 interval
                has_nutr = (d_off <= 2) if wk == 1 else (d_off <= 6)
                daily_14_rows.append(
                    {
                        "date": dd,
                        "calories_kcal": (1850.0 - 90.0 * (wk % 5)) if has_nutr else np.nan,
                        "activity_total_kcal": 2400.0,
                        "steps": 5500.0 + 600.0 * (wk % 5),
                        "is_period_day": (wk % 4 == 0 and d_off == 7),
                    }
                )

    _, stats_unlocked = build_gold_weight_intervals(
        pd.DataFrame(cw_14_rows),
        pd.DataFrame(daily_14_rows),
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        thresholds_cfg,
    )
    steps_stat = stats_unlocked.loc[
        (stats_unlocked["target_metric"] == "loss_rate_kg_per_wk")
        & (stats_unlocked["predictor_name"] == "steps_mean")
    ].iloc[0]
    kcal_stat = stats_unlocked.loc[
        (stats_unlocked["target_metric"] == "loss_rate_kg_per_wk")
        & (stats_unlocked["predictor_name"] == "kcal_mean")
    ].iloc[0]

    assert steps_stat["maturity_level"] == "Level 1 (Early Signal)"
    assert int(steps_stat["valid_n"]) == 14
    assert np.isfinite(steps_stat["spearman_rho"]) and steps_stat["spearman_rho"] > 0.5
    # kcal_mean excluded the 1 low-coverage interval (2/7 = 0.2857 < 0.60) -> valid_n == 13
    assert int(kcal_stat["valid_n"]) == 13
    assert kcal_stat["maturity_level"] == "Level 1 (Early Signal)"
    assert np.isfinite(kcal_stat["spearman_rho"]) and kcal_stat["spearman_rho"] < -0.5


def test_treatment_scorecards_dual_baseline_overlap_magnesium_gate_and_auto_escalation() -> None:
    thresholds_cfg, _ = load_engine_configs()
    treatments_cfg = load_treatments_config()

    # Build deterministic daily history covering:
    # - Pre-treatment baseline: 2026-08-01 .. 2026-09-30 (avg_hr=72.0, deep_min=80.0)
    # - Tirzepatide 2.5mg phase: 2026-10-21 .. 2026-11-18 (avg_hr=75.0)
    # - Tirzepatide 5.0mg phase: 2026-11-19 .. 2026-11-30 (avg_hr=78.0)
    # - Magnesium Glycinate started 2026-11-25 -> 6 days through 2026-11-30 (<14 -> Level 0 Collecting_Baseline!)
    dates = pd.date_range("2026-08-01", "2026-11-30", freq="D").strftime("%Y-%m-%d").tolist()
    avg_hrs = []
    for ds in dates:
        if ds <= "2026-09-30":
            avg_hrs.append(72.0 + (0.5 if int(ds[-1]) % 2 == 0 else -0.5))
        elif "2026-10-21" <= ds <= "2026-11-18":
            avg_hrs.append(75.0)
        elif ds >= "2026-11-19":
            avg_hrs.append(78.0)
        else:
            avg_hrs.append(73.0)

    gold_daily = pd.DataFrame(
        {
            "date": dates,
            "avg_hr": avg_hrs,
            "sleep_hr_avg": [h - 8.0 for h in avg_hrs],
            "resting_hr": [h - 6.0 for h in avg_hrs],
            "p95_hr": [h + 18.0 for h in avg_hrs],
            "hrv_rmssd_ms": [42.0 if d <= "2026-09-30" else 36.0 for d in dates],
            "asleep_min": [420.0] * len(dates),
            "deep_min": [80.0 if d < "2026-11-25" else 92.0 for d in dates],
            "rem_min": [95.0] * len(dates),
            "steps": [7000.0] * len(dates),
            "calories_kcal": [1600.0] * len(dates),
            "protein_g": [95.0] * len(dates),
            "fiber_g": [24.0] * len(dates),
            "hunger": [1.5] * len(dates),
            "food_noise": [1.0] * len(dates),
            "days_since_shot": [0.0] * len(dates),
            "micro_journal_status": ["Present" if d >= "2026-11-20" else "Source_Gap" for d in dates],
            "glapp_mood_status": ["Present" if d >= "2026-10-21" else "Source_Gap" for d in dates],
        }
    )

    # Include Thyronorm 75 mcg bottle-finishing entry post-2026-11-10 AND a future 7.5mg shot on 2026-12-17
    silver_meds = pd.DataFrame(
        [
            {"date": "2026-11-25", "Med_or_Supplement": "Thyronorm", "Dose": "75 mcg", "Status": "Taken", "Notes": ""},
            {"date": "2026-11-25", "Med_or_Supplement": "Magnesium Glycinate", "Dose": "400 mg", "Status": "Taken", "Notes": ""},
        ]
    )
    silver_shots = pd.DataFrame(
        [
            {"date": "2026-10-21", "medication": "tirzepatide", "dose_mg": "2.5"},
            {"date": "2026-11-19", "medication": "tirzepatide", "dose_mg": "5.0"},
        ]
    )
    silver_tables = {
        "Silver_Medications": pd.DataFrame(),
        "Silver_Procedures": pd.DataFrame(),
        "Silver_Glapp_Shots": silver_shots,
        "Silver_Sleep_Daily": pd.DataFrame({"date": dates, "sleep_onset_latency_min_est": [10.0] * len(dates)}),
        "Silver_Meds_Log": silver_meds,
        "Silver_Symptoms_Log": pd.DataFrame(),
        "Silver_Glapp_Side_Effects": pd.DataFrame(),
        "Silver_Nutrition_Log": pd.DataFrame(),
    }

    sc = build_gold_treatment_scorecards(
        gold_daily=gold_daily,
        gold_weight_intervals=pd.DataFrame(),
        gold_sleep_spikes=pd.DataFrame(),
        gold_event_windows=pd.DataFrame(),
        cardiac_daily=pd.DataFrame(),
        silver_tables=silver_tables,
        thresholds_cfg=thresholds_cfg,
        treatments_cfg=treatments_cfg,
    )

    # 1. Tirzepatide_5.0mg computes BOTH delta_vs_pre_treatment (+6.0) AND delta_vs_prior_dose (+3.0 vs 2.5mg)
    t5_hr = sc.loc[(sc["treatment_id"] == "Tirzepatide_5.0mg") & (sc["metric_name"] == "avg_hr")].iloc[0]
    assert t5_hr["prior_dose_id"] == "Tirzepatide_2.5mg"
    assert t5_hr["pre_treatment_baseline_mean"] == pytest.approx(72.0, abs=0.05)
    assert t5_hr["prior_dose_baseline_mean"] == pytest.approx(75.0)
    assert t5_hr["current_full_phase_mean"] == pytest.approx(78.0)
    assert t5_hr["delta_vs_pre_treatment"] == pytest.approx(6.0, abs=0.05)
    assert t5_hr["delta_vs_prior_dose"] == pytest.approx(3.0)

    # 2. Oct 2026 therapies flag overlap_warning = True automatically
    for oct_id in ("Vit_D_600K_IM", "IV_Iron_FCM_1g", "Tirzepatide_2.5mg", "Trazer_F_Forte", "Matilda_Forte"):
        sub_oct = sc.loc[sc["treatment_id"] == oct_id]
        assert not sub_oct.empty
        assert bool(sub_oct.iloc[0]["overlap_warning"]) is True

    # 3. Magnesium_Glycinate stays at Level 0 (Collecting Baseline) when valid_n == 6 (< 14)
    mag_deep = sc.loc[(sc["treatment_id"] == "Magnesium_Glycinate") & (sc["metric_name"] == "deep_min")].iloc[0]
    assert int(mag_deep["current_valid_n"]) == 6
    assert mag_deep["maturity_level"] == "Level 0 (Collecting Baseline)"
    assert mag_deep["clinical_window_status"] == "Collecting_Baseline"

    # 4. Thyronorm_100mcg reconciles 75 mcg bottle-finishing entries cleanly without error
    thy_hr = sc.loc[(sc["treatment_id"] == "Thyronorm_100mcg") & (sc["metric_name"] == "avg_hr")].iloc[0]
    assert "Reconciled Meds_Log 75 mcg" in str(thy_hr["caveat_note"])

    # 5. Extensible auto-registration when a 7.5 mg shot appears in Glapp_Data_shots
    silver_tables_75 = dict(silver_tables)
    silver_tables_75["Silver_Glapp_Shots"] = pd.DataFrame(
        [
            {"date": "2026-10-21", "medication": "tirzepatide", "dose_mg": "2.5"},
            {"date": "2026-11-19", "medication": "tirzepatide", "dose_mg": "5.0"},
            {"date": "2026-11-28", "medication": "tirzepatide", "dose_mg": "7.5"},
        ]
    )
    sc_75 = build_gold_treatment_scorecards(
        gold_daily=gold_daily,
        gold_weight_intervals=pd.DataFrame(),
        gold_sleep_spikes=pd.DataFrame(),
        gold_event_windows=pd.DataFrame(),
        cardiac_daily=pd.DataFrame(),
        silver_tables=silver_tables_75,
        thresholds_cfg=thresholds_cfg,
        treatments_cfg=treatments_cfg,
    )
    assert "Tirzepatide_7.5mg" in set(sc_75["treatment_id"])
    t75_row = sc_75.loc[sc_75["treatment_id"] == "Tirzepatide_7.5mg"].iloc[0]
    assert t75_row["prior_dose_id"] == "Tirzepatide_5.0mg"


def test_effective_n_autocorrelation_shrinkage_and_small_n_fallback() -> None:
    # Small n < 5 falls back to valid_n
    assert compute_effective_n([10.0, 12.0, 14.0, 16.0]) == 4
    assert compute_effective_n([np.nan, np.nan]) == 0

    # Deterministic AR(1) series with strong positive lag-1 autocorrelation -> effective_n < valid_n
    rng = np.random.default_rng(123)
    n = 80
    ar_series = np.zeros(n, dtype=float)
    for t in range(1, n):
        ar_series[t] = 0.7 * ar_series[t - 1] + rng.normal(0.0, 1.0)

    neff, r1 = compute_effective_n_with_r1(ar_series)
    expected = int(np.clip(np.round(n * (1.0 - r1) / (1.0 + r1)), 1, n))
    assert r1 > 0.5
    assert neff == expected
    assert 1 <= neff < n


def test_event_study_tier3_its_refuses_overlapping_oct_anchors_and_runs_when_isolated() -> None:
    thresholds_cfg, _ = load_engine_configs()
    dates = pd.date_range("2026-05-01", "2026-12-25", freq="D").strftime("%Y-%m-%d").tolist()
    gold_daily = pd.DataFrame(
        {
            "date": dates,
            "work_mode": ["Office_Mon_Wed"] * len(dates),
            "cycle_phase": ["Follicular_D1_14"] * len(dates),
            "hr_hours_covered": [22] * len(dates),
            "steps": [7000.0] * len(dates),
            "avg_hr": [74.0 if d < "2026-10-10" else 70.0 for d in dates],
            "daytime_avg_hr": [78.0] * len(dates),
            "resting_hr": [65.0] * len(dates),
            "p95_hr": [95.0] * len(dates),
            "hrv_rmssd_ms": [40.0] * len(dates),
            "sleep_hr_avg": [64.0] * len(dates),
            "asleep_min": [420.0] * len(dates),
            "deep_min": [85.0] * len(dates),
            "rem_min": [95.0] * len(dates),
        }
    )

    # 1. Standard Oct-Nov 2026 anchors overlap within +/-28d -> Tier 3 ITS MUST refuse to run
    ev_df = build_gold_intervention_events(gold_daily, pd.DataFrame(), thresholds_cfg)
    oct_rows = ev_df.loc[ev_df["anchor_id"].isin(["Vit_D_600K_IM", "IV_Iron_FCM_1g", "Tirzepatide_2.5mg"])]
    assert (oct_rows["overlap_warning"] == True).all()  # noqa: E712
    assert (oct_rows["its_status"] == "Gated_Overlap_Or_Insufficient_Window").all()
    assert oct_rows["tier3_its_level_change"].isna().all()

    # 2. Isolated single anchor with >=28 pre and >=28 post valid days -> Tier 3 ITS unlocks
    iso_cfg = dict(thresholds_cfg)
    iso_cfg["intervention_anchors"] = [
        {"anchor_id": "Isolated_Test_Intervention", "anchor_date": "2026-10-10", "label": "Isolated", "end_date": None}
    ]
    ev_iso = build_gold_intervention_events(gold_daily, pd.DataFrame(), iso_cfg)
    hr_iso = ev_iso.loc[ev_iso["metric_name"] == "avg_hr"].iloc[0]
    assert bool(hr_iso["overlap_warning"]) is False
    assert hr_iso["its_status"] == "Unlocked_Segmented_ITS"
    assert hr_iso["tier3_its_level_change"] == pytest.approx(-4.0, abs=0.2)


def test_v20_pipeline_end_to_end_reverse_causality_and_audit_enforcement(
    synthetic_data_dir: Path, tmp_path: Path
) -> None:
    adapter = LocalCSVAdapter(synthetic_data_dir, tmp_path / "out_v20")
    outputs = run_v20_pipeline(adapter, persist=True)

    # Verify same-day fat/kcal <-> GI flagged with reverse_causality_risk = True
    corr = outputs["Gold_Stats_Correlations"].set_index("pair_id")
    assert bool(corr.loc["DIAG_SAMEDAY_FAT_GI", "reverse_causality_risk"]) is True
    assert bool(corr.loc["DIAG_SAMEDAY_KCAL_GI", "reverse_causality_risk"]) is True
    assert bool(corr.loc["EV_TM1_EVE_FAT18_GI", "reverse_causality_risk"]) is False

    # Verify NO hardcoded phase labels ("nadir", "rebound") in Gold_Tirz_Folded_Wave
    wave_df = outputs["Gold_Tirz_Folded_Wave"]
    serialized_wave = wave_df.to_csv(index=False).lower()
    assert "nadir" not in serialized_wave
    assert "rebound" not in serialized_wave

    # Verify mandatory V2.0 audit columns & valid epistemic labels on EVERY V2.0 stats table
    v20_table_names = [
        "Gold_Weight_Intervals",
        "Gold_Weight_Interval_Stats",
        "Gold_Weight_Trend",
        "Gold_Weight_Clinical_Link",
        "Gold_Treatment_Scorecards",
        "Gold_Stats_Correlations",
        "Gold_Tirz_Folded_Wave",
        "Gold_Intervention_Events",
        "Gold_Cardiac_Efficiency",
        "Gold_Cardiac_Efficiency_Buckets",
        "Gold_Cycle_Phase_Stats",
        "Gold_Episode_Chains",
    ]
    for tname in v20_table_names:
        tbl = outputs[tname]
        for acol in MANDATORY_AUDIT_COLUMNS:
            assert acol in tbl.columns, f"{tname} missing mandatory audit column '{acol}'"
        if not tbl.empty:
            bad = set(tbl["epistemic_label"].dropna().unique()) - EPISTEMIC_CLASSES
            assert not bad, f"{tname} has invalid epistemic_label: {bad}"

    audit_df = outputs["Stats_Run_Audit"]
    assert len(audit_df) >= 9
    assert (tmp_path / "out_v20" / "Stats_Run_Audit.parquet").exists()


def test_stats_strict_privacy_boundary_and_private_smoking_v20_zero_leakage(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # 1. Static AST + literal scan across all files in src/health_engine/stats/
    stats_dir = Path(__file__).resolve().parents[1] / "src" / "health_engine" / "stats"
    py_files = sorted(stats_dir.glob("*.py"))
    assert len(py_files) >= 8

    for py_path in py_files:
        txt = py_path.read_text(encoding="utf-8")
        assert "smoking_log" not in txt.lower(), f"Forbidden Smoking_Log reference in {py_path}"
        assert "private_smoking" not in txt.lower(), f"Forbidden private_smoking reference in {py_path}"
        tree = ast.parse(txt, filename=str(py_path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith("private_smoking")
            elif isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("private_smoking")

    # 2. Verify private_smoking.stats_smoking onset vs post separation & silent CLI
    priv_in = tmp_path / "priv_v20_in" / "Micro Journal"
    priv_in.mkdir(parents=True)
    raw_smk = pd.DataFrame(
        [
            {
                "Date": "2026-11-22",
                "Time": "16:10",
                "Event_Type": "Smoked",
                "Count": "1",
                "Craving_Intensity": "4",
                "Delay_Minutes": "12",
                "Trigger": "Secret_Trigger_Alpha",
                "Mood": "Stressed",
                "Barrier_or_Reason": "Secret_Barrier_Beta",
            },
            {
                "Date": "2026-11-23",
                "Time": "11:20",
                "Event_Type": "Craving Resisted",
                "Count": "0",
                "Craving_Intensity": "3",
                "Delay_Minutes": "30",
                "Trigger": "Coffee_Break",
                "Mood": "Calm",
                "Barrier_or_Reason": "Walked_Around",
            },
        ]
    )
    raw_smk.to_csv(priv_in / "Smoking_Log.csv", index=False)

    silver_smk = process_silver_smoking(raw_smk)
    daily_ctx = pd.DataFrame(
        [
            {"date": "2026-11-22", "shot_day_int": 0, "active_tirz_dose_mg": 5.0, "asleep_min": 330.0},
            {"date": "2026-11-23", "shot_day_int": 1, "active_tirz_dose_mg": 5.0, "asleep_min": 450.0},
        ]
    )
    wint_ctx = pd.DataFrame(
        [
            {
                "interval_id": "WINT-001",
                "start_date": "2026-11-20",
                "end_date": "2026-11-26",
                "interval_days": 6,
                "in_all_weighins": True,
                "loss_rate_kg_per_wk": 0.47,
            }
        ]
    )
    v20_smk = build_private_smoking_stats(silver_smk, daily_context_df=daily_ctx, weight_intervals_df=wint_ctx)
    desc = v20_smk["Private_Smoking_Descriptive"]
    wint = v20_smk["Private_Weight_Interval_Smoking"]
    sc = v20_smk["Private_Treatment_Smoking_Scorecard"]

    assert set(desc["onset_dimension"]) == {
        "hour_ist",
        "Trigger",
        "Mood",
        "shot_day_int",
        "prior_night_sleep_tier",
    }
    assert {"<6h", ">=7h"}.issubset(
        set(desc.loc[desc["onset_dimension"] == "prior_night_sleep_tier", "onset_stratum"])
    )
    assert len(wint) == 1
    assert wint.iloc[0]["maturity_level"] == "Level 0 (Descriptive Ledger)"
    assert (sc["maturity_level"] == "Level 0 (Collecting Baseline)").all()

    rc = run_private_main(
        [
            "--source",
            "csv",
            "--data-dir",
            str(tmp_path / "priv_v20_in"),
            "--output-dir",
            str(tmp_path / "priv_v20_out"),
        ]
    )
    assert rc == 0
    out_std = capsys.readouterr().out
    assert "Secret_Trigger_Alpha" not in out_std
    assert "Secret_Barrier_Beta" not in out_std
    assert (tmp_path / "priv_v20_out" / "Private_Smoking_Descriptive.parquet").exists()
    assert (tmp_path / "priv_v20_out" / "Private_Treatment_Smoking_Scorecard.parquet").exists()
