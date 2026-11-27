"""Comprehensive unit & integration tests for the V2.5 Progressive ML Engine."""

from __future__ import annotations

import ast
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from health_engine.config_loader import load_engine_configs, load_journal_themes_config
from health_engine.io_adapter import LocalCSVAdapter
from health_engine.ml.clustering import (
    PHENOTYPE_DRAIN,
    PHENOTYPE_EXERTION,
    PHENOTYPE_RECOVERY,
    assign_posthoc_phenotype_labels,
    build_gold_phenotypes,
)
from health_engine.ml.evaluator import (
    ModelAuditLogger,
    evaluate_progressive_regressor,
    generate_chronological_splits,
)
from health_engine.ml.journal_nlp import build_gold_journal_nlp
from health_engine.ml.regressors import (
    assert_gi_no_sameday_nutrition,
    assert_modifiable_hrv_features,
    build_gold_ml_regressors,
    run_gi_severity_model,
)
from health_engine.run_v25 import run_v25_pipeline
from private_smoking.ml_smoking import (
    build_private_smoking_ml,
    validate_onset_feature_matrix,
)
from private_smoking.run_private_ml import main as run_private_ml_main
from private_smoking.silver_smoking import process_silver_smoking


def test_timeseries_split_strictly_chronological_zero_future_leakage() -> None:
    dates = pd.date_range("2026-08-01", periods=60, freq="D").strftime("%Y-%m-%d").tolist()
    splits = generate_chronological_splits(dates, n_splits=5, gap=1)
    assert len(splits) == 5

    for _, tr_idx, te_idx in splits:
        train_dates = [dates[i] for i in tr_idx]
        test_dates = [dates[i] for i in te_idx]
        # Strict chronological ordering + gap=1 day separation
        assert max(train_dates) < min(test_dates)
        gap_days = (pd.Timestamp(min(test_dates)) - pd.Timestamp(max(train_dates))).days
        assert gap_days >= 2

    # Unsorted input to generate_chronological_splits MUST raise ValueError
    unsorted = list(reversed(dates[:20]))
    with pytest.raises(ValueError, match="not sorted chronologically"):
        generate_chronological_splits(unsorted, n_splits=3, gap=1)


def test_model_1b_excludes_sleep_hr_and_model_3_rejects_sameday_nutrition() -> None:
    thresholds_cfg, _ = load_engine_configs()

    # 1. Model 1B modifiable guard raises ValueError if sleep_hr_avg is injected
    with pytest.raises(ValueError, match="forbids autonomic-twin sleep HR"):
        assert_modifiable_hrv_features(["bedtime_hr_rel_mid", "asleep_min", "sleep_hr_avg"])

    # 2. Model 3 GI predictor raises ValueError if same-day calories_kcal or fat_g is injected
    with pytest.raises(ValueError, match="forbids same-day nutrition"):
        assert_gi_no_sameday_nutrition(["prev_evening_fat_g", "calories_kcal"])

    with pytest.raises(ValueError, match="forbids same-day nutrition"):
        run_gi_severity_model(
            pd.DataFrame({"date": ["2026-11-01"], "gi_severity_sum": [1.0], "fat_g": [50.0]}),
            ["fat_g"],
            thresholds_cfg,
        )

    # 3. End-to-end check on 50 days: Model 1A has sleep_hr_avg while Model 1B strictly excludes it
    rng = np.random.default_rng(7)
    dates = pd.date_range("2026-10-01", periods=50, freq="D").strftime("%Y-%m-%d").tolist()
    bed = rng.uniform(-1.5, 2.0, size=50)
    asleep = 430.0 - 25.0 * bed + rng.normal(0, 10, size=50)
    sl_hr = 64.0 + 1.5 * bed + rng.normal(0, 1.0, size=50)
    hrv = 45.0 - 1.2 * (sl_hr - 64.0) - 2.0 * bed + rng.normal(0, 1.5, size=50)

    gold_daily = pd.DataFrame(
        {
            "date": dates,
            "hrv_rmssd_ms": hrv,
            "sleep_hr_avg": sl_hr,
            "bedtime_hr_rel_mid": bed,
            "asleep_min": asleep,
            "steps": rng.uniform(4500, 9500, size=50),
            "zone_mins": rng.uniform(5, 40, size=50),
            "workout_mins": rng.uniform(0, 45, size=50),
            "active_hours": rng.uniform(4, 10, size=50),
            "avg_hr": sl_hr + 10.0,
            "work_mode": ["Office_Mon_Wed" if i % 3 == 0 else "WFH_Tue_Thu_Fri" for i in range(50)],
            "cycle_phase": ["Follicular_D1_14"] * 50,
            "tirz_pk_est": [2.5] * 50,
            "active_tirz_dose_mg": [2.5] * 50,
            "shot_day_int": [i % 7 for i in range(50)],
        }
    )
    _, drivers = build_gold_ml_regressors(
        gold_daily=gold_daily,
        gold_event_windows=pd.DataFrame(),
        gold_weight_intervals=pd.DataFrame(),
        thresholds_cfg=thresholds_cfg,
    )
    m1a_feats = set(drivers.loc[drivers["model_id"] == "ML_HRV_1A_FULL", "feature_name"])
    m1b_feats = set(drivers.loc[drivers["model_id"] == "ML_HRV_1B_MODIFIABLE", "feature_name"])
    assert "sleep_hr_avg" in m1a_feats
    assert "sleep_hr_avg" not in m1b_feats
    assert "bedtime_hr_rel_mid" in m1b_feats


def test_progressive_maturity_gates_lock_and_unlock_linear_rf_shap_and_weight_velocity() -> None:
    thresholds_cfg, _ = load_engine_configs()
    rng = np.random.default_rng(99)

    # 1. Level 0-1 (valid_n = 20 < 30) -> Gated_Insufficient_N
    df_20 = pd.DataFrame(
        {
            "date": pd.date_range("2026-10-01", periods=20, freq="D").strftime("%Y-%m-%d"),
            "x1": rng.normal(size=20),
            "y": rng.normal(size=20),
        }
    )
    s_20, d_20 = evaluate_progressive_regressor(
        model_id="TEST_L0",
        model_label="Test L0",
        df_scope=df_20,
        date_col="date",
        target_col="y",
        feature_cols=["x1"],
        thresholds_cfg=thresholds_cfg,
    )
    assert s_20["status"] == "Gated_Insufficient_N"
    assert d_20 == []

    # 2. Level 2 (valid_n = 55 < 100) -> Ridge/ElasticNet active, RF/SHAP locked (`Gated_Below_N100`)
    x1_55 = rng.normal(size=55)
    x2_55 = rng.normal(size=55)
    df_55 = pd.DataFrame(
        {
            "date": pd.date_range("2026-09-01", periods=55, freq="D").strftime("%Y-%m-%d"),
            "x1": x1_55,
            "x2": x2_55,
            "y": 3.0 * x1_55 - 1.5 * x2_55 + rng.normal(0, 0.4, size=55),
        }
    )
    s_55, d_55 = evaluate_progressive_regressor(
        model_id="TEST_L2",
        model_label="Test L2",
        df_scope=df_55,
        date_col="date",
        target_col="y",
        feature_cols=["x1", "x2"],
        thresholds_cfg=thresholds_cfg,
    )
    assert s_55["status"] == "Active"
    assert s_55["maturity_level"] == "Level 2 (Regularized Linear)"
    assert s_55["tree_unlock_status"] == "Gated_Below_N100"
    assert np.isnan(s_55["ml_cv_r2"])
    assert s_55["linear_cv_r2"] > 0.7
    assert len(d_55) == 2
    assert all(np.isnan(r["mean_abs_shap"]) for r in d_55)

    # 3. Weight Velocity ML with 10 intervals (< 24) -> Gated_Below_N24_Intervals cleanly
    wi_10 = pd.DataFrame(
        {
            "end_date": pd.date_range("2026-09-01", periods=10, freq="7D").strftime("%Y-%m-%d"),
            "in_all_weighins": [True] * 10,
            "is_fasted_interval": [True] * 10,
            "nutrition_coverage_pct": [0.85] * 10,
            "loss_rate_kg_per_wk": rng.uniform(0.3, 1.2, size=10),
            "est_deficit_mean": rng.uniform(400, 900, size=10),
            "protein_g_mean": rng.uniform(80, 110, size=10),
            "carbs_g_mean": rng.uniform(110, 150, size=10),
            "steps_mean": rng.uniform(5000, 9000, size=10),
            "asleep_hr_mean": rng.uniform(6.0, 7.5, size=10),
            "active_tirz_dose_mg": [2.5] * 10,
            "period_at_weighin_flag": [False] * 10,
            "constipation_48h_before_weighin_flag": [False] * 10,
        }
    )
    sum_all, _ = build_gold_ml_regressors(
        gold_daily=pd.DataFrame({"date": ["2026-11-01"], "hrv_rmssd_ms": [40.0]}),
        gold_event_windows=pd.DataFrame(),
        gold_weight_intervals=wi_10,
        thresholds_cfg=thresholds_cfg,
    )
    w_row = sum_all.loc[sum_all["model_id"] == "ML_WEIGHT_4_VELOCITY"].iloc[0]
    assert w_row["status"] == "Gated_Below_N24_Intervals"
    assert np.isnan(w_row["linear_cv_r2"])

    # 4. Level 3 (valid_n = 130 >= 100, effective_n >= 40) with strong nonlinear XOR/quadrant signal -> Unlocks RF + SHAP!
    x1_130 = rng.uniform(-2.0, 2.0, size=130)
    x2_130 = rng.uniform(-2.0, 2.0, size=130)
    x3_130 = rng.normal(0, 1.0, size=130)
    y_130 = np.where((x1_130 > 0) & (x2_130 > 0), 8.0, np.where((x1_130 < 0) & (x2_130 < 0), 7.5, -2.0)) + rng.normal(0, 0.25, size=130)
    df_130 = pd.DataFrame(
        {
            "date": pd.date_range("2026-06-01", periods=130, freq="D").strftime("%Y-%m-%d"),
            "x1": x1_130,
            "x2": x2_130,
            "x3": x3_130,
            "y": y_130,
        }
    )
    audit = ModelAuditLogger()
    s_130, d_130 = evaluate_progressive_regressor(
        model_id="TEST_L3_SHAP",
        model_label="Test L3 Nonlinear",
        df_scope=df_130,
        date_col="date",
        target_col="y",
        feature_cols=["x1", "x2", "x3"],
        thresholds_cfg=thresholds_cfg,
        audit_logger=audit,
    )
    assert s_130["tree_unlock_status"] == "Unlocked_RandomForest_Depth4"
    assert s_130["shap_unlock_status"] == "Unlocked_SHAP_TreeExplainer"
    assert s_130["ml_cv_r2"] > 0.20
    assert s_130["ml_cv_r2"] > s_130["linear_cv_r2"] + 0.15
    assert all(np.isfinite(r["mean_abs_shap"]) and r["mean_abs_shap"] >= 0 for r in d_130)
    assert all("does not prove biological causation" in r["ui_phrasing"] for r in d_130)
    assert len(audit.to_dataframe()) == 1


def test_kmeans_ari_stability_and_permutation_invariant_posthoc_labeling() -> None:
    # 1. Verify post-hoc centroid label assignment across ALL 6 cluster ID permutations
    archetypes = [
        {"name": PHENOTYPE_RECOVERY, "asleep_min": 465.0, "bedtime_hr_rel_mid": -1.1, "steps": 6800.0},
        {"name": PHENOTYPE_EXERTION, "asleep_min": 355.0, "bedtime_hr_rel_mid": 0.4, "steps": 10400.0},
        {"name": PHENOTYPE_DRAIN, "asleep_min": 340.0, "bedtime_hr_rel_mid": 2.3, "steps": 3600.0},
    ]
    for perm in permutations([0, 1, 2]):
        rows = []
        expected_map = {}
        for arch, cid in zip(archetypes, perm):
            rows.append(
                {
                    "cluster_id": cid,
                    "asleep_min": arch["asleep_min"],
                    "bedtime_hr_rel_mid": arch["bedtime_hr_rel_mid"],
                    "steps": arch["steps"],
                }
            )
            expected_map[cid] = arch["name"]
        assigned = assign_posthoc_phenotype_labels(pd.DataFrame(rows))
        assert assigned == expected_map

    # 2. End-to-end clustering + 25x bootstrap ARI check on 90 synthetic days
    thresholds_cfg, _ = load_engine_configs()
    rng = np.random.default_rng(42)
    dates = pd.date_range("2026-08-01", periods=90, freq="D").strftime("%Y-%m-%d").tolist()
    records = []
    for i, ds in enumerate(dates):
        mode = i % 3
        if mode == 0:  # Earlier-Bedtime Recovery
            records.append(
                {
                    "date": ds,
                    "vitals_valid_coverage": True,
                    "steps": 6800 + rng.normal(0, 300),
                    "active_hours": 7.0 + rng.normal(0, 0.5),
                    "avg_hr": 71.0 + rng.normal(0, 1.0),
                    "asleep_min": 465.0 + rng.normal(0, 12.0),
                    "bedtime_hr_rel_mid": -1.2 + rng.normal(0, 0.25),
                    "hrv_rmssd_ms": 46.0,
                    "deep_min": 95.0,
                    "rem_min": 105.0,
                    "sleep_hr_avg": 62.0,
                }
            )
        elif mode == 1:  # High-Exertion Short Sleep
            records.append(
                {
                    "date": ds,
                    "vitals_valid_coverage": True,
                    "steps": 11200 + rng.normal(0, 400),
                    "active_hours": 11.0 + rng.normal(0, 0.5),
                    "avg_hr": 79.0 + rng.normal(0, 1.0),
                    "asleep_min": 355.0 + rng.normal(0, 12.0),
                    "bedtime_hr_rel_mid": 0.5 + rng.normal(0, 0.25),
                    "hrv_rmssd_ms": 37.0,
                    "deep_min": 75.0,
                    "rem_min": 80.0,
                    "sleep_hr_avg": 67.0,
                }
            )
        else:  # Late-Night Sedentary Drain
            records.append(
                {
                    "date": ds,
                    "vitals_valid_coverage": True,
                    "steps": 3100 + rng.normal(0, 300),
                    "active_hours": 3.0 + rng.normal(0, 0.5),
                    "avg_hr": 76.0 + rng.normal(0, 1.0),
                    "asleep_min": 330.0 + rng.normal(0, 12.0),
                    "bedtime_hr_rel_mid": 2.6 + rng.normal(0, 0.25),
                    "hrv_rmssd_ms": 31.0,
                    "deep_min": 58.0,
                    "rem_min": 70.0,
                    "sleep_hr_avg": 70.0,
                }
            )

    daily_pheno, profiles = build_gold_phenotypes(pd.DataFrame(records), thresholds_cfg)
    assert len(daily_pheno) == 90
    assert len(profiles) == 3
    assert set(profiles["phenotype_label"]) == {
        PHENOTYPE_RECOVERY,
        PHENOTYPE_EXERTION,
        PHENOTYPE_DRAIN,
    }
    assert float(profiles.iloc[0]["median_ari"]) >= 0.70
    assert profiles.iloc[0]["stability_flag"] == "Stable"
    assert (
        daily_pheno.iloc[-1]["recovery_days_7d"]
        + daily_pheno.iloc[-1]["high_exertion_short_sleep_7d"]
        + daily_pheno.iloc[-1]["late_drain_days_7d"]
    ) == 7


def test_private_smoking_ml_zero_leakage_and_progressive_classifier_gates(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # 1. Runtime ValueError if Delay_Minutes, Count, Barrier_or_Reason, or Event_Type enters X
    for bad_col in ("Delay_Minutes", "Count", "Barrier_or_Reason", "Event_Type", "delay_minutes_post"):
        with pytest.raises(ValueError, match="Zero-leakage boundary violation"):
            validate_onset_feature_matrix(pd.DataFrame({"hour_sin": [0.5], bad_col: [1]}))

    # 2. Gate 0: < 30 events -> Level 0 Descriptive Risk Heatmap ONLY (no classifier trained)
    raw_10 = pd.DataFrame(
        [
            {
                "Date": f"2026-11-{10 + i:02d}",
                "Time": "15:30",
                "Event_Type": "Smoked" if i % 2 == 0 else "Craving Resisted",
                "Count": "1" if i % 2 == 0 else "0",
                "Craving_Intensity": "4",
                "Delay_Minutes": "10",
                "Trigger": "Private_Trigger_Token",
                "Mood": "Tense",
                "Barrier_or_Reason": "Private_Barrier_Token",
            }
            for i in range(10)
        ]
    )
    ml_10 = build_private_smoking_ml(process_silver_smoking(raw_10))
    assert ml_10["Private_Smoking_ML_Summary"].iloc[0]["status"] == "Level 0: Descriptive Risk Heatmap"
    assert ml_10["Private_Smoking_ML_Summary"].iloc[0]["selected_algorithm"] == "None_Gated"
    assert np.isnan(ml_10["Private_Smoking_ML_Summary"].iloc[0]["logistic_pr_auc"])

    # 3. Gate 1: 48 events (24 Smoked, 24 Resisted -> 30..99 & min_class >= 10) -> L1 Logistic unlocked
    rows_48 = []
    for i in range(48):
        is_smk = (i % 2 == 0)
        rows_48.append(
            {
                "Date": f"2026-10-{(i // 2) + 1:02d}",
                "Time": "02:30" if is_smk else "14:00",
                "Event_Type": "Smoked" if is_smk else "Craving Resisted",
                "Count": "1" if is_smk else "0",
                "Craving_Intensity": "5" if is_smk else "2",
                "Delay_Minutes": "3" if is_smk else "22",
                "Trigger": "Stress" if is_smk else "Habit",
                "Mood": "Anxious" if is_smk else "Calm",
                "Barrier_or_Reason": "None" if is_smk else "Walked",
            }
        )
    ml_48 = build_private_smoking_ml(process_silver_smoking(pd.DataFrame(rows_48)))
    s48 = ml_48["Private_Smoking_ML_Summary"].iloc[0]
    assert s48["selected_algorithm"] == "L1_LogisticRegression_Balanced"
    assert np.isfinite(s48["logistic_pr_auc"]) and s48["logistic_pr_auc"] > 0.6
    assert np.isfinite(s48["logistic_roc_auc"])
    assert np.isfinite(s48["logistic_balanced_acc"])
    assert np.isnan(s48["rf_pr_auc"])
    # Delay stats unlocked (48 >= 15)
    dly48 = ml_48["Private_Delay_Outcome_Stats"]
    assert (dly48["gate_status"] == "Unlocked_Delay_Comparison").all()
    assert dly48["resist_rate"].notna().all()

    # 4. Gate 2: 120 events (60 Smoked, 60 Resisted -> >=100 & min_class >= 25) -> RF + 3D SHAP [:,:,1] unlocked!
    rows_120 = []
    base_dt = pd.Timestamp("2026-08-01")
    for i in range(120):
        is_smk = (i % 2 == 0) or (i % 5 == 0)
        ds = (base_dt + pd.Timedelta(days=i // 2)).strftime("%Y-%m-%d")
        rows_120.append(
            {
                "Event_ID": f"EV-{i:04d}",
                "Date": ds,
                "Time": f"{(2 if is_smk else 15):02d}:{(i % 50):02d}",
                "Event_Type": "Smoked" if is_smk else "Craving Resisted",
                "Count": "1" if is_smk else "0",
                "Craving_Intensity": "5" if is_smk else "2",
                "Delay_Minutes": "2" if is_smk else "25",
                "Trigger": "Late_Work_Secret" if is_smk else "Coffee",
                "Mood": "Overwhelmed" if is_smk else "Focused",
                "Barrier_or_Reason": "Stepped_Out" if is_smk else "Breathing",
            }
        )
    priv_dir = tmp_path / "priv_ml_in" / "Micro Journal"
    priv_dir.mkdir(parents=True)
    pd.DataFrame(rows_120).to_csv(priv_dir / "Smoking_Log.csv", index=False)

    rc = run_private_ml_main(
        [
            "--source",
            "csv",
            "--data-dir",
            str(tmp_path / "priv_ml_in"),
            "--output-dir",
            str(tmp_path / "priv_ml_out"),
        ]
    )
    assert rc == 0
    out_txt = capsys.readouterr().out
    assert "Late_Work_Secret" not in out_txt
    assert "Stepped_Out" not in out_txt

    s120 = pd.read_csv(tmp_path / "priv_ml_out" / "Private_Smoking_ML_Summary.csv").iloc[0]
    assert np.isfinite(s120["rf_pr_auc"])
    assert s120["selected_algorithm"] in ("RandomForestClassifier_SHAP", "L1_LogisticRegression_Balanced")


def test_ml_package_privacy_ast_scan_and_journal_nlp_pipeline(
    tmp_path: Path,
) -> None:
    # 1. Static AST + literal scan across ALL files in src/health_engine/ml/
    ml_dir = Path(__file__).resolve().parents[1] / "src" / "health_engine" / "ml"
    py_files = sorted(ml_dir.glob("*.py"))
    assert len(py_files) >= 5

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

    # 2. Verify Journal NLP theme tagging + Episode Rules lift gating (<15 suppresses lift, >=15 unlocks)
    thresholds_cfg, _ = load_engine_configs()
    themes_cfg = load_journal_themes_config()
    jm = pd.DataFrame(
        [
            {
                "Event_ID": "MJ-1",
                "Source_Event_Key": "EP-1",
                "date": "2026-11-26",
                "time_ist": "04:00",
                "Mood_Tags": "Frustrated, Overwhelmed",
                "Context_Tag": "Work",
                "Summary": "Finishing work at 4:00 AM with right shoulder pain",
            }
        ]
    )
    sym = pd.DataFrame(
        [
            {
                "Event_ID": "SYM-1",
                "Source_Event_Key": "EP-1",
                "date": "2026-11-26",
                "time_ist": "04:00",
                "Symptom": "Right Shoulder Pain",
                "Normalized_Symptom": "Right Shoulder Pain",
                "Severity": "Severe",
                "Context_Tag": "Work",
                "Notes": "",
            }
        ]
    )
    gd = pd.DataFrame({"date": ["2026-11-25", "2026-11-26"], "days_since_shot": [6.0, 0.0], "micro_journal_status": ["Not_Logged", "Present"]})

    themes_df, rules_df, enriched_gd = build_gold_journal_nlp(
        gold_daily=gd,
        silver_journal_mood=jm,
        silver_meds=pd.DataFrame(),
        silver_symptoms=sym,
        silver_nutr_log=pd.DataFrame(),
        thresholds_cfg=thresholds_cfg,
        themes_cfg=themes_cfg,
    )
    assert "Work_Overwhelm" in themes_df.iloc[0]["detected_themes"]
    assert "Late_Night_Work" in themes_df.iloc[0]["detected_themes"]
    assert bool(themes_df.iloc[0]["human_reviewed_flag"]) is False
    assert themes_df.iloc[0]["tfidf_discovery_status"] == "Gated_Below_N100_Notes"
    # Unlogged 2026-11-25 stays strictly NaN while logged 2026-11-26 gets numeric count
    assert np.isnan(enriched_gd.loc[enriched_gd["date"] == "2026-11-25", "theme_work_overwhelm_cnt"].iloc[0])
    assert enriched_gd.loc[enriched_gd["date"] == "2026-11-26", "theme_work_overwhelm_cnt"].iloc[0] == 1.0
    # Lift suppressed when total_episodes < 15
    assert rules_df["lift"].isna().all()
    assert (rules_df["maturity_level"] == "Level 0: Descriptive Episode Log").all()

    # 3. Full run_v25_pipeline integration check on ./sample_data
    repo_sample = Path(__file__).resolve().parents[1] / "sample_data"
    adapter = LocalCSVAdapter(repo_sample, tmp_path / "v25_out")
    out = run_v25_pipeline(adapter, persist=True)
    assert not out["Gold_ML_Model_Summary"].empty
    assert not out["Gold_Phenotype_Profiles"].empty
    assert not out["Model_Run_Audit"].empty
