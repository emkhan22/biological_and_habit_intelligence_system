"""Deterministic synthetic CSV fixtures covering every V1.5 Silver & Gold edge case."""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import pytest


def _write_csv(base: Path, workbook: str, tab: str, rows: list[dict[str, object]]) -> None:
    folder = base / workbook
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(folder / f"{tab}.csv", index=False)


def populate_synthetic_health_dataset(base_dir: Path) -> Path:
    """Create a complete deterministic synthetic dataset in `base_dir`."""
    # 1. HealthConnectDaily!Sleep_Sessions
    # Includes:
    # - Unstaged Google Fit session (stage_coverage_pct=0) -> MUST be excluded
    # - Short Fitbit nap <120m (asleep=90m) -> MUST be flagged nap_or_partial_wear & excluded from nocturnal
    # - Multiple valid nocturnal sessions on 2026-11-23 -> longest nocturnal selected
    # - Pre-Oct baseline sessions (2026-06-10..2026-06-14) + Active GLP-1 sessions (2026-11-22..2026-11-26)
    sleep_sessions = [
        {
            "date": "2026-06-10",
            "session_id": "101",
            "start_time_ist": "2026-06-09 23:00:00",
            "end_time_ist": "2026-06-10 07:00:00",
            "session_duration_min": "480",
            "sleep_onset_latency_min_est": "10",
            "awake_episode_count": "2",
            "awake_min": "30",
            "longest_awake_episode_min": "20",
            "light_min": "250",
            "deep_min": "100",
            "rem_min": "100",
            "unspecified_sleep_min": "0",
            "sleep_hr_avg": "64.0",
            "sleep_hr_median": "63",
            "sleep_hr_p95": "70",
            "sleep_hr_max": "85",
            "sleep_hr_sample_count": "450",
            "source_app": "Fitbit",
            "stage_coverage_pct": "100",
            "selection_note": "Fitbit staged",
        },
        {
            "date": "2026-06-11",
            "session_id": "102",
            "start_time_ist": "2026-06-10 23:30:00",
            "end_time_ist": "2026-06-11 07:30:00",
            "session_duration_min": "480",
            "sleep_onset_latency_min_est": "12",
            "awake_episode_count": "3",
            "awake_min": "40",
            "longest_awake_episode_min": "20",
            "light_min": "240",
            "deep_min": "110",
            "rem_min": "90",
            "unspecified_sleep_min": "0",
            "sleep_hr_avg": "65.0",
            "sleep_hr_median": "64",
            "sleep_hr_p95": "72",
            "sleep_hr_max": "88",
            "sleep_hr_sample_count": "440",
            "source_app": "Fitbit",
            "stage_coverage_pct": "100",
            "selection_note": "Fitbit staged",
        },
        {
            "date": "2026-06-12",
            "session_id": "103",
            "start_time_ist": "2026-06-12 00:30:00",
            "end_time_ist": "2026-06-12 08:00:00",
            "session_duration_min": "450",
            "sleep_onset_latency_min_est": "8",
            "awake_episode_count": "2",
            "awake_min": "30",
            "longest_awake_episode_min": "15",
            "light_min": "230",
            "deep_min": "95",
            "rem_min": "95",
            "unspecified_sleep_min": "0",
            "sleep_hr_avg": "63.0",
            "sleep_hr_median": "62",
            "sleep_hr_p95": "69",
            "sleep_hr_max": "82",
            "sleep_hr_sample_count": "420",
            "source_app": "Fitbit",
            "stage_coverage_pct": "100",
            "selection_note": "Fitbit staged",
        },
        # Unstaged Google Fit session -> must be excluded
        {
            "date": "2026-11-22",
            "session_id": "700",
            "start_time_ist": "2026-11-21 23:30:00",
            "end_time_ist": "2026-11-22 07:30:00",
            "session_duration_min": "480",
            "sleep_onset_latency_min_est": "",
            "awake_episode_count": "0",
            "awake_min": "0",
            "longest_awake_episode_min": "0",
            "light_min": "0",
            "deep_min": "0",
            "rem_min": "0",
            "unspecified_sleep_min": "480",
            "sleep_hr_avg": "",
            "sleep_hr_median": "",
            "sleep_hr_p95": "",
            "sleep_hr_max": "",
            "sleep_hr_sample_count": "0",
            "source_app": "Fit",
            "stage_coverage_pct": "0",
            "selection_note": "Google Fit unstaged",
        },
        # Short Fitbit nap (<120 min asleep) -> must be excluded from nocturnal
        {
            "date": "2026-11-23",
            "session_id": "701",
            "start_time_ist": "2026-11-23 15:00:00",
            "end_time_ist": "2026-11-23 16:40:00",
            "session_duration_min": "100",
            "sleep_onset_latency_min_est": "5",
            "awake_episode_count": "1",
            "awake_min": "10",
            "longest_awake_episode_min": "10",
            "light_min": "50",
            "deep_min": "20",
            "rem_min": "20",
            "unspecified_sleep_min": "0",
            "sleep_hr_avg": "70.0",
            "sleep_hr_median": "69",
            "sleep_hr_p95": "75",
            "sleep_hr_max": "84",
            "sleep_hr_sample_count": "2400",
            "source_app": "Fitbit",
            "stage_coverage_pct": "100",
            "selection_note": "Afternoon nap",
        },
        # Primary nocturnal session on 2026-11-23 (bedtime 23:00 = -1.0; light=180m=3h, deep=60m=1h, rem=120m=2h, asleep=360m=6h)
        {
            "date": "2026-11-23",
            "session_id": "702",
            "start_time_ist": "2026-11-22 23:00:00",
            "end_time_ist": "2026-11-23 05:30:00",
            "session_duration_min": "390",
            "sleep_onset_latency_min_est": "10",
            "awake_episode_count": "3",
            "awake_min": "30",
            "longest_awake_episode_min": "15",
            "light_min": "180",
            "deep_min": "60",
            "rem_min": "120",
            "unspecified_sleep_min": "0",
            "sleep_hr_avg": "69.0",
            "sleep_hr_median": "68",
            "sleep_hr_p95": "78",
            "sleep_hr_max": "96",
            "sleep_hr_sample_count": "9000",
            "source_app": "Fitbit",
            "stage_coverage_pct": "100",
            "selection_note": "Primary nocturnal",
        },
        # Late-night bedtime on 2026-11-24 (02:00 IST -> +2.0 rel mid)
        {
            "date": "2026-11-24",
            "session_id": "703",
            "start_time_ist": "2026-11-24 02:00:00",
            "end_time_ist": "2026-11-24 08:30:00",
            "session_duration_min": "390",
            "sleep_onset_latency_min_est": "10",
            "awake_episode_count": "2",
            "awake_min": "30",
            "longest_awake_episode_min": "20",
            "light_min": "210",
            "deep_min": "60",
            "rem_min": "90",
            "unspecified_sleep_min": "0",
            "sleep_hr_avg": "71.0",
            "sleep_hr_median": "70",
            "sleep_hr_p95": "80",
            "sleep_hr_max": "98",
            "sleep_hr_sample_count": "9200",
            "source_app": "Fitbit",
            "stage_coverage_pct": "100",
            "selection_note": "Late bedtime",
        },
        {
            "date": "2026-11-25",
            "session_id": "704",
            "start_time_ist": "2026-11-25 01:00:00",
            "end_time_ist": "2026-11-25 08:00:00",
            "session_duration_min": "420",
            "sleep_onset_latency_min_est": "8",
            "awake_episode_count": "2",
            "awake_min": "30",
            "longest_awake_episode_min": "18",
            "light_min": "210",
            "deep_min": "90",
            "rem_min": "90",
            "unspecified_sleep_min": "0",
            "sleep_hr_avg": "68.0",
            "sleep_hr_median": "67",
            "sleep_hr_p95": "76",
            "sleep_hr_max": "92",
            "sleep_hr_sample_count": "9800",
            "source_app": "Fitbit",
            "stage_coverage_pct": "100",
            "selection_note": "Fitbit staged",
        },
    ]
    _write_csv(base_dir, "HealthConnectDaily", "Sleep_Sessions", sleep_sessions)

    _write_csv(
        base_dir,
        "HealthConnectDaily",
        "Sleep_Stage_Timeline",
        [
            {
                "date": "2026-11-23",
                "session_id": "702",
                "stage_start_ist": "2026-11-22 23:00:00",
                "stage_end_ist": "2026-11-23 02:00:00",
                "stage_code": "4",
                "stage_name": "Light",
            },
            {
                "date": "2026-11-23",
                "session_id": "702",
                "stage_start_ist": "2026-11-23 02:00:00",
                "stage_end_ist": "2026-11-23 03:00:00",
                "stage_code": "5",
                "stage_name": "Deep",
            },
            {
                "date": "2026-11-23",
                "session_id": "702",
                "stage_start_ist": "2026-11-23 03:00:00",
                "stage_end_ist": "2026-11-23 05:00:00",
                "stage_code": "6",
                "stage_name": "REM",
            },
            {
                "date": "2026-11-23",
                "session_id": "702",
                "stage_start_ist": "2026-11-23 05:00:00",
                "stage_end_ist": "2026-11-23 05:30:00",
                "stage_code": "1",
                "stage_name": "Awake",
            },
        ],
    )

    # Sleep HR Events on 2026-11-23 session 702:
    # 3 Light spikes (over 3.0 Light hrs -> 1.0/hr), 2 REM spikes (over 2.0 REM hrs -> 1.0/hr),
    # 1 Deep spike (over 1.0 Deep hr -> 1.0/hr), 2 Awake/transition spikes -> total 8 spikes over 6.0 asleep hrs = 1.3333/hr
    _write_csv(
        base_dir,
        "HealthConnectDaily",
        "Sleep_HR_Events",
        [
            {"date": "2026-11-23", "session_id": "702", "event_start_ist": "2026-11-22 23:30:00", "event_end_ist": "2026-11-22 23:30:45", "duration_sec": "45", "baseline_hr_bpm": "66", "elevated_median_hr_bpm": "78", "peak_hr_bpm": "84", "delta_bpm": "12", "stage_context": "Light", "method_note": "algo"},
            {"date": "2026-11-23", "session_id": "702", "event_start_ist": "2026-11-23 00:15:00", "event_end_ist": "2026-11-23 00:15:50", "duration_sec": "50", "baseline_hr_bpm": "65", "elevated_median_hr_bpm": "79", "peak_hr_bpm": "85", "delta_bpm": "14", "stage_context": "Light", "method_note": "algo"},
            {"date": "2026-11-23", "session_id": "702", "event_start_ist": "2026-11-23 01:10:00", "event_end_ist": "2026-11-23 01:10:40", "duration_sec": "40", "baseline_hr_bpm": "65", "elevated_median_hr_bpm": "76", "peak_hr_bpm": "82", "delta_bpm": "11", "stage_context": "Light", "method_note": "algo"},
            {"date": "2026-11-23", "session_id": "702", "event_start_ist": "2026-11-23 02:20:00", "event_end_ist": "2026-11-23 02:20:35", "duration_sec": "35", "baseline_hr_bpm": "64", "elevated_median_hr_bpm": "75", "peak_hr_bpm": "80", "delta_bpm": "11", "stage_context": "Deep", "method_note": "algo"},
            {"date": "2026-11-23", "session_id": "702", "event_start_ist": "2026-11-23 03:30:00", "event_end_ist": "2026-11-23 03:31:00", "duration_sec": "60", "baseline_hr_bpm": "68", "elevated_median_hr_bpm": "84", "peak_hr_bpm": "92", "delta_bpm": "16", "stage_context": "REM", "method_note": "algo"},
            {"date": "2026-11-23", "session_id": "702", "event_start_ist": "2026-11-23 04:15:00", "event_end_ist": "2026-11-23 04:15:50", "duration_sec": "50", "baseline_hr_bpm": "69", "elevated_median_hr_bpm": "85", "peak_hr_bpm": "96", "delta_bpm": "16", "stage_context": "REM", "method_note": "algo"},
            {"date": "2026-11-23", "session_id": "702", "event_start_ist": "2026-11-23 05:05:00", "event_end_ist": "2026-11-23 05:05:40", "duration_sec": "40", "baseline_hr_bpm": "72", "elevated_median_hr_bpm": "88", "peak_hr_bpm": "94", "delta_bpm": "16", "stage_context": "Awake", "method_note": "algo"},
            {"date": "2026-11-23", "session_id": "702", "event_start_ist": "2026-11-23 04:59:30", "event_end_ist": "2026-11-23 05:00:10", "duration_sec": "40", "baseline_hr_bpm": "70", "elevated_median_hr_bpm": "82", "peak_hr_bpm": "88", "delta_bpm": "12", "stage_context": "Light / Awake", "method_note": "algo"},
        ],
    )

    # 2. Vitals + Heart_Rate_Hourly across Pre-May 2026 (low sample count, 20h covered -> PASS)
    #    and Post-May 2026 (high sample count, 20h covered -> PASS; 10h covered -> FAIL)
    vitals_rows = [
        {"date": "2026-04-15", "avg_hr": "76", "min_hr": "58", "max_hr": "118", "hr_sample_count": "1180", "p95_hr": "96", "resting_hr": "66", "hrv_rmssd_ms": "38", "breathing_rate_rpm": "15", "spo2_pct": ""},
        {"date": "2026-06-10", "avg_hr": "74", "min_hr": "56", "max_hr": "120", "hr_sample_count": "28500", "p95_hr": "94", "resting_hr": "64", "hrv_rmssd_ms": "42", "breathing_rate_rpm": "15", "spo2_pct": ""},
        {"date": "2026-06-11", "avg_hr": "75", "min_hr": "57", "max_hr": "122", "hr_sample_count": "29100", "p95_hr": "95", "resting_hr": "65", "hrv_rmssd_ms": "40", "breathing_rate_rpm": "15", "spo2_pct": ""},
        {"date": "2026-06-12", "avg_hr": "73", "min_hr": "55", "max_hr": "115", "hr_sample_count": "27900", "p95_hr": "92", "resting_hr": "63", "hrv_rmssd_ms": "44", "breathing_rate_rpm": "14", "spo2_pct": ""},
        {"date": "2026-11-22", "avg_hr": "78", "min_hr": "60", "max_hr": "125", "hr_sample_count": "30200", "p95_hr": "98", "resting_hr": "68", "hrv_rmssd_ms": "35", "breathing_rate_rpm": "16", "spo2_pct": ""},
        {"date": "2026-11-23", "avg_hr": "79", "min_hr": "61", "max_hr": "128", "hr_sample_count": "29800", "p95_hr": "99", "resting_hr": "69", "hrv_rmssd_ms": "33", "breathing_rate_rpm": "16", "spo2_pct": ""},
        # High sample count (24000) but only 10 hours covered -> MUST FAIL vitals_valid_coverage
        {"date": "2026-11-24", "avg_hr": "82", "min_hr": "65", "max_hr": "130", "hr_sample_count": "24000", "p95_hr": "105", "resting_hr": "71", "hrv_rmssd_ms": "30", "breathing_rate_rpm": "16", "spo2_pct": ""},
    ]
    _write_csv(base_dir, "HealthConnectDaily", "Vitals", vitals_rows)

    hr_hourly_rows: list[dict[str, object]] = []
    for ds, n_hrs, per_hr_samples, base_hr in [
        ("2026-04-15", 20, 59, 76.0),     # pre-May 2026 1-min era, 20h -> valid
        ("2026-06-10", 22, 1300, 74.0),
        ("2026-06-11", 21, 1350, 75.0),
        ("2026-06-12", 20, 1390, 73.0),
        ("2026-11-22", 21, 1400, 78.0),
        ("2026-11-23", 19, 1500, 79.0),
        ("2026-11-24", 10, 2400, 82.0),   # only 10h covered -> invalid!
    ]:
        for h in range(n_hrs):
            hr_hourly_rows.append(
                {
                    "date": ds,
                    "hour_ist": str(h),
                    "avg_hr": str(base_hr + (4.0 if 9 <= h <= 18 else -4.0)),
                    "median_hr": str(base_hr),
                    "min_hr": "60",
                    "max_hr": "105",
                    "sample_count": str(per_hr_samples),
                }
            )
    _write_csv(base_dir, "HealthConnectDaily", "Heart_Rate_Hourly", hr_hourly_rows)

    # 3. Steps, Workouts (with duplicate), Nutrition, Nutrition_Log, Body_Composition, Body_Measurements
    _write_csv(
        base_dir,
        "HealthConnectDaily",
        "Steps",
        [
            {"date": "2026-06-10", "steps": "6500", "fitbit_steps": "6500", "phone_fallback_steps": "0"},
            {"date": "2026-06-11", "steps": "7200", "fitbit_steps": "7200", "phone_fallback_steps": "0"},
            {"date": "2026-06-12", "steps": "8100", "fitbit_steps": "8100", "phone_fallback_steps": "0"},
            {"date": "2026-11-22", "steps": "5400", "fitbit_steps": "5400", "phone_fallback_steps": "0"},
            {"date": "2026-11-23", "steps": "7800", "fitbit_steps": "7800", "phone_fallback_steps": "0"},
        ],
    )

    _write_csv(
        base_dir,
        "HealthConnectDaily",
        "Workouts",
        [
            {"date": "2026-11-23", "start_time_ist": "2026-11-23 18:00:00", "end_time_ist": "2026-11-23 18:45:00", "duration_mins": "45", "exercise_type": "Strength training", "title": "Upper Body", "source_app": "Fitbit", "avg_hr": "118", "max_hr": "145", "moderate_mins": "20", "vigorous_mins": "15", "peak_mins": "0", "zone_mins": "35", "notes": ""},
            # Exact duplicate -> must be removed
            {"date": "2026-11-23", "start_time_ist": "2026-11-23 18:00:00", "end_time_ist": "2026-11-23 18:45:00", "duration_mins": "45", "exercise_type": "Strength training", "title": "Upper Body", "source_app": "Fitbit", "avg_hr": "118", "max_hr": "145", "moderate_mins": "20", "vigorous_mins": "15", "peak_mins": "0", "zone_mins": "35", "notes": ""},
        ],
    )

    _write_csv(
        base_dir,
        "HealthConnectDaily",
        "Nutrition",
        [
            {"date": "2026-11-22", "calories_kcal": "1650", "protein_g": "95", "carbs_g": "140", "fat_g": "68", "fiber_g": "22", "sugar_g": "18", "hydration_liters": "2.4", "entries_logged": "4"},
            {"date": "2026-11-23", "calories_kcal": "1520", "protein_g": "105", "carbs_g": "130", "fat_g": "52", "fiber_g": "28", "sugar_g": "15", "hydration_liters": "1.6", "entries_logged": "3"},
        ],
    )

    _write_csv(
        base_dir,
        "HealthConnectDaily",
        "Nutrition_Log",
        [
            {"date": "2026-11-22", "time_ist": "09:50", "meal_type": "Breakfast", "meal_name": "Eggs & Toast", "calories_kcal": "420", "protein_g": "28", "carbs_g": "30", "fat_g": "18", "fiber_g": "5", "sugar_g": "3", "source_app": "HealthifyMe"},
            {"date": "2026-11-22", "time_ist": "13:30", "meal_type": "Lunch", "meal_name": "Chicken Bowl", "calories_kcal": "630", "protein_g": "45", "carbs_g": "60", "fat_g": "20", "fiber_g": "10", "sugar_g": "5", "source_app": "HealthifyMe"},
            {"date": "2026-11-22", "time_ist": "21:30", "meal_type": "Dinner", "meal_name": "Late Paneer Curry", "calories_kcal": "600", "protein_g": "22", "carbs_g": "50", "fat_g": "30", "fiber_g": "7", "sugar_g": "10", "source_app": "HealthifyMe"},
            {"date": "2026-11-23", "time_ist": "09:15", "meal_type": "Breakfast", "meal_name": "Whey + Oats", "calories_kcal": "400", "protein_g": "35", "carbs_g": "45", "fat_g": "8", "fiber_g": "8", "sugar_g": "4", "source_app": "HealthifyMe"},
            {"date": "2026-11-23", "time_ist": "14:00", "meal_type": "Lunch", "meal_name": "Salmon & Rice", "calories_kcal": "620", "protein_g": "42", "carbs_g": "55", "fat_g": "24", "fiber_g": "10", "sugar_g": "4", "source_app": "HealthifyMe"},
            {"date": "2026-11-23", "time_ist": "19:30", "meal_type": "Dinner", "meal_name": "Grilled Chicken Salad", "calories_kcal": "500", "protein_g": "28", "carbs_g": "30", "fat_g": "20", "fiber_g": "10", "sugar_g": "7", "source_app": "HealthifyMe"},
        ],
    )

    _write_csv(
        base_dir,
        "HealthConnectDaily",
        "Body_Composition",
        [
            {"date": "2026-10-07", "time_ist": "08:00", "weight_kg": "108.3", "bmi": "40.8", "body_fat_pct": "44.2", "fat_mass_kg": "47.9", "muscle_mass_kg": "56.8", "skeletal_muscle_pct": "34.1", "visceral_fat": "14", "bone_mass_kg": "3.6", "protein_mass_kg": "11.2"},
            {"date": "2026-11-22", "time_ist": "08:15", "weight_kg": "104.1", "bmi": "39.2", "body_fat_pct": "42.8", "fat_mass_kg": "44.6", "muscle_mass_kg": "56.0", "skeletal_muscle_pct": "34.8", "visceral_fat": "13", "bone_mass_kg": "3.5", "protein_mass_kg": "11.1"},
        ],
    )

    _write_csv(
        base_dir,
        "HealthConnectDaily",
        "Body_Measurements",
        [
            {"date": "2026-11-22", "time_ist": "08:20", "waist_cm": "105.95", "hip_cm": "118.0"},
        ],
    )

    # 4. Menstrual Cycle tabs across Flo (<=2026-11-05) and Fitbit/HC (>=2026-11-06) + Spotting
    _write_csv(
        base_dir,
        "HealthConnectDaily",
        "Cycles",
        [
            {"cycle_id": "cyc-aug", "cycle_start": "2026-08-13", "cycle_end": "2026-08-17", "span_days": "5", "cycle_length_days": "83", "marked_flow_days": "5", "unmarked_flow_days": "0", "source_app": "Flo"},
            {"cycle_id": "cyc-nov", "cycle_start": "2026-11-04", "cycle_end": "2026-11-05", "span_days": "2", "cycle_length_days": "", "marked_flow_days": "1", "unmarked_flow_days": "1", "source_app": "Flo"},
        ],
    )
    _write_csv(
        base_dir,
        "HealthConnectDaily",
        "Period_Flow_Daily",
        [
            {"date": "2026-11-04", "cycle_start": "2026-11-04", "cycle_day": "1", "flow_code": "2", "flow_level": "Medium"},
            {"date": "2026-11-05", "cycle_start": "2026-11-04", "cycle_day": "2", "flow_code": "-1", "flow_level": "Flow Unmarked"},
        ],
    )
    _write_csv(
        base_dir,
        "HealthConnectDaily",
        "Archived_Menstrual_Cycle",
        [
            {"date": "2026-10-20", "record_type": "Spotting", "end_date": "", "flow_level": "Spotting", "source_app": "Flo"},
        ],
    )
    _write_csv(
        base_dir,
        "HealthConnectDaily",
        "Menstrual_Cycle",
        [
            {"date": "2026-11-06", "record_type": "Period", "end_date": "2026-11-08", "flow_level": "Light", "source_app": "Fitbit"},
        ],
    )

    # 5. Glapp Data
    _write_csv(
        base_dir,
        "Glapp Data",
        "shots",
        [
            {"date": "2026-10-21", "time": "08:00", "medication": "tirzepatide", "dose_mg": "2.5"},
            {"date": "2026-11-22", "time": "08:30", "medication": "tirzepatide", "dose_mg": "5.0"},
        ],
    )
    _write_csv(
        base_dir,
        "Glapp Data",
        "mood",
        [
            {"date": "2026-11-22", "mood": "4", "hunger": "1", "food_noise": "1", "feeling": "steady"},
            {"date": "2026-11-23", "mood": "3", "hunger": "1", "food_noise": "0", "feeling": "mildly tired"},
            {"date": "2026-11-24", "mood": "4", "hunger": "2", "food_noise": "1", "feeling": "good"},
        ],
    )
    _write_csv(
        base_dir,
        "Glapp Data",
        "side_effects",
        [
            {"date": "2026-11-23", "side_effect": "nausea", "level": "2"},
            {"date": "2026-11-24", "side_effect": "constipation", "level": "1"},
        ],
    )

    # 6. Micro Journal (ONLY the 3 non-smoking tabs)
    _write_csv(
        base_dir,
        "Micro Journal",
        "Meds_Log",
        [
            {"Date": "2026-11-22", "Time": "09:00", "Med_or_Supplement": "Thyronorm", "Dose": "75 mcg", "Timing_Context": "Morning", "Status": "Taken", "Notes": ""},
            # Exact duplicate morning Thyronorm -> must be deduplicated
            {"Date": "2026-11-22", "Time": "09:00", "Med_or_Supplement": "Thyronorm", "Dose": "75 mcg", "Timing_Context": "Morning", "Status": "Taken", "Notes": "duplicate entry"},
            {"Date": "2026-11-22", "Time": "14:00", "Med_or_Supplement": "Vitamin D3 (D-Rise)", "Dose": "60000 IU", "Timing_Context": "After Lunch", "Status": "Taken", "Notes": ""},
            {"Date": "2026-11-22", "Time": "22:00", "Med_or_Supplement": "Magnesium Glycinate", "Dose": "400 mg", "Timing_Context": "Night", "Status": "Skipped", "Notes": "nauseous on shot day"},
            {"Date": "2026-11-23", "Time": "09:00", "Med_or_Supplement": "Thyronorm", "Dose": "75 mcg", "Timing_Context": "Morning", "Status": "Taken", "Notes": ""},
        ],
    )
    _write_csv(
        base_dir,
        "Micro Journal",
        "Symptoms_Log",
        [
            {"Date": "2026-11-23", "Time": "07:30", "Symptom": "Right Foot Cramps", "Severity": "Moderate", "Context_Tag": "Morning", "Notes": ""},
            # Plural/singular duplicate at same Date+Time -> must normalize & deduplicate
            {"Date": "2026-11-23", "Time": "07:30", "Symptom": "Right Foot Cramp", "Severity": "Moderate", "Context_Tag": "Morning", "Notes": ""},
            {"Date": "2026-11-23", "Time": "08:15", "Symptom": "Acid Reflux", "Severity": "Mild", "Context_Tag": "Morning", "Notes": ""},
            {"Date": "2026-11-24", "Time": "03:30", "Symptom": "Right Shoulder Pain", "Severity": "Severe", "Context_Tag": "Work", "Notes": "Working late at 03:30 AM"},
            {"Date": "2026-11-24", "Time": "10:00", "Symptom": "Headaches", "Severity": "Mild", "Context_Tag": "Morning", "Notes": ""},
        ],
    )
    _write_csv(
        base_dir,
        "Micro Journal",
        "Journal_Mood",
        [
            {"Date": "2026-11-24", "Time": "03:30", "Entry_Type": "Snippet", "Mood_Tags": "Focused", "Context_Tag": "Work", "Summary": "Late night work sprint"},
        ],
    )

    # 7. AggregatedMedicalRecords!Lab_Results
    # Covers:
    # - Platelet Count in cells/cumm (250000 -> 250.0 10^3/µL)
    # - WBC Count in cells/µL (7400 -> 7.4 10^3/µL)
    # - Total T3 in ng/dL (115 -> 1.15 ng/mL)
    # - Same-day verified fasting draw on 2026-10-10 (FBS=95, Fasting Insulin=16.2, TG=150, HDL=45, Total Chol=195, MCV=82, RBC=4.8, AST=28, ALT=36, Neut=60, Lymph=30)
    # - Non-fasting eAG on 2026-10-15 + Fasting Insulin on 2026-10-16 (mismatched date & non-fasting -> MUST NOT produce HOMA-IR or TyG!)
    # - Auto-date corrupted urinalysis ("2026-05-02" in Original_Value) -> MUST quarantine!
    lab_rows = [
        {"Record_ID": "LAB-001", "Original_Test_Name": "Platelet Count", "Standardized_Test_Name": "Platelet Count", "Original_Value": "250000", "Numeric_Value": "250000", "Unit": "cells/cumm", "Standardized_Unit": "cells/µL", "Specimen_Type": "EDTA Blood", "Collection_DateTime": "2026-10-10 08:00", "Source_Page": "1"},
        {"Record_ID": "LAB-002", "Original_Test_Name": "Total Leucocyte Count", "Standardized_Test_Name": "WBC Count (Total Leucocyte Count)", "Original_Value": "7400", "Numeric_Value": "7400", "Unit": "cells/cumm", "Standardized_Unit": "cells/µL", "Specimen_Type": "EDTA Blood", "Collection_DateTime": "2026-10-10 08:00", "Source_Page": "1"},
        {"Record_ID": "LAB-003", "Original_Test_Name": "Triiodothyronine Total", "Standardized_Test_Name": "Total T3", "Original_Value": "115", "Numeric_Value": "115", "Unit": "ng/dl", "Standardized_Unit": "ng/dL", "Specimen_Type": "Serum", "Collection_DateTime": "2026-10-10 08:00", "Source_Page": "2"},
        {"Record_ID": "LAB-004", "Original_Test_Name": "Fasting Blood Sugar", "Standardized_Test_Name": "Fasting Blood Glucose (FBS)", "Original_Value": "95", "Numeric_Value": "95", "Unit": "mg/dL", "Standardized_Unit": "mg/dL", "Specimen_Type": "Fluoride Plasma", "Collection_DateTime": "2026-10-10 08:00", "Source_Page": "2"},
        {"Record_ID": "LAB-005", "Original_Test_Name": "Fasting Insulin", "Standardized_Test_Name": "Fasting Insulin", "Original_Value": "16.2", "Numeric_Value": "16.2", "Unit": "mcU/mL", "Standardized_Unit": "µU/mL", "Specimen_Type": "Serum", "Collection_DateTime": "2026-10-10 08:00", "Source_Page": "2"},
        {"Record_ID": "LAB-006", "Original_Test_Name": "Triglycerides", "Standardized_Test_Name": "Triglycerides", "Original_Value": "150", "Numeric_Value": "150", "Unit": "mg/dL", "Standardized_Unit": "mg/dL", "Specimen_Type": "Serum", "Collection_DateTime": "2026-10-10 08:00", "Source_Page": "3"},
        {"Record_ID": "LAB-007", "Original_Test_Name": "HDL Cholesterol", "Standardized_Test_Name": "HDL Cholesterol", "Original_Value": "45", "Numeric_Value": "45", "Unit": "mg/dL", "Standardized_Unit": "mg/dL", "Specimen_Type": "Serum", "Collection_DateTime": "2026-10-10 08:00", "Source_Page": "3"},
        {"Record_ID": "LAB-008", "Original_Test_Name": "Total Cholesterol", "Standardized_Test_Name": "Total Cholesterol", "Original_Value": "195", "Numeric_Value": "195", "Unit": "mg/dL", "Standardized_Unit": "mg/dL", "Specimen_Type": "Serum", "Collection_DateTime": "2026-10-10 08:00", "Source_Page": "3"},
        {"Record_ID": "LAB-009", "Original_Test_Name": "MCV", "Standardized_Test_Name": "MCV", "Original_Value": "84", "Numeric_Value": "84", "Unit": "fL", "Standardized_Unit": "fL", "Specimen_Type": "EDTA Blood", "Collection_DateTime": "2026-10-10 08:00", "Source_Page": "1"},
        {"Record_ID": "LAB-010", "Original_Test_Name": "RBC Count", "Standardized_Test_Name": "RBC Count", "Original_Value": "4.8", "Numeric_Value": "4.8", "Unit": "mil/cumm", "Standardized_Unit": "10^6/µL", "Specimen_Type": "EDTA Blood", "Collection_DateTime": "2026-10-10 08:00", "Source_Page": "1"},
        {"Record_ID": "LAB-011", "Original_Test_Name": "SGOT (AST)", "Standardized_Test_Name": "AST (SGOT)", "Original_Value": "28", "Numeric_Value": "28", "Unit": "U/L", "Standardized_Unit": "U/L", "Specimen_Type": "Serum", "Collection_DateTime": "2026-10-10 08:00", "Source_Page": "4"},
        {"Record_ID": "LAB-012", "Original_Test_Name": "SGPT (ALT)", "Standardized_Test_Name": "ALT (SGPT)", "Original_Value": "36", "Numeric_Value": "36", "Unit": "U/L", "Standardized_Unit": "U/L", "Specimen_Type": "Serum", "Collection_DateTime": "2026-10-10 08:00", "Source_Page": "4"},
        {"Record_ID": "LAB-013", "Original_Test_Name": "Neutrophils", "Standardized_Test_Name": "Neutrophils (%)", "Original_Value": "60", "Numeric_Value": "60", "Unit": "%", "Standardized_Unit": "%", "Specimen_Type": "EDTA Blood", "Collection_DateTime": "2026-10-10 08:00", "Source_Page": "1"},
        {"Record_ID": "LAB-014", "Original_Test_Name": "Lymphocytes", "Standardized_Test_Name": "Lymphocytes (%)", "Original_Value": "30", "Numeric_Value": "30", "Unit": "%", "Standardized_Unit": "%", "Specimen_Type": "EDTA Blood", "Collection_DateTime": "2026-10-10 08:00", "Source_Page": "1"},
        # Non-fasting glucose on 2026-10-15 + Insulin/TG on same/mismatched dates -> MUST NOT compute HOMA-IR or TyG
        {"Record_ID": "LAB-015", "Original_Test_Name": "Random Blood Glucose", "Standardized_Test_Name": "Fasting Blood Glucose (FBS)", "Original_Value": "130", "Numeric_Value": "130", "Unit": "mg/dL", "Standardized_Unit": "mg/dL", "Specimen_Type": "Random Plasma", "Collection_DateTime": "2026-10-15 17:00", "Source_Page": "1"},
        {"Record_ID": "LAB-016", "Original_Test_Name": "Fasting Insulin", "Standardized_Test_Name": "Fasting Insulin", "Original_Value": "14.0", "Numeric_Value": "14.0", "Unit": "µU/mL", "Standardized_Unit": "µU/mL", "Specimen_Type": "Serum", "Collection_DateTime": "2026-10-15 17:00", "Source_Page": "1"},
        {"Record_ID": "LAB-017", "Original_Test_Name": "Triglycerides", "Standardized_Test_Name": "Triglycerides", "Original_Value": "140", "Numeric_Value": "140", "Unit": "mg/dL", "Standardized_Unit": "mg/dL", "Specimen_Type": "Serum", "Collection_DateTime": "2026-10-15 17:00", "Source_Page": "1"},
        # Mismatched dates (FBS on 2026-10-18, Insulin on 2026-10-19) -> MUST NOT compute HOMA-IR
        {"Record_ID": "LAB-018", "Original_Test_Name": "Fasting Blood Sugar", "Standardized_Test_Name": "Fasting Blood Glucose (FBS)", "Original_Value": "92", "Numeric_Value": "92", "Unit": "mg/dL", "Standardized_Unit": "mg/dL", "Specimen_Type": "Fluoride Plasma", "Collection_DateTime": "2026-10-18 08:00", "Source_Page": "1"},
        {"Record_ID": "LAB-019", "Original_Test_Name": "Fasting Insulin", "Standardized_Test_Name": "Fasting Insulin", "Original_Value": "12.0", "Numeric_Value": "12.0", "Unit": "µU/mL", "Standardized_Unit": "µU/mL", "Specimen_Type": "Serum", "Collection_DateTime": "2026-10-19 08:00", "Source_Page": "1"},
        # Spreadsheet auto-date corruption in urinalysis ("2-5" turned into "2026-05-02")
        {"Record_ID": "LAB-020", "Original_Test_Name": "Pus Cells", "Standardized_Test_Name": "Urine Pus Cells (WBCs)", "Original_Value": "2026-05-02", "Numeric_Value": "", "Unit": "/hpf", "Standardized_Unit": "/hpf", "Specimen_Type": "Urine", "Collection_DateTime": "2026-10-10 08:00", "Source_Page": "5"},
    ]
    _write_csv(base_dir, "AggregatedMedicalRecords", "Lab_Results", lab_rows)
    return base_dir


@pytest.fixture()
def synthetic_data_dir(tmp_path: Path) -> Path:
    return populate_synthetic_health_dataset(tmp_path / "sample_data")
