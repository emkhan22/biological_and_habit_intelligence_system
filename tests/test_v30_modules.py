"""Unit tests for V3.0 Behavioral-Metabolic Statistical & AI Modules (`test_v30_modules.py`).

Verifies:
1. `habit_durability.py` -> `Gold_Habit_Durability`:
   - `Pre_GLP1` vs `On_GLP1` Peak (Days 0-2) vs Trough (Days 4-6) carryover.
   - Automatic `Off_GLP1_Transition` + `Post_GLP1_Week_1..N` trigger on `days_since_shot > 10`.
   - Sustained dose step-down triggers `Off_GLP1_Transition`, whereas a single one-off lower dose does NOT.
2. `sweet_hunger.py` -> `Gold_Sweet_Hunger_Analysis`:
   - Deterministic sweet detection + strict logged-day coverage (unlogged days never zero-filled).
   - `reverse_causality_risk = True` when prior hunger/sleep debt precedes sweet intake.
   - Level 0 gating suppresses Spearman rho/p-value when `< 15` valid windows.
3. `calorie_balance_audit.py` -> `Gold_Calorie_Balance_Audit`:
   - Fitdays Katch-McArdle BMR + TDEE deficit math.
   - Exact classifications: `Within_Expected_Band`, `Potential_Under_Logging`, `Potential_Over_Logging`,
     and `Unclear_Water_Cycle_Or_Log_Gap`.
4. `enricher_sweets.py` (`1D_sweet_meal_classifier`) incremental caching by `normalized_meal_name`
   and smoking-free `AI_Briefing_Packet` with V3.0 sections.
"""

from __future__ import annotations

from datetime import date, timedelta
import json
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

from health_engine.ai.enricher_sweets import run_module_1d_sweet_classifier
from health_engine.ai.packet_builder import assert_packet_smoking_free, build_ai_briefing_packet
from health_engine.ai.router import ModelRouter
from health_engine.stats.calorie_balance_audit import build_gold_calorie_balance_audit
from health_engine.stats.habit_durability import (
    build_gold_habit_durability,
    classify_glp1_regimes,
)
from health_engine.stats.sweet_hunger import build_gold_sweet_hunger_analysis


def _make_daily_frame(n_days: int = 45, start: str = "2026-10-10") -> pd.DataFrame:
    d0 = date.fromisoformat(start)
    rows: list[dict[str, Any]] = []
    for i in range(n_days):
        ds = (d0 + timedelta(days=i)).isoformat()
        on_glp = ds >= "2026-10-21"
        dss = (i - 11) % 7 if on_glp else np.nan
        rows.append(
            {
                "date": ds,
                "steps": 8500.0 if i % 2 == 0 else 6800.0,
                "active_hours": 9.0,
                "strength_sessions_cnt": 1.0 if i % 3 == 0 else 0.0,
                "asleep_min": 420.0,
                "bedtime_hr_rel_mid": 0.5,
                "calories_kcal": 1450.0,
                "activity_total_kcal": 2250.0,
                "protein_g": 95.0,
                "fiber_g": 24.0,
                "sugar_g": 20.0,
                "eating_window_hr": 9.5,
                "late_kcal_after_21": 0.0,
                "nutrition_status": "Present",
                "hunger": 1.5 if on_glp and dss <= 2 else 3.0,
                "food_noise": 1.0 if on_glp and dss <= 2 else 2.5,
                "days_since_shot": float(dss) if on_glp else np.nan,
                "shot_day_int": float(dss) if on_glp else np.nan,
                "active_tirz_dose_mg": 5.0 if on_glp else np.nan,
                "weight_kg": round(106.0 - 0.1 * i, 2) if i % 7 == 0 else np.nan,
                "latest_weight_kg": round(106.0 - 0.1 * i, 2),
            }
        )
    return pd.DataFrame(rows)


def test_habit_durability_peak_trough_off_glp1_and_sustained_stepdown_guard() -> None:
    df = _make_daily_frame(n_days=50, start="2026-10-10")

    # 1. Inject missed/delayed shot (>10 days) for last 14 days -> must trigger Off_GLP1_Transition & Post_GLP1_Week_1..2
    df_gap = df.copy()
    for offset, idx in enumerate(range(36, 50)):
        df_gap.loc[idx, "days_since_shot"] = 11.0 + offset
        df_gap.loc[idx, "protein_g"] = 78.0
        df_gap.loc[idx, "hunger"] = 4.0

    hab = build_gold_habit_durability(df_gap)
    assert "On_GLP1_Peak_vs_Trough" in set(hab["row_type"])
    post_wks = hab.loc[hab["row_type"] == "Post_GLP1_Week"]
    assert list(post_wks["period_id"]) == ["Post_GLP1_Week_1", "Post_GLP1_Week_2"]
    assert (post_wks["glp1_regime"] == "Off_GLP1_Transition").all()
    assert float(post_wks.iloc[0]["delta_protein_vs_on_glp1"]) < 0.0
    assert float(post_wks.iloc[0]["delta_hunger_vs_on_glp1"]) > 0.0

    # 2. One-off lower dose (5 -> 5 -> 2.5 -> 5) must NOT trigger sustained taper!
    shots_oneoff = pd.DataFrame(
        [
            {"date": "2026-10-21", "dose_mg": 5.0},
            {"date": "2026-10-28", "dose_mg": 5.0},
            {"date": "2026-11-04", "dose_mg": 2.5},  # one-off adjustment
            {"date": "2026-11-11", "dose_mg": 5.0},  # returns to 5.0
        ]
    )
    reg_oneoff = classify_glp1_regimes(df, silver_shots=shots_oneoff)
    assert (reg_oneoff.loc[reg_oneoff["date"] >= "2026-10-21", "glp1_regime"] == "On_GLP1").all()

    # 3. Sustained stepdown (5 -> 5 -> 2.5 -> 2.5) MUST trigger Off_GLP1_Transition from 2026-11-04 onward
    shots_sustained = pd.DataFrame(
        [
            {"date": "2026-10-21", "dose_mg": 5.0},
            {"date": "2026-10-28", "dose_mg": 5.0},
            {"date": "2026-11-04", "dose_mg": 2.5},
            {"date": "2026-11-11", "dose_mg": 2.5},
        ]
    )
    reg_sust = classify_glp1_regimes(df, silver_shots=shots_sustained)
    assert (reg_sust.loc[reg_sust["date"] >= "2026-11-04", "glp1_regime"] == "Off_GLP1_Transition").all()


def test_sweet_hunger_reverse_causality_and_logged_coverage() -> None:
    df = _make_daily_frame(n_days=24, start="2026-11-01")
    # Mark 4 days as unlogged -> must NOT be counted as non-sweet days
    df.loc[df.index[:4], "nutrition_status"] = "Not_Logged"
    df.loc[df.index[:4], "calories_kcal"] = np.nan
    # Make day before sweet have high hunger + sleep debt to verify reverse-causality flagging
    df.loc[df.index[6], "hunger"] = 4.0
    df.loc[df.index[7], "asleep_min"] = 310.0

    nutr_rows: list[dict[str, Any]] = []
    for i in range(4, 24):
        ds = str(df.loc[i, "date"])
        if i in {7, 11, 15, 19}:
            nutr_rows.append(
                {"date": ds, "time_ist": "15:30", "meal_type": "Snack", "meal_name": "Gulab Jamun", "calories_kcal": 280, "sugar_g": 32, "carbs_g": 45}
            )
            nutr_rows.append(
                {"date": ds, "time_ist": "20:00", "meal_type": "Dinner", "meal_name": "Dal + 2 Phulkas", "calories_kcal": 520, "sugar_g": 4, "carbs_g": 50}
            )
        else:
            nutr_rows.append(
                {"date": ds, "time_ist": "13:00", "meal_type": "Lunch", "meal_name": "Grilled Fish + Sabzi", "calories_kcal": 480, "sugar_g": 5, "carbs_g": 35}
            )
            nutr_rows.append(
                {"date": ds, "time_ist": "19:30", "meal_type": "Dinner", "meal_name": "Paneer Bhurji", "calories_kcal": 410, "sugar_g": 4, "carbs_g": 20}
            )

    sw_df = build_gold_sweet_hunger_analysis(df, pd.DataFrame(nutr_rows))
    assert not sw_df.empty
    next_day_hun = sw_df.loc[sw_df["outcome_metric"] == "next_day_hunger"].iloc[0]
    assert int(next_day_hun["nutrition_logged_days"]) == 20
    assert int(next_day_hun["sweet_days_n"]) == 4
    assert bool(next_day_hun["reverse_causality_risk"]) is True
    assert float(next_day_hun["pct_sweet_preceded_by_hunger_or_sleep_debt"]) > 0.0

    # Small N (< 15 logged days) MUST suppress Spearman rho/p-value at Level 0
    small_df = df.iloc[4:14].copy().reset_index(drop=True)
    small_sw = build_gold_sweet_hunger_analysis(small_df, pd.DataFrame(nutr_rows))
    small_row = small_sw.loc[small_sw["outcome_metric"] == "next_day_hunger"].iloc[0]
    assert pd.isna(small_row["spearman_rho"])
    assert "Level 0" in str(small_row["maturity_level"])


def test_calorie_balance_audit_four_classifications() -> None:
    df = _make_daily_frame(n_days=35, start="2026-10-20")
    df["activity_total_kcal"] = 2200.0
    df["calories_kcal"] = 1450.0  # expected deficit = 750 kcal/day -> ~1.36 kg per 14d

    bc = pd.DataFrame([{"date": "2026-10-20", "fat_free_body_weight_kg": 52.0}])
    intervals = pd.DataFrame(
        [
            # 1. Within expected band (~750 kcal/d expected vs ~770 kcal/d actual -> delta -1.4 kg over 14d)
            {
                "interval_id": "W-EXPECTED",
                "start_date": "2026-10-20",
                "end_date": "2026-11-03",
                "delta_kg": -1.4,
                "is_fasted_interval": True,
                "period_at_weighin_flag": False,
                "constipation_48h_before_weighin_flag": False,
            },
            # 2. Potential_Under_Logging (logged 750 kcal/d deficit over 14d, but actual delta = -0.1 kg -> ~695 kcal/d discrepancy)
            {
                "interval_id": "W-UNDERLOG",
                "start_date": "2026-11-03",
                "end_date": "2026-11-17",
                "delta_kg": -0.1,
                "is_fasted_interval": True,
                "period_at_weighin_flag": False,
                "constipation_48h_before_weighin_flag": False,
            },
            # 3. Potential_Over_Logging (logged 750 kcal/d deficit over 14d, actual delta = -2.6 kg -> actual > expected by >250)
            {
                "interval_id": "W-OVERLOG",
                "start_date": "2026-10-25",
                "end_date": "2026-11-08",
                "delta_kg": -2.6,
                "is_fasted_interval": True,
                "period_at_weighin_flag": False,
                "constipation_48h_before_weighin_flag": False,
            },
            # 4. Unclear_Water_Cycle_Or_Log_Gap (non-fasted anchor)
            {
                "interval_id": "W-UNCLEAR",
                "start_date": "2026-11-08",
                "end_date": "2026-11-15",
                "delta_kg": -0.5,
                "is_fasted_interval": False,
                "period_at_weighin_flag": True,
                "constipation_48h_before_weighin_flag": False,
            },
        ]
    )

    aud = build_gold_calorie_balance_audit(df, intervals, silver_body_comp=bc)
    by_id = aud.set_index("window_id")
    assert by_id.loc["W-EXPECTED", "audit_classification"] == "Within_Expected_Band"
    assert by_id.loc["W-EXPECTED", "bmr_method"] == "Fitdays_Katch_McArdle_FFM"
    assert by_id.loc["W-UNDERLOG", "audit_classification"] == "Potential_Under_Logging"
    assert by_id.loc["W-OVERLOG", "audit_classification"] == "Potential_Over_Logging"
    assert by_id.loc["W-UNCLEAR", "audit_classification"] == "Unclear_Water_Cycle_Or_Log_Gap"


def test_minimal_cost_1d_sweet_classifier_caching_and_packet_integration() -> None:
    nl = pd.DataFrame(
        [
            {"date": "2026-11-25", "meal_name": "Masala Chai", "meal_type": "Snack", "sugar_g": 8, "carbs_g": 12},
            {"date": "2026-11-26", "meal_name": "Masala Chai", "meal_type": "Snack", "sugar_g": 9, "carbs_g": 13},
            {"date": "2026-11-26", "meal_name": "Grilled Fish Curry", "meal_type": "Lunch", "sugar_g": 2, "carbs_g": 10},
        ]
    )

    calls: list[dict[str, Any]] = []

    class _FakeModels:
        def generate_content(self, *, model: str, contents: str, config: Any) -> Any:
            calls.append({"model": model, "contents": contents})
            payload = {
                "items": [
                    {
                        "normalized_meal_name": "masala chai",
                        "is_sweet_ai": True,
                        "sweet_category": "Sweetened_Beverage",
                        "rationale": "Indian milk tea with added sugar.",
                    }
                ]
            }
            return SimpleNamespace(
                text=json.dumps(payload),
                usage_metadata=SimpleNamespace(prompt_token_count=150, candidates_token_count=50, thoughts_token_count=0, tool_use_prompt_token_count=0),
                candidates=[],
            )

    router = ModelRouter(client=SimpleNamespace(models=_FakeModels()))
    tags_1 = run_module_1d_sweet_classifier(silver_nutr_log=nl, existing_tags_df=None, router=router)
    assert len(calls) == 1
    assert calls[0]["model"] == "gemini-3.5-flash-lite"
    assert len(tags_1) == 1
    assert bool(tags_1.iloc[0]["is_sweet_ai"]) is True

    # Re-run on same meals -> ZERO new API calls!
    tags_2 = run_module_1d_sweet_classifier(silver_nutr_log=nl, existing_tags_df=tags_1, router=router)
    assert len(calls) == 1
    assert len(tags_2) == 1

    # Verify packet includes all 3 V3.0 blocks and passes smoking-free assertion
    df = _make_daily_frame(n_days=20)
    pkt = build_ai_briefing_packet(
        {
            "Gold_Daily_Unified": df,
            "Gold_Habit_Durability": build_gold_habit_durability(df),
            "Gold_Sweet_Hunger_Analysis": build_gold_sweet_hunger_analysis(df, nl, sweet_tags_ai_df=tags_2),
            "Gold_Calorie_Balance_Audit": build_gold_calorie_balance_audit(df, None),
        }
    )
    assert "habit_durability" in pkt
    assert "sweet_hunger" in pkt
    assert "calorie_balance_audit" in pkt
    assert_packet_smoking_free(pkt)
