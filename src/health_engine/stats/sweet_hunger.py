"""Sweet Intake -> Hunger Window Analysis (`sweet_hunger.py` -> `Gold_Sweet_Hunger_Analysis`).

Rules:
1. Identifies sweet/high-sugar/dessert/sweet-snack items and timing from `Silver_Nutrition_Log`
   using deterministic Indian + general food/sugar-density rules, optionally merged with
   cached `gemini-3.5-flash-lite` tags (`Gold_Sweet_Meal_Tags_AI`) for ambiguous entries.
2. Evaluates strictly on `nutrition_status == 'Present'` (logged) days so unlogged days
   remain `NaN` (`Not_Logged`) and are NEVER mistaken for zero sweets.
3. Aligns sweet-intake days/windows with subsequent hunger, food noise, and next-meal kcal
   vs non-sweet logged days, gated by the Daily Maturity Scale (`Level 0` `< 15` valid windows
   suppresses Spearman rho/p-values; `Level 1/2` requires adequate `nutrition_coverage_pct`).
4. Flags `reverse_causality_risk = True` whenever elevated prior-day hunger (`>= 3`),
   high food noise (`>= 3`), or sleep debt (`asleep_min < 360` or `bedtime_hr_rel_mid > 2.0`)
   preceded the sweet intake log.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from health_engine.config_loader import to_iso_date_str
from health_engine.stats.maturity import (
    StatsAuditLogger,
    classify_daily_maturity,
    compute_effective_n,
    compute_missingness_pct,
    enforce_maturity_audit,
)

SWEET_HUNGER_COLUMNS: list[str] = [
    "window_alignment",
    "outcome_metric",
    "sweet_days_n",
    "non_sweet_logged_days_n",
    "nutrition_logged_days",
    "nutrition_coverage_pct",
    "sweet_window_mean",
    "sweet_window_median",
    "non_sweet_window_mean",
    "non_sweet_window_median",
    "sweet_minus_nonsweet_diff",
    "spearman_rho",
    "p_value",
    "reverse_causality_risk",
    "pct_sweet_preceded_by_hunger_or_sleep_debt",
    "sweet_items_detected_total",
    "ambiguous_items_count",
    "raw_n",
    "valid_n",
    "effective_n",
    "missingness_pct",
    "predictor_count",
    "stability_flag",
    "maturity_level",
    "epistemic_label",
    "caveat_note",
    "epistemic_class",
]

CLEAR_SWEET_PATTERNS: tuple[str, ...] = (
    "mithai",
    "gulab jamun",
    "jalebi",
    "rasmalai",
    "rasgulla",
    "kheer",
    "halwa",
    "barfi",
    "burfi",
    "laddu",
    "ladoo",
    "peda",
    "kaju katli",
    "soan papdi",
    "payasam",
    "rabri",
    "kulfi",
    "ice cream",
    "ice-cream",
    "chocolate",
    "brownie",
    "cake",
    "pastry",
    "cookie",
    "biscuit",
    "donut",
    "doughnut",
    "cupcake",
    "muffin",
    "dessert",
    "candy",
    "toffee",
    "pudding",
    "tiramisu",
    "waffle",
    "pancake",
    "sweet",
    "cola",
    "pepsi",
    "coke",
    "fanta",
    "sprite",
    "cold drink",
    "sharbat",
    "rooh afza",
    "falooda",
)

AMBIGUOUS_SWEET_PATTERNS: tuple[str, ...] = (
    "chai",
    "tea",
    "coffee",
    "latte",
    "mocha",
    "shake",
    "milkshake",
    "smoothie",
    "lassi",
    "yogurt",
    "curd",
    "dahi",
    "granola",
    "muesli",
    "bar",
    "protein bar",
    "oats",
    "cornflakes",
    "peanut butter",
    "jam",
    "honey",
    "dates",
    "fruit",
    "mango",
    "banana",
    "chikki",
    "dry fruit",
)


def normalize_meal_name(name: Any) -> str:
    s = str(name or "").strip().lower()
    return re.sub(r"\s+", " ", s)


def classify_meal_sweetness_deterministic(
    meal_name: Any,
    meal_type: Any = "",
    sugar_g: Any = np.nan,
    carbs_g: Any = np.nan,
) -> tuple[bool, bool, str]:
    """Return `(is_sweet, is_ambiguous, rule_reason)` for a single `Silver_Nutrition_Log` entry."""
    norm = normalize_meal_name(meal_name)
    mtype = str(meal_type or "").strip().lower()
    sug = pd.to_numeric(sugar_g, errors="coerce")
    carb = pd.to_numeric(carbs_g, errors="coerce")

    if any(pat in norm for pat in CLEAR_SWEET_PATTERNS) or "dessert" in mtype:
        return True, False, "Deterministic_Sweet_Keyword"

    if pd.notna(sug) and float(sug) >= 18.0:
        if pd.notna(carb) and float(carb) > 0 and (float(sug) / float(carb)) >= 0.40:
            return True, False, "High_Sugar_Density_Ge18g_40pct"

    if any(pat in norm for pat in AMBIGUOUS_SWEET_PATTERNS):
        if pd.notna(sug) and float(sug) >= 12.0:
            return True, True, "Ambiguous_Item_High_Sugar"
        return False, True, "Ambiguous_Needs_Optional_Lite_Tag"

    return False, False, "Savory_Or_NonSweet"


def _parse_hhmm_min(val: Any) -> float:
    m = re.search(r"(\d{1,2}):(\d{2})", str(val or "").strip())
    if not m:
        return np.nan
    return float(int(m.group(1)) * 60 + int(m.group(2)))


def build_gold_sweet_hunger_analysis(
    gold_daily: pd.DataFrame,
    silver_nutr_log: pd.DataFrame | None,
    sweet_tags_ai_df: pd.DataFrame | None = None,
    thresholds_cfg: dict[str, Any] | None = None,
    audit_logger: StatsAuditLogger | None = None,
) -> pd.DataFrame:
    """Build `Gold_Sweet_Hunger_Analysis` aligning sweet-intake windows with subsequent hunger & food noise."""
    if gold_daily is None or gold_daily.empty or "date" not in gold_daily.columns:
        return pd.DataFrame(columns=SWEET_HUNGER_COLUMNS)

    df = gold_daily.sort_values("date").reset_index(drop=True).copy()
    # Restrict scope to active nutrition-tracking period (or all days with nutrition_status == 'Present')
    nutr_present = (
        df["nutrition_status"].astype(str) == "Present"
        if "nutrition_status" in df.columns
        else pd.to_numeric(df.get("calories_kcal", np.nan), errors="coerce").notna()
    )
    if not nutr_present.any():
        return pd.DataFrame(columns=SWEET_HUNGER_COLUMNS)

    first_nutr_date = str(df.loc[nutr_present, "date"].min())[:10]
    scope = df.loc[df["date"].astype(str) >= first_nutr_date].reset_index(drop=True).copy()
    scope_nutr_present = (
        scope["nutrition_status"].astype(str) == "Present"
        if "nutrition_status" in scope.columns
        else pd.to_numeric(scope.get("calories_kcal", np.nan), errors="coerce").notna()
    )
    total_scope_days = int(len(scope))
    nutr_logged_days = int(scope_nutr_present.sum())
    nutr_cov_pct = round(nutr_logged_days / total_scope_days, 4) if total_scope_days > 0 else 0.0

    # Optional AI sweet-tag overrides cached by normalized meal_name
    ai_tag_map: dict[str, bool] = {}
    if sweet_tags_ai_df is not None and not sweet_tags_ai_df.empty:
        for r in sweet_tags_ai_df.to_dict(orient="records"):
            nm = normalize_meal_name(r.get("normalized_meal_name") or r.get("meal_name"))
            if nm:
                val = r.get("is_sweet_ai")
                ai_tag_map[nm] = str(val).strip().lower() in {"true", "1", "yes"}

    # Classify Silver_Nutrition_Log rows
    sweet_days_set: set[str] = set()
    sweet_sugar_by_day: dict[str, float] = {}
    next_meal_kcal_after_sweet: dict[str, float] = {}
    next_meal_kcal_nonsweet: dict[str, float] = {}
    total_sweet_items = 0
    total_ambig_items = 0

    if silver_nutr_log is not None and not silver_nutr_log.empty:
        nl = silver_nutr_log.copy()
        nl["_ds"] = nl["date"].map(to_iso_date_str) if "date" in nl.columns else None
        nl = nl.loc[nl["_ds"].notna()].copy()
        nl["_min"] = nl["time_ist"].map(_parse_hhmm_min) if "time_ist" in nl.columns else np.nan
        nl["_kcal"] = pd.to_numeric(nl.get("calories_kcal", np.nan), errors="coerce")
        nl["_sug"] = pd.to_numeric(nl.get("sugar_g", np.nan), errors="coerce")

        is_sweet_flags: list[bool] = []
        for r in nl.to_dict(orient="records"):
            norm_m = normalize_meal_name(r.get("meal_name"))
            det_sweet, is_ambig, _ = classify_meal_sweetness_deterministic(
                r.get("meal_name"), r.get("meal_type"), r.get("sugar_g"), r.get("carbs_g")
            )
            if is_ambig:
                total_ambig_items += 1
            final_sweet = ai_tag_map[norm_m] if norm_m in ai_tag_map else det_sweet
            if final_sweet:
                total_sweet_items += 1
            is_sweet_flags.append(final_sweet)
        nl["_is_sweet"] = is_sweet_flags

        for ds, grp in nl.groupby("_ds"):
            grp_s = grp.sort_values("_min")
            sw_rows = grp_s.loc[grp_s["_is_sweet"]]
            if not sw_rows.empty:
                sweet_days_set.add(str(ds))
                sweet_sugar_by_day[str(ds)] = float(sw_rows["_sug"].fillna(0.0).sum())
                first_sw_min = float(sw_rows["_min"].iloc[0]) if pd.notna(sw_rows["_min"].iloc[0]) else np.nan
                if np.isfinite(first_sw_min):
                    later = grp_s.loc[grp_s["_min"] > first_sw_min, "_kcal"].dropna()
                    if not later.empty:
                        next_meal_kcal_after_sweet[str(ds)] = float(later.iloc[0])
            else:
                if len(grp_s) >= 2:
                    later_ns = grp_s["_kcal"].dropna()
                    if len(later_ns) >= 2:
                        next_meal_kcal_nonsweet[str(ds)] = float(later_ns.iloc[-1])

    # Build day-level analytic frame strictly on logged nutrition days
    scope["_is_sweet_day"] = np.where(
        scope_nutr_present,
        scope["date"].astype(str).str.slice(0, 10).isin(sweet_days_set).astype(float),
        np.nan,
    )
    scope["_sweet_sugar_g"] = np.where(
        scope_nutr_present,
        scope["date"].astype(str).str.slice(0, 10).map(lambda d: sweet_sugar_by_day.get(d, 0.0)),
        np.nan,
    )
    scope["_next_day_hunger"] = pd.to_numeric(scope.get("hunger", np.nan), errors="coerce").shift(-1)
    scope["_next_day_food_noise"] = pd.to_numeric(scope.get("food_noise", np.nan), errors="coerce").shift(-1)
    scope["_same_day_hunger"] = pd.to_numeric(scope.get("hunger", np.nan), errors="coerce")
    scope["_same_day_food_noise"] = pd.to_numeric(scope.get("food_noise", np.nan), errors="coerce")

    # Preceding hunger / sleep-debt reverse-causality check
    prev_hunger = pd.to_numeric(scope.get("hunger", np.nan), errors="coerce").shift(1)
    prev_fn = pd.to_numeric(scope.get("food_noise", np.nan), errors="coerce").shift(1)
    asleep_min = pd.to_numeric(scope.get("asleep_min", np.nan), errors="coerce")
    bed_rel = pd.to_numeric(scope.get("bedtime_hr_rel_mid", np.nan), errors="coerce")
    preceded_mask = (
        (prev_hunger >= 3.0)
        | (prev_fn >= 3.0)
        | (asleep_min < 360.0)
        | (bed_rel > 2.0)
    )
    sweet_logged_mask = scope["_is_sweet_day"] == 1.0
    sweet_days_n = int(sweet_logged_mask.sum())
    non_sweet_days_n = int((scope["_is_sweet_day"] == 0.0).sum())
    pct_preceded = (
        round(float(preceded_mask.loc[sweet_logged_mask].mean() * 100.0), 1)
        if sweet_days_n > 0
        else np.nan
    )
    rev_causality_flag = bool(
        (pd.notna(pct_preceded) and pct_preceded >= 25.0) or sweet_days_n > 0
    )

    specs: list[tuple[str, str, pd.Series]] = [
        ("Next_Day_Post_Sweet", "next_day_hunger", scope["_next_day_hunger"]),
        ("Next_Day_Post_Sweet", "next_day_food_noise", scope["_next_day_food_noise"]),
        ("Same_Day_Post_Sweet", "same_day_hunger", scope["_same_day_hunger"]),
        ("Same_Day_Post_Sweet", "same_day_food_noise", scope["_same_day_food_noise"]),
    ]

    rows: list[dict[str, Any]] = []
    for align_lbl, out_name, y_series in specs:
        valid_mask = scope["_is_sweet_day"].notna() & y_series.notna()
        valid_n = int(valid_mask.sum())
        sw_vals = y_series.loc[valid_mask & (scope["_is_sweet_day"] == 1.0)]
        ns_vals = y_series.loc[valid_mask & (scope["_is_sweet_day"] == 0.0)]

        sw_mean = round(float(sw_vals.mean()), 3) if not sw_vals.empty else np.nan
        sw_med = round(float(sw_vals.median()), 3) if not sw_vals.empty else np.nan
        ns_mean = round(float(ns_vals.mean()), 3) if not ns_vals.empty else np.nan
        ns_med = round(float(ns_vals.median()), 3) if not ns_vals.empty else np.nan
        diff_val = (
            round(sw_mean - ns_mean, 3)
            if pd.notna(sw_mean) and pd.notna(ns_mean)
            else np.nan
        )

        _, mat, _ = classify_daily_maturity(valid_n)
        eff_n = compute_effective_n(y_series.loc[valid_mask]) if valid_n >= 5 else valid_n

        rho = np.nan
        pval = np.nan
        epistemic = "Descriptive"
        stab = "Insufficient_N" if valid_n < 15 else "Exploratory_Signal"

        # Level 1/2 gate: require >=15 valid windows, >=3 sweet days, and >=60% nutrition coverage
        if valid_n >= 15 and len(sw_vals) >= 3 and len(ns_vals) >= 3 and nutr_cov_pct >= 0.60:
            x_v = scope.loc[valid_mask, "_sweet_sugar_g"].to_numpy(dtype=float)
            y_v = y_series.loc[valid_mask].to_numpy(dtype=float)
            if np.nanstd(x_v) > 0 and np.nanstd(y_v) > 0:
                r_res = sp_stats.spearmanr(x_v, y_v)
                rho = round(float(r_res.statistic), 4) if np.isfinite(r_res.statistic) else np.nan
                pval = round(float(r_res.pvalue), 4) if np.isfinite(r_res.pvalue) else np.nan
                epistemic = "Correlation"
                stab = "Moderate" if valid_n >= 30 else "Exploratory_Signal"

        caveat = (
            f"Evaluated strictly on {nutr_logged_days}/{total_scope_days} logged nutrition days "
            f"(coverage={nutr_cov_pct*100:.1f}%); unlogged days excluded. "
            f"Reverse-causality risk flagged (prior hunger/sleep debt preceded {pct_preceded if pd.notna(pct_preceded) else 0:.1f}% of sweet days)."
        )
        rows.append(
            {
                "window_alignment": align_lbl,
                "outcome_metric": out_name,
                "sweet_days_n": int(len(sw_vals)),
                "non_sweet_logged_days_n": int(len(ns_vals)),
                "nutrition_logged_days": nutr_logged_days,
                "nutrition_coverage_pct": nutr_cov_pct,
                "sweet_window_mean": sw_mean,
                "sweet_window_median": sw_med,
                "non_sweet_window_mean": ns_mean,
                "non_sweet_window_median": ns_med,
                "sweet_minus_nonsweet_diff": diff_val,
                "spearman_rho": rho,
                "p_value": pval,
                "reverse_causality_risk": rev_causality_flag,
                "pct_sweet_preceded_by_hunger_or_sleep_debt": pct_preceded,
                "sweet_items_detected_total": total_sweet_items,
                "ambiguous_items_count": total_ambig_items,
                "raw_n": total_scope_days,
                "valid_n": valid_n,
                "effective_n": eff_n,
                "missingness_pct": compute_missingness_pct(total_scope_days, valid_n),
                "predictor_count": 1,
                "stability_flag": stab,
                "maturity_level": mat,
                "epistemic_label": epistemic,
                "caveat_note": caveat,
            }
        )

    # Next-meal caloric rebound row (same-day subsequent meal after sweet vs non-sweet)
    sw_nm = list(next_meal_kcal_after_sweet.values())
    ns_nm = list(next_meal_kcal_nonsweet.values())
    nm_valid = len(sw_nm) + len(ns_nm)
    _, nm_mat, _ = classify_daily_maturity(nm_valid)
    rows.append(
        {
            "window_alignment": "Same_Day_Next_Meal",
            "outcome_metric": "subsequent_meal_kcal",
            "sweet_days_n": len(sw_nm),
            "non_sweet_logged_days_n": len(ns_nm),
            "nutrition_logged_days": nutr_logged_days,
            "nutrition_coverage_pct": nutr_cov_pct,
            "sweet_window_mean": round(float(np.mean(sw_nm)), 1) if sw_nm else np.nan,
            "sweet_window_median": round(float(np.median(sw_nm)), 1) if sw_nm else np.nan,
            "non_sweet_window_mean": round(float(np.mean(ns_nm)), 1) if ns_nm else np.nan,
            "non_sweet_window_median": round(float(np.median(ns_nm)), 1) if ns_nm else np.nan,
            "sweet_minus_nonsweet_diff": (
                round(float(np.mean(sw_nm) - np.mean(ns_nm)), 1) if sw_nm and ns_nm else np.nan
            ),
            "spearman_rho": np.nan,
            "p_value": np.nan,
            "reverse_causality_risk": rev_causality_flag,
            "pct_sweet_preceded_by_hunger_or_sleep_debt": pct_preceded,
            "sweet_items_detected_total": total_sweet_items,
            "ambiguous_items_count": total_ambig_items,
            "raw_n": total_scope_days,
            "valid_n": nm_valid,
            "effective_n": nm_valid,
            "missingness_pct": compute_missingness_pct(total_scope_days, nm_valid),
            "predictor_count": 1,
            "stability_flag": "Descriptive_Only" if nm_valid >= 10 else "Insufficient_N",
            "maturity_level": nm_mat,
            "epistemic_label": "Descriptive",
            "caveat_note": "Compares same-day subsequent meal kcal after a sweet entry vs non-sweet logged days; unlogged days excluded.",
        }
    )

    raw_out = pd.DataFrame(rows)
    out_df = enforce_maturity_audit(
        raw_out,
        module_name="sweet_hunger",
        analysis_id="SWEET_INTAKE_HUNGER_WINDOWS",
        audit_logger=audit_logger,
        suppressed_at_level0=("spearman_rho", "p_value"),
        default_epistemic_label="Descriptive",
        summary_cv_or_ci=f"sweet_days={sweet_days_n}, non_sweet_logged={non_sweet_days_n}, coverage={nutr_cov_pct:.2f}",
        summary_caveat="Strict logged-day denominator with reverse-causality flagging for prior hunger/sleep debt.",
    )
    for col in SWEET_HUNGER_COLUMNS:
        if col not in out_df.columns:
            out_df[col] = np.nan
    return out_df[SWEET_HUNGER_COLUMNS]
