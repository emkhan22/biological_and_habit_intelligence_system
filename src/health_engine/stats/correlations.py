"""Event-Window & Confounder-Adjusted Correlations (`correlations.py` -> `Gold_Stats_Correlations`).

Rules (Prompt Section 4.1 & AGENTS.md Rules 1, 3, 4):
- Evaluates prespecified biologically ordered pairs across:
  1. Circadian / Sleep (`Gold_Daily_Unified`)
  2. Event-Window Nutrition -> Sleep / GI (`Gold_Event_Windows` + `Gold_Daily_Unified`)
  3. GLP-1 Validation (`Gold_Daily_Unified` active GLP-1 scope)
- Enforces Daily / Event Series Maturity Gate:
  - `Level 0 (valid_n < 15)`: descriptive summaries only; suppresses correlation coefficients & p-values.
  - `Level 1 (15 <= valid_n < 30)`: Spearman `rho` + 1,000-resample 95% bootstrap CI + explicit `Exploratory` badge.
  - `Level 2 (valid_n >= 30)`: unlocks confounder-adjusted rank-partial correlations (`partial_rho`, 95% CI, `partial_p_value`).
- Explicitly tags same-day `calories/fat <-> GI` comparisons with `reverse_causality_risk = True`.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from health_engine.stats.maturity import (
    StatsAuditLogger,
    bootstrap_bivariate_ci,
    classify_daily_maturity,
    compute_effective_n,
    compute_missingness_pct,
    enforce_maturity_audit,
)


@dataclass(frozen=True)
class CorrelationPairSpec:
    pair_id: str
    domain: str
    x_col: str
    y_col: str
    alignment_spec: str
    covariates: tuple[str, ...] = ()
    glp1_scope_only: bool = False
    reverse_causality_risk: bool = False
    hypothesis_note: str = ""


PRESPECIFIED_PAIRS: tuple[CorrelationPairSpec, ...] = (
    # 1. Circadian & Sleep
    CorrelationPairSpec(
        "CIRC_BEDTIME_ASLEEP",
        "Circadian_Sleep",
        "bedtime_hr_rel_mid",
        "asleep_min",
        "same_night",
        ("is_weekend_num", "active_glp1_num"),
        False,
        False,
        "Later bedtime vs same-night sleep duration.",
    ),
    CorrelationPairSpec(
        "CIRC_BEDTIME_DEEP",
        "Circadian_Sleep",
        "bedtime_hr_rel_mid",
        "deep_min",
        "same_night",
        ("asleep_min", "active_glp1_num"),
        False,
        False,
        "Bedtime timing vs slow-wave sleep minutes.",
    ),
    CorrelationPairSpec(
        "CIRC_BEDTIME_HRV",
        "Circadian_Sleep",
        "bedtime_hr_rel_mid",
        "hrv_rmssd_ms",
        "same_night",
        ("asleep_min", "steps", "active_glp1_num"),
        False,
        False,
        "Bedtime timing vs nocturnal/daily HRV RMSSD.",
    ),
    CorrelationPairSpec(
        "CIRC_SLEEPHR_HRV",
        "Circadian_Sleep",
        "sleep_hr_avg",
        "hrv_rmssd_ms",
        "same_night",
        ("asleep_min", "steps", "active_glp1_num"),
        False,
        False,
        "Nocturnal heart rate vs vagal HRV RMSSD co-regulation.",
    ),
    CorrelationPairSpec(
        "CIRC_ASLEEP_NEXTDAY_STEPS",
        "Circadian_Sleep",
        "asleep_min",
        "steps",
        "wake_morning_t -> daytime_t",
        ("is_weekend_num", "active_glp1_num"),
        False,
        False,
        "Nocturnal sleep duration ending morning t vs daytime t step count.",
    ),
    CorrelationPairSpec(
        "CIRC_ASLEEP_NEXTDAY_WORKOUT",
        "Circadian_Sleep",
        "asleep_min",
        "workout_mins",
        "wake_morning_t -> daytime_t",
        ("is_weekend_num", "active_glp1_num"),
        False,
        False,
        "Nocturnal sleep duration ending morning t vs daytime t workout minutes.",
    ),
    CorrelationPairSpec(
        "CIRC_ASLEEP_NEXTDAY_HUNGER",
        "Circadian_Sleep",
        "asleep_min",
        "hunger",
        "wake_morning_t -> daytime_t",
        ("tirz_pk_zero_pre",),
        True,
        False,
        "Prior-night sleep duration vs subsequent daytime hunger rating.",
    ),
    CorrelationPairSpec(
        "CIRC_ASLEEP_NEXTDAY_FOODNOISE",
        "Circadian_Sleep",
        "asleep_min",
        "food_noise",
        "wake_morning_t -> daytime_t",
        ("tirz_pk_zero_pre",),
        True,
        False,
        "Prior-night sleep duration vs subsequent daytime food-noise rating.",
    ),
    # 2. Event-Window Nutrition -> Sleep / GI
    CorrelationPairSpec(
        "EV_MEAL_TO_BED_SLEEPHR",
        "Event_Window_Nutrition",
        "exp_last_meal_to_bed_min",
        "out_following_night_sleep_hr_avg",
        "day_t_last_meal -> night_t_to_t+1",
        ("tirz_pk_zero_pre", "steps"),
        False,
        False,
        "Fasting buffer before sleep vs following-night average sleep HR.",
    ),
    CorrelationPairSpec(
        "EV_MEAL_TO_BED_SPIKES",
        "Event_Window_Nutrition",
        "exp_last_meal_to_bed_min",
        "out_following_night_spikes_per_asleep_hr",
        "day_t_last_meal -> night_t_to_t+1",
        ("tirz_pk_zero_pre", "steps"),
        False,
        False,
        "Fasting buffer before sleep vs following-night sleep HR spike rate.",
    ),
    CorrelationPairSpec(
        "EV_LATE_KCAL21_SLEEPHR",
        "Event_Window_Nutrition",
        "exp_late_kcal_after_21_t",
        "out_following_night_sleep_hr_avg",
        "day_t_after_21h -> night_t_to_t+1",
        ("tirz_pk_zero_pre", "steps"),
        False,
        False,
        "Calories consumed >=21:00 IST vs following-night average sleep HR.",
    ),
    CorrelationPairSpec(
        "EV_LATE_KCAL21_SPIKES",
        "Event_Window_Nutrition",
        "exp_late_kcal_after_21_t",
        "out_following_night_spikes_per_asleep_hr",
        "day_t_after_21h -> night_t_to_t+1",
        ("tirz_pk_zero_pre", "steps"),
        False,
        False,
        "Calories consumed >=21:00 IST vs following-night sleep HR spike rate.",
    ),
    CorrelationPairSpec(
        "EV_KCAL_HRV",
        "Event_Window_Nutrition",
        "calories_kcal",
        "hrv_rmssd_ms",
        "day_t -> night/daily_t",
        ("tirz_pk_zero_pre", "steps"),
        False,
        False,
        "Daily caloric load vs HRV RMSSD adjusting for Tirzepatide PK and steps.",
    ),
    CorrelationPairSpec(
        "EV_TM1_EVE_FAT18_GI",
        "Event_Window_Nutrition",
        "exp_t_minus_1_evening_fat_after_18",
        "out_day_t_gi_symptom_score",
        "evening_t-1 -> overnight_morning_t",
        ("tirz_pk_zero_pre",),
        False,
        False,
        "Prior-evening (>=18:00) dietary fat vs overnight/next-morning GI score.",
    ),
    CorrelationPairSpec(
        "EV_TM1_EVE_KCAL18_GI",
        "Event_Window_Nutrition",
        "exp_t_minus_1_evening_kcal_after_18",
        "out_day_t_gi_symptom_score",
        "evening_t-1 -> overnight_morning_t",
        ("tirz_pk_zero_pre",),
        False,
        False,
        "Prior-evening (>=18:00) kcal vs overnight/next-morning GI score.",
    ),
    CorrelationPairSpec(
        "EV_TM1_HYDRATION_CONSTIP",
        "Event_Window_Nutrition",
        "exp_t_minus_1_hydration_liters",
        "out_day_t_constipation_num",
        "day_t-1 -> day_t",
        ("tirz_pk_zero_pre",),
        False,
        False,
        "Prior-day logged hydration liters vs next-day constipation indicator.",
    ),
    CorrelationPairSpec(
        "EV_TM1_HYDRATION_HEADACHE",
        "Event_Window_Nutrition",
        "exp_t_minus_1_hydration_liters",
        "out_day_t_headache_num",
        "day_t-1 -> day_t",
        ("asleep_min",),
        False,
        False,
        "Prior-day logged hydration liters vs next-day headache indicator.",
    ),
    CorrelationPairSpec(
        "EV_FIBER_24H_GI",
        "Event_Window_Nutrition",
        "exp_rolling_24h_fiber_g",
        "out_day_t_gi_symptom_score",
        "trailing_24h_t-1 -> day_t",
        ("tirz_pk_zero_pre",),
        False,
        False,
        "Trailing 24h fiber intake vs subsequent GI score.",
    ),
    CorrelationPairSpec(
        "EV_FIBER_48H_CONSTIP",
        "Event_Window_Nutrition",
        "exp_rolling_48h_fiber_avg_g",
        "out_day_t_constipation_score",
        "trailing_48h_t-2..t-1 -> day_t",
        ("tirz_pk_zero_pre",),
        False,
        False,
        "Trailing 48h mean fiber intake vs subsequent constipation score.",
    ),
    # Reverse-causality guardrail diagnostic pairs (same-day calories/fat <-> GI)
    CorrelationPairSpec(
        "DIAG_SAMEDAY_FAT_GI",
        "Reverse_Causality_Diagnostic",
        "fat_g",
        "out_day_t_gi_symptom_score",
        "same_day_concurrent",
        ("tirz_pk_zero_pre",),
        False,
        True,
        "REVERSE-CAUSALITY RISK: nausea/GI flare on day t can suppress same-day fat intake.",
    ),
    CorrelationPairSpec(
        "DIAG_SAMEDAY_KCAL_GI",
        "Reverse_Causality_Diagnostic",
        "calories_kcal",
        "out_day_t_gi_symptom_score",
        "same_day_concurrent",
        ("tirz_pk_zero_pre",),
        False,
        True,
        "REVERSE-CAUSALITY RISK: nausea/GI flare on day t can suppress same-day caloric intake.",
    ),
    # 3. GLP-1 Pharmacokinetic & Appetite Validation
    CorrelationPairSpec(
        "GLP1_PK_HUNGER",
        "GLP1_Validation",
        "tirz_pk_est",
        "hunger",
        "same_day_12h_pk",
        ("steps",),
        True,
        False,
        "Estimated Tirzepatide plasma-equivalent load vs daily hunger score.",
    ),
    CorrelationPairSpec(
        "GLP1_PK_FOODNOISE",
        "GLP1_Validation",
        "tirz_pk_est",
        "food_noise",
        "same_day_12h_pk",
        ("steps",),
        True,
        False,
        "Estimated Tirzepatide load vs daily food-noise score.",
    ),
    CorrelationPairSpec(
        "GLP1_PK_KCAL",
        "GLP1_Validation",
        "tirz_pk_est",
        "calories_kcal",
        "same_day_12h_pk",
        ("steps",),
        True,
        False,
        "Estimated Tirzepatide load vs daily caloric intake.",
    ),
    CorrelationPairSpec(
        "GLP1_PK_GI",
        "GLP1_Validation",
        "tirz_pk_est",
        "out_day_t_gi_symptom_score",
        "same_day_12h_pk",
        ("fat_g",),
        True,
        False,
        "Estimated Tirzepatide load vs daily GI symptom severity.",
    ),
    CorrelationPairSpec(
        "GLP1_PK_RHR",
        "GLP1_Validation",
        "tirz_pk_est",
        "resting_hr",
        "same_day_12h_pk",
        ("steps", "asleep_min"),
        True,
        False,
        "Estimated Tirzepatide load vs resting heart rate.",
    ),
    CorrelationPairSpec(
        "GLP1_PK_HRV",
        "GLP1_Validation",
        "tirz_pk_est",
        "hrv_rmssd_ms",
        "same_day_12h_pk",
        ("steps", "asleep_min"),
        True,
        False,
        "Estimated Tirzepatide load vs HRV RMSSD.",
    ),
    CorrelationPairSpec(
        "GLP1_DSS_HUNGER",
        "GLP1_Validation",
        "days_since_shot",
        "hunger",
        "days_post_injection",
        ("active_tirz_dose_mg",),
        True,
        False,
        "Days elapsed since injection vs hunger rating.",
    ),
    CorrelationPairSpec(
        "GLP1_DSS_FOODNOISE",
        "GLP1_Validation",
        "days_since_shot",
        "food_noise",
        "days_post_injection",
        ("active_tirz_dose_mg",),
        True,
        False,
        "Days elapsed since injection vs food-noise rating.",
    ),
    CorrelationPairSpec(
        "GLP1_DSS_KCAL",
        "GLP1_Validation",
        "days_since_shot",
        "calories_kcal",
        "days_post_injection",
        ("active_tirz_dose_mg",),
        True,
        False,
        "Days elapsed since injection vs daily caloric intake.",
    ),
    CorrelationPairSpec(
        "GLP1_DSS_GI",
        "GLP1_Validation",
        "days_since_shot",
        "out_day_t_gi_symptom_score",
        "days_post_injection",
        ("active_tirz_dose_mg",),
        True,
        False,
        "Days elapsed since injection vs GI symptom score.",
    ),
    CorrelationPairSpec(
        "GLP1_DSS_RHR",
        "GLP1_Validation",
        "days_since_shot",
        "resting_hr",
        "days_post_injection",
        ("active_tirz_dose_mg", "steps"),
        True,
        False,
        "Days elapsed since injection vs resting heart rate.",
    ),
    CorrelationPairSpec(
        "GLP1_DSS_HRV",
        "GLP1_Validation",
        "days_since_shot",
        "hrv_rmssd_ms",
        "days_post_injection",
        ("active_tirz_dose_mg", "steps"),
        True,
        False,
        "Days elapsed since injection vs HRV RMSSD.",
    ),
    CorrelationPairSpec(
        "GLP1_HUNGER_KCAL",
        "GLP1_Validation",
        "hunger",
        "calories_kcal",
        "same_day",
        ("tirz_pk_est",),
        True,
        False,
        "Subjective hunger rating vs logged daily caloric intake.",
    ),
    CorrelationPairSpec(
        "GLP1_FOODNOISE_KCAL",
        "GLP1_Validation",
        "food_noise",
        "calories_kcal",
        "same_day",
        ("tirz_pk_est",),
        True,
        False,
        "Subjective food-noise rating vs logged daily caloric intake.",
    ),
)


def _partial_spearman_with_ci(
    df_sub: pd.DataFrame,
    x_col: str,
    y_col: str,
    cov_cols: list[str],
    n_boot: int = 1000,
    seed: int = 42,
    min_n: int = 30,
) -> tuple[float, float, float, float, list[str]]:
    """Residualize rank(X) and rank(Y) on available non-constant covariates (when complete n >= min_n)."""
    avail_covs: list[str] = []
    for cc in cov_cols:
        if cc in df_sub.columns:
            s = pd.to_numeric(df_sub[cc], errors="coerce")
            if int((df_sub[x_col].notna() & df_sub[y_col].notna() & s.notna()).sum()) >= min_n:
                avail_covs.append(cc)

    work = df_sub[[x_col, y_col, *avail_covs]].apply(pd.to_numeric, errors="coerce").dropna()
    n = len(work)
    if n < min_n:
        return np.nan, np.nan, np.nan, np.nan, []

    xv = work[x_col].to_numpy(dtype=float)
    yv = work[y_col].to_numpy(dtype=float)
    if np.std(xv) <= 1e-12 or np.std(yv) <= 1e-12:
        return np.nan, np.nan, np.nan, np.nan, []

    active_covs = [c for c in avail_covs if np.std(work[c].to_numpy(dtype=float)) > 1e-10]
    if not active_covs:
        sp = sp_stats.spearmanr(xv, yv)
        rho = round(float(sp.statistic), 4)
        ci_lo, ci_hi, _, _ = bootstrap_bivariate_ci(
            xv,
            yv,
            stat_fn=lambda a, b: float(sp_stats.spearmanr(a, b).statistic),
            point_est=rho,
            n_resamples=n_boot,
            seed=seed,
            min_required_n=min_n,
        )
        return rho, ci_lo, ci_hi, round(float(sp.pvalue), 6), []

    c_mat = work[active_covs].to_numpy(dtype=float)

    def _calc_part(xa: np.ndarray, ya: np.ndarray, ca: np.ndarray) -> float:
        rx = sp_stats.rankdata(xa).astype(float)
        ry = sp_stats.rankdata(ya).astype(float)
        keep = [j for j in range(ca.shape[1]) if np.std(ca[:, j]) > 1e-10]
        if not keep:
            return float(sp_stats.spearmanr(xa, ya).statistic)
        des = np.column_stack([np.ones(len(xa), dtype=float), ca[:, keep]])
        bx, _, _, _ = np.linalg.lstsq(des, rx, rcond=None)
        by, _, _, _ = np.linalg.lstsq(des, ry, rcond=None)
        ex = rx - des @ bx
        ey = ry - des @ by
        if np.std(ex) <= 1e-12 or np.std(ey) <= 1e-12:
            return np.nan
        return float(np.corrcoef(ex, ey)[0, 1])

    prho = _calc_part(xv, yv, c_mat)
    if not np.isfinite(prho):
        return np.nan, np.nan, np.nan, np.nan, active_covs

    df_deg = max(1, n - 2 - len(active_covs))
    if abs(prho) >= 1.0 - 1e-12:
        pp = 0.0
    else:
        t_stat = prho * math.sqrt(df_deg / max(1e-12, 1.0 - prho * prho))
        pp = float(2.0 * sp_stats.t.sf(abs(t_stat), df=df_deg))

    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=float)
    boots.fill(np.nan)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if np.std(xv[idx]) <= 1e-12 or np.std(yv[idx]) <= 1e-12:
            continue
        boots[b] = _calc_part(xv[idx], yv[idx], c_mat[idx])
    fin = boots[np.isfinite(boots)]
    if len(fin) >= max(50, n_boot // 4):
        p_lo, p_hi = np.percentile(fin, [2.5, 97.5])
        ci_l, ci_h = round(float(p_lo), 4), round(float(p_hi), 4)
    else:
        ci_l, ci_h = np.nan, np.nan

    return round(prho, 4), ci_l, ci_h, round(pp, 6), active_covs


def build_gold_stats_correlations(
    gold_daily: pd.DataFrame,
    gold_event_windows: pd.DataFrame,
    thresholds_cfg: dict[str, Any],
    audit_logger: StatsAuditLogger | None = None,
) -> pd.DataFrame:
    """Compute Gold_Stats_Correlations enforcing Daily Maturity Gate & Reverse-Causality flags."""
    cols = [
        "pair_id",
        "domain",
        "x_variable",
        "y_variable",
        "alignment_spec",
        "reverse_causality_risk",
        "maturity_badge",
        "x_mean",
        "x_median",
        "y_mean",
        "y_median",
        "spearman_rho",
        "ci_95_low",
        "ci_95_high",
        "p_value",
        "covariates_adjusted",
        "partial_rho",
        "partial_ci_95_low",
        "partial_ci_95_high",
        "partial_p_value",
        "raw_n",
        "valid_n",
        "effective_n",
        "missingness_pct",
        "predictor_count",
        "stability_flag",
        "maturity_level",
        "epistemic_label",
        "caveat_note",
    ]
    if gold_daily.empty:
        return enforce_maturity_audit(
            pd.DataFrame(columns=cols),
            module_name="correlations",
            analysis_id="EVENT_AND_DAILY_CORRELATIONS",
            audit_logger=audit_logger,
        )

    dgates = thresholds_cfg.get("v20_maturity_gates", {}).get("daily_series", {})
    lvl1_min = int(dgates.get("level1_exploratory_min_n", 15))
    lvl2_min = int(dgates.get("level2_modeling_min_n", 30))
    bcfg = thresholds_cfg.get("v20_maturity_gates", {}).get("bootstrap", {})
    n_boot = int(bcfg.get("n_resamples", 1000))
    seed = int(bcfg.get("random_seed", 42))
    glp1_start = str(thresholds_cfg["phase_anchors"]["active_glp1_phase"]["start_date"])

    # Merge Gold_Daily_Unified and Gold_Event_Windows on date without colliding duplicate columns
    df = gold_daily.sort_values("date").reset_index(drop=True).copy()
    if not gold_event_windows.empty:
        ev_extra = [
            c
            for c in gold_event_windows.columns
            if c == "date" or c not in df.columns
        ]
        df = df.merge(gold_event_windows[ev_extra], on="date", how="left")

    # Numeric indicator helpers for partial correlation adjustment
    df["is_weekend_num"] = df.get("work_mode", pd.Series("", index=df.index)).astype(str).eq("Weekend").astype(float)
    df["active_glp1_num"] = (df["date"].astype(str) >= glp1_start).astype(float)
    pk_num = pd.to_numeric(df.get("tirz_pk_est", pd.Series(np.nan, index=df.index)), errors="coerce")
    # Before GLP-1 initiation, true plasma Tirzepatide concentration is True_Zero (0.0 mg)
    df["tirz_pk_zero_pre"] = np.where(df["date"].astype(str) < glp1_start, 0.0, pk_num)

    for bcol, ncol in (
        ("out_day_t_constipation_flag", "out_day_t_constipation_num"),
        ("out_day_t_headache_flag", "out_day_t_headache_num"),
    ):
        if bcol in df.columns:
            df[ncol] = df[bcol].map(lambda v: 1.0 if v is True else (0.0 if v is False else np.nan))
        else:
            df[ncol] = np.nan

    out_rows: list[dict[str, Any]] = []

    for spec in PRESPECIFIED_PAIRS:
        scope_df = (
            df.loc[df["date"].astype(str) >= glp1_start].copy()
            if spec.glp1_scope_only
            else df.copy()
        )
        raw_n = len(scope_df)
        x_s = (
            pd.to_numeric(scope_df[spec.x_col], errors="coerce")
            if spec.x_col in scope_df.columns
            else pd.Series(np.nan, index=scope_df.index, dtype=float)
        )
        y_s = (
            pd.to_numeric(scope_df[spec.y_col], errors="coerce")
            if spec.y_col in scope_df.columns
            else pd.Series(np.nan, index=scope_df.index, dtype=float)
        )

        valid_mask = x_s.notna() & y_s.notna()
        valid_n = int(valid_mask.sum())
        lvl_num, mat_lbl, badge = classify_daily_maturity(valid_n, lvl1_min, lvl2_min)

        xv = x_s.loc[valid_mask].to_numpy(dtype=float)
        yv = y_s.loc[valid_mask].to_numpy(dtype=float)
        v_dates = scope_df.loc[valid_mask, "date"].tolist()

        x_mean = round(float(np.mean(xv)), 4) if valid_n > 0 else np.nan
        x_med = round(float(np.median(xv)), 4) if valid_n > 0 else np.nan
        y_mean = round(float(np.mean(yv)), 4) if valid_n > 0 else np.nan
        y_med = round(float(np.median(yv)), 4) if valid_n > 0 else np.nan

        eff_n = compute_effective_n(y_s.loc[valid_mask], dates=v_dates) if valid_n > 0 else 0

        rho = np.nan
        ci_lo = np.nan
        ci_hi = np.nan
        pval = np.nan
        cov_used_str = ""
        prho = np.nan
        pci_lo = np.nan
        pci_hi = np.nan
        ppval = np.nan
        stab = "Insufficient_N"
        ep_lbl = "Descriptive"
        pred_cnt = 1

        if lvl_num >= 1 and valid_n >= lvl1_min:
            ep_lbl = "Correlation"
            if np.std(xv) > 1e-12 and np.std(yv) > 1e-12:
                sp = sp_stats.spearmanr(xv, yv)
                rho = round(float(sp.statistic), 4)
                pval = round(float(sp.pvalue), 6)
                ci_lo, ci_hi, _, stab = bootstrap_bivariate_ci(
                    xv,
                    yv,
                    stat_fn=lambda a, b: float(sp_stats.spearmanr(a, b).statistic),
                    point_est=rho,
                    n_resamples=n_boot,
                    seed=seed,
                    min_required_n=lvl1_min,
                )
            else:
                stab = "Unstable"

        if lvl_num >= 2 and valid_n >= lvl2_min and spec.covariates:
            scope_df_num = scope_df.copy()
            scope_df_num[spec.x_col] = x_s
            scope_df_num[spec.y_col] = y_s
            prho, pci_lo, pci_hi, ppval, used_covs = _partial_spearman_with_ci(
                scope_df_num,
                spec.x_col,
                spec.y_col,
                list(spec.covariates),
                n_boot=n_boot,
                seed=seed,
                min_n=lvl2_min,
            )
            cov_used_str = ", ".join(used_covs)
            pred_cnt = 1 + len(used_covs)

        caveat_parts = [spec.hypothesis_note]
        if spec.reverse_causality_risk:
            caveat_parts.append("Flagged reverse_causality_risk=True (concurrent day exposure/symptom).")
        if lvl_num == 0:
            caveat_parts.append(f"Gated at Level 0 (valid_n={valid_n} < {lvl1_min}); correlations suppressed.")
        elif lvl_num == 1:
            caveat_parts.append(f"Exploratory signal (valid_n={valid_n}, effective_n={eff_n}); partial correlations unlock at n>={lvl2_min}.")
        else:
            caveat_parts.append(f"Associational correlation only (effective_n={eff_n}), never causal proof.")

        out_rows.append(
            {
                "pair_id": spec.pair_id,
                "domain": spec.domain,
                "x_variable": spec.x_col,
                "y_variable": spec.y_col,
                "alignment_spec": spec.alignment_spec,
                "reverse_causality_risk": spec.reverse_causality_risk,
                "maturity_badge": badge,
                "x_mean": x_mean,
                "x_median": x_med,
                "y_mean": y_mean,
                "y_median": y_med,
                "spearman_rho": rho,
                "ci_95_low": ci_lo,
                "ci_95_high": ci_hi,
                "p_value": pval,
                "covariates_adjusted": cov_used_str,
                "partial_rho": prho,
                "partial_ci_95_low": pci_lo,
                "partial_ci_95_high": pci_hi,
                "partial_p_value": ppval,
                "raw_n": raw_n,
                "valid_n": valid_n,
                "effective_n": eff_n,
                "missingness_pct": compute_missingness_pct(raw_n, valid_n),
                "predictor_count": pred_cnt,
                "stability_flag": stab,
                "maturity_level": mat_lbl,
                "epistemic_label": ep_lbl,
                "caveat_note": " ".join(p for p in caveat_parts if p),
            }
        )

    out_df = pd.DataFrame(out_rows, columns=cols)
    return enforce_maturity_audit(
        out_df,
        module_name="correlations",
        analysis_id="EVENT_AND_DAILY_CORRELATIONS",
        audit_logger=audit_logger,
        suppressed_at_level0=(
            "spearman_rho",
            "ci_95_low",
            "ci_95_high",
            "p_value",
            "partial_rho",
            "partial_ci_95_low",
            "partial_ci_95_high",
            "partial_p_value",
        ),
        default_epistemic_label="Correlation",
        summary_caveat="Prespecified time-ordered and confounder-adjusted correlations; never causal proof.",
    )
