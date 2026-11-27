"""Isolated V2.5 Progressive ML for `private_smoking` (`Smoking_Private_Store` ONLY).

Security, Privacy & Zero-Leakage Guarantees (AGENTS.md Rule 2 & Prompt Section 5):
- Zero imports from `health_engine`; writes exclusively to `Smoking_Private_Store`.
- Hard `ONSET_FEATURES` allowlist and `FORBIDDEN_POST_EVENT_COLS` runtime `ValueError` assertion
  preventing `Delay_Minutes`, `Count`, `Barrier_or_Reason`, or `Event_Type` from ever entering `X`.
- Maturity & class-imbalance gated binary classifier (`Smoked=1` vs `Resisted_Craving=0`):
  1. `valid_events < 30` OR `min_class_count < 10`: `Level 0: Descriptive Risk Heatmap` ONLY.
  2. `30 <= valid_events < 100` AND `min_class_count >= 10`: L1 `LogisticRegression(l1_ratio=1.0,
     solver='saga', class_weight='balanced')` with chronological `TimeSeriesSplit` evaluated on
     PR-AUC, ROC-AUC, and Balanced Accuracy.
  3. `valid_events >= 100` AND `min_class_count >= 25`: `RandomForestClassifier(max_depth=4,
     class_weight='balanced')` vs Logistic Regression + `shap.TreeExplainer` slicing `[:, :, 1]`.
- Separate post-event `Private_Delay_Outcome_Stats` (`<5m`, `5-15m`, `>15m`, gated at `n >= 15`).
- Zero raw notes, triggers, timestamps, or counts printed to stdout/CI logs.
"""

from __future__ import annotations

from datetime import datetime
import math
from typing import Any
import warnings
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

IST = ZoneInfo("Asia/Kolkata")

ONSET_FEATURES: tuple[str, ...] = (
    "hour_sin",
    "hour_cos",
    "late_night_00_05_flag",
    "prior_night_asleep_min",
    "prior_night_bedtime_hr",
    "days_since_shot",
    "active_tirz_dose_mg",
    "is_weighin_day",
    "Trigger",
    "Mood",
    "prior_hour_hr_zscore",
)

NUMERIC_ONSET_FEATURES: tuple[str, ...] = (
    "hour_sin",
    "hour_cos",
    "late_night_00_05_flag",
    "prior_night_asleep_min",
    "prior_night_bedtime_hr",
    "days_since_shot",
    "active_tirz_dose_mg",
    "is_weighin_day",
    "prior_hour_hr_zscore",
)

CATEGORICAL_ONSET_FEATURES: tuple[str, ...] = ("Trigger", "Mood")

FORBIDDEN_POST_EVENT_COLS: frozenset[str] = frozenset(
    {
        "Delay_Minutes",
        "delay_minutes_post",
        "Count",
        "count_smoked_post",
        "Barrier_or_Reason",
        "barrier_or_reason_post",
        "Event_Type",
        "event_type",
        "is_resisted_outcome",
    }
)

_FORBIDDEN_LOWER: frozenset[str] = frozenset(c.lower() for c in FORBIDDEN_POST_EVENT_COLS)


def validate_onset_feature_matrix(x_df: pd.DataFrame) -> None:
    """Hard runtime assertion: raise ValueError if any post-event column enters X."""
    leaked: list[str] = []
    for col in x_df.columns:
        c_str = str(col).strip()
        c_low = c_str.lower()
        if (
            c_str in FORBIDDEN_POST_EVENT_COLS
            or c_low in _FORBIDDEN_LOWER
            or any(
                tok in c_low
                for tok in ("delay_minutes", "count_smoked", "barrier_or_reason", "event_type", "is_resisted")
            )
        ):
            leaked.append(c_str)
    if leaked:
        raise ValueError(
            f"Zero-leakage boundary violation in private_smoking ML: forbidden post-event columns in X: {leaked}"
        )


def _sleep_tier(asleep_min: float) -> str:
    if not np.isfinite(asleep_min):
        return "Unmeasured"
    if asleep_min < 360.0:
        return "<6h"
    if asleep_min >= 420.0:
        return ">=7h"
    return "6h_to_7h"


def _delay_bucket(delay_min: float) -> str:
    if not np.isfinite(delay_min):
        return "Unlogged"
    if delay_min < 5.0:
        return "<5m"
    if delay_min <= 15.0:
        return "5-15m"
    return ">15m"


def _build_onset_dataset(
    onset_df: pd.DataFrame,
    post_df: pd.DataFrame,
    daily_context_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Assemble strictly chronologically sorted onset feature frame + binary outcome y."""
    if onset_df.empty or post_df.empty:
        return pd.DataFrame()

    ctx_by_date: dict[str, dict[str, Any]] = {}
    if daily_context_df is not None and not daily_context_df.empty and "date" in daily_context_df.columns:
        for r in daily_context_df.to_dict(orient="records"):
            ds = str(r.get("date") or "")
            if ds:
                ctx_by_date[ds] = r

    joined = onset_df.merge(
        post_df[["event_key", "event_type", "is_resisted_outcome", "count_smoked_post"]],
        on="event_key",
        how="inner",
    )
    if joined.empty:
        return pd.DataFrame()

    joined["_sort_time"] = joined["onset_time_ist"].fillna("").astype(str)
    joined = joined.sort_values(["date", "_sort_time", "event_key"], kind="mergesort").reset_index(drop=True)

    rows: list[dict[str, Any]] = []
    for r in joined.to_dict(orient="records"):
        ds = str(r.get("date") or "")
        hr_val = pd.to_numeric(pd.Series([r.get("onset_hour_ist")]), errors="coerce").iloc[0]
        if np.isfinite(hr_val):
            rad = 2.0 * math.pi * (float(hr_val) / 24.0)
            h_sin = round(math.sin(rad), 6)
            h_cos = round(math.cos(rad), 6)
            late_00_05 = 1.0 if (0.0 <= float(hr_val) < 5.0) else 0.0
            hr_bucket = f"{int(float(hr_val)):02d}:00"
        else:
            h_sin = np.nan
            h_cos = np.nan
            late_00_05 = np.nan
            hr_bucket = "Unlogged_Hour"

        ctx = ctx_by_date.get(ds, {})
        asleep_m = pd.to_numeric(pd.Series([ctx.get("asleep_min")]), errors="coerce").iloc[0]
        bed_hr = pd.to_numeric(pd.Series([ctx.get("bedtime_hr_rel_mid")]), errors="coerce").iloc[0]
        dss = pd.to_numeric(pd.Series([ctx.get("days_since_shot")]), errors="coerce").iloc[0]
        sday = pd.to_numeric(pd.Series([ctx.get("shot_day_int")]), errors="coerce").iloc[0]
        dose = pd.to_numeric(pd.Series([ctx.get("active_tirz_dose_mg")]), errors="coerce").iloc[0]
        w_kg = pd.to_numeric(pd.Series([ctx.get("weight_kg")]), errors="coerce").iloc[0]
        hr_z = pd.to_numeric(pd.Series([ctx.get("daytime_hr_z_classical_pre_oct", np.nan)]), errors="coerce").iloc[0]

        ev_low = str(r.get("event_type") or "").strip().lower()
        is_res = bool(r.get("is_resisted_outcome", False))
        cnt_v = pd.to_numeric(pd.Series([r.get("count_smoked_post")]), errors="coerce").iloc[0]

        if is_res:
            y_bin = 0
        elif "smok" in ev_low or (np.isfinite(cnt_v) and float(cnt_v) > 0.0):
            y_bin = 1
        else:
            continue

        rows.append(
            {
                "event_key": str(r["event_key"]),
                "date": ds,
                "onset_time_ist": str(r.get("onset_time_ist") or ""),
                "hour_bucket": hr_bucket,
                "shot_day_stratum": f"Day_{int(sday)}" if np.isfinite(sday) else "Unlinked",
                "sleep_tier": _sleep_tier(float(asleep_m) if np.isfinite(asleep_m) else np.nan),
                "hour_sin": h_sin,
                "hour_cos": h_cos,
                "late_night_00_05_flag": late_00_05,
                "prior_night_asleep_min": float(asleep_m) if np.isfinite(asleep_m) else np.nan,
                "prior_night_bedtime_hr": float(bed_hr) if np.isfinite(bed_hr) else np.nan,
                "days_since_shot": float(dss) if np.isfinite(dss) else np.nan,
                "active_tirz_dose_mg": float(dose) if np.isfinite(dose) else 0.0,
                "is_weighin_day": 1.0 if np.isfinite(w_kg) else 0.0,
                "Trigger": str(r.get("trigger_onset") or "").strip() or "Unlogged",
                "Mood": str(r.get("mood_onset") or "").strip() or "Unlogged",
                "prior_hour_hr_zscore": float(hr_z) if np.isfinite(hr_z) else np.nan,
                "y_smoked": int(y_bin),
            }
        )

    return pd.DataFrame(rows)


def _make_smoking_clf_pipeline(estimator: Any, num_cols: list[str], cat_cols: list[str]) -> Pipeline:
    transformers: list[tuple[str, Any, list[str]]] = []
    if num_cols:
        num_pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("num", num_pipe, num_cols))
    if cat_cols:
        cat_pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="constant", fill_value="Unlogged")),
                ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
        transformers.append(("cat", cat_pipe, cat_cols))

    pre = ColumnTransformer(transformers=transformers, remainder="drop")
    return Pipeline([("pre", pre), ("clf", estimator)])


def build_private_smoking_ml(
    silver_smoking_tables: dict[str, pd.DataFrame],
    daily_context_df: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """Run isolated V2.5 Private Smoking ML & Post-Event Delay Outcome Analysis."""
    run_ts = datetime.now(tz=IST).strftime("%d-%m-%Y %H:%M:%S IST")
    onset_df = silver_smoking_tables.get("Silver_Smoking_Onset_Context", pd.DataFrame()).copy()
    post_df = silver_smoking_tables.get("Silver_Smoking_Post_Outcomes", pd.DataFrame()).copy()

    summary_cols = [
        "model_id",
        "status",
        "selected_algorithm",
        "valid_events",
        "smoked_class_n",
        "resisted_class_n",
        "min_class_count",
        "dummy_pr_auc",
        "dummy_balanced_acc",
        "logistic_pr_auc",
        "logistic_roc_auc",
        "logistic_balanced_acc",
        "rf_pr_auc",
        "rf_roc_auc",
        "rf_balanced_acc",
        "maturity_level",
        "epistemic_label",
        "epistemic_class",
        "caveat_note",
    ]
    driver_cols = [
        "driver_mode",
        "dimension_or_feature",
        "stratum_value",
        "event_count",
        "smoked_rate",
        "model_weight_or_shap",
        "direction",
        "maturity_level",
        "epistemic_label",
        "epistemic_class",
        "ui_phrasing",
    ]
    delay_cols = [
        "delay_bucket",
        "events_n",
        "resisted_n",
        "smoked_n",
        "resist_rate",
        "mean_delay_minutes",
        "gate_status",
        "maturity_level",
        "epistemic_label",
        "epistemic_class",
        "caveat_note",
    ]
    audit_cols = [
        "model_id",
        "model_version",
        "target_metric",
        "algorithm",
        "feature_list",
        "train_date_range",
        "validation_method",
        "raw_n",
        "valid_n",
        "min_class_count",
        "pr_auc",
        "roc_auc",
        "balanced_accuracy",
        "maturity_level",
        "epistemic_label",
        "epistemic_class",
        "run_timestamp_ist",
    ]

    ds_frame = _build_onset_dataset(onset_df, post_df, daily_context_df=daily_context_df)
    raw_n = len(onset_df)

    # 1. Separate Post-Event Delay Analysis (`Private_Delay_Outcome_Stats`)
    delay_rows: list[dict[str, Any]] = []
    if not post_df.empty and "delay_minutes_post" in post_df.columns:
        p_work = post_df.copy()
        p_work["_dly"] = pd.to_numeric(p_work["delay_minutes_post"], errors="coerce")
        valid_dly = p_work.loc[p_work["_dly"].notna()].copy()
        n_dly = len(valid_dly)
        dly_unlocked = n_dly >= 15
        dly_gate = "Unlocked_Delay_Comparison" if dly_unlocked else "Gated_Below_N15_Delay_Events"
        dly_mat = "Level 1 (Post-Event Outcome Summary)" if dly_unlocked else "Level 0 (Collecting Delay Baseline)"

        if n_dly > 0:
            valid_dly["_bkt"] = [_delay_bucket(float(v)) for v in valid_dly["_dly"]]
            for bkt in ("<5m", "5-15m", ">15m"):
                bsub = valid_dly.loc[valid_dly["_bkt"] == bkt]
                bn = len(bsub)
                if bn == 0:
                    continue
                res_n = int(bsub["is_resisted_outcome"].fillna(False).astype(bool).sum())
                smk_n = bn - res_n
                delay_rows.append(
                    {
                        "delay_bucket": bkt,
                        "events_n": bn,
                        "resisted_n": res_n,
                        "smoked_n": smk_n,
                        "resist_rate": round(res_n / float(bn), 4) if dly_unlocked else np.nan,
                        "mean_delay_minutes": round(float(bsub["_dly"].mean()), 2),
                        "gate_status": dly_gate,
                        "maturity_level": dly_mat,
                        "epistemic_label": "Descriptive",
                        "epistemic_class": "Descriptive",
                        "caveat_note": (
                            "Post-event delay outcome comparison (strictly excluded from pre-decision ML features)."
                            if dly_unlocked
                            else f"Resist rate gated until >= 15 logged delay events accrue (current n={n_dly})."
                        ),
                    }
                )
    delay_out = pd.DataFrame(delay_rows, columns=delay_cols)

    # Handle empty onset dataset
    if ds_frame.empty:
        sum_empty = pd.DataFrame(
            [
                {
                    "model_id": "PRIVATE_SMOKING_ONSET_CLF",
                    "status": "Level 0: Descriptive Risk Heatmap",
                    "selected_algorithm": "None_Gated",
                    "valid_events": 0,
                    "smoked_class_n": 0,
                    "resisted_class_n": 0,
                    "min_class_count": 0,
                    "dummy_pr_auc": np.nan,
                    "dummy_balanced_acc": np.nan,
                    "logistic_pr_auc": np.nan,
                    "logistic_roc_auc": np.nan,
                    "logistic_balanced_acc": np.nan,
                    "rf_pr_auc": np.nan,
                    "rf_roc_auc": np.nan,
                    "rf_balanced_acc": np.nan,
                    "maturity_level": "Level 0: Descriptive Risk Heatmap",
                    "epistemic_label": "Descriptive",
                    "epistemic_class": "Descriptive",
                    "caveat_note": "Gated at Level 0 (<30 events or min_class_count < 10); classifier not trained.",
                }
            ],
            columns=summary_cols,
        )
        aud_empty = pd.DataFrame(
            [
                {
                    "model_id": "PRIVATE_SMOKING_ONSET_CLF",
                    "model_version": "v2.5.0",
                    "target_metric": "y_smoked_vs_resisted",
                    "algorithm": "None_Gated",
                    "feature_list": ", ".join(ONSET_FEATURES),
                    "train_date_range": "",
                    "validation_method": "Gated_Level0",
                    "raw_n": raw_n,
                    "valid_n": 0,
                    "min_class_count": 0,
                    "pr_auc": np.nan,
                    "roc_auc": np.nan,
                    "balanced_accuracy": np.nan,
                    "maturity_level": "Level 0: Descriptive Risk Heatmap",
                    "epistemic_label": "Descriptive",
                    "epistemic_class": "Descriptive",
                    "run_timestamp_ist": run_ts,
                }
            ],
            columns=audit_cols,
        )
        return {
            "Private_Smoking_ML_Summary": sum_empty,
            "Private_Smoking_Risk_Drivers": pd.DataFrame(columns=driver_cols),
            "Private_Delay_Outcome_Stats": delay_out,
            "Private_Model_Audit": aud_empty,
        }

    # Build strict X matrix from ONSET_FEATURES ONLY and validate zero post-event leakage!
    x_df = ds_frame[list(ONSET_FEATURES)].copy()
    validate_onset_feature_matrix(x_df)
    y_arr = ds_frame["y_smoked"].to_numpy(dtype=int)

    valid_events = len(ds_frame)
    n_smoked = int((y_arr == 1).sum())
    n_resisted = int((y_arr == 0).sum())
    min_cls = min(n_smoked, n_resisted)
    d_span = f"{ds_frame['date'].iloc[0]}..{ds_frame['date'].iloc[-1]}"

    # Gate 0: valid_events < 30 OR min_class_count < 10 -> Level 0 Descriptive Risk Heatmap ONLY
    if valid_events < 30 or min_cls < 10:
        heatmap_rows: list[dict[str, Any]] = []
        for dim_lbl, col_k in (
            ("hour_ist", "hour_bucket"),
            ("Trigger", "Trigger"),
            ("shot_day_int", "shot_day_stratum"),
            ("sleep_tier", "sleep_tier"),
        ):
            for sval, grp in ds_frame.groupby(col_k, sort=True):
                en = len(grp)
                srate = round(float(grp["y_smoked"].mean()), 4)
                heatmap_rows.append(
                    {
                        "driver_mode": "Descriptive_Risk_Heatmap",
                        "dimension_or_feature": dim_lbl,
                        "stratum_value": str(sval),
                        "event_count": en,
                        "smoked_rate": srate,
                        "model_weight_or_shap": np.nan,
                        "direction": "Descriptive_Count",
                        "maturity_level": "Level 0: Descriptive Risk Heatmap",
                        "epistemic_label": "Descriptive",
                        "epistemic_class": "Descriptive",
                        "ui_phrasing": "Descriptive onset risk frequency only; classifier locked until >=30 events & >=10 per class.",
                    }
                )
        sum_l0 = pd.DataFrame(
            [
                {
                    "model_id": "PRIVATE_SMOKING_ONSET_CLF",
                    "status": "Level 0: Descriptive Risk Heatmap",
                    "selected_algorithm": "None_Gated",
                    "valid_events": valid_events,
                    "smoked_class_n": n_smoked,
                    "resisted_class_n": n_resisted,
                    "min_class_count": min_cls,
                    "dummy_pr_auc": np.nan,
                    "dummy_balanced_acc": np.nan,
                    "logistic_pr_auc": np.nan,
                    "logistic_roc_auc": np.nan,
                    "logistic_balanced_acc": np.nan,
                    "rf_pr_auc": np.nan,
                    "rf_roc_auc": np.nan,
                    "rf_balanced_acc": np.nan,
                    "maturity_level": "Level 0: Descriptive Risk Heatmap",
                    "epistemic_label": "Descriptive",
                    "epistemic_class": "Descriptive",
                    "caveat_note": (
                        f"Classifier locked (valid_events={valid_events}, min_class_count={min_cls}); "
                        "requires >=30 events and >=10 per class."
                    ),
                }
            ],
            columns=summary_cols,
        )
        aud_l0 = pd.DataFrame(
            [
                {
                    "model_id": "PRIVATE_SMOKING_ONSET_CLF",
                    "model_version": "v2.5.0",
                    "target_metric": "y_smoked_vs_resisted",
                    "algorithm": "None_Gated",
                    "feature_list": ", ".join(ONSET_FEATURES),
                    "train_date_range": d_span,
                    "validation_method": "Gated_Level0_Heatmap_Only",
                    "raw_n": raw_n,
                    "valid_n": valid_events,
                    "min_class_count": min_cls,
                    "pr_auc": np.nan,
                    "roc_auc": np.nan,
                    "balanced_accuracy": np.nan,
                    "maturity_level": "Level 0: Descriptive Risk Heatmap",
                    "epistemic_label": "Descriptive",
                    "epistemic_class": "Descriptive",
                    "run_timestamp_ist": run_ts,
                }
            ],
            columns=audit_cols,
        )
        return {
            "Private_Smoking_ML_Summary": sum_l0,
            "Private_Smoking_Risk_Drivers": pd.DataFrame(heatmap_rows, columns=driver_cols),
            "Private_Delay_Outcome_Stats": delay_out,
            "Private_Model_Audit": aud_l0,
        }

    # Gate 1 (30..99, min_cls >= 10): L1 LogisticRegression(l1_ratio=1.0, solver='saga', class_weight='balanced')
    # Gate 2 (>= 100, min_cls >= 25): also RandomForestClassifier(max_depth=4, class_weight='balanced') + SHAP[:,:,1]
    num_cols = [c for c in NUMERIC_ONSET_FEATURES if x_df[c].notna().any()]
    cat_cols = list(CATEGORICAL_ONSET_FEATURES)

    tss = TimeSeriesSplit(n_splits=4)
    y_oof_list: list[np.ndarray] = []
    dum_prob_list: list[np.ndarray] = []
    dum_pred_list: list[np.ndarray] = []
    log_prob_list: list[np.ndarray] = []
    log_pred_list: list[np.ndarray] = []
    rf_prob_list: list[np.ndarray] = []
    rf_pred_list: list[np.ndarray] = []

    rf_unlocked = (valid_events >= 100) and (min_cls >= 25)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        for tr_idx, te_idx in tss.split(x_df):
            y_tr, y_te = y_arr[tr_idx], y_arr[te_idx]
            if len(np.unique(y_tr)) < 2:
                continue
            x_tr, x_te = x_df.iloc[tr_idx], x_df.iloc[te_idx]
            y_oof_list.append(y_te)

            p_dum = _make_smoking_clf_pipeline(
                DummyClassifier(strategy="prior"), num_cols, cat_cols
            ).fit(x_tr, y_tr)
            dum_prob_list.append(p_dum.predict_proba(x_te)[:, 1])
            dum_pred_list.append(p_dum.predict(x_te))

            p_log = _make_smoking_clf_pipeline(
                LogisticRegression(
                    l1_ratio=1.0,
                    solver="saga",
                    C=1.0,
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=42,
                ),
                num_cols,
                cat_cols,
            ).fit(x_tr, y_tr)
            log_prob_list.append(p_log.predict_proba(x_te)[:, 1])
            log_pred_list.append(p_log.predict(x_te))

            if rf_unlocked:
                p_rf = _make_smoking_clf_pipeline(
                    RandomForestClassifier(
                        n_estimators=200,
                        max_depth=4,
                        min_samples_leaf=5,
                        class_weight="balanced",
                        random_state=42,
                    ),
                    num_cols,
                    cat_cols,
                ).fit(x_tr, y_tr)
                rf_prob_list.append(p_rf.predict_proba(x_te)[:, 1])
                rf_pred_list.append(p_rf.predict(x_te))

    y_oof = np.concatenate(y_oof_list)
    dum_prob = np.concatenate(dum_prob_list)
    dum_pred = np.concatenate(dum_pred_list)
    log_prob = np.concatenate(log_prob_list)
    log_pred = np.concatenate(log_pred_list)

    has_both_oof = len(np.unique(y_oof)) >= 2
    dum_pr = round(float(average_precision_score(y_oof, dum_prob)), 4) if has_both_oof else np.nan
    dum_bacc = round(float(balanced_accuracy_score(y_oof, dum_pred)), 4) if has_both_oof else np.nan
    log_pr = round(float(average_precision_score(y_oof, log_prob)), 4) if has_both_oof else np.nan
    log_roc = round(float(roc_auc_score(y_oof, log_prob)), 4) if has_both_oof else np.nan
    log_bacc = round(float(balanced_accuracy_score(y_oof, log_pred)), 4) if has_both_oof else np.nan

    rf_pr = rf_roc = rf_bacc = np.nan
    if rf_unlocked and rf_prob_list:
        rf_prob = np.concatenate(rf_prob_list)
        rf_pred = np.concatenate(rf_pred_list)
        if has_both_oof:
            rf_pr = round(float(average_precision_score(y_oof, rf_prob)), 4)
            rf_roc = round(float(roc_auc_score(y_oof, rf_prob)), 4)
            rf_bacc = round(float(balanced_accuracy_score(y_oof, rf_pred)), 4)

    # Fit full-sample interpretability pipeline
    driver_rows: list[dict[str, Any]] = []
    if rf_unlocked and np.isfinite(rf_pr) and (not np.isfinite(log_pr) or rf_pr >= log_pr * 0.98):
        sel_algo = "RandomForestClassifier_SHAP"
        mat_lbl = "Level 3 (Tree Classifier + SHAP)"
        full_pipe = _make_smoking_clf_pipeline(
            RandomForestClassifier(
                n_estimators=200,
                max_depth=4,
                min_samples_leaf=5,
                class_weight="balanced",
                random_state=42,
            ),
            num_cols,
            cat_cols,
        ).fit(x_df, y_arr)
        x_trans = full_pipe.named_steps["pre"].transform(x_df)
        feat_names = [str(f) for f in full_pipe.named_steps["pre"].get_feature_names_out()]
        sv = shap.TreeExplainer(full_pipe.named_steps["clf"]).shap_values(x_trans)
        sv_arr = np.asarray(sv)
        # In shap >= 0.45 (verified shap 0.54.1), binary classifier returns 3D (n_samples, n_features, 2)
        if sv_arr.ndim == 3:
            pos_shap = sv_arr[:, :, 1]
        elif isinstance(sv, list) and len(sv) >= 2:
            pos_shap = np.asarray(sv[1], dtype=float)
        else:
            pos_shap = sv_arr
        mean_abs = np.mean(np.abs(pos_shap), axis=0)
        for fname, mval in sorted(zip(feat_names, mean_abs), key=lambda t: t[1], reverse=True):
            driver_rows.append(
                {
                    "driver_mode": "SHAP_Positive_Class_Importance",
                    "dimension_or_feature": fname,
                    "stratum_value": "Model_Feature",
                    "event_count": valid_events,
                    "smoked_rate": round(float(y_arr.mean()), 4),
                    "model_weight_or_shap": round(float(mval), 5),
                    "direction": "Predictive_Magnitude",
                    "maturity_level": mat_lbl,
                    "epistemic_label": "Prediction",
                    "epistemic_class": "Prediction",
                    "ui_phrasing": f"{fname} contributed to onset risk prediction (PR-AUC={rf_pr:.2f}); not causal proof.",
                }
            )
        best_pr, best_roc, best_bacc = rf_pr, rf_roc, rf_bacc
    else:
        sel_algo = "L1_LogisticRegression_Balanced"
        mat_lbl = "Level 2 (L1 Logistic Classifier)"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            full_pipe = _make_smoking_clf_pipeline(
                LogisticRegression(
                    l1_ratio=1.0,
                    solver="saga",
                    C=1.0,
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=42,
                ),
                num_cols,
                cat_cols,
            ).fit(x_df, y_arr)
        feat_names = [str(f) for f in full_pipe.named_steps["pre"].get_feature_names_out()]
        coefs = np.asarray(full_pipe.named_steps["clf"].coef_, dtype=float).ravel()
        for fname, cval in sorted(zip(feat_names, coefs), key=lambda t: abs(t[1]), reverse=True):
            driver_rows.append(
                {
                    "driver_mode": "L1_Logistic_LogOdds_Weight",
                    "dimension_or_feature": fname,
                    "stratum_value": "Model_Feature",
                    "event_count": valid_events,
                    "smoked_rate": round(float(y_arr.mean()), 4),
                    "model_weight_or_shap": round(float(cval), 5),
                    "direction": "Zero_Shrunk" if abs(cval) <= 1e-8 else ("Higher_Risk" if cval > 0 else "Higher_Resistance"),
                    "maturity_level": mat_lbl,
                    "epistemic_label": "Prediction",
                    "epistemic_class": "Prediction",
                    "ui_phrasing": f"{fname} carried predictive weight (PR-AUC={log_pr:.2f}); predictive weight does not prove causation.",
                }
            )
        best_pr, best_roc, best_bacc = log_pr, log_roc, log_bacc

    sum_out = pd.DataFrame(
        [
            {
                "model_id": "PRIVATE_SMOKING_ONSET_CLF",
                "status": "Active",
                "selected_algorithm": sel_algo,
                "valid_events": valid_events,
                "smoked_class_n": n_smoked,
                "resisted_class_n": n_resisted,
                "min_class_count": min_cls,
                "dummy_pr_auc": dum_pr,
                "dummy_balanced_acc": dum_bacc,
                "logistic_pr_auc": log_pr,
                "logistic_roc_auc": log_roc,
                "logistic_balanced_acc": log_bacc,
                "rf_pr_auc": rf_pr,
                "rf_roc_auc": rf_roc,
                "rf_balanced_acc": rf_bacc,
                "maturity_level": mat_lbl,
                "epistemic_label": "Prediction",
                "epistemic_class": "Prediction",
                "caveat_note": "Strictly pre-decision onset features only; evaluated chronologically via PR-AUC, ROC-AUC, and Balanced Accuracy.",
            }
        ],
        columns=summary_cols,
    )
    aud_out = pd.DataFrame(
        [
            {
                "model_id": "PRIVATE_SMOKING_ONSET_CLF",
                "model_version": "v2.5.0",
                "target_metric": "y_smoked_vs_resisted",
                "algorithm": sel_algo,
                "feature_list": ", ".join(ONSET_FEATURES),
                "train_date_range": d_span,
                "validation_method": "TimeSeriesSplit(n_splits=4)",
                "raw_n": raw_n,
                "valid_n": valid_events,
                "min_class_count": min_cls,
                "pr_auc": best_pr,
                "roc_auc": best_roc,
                "balanced_accuracy": best_bacc,
                "maturity_level": mat_lbl,
                "epistemic_label": "Prediction",
                "epistemic_class": "Prediction",
                "run_timestamp_ist": run_ts,
            }
        ],
        columns=audit_cols,
    )

    return {
        "Private_Smoking_ML_Summary": sum_out,
        "Private_Smoking_Risk_Drivers": pd.DataFrame(driver_rows, columns=driver_cols),
        "Private_Delay_Outcome_Stats": delay_out,
        "Private_Model_Audit": aud_out,
    }
