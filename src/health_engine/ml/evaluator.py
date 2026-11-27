"""Core V2.5 ML Validation, Stability, & Progressive Maturity Guardrails (`evaluator.py`).

Enforced across EVERY V2.5 predictive model (Prompt Section 1 & AGENTS.md Rules 1, 3, 4, 6):
1. Strict Chronological Validation (`TimeSeriesSplit` with `gap=1` daily / `gap=0` intervals)
   and assertion that `max(train_dates) < min(test_dates)` on every fold.
2. Leak-free `sklearn.pipeline.Pipeline` (`SimpleImputer` + `StandardScaler` + `estimator`)
   fit strictly inside each training fold.
3. Baseline-First Comparison (`DummyRegressor` + `RidgeCV` / `ElasticNetCV`).
4. Progressive Maturity Gates:
   - Level 0-1 (`valid_n < 30` daily / `< 24` intervals): block all ML (`Gated_Insufficient_N`).
   - Level 2 (`30 <= valid_n < 100` daily / `>= 24` intervals, or `effective_n < 40`):
     activate `RidgeCV` & `ElasticNetCV` ONLY; keep RF & SHAP locked.
   - Level 3 (`valid_n >= 100` AND `effective_n >= 40`): unlock depth-capped
     `RandomForestRegressor(max_depth=4, min_samples_leaf=5)` head-to-head vs linear.
   - SHAP Unlock Gate: run `shap.TreeExplainer` ONLY when RF beats dummy with `cv_r2 > 0.05`,
     matches/beats linear MAE within 5%, and passes top-3 fold feature-rank stability (`>= 0.60`).
5. Epistemic Honesty Contract (`epistemic_label = 'Prediction'` + non-causal `ui_phrasing`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import warnings

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
import shap
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, ElasticNetCV, Ridge, RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from health_engine.config_loader import now_ist_str, validate_epistemic_class
from health_engine.stats.maturity import compute_effective_n, compute_missingness_pct

MODEL_RUN_AUDIT_COLUMNS: list[str] = [
    "model_id",
    "model_version",
    "target_metric",
    "algorithm",
    "feature_list",
    "train_date_range",
    "validation_method",
    "raw_n",
    "valid_n",
    "effective_n",
    "predictor_count",
    "missingness_pct",
    "dummy_baseline_mae",
    "linear_cv_r2",
    "linear_cv_mae",
    "ml_cv_r2",
    "ml_cv_mae",
    "stability_score",
    "stability_flag",
    "maturity_level",
    "epistemic_label",
    "data_cleaning_version",
    "run_timestamp_ist",
    "epistemic_class",
]


@dataclass
class ModelAuditLogger:
    """Collects standardized V2.5 reproducibility records for `Model_Run_Audit`."""

    run_timestamp_ist: str = field(default_factory=now_ist_str)
    data_cleaning_version: str = "v1.5_silver_gold"
    model_version: str = "v2.5.0"
    records: list[dict[str, Any]] = field(default_factory=list)

    def log_model(
        self,
        *,
        model_id: str,
        target_metric: str,
        algorithm: str,
        feature_list: list[str] | tuple[str, ...],
        train_date_range: str,
        validation_method: str,
        raw_n: int,
        valid_n: int,
        effective_n: int,
        predictor_count: int,
        missingness_pct: float,
        dummy_baseline_mae: float,
        linear_cv_r2: float,
        linear_cv_mae: float,
        ml_cv_r2: float,
        ml_cv_mae: float,
        stability_score: float,
        stability_flag: str,
        maturity_level: str,
        epistemic_label: str = "Prediction",
    ) -> None:
        validate_epistemic_class(epistemic_label)

        def _fmt_num(v: float) -> float:
            return round(float(v), 4) if (v is not None and np.isfinite(v)) else np.nan

        self.records.append(
            {
                "model_id": str(model_id),
                "model_version": self.model_version,
                "target_metric": str(target_metric),
                "algorithm": str(algorithm),
                "feature_list": ", ".join(str(f) for f in feature_list),
                "train_date_range": str(train_date_range),
                "validation_method": str(validation_method),
                "raw_n": int(raw_n),
                "valid_n": int(valid_n),
                "effective_n": int(effective_n),
                "predictor_count": int(predictor_count),
                "missingness_pct": round(float(missingness_pct), 2),
                "dummy_baseline_mae": _fmt_num(dummy_baseline_mae),
                "linear_cv_r2": _fmt_num(linear_cv_r2),
                "linear_cv_mae": _fmt_num(linear_cv_mae),
                "ml_cv_r2": _fmt_num(ml_cv_r2),
                "ml_cv_mae": _fmt_num(ml_cv_mae),
                "stability_score": _fmt_num(stability_score),
                "stability_flag": str(stability_flag),
                "maturity_level": str(maturity_level),
                "epistemic_label": epistemic_label,
                "data_cleaning_version": self.data_cleaning_version,
                "run_timestamp_ist": self.run_timestamp_ist,
                "epistemic_class": epistemic_label,
            }
        )

    def to_dataframe(self) -> pd.DataFrame:
        if not self.records:
            return pd.DataFrame(columns=MODEL_RUN_AUDIT_COLUMNS)
        return pd.DataFrame(self.records, columns=MODEL_RUN_AUDIT_COLUMNS)


def make_leakfree_pipeline(estimator: Any) -> Pipeline:
    """Wrap estimator inside a leak-free Pipeline with fold-local median imputation & scaling."""
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
            ("model", estimator),
        ]
    )


def generate_chronological_splits(
    dates: list[str] | np.ndarray | pd.Series,
    n_splits: int = 5,
    gap: int = 1,
) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """Generate strict chronological TimeSeriesSplit folds and verify zero future date leakage."""
    d_arr = np.asarray([str(d) for d in dates], dtype=object)
    n = len(d_arr)
    if n < (n_splits + 2):
        raise ValueError(f"Insufficient samples ({n}) for TimeSeriesSplit(n_splits={n_splits}).")

    # Verify input is sorted chronologically
    for idx in range(1, n):
        if str(d_arr[idx]) < str(d_arr[idx - 1]):
            raise ValueError(
                f"Input matrix is not sorted chronologically at index {idx}: {d_arr[idx-1]} > {d_arr[idx]}"
            )

    eff_splits = min(n_splits, max(2, (n - gap - 4) // 4))
    tss = TimeSeriesSplit(n_splits=eff_splits, gap=gap)
    splits: list[tuple[int, np.ndarray, np.ndarray]] = []
    for fold_idx, (train_idx, test_idx) in enumerate(tss.split(np.zeros(n))):
        max_train_date = str(d_arr[train_idx].max())
        min_test_date = str(d_arr[test_idx].min())
        if max_train_date >= min_test_date:
            raise RuntimeError(
                f"Chronological leakage detected in fold {fold_idx}: "
                f"max(train_date)={max_train_date} >= min(test_date)={min_test_date}"
            )
        splits.append((fold_idx, train_idx, test_idx))
    return splits


def format_noncausal_ui_phrasing(feature_name: str, cv_r2: float) -> str:
    """Return mandatory non-causal predictive phrasing string (AGENTS.md Rule 3)."""
    r2_txt = f"{cv_r2:.2f}" if np.isfinite(cv_r2) else "N/A"
    return (
        f"{feature_name} was a top predictive feature in this model (CV R² = {r2_txt}); "
        "predictive weight does not prove biological causation."
    )


def _moving_block_bootstrap_linear_signs(
    x_mat: np.ndarray,
    y_vec: np.ndarray,
    base_estimator: Any,
    full_coefs: np.ndarray,
    n_boot: int = 30,
    block_len: int = 5,
    seed: int = 42,
) -> np.ndarray:
    """Compute sign-agreement rate [0..1] per feature across moving-block bootstrap refits."""
    n, p = x_mat.shape
    if n < max(12, block_len * 2) or p == 0:
        return np.full(p, np.nan, dtype=float)

    rng = np.random.default_rng(seed)
    b_len = max(2, min(block_len, n // 3))
    n_blocks = int(np.ceil(n / float(b_len)))
    max_start = max(1, n - b_len + 1)

    coef_signs = np.zeros((n_boot, p), dtype=float)
    valid_boots = 0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        for b in range(n_boot):
            starts = rng.integers(0, max_start, size=n_blocks)
            idx_list: list[int] = []
            for st in starts:
                idx_list.extend(range(int(st), min(n, int(st) + b_len)))
            b_idx = np.asarray(idx_list[:n], dtype=int)
            yb = y_vec[b_idx]
            if np.std(yb) <= 1e-12:
                continue
            pipe = make_leakfree_pipeline(clone(base_estimator))
            pipe.fit(x_mat[b_idx], yb)
            c = np.asarray(pipe.named_steps["model"].coef_, dtype=float).ravel()
            coef_signs[valid_boots] = np.sign(c)
            valid_boots += 1

    if valid_boots == 0:
        return np.full(p, np.nan, dtype=float)

    target_signs = np.sign(full_coefs)
    agree = np.zeros(p, dtype=float)
    for j in range(p):
        if abs(full_coefs[j]) <= 1e-8:
            # Shrunk to zero by L1/L2: stability reflects how often it stayed near-zero or non-opposing
            agree[j] = float(np.mean(coef_signs[:valid_boots, j] == 0.0))
        else:
            agree[j] = float(np.mean(coef_signs[:valid_boots, j] == target_signs[j]))
    return np.round(agree, 4)


def _classify_driver_stability(score: float) -> str:
    if not np.isfinite(score):
        return "Insufficient_N"
    if score >= 0.80:
        return "Stable"
    if score >= 0.60:
        return "Moderate"
    return "Unstable"


def evaluate_progressive_regressor(
    *,
    model_id: str,
    model_label: str,
    df_scope: pd.DataFrame,
    date_col: str,
    target_col: str,
    feature_cols: list[str],
    thresholds_cfg: dict[str, Any],
    audit_logger: ModelAuditLogger | None = None,
    is_interval_grain: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run leak-free TimeSeriesSplit evaluation across Dummy, Ridge/ElasticNet, and gated RF+SHAP."""
    ml_gates = thresholds_cfg.get("v25_ml_gates", {})
    d_gates = ml_gates.get("daily_series", {})
    i_gates = ml_gates.get("weight_intervals", {})
    shap_gates = ml_gates.get("shap_unlock", {})

    if is_interval_grain:
        min_l2_n = int(i_gates.get("level2_linear_min_intervals", 24))
        min_l3_n = 100
        min_l3_eff_n = 40
        n_splits = int(i_gates.get("ts_cv_splits", 4))
        cv_gap = int(i_gates.get("ts_cv_gap", 0))
        gated_l0_status = f"Gated_Below_N{min_l2_n}_Intervals"
        block_len = 2
    else:
        min_l2_n = int(d_gates.get("level2_linear_min_n", 30))
        min_l3_n = int(d_gates.get("level3_tree_min_n", 100))
        min_l3_eff_n = int(d_gates.get("level3_tree_min_effective_n", 40))
        n_splits = int(d_gates.get("ts_cv_splits", 5))
        cv_gap = int(d_gates.get("ts_cv_gap", 1))
        gated_l0_status = "Gated_Insufficient_N"
        block_len = int(d_gates.get("block_length_days", 5))

    n_boot = int(d_gates.get("block_bootstraps", 30))
    shap_min_r2 = float(shap_gates.get("min_cv_r2", 0.05))
    shap_max_mae_ratio = float(shap_gates.get("max_mae_ratio_vs_linear", 1.05))
    shap_min_top3_agree = float(shap_gates.get("min_top3_fold_rank_agreement", 0.60))

    raw_n = len(df_scope)
    if df_scope.empty or not feature_cols or target_col not in df_scope.columns:
        summary = {
            "model_id": model_id,
            "model_label": model_label,
            "target_metric": target_col,
            "status": gated_l0_status,
            "selected_algorithm": "None_Gated",
            "tree_unlock_status": gated_l0_status,
            "shap_unlock_status": gated_l0_status,
            "train_date_range": "",
            "validation_method": f"TimeSeriesSplit(n_splits={n_splits}, gap={cv_gap})",
            "raw_n": raw_n,
            "valid_n": 0,
            "effective_n": 0,
            "predictor_count": len(feature_cols),
            "missingness_pct": 100.0,
            "dummy_baseline_mae": np.nan,
            "ridge_cv_r2": np.nan,
            "ridge_cv_mae": np.nan,
            "elasticnet_cv_r2": np.nan,
            "elasticnet_cv_mae": np.nan,
            "linear_cv_r2": np.nan,
            "linear_cv_mae": np.nan,
            "ml_cv_r2": np.nan,
            "ml_cv_mae": np.nan,
            "stability_score": np.nan,
            "stability_flag": "Insufficient_N",
            "maturity_level": "Level 0-1 (Gated)",
            "epistemic_label": "Prediction",
            "epistemic_class": "Prediction",
            "ui_phrasing": f"{model_label} is currently gated pending sufficient clean observations (n < {min_l2_n}).",
        }
        if audit_logger is not None:
            audit_logger.log_model(
                model_id=model_id,
                target_metric=target_col,
                algorithm="None_Gated",
                feature_list=feature_cols,
                train_date_range="",
                validation_method=summary["validation_method"],
                raw_n=raw_n,
                valid_n=0,
                effective_n=0,
                predictor_count=len(feature_cols),
                missingness_pct=100.0,
                dummy_baseline_mae=np.nan,
                linear_cv_r2=np.nan,
                linear_cv_mae=np.nan,
                ml_cv_r2=np.nan,
                ml_cv_mae=np.nan,
                stability_score=np.nan,
                stability_flag="Insufficient_N",
                maturity_level="Level 0-1 (Gated)",
            )
        return summary, []

    # Sort strictly ascending by date_col (deduplicating identical dates if any)
    work = df_scope.sort_values(date_col, kind="mergesort").drop_duplicates(subset=[date_col], keep="last").reset_index(drop=True).copy()
    y_s = pd.to_numeric(work[target_col], errors="coerce")

    # Select candidate features that have >= 60% non-null coverage across non-null target rows
    # (prevents sparse auxiliary columns from silently wiping out valid rows while obeying Rule 4 No Silent Zero-Filling)
    y_valid_mask = y_s.notna()
    n_y_valid = int(y_valid_mask.sum())
    usable_features: list[str] = []
    for fc in feature_cols:
        if fc not in work.columns:
            continue
        fs = pd.to_numeric(work[fc], errors="coerce")
        cov_on_y = int((y_valid_mask & fs.notna()).sum())
        if n_y_valid > 0 and (cov_on_y >= max(min_l2_n, int(0.60 * n_y_valid))):
            if float(fs.loc[y_valid_mask].std(ddof=0)) > 1e-12:
                usable_features.append(fc)

    if not usable_features:
        usable_features = [fc for fc in feature_cols if fc in work.columns]

    # Filter to rows with non-null target and non-null core usable features
    num_block = work[usable_features].apply(pd.to_numeric, errors="coerce")
    complete_mask = y_s.notna() & num_block.notna().all(axis=1)
    clean_df = work.loc[complete_mask].sort_values(date_col, kind="mergesort").reset_index(drop=True)

    valid_n = len(clean_df)
    miss_pct = compute_missingness_pct(raw_n, valid_n)
    val_method_str = f"TimeSeriesSplit(n_splits={n_splits}, gap={cv_gap})"

    if valid_n == 0:
        eff_n = 0
        date_span = ""
    else:
        eff_n = compute_effective_n(
            pd.to_numeric(clean_df[target_col], errors="coerce"),
            dates=clean_df[date_col].astype(str).tolist(),
        )
        date_span = f"{clean_df[date_col].iloc[0]}..{clean_df[date_col].iloc[-1]}"

    # Gate 1: Level 0-1 (< 30 daily or < 24 weight intervals)
    if valid_n < min_l2_n or len(usable_features) == 0:
        summary = {
            "model_id": model_id,
            "model_label": model_label,
            "target_metric": target_col,
            "status": gated_l0_status,
            "selected_algorithm": "None_Gated",
            "tree_unlock_status": gated_l0_status,
            "shap_unlock_status": gated_l0_status,
            "train_date_range": date_span,
            "validation_method": val_method_str,
            "raw_n": raw_n,
            "valid_n": valid_n,
            "effective_n": eff_n,
            "predictor_count": len(usable_features),
            "missingness_pct": miss_pct,
            "dummy_baseline_mae": np.nan,
            "ridge_cv_r2": np.nan,
            "ridge_cv_mae": np.nan,
            "elasticnet_cv_r2": np.nan,
            "elasticnet_cv_mae": np.nan,
            "linear_cv_r2": np.nan,
            "linear_cv_mae": np.nan,
            "ml_cv_r2": np.nan,
            "ml_cv_mae": np.nan,
            "stability_score": np.nan,
            "stability_flag": "Insufficient_N",
            "maturity_level": f"Level 0-1 ({gated_l0_status})",
            "epistemic_label": "Prediction",
            "epistemic_class": "Prediction",
            "ui_phrasing": (
                f"{model_label} is locked ({gated_l0_status}: valid_n={valid_n} < {min_l2_n}); "
                "no predictive weights are emitted."
            ),
        }
        if audit_logger is not None:
            audit_logger.log_model(
                model_id=model_id,
                target_metric=target_col,
                algorithm="None_Gated",
                feature_list=usable_features,
                train_date_range=date_span,
                validation_method=val_method_str,
                raw_n=raw_n,
                valid_n=valid_n,
                effective_n=eff_n,
                predictor_count=len(usable_features),
                missingness_pct=miss_pct,
                dummy_baseline_mae=np.nan,
                linear_cv_r2=np.nan,
                linear_cv_mae=np.nan,
                ml_cv_r2=np.nan,
                ml_cv_mae=np.nan,
                stability_score=np.nan,
                stability_flag="Insufficient_N",
                maturity_level=f"Level 0-1 ({gated_l0_status})",
            )
        return summary, []

    # Drop any feature that became zero-variance inside clean_df
    active_features = [
        fc
        for fc in usable_features
        if float(pd.to_numeric(clean_df[fc], errors="coerce").std(ddof=0)) > 1e-12
    ]
    x_mat = clean_df[active_features].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    y_vec = pd.to_numeric(clean_df[target_col], errors="coerce").to_numpy(dtype=float)
    dates_vec = clean_df[date_col].astype(str).to_numpy()

    splits = generate_chronological_splits(dates_vec, n_splits=n_splits, gap=cv_gap)
    p = len(active_features)

    # Estimators (verified against scikit-learn 1.9.1 signatures: explicit alphas, no n_alphas)
    dummy_est = DummyRegressor(strategy="mean")
    ridge_est = RidgeCV(alphas=np.logspace(-3, 3, 30))
    enet_est = ElasticNetCV(
        l1_ratio=[0.1, 0.5, 0.9],
        alphas=np.logspace(-3, 2, 30),
        max_iter=3000,
        random_state=42,
    )

    y_oof_list: list[np.ndarray] = []
    dummy_oof_list: list[np.ndarray] = []
    ridge_oof_list: list[np.ndarray] = []
    enet_oof_list: list[np.ndarray] = []

    ridge_fold_coefs = np.zeros((len(splits), p), dtype=float)
    enet_fold_coefs = np.zeros((len(splits), p), dtype=float)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        for f_i, tr_idx, te_idx in splits:
            x_tr, y_tr = x_mat[tr_idx], y_vec[tr_idx]
            x_te, y_te = x_mat[te_idx], y_vec[te_idx]
            y_oof_list.append(y_te)

            p_dum = make_leakfree_pipeline(clone(dummy_est)).fit(x_tr, y_tr)
            dummy_oof_list.append(p_dum.predict(x_te))

            p_rid = make_leakfree_pipeline(clone(ridge_est)).fit(x_tr, y_tr)
            ridge_oof_list.append(p_rid.predict(x_te))
            ridge_fold_coefs[f_i] = np.asarray(p_rid.named_steps["model"].coef_, dtype=float).ravel()

            inner_cv = min(3, max(2, len(tr_idx) // 5))
            enet_fold = ElasticNetCV(
                l1_ratio=[0.1, 0.5, 0.9],
                alphas=np.logspace(-3, 2, 30),
                cv=inner_cv,
                max_iter=3000,
                random_state=42,
            )
            p_en = make_leakfree_pipeline(enet_fold).fit(x_tr, y_tr)
            enet_oof_list.append(p_en.predict(x_te))
            enet_fold_coefs[f_i] = np.asarray(p_en.named_steps["model"].coef_, dtype=float).ravel()

    y_oof = np.concatenate(y_oof_list)
    dummy_oof = np.concatenate(dummy_oof_list)
    ridge_oof = np.concatenate(ridge_oof_list)
    enet_oof = np.concatenate(enet_oof_list)

    dummy_mae = float(mean_absolute_error(y_oof, dummy_oof))
    ridge_mae = float(mean_absolute_error(y_oof, ridge_oof))
    enet_mae = float(mean_absolute_error(y_oof, enet_oof))
    ridge_r2 = float(r2_score(y_oof, ridge_oof)) if np.std(y_oof) > 1e-12 else 0.0
    enet_r2 = float(r2_score(y_oof, enet_oof)) if np.std(y_oof) > 1e-12 else 0.0

    if enet_mae < ridge_mae:
        best_lin_name = "ElasticNetCV"
        best_lin_est: Any = enet_est
        best_lin_r2 = enet_r2
        best_lin_mae = enet_mae
        best_lin_fold_coefs = enet_fold_coefs
    else:
        best_lin_name = "RidgeCV"
        best_lin_est = ridge_est
        best_lin_r2 = ridge_r2
        best_lin_mae = ridge_mae
        best_lin_fold_coefs = ridge_fold_coefs

    # Full-sample linear fit inside leak-free Pipeline for standardized weights
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        full_lin_pipe = make_leakfree_pipeline(clone(best_lin_est)).fit(x_mat, y_vec)
    fitted_lin_model = full_lin_pipe.named_steps["model"]
    full_lin_coefs = np.asarray(fitted_lin_model.coef_, dtype=float).ravel()

    if isinstance(fitted_lin_model, ElasticNetCV):
        boot_est: Any = ElasticNet(
            alpha=float(fitted_lin_model.alpha_),
            l1_ratio=float(fitted_lin_model.l1_ratio_),
            max_iter=2000,
            random_state=42,
        )
    else:
        boot_est = Ridge(alpha=float(getattr(fitted_lin_model, "alpha_", 1.0)))

    boot_sign_agree = _moving_block_bootstrap_linear_signs(
        x_mat=x_mat,
        y_vec=y_vec,
        base_estimator=boot_est,
        full_coefs=full_lin_coefs,
        n_boot=n_boot,
        block_len=block_len,
    )

    # Gate 2 vs Level 3 Tree Ensemble & SHAP Unlock Gate
    tree_unlocked = (valid_n >= min_l3_n) and (eff_n >= min_l3_eff_n)
    ml_r2 = np.nan
    ml_mae = np.nan
    shap_unlocked = False
    shap_mean_abs = np.full(p, np.nan, dtype=float)
    rf_fold_rank_agree = np.nan

    if not tree_unlocked:
        maturity_lbl = "Level 2 (Regularized Linear)"
        tree_status = "Gated_Below_N100" if valid_n < min_l3_n else "Gated_Effective_N_Below_40"
        shap_status = tree_status
        selected_algo = best_lin_name
    else:
        maturity_lbl = "Level 3 (Tree Ensemble + Linear Benchmark)"
        tree_status = "Unlocked_RandomForest_Depth4"
        rf_est = RandomForestRegressor(
            n_estimators=200,
            max_depth=4,
            min_samples_leaf=5,
            random_state=42,
        )
        rf_oof_list: list[np.ndarray] = []
        rf_fold_imps = np.zeros((len(splits), p), dtype=float)
        for f_i, tr_idx, te_idx in splits:
            p_rf = make_leakfree_pipeline(clone(rf_est)).fit(x_mat[tr_idx], y_vec[tr_idx])
            rf_oof_list.append(p_rf.predict(x_mat[te_idx]))
            rf_fold_imps[f_i] = np.asarray(
                p_rf.named_steps["model"].feature_importances_, dtype=float
            )

        rf_oof = np.concatenate(rf_oof_list)
        ml_mae = float(mean_absolute_error(y_oof, rf_oof))
        ml_r2 = float(r2_score(y_oof, rf_oof)) if np.std(y_oof) > 1e-12 else 0.0

        full_rf_pipe = make_leakfree_pipeline(clone(rf_est)).fit(x_mat, y_vec)
        full_rf_model = full_rf_pipe.named_steps["model"]
        full_rf_imps = np.asarray(full_rf_model.feature_importances_, dtype=float)

        k_top = min(3, p)
        full_top_set = set(np.argsort(-full_rf_imps)[:k_top].tolist())
        fold_shares = [
            len(full_top_set & set(np.argsort(-rf_fold_imps[f_i])[:k_top].tolist())) / float(k_top)
            for f_i in range(len(splits))
        ]
        rf_fold_rank_agree = float(np.mean(fold_shares))

        beats_dummy = (ml_r2 > shap_min_r2) and (ml_mae < dummy_mae)
        competitive_with_linear = ml_mae <= (best_lin_mae * shap_max_mae_ratio)
        stable_top3 = rf_fold_rank_agree >= shap_min_top3_agree

        if beats_dummy and competitive_with_linear and stable_top3:
            shap_unlocked = True
            shap_status = "Unlocked_SHAP_TreeExplainer"
            selected_algo = "RandomForestRegressor_SHAP"
            x_trans = full_rf_pipe.named_steps["scaler"].transform(
                full_rf_pipe.named_steps["imputer"].transform(x_mat)
            )
            sv = shap.TreeExplainer(full_rf_model).shap_values(x_trans)
            sv_arr = np.asarray(sv, dtype=float)
            if sv_arr.ndim == 3:
                sv_arr = sv_arr[:, :, 0]
            shap_mean_abs = np.mean(np.abs(sv_arr), axis=0)
        else:
            reasons: list[str] = []
            if not beats_dummy:
                reasons.append("RF_Below_R2_0.05_Or_Dummy")
            if not competitive_with_linear:
                reasons.append("Linear_Beat_RF_By_GT_5pct")
            if not stable_top3:
                reasons.append("Top3_Rank_Unstable")
            shap_status = "Gated_" + "_".join(reasons)
            selected_algo = best_lin_name if best_lin_mae <= ml_mae else "RandomForestRegressor"

    # Assemble per-feature drivers with fold & block-bootstrap sign stability
    driver_rows: list[dict[str, Any]] = []
    fold_sign_scores: list[float] = []

    ref_r2_for_ui = ml_r2 if shap_unlocked else best_lin_r2

    for j, fname in enumerate(active_features):
        lin_c = float(full_lin_coefs[j])
        if abs(lin_c) <= 1e-8:
            fold_same = float(np.mean(np.abs(best_lin_fold_coefs[:, j]) <= 1e-8))
        else:
            fold_same = float(np.mean(np.sign(best_lin_fold_coefs[:, j]) == np.sign(lin_c)))
        boot_same = float(boot_sign_agree[j]) if np.isfinite(boot_sign_agree[j]) else fold_same
        comb_stab = round(0.5 * fold_same + 0.5 * boot_same, 4)
        fold_sign_scores.append(comb_stab)

        if shap_unlocked:
            primary_weight = round(float(shap_mean_abs[j]), 5)
            sp_dir = sp_stats.spearmanr(x_mat[:, j], y_vec).statistic
            direction = "Positive" if (np.isfinite(sp_dir) and sp_dir >= 0) else "Negative"
            metric_type = "mean_abs_shap"
        else:
            primary_weight = round(lin_c, 5)
            direction = "Zero_Shrunk" if abs(lin_c) <= 1e-8 else ("Positive" if lin_c > 0 else "Negative")
            metric_type = f"standardized_{best_lin_name.lower()}_coef"

        driver_rows.append(
            {
                "model_id": model_id,
                "target_metric": target_col,
                "algorithm": selected_algo,
                "feature_name": fname,
                "weight_metric_type": metric_type,
                "feature_weight": primary_weight,
                "linear_std_coef": round(lin_c, 5),
                "mean_abs_shap": round(float(shap_mean_abs[j]), 5) if shap_unlocked else np.nan,
                "direction": direction,
                "pct_folds_same_sign": round(fold_same, 4),
                "pct_boot_same_sign": round(boot_same, 4),
                "stability_score": comb_stab,
                "feature_stability_flag": _classify_driver_stability(comb_stab),
                "maturity_level": maturity_lbl,
                "epistemic_label": "Prediction",
                "epistemic_class": "Prediction",
                "ui_phrasing": format_noncausal_ui_phrasing(fname, ref_r2_for_ui),
            }
        )

    # Sort drivers descending by magnitude of feature_weight
    driver_rows.sort(key=lambda r: abs(float(r["feature_weight"])), reverse=True)
    for rank_i, drow in enumerate(driver_rows, start=1):
        drow["importance_rank"] = rank_i

    top_feat = driver_rows[0]["feature_name"] if driver_rows else "None"
    model_stab_score = (
        round(float(rf_fold_rank_agree), 4)
        if (shap_unlocked and np.isfinite(rf_fold_rank_agree))
        else (round(float(np.mean(fold_sign_scores[: min(3, len(fold_sign_scores))])), 4) if fold_sign_scores else np.nan)
    )
    model_stab_flag = _classify_driver_stability(model_stab_score)

    summary = {
        "model_id": model_id,
        "model_label": model_label,
        "target_metric": target_col,
        "status": "Active",
        "selected_algorithm": selected_algo,
        "tree_unlock_status": tree_status,
        "shap_unlock_status": shap_status,
        "train_date_range": date_span,
        "validation_method": val_method_str,
        "raw_n": raw_n,
        "valid_n": valid_n,
        "effective_n": eff_n,
        "predictor_count": p,
        "missingness_pct": miss_pct,
        "dummy_baseline_mae": round(dummy_mae, 4),
        "ridge_cv_r2": round(ridge_r2, 4),
        "ridge_cv_mae": round(ridge_mae, 4),
        "elasticnet_cv_r2": round(enet_r2, 4),
        "elasticnet_cv_mae": round(enet_mae, 4),
        "linear_cv_r2": round(best_lin_r2, 4),
        "linear_cv_mae": round(best_lin_mae, 4),
        "ml_cv_r2": round(ml_r2, 4) if np.isfinite(ml_r2) else np.nan,
        "ml_cv_mae": round(ml_mae, 4) if np.isfinite(ml_mae) else np.nan,
        "stability_score": model_stab_score,
        "stability_flag": model_stab_flag,
        "maturity_level": maturity_lbl,
        "epistemic_label": "Prediction",
        "epistemic_class": "Prediction",
        "ui_phrasing": format_noncausal_ui_phrasing(top_feat, ref_r2_for_ui),
    }

    if audit_logger is not None:
        audit_logger.log_model(
            model_id=model_id,
            target_metric=target_col,
            algorithm=selected_algo,
            feature_list=active_features,
            train_date_range=date_span,
            validation_method=val_method_str,
            raw_n=raw_n,
            valid_n=valid_n,
            effective_n=eff_n,
            predictor_count=p,
            missingness_pct=miss_pct,
            dummy_baseline_mae=dummy_mae,
            linear_cv_r2=best_lin_r2,
            linear_cv_mae=best_lin_mae,
            ml_cv_r2=ml_r2,
            ml_cv_mae=ml_mae,
            stability_score=model_stab_score,
            stability_flag=model_stab_flag,
            maturity_level=maturity_lbl,
        )

    return summary, driver_rows
