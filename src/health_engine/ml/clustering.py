"""Unsupervised Physiological State Phenotyping (`clustering.py` -> `Gold_Daily_Phenotypes` & `Gold_Phenotype_Profiles`).

Rules (Prompt Section 3 & AGENTS.md Rules 1, 3, 4, 5):
- Input features (standardized via `StandardScaler` on clean Fitbit staged days with valid HR coverage):
  `steps`, `active_hours`, `avg_hr`, `asleep_min`, `bedtime_hr_rel_mid`.
- Fit `KMeans(n_clusters=3, n_init=30, random_state=42)` and compute silhouette scores for `k in {2, 3, 4}`.
- Run 25 random 80%-subsample refits and compute median Adjusted Rand Index (ARI) vs full-sample labels:
  `'Stable'` (`ARI >= 0.70`), `'Moderate'` (`0.50 <= ARI < 0.70`), `'Unstable'` (`ARI < 0.50`).
- Assign semantic labels post-hoc from fitted centroids (never assuming fixed cluster IDs):
  1. Highest standardized `z(asleep_min) - z(bedtime_hr_rel_mid)` -> `Earlier-Bedtime Recovery`
  2. Remaining cluster with higher `steps` -> `High-Exertion Short Sleep`
  3. Remaining cluster with later bedtime & lower steps -> `Late-Night Sedentary Drain`
- Compute rolling 7-calendar-day state counts (`recovery_days_7d`, `high_exertion_short_sleep_7d`,
  `late_drain_days_7d`) and `latest_day_phenotype`.
- Explicitly defer Hidden Markov Models (HMM) to V3.0.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from health_engine.config_loader import add_display_date_col
from health_engine.ml.evaluator import ModelAuditLogger
from health_engine.stats.maturity import compute_effective_n, compute_missingness_pct

CLUSTERING_FEATURES: tuple[str, ...] = (
    "steps",
    "active_hours",
    "avg_hr",
    "asleep_min",
    "bedtime_hr_rel_mid",
)

EXTERNAL_VALIDATION_METRICS: tuple[str, ...] = (
    "hrv_rmssd_ms",
    "deep_min",
    "rem_min",
    "sleep_hr_avg",
)

PHENOTYPE_RECOVERY = "Earlier-Bedtime Recovery"
PHENOTYPE_EXERTION = "High-Exertion Short Sleep"
PHENOTYPE_DRAIN = "Late-Night Sedentary Drain"


def assign_posthoc_phenotype_labels(centroids_df: pd.DataFrame) -> dict[int, str]:
    """Deterministically map `cluster_id -> semantic_phenotype_label` from centroid means.

    Invariant to any permutation of cluster IDs `{0, 1, 2}`:
    1. Highest standardized `z(asleep_min) - z(bedtime_hr_rel_mid)` -> `Earlier-Bedtime Recovery`
    2. Remaining cluster with higher `steps` -> `High-Exertion Short Sleep`
    3. Final remaining cluster -> `Late-Night Sedentary Drain`
    """
    req = {"cluster_id", "asleep_min", "bedtime_hr_rel_mid", "steps"}
    if not req.issubset(centroids_df.columns) or len(centroids_df) < 3:
        return {
            int(r["cluster_id"]): f"Cluster_{int(r['cluster_id'])}"
            for r in centroids_df.to_dict(orient="records")
        }

    work = centroids_df.copy()
    work["cluster_id_int"] = pd.to_numeric(work["cluster_id"], errors="coerce").astype(int)
    asl = pd.to_numeric(work["asleep_min"], errors="coerce").to_numpy(dtype=float)
    bed = pd.to_numeric(work["bedtime_hr_rel_mid"], errors="coerce").to_numpy(dtype=float)
    stp = pd.to_numeric(work["steps"], errors="coerce").to_numpy(dtype=float)

    asl_sd = float(np.std(asl)) if float(np.std(asl)) > 1e-12 else 1.0
    bed_sd = float(np.std(bed)) if float(np.std(bed)) > 1e-12 else 1.0
    z_asl = (asl - float(np.mean(asl))) / asl_sd
    z_bed = (bed - float(np.mean(bed))) / bed_sd

    # Higher asleep_min and earlier (lower) bedtime_hr_rel_mid maximizes recovery_score
    work["_rec_score"] = z_asl - z_bed
    work["_steps_val"] = stp

    # Deterministic tie-breaking by higher asleep_min then lower bedtime_hr_rel_mid
    work_sorted_rec = work.sort_values(
        by=["_rec_score", "asleep_min", "bedtime_hr_rel_mid"],
        ascending=[False, False, True],
    )
    rec_cid = int(work_sorted_rec.iloc[0]["cluster_id_int"])

    rem = work.loc[work["cluster_id_int"] != rec_cid].sort_values(
        by=["_steps_val", "bedtime_hr_rel_mid"],
        ascending=[False, True],
    )
    exert_cid = int(rem.iloc[0]["cluster_id_int"])
    drain_cid = int(rem.iloc[1]["cluster_id_int"])

    return {
        rec_cid: PHENOTYPE_RECOVERY,
        exert_cid: PHENOTYPE_EXERTION,
        drain_cid: PHENOTYPE_DRAIN,
    }


def _classify_ari_stability(
    median_ari: float,
    stable_thresh: float = 0.70,
    moderate_thresh: float = 0.50,
) -> str:
    if not np.isfinite(median_ari):
        return "Insufficient_N"
    if median_ari >= stable_thresh:
        return "Stable"
    if median_ari >= moderate_thresh:
        return "Moderate"
    return "Unstable"


def build_gold_phenotypes(
    gold_daily: pd.DataFrame,
    thresholds_cfg: dict[str, Any],
    audit_logger: ModelAuditLogger | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build (Gold_Daily_Phenotypes, Gold_Phenotype_Profiles) on clean Fitbit days."""
    daily_cols = [
        "date",
        "display_date",
        "cluster_id",
        "phenotype_label",
        "steps",
        "active_hours",
        "avg_hr",
        "asleep_min",
        "bedtime_hr_rel_mid",
        "z_steps",
        "z_active_hours",
        "z_avg_hr",
        "z_asleep_min",
        "z_bedtime_hr_rel_mid",
        "recovery_days_7d",
        "high_exertion_short_sleep_7d",
        "late_drain_days_7d",
        "latest_day_phenotype",
        "median_ari",
        "stability_flag",
        "hmm_status",
        "epistemic_label",
        "epistemic_class",
    ]

    profile_cols = [
        "cluster_id",
        "phenotype_label",
        "days_count",
        "share_pct",
        "steps_mean",
        "steps_median",
        "steps_iqr",
        "active_hours_mean",
        "active_hours_median",
        "active_hours_iqr",
        "avg_hr_mean",
        "avg_hr_median",
        "avg_hr_iqr",
        "asleep_min_mean",
        "asleep_min_median",
        "asleep_min_iqr",
        "bedtime_hr_mean",
        "bedtime_hr_median",
        "bedtime_hr_iqr",
        "ext_hrv_rmssd_mean",
        "ext_hrv_rmssd_median",
        "ext_deep_min_mean",
        "ext_rem_min_mean",
        "ext_sleep_hr_avg_mean",
        "silhouette_k2",
        "silhouette_k3",
        "silhouette_k4",
        "median_ari",
        "stability_flag",
        "maturity_level",
        "hmm_status",
        "epistemic_label",
        "epistemic_class",
        "caveat_note",
    ]

    ccfg = thresholds_cfg.get("v25_ml_gates", {}).get("clustering", {})
    min_days = int(ccfg.get("min_valid_days", 30))
    n_clusters = int(ccfg.get("n_clusters", 3))
    n_init = int(ccfg.get("n_init", 30))
    seed = int(ccfg.get("random_state", 42))
    n_boot = int(ccfg.get("bootstrap_subsamples", 25))
    sub_frac = float(ccfg.get("subsample_fraction", 0.80))
    ari_stable = float(ccfg.get("ari_stable_threshold", 0.70))
    ari_mod = float(ccfg.get("ari_moderate_threshold", 0.50))

    raw_n = len(gold_daily)
    if gold_daily.empty:
        if audit_logger is not None:
            audit_logger.log_model(
                model_id="ML_CLUSTER_KMEANS_3STATE",
                target_metric="unsupervised_daily_phenotype",
                algorithm="None_Gated",
                feature_list=list(CLUSTERING_FEATURES),
                train_date_range="",
                validation_method=f"{n_boot}x_{int(sub_frac*100)}pct_Subsample_ARI",
                raw_n=0,
                valid_n=0,
                effective_n=0,
                predictor_count=len(CLUSTERING_FEATURES),
                missingness_pct=100.0,
                dummy_baseline_mae=np.nan,
                linear_cv_r2=np.nan,
                linear_cv_mae=np.nan,
                ml_cv_r2=np.nan,
                ml_cv_mae=np.nan,
                stability_score=np.nan,
                stability_flag="Insufficient_N",
                maturity_level="Level 0-1 (Gated_Insufficient_N)",
                epistemic_label="Descriptive",
            )
        return pd.DataFrame(columns=daily_cols), pd.DataFrame(columns=profile_cols)

    df = gold_daily.sort_values("date").reset_index(drop=True).copy()
    vcov = df.get("vitals_valid_coverage", pd.Series(True, index=df.index)).fillna(False).astype(bool)
    num_feats = pd.DataFrame(
        {
            c: pd.to_numeric(df.get(c, pd.Series(np.nan, index=df.index)), errors="coerce")
            for c in CLUSTERING_FEATURES
        }
    )
    valid_mask = vcov & (num_feats["asleep_min"] >= 120.0) & num_feats.notna().all(axis=1)
    clean_df = df.loc[valid_mask].sort_values("date").reset_index(drop=True).copy()
    for c in CLUSTERING_FEATURES:
        clean_df[c] = pd.to_numeric(clean_df[c], errors="coerce")

    valid_n = len(clean_df)
    miss_pct = compute_missingness_pct(raw_n, valid_n)

    if valid_n < min_days:
        if audit_logger is not None:
            audit_logger.log_model(
                model_id="ML_CLUSTER_KMEANS_3STATE",
                target_metric="unsupervised_daily_phenotype",
                algorithm="None_Gated",
                feature_list=list(CLUSTERING_FEATURES),
                train_date_range=(
                    f"{clean_df['date'].iloc[0]}..{clean_df['date'].iloc[-1]}" if valid_n > 0 else ""
                ),
                validation_method=f"{n_boot}x_{int(sub_frac*100)}pct_Subsample_ARI",
                raw_n=raw_n,
                valid_n=valid_n,
                effective_n=valid_n,
                predictor_count=len(CLUSTERING_FEATURES),
                missingness_pct=miss_pct,
                dummy_baseline_mae=np.nan,
                linear_cv_r2=np.nan,
                linear_cv_mae=np.nan,
                ml_cv_r2=np.nan,
                ml_cv_mae=np.nan,
                stability_score=np.nan,
                stability_flag="Insufficient_N",
                maturity_level="Level 0-1 (Gated_Insufficient_N)",
                epistemic_label="Descriptive",
            )
        return pd.DataFrame(columns=daily_cols), pd.DataFrame(columns=profile_cols)

    x_raw = clean_df[list(CLUSTERING_FEATURES)].to_numpy(dtype=float)
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_raw)

    km3 = KMeans(n_clusters=n_clusters, n_init=n_init, random_state=seed)
    labels_full = km3.fit_predict(x_scaled)
    clean_df["cluster_id"] = labels_full

    # Diagnostic silhouette scores for k in {2, 3, 4}
    sil_scores: dict[int, float] = {}
    for k_cand in (2, 3, 4):
        if valid_n > k_cand + 1:
            km_k = KMeans(n_clusters=k_cand, n_init=20, random_state=seed)
            lbl_k = km_k.fit_predict(x_scaled)
            if len(set(lbl_k)) > 1:
                sil_scores[k_cand] = round(float(silhouette_score(x_scaled, lbl_k)), 4)
            else:
                sil_scores[k_cand] = np.nan
        else:
            sil_scores[k_cand] = np.nan

    # 25x 80%-subsample bootstrap Adjusted Rand Index (ARI) check
    rng = np.random.default_rng(seed)
    sub_size = max(n_clusters + 2, int(round(sub_frac * valid_n)))
    ari_vals: list[float] = []
    for b in range(n_boot):
        sub_idx = np.sort(rng.choice(valid_n, size=sub_size, replace=False))
        x_sub = StandardScaler().fit_transform(x_raw[sub_idx])
        km_sub = KMeans(n_clusters=n_clusters, n_init=20, random_state=seed + b + 1)
        lbl_sub = km_sub.fit_predict(x_sub)
        ari_vals.append(float(adjusted_rand_score(labels_full[sub_idx], lbl_sub)))

    median_ari = round(float(np.median(ari_vals)), 4) if ari_vals else np.nan
    stab_flag = _classify_ari_stability(median_ari, ari_stable, ari_mod)
    mat_lbl = "Level 3 (Stable Phenotypes)" if valid_n >= 100 else "Level 2 (Exploratory Phenotypes)"

    # Post-hoc semantic centroid labeling
    centroids_raw = (
        clean_df.groupby("cluster_id", as_index=False)[list(CLUSTERING_FEATURES)].mean()
    )
    label_map = assign_posthoc_phenotype_labels(centroids_raw)
    clean_df["phenotype_label"] = clean_df["cluster_id"].map(label_map)

    for j, cname in enumerate(CLUSTERING_FEATURES):
        clean_df[f"z_{cname}"] = np.round(x_scaled[:, j], 4)

    # Rolling 7-calendar-day state counts on every valid Fitbit day
    date_objs = [date.fromisoformat(str(d)) for d in clean_df["date"]]
    pheno_arr = clean_df["phenotype_label"].astype(str).tolist()
    rec_7d: list[int] = []
    ex_7d: list[int] = []
    drn_7d: list[int] = []

    for i, d_curr in enumerate(date_objs):
        d_lo = d_curr - timedelta(days=6)
        r_c = e_c = d_c = 0
        for j in range(i, -1, -1):
            if date_objs[j] < d_lo:
                break
            pl = pheno_arr[j]
            if pl == PHENOTYPE_RECOVERY:
                r_c += 1
            elif pl == PHENOTYPE_EXERTION:
                e_c += 1
            elif pl == PHENOTYPE_DRAIN:
                d_c += 1
        rec_7d.append(r_c)
        ex_7d.append(e_c)
        drn_7d.append(d_c)

    latest_pheno = str(clean_df["phenotype_label"].iloc[-1])
    clean_df["recovery_days_7d"] = rec_7d
    clean_df["high_exertion_short_sleep_7d"] = ex_7d
    clean_df["late_drain_days_7d"] = drn_7d
    clean_df["latest_day_phenotype"] = latest_pheno
    clean_df["median_ari"] = median_ari
    clean_df["stability_flag"] = stab_flag
    clean_df["hmm_status"] = "Deferred_V3.0_Requires_6Mo_Continuous"
    clean_df["epistemic_label"] = "Descriptive"
    clean_df["epistemic_class"] = "Descriptive"

    daily_pheno_df = add_display_date_col(clean_df, "date")[daily_cols]

    # Build Gold_Phenotype_Profiles (centroids + IQR + external validation metrics)
    profile_rows: list[dict[str, Any]] = []
    for cid in sorted(clean_df["cluster_id"].unique()):
        csub = clean_df.loc[clean_df["cluster_id"] == cid]
        cnt = len(csub)

        def _stat_triple(col_k: str) -> tuple[float, float, float]:
            vals = pd.to_numeric(csub[col_k], errors="coerce").dropna().to_numpy(dtype=float)
            if len(vals) == 0:
                return np.nan, np.nan, np.nan
            q75, q25 = np.percentile(vals, [75, 25])
            return (
                round(float(np.mean(vals)), 3),
                round(float(np.median(vals)), 3),
                round(float(q75 - q25), 3),
            )

        def _ext_mean_med(col_k: str) -> tuple[float, float]:
            if col_k not in csub.columns:
                return np.nan, np.nan
            vals = pd.to_numeric(csub[col_k], errors="coerce").dropna().to_numpy(dtype=float)
            if len(vals) == 0:
                return np.nan, np.nan
            return round(float(np.mean(vals)), 3), round(float(np.median(vals)), 3)

        st_m, st_med, st_iqr = _stat_triple("steps")
        ah_m, ah_med, ah_iqr = _stat_triple("active_hours")
        hr_m, hr_med, hr_iqr = _stat_triple("avg_hr")
        as_m, as_med, as_iqr = _stat_triple("asleep_min")
        bt_m, bt_med, bt_iqr = _stat_triple("bedtime_hr_rel_mid")

        hrv_m, hrv_med = _ext_mean_med("hrv_rmssd_ms")
        dp_m, _ = _ext_mean_med("deep_min")
        rm_m, _ = _ext_mean_med("rem_min")
        shr_m, _ = _ext_mean_med("sleep_hr_avg")

        profile_rows.append(
            {
                "cluster_id": int(cid),
                "phenotype_label": label_map[int(cid)],
                "days_count": cnt,
                "share_pct": round((cnt / float(valid_n)) * 100.0, 2),
                "steps_mean": st_m,
                "steps_median": st_med,
                "steps_iqr": st_iqr,
                "active_hours_mean": ah_m,
                "active_hours_median": ah_med,
                "active_hours_iqr": ah_iqr,
                "avg_hr_mean": hr_m,
                "avg_hr_median": hr_med,
                "avg_hr_iqr": hr_iqr,
                "asleep_min_mean": as_m,
                "asleep_min_median": as_med,
                "asleep_min_iqr": as_iqr,
                "bedtime_hr_mean": bt_m,
                "bedtime_hr_median": bt_med,
                "bedtime_hr_iqr": bt_iqr,
                "ext_hrv_rmssd_mean": hrv_m,
                "ext_hrv_rmssd_median": hrv_med,
                "ext_deep_min_mean": dp_m,
                "ext_rem_min_mean": rm_m,
                "ext_sleep_hr_avg_mean": shr_m,
                "silhouette_k2": sil_scores.get(2, np.nan),
                "silhouette_k3": sil_scores.get(3, np.nan),
                "silhouette_k4": sil_scores.get(4, np.nan),
                "median_ari": median_ari,
                "stability_flag": stab_flag,
                "maturity_level": mat_lbl,
                "hmm_status": "Deferred_V3.0_Requires_6Mo_Continuous",
                "epistemic_label": "Descriptive",
                "epistemic_class": "Descriptive",
                "caveat_note": (
                    f"Post-hoc centroid phenotype (k=3 silhouette={sil_scores.get(3, np.nan)}, "
                    f"median 80%-subsample ARI={median_ari}); HMM parked for V3.0."
                ),
            }
        )

    profiles_df = pd.DataFrame(profile_rows, columns=profile_cols)
    eff_n = compute_effective_n(clean_df["asleep_min"], dates=clean_df["date"].tolist())

    if audit_logger is not None:
        audit_logger.log_model(
            model_id="ML_CLUSTER_KMEANS_3STATE",
            target_metric="unsupervised_daily_phenotype",
            algorithm="KMeans(k=3, n_init=30)",
            feature_list=list(CLUSTERING_FEATURES),
            train_date_range=f"{clean_df['date'].iloc[0]}..{clean_df['date'].iloc[-1]}",
            validation_method=f"{n_boot}x_{int(sub_frac*100)}pct_Subsample_ARI + Silhouette(k=2..4)",
            raw_n=raw_n,
            valid_n=valid_n,
            effective_n=eff_n,
            predictor_count=len(CLUSTERING_FEATURES),
            missingness_pct=miss_pct,
            dummy_baseline_mae=np.nan,
            linear_cv_r2=np.nan,
            linear_cv_mae=np.nan,
            ml_cv_r2=sil_scores.get(3, np.nan),
            ml_cv_mae=np.nan,
            stability_score=median_ari,
            stability_flag=stab_flag,
            maturity_level=mat_lbl,
            epistemic_label="Descriptive",
        )

    return daily_pheno_df, profiles_df
