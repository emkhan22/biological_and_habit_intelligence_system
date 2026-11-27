"""Micro Journal Structured NLP Themes & Episode Rules (`journal_nlp.py` -> `Gold_Journal_Themes` & `Gold_Episode_Rules`).

Rules (Prompt Section 4 & AGENTS.md Rules 1, 2, 3, 4):
- Reads ONLY non-smoking `Silver_Journal_Mood`, `Silver_Meds_Log`, and `Silver_Symptoms_Log`.
- Applies deterministic multi-label taxonomy (`config/journal_themes.yaml`) across 10 themes:
  `Work_Overwhelm`, `Late_Night_Work`, `Weigh_In_Anxiety`, `Appetite_Suppression_Meal_Skip`,
  `GI_Distress`, `Musculoskeletal_Pain`, `Nocturnal_Cramp`, `Missed_Walk_Guilt`,
  `Positive_Momentum`, `Fatigue_Sleep_Debt`.
- Outputs `Gold_Journal_Themes` (`event_id`, `date`, `detected_themes`,
  `sentiment_polarity_simple`, `human_reviewed_flag = False`) without exposing raw free-text notes,
  and enriches `Gold_Daily_Unified` with daily theme counts (`NaN` when journal unlogged,
  `0` `True_Zero` on active logged days without that theme).
- Gates unsupervised TF-IDF theme discovery behind `valid_notes >= 100` (`Gated_Below_N100_Notes`).
- Aggregates co-occurring episode rules (`Gold_Episode_Rules`) with `episode_count`, `support`,
  `confidence`, and `lift` (suppressing `lift` at `Level 0` until `total_episodes >= 15`).
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

import numpy as np
import pandas as pd

from health_engine.config_loader import add_display_date_col, to_iso_date_str
from health_engine.ml.evaluator import ModelAuditLogger


def _parse_hour_ist(val: Any) -> float:
    if val is None or pd.isna(val):
        return np.nan
    m = re.search(r"(\d{1,2}):(\d{2})", str(val).strip())
    if not m:
        return np.nan
    return float(int(m.group(1))) + float(int(m.group(2))) / 60.0


def _sanitize_event_id(raw_id: Any, prefix: str, ds: str, tm: str, idx: int) -> str:
    s = str(raw_id or "").strip()
    if s:
        # Scrub any internal key token while keeping deterministic ID
        return re.sub(r"smk", "ep", s, flags=re.IGNORECASE)
    digest = hashlib.sha256(f"{prefix}|{ds}|{tm}|{idx}".encode("utf-8")).hexdigest()[:8]
    return f"{prefix}-{ds.replace('-', '')}-{digest}"


def _detect_themes_for_record(
    *,
    mood_txt: str,
    ctx_txt: str,
    sym_txt: str,
    body_txt: str,
    hour_val: float,
    themes_list: list[dict[str, Any]],
) -> list[str]:
    matched: list[str] = []
    mood_l = mood_txt.lower()
    ctx_l = ctx_txt.lower()
    sym_l = sym_txt.lower()
    body_l = body_txt.lower()
    combined_l = f"{mood_l} | {ctx_l} | {sym_l} | {body_l}"

    for tdef in themes_list:
        tid = str(tdef["theme_id"])
        hit = False

        # 1. Circadian hour window + context/text check (e.g. Late_Night_Work 00:00..05:00)
        hwin = tdef.get("hour_window_ist")
        if hwin and len(hwin) == 2 and np.isfinite(hour_val):
            if float(hwin[0]) <= hour_val < float(hwin[1]):
                ctx_pats = [str(p).lower() for p in tdef.get("context_patterns", [])]
                if not ctx_pats or any(p in ctx_l or p in body_l for p in ctx_pats):
                    hit = True

        # 2. Symptom patterns
        for sp in tdef.get("symptom_patterns", []):
            if str(sp).lower() in sym_l:
                hit = True
                break

        # 3. Mood patterns (with optional work-context guard)
        if not hit and tdef.get("mood_patterns"):
            req_work = bool(tdef.get("require_work_context_for_mood", False))
            has_work = ("work" in ctx_l) or ("work" in body_l)
            if (not req_work) or has_work:
                if any(str(mp).lower() in mood_l for mp in tdef.get("mood_patterns", [])):
                    hit = True

        # 4. Text phrase patterns
        if not hit and tdef.get("text_patterns"):
            if any(str(tp).lower() in combined_l for tp in tdef.get("text_patterns", [])):
                hit = True

        if hit:
            matched.append(tid)

    return matched


def _classify_simple_sentiment(
    mood_txt: str,
    sym_txt: str,
    body_txt: str,
    status_txt: str,
    pos_tokens: list[str],
    neg_tokens: list[str],
) -> str:
    combined = f"{mood_txt} {sym_txt} {body_txt} {status_txt}".lower()
    pos_hit = any(tok in combined for tok in pos_tokens)
    neg_hit = any(tok in combined for tok in neg_tokens) or bool(sym_txt.strip())
    if pos_hit and neg_hit:
        return "Mixed"
    if pos_hit:
        return "Positive"
    if neg_hit:
        return "Negative"
    return "Neutral"


def build_gold_journal_nlp(
    gold_daily: pd.DataFrame,
    silver_journal_mood: pd.DataFrame,
    silver_meds: pd.DataFrame,
    silver_symptoms: pd.DataFrame,
    silver_nutr_log: pd.DataFrame,
    thresholds_cfg: dict[str, Any],
    themes_cfg: dict[str, Any],
    audit_logger: ModelAuditLogger | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (Gold_Journal_Themes, Gold_Episode_Rules, enriched_Gold_Daily_Unified)."""
    theme_cols = [
        "event_id",
        "episode_key",
        "source_tab",
        "date",
        "display_date",
        "time_ist",
        "detected_themes",
        "theme_count",
        "sentiment_polarity_simple",
        "human_reviewed_flag",
        "tfidf_discovery_status",
        "epistemic_label",
        "epistemic_class",
    ]
    rule_cols = [
        "rule_id",
        "antecedent",
        "consequent",
        "total_episodes",
        "antecedent_count",
        "episode_count",
        "support",
        "confidence",
        "lift",
        "maturity_level",
        "epistemic_label",
        "epistemic_class",
        "caveat_note",
    ]

    j_gates = thresholds_cfg.get("v25_ml_gates", {}).get("journal_nlp", {})
    min_tfidf_notes = int(
        themes_cfg.get("unsupervised_tfidf_min_notes", j_gates.get("unsupervised_tfidf_min_notes", 100))
    )
    min_lift_eps = int(
        themes_cfg.get("episode_lift_min_episodes", j_gates.get("episode_lift_min_episodes", 15))
    )

    themes_list: list[dict[str, Any]] = list(themes_cfg.get("themes", []))
    all_theme_ids = [str(t["theme_id"]) for t in themes_list]
    lex = themes_cfg.get("sentiment_lexicon", {})
    pos_tokens = [str(x).lower() for x in lex.get("positive_tokens", [])]
    neg_tokens = [str(x).lower() for x in lex.get("negative_tokens", [])]

    # Collect structured entries across the 3 non-smoking Micro Journal tables
    raw_entries: list[dict[str, Any]] = []

    if not silver_journal_mood.empty:
        for i, r in enumerate(silver_journal_mood.to_dict(orient="records")):
            ds = to_iso_date_str(r.get("date") or r.get("Date"))
            if not ds:
                continue
            tm = str(r.get("time_ist") or r.get("Time") or "").strip()
            ev_id = _sanitize_event_id(r.get("Event_ID"), "JM", ds, tm, i)
            ep_key = _sanitize_event_id(r.get("Source_Event_Key") or ev_id, "EP", ds, tm, i)
            raw_entries.append(
                {
                    "event_id": ev_id,
                    "episode_key": ep_key,
                    "source_tab": "Journal_Mood",
                    "date": ds,
                    "time_ist": tm,
                    "mood_txt": str(r.get("Mood_Tags") or ""),
                    "ctx_txt": str(r.get("Context_Tag") or ""),
                    "sym_txt": "",
                    "status_txt": "",
                    "med_txt": "",
                    "body_txt": f"{r.get('Summary', '')} {r.get('Raw_Note', '')}",
                }
            )

    if not silver_symptoms.empty:
        for i, r in enumerate(silver_symptoms.to_dict(orient="records")):
            ds = to_iso_date_str(r.get("date") or r.get("Date"))
            if not ds:
                continue
            tm = str(r.get("time_ist") or r.get("Time") or "").strip()
            ev_id = _sanitize_event_id(r.get("Event_ID"), "SYM", ds, tm, i)
            ep_key = _sanitize_event_id(r.get("Source_Event_Key") or ev_id, "EP", ds, tm, i)
            raw_entries.append(
                {
                    "event_id": ev_id,
                    "episode_key": ep_key,
                    "source_tab": "Symptoms_Log",
                    "date": ds,
                    "time_ist": tm,
                    "mood_txt": "",
                    "ctx_txt": str(r.get("Context_Tag") or ""),
                    "sym_txt": f"{r.get('Normalized_Symptom', '')} {r.get('Symptom', '')}",
                    "status_txt": str(r.get("Severity") or ""),
                    "med_txt": "",
                    "body_txt": f"{r.get('Notes', '')} {r.get('Raw_Note', '')}",
                }
            )

    if not silver_meds.empty:
        for i, r in enumerate(silver_meds.to_dict(orient="records")):
            ds = to_iso_date_str(r.get("date") or r.get("Date"))
            if not ds:
                continue
            tm = str(r.get("time_ist") or r.get("Time") or "").strip()
            ev_id = _sanitize_event_id(r.get("Event_ID"), "MED", ds, tm, i)
            ep_key = _sanitize_event_id(r.get("Source_Event_Key") or ev_id, "EP", ds, tm, i)
            raw_entries.append(
                {
                    "event_id": ev_id,
                    "episode_key": ep_key,
                    "source_tab": "Meds_Log",
                    "date": ds,
                    "time_ist": tm,
                    "mood_txt": "",
                    "ctx_txt": str(r.get("Timing_Context") or ""),
                    "sym_txt": "",
                    "status_txt": str(r.get("Status") or ""),
                    "med_txt": str(r.get("Med_or_Supplement") or ""),
                    "body_txt": str(r.get("Notes") or ""),
                }
            )

    valid_notes_n = len(raw_entries)
    tfidf_status = (
        "Unlocked_TFIDF_Clustering"
        if valid_notes_n >= min_tfidf_notes
        else f"Gated_Below_N{min_tfidf_notes}_Notes"
    )

    theme_rows: list[dict[str, Any]] = []
    daily_theme_counts: dict[str, dict[str, int]] = {}

    for rec in raw_entries:
        ds = str(rec["date"])
        hr_val = _parse_hour_ist(rec["time_ist"])
        t_found = _detect_themes_for_record(
            mood_txt=rec["mood_txt"],
            ctx_txt=rec["ctx_txt"],
            sym_txt=rec["sym_txt"],
            body_txt=rec["body_txt"],
            hour_val=hr_val,
            themes_list=themes_list,
        )
        polarity = _classify_simple_sentiment(
            mood_txt=rec["mood_txt"],
            sym_txt=rec["sym_txt"],
            body_txt=rec["body_txt"],
            status_txt=rec["status_txt"],
            pos_tokens=pos_tokens,
            neg_tokens=neg_tokens,
        )

        d_bucket = daily_theme_counts.setdefault(ds, {tid: 0 for tid in all_theme_ids})
        for tid in t_found:
            d_bucket[tid] = d_bucket.get(tid, 0) + 1

        theme_rows.append(
            {
                "event_id": rec["event_id"],
                "episode_key": rec["episode_key"],
                "source_tab": rec["source_tab"],
                "date": ds,
                "time_ist": rec["time_ist"],
                "detected_themes": ", ".join(t_found),
                "theme_count": len(t_found),
                "sentiment_polarity_simple": polarity,
                "human_reviewed_flag": False,
                "tfidf_discovery_status": tfidf_status,
                "epistemic_label": "Descriptive",
                "epistemic_class": "Descriptive",
            }
        )

    if theme_rows:
        themes_df = add_display_date_col(
            pd.DataFrame(theme_rows).sort_values(["date", "time_ist", "event_id"]).reset_index(drop=True),
            "date",
        )[theme_cols]
    else:
        themes_df = pd.DataFrame(columns=theme_cols)

    # Enrich Gold_Daily_Unified with daily theme counts (strictly NaN when journal unlogged!)
    enriched_daily = gold_daily.copy()
    if not enriched_daily.empty:
        mj_logged_dates = set(daily_theme_counts.keys())
        if "micro_journal_status" in enriched_daily.columns:
            mj_logged_dates |= set(
                enriched_daily.loc[
                    enriched_daily["micro_journal_status"].fillna("").astype(str) == "Present",
                    "date",
                ].astype(str)
            )
        ds_series = enriched_daily["date"].astype(str).tolist()
        for tid in all_theme_ids:
            col_slug = f"theme_{tid.lower()}_cnt"
            enriched_daily[col_slug] = [
                float(daily_theme_counts.get(d, {}).get(tid, 0)) if d in mj_logged_dates else np.nan
                for d in ds_series
            ]

    # 2. Build Episode Co-Occurrence Rules (`Gold_Episode_Rules`)
    # Assemble binary tags per episode window (grouped by date + episode_key AND date-level chains)
    dss_map: dict[str, float] = {}
    if not gold_daily.empty and "days_since_shot" in gold_daily.columns:
        dss_map = pd.to_numeric(gold_daily.set_index("date")["days_since_shot"], errors="coerce").to_dict()

    lunch_dates: set[str] = set()
    any_nutr_dates: set[str] = set()
    if not silver_nutr_log.empty and "date" in silver_nutr_log.columns:
        for r in silver_nutr_log.to_dict(orient="records"):
            ds = to_iso_date_str(r.get("date"))
            if not ds:
                continue
            any_nutr_dates.add(ds)
            hr = _parse_hour_ist(r.get("time_ist"))
            if "lunch" in str(r.get("meal_type") or "").lower() or (np.isfinite(hr) and 11.5 <= hr <= 16.5):
                lunch_dates.add(ds)

    # Build episode baskets keyed by (date, episode_key) supplemented by same-day context
    ep_baskets: dict[tuple[str, str], set[str]] = {}
    day_level_tags: dict[str, set[str]] = {}

    for rec, trow in zip(raw_entries, theme_rows):
        ds = str(rec["date"])
        ek = str(rec["episode_key"])
        bset = ep_baskets.setdefault((ds, ek), set())
        dset = day_level_tags.setdefault(ds, set())

        for tid in [t.strip() for t in str(trow["detected_themes"]).split(",") if t.strip()]:
            bset.add(tid)
            dset.add(tid)

        dss_v = dss_map.get(ds, np.nan)
        if np.isfinite(dss_v) and float(dss_v) in (0.0, 1.0):
            bset.add("Tirz_Shot_Day")
            dset.add("Tirz_Shot_Day")

        if (ds in any_nutr_dates and ds not in lunch_dates) or ("Appetite_Suppression_Meal_Skip" in bset):
            bset.add("Skipped_Afternoon_Meal")
            dset.add("Skipped_Afternoon_Meal")

        st_low = rec["status_txt"].lower()
        ctx_low = rec["ctx_txt"].lower()
        med_low = rec["med_txt"].lower()
        sym_low = rec["sym_txt"].lower()

        if ("skip" in st_low or "miss" in st_low) and (
            "lunch" in ctx_low or any(k in med_low for k in ("matilda", "trazer", "d-rise", "fish oil", "omega"))
        ):
            bset.add("After_Lunch_Meds_Skipped")
            dset.add("After_Lunch_Meds_Skipped")

        if ("skip" in st_low or "miss" in st_low) and "magnesium" in med_low:
            bset.add("Evening_Magnesium_Skipped")
            dset.add("Evening_Magnesium_Skipped")

        if "right shoulder" in sym_low or "Musculoskeletal_Pain" in bset:
            bset.add("Right_Shoulder_Pain")
            dset.add("Right_Shoulder_Pain")

        if "bloat" in sym_low or "bloat" in rec["body_txt"].lower():
            bset.add("Evening_Bloating")
            dset.add("Evening_Bloating")

        if "Missed_Walk_Guilt" in bset:
            bset.add("Missed_Walk")
            dset.add("Missed_Walk")

        if "Late_Night_Work" in bset:
            bset.add("Late_Work_00_05")
            dset.add("Late_Work_00_05")

    # Propagate same-day contextual antecedents (Tirz_Shot_Day, Skipped_Afternoon_Meal, Missed_Walk, Late_Work_00_05)
    for (ds, ek), bset in ep_baskets.items():
        for ctx_tag in ("Tirz_Shot_Day", "Skipped_Afternoon_Meal", "Missed_Walk", "Late_Work_00_05", "Evening_Bloating"):
            if ctx_tag in day_level_tags.get(ds, set()):
                bset.add(ctx_tag)

    baskets_list = list(ep_baskets.values())
    total_eps = len(baskets_list)
    lift_unlocked = total_eps >= min_lift_eps
    rule_mat_lbl = (
        "Level 1 (Association Rules Unlocked)"
        if lift_unlocked
        else "Level 0: Descriptive Episode Log"
    )

    candidate_rules: tuple[tuple[str, tuple[str, ...], str], ...] = (
        ("RULE_SHOT_MEALSKIP_MEDSKIP", ("Tirz_Shot_Day", "Skipped_Afternoon_Meal"), "After_Lunch_Meds_Skipped"),
        ("RULE_LATEWORK_SHOULDER", ("Late_Work_00_05",), "Right_Shoulder_Pain"),
        ("RULE_MISSEDWALK_BLOATING", ("Missed_Walk",), "Evening_Bloating"),
        ("RULE_WEIGHIN_ANXIETY_SHOTDAY", ("Tirz_Shot_Day",), "Weigh_In_Anxiety"),
        ("RULE_LATEWORK_FATIGUE", ("Late_Work_00_05",), "Fatigue_Sleep_Debt"),
    )

    rule_rows: list[dict[str, Any]] = []
    for rid, ant_tuple, cons_tag in candidate_rules:
        ant_set = set(ant_tuple)
        if total_eps == 0:
            ant_cnt = both_cnt = cons_cnt = 0
            supp = conf = lift_v = np.nan
        else:
            ant_cnt = sum(1 for b in baskets_list if ant_set.issubset(b))
            cons_cnt = sum(1 for b in baskets_list if cons_tag in b)
            both_cnt = sum(1 for b in baskets_list if ant_set.issubset(b) and cons_tag in b)
            supp = round(both_cnt / float(total_eps), 4)
            conf = round(both_cnt / float(ant_cnt), 4) if ant_cnt > 0 else np.nan
            p_cons = cons_cnt / float(total_eps)
            if lift_unlocked and ant_cnt > 0 and p_cons > 0:
                lift_v = round((both_cnt / float(ant_cnt)) / p_cons, 4)
            else:
                lift_v = np.nan

        rule_rows.append(
            {
                "rule_id": rid,
                "antecedent": " + ".join(ant_tuple),
                "consequent": cons_tag,
                "total_episodes": total_eps,
                "antecedent_count": ant_cnt,
                "episode_count": both_cnt,
                "support": supp,
                "confidence": conf,
                "lift": lift_v,
                "maturity_level": rule_mat_lbl,
                "epistemic_label": "Descriptive",
                "epistemic_class": "Descriptive",
                "caveat_note": (
                    f"Lift suppressed at Level 0 (total_episodes={total_eps} < {min_lift_eps})."
                    if not lift_unlocked
                    else "Co-occurrence association rule; descriptive episode pattern, not causal proof."
                ),
            }
        )

    rules_df = pd.DataFrame(rule_rows, columns=rule_cols)

    if audit_logger is not None:
        dates_seen = sorted({r["date"] for r in raw_entries})
        d_span = f"{dates_seen[0]}..{dates_seen[-1]}" if dates_seen else ""
        audit_logger.log_model(
            model_id="ML_NLP_JOURNAL_TAXONOMY",
            target_metric="multi_label_journal_themes",
            algorithm=f"Deterministic_YAML_Taxonomy ({tfidf_status})",
            feature_list=all_theme_ids,
            train_date_range=d_span,
            validation_method="Human_Auditable_Rule_Taxonomy",
            raw_n=valid_notes_n,
            valid_n=valid_notes_n,
            effective_n=total_eps,
            predictor_count=len(all_theme_ids),
            missingness_pct=0.0 if valid_notes_n > 0 else 100.0,
            dummy_baseline_mae=np.nan,
            linear_cv_r2=np.nan,
            linear_cv_mae=np.nan,
            ml_cv_r2=np.nan,
            ml_cv_mae=np.nan,
            stability_score=1.0 if valid_notes_n > 0 else np.nan,
            stability_flag="Stable" if valid_notes_n >= min_lift_eps else "Insufficient_N",
            maturity_level=rule_mat_lbl,
            epistemic_label="Descriptive",
        )

    return themes_df, rules_df, enriched_daily
