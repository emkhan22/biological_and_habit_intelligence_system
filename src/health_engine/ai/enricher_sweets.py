"""Module 1D: Minimal-Cost Ambiguous Sweet-Meal Classifier (`enricher_sweets.py` -> `Gold_Sweet_Meal_Tags_AI`).

Uses `gemini-3.5-flash-lite` ONLY for ambiguous meal names in `Silver_Nutrition_Log`
(e.g., chai, shake, bar, oats, lassi) and caches every classification by `normalized_meal_name`
so each unique ambiguous meal is billed at most once ever. Automatically skipped if budget >= ₹450
or falls back deterministically at >= ₹500 / `--dry-run`.
"""

from __future__ import annotations

import json
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, Field

from health_engine.ai.packet_builder import assert_packet_smoking_free
from health_engine.ai.router import ModelRouter
from health_engine.config_loader import now_ist_str
from health_engine.stats.sweet_hunger import (
    classify_meal_sweetness_deterministic,
    normalize_meal_name,
)

SWEET_TAGS_AI_COLUMNS: list[str] = [
    "normalized_meal_name",
    "sample_meal_name",
    "is_sweet_ai",
    "sweet_category",
    "rationale",
    "engine_badge",
    "model_used",
    "tagged_at_ist",
    "epistemic_label",
    "epistemic_class",
]


class SweetMealTagItem(BaseModel):
    normalized_meal_name: str
    is_sweet_ai: bool = Field(description="True if item is a sweet, dessert, sugary drink, or sweetened snack.")
    sweet_category: Literal["Dessert_Mithai", "Sweetened_Beverage", "Sweet_Snack_Or_Bar", "Natural_Fruit", "Unsweetened_Savory"]
    rationale: str


class SweetMealTagBatchSchema(BaseModel):
    items: list[SweetMealTagItem]


SYSTEM_PROMPT_1D = """You are an Indian & clinical nutrition meal classifier.
For each ambiguous meal name (with average logged sugar_g and carbs_g), classify whether it is a sweet/dessert/sweetened snack/sugary drink (`is_sweet_ai = true`) vs an unsweetened staple or plain protein shake (`is_sweet_ai = false`).
Keep rationale under 15 words."""


def _fallback_sweet_tags(batch: list[dict[str, Any]]) -> SweetMealTagBatchSchema:
    items: list[SweetMealTagItem] = []
    for r in batch:
        sug = float(r.get("avg_sugar_g") or 0.0)
        is_sw = sug >= 12.0
        cat = "Sweet_Snack_Or_Bar" if is_sw else "Unsweetened_Savory"
        items.append(
            SweetMealTagItem(
                normalized_meal_name=str(r["normalized_meal_name"]),
                is_sweet_ai=is_sw,
                sweet_category=cat,  # type: ignore[arg-type]
                rationale=f"Deterministic sugar threshold ({sug:.1f}g avg sugar).",
            )
        )
    return SweetMealTagBatchSchema(items=items)


def run_module_1d_sweet_classifier(
    *,
    silver_nutr_log: pd.DataFrame | None,
    existing_tags_df: pd.DataFrame | None,
    router: ModelRouter,
    batch_size: int = 25,
) -> pd.DataFrame:
    """Incrementally classify ONLY newly seen ambiguous meal names via `gemini-3.5-flash-lite`."""
    existing_rows: list[dict[str, Any]] = []
    cached_names: set[str] = set()
    if existing_tags_df is not None and not existing_tags_df.empty and "normalized_meal_name" in existing_tags_df.columns:
        for r in existing_tags_df.to_dict(orient="records"):
            nm = normalize_meal_name(r.get("normalized_meal_name"))
            if nm:
                cached_names.add(nm)
                existing_rows.append(dict(r))

    ambiguous_candidates: dict[str, dict[str, Any]] = {}
    if silver_nutr_log is not None and not silver_nutr_log.empty and "meal_name" in silver_nutr_log.columns:
        for r in silver_nutr_log.to_dict(orient="records"):
            nm = normalize_meal_name(r.get("meal_name"))
            if not nm or nm in cached_names:
                continue
            _, is_ambig, _ = classify_meal_sweetness_deterministic(
                r.get("meal_name"), r.get("meal_type"), r.get("sugar_g"), r.get("carbs_g")
            )
            if not is_ambig:
                continue
            sug = pd.to_numeric(r.get("sugar_g"), errors="coerce")
            carb = pd.to_numeric(r.get("carbs_g"), errors="coerce")
            entry = ambiguous_candidates.setdefault(
                nm,
                {
                    "normalized_meal_name": nm,
                    "sample_meal_name": str(r.get("meal_name") or nm),
                    "_sug_list": [],
                    "_carb_list": [],
                },
            )
            if pd.notna(sug):
                entry["_sug_list"].append(float(sug))
            if pd.notna(carb):
                entry["_carb_list"].append(float(carb))

    to_classify: list[dict[str, Any]] = []
    for nm, info in sorted(ambiguous_candidates.items()):
        s_list = info["_sug_list"]
        c_list = info["_carb_list"]
        to_classify.append(
            {
                "normalized_meal_name": nm,
                "sample_meal_name": info["sample_meal_name"],
                "avg_sugar_g": round(sum(s_list) / len(s_list), 1) if s_list else None,
                "avg_carbs_g": round(sum(c_list) / len(c_list), 1) if c_list else None,
            }
        )

    if not to_classify:
        if not existing_rows:
            return pd.DataFrame(columns=SWEET_TAGS_AI_COLUMNS)
        out = pd.DataFrame(existing_rows)
        for col in SWEET_TAGS_AI_COLUMNS:
            if col not in out.columns:
                out[col] = None
        return out[SWEET_TAGS_AI_COLUMNS].sort_values("normalized_meal_name").reset_index(drop=True)

    new_rows: list[dict[str, Any]] = []
    ts_now = now_ist_str()

    for start_i in range(0, len(to_classify), batch_size):
        batch = to_classify[start_i : start_i + batch_size]
        assert_packet_smoking_free(batch, path="module_1d_sweets")
        res = router.generate_structured(
            module_id="1D_sweet_meal_classifier",
            system_prompt=SYSTEM_PROMPT_1D,
            user_prompt="Classify these ambiguous meals:\n" + json.dumps(batch, ensure_ascii=False, indent=2),
            schema_cls=SweetMealTagBatchSchema,
            fallback_fn=lambda b=batch: _fallback_sweet_tags(b),
        )
        if res.status == "Skipped_NonEssential_Above_450_INR" or res.parsed is None:
            break
        parsed: SweetMealTagBatchSchema = res.parsed
        by_nm = {normalize_meal_name(it.normalized_meal_name): it for it in parsed.items}
        for rec in batch:
            nm = rec["normalized_meal_name"]
            item = by_nm.get(nm) or _fallback_sweet_tags([rec]).items[0]
            new_rows.append(
                {
                    "normalized_meal_name": nm,
                    "sample_meal_name": rec["sample_meal_name"],
                    "is_sweet_ai": bool(item.is_sweet_ai),
                    "sweet_category": item.sweet_category,
                    "rationale": item.rationale,
                    "engine_badge": res.engine_badge,
                    "model_used": res.model_used,
                    "tagged_at_ist": ts_now,
                    "epistemic_label": "Descriptive",
                    "epistemic_class": "Descriptive",
                }
            )

    if not existing_rows and not new_rows:
        return pd.DataFrame(columns=SWEET_TAGS_AI_COLUMNS)
    combined = pd.DataFrame(existing_rows + new_rows)
    for col in SWEET_TAGS_AI_COLUMNS:
        if col not in combined.columns:
            combined[col] = None
    return combined[SWEET_TAGS_AI_COLUMNS].drop_duplicates(
        subset=["normalized_meal_name"], keep="last"
    ).sort_values("normalized_meal_name").reset_index(drop=True)
