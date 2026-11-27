"""Module 4B: Strictly Isolated Private Craving Coach (`src/private_smoking/ai_smoking.py`).

Security & Privacy Guarantees (AGENTS.md Rule 2):
- NEVER imports `health_engine.ai` packet builder or writes to `Health_Intelligence_Store_v2`.
- Reads ONLY from `Smoking_Private_Store` tables (`Silver_Smoking_Onset_Context`,
  `Silver_Smoking_Post_Outcomes`, `Private_Smoking_Descriptive`, `Private_Smoking_Risk_Drivers`,
  `Private_Delay_Outcome_Stats`, and `Private_AI_Cost_Ledger`).
- Gates execution strictly behind `>= 5` events in the trailing 30 days (`Gated_Below_5_Events_30d`).
- Routes exclusively to `gemini-4-hydrogen` (`thinking_level="medium"`, Google Search OFF,
  no `temperature`/`top_p`/`top_k`/`cached_content`).
- Logs anonymous token/INR metrics to `Private_AI_Cost_Ledger` and coaching output to
  `Private_AI_Coach` ONLY inside `Smoking_Private_Store`, with ZERO raw text printed to stdout.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
import json
import os
import time
from typing import Any, Literal
import uuid
from zoneinfo import ZoneInfo

from google import genai
from google.genai import types
from google.genai.errors import APIError
import pandas as pd
from pydantic import BaseModel, Field, ValidationError

IST = ZoneInfo("Asia/Kolkata")

PRIVATE_COACH_COLUMNS: list[str] = [
    "coach_run_id",
    "report_date",
    "display_date",
    "generated_at_ist",
    "status",
    "events_30d_gate_passed",
    "trigger_pattern_synthesis",
    "friction_tactics_json",
    "delay_leverage_note",
    "engine_badge",
    "model_used",
    "epistemic_label",
    "epistemic_class",
]

PRIVATE_COST_LEDGER_COLUMNS: list[str] = [
    "call_id",
    "timestamp_ist",
    "month_yyyy_mm",
    "module_id",
    "model_used",
    "prompt_tokens",
    "candidates_tokens",
    "thoughts_tokens",
    "estimated_usd",
    "estimated_inr",
    "status",
    "fallback_used",
    "epistemic_class",
]


class FrictionTacticItem(BaseModel):
    tactic_title: str = Field(description="Short practical behavioral friction tactic title.")
    when_to_deploy: str = Field(description="Onset window or trigger state where this tactic applies.")
    concrete_protocol: str = Field(description="Immediate 2-10 minute friction step.")


class PrivateCravingCoachSchema(BaseModel):
    trigger_pattern_synthesis: str
    friction_tactics: list[FrictionTacticItem]
    delay_leverage_note: str
    epistemic_label: Literal["Descriptive", "Causal_Hypothesis"]


SYSTEM_PROMPT_4B = """You are a confidential, non-judgmental Behavioral Friction & Craving Coach.
Given pre-computed onset trigger distributions, circadian hour buckets, GLP-1 wear-off / sleep-debt context, and delay-bucket resist rates over the past 30 days:
1. Synthesize the primary onset trigger patterns without shaming or moralizing.
2. Propose 2-3 concrete, low-willpower behavioral friction tactics (2-10 minute delay protocols, environment shifts, hydration/oral substitution, bedtime protection).
3. Set epistemic_label to 'Descriptive' or 'Causal_Hypothesis' (never claim causal certainty from small-N observational logs)."""


def _now_ist() -> datetime:
    return datetime.now(tz=IST)


def _now_ist_str() -> str:
    return _now_ist().strftime("%d-%m-%Y %H:%M:%S IST")


def _disp_date(iso_ds: str) -> str:
    if len(iso_ds) >= 10 and iso_ds[4] == "-" and iso_ds[7] == "-":
        return f"{iso_ds[8:10]}-{iso_ds[5:7]}-{iso_ds[0:4]}"
    return iso_ds


def _private_mtd_inr(ledger_df: pd.DataFrame | None, month_key: str) -> float:
    if ledger_df is None or ledger_df.empty or "month_yyyy_mm" not in ledger_df.columns:
        return 0.0
    sub = ledger_df.loc[ledger_df["month_yyyy_mm"].astype(str).str.strip() == month_key]
    vals = pd.to_numeric(sub.get("estimated_inr", pd.Series(dtype=float)), errors="coerce").dropna()
    return round(float(vals.sum()), 4) if not vals.empty else 0.0


def _fallback_private_coach(summary_payload: dict[str, Any]) -> PrivateCravingCoachSchema:
    top_trigs = summary_payload.get("top_onset_triggers_30d", {})
    top_trig_str = ", ".join(f"{k} ({v})" for k, v in list(top_trigs.items())[:3]) or "logged onset cues"
    return PrivateCravingCoachSchema(
        trigger_pattern_synthesis=(
            f"Trailing 30-day onset events cluster around {top_trig_str}. "
            "Patterns are descriptive summaries of pre-decision onset states."
        ),
        friction_tactics=[
            FrictionTacticItem(
                tactic_title="10-Minute Timer + Cold Water Reset",
                when_to_deploy="At initial craving onset during high-risk work/late-night hours",
                concrete_protocol="Start a 10-minute timer, drink 300 ml cold water, and step away from your desk before deciding.",
            ),
            FrictionTacticItem(
                tactic_title="Physical Access Friction Buffer",
                when_to_deploy="Evening / post-meal or late-work stress windows",
                concrete_protocol="Keep supplies out of immediate reach so every episode requires a deliberate multi-step delay.",
            ),
        ],
        delay_leverage_note="Inserting even a 5-10 minute onset delay shifts automatic cue-response loops toward conscious choice.",
        epistemic_label="Descriptive",
    )


def build_private_smoking_ai(
    private_tables: dict[str, pd.DataFrame],
    *,
    existing_coach_df: pd.DataFrame | None = None,
    existing_ledger_df: pd.DataFrame | None = None,
    api_key: str | None = None,
    injected_client: Any | None = None,
    dry_run: bool = False,
    min_events_30d: int = 5,
    monthly_subcap_inr: float = 75.0,
    usd_to_inr: float = 95.23,
) -> dict[str, pd.DataFrame]:
    """Run isolated Module 4B and return `{'Private_AI_Coach': ..., 'Private_AI_Cost_Ledger': ...}`."""
    onset_df = private_tables.get("Silver_Smoking_Onset_Context", pd.DataFrame())
    delay_df = private_tables.get("Private_Delay_Outcome_Stats", pd.DataFrame())
    drivers_df = private_tables.get("Private_Smoking_Risk_Drivers", pd.DataFrame())

    ledger_rows: list[dict[str, Any]] = (
        [dict(r) for r in existing_ledger_df.to_dict(orient="records")]
        if existing_ledger_df is not None and not existing_ledger_df.empty
        else []
    )
    coach_rows: list[dict[str, Any]] = (
        [dict(r) for r in existing_coach_df.to_dict(orient="records")]
        if existing_coach_df is not None and not existing_coach_df.empty
        else []
    )

    if not onset_df.empty and "date" in onset_df.columns:
        max_ds = str(onset_df["date"].dropna().astype(str).max())[:10]
    else:
        max_ds = _now_ist().strftime("%Y-%m-%d")

    cutoff_ds = (date.fromisoformat(max_ds) - timedelta(days=29)).isoformat()
    recent_onset = (
        onset_df.loc[onset_df["date"].astype(str) >= cutoff_ds].copy()
        if not onset_df.empty and "date" in onset_df.columns
        else pd.DataFrame()
    )
    events_30d = int(len(recent_onset))
    ts_now = _now_ist_str()
    month_key = _now_ist().strftime("%Y-%m")
    run_id = f"PCOACH-{max_ds.replace('-', '')}"

    # Gate 1: Require >= 5 events in trailing 30 days
    if events_30d < min_events_30d:
        coach_rows = [r for r in coach_rows if str(r.get("coach_run_id", "")) != run_id]
        coach_rows.append(
            {
                "coach_run_id": run_id,
                "report_date": max_ds,
                "display_date": _disp_date(max_ds),
                "generated_at_ist": ts_now,
                "status": "Gated_Below_5_Events_30d",
                "events_30d_gate_passed": False,
                "trigger_pattern_synthesis": "Fewer than 5 logged events in the trailing 30 days; weekly AI synthesis skipped.",
                "friction_tactics_json": "[]",
                "delay_leverage_note": "Continue logging onset context when cravings arise.",
                "engine_badge": "Gated_Low_Activity",
                "model_used": "None_Gated",
                "epistemic_label": "Descriptive",
                "epistemic_class": "Descriptive",
            }
        )
        return {
            "Private_AI_Coach": pd.DataFrame(coach_rows, columns=PRIVATE_COACH_COLUMNS),
            "Private_AI_Cost_Ledger": pd.DataFrame(ledger_rows, columns=PRIVATE_COST_LEDGER_COLUMNS),
        }

    # Build compact anonymized aggregate summary strictly from Smoking_Private_Store tables
    trig_counts = (
        {
            str(k): int(v)
            for k, v in recent_onset.get("trigger_onset", pd.Series(dtype=str))
            .fillna("")
            .loc[lambda s: s != ""]
            .value_counts()
            .head(5)
            .items()
        }
    )
    work_counts = (
        {
            str(k): int(v)
            for k, v in recent_onset.get("work_mode", pd.Series(dtype=str))
            .fillna("")
            .loc[lambda s: s != ""]
            .value_counts()
            .items()
        }
    )
    summary_payload = {
        "window_days": 30,
        "events_30d": events_30d,
        "top_onset_triggers_30d": trig_counts,
        "work_mode_mix_30d": work_counts,
        "risk_drivers_summary": (
            drivers_df[["dimension_or_feature", "stratum_value", "event_count", "smoked_rate"]]
            .head(8)
            .to_dict(orient="records")
            if not drivers_df.empty
            and {"dimension_or_feature", "stratum_value", "event_count", "smoked_rate"}.issubset(drivers_df.columns)
            else []
        ),
        "delay_outcome_buckets": (
            delay_df[["delay_bucket", "events_n", "resist_rate"]].to_dict(orient="records")
            if not delay_df.empty and {"delay_bucket", "events_n", "resist_rate"}.issubset(delay_df.columns)
            else []
        ),
    }

    resolved_key = api_key if api_key is not None else os.environ.get("GEMINI_API_KEY", "").strip()
    mtd_inr = _private_mtd_inr(pd.DataFrame(ledger_rows) if ledger_rows else None, month_key)
    allow_live = (not dry_run) and (bool(resolved_key) or injected_client is not None) and (mtd_inr < monthly_subcap_inr)

    model_name = "gemini-4-hydrogen"
    parsed_coach: PrivateCravingCoachSchema | None = None
    engine_badge = "Deterministic_Fallback"
    model_used = "Deterministic_Python"
    call_status = "Dry_Run" if dry_run else ("Budget_Subcap_Reached" if mtd_inr >= monthly_subcap_inr else "Missing_API_Key")
    prompt_t = 0
    cand_t = 0
    thought_t = 0
    fallback_used = True

    if allow_live:
        cfg = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT_4B,
            response_mime_type="application/json",
            response_schema=PrivateCravingCoachSchema,
            thinking_config=types.ThinkingConfig(thinking_level="medium"),
        )
        # Gemini 4 convention invariants
        assert cfg.temperature is None and cfg.top_p is None and cfg.top_k is None and cfg.cached_content is None

        client = injected_client or genai.Client(
            api_key=resolved_key,
            http_options=types.HttpOptions(timeout=360_000),
        )
        prompt_str = (
            "Synthesize 2-3 practical behavioral friction tactics from this 30-day onset summary:\n"
            + json.dumps(summary_payload, ensure_ascii=False, indent=2)
        )

        for attempt in range(2):
            try:
                resp = client.models.generate_content(
                    model=model_name,
                    contents=prompt_str,
                    config=cfg,
                )
                usage = getattr(resp, "usage_metadata", None)
                if usage is not None:
                    prompt_t += int(getattr(usage, "prompt_token_count", 0) or 0)
                    cand_t += int(getattr(usage, "candidates_token_count", 0) or 0)
                    thought_t += int(getattr(usage, "thoughts_token_count", 0) or 0)
                raw_txt = str(getattr(resp, "text", "") or "").strip()
                parsed_coach = PrivateCravingCoachSchema.model_validate_json(raw_txt)
                engine_badge = f"Gemini_AI ({model_name})"
                model_used = model_name
                call_status = "Success"
                fallback_used = False
                break
            except (ValidationError, ValueError):
                call_status = "Fallback_Schema_Invalid"
                continue
            except APIError as exc:
                if getattr(exc, "code", None) in {429, 500, 502, 503, 504} and attempt == 0:
                    time.sleep(2.0)
                    continue
                call_status = "Fallback_API_Error"
                break
            except Exception:
                call_status = "Fallback_Exception"
                break

    if parsed_coach is None:
        parsed_coach = _fallback_private_coach(summary_payload)

    in_usd = (prompt_t / 1_000_000.0) * 1.50
    out_usd = ((cand_t + thought_t) / 1_000_000.0) * 7.50
    est_usd = round(in_usd + out_usd, 6)
    est_inr = round(est_usd * usd_to_inr, 4)

    ledger_rows.append(
        {
            "call_id": f"PAIC-{uuid.uuid4().hex[:12]}",
            "timestamp_ist": ts_now,
            "month_yyyy_mm": month_key,
            "module_id": "4B_private_craving_coach",
            "model_used": model_used,
            "prompt_tokens": prompt_t,
            "candidates_tokens": cand_t,
            "thoughts_tokens": thought_t,
            "estimated_usd": est_usd,
            "estimated_inr": est_inr,
            "status": call_status,
            "fallback_used": fallback_used,
            "epistemic_class": "Measured",
        }
    )

    coach_rows = [r for r in coach_rows if str(r.get("coach_run_id", "")) != run_id]
    coach_rows.append(
        {
            "coach_run_id": run_id,
            "report_date": max_ds,
            "display_date": _disp_date(max_ds),
            "generated_at_ist": ts_now,
            "status": call_status,
            "events_30d_gate_passed": True,
            "trigger_pattern_synthesis": parsed_coach.trigger_pattern_synthesis,
            "friction_tactics_json": json.dumps(
                [t.model_dump() for t in parsed_coach.friction_tactics], ensure_ascii=False
            ),
            "delay_leverage_note": parsed_coach.delay_leverage_note,
            "engine_badge": engine_badge,
            "model_used": model_used,
            "epistemic_label": parsed_coach.epistemic_label,
            "epistemic_class": parsed_coach.epistemic_label,
        }
    )

    return {
        "Private_AI_Coach": pd.DataFrame(coach_rows, columns=PRIVATE_COACH_COLUMNS),
        "Private_AI_Cost_Ledger": pd.DataFrame(ledger_rows, columns=PRIVATE_COST_LEDGER_COLUMNS),
    }
