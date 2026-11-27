"""Unit & Mocked Integration Tests for V2.8 Model-Routed AI Synthesis Layer (`test_v28_ai.py`).

Verifies:
1. ₹500/month INR budget circuit breaker:
   - `< ₹350`: primary Gemini 4 / 3.5 models active
   - `>= ₹350` (70%): auto-downgrades `2A` & `4A` from `gemini-4-hydrogen` -> `gemini-3.5-flash-lite`
   - `>= ₹450` (90%): skips non-essential `1C` & `3C` while keeping essential modules active
   - `>= ₹500` (100%): hard-stops all Gemini API calls and returns `Deterministic_Fallback`
   - Thinking tokens (`thoughts_token_count`) billed at output rates (`usd_to_inr = 95.23`)
2. Gemini 4 API guardrails:
   - Never passes `temperature`, `top_p`, `top_k`, `cached_content`, or `thinking_budget`
   - Rejects `thinking_level="minimal"` on `gemini-4-neon`, `gemini-4-hydrogen`, and `gemini-4-argon`
3. Smoking-free `AI_Briefing_Packet` builder & runtime `PrivacyBoundaryViolationError` gate
4. Incremental caching for Module 1B (`event_id`) and Module 1C (`date`) preventing re-billing
5. Pydantic v2 validation retry -> graceful deterministic fallback on malformed JSON or causal overclaim
6. Search-grounded citation extraction (`grounding_metadata`) + Module 4A stateless Q&A & Simulator Advisor
7. Isolated `private_smoking.ai_smoking` (`>= 5` events/30d gate, zero `Health_Intelligence_Store` writes, silent stdout)
"""

from __future__ import annotations

import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from health_engine.ai.clinical_peer_check import run_module_3a_peer_expectedness
from health_engine.ai.doctor_prep import run_module_3b_doctor_prep
from health_engine.ai.enricher_journal import run_module_1b_journal_enricher
from health_engine.ai.enricher_meals import run_module_1c_meal_auditor
from health_engine.ai.fallbacks import fallback_daily_brief
from health_engine.ai.interactive_service import (
    advise_simulator_target,
    answer_health_question,
)
from health_engine.ai.packet_builder import (
    assert_packet_smoking_free,
    build_ai_briefing_packet,
)
from health_engine.ai.research_pulse import run_module_3c_research_pulse
from health_engine.ai.router import (
    BudgetCircuitBreaker,
    GeminiConfigGuard,
    ModelRouter,
)
from health_engine.ai.schemas import DailyBriefSchema
from health_engine.config_loader import load_ai_routing_config, now_ist
from health_engine.io_adapter import LocalCSVAdapter, PrivacyBoundaryViolationError
from health_engine.run_v28_ai import run_v28_ai_pipeline
from private_smoking.ai_smoking import build_private_smoking_ai
from private_smoking.run_private_ai import main as run_private_ai_main
from private_smoking.silver_smoking import process_silver_smoking


class _FakeModelsEndpoint:
    def __init__(self, responder: Any) -> None:
        self._responder = responder
        self.calls: list[dict[str, Any]] = []

    def generate_content(self, *, model: str, contents: str, config: Any) -> Any:
        call_rec = {"model": model, "contents": contents, "config": config}
        self.calls.append(call_rec)
        return self._responder(call_rec, len(self.calls))


class _FakeGenAIClient:
    def __init__(self, responder: Any) -> None:
        self.models = _FakeModelsEndpoint(responder)


def _make_fake_response(
    text_json: str,
    *,
    prompt_tokens: int = 1200,
    candidates_tokens: int = 400,
    thoughts_tokens: int = 800,
    citations: list[tuple[str, str]] | None = None,
    queries: list[str] | None = None,
) -> Any:
    chunks = [
        SimpleNamespace(web=SimpleNamespace(title=t, uri=u))
        for (t, u) in (citations or [])
    ]
    g_meta = SimpleNamespace(
        grounding_chunks=chunks,
        web_search_queries=list(queries or []),
    )
    usage = SimpleNamespace(
        prompt_token_count=prompt_tokens,
        tool_use_prompt_token_count=0,
        candidates_token_count=candidates_tokens,
        thoughts_token_count=thoughts_tokens,
    )
    return SimpleNamespace(
        text=text_json,
        usage_metadata=usage,
        candidates=[SimpleNamespace(grounding_metadata=g_meta)],
    )


def test_budget_circuit_breaker_tiers_and_thinking_token_accounting() -> None:
    cfg = load_ai_routing_config()
    mk = now_ist().strftime("%Y-%m")

    # 1. Fresh month ($0 / ₹0): primary routes active
    breaker_0 = BudgetCircuitBreaker(cfg)
    d_2a = breaker_0.evaluate_route("2A_daily_brief", month_key=mk)
    assert d_2a.allow_api is True
    assert d_2a.model == "gemini-4-hydrogen"
    assert d_2a.thinking_level == "medium"
    assert d_2a.downgraded is False

    # Verify thinking token output-rate accounting on gemini-4-hydrogen ($1.50/1M in, $7.50/1M out)
    # 100k input = $0.15; 20k candidates + 80k thoughts = 100k output = $0.75 -> total $0.90 * 95.23 = ₹85.707
    usd, inr = breaker_0.estimate_call_cost(
        model="gemini-4-hydrogen",
        prompt_tokens=100_000,
        candidates_tokens=20_000,
        thoughts_tokens=80_000,
        grounded_search_queries=2,
        month_key=mk,
    )
    assert usd == pytest.approx(0.90, abs=1e-5)
    assert inr == pytest.approx(85.707, abs=1e-2)

    # 2. >= ₹350 (70% tier): 2A & 4A downgrade to gemini-3.5-flash-lite; 2B & 3C remain on primary
    ledger_360 = pd.DataFrame([{"month_yyyy_mm": mk, "estimated_inr": 360.0, "grounded_search_queries": 4}])
    breaker_360 = BudgetCircuitBreaker(cfg, ledger_df=ledger_360)
    d_2a_down = breaker_360.evaluate_route("2A_daily_brief", month_key=mk)
    d_4a_down = breaker_360.evaluate_route("4A_interactive_qa_and_simulator", month_key=mk)
    d_2b_keep = breaker_360.evaluate_route("2B_weekly_review", month_key=mk)
    d_3c_keep = breaker_360.evaluate_route("3C_research_pulse", month_key=mk)
    assert d_2a_down.downgraded is True and d_2a_down.model == "gemini-3.5-flash-lite"
    assert d_4a_down.downgraded is True and d_4a_down.model == "gemini-3.5-flash-lite"
    assert d_2b_keep.model == "gemini-4-neon"
    assert d_3c_keep.allow_api is True and d_3c_keep.skip_job is False

    # 3. >= ₹450 (90% tier): non-essential 1C & 3C skipped; essential 2A & 3A still allowed
    ledger_460 = pd.DataFrame([{"month_yyyy_mm": mk, "estimated_inr": 460.0, "grounded_search_queries": 6}])
    breaker_460 = BudgetCircuitBreaker(cfg, ledger_df=ledger_460)
    d_1c_skip = breaker_460.evaluate_route("1C_meal_bioavailability", month_key=mk)
    d_3c_skip = breaker_460.evaluate_route("3C_research_pulse", month_key=mk)
    d_3a_ess = breaker_460.evaluate_route("3A_peer_expectedness", month_key=mk)
    assert d_1c_skip.skip_job is True and d_1c_skip.allow_api is False
    assert d_3c_skip.skip_job is True and d_3c_skip.allow_api is False
    assert d_3a_ess.allow_api is True and d_3a_ess.model == "gemini-4-neon"

    # 4. >= ₹500 (100% hard stop): zero Gemini calls dispatched, returns Deterministic_Fallback
    ledger_505 = pd.DataFrame([{"month_yyyy_mm": mk, "estimated_inr": 505.0, "grounded_search_queries": 10}])
    fake_client = _FakeGenAIClient(lambda _c, _n: pytest.fail("API must NEVER be called >= ₹500!"))
    router_505 = ModelRouter(routing_cfg=cfg, ledger_df=ledger_505, client=fake_client)
    sample_fb = fallback_daily_brief({"meta": {"report_date_display": "27-11-2026"}})
    res_505 = router_505.generate_structured(
        module_id="2A_daily_brief",
        system_prompt="sys",
        user_prompt="usr",
        schema_cls=DailyBriefSchema,
        fallback_fn=lambda: sample_fb,
        month_key=mk,
    )
    assert len(fake_client.models.calls) == 0
    assert res_505.engine_badge == "Deterministic_Fallback"
    assert res_505.fallback_used is True
    assert res_505.status == "Budget_Hard_Stop_500_INR"


def test_gemini4_api_guardrails_forbid_sampling_caching_and_minimal_thinking() -> None:
    for g4_model, lvl in [
        ("gemini-4-hydrogen", "medium"),
        ("gemini-4-neon", "high"),
        ("gemini-4-argon", "high"),
    ]:
        cfg = GeminiConfigGuard.validate_and_build_config(
            model=g4_model,
            system_instruction="test instruction",
            response_schema=DailyBriefSchema,
            thinking_level=lvl,
            use_google_search=(g4_model != "gemini-4-hydrogen"),
        )
        assert cfg.temperature is None
        assert cfg.top_p is None
        assert cfg.top_k is None
        assert cfg.cached_content is None
        assert cfg.thinking_config is not None
        assert cfg.thinking_config.thinking_budget is None
        assert str(cfg.thinking_config.thinking_level).endswith(lvl.upper())

    # Passing temperature, top_p, top_k, or cached_content on Gemini 4 MUST raise ValueError
    for bad_kw in [
        {"temperature": 0.2},
        {"top_p": 0.9},
        {"top_k": 40},
        {"cached_content": "cachedContents/abc"},
        {"thinking_budget": 2048},
    ]:
        with pytest.raises(ValueError, match="Gemini 4 API violation"):
            GeminiConfigGuard.validate_and_build_config(
                model="gemini-4-hydrogen",
                system_instruction="sys",
                response_schema=DailyBriefSchema,
                thinking_level="medium",
                extra_kwargs=bad_kw,
            )

    # Passing thinking_level="minimal" to gemini-4-neon (or any Gemini 4 model) MUST raise ValueError
    for g4_model in ("gemini-4-neon", "gemini-4-hydrogen", "gemini-4-argon"):
        with pytest.raises(ValueError, match="minimal"):
            GeminiConfigGuard.validate_and_build_config(
                model=g4_model,
                system_instruction="sys",
                response_schema=DailyBriefSchema,
                thinking_level="minimal",
            )


def test_briefing_packet_structure_and_strict_smoking_privacy_gate(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    adapter = LocalCSVAdapter(repo_root / "sample_data", tmp_path / "out")

    res = run_v28_ai_pipeline(
        adapter=adapter,
        packet_output_dir=tmp_path / "out",
        run_daily=True,
        dry_run=True,
        persist=True,
    )
    packet = res["packet"]

    expected_sections = {
        "meta",
        "today_and_7d",
        "tirzepatide_and_treatments",
        "weight_alignment",
        "nutrition_and_adherence",
        "stats_and_ml_verified",
        "clinical_context",
    }
    assert expected_sections.issubset(set(packet.keys()))
    assert "readiness_pillars" in packet["today_and_7d"]
    assert len(packet["meta"]["report_date_display"]) == 10
    assert packet["meta"]["report_date_display"][2] == "-"

    # Injecting any smoking key or SMK- value anywhere in packet MUST raise PrivacyBoundaryViolationError
    corrupted_key = copy.deepcopy(packet)
    corrupted_key["nutrition_and_adherence"]["smoking_events_cnt"] = 1
    with pytest.raises(PrivacyBoundaryViolationError):
        assert_packet_smoking_free(corrupted_key)

    corrupted_val = copy.deepcopy(packet)
    corrupted_val["clinical_context"]["notes"] = "Reference SMK-20261122-abc123 from Smoking_Log"
    with pytest.raises(PrivacyBoundaryViolationError):
        assert_packet_smoking_free(corrupted_val)

    with pytest.raises(PrivacyBoundaryViolationError):
        answer_health_question("Summarize my Smoking_Log entries", packet, router=ModelRouter(dry_run=True))


def test_incremental_caching_prevents_rebilling_1b_and_1c() -> None:
    mood_df = pd.DataFrame(
        [
            {
                "Event_ID": "JM-20261125-001",
                "date": "2026-11-25",
                "time_ist": "02:30",
                "Mood_Tags": "Stressed",
                "Context_Tag": "Late work",
                "Summary": "Bahut kaam tha, right shoulder stiff after laptop session",
            }
        ]
    )
    sym_df = pd.DataFrame(
        [
            {
                "Event_ID": "SYM-20261125-001",
                "date": "2026-11-25",
                "time_ist": "07:45",
                "Normalized_Symptom": "Calf Cramp",
                "Context_Tag": "Morning wake",
                "Notes": "Mild left calf cramp at 7:45am",
            }
        ]
    )

    def _responder_1b(call_rec: dict[str, Any], _n: int) -> Any:
        payload_str = call_rec["contents"].split("batch:\n", 1)[1]
        batch_in = json.loads(payload_str)
        items = [
            {
                "event_id": r["event_id"],
                "detected_themes_csv": "Late_Night_Work, Musculoskeletal_Pain",
                "trigger_context": "Late laptop work session",
                "pain_radiation_note": "Right shoulder stiffness",
                "sentiment_polarity": "Negative",
                "epistemic_label": "Descriptive",
            }
            for r in batch_in
        ]
        return _make_fake_response(json.dumps({"items": items}), thoughts_tokens=0)

    fake_1b = _FakeGenAIClient(_responder_1b)
    router_1b = ModelRouter(client=fake_1b)

    # First run: enriches 2 unenriched events (1 API call)
    enriched_1 = run_module_1b_journal_enricher(
        silver_journal_mood=mood_df,
        silver_symptoms=sym_df,
        det_themes_df=None,
        existing_ai_df=None,
        router=router_1b,
    )
    assert len(enriched_1) == 2
    assert len(fake_1b.models.calls) == 1

    # Second run with identical inputs: ZERO API calls!
    enriched_2 = run_module_1b_journal_enricher(
        silver_journal_mood=mood_df,
        silver_symptoms=sym_df,
        det_themes_df=None,
        existing_ai_df=enriched_1,
        router=router_1b,
    )
    assert len(enriched_2) == 2
    assert len(fake_1b.models.calls) == 1

    # Third run appending 1 new event: sends ONLY the 1 new event_id!
    mood_df_plus = pd.concat(
        [
            mood_df,
            pd.DataFrame(
                [
                    {
                        "Event_ID": "JM-20261126-002",
                        "date": "2026-11-26",
                        "time_ist": "21:00",
                        "Mood_Tags": "Calm",
                        "Context_Tag": "Evening walk",
                        "Summary": "Good post-dinner walk",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    enriched_3 = run_module_1b_journal_enricher(
        silver_journal_mood=mood_df_plus,
        silver_symptoms=sym_df,
        det_themes_df=None,
        existing_ai_df=enriched_2,
        router=router_1b,
    )
    assert len(enriched_3) == 3
    assert len(fake_1b.models.calls) == 2
    assert "JM-20261125-001" not in fake_1b.models.calls[1]["contents"]
    assert "JM-20261126-002" in fake_1b.models.calls[1]["contents"]

    # Now verify Module 1C incremental caching by date
    nutr_1 = pd.DataFrame(
        [{"date": "2026-11-25", "time_ist": "13:00", "meal_type": "Lunch", "meal_name": "Fish curry + 2 phulkas", "calories_kcal": "540", "protein_g": "38", "fat_g": "16", "fiber_g": "7"}]
    )
    meds_1 = pd.DataFrame(
        [{"date": "2026-11-25", "time_ist": "08:00", "Med_or_Supplement": "Thyronorm", "Dose": "100mcg", "Status": "Taken", "Timing_Context": "Fasted"}]
    )

    def _responder_1c(call_rec: dict[str, Any], _n: int) -> Any:
        payload_str = call_rec["contents"].split("date:\n", 1)[1]
        days_in = json.loads(payload_str)
        items = [
            {
                "date": d["date"],
                "fat_cofactor_assessment": "16g meal fat supports fat-soluble absorption.",
                "thyronorm_fasting_assessment": "5h gap between 08:00 Thyronorm and 13:00 lunch.",
                "late_spicy_gi_sleep_note": "No post-21:00 heavy spicy meal.",
                "protein_fiber_tweaks": "Add curd + cucumber koshimbir at dinner.",
                "bioavailability_status": "Optimal",
                "epistemic_label": "Descriptive",
            }
            for d in days_in
        ]
        return _make_fake_response(json.dumps({"items": items}), thoughts_tokens=0)

    fake_1c = _FakeGenAIClient(_responder_1c)
    router_1c = ModelRouter(client=fake_1c)

    meals_ai_1 = run_module_1c_meal_auditor(
        silver_nutr_log=nutr_1,
        silver_meds_log=meds_1,
        gold_daily=None,
        gold_event_windows=None,
        existing_ai_df=None,
        router=router_1c,
    )
    assert len(meals_ai_1) == 1
    assert len(fake_1c.models.calls) == 1

    # Re-run on same date -> ZERO new API calls
    meals_ai_2 = run_module_1c_meal_auditor(
        silver_nutr_log=nutr_1,
        silver_meds_log=meds_1,
        gold_daily=None,
        gold_event_windows=None,
        existing_ai_df=meals_ai_1,
        router=router_1c,
    )
    assert len(meals_ai_2) == 1
    assert len(fake_1c.models.calls) == 1


def test_pydantic_validation_retry_then_deterministic_fallback() -> None:
    packet = {"meta": {"report_date_iso": "2026-11-27", "report_date_display": "27-11-2026"}}

    # Case A: Both attempts return invalid JSON / forbidden causal claim -> retries once (2 total calls) -> Deterministic_Fallback
    def _bad_responder(_call: dict[str, Any], n: int) -> Any:
        if n == 1:
            return _make_fake_response("{not valid json")
        # Attempt 2 violates AGENTS.md Rule 3 causal phrase under 'Correlation'
        bad_payload = {
            "overview_one_liner": "Summary",
            "headline": "Headline",
            "inside_your_body_summary": "Body",
            "top_3_patterns": [
                {
                    "pattern_title": "Bedtime",
                    "epistemic_label": "Correlation",
                    "supporting_metric": "rho=-0.8",
                    "caveat": "This definitively causes deeper sleep.",
                }
            ],
            "action_nudges": [
                {
                    "title": "Sleep early",
                    "why": "Helpful",
                    "your_data": "01:00",
                    "evidence": "Obs",
                    "epistemic_label": "Descriptive",
                }
            ],
            "epistemic_label": "Descriptive",
        }
        return _make_fake_response(json.dumps(bad_payload))

    fake_bad = _FakeGenAIClient(_bad_responder)
    router_bad = ModelRouter(client=fake_bad)
    res_fb = router_bad.generate_structured(
        module_id="2A_daily_brief",
        system_prompt="sys",
        user_prompt="usr",
        schema_cls=DailyBriefSchema,
        fallback_fn=lambda: fallback_daily_brief(packet),
    )
    assert len(fake_bad.models.calls) == 2
    assert res_fb.fallback_used is True
    assert res_fb.engine_badge == "Deterministic_Fallback"
    assert isinstance(res_fb.parsed, DailyBriefSchema)


def test_grounded_url_extraction_and_4a_stateless_service() -> None:
    packet = {
        "meta": {"report_date_iso": "2026-11-27", "report_date_display": "27-11-2026"},
        "today_and_7d": {"latest_phenotype": "Earlier-Bedtime Recovery", "readiness_pillars": {}},
        "tirzepatide_and_treatments": {"active_tirz_dose_mg": 5.0},
        "weight_alignment": {"ema_trend_loss_kg_per_wk": 0.85, "masld_progress": []},
        "clinical_context": {"open_retest_windows": []},
    }

    def _grounded_responder(call_rec: dict[str, Any], _n: int) -> Any:
        cfg = call_rec["config"]
        # Verify GoogleSearch tool + response_schema are combined in a single call
        assert cfg.response_schema is not None
        assert cfg.tools is not None and len(cfg.tools) == 1
        assert cfg.tools[0].google_search is not None

        schema_name = cfg.response_schema.__name__
        if schema_name == "PeerExpectednessSchema":
            body = {
                "summary_headline": "Expected GLP-1 adaptation profile",
                "items": [
                    {
                        "symptom_or_shift": "Post-shot satiety peak",
                        "active_therapy_context": "Tirzepatide 5 mg",
                        "safety_tag": "Expected",
                        "clinical_rationale": "Consistent with SURMOUNT-1 trial GI/satiety kinetics.",
                        "reference_source_title": "NEJM SURMOUNT-1",
                        "reference_url": "https://www.nejm.org/doi/full/10.1056/NEJMoa2206038",
                        "epistemic_label": "Screening_Context",
                    }
                ],
                "regimen_interaction_note": "Maintain 45m fasting gap on levothyroxine.",
                "epistemic_label": "Screening_Context",
            }
        elif schema_name == "DoctorVisitPrepSchema":
            body = {
                "specialty": "Endocrinology",
                "printable_clinical_summary": "Endocrinology review summary.",
                "lab_delta_checklist": ["TSH & FT4", "Ferritin"],
                "treatment_scorecard_highlights": "Tirzepatide 5mg + Thyronorm 100mcg.",
                "top_5_questions": [
                    {
                        "priority_rank": 1,
                        "question": "Should we maintain 5mg for another 4 weeks?",
                        "supporting_patient_data": "EMA 0.85 kg/wk",
                        "guideline_context": "Lowest effective dose principle",
                    }
                ],
                "epistemic_label": "Screening_Context",
                "caveat_note": "For physician discussion.",
            }
        elif schema_name == "ResearchPulseSchema":
            body = {
                "digest_headline": "Weekly Metabolic Literature",
                "studies": [
                    {
                        "study_or_guideline_title": "SYNERGY-NASH Tirzepatide Trial",
                        "publication_date_or_year": "2024-2026",
                        "topic_domain": "MASLD_Fatty_Liver",
                        "key_finding_plain_english": "Significant MASH resolution without worsening fibrosis.",
                        "why_it_matters_for_profile": "Supports steady 7-10% weight milestone tracking.",
                        "source_url": "https://www.nejm.org/doi/full/10.1056/NEJMoa2401943",
                        "epistemic_label": "Screening_Context",
                    }
                ],
                "epistemic_label": "Screening_Context",
            }
        else:
            body = {
                "what_your_data_shows": "EMA velocity is 0.85 kg/wk in Earlier-Bedtime Recovery.",
                "what_research_says": "SURMOUNT trials show peak satiety Days 1-3 post-dose.",
                "citations": [{"title": "NEJM SURMOUNT-1", "url": "https://www.nejm.org/doi/full/10.1056/NEJMoa2206038"}],
                "confidence_tier": "Moderate_Observational",
                "epistemic_label": "Screening_Context",
                "ask_your_doctor_note": "Confirm titration pace with your endocrinologist.",
            }

        return _make_fake_response(
            json.dumps(body),
            citations=[
                ("NEJM SURMOUNT-1", "https://www.nejm.org/doi/full/10.1056/NEJMoa2206038"),
                ("NEJM SYNERGY-NASH", "https://www.nejm.org/doi/full/10.1056/NEJMoa2401943"),
            ],
            queries=["tirzepatide SURMOUNT GI kinetics", "tirzepatide MASLD SYNERGY-NASH"],
        )

    fake_grounded = _FakeGenAIClient(_grounded_responder)
    router = ModelRouter(client=fake_grounded)

    df_3a = run_module_3a_peer_expectedness(packet=packet, existing_peer_df=None, router=router)
    assert len(df_3a) == 1
    assert bool(df_3a.iloc[0]["url_grounding_verified"]) is True
    assert "nejm.org" in str(df_3a.iloc[0]["reference_url"])

    df_3b = run_module_3b_doctor_prep(
        specialty="Endocrinology", packet=packet, existing_doc_df=None, router=router
    )
    assert len(df_3b) == 1
    assert df_3b.iloc[0]["model_used"] == "gemini-4-argon"

    df_3c = run_module_3c_research_pulse(packet=packet, existing_pulse_df=None, router=router)
    assert len(df_3c) == 1
    assert bool(df_3c.iloc[0]["url_grounding_verified"]) is True

    qa_out = answer_health_question(
        "Why is my appetite lowest 24-48h after my Tirzepatide shot?",
        packet,
        use_search=True,
        router=router,
    )
    assert qa_out["epistemic_label"] == "Screening_Context"
    assert any("nejm.org" in c["url"] for c in qa_out["citations"])

    # Simulator advisor (use_search=False)
    sim_router = ModelRouter(dry_run=True)
    sim_out = advise_simulator_target(
        92.0, {"protein_g": 105, "bedtime_ist": "00:30", "steps": 9000}, packet, router=sim_router
    )
    assert sim_out["epistemic_label"] == "Prediction"
    assert len(sim_out["prioritized_lever_adjustments"]) >= 2

    ledger_df = router.breaker.to_dataframe()
    assert int(ledger_df["grounded_search_queries"].sum()) == 8


def test_private_smoking_ai_isolation_gate_and_silent_stdout(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # 1. Static AST verification: ai_smoking.py & run_private_ai.py never import health_engine
    priv_dir = Path(__file__).resolve().parents[1] / "src" / "private_smoking"
    for py_path in sorted(priv_dir.glob("*ai*.py")):
        tree = ast.parse(py_path.read_text(encoding="utf-8"), filename=str(py_path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith("health_engine")
            elif isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("health_engine")

    # 2. < 5 events in 30d -> Gated_Below_5_Events_30d with ZERO Gemini calls
    small_raw = pd.DataFrame(
        [
            {"Date": "2026-11-24", "Time": "23:15", "Event_Type": "Smoked", "Count": "1", "Trigger": "Stress_Secret_Marker", "Craving_Intensity": "4"},
            {"Date": "2026-11-25", "Time": "16:00", "Event_Type": "Craving Resisted", "Count": "0", "Trigger": "Boredom", "Craving_Intensity": "3"},
        ]
    )
    small_silver = process_silver_smoking(small_raw)
    never_client = _FakeGenAIClient(lambda _c, _n: pytest.fail("Should gate <5 events!"))
    gated_out = build_private_smoking_ai(small_silver, injected_client=never_client)
    assert gated_out["Private_AI_Coach"].iloc[0]["status"] == "Gated_Below_5_Events_30d"
    assert len(gated_out["Private_AI_Cost_Ledger"]) == 0

    # 3. >= 5 events in 30d -> calls gemini-4-hydrogen (medium thinking, no search) & silent CLI stdout
    priv_in = tmp_path / "priv_in" / "Micro Journal"
    priv_in.mkdir(parents=True)
    six_rows = [
        {
            "Date": f"2026-11-2{i}",
            "Time": "23:30",
            "Event_Type": "Smoked" if i % 2 == 0 else "Craving Resisted",
            "Count": "1" if i % 2 == 0 else "0",
            "Craving_Intensity": "4",
            "Delay_Minutes": "12",
            "Trigger": "Late_Work_Private_Marker_XYZ",
            "Mood": "Anxious",
            "Barrier_or_Reason": "Private_Note_XYZ",
        }
        for i in range(1, 7)
    ]
    six_df = pd.DataFrame(six_rows)
    six_df.to_csv(priv_in / "Smoking_Log.csv", index=False)

    def _priv_responder(call_rec: dict[str, Any], _n: int) -> Any:
        assert call_rec["model"] == "gemini-4-hydrogen"
        assert call_rec["config"].tools is None
        assert call_rec["config"].temperature is None
        payload = {
            "trigger_pattern_synthesis": "Late-evening work cues dominate onset episodes.",
            "friction_tactics": [
                {
                    "tactic_title": "10-Minute Cold Water Delay",
                    "when_to_deploy": "23:00+ work stress",
                    "concrete_protocol": "Drink water and walk to another room for 10 minutes.",
                }
            ],
            "delay_leverage_note": "10+ minute delays markedly improve resistance.",
            "epistemic_label": "Descriptive",
        }
        return _make_fake_response(json.dumps(payload))

    fake_priv = _FakeGenAIClient(_priv_responder)
    active_out = build_private_smoking_ai(process_silver_smoking(six_df), injected_client=fake_priv)
    assert active_out["Private_AI_Coach"].iloc[0]["status"] == "Success"
    assert active_out["Private_AI_Coach"].iloc[0]["model_used"] == "gemini-4-hydrogen"
    assert len(active_out["Private_AI_Cost_Ledger"]) == 1

    # Verify CLI --dry-run writes ONLY to private_out and prints zero raw markers to stdout
    pub_out = tmp_path / "public_store_out"
    priv_out = tmp_path / "private_store_out"
    rc = run_private_ai_main(
        [
            "--source",
            "csv",
            "--data-dir",
            str(tmp_path / "priv_in"),
            "--output-dir",
            str(priv_out),
            "--dry-run",
        ]
    )
    assert rc == 0
    captured = capsys.readouterr()
    assert "Late_Work_Private_Marker_XYZ" not in captured.out
    assert "Private_Note_XYZ" not in captured.out
    assert not pub_out.exists()
    assert (priv_out / "Private_AI_Coach.csv").exists()
    assert (priv_out / "Private_AI_Cost_Ledger.csv").exists()
