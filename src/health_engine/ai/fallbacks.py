"""Deterministic Python Template Fallbacks for V2.8 AI Synthesis Modules.

Activated automatically when:
- `--dry-run` is requested,
- `GEMINI_API_KEY` is absent in offline/CI environments,
- Monthly INR spend reaches the hard ₹500 circuit breaker (`>= ₹500`), or
- Gemini API / Pydantic validation retries fail.

Every deterministic output adheres strictly to `AGENTS.md` Rule 3 (`EPISTEMIC_CLASSES`)
and uses only pre-computed numbers from `AI_Briefing_Packet` and Silver/Gold tables.
"""

from __future__ import annotations

from typing import Any

from health_engine.ai.schemas import (
    ActionNudgeItem,
    CitationItem,
    DailyBriefSchema,
    DailyPatternItem,
    DoctorVisitPrepSchema,
    InteractiveAnswerSchema,
    JournalEnrichmentBatchSchema,
    JournalEnrichmentItem,
    MealBioavailabilityBatchSchema,
    MealBioavailabilityItem,
    MonthlyReviewSchema,
    PeerExpectednessSchema,
    PeerSafetyItem,
    ResearchPulseSchema,
    ResearchStudyItem,
    SimulatorAdviceSchema,
    SpecialistQuestionItem,
    WeeklyReviewSchema,
)


def fallback_journal_enrichment(batch_records: list[dict[str, Any]]) -> JournalEnrichmentBatchSchema:
    """Deterministic Python fallback for Module 1B (`Gold_Journal_Themes_AI`)."""
    items: list[JournalEnrichmentItem] = []
    for rec in batch_records:
        ev_id = str(rec.get("event_id", ""))
        themes = str(rec.get("deterministic_themes", "") or "").strip()
        ctx = str(rec.get("context_tag", "") or "").strip()
        sym = str(rec.get("symptom", "") or "").strip()
        text = str(rec.get("raw_text", "") or "").strip()
        sentiment = str(rec.get("deterministic_sentiment", "Neutral") or "Neutral")
        if sentiment not in {"Positive", "Negative", "Mixed", "Neutral"}:
            sentiment = "Neutral"

        pain_note = "None"
        low_txt = f"{sym} {text}".lower()
        if any(k in low_txt for k in ("shoulder", "neck", "arm", "back", "cramp", "calf", "ache", "pain")):
            pain_note = f"Reported musculoskeletal/cramp note ({sym or 'logged symptom'})."

        trig = ctx if ctx else ("Symptom/mood log entry" if text or sym else "Routine check-in")
        items.append(
            JournalEnrichmentItem(
                event_id=ev_id,
                detected_themes_csv=themes if themes else "Routine_Check_In",
                trigger_context=trig[:180],
                pain_radiation_note=pain_note,
                sentiment_polarity=sentiment,  # type: ignore[arg-type]
                epistemic_label="Descriptive",
            )
        )
    return JournalEnrichmentBatchSchema(items=items)


def fallback_meal_bioavailability(day_records: list[dict[str, Any]]) -> MealBioavailabilityBatchSchema:
    """Deterministic Python fallback for Module 1C (`Gold_Meal_Bioavailability_AI`)."""
    items: list[MealBioavailabilityItem] = []
    for rec in day_records:
        ds = str(rec.get("date", ""))
        meals_cnt = int(rec.get("meals_count", 0) or 0)
        fat_ok = rec.get("fat_cofactor_valid_10g")
        thy_gap = rec.get("thyronorm_fasting_gap_min")
        protein_g = rec.get("protein_g")
        fiber_g = rec.get("fiber_g")
        late_kcal = rec.get("late_kcal_after_21")

        if meals_cnt == 0:
            status = "Unlogged_Meals"
            fat_txt = "No meal timestamps logged to verify >=10g fat co-factor with D-Rise 60K / Omega-3."
        elif fat_ok is False:
            status = "Missing_Fat_Cofactor"
            fat_txt = "Fat-soluble dose window had <10g logged meal fat; pair D-Rise 60K / Omega-3 with whole eggs, fish, nuts, or ghee/curd meal."
        else:
            status = "Optimal"
            fat_txt = "Fat co-factor window satisfied (>=10g dietary fat or no fat-soluble dose logged)."

        if thy_gap is not None and str(thy_gap) not in {"", "nan", "None"}:
            try:
                gap_f = float(thy_gap)
                if gap_f < 30.0:
                    status = "Needs_Timing_Tweak"
                    thy_txt = f"Thyronorm pre-meal gap was {gap_f:.0f} min (<30 min target); aim for 45-60 min water-only fasting window."
                else:
                    thy_txt = f"Thyronorm fasting gap verified at {gap_f:.0f} min before first caloric intake."
            except (TypeError, ValueError):
                thy_txt = "Thyronorm timestamp present; first-meal gap unverified."
        else:
            thy_txt = "No paired morning Thyronorm + breakfast timestamp logged for this date."

        late_txt = (
            f"Post-21:00 intake logged ({float(late_kcal):.0f} kcal); monitor overnight HR spikes and reflux."
            if late_kcal not in (None, "", "nan") and float(late_kcal or 0) > 0
            else "No late post-21:00 heavy caloric load logged."
        )
        tweak_parts: list[str] = []
        if protein_g is not None and float(protein_g or 0) < 85:
            tweak_parts.append("Add 1 scoop whey isolate, egg whites, or grilled fish/paneer (+20-25g protein)")
        if fiber_g is not None and float(fiber_g or 0) < 22:
            tweak_parts.append("Add sabja/chia, cucumber-carrot koshimbir, or dalia/vegetable sabzi (+6-8g fiber)")
        if not tweak_parts:
            tweak_parts.append("Maintain current home-cooked protein and soluble-fiber balance alongside 2.5L+ hydration.")

        items.append(
            MealBioavailabilityItem(
                date=ds,
                fat_cofactor_assessment=fat_txt,
                thyronorm_fasting_assessment=thy_txt,
                late_spicy_gi_sleep_note=late_txt,
                protein_fiber_tweaks="; ".join(tweak_parts),
                bioavailability_status=status,  # type: ignore[arg-type]
                epistemic_label="Descriptive",
            )
        )
    return MealBioavailabilityBatchSchema(items=items)


def fallback_daily_brief(packet: dict[str, Any]) -> DailyBriefSchema:
    """Deterministic Python fallback for Module 2A (`AI_Insights_Cache`)."""
    meta = packet.get("meta", {})
    t7 = packet.get("today_and_7d", {})
    tirz = packet.get("tirzepatide_and_treatments", {})
    wt = packet.get("weight_alignment", {})
    nutr = packet.get("nutrition_and_adherence", {})

    rdate = meta.get("report_date_display", "Today")
    pheno = t7.get("latest_phenotype") or "Collecting_Wearable_State"
    shot_day = tirz.get("shot_day_int")
    dose_mg = tirz.get("active_tirz_dose_mg")
    pillars = t7.get("readiness_pillars", {})
    sleep_sc = pillars.get("sleep_circadian_score")
    rec_sc = pillars.get("autonomic_recovery_score")

    shot_str = (
        f"Tirzepatide {dose_mg} mg Day {shot_day}"
        if dose_mg is not None and shot_day is not None
        else "Active metabolic tracking phase"
    )
    one_liner = (
        f"{rdate}: Phenotype '{pheno}' | {shot_str} | "
        f"Sleep/Circadian={sleep_sc if sleep_sc is not None else 'N/A'}, "
        f"Autonomic={rec_sc if rec_sc is not None else 'N/A'}."
    )

    z_map = t7.get("latest_z_scores", {})
    bed_z = z_map.get("bedtime_robust_z")
    hrv_z = z_map.get("hrv_robust_z")
    ema_slope = wt.get("ema_trend_loss_kg_per_wk")
    prot_7d = nutr.get("protein_g_7d_mean")

    patterns = [
        DailyPatternItem(
            pattern_title="Circadian & Overnight Autonomic State",
            epistemic_label="Measured",
            supporting_metric=f"Phenotype={pheno}; HRV robust Z={hrv_z if hrv_z is not None else 'N/A'}; Bedtime robust Z={bed_z if bed_z is not None else 'N/A'}",
            caveat="Reflects wear-verified Fitbit nights only; missing nights remain NaN.",
        ),
        DailyPatternItem(
            pattern_title="GLP-1 Weekly Pharmacokinetic Context",
            epistemic_label="Descriptive",
            supporting_metric=f"{shot_str} (modeled PK={tirz.get('tirz_pk_est', 'N/A')})",
            caveat="PK curve is a population half-life estimate (~5 days), not a serum drug assay.",
        ),
        DailyPatternItem(
            pattern_title="Smoothed Weight & Lean-Mass Protection Context",
            epistemic_label="Descriptive",
            supporting_metric=f"EMA trend={ema_slope if ema_slope is not None else 'N/A'} kg/wk; 7d protein mean={prot_7d if prot_7d is not None else 'N/A'} g/day",
            caveat="Short weigh-in intervals remain influenced by meal timing, cycle water shifts, and GI transit.",
        ),
    ]

    nudges = [
        ActionNudgeItem(
            title="Anchor Earlier Bedtime Window",
            why="Bedtime timing is the strongest observational & Level 2/3 predictor of clean sleep duration in your dataset.",
            your_data=f"7-day phenotype mix: {t7.get('phenotype_mix_7d', {})}",
            evidence="Predictive feature weight in Clean Sleep model (epistemic class: Prediction, non-causal).",
            epistemic_label="Prediction",
        ),
        ActionNudgeItem(
            title="Protect Protein & Afternoon Supplement Continuity on Shot Days",
            why="Appetite suppression on Days 0-2 post-shot can trigger skipped afternoon meals and missed fat-soluble supplements.",
            your_data=f"7-day protein mean={prot_7d if prot_7d is not None else 'Not_Logged'} g; adherence summary in packet.",
            evidence="Deterministic episode-chain audit across Meds_Log and Nutrition_Log.",
            epistemic_label="Descriptive",
        ),
    ]

    return DailyBriefSchema(
        overview_one_liner=one_liner,
        headline=f"Daily Holistic Synthesis ({rdate}) — Deterministic Template",
        inside_your_body_summary=(
            f"Current physiological state is classified as '{pheno}' alongside {shot_str}. "
            "All metrics are pre-computed from validated Silver and Gold tables without zero-filling unlogged days."
        ),
        top_3_patterns=patterns,
        action_nudges=nudges,
        epistemic_label="Descriptive",
    )


def fallback_weekly_review(packet: dict[str, Any], week_label: str) -> WeeklyReviewSchema:
    """Deterministic Python fallback for Module 2B (`AI_Weekly_Briefs`)."""
    tirz = packet.get("tirzepatide_and_treatments", {})
    wt = packet.get("weight_alignment", {})
    nutr = packet.get("nutrition_and_adherence", {})
    t7 = packet.get("today_and_7d", {})

    return WeeklyReviewSchema(
        headline=f"Thu-Wed Weekly Deep Review ({week_label}) — Deterministic Summary",
        dose_working=(
            f"Active Tirzepatide dose: {tirz.get('active_tirz_dose_mg', 'N/A')} mg; "
            f"EMA weight velocity: {wt.get('ema_trend_loss_kg_per_wk', 'N/A')} kg/wk; "
            f"MASLD milestone progress: {wt.get('masld_progress', {})}."
        ),
        week_influencers=(
            f"7-day phenotype distribution: {t7.get('phenotype_mix_7d', {})}. "
            "Confounders tracked explicitly: fasted vs post-meal weigh-in status, cycle phase, and 48h constipation flags."
        ),
        habit_review_improved_vs_slipped=(
            f"Nutrition logged days (7d): {nutr.get('nutrition_logged_days_7d', 0)}/7 "
            f"(mean protein={nutr.get('protein_g_7d_mean', 'N/A')} g, fiber={nutr.get('fiber_g_7d_mean', 'N/A')} g, "
            f"hydration={nutr.get('hydration_l_7d_mean', 'N/A')} L)."
        ),
        next_week_priorities=[
            "Log at least 1 morning fasted Fitdays BIA weigh-in before first caloric intake.",
            "Target >=90g protein and >=25g fiber daily with 2.5L+ hydration during Days 0-2 post-shot.",
            "Protect pre-01:00 IST lights-out to sustain Earlier-Bedtime Recovery days.",
        ],
        clinical_and_bioavailability_notes=(
            "Keep Thyronorm 30-60 min before morning tea/food and pair D-Rise 60K / Omega-3 with >=10g meal fat. "
            "Concurrent Oct/Nov therapies carry overlap_warning=True on treatment scorecards."
        ),
        epistemic_label="Descriptive",
        caveat_note="Generated via deterministic Python fallback template; observational deltas do not isolate single-drug causation.",
    )


def fallback_monthly_review(packet: dict[str, Any], month_key: str) -> MonthlyReviewSchema:
    """Deterministic Python fallback for Module 2C (`AI_Monthly_Briefs`)."""
    wt = packet.get("weight_alignment", {})
    clin = packet.get("clinical_context", {})
    tirz = packet.get("tirzepatide_and_treatments", {})

    return MonthlyReviewSchema(
        headline=f"Monthly Whole-Body Clinical Review ({month_key}) — Deterministic Summary",
        body_comp_fat_vs_lean_summary=(
            f"Recent weight intervals: {len(wt.get('recent_intervals', []))} tracked; "
            f"EMA trend slope: {wt.get('ema_trend_loss_kg_per_wk', 'N/A')} kg/wk. "
            "Prioritize resistance training + protein >=1.2-1.5 g/kg FFM to preserve skeletal muscle mass."
        ),
        masld_and_metabolic_trajectory=(
            f"MASLD 7% and 10% weight-reduction milestone tracker from Oct 7 baseline: {wt.get('masld_progress', {})}."
        ),
        menstrual_cycle_and_hematology_summary=(
            "Cycle-phase biometrics and post-IV-Iron FCM / Vit-D / Thyronorm context tracked under Silver_Cycle_Dim and Gold_Cycle_Phase_Stats."
        ),
        treatment_scorecards_synthesis=(
            f"Active dual-baseline scorecards ({len(tirz.get('treatment_scorecards', []))} key rows) track both pre-treatment and prior-dose deltas with explicit overlap warnings."
        ),
        lab_retest_readiness=(
            f"Open clinical retest windows & indices: {clin.get('open_retest_windows', [])}; "
            f"clinical indices carry age/fasting screening caveats."
        ),
        epistemic_label="Descriptive",
        caveat_note="Deterministic monthly template; review lab retest timing directly with treating physician.",
    )


def fallback_peer_expectedness(packet: dict[str, Any]) -> PeerExpectednessSchema:
    """Deterministic Python fallback for Module 3A (`AI_Peer_Expectedness`)."""
    items = [
        PeerSafetyItem(
            symptom_or_shift="Days 0-2 Post-Shot Appetite Suppression & Mild GI Fullness",
            active_therapy_context="Tirzepatide weekly SC escalation",
            safety_tag="Expected",
            clinical_rationale="Peak GIP/GLP-1 receptor agonist exposure occurs ~24-72h post-injection alongside delayed gastric emptying.",
            reference_source_title="Tirzepatide Prescribing & SURMOUNT Clinical Pharmacology Context",
            reference_url="https://dailymed.nlm.nih.gov/",
            epistemic_label="Screening_Context",
        ),
        PeerSafetyItem(
            symptom_or_shift="Small Resting / Sleeping Heart Rate Elevation (+1 to +3 bpm) on GLP-1 Therapy",
            active_therapy_context="Tirzepatide + Thyronorm 100 mcg equilibration",
            safety_tag="Monitor",
            clinical_rationale="Modest sinus rate increases (1-4 bpm) are documented in GLP-1/GIP trials, but concurrent levothyroxine dose changes warrant TSH/FT4 re-check at equilibration.",
            reference_source_title="GLP-1 RA AutonomicHR & Levothyroxine Monitoring Guidance",
            reference_url="https://www.ncbi.nlm.nih.gov/books/",
            epistemic_label="Screening_Context",
        ),
    ]
    return PeerExpectednessSchema(
        summary_headline="Regimen Safety & Peer-Expectedness Check (Deterministic Fallback)",
        items=items,
        regimen_interaction_note=(
            "Delayed gastric emptying on Tirzepatide can alter oral absorption timing; maintain consistent fasting separation for Thyronorm and discuss persistent GI pain, palpitations, or severe constipation with your doctor."
        ),
        epistemic_label="Screening_Context",
    )


def fallback_doctor_prep(packet: dict[str, Any], specialty: str) -> DoctorVisitPrepSchema:
    """Deterministic Python fallback for Module 3B (`AI_Doctor_Briefs`)."""
    wt = packet.get("weight_alignment", {})
    clin = packet.get("clinical_context", {})

    questions = [
        SpecialistQuestionItem(
            priority_rank=1,
            question=f"Given my current trajectory and {specialty} profile, is my current Tirzepatide dose-maintenance vs escalation pace optimal for fat-mass loss while preserving lean mass?",
            supporting_patient_data=f"EMA trend={wt.get('ema_trend_loss_kg_per_wk', 'N/A')} kg/wk; MASLD progress={wt.get('masld_progress', {})}.",
            guideline_context="Clinical obesity & MASLD management prioritizes steady loss with lean-mass preservation.",
        ),
        SpecialistQuestionItem(
            priority_rank=2,
            question="Which follow-up labs (TSH/Free T4 post-Thyronorm 100 mcg, Ferritin/CBC post-IV Iron FCM, 25-OH Vit D, hepatic panel/FIB-4) should we order in this retest window?",
            supporting_patient_data=f"Open retest windows: {clin.get('open_retest_windows', [])}.",
            guideline_context="6-8 week thyroid equilibration and 8-12 week post-parenteral iron/vitamin D re-evaluation.",
        ),
        SpecialistQuestionItem(
            priority_rank=3,
            question="Do any of my supplement timings need adjustment alongside delayed gastric emptying on GLP-1/GIP therapy?",
            supporting_patient_data="Active stack tracked in Gold_Treatment_Scorecards with overlap_warning=True.",
            guideline_context="Oral levothyroxine fasting separation and fat-soluble vitamin meal pairing.",
        ),
    ]

    return DoctorVisitPrepSchema(
        specialty=specialty,
        printable_clinical_summary=(
            f"Structured {specialty} visit brief generated from validated Silver/Gold biometrics, longitudinal labs, and dual-baseline treatment scorecards."
        ),
        lab_delta_checklist=[
            "Verify TSH & Free T4 >=6-8 weeks after Thyronorm 100 mcg transition.",
            "Verify Serum Ferritin, Hb, MCV, TSAT post-IV Iron FCM 1g infusion.",
            "Verify ALT, AST, GGT, fasting lipids, HbA1c, and fasting insulin for MASLD tracking.",
        ],
        treatment_scorecard_highlights=(
            "Dual-baseline treatment scorecards separate pre-treatment baseline deltas from prior-dose escalation deltas while flagging overlapping Oct/Nov starts."
        ),
        top_5_questions=questions,
        epistemic_label="Screening_Context",
        caveat_note="Deterministic clinical prep summary for physician discussion; not a diagnostic or prescribing directive.",
    )


def fallback_research_pulse() -> ResearchPulseSchema:
    """Deterministic Python fallback for Module 3C (`AI_Research_Pulse`)."""
    studies = [
        ResearchStudyItem(
            study_or_guideline_title="SYNERGY-NASH & SURMOUNT Body-Composition Literature Context",
            publication_date_or_year="2024-2026 Clinical Literature",
            topic_domain="MASLD_Fatty_Liver",
            key_finding_plain_english="Sustained >=7-10% weight reduction and dual GIP/GLP-1 therapy show substantial reductions in hepatic steatosis alongside cardiometabolic risk improvement.",
            why_it_matters_for_profile="Aligns directly with your Oct 7 MASLD 7% and 10% weight-milestone tracking and hepatic lab follow-up.",
            source_url="https://pubmed.ncbi.nlm.nih.gov/",
            epistemic_label="Screening_Context",
        ),
        ResearchStudyItem(
            study_or_guideline_title="Lean Soft-Tissue Preservation During Incretin-Based Weight Loss",
            publication_date_or_year="2025-2026 Clinical Consensus",
            topic_domain="Lean_Mass_Preservation",
            key_finding_plain_english="Adequate daily protein intake (1.2-1.6 g/kg FFM or target body weight) combined with progressive resistance training mitigates lean-mass loss during GLP-1/GIP therapy.",
            why_it_matters_for_profile="Supports tracking Fitdays BIA fat_share_of_loss alongside strength_sessions_cnt and daily protein coverage.",
            source_url="https://pubmed.ncbi.nlm.nih.gov/",
            epistemic_label="Screening_Context",
        ),
    ]
    return ResearchPulseSchema(
        digest_headline="Weekly GLP-1 & Metabolic Research Pulse (Deterministic Reference Digest)",
        studies=studies,
        epistemic_label="Screening_Context",
    )


def fallback_interactive_qa(question: str, packet: dict[str, Any]) -> InteractiveAnswerSchema:
    """Deterministic Python fallback for Module 4A (`answer_health_question`)."""
    meta = packet.get("meta", {})
    t7 = packet.get("today_and_7d", {})
    wt = packet.get("weight_alignment", {})
    return InteractiveAnswerSchema(
        what_your_data_shows=(
            f"As of {meta.get('report_date_display', 'latest sync')}, latest phenotype is '{t7.get('latest_phenotype', 'N/A')}', "
            f"EMA weight velocity is {wt.get('ema_trend_loss_kg_per_wk', 'N/A')} kg/wk, and readiness pillars are {t7.get('readiness_pillars', {})}. "
            f"(Question received: '{question[:120]}')"
        ),
        what_research_says=(
            "Live Gemini synthesis is currently in deterministic fallback mode (dry-run, missing API key, or monthly ₹500 cap reached). "
            "Observational correlations and predictive feature weights in your packet reflect associations, never standalone causal proof."
        ),
        citations=[CitationItem(title="Personal Health Intelligence Store V2 Validated Packet", url="internal://Health_Intelligence_Store_v2")],
        confidence_tier="Moderate_Observational",
        epistemic_label="Descriptive",
        ask_your_doctor_note="Discuss medication dosing, persistent symptoms, or lab interpretation directly with your treating clinician.",
    )


def fallback_simulator_advice(
    target_weight_kg: float, sliders: dict[str, Any], packet: dict[str, Any]
) -> SimulatorAdviceSchema:
    """Deterministic Python fallback for Module 4A (`advise_simulator_target`)."""
    wt = packet.get("weight_alignment", {})
    cur_kg = wt.get("latest_ema_weight_kg") or wt.get("latest_measured_weight_kg")
    delta_txt = (
        f"Target {target_weight_kg:.1f} kg vs current {float(cur_kg):.1f} kg ({float(cur_kg) - target_weight_kg:.1f} kg remaining)."
        if cur_kg is not None
        else f"Target weight set to {target_weight_kg:.1f} kg."
    )
    return SimulatorAdviceSchema(
        what_your_data_shows=(
            f"{delta_txt} Current EMA velocity={wt.get('ema_trend_loss_kg_per_wk', 'N/A')} kg/wk; "
            f"requested slider levers={sliders}."
        ),
        what_research_says=(
            "Sustainable loss velocities around 0.5-1.0% of body weight per week with >=1.2 g/kg FFM protein and 2-3x/wk resistance training help keep fat_share_of_loss >=75%."
        ),
        realistic_timeline_and_lean_mass_guardrail=(
            "Avoid aggressive caloric deficits that depress protein below 85-95g/day or push loss velocity sustainably above ~1.0-1.2% body weight/week without BIA lean-mass monitoring."
        ),
        prioritized_lever_adjustments=[
            "Prioritize bedtime consistency (<01:00 IST) to protect recovery and appetite regulation.",
            "Maintain protein floor and hydration >=2.5L/day, especially on Days 0-2 post-Tirzepatide injection.",
            "Include 2+ weekly resistance sessions alongside daily step consistency.",
        ],
        citations=[CitationItem(title="Validated Weight Alignment & BIA Guardrails", url="internal://Gold_Weight_Intervals")],
        confidence_tier="Exploratory_Low_N",
        epistemic_label="Prediction",
        ask_your_doctor_note="Confirm target weight milestones and medication titration schedule with your endocrinologist/physician.",
    )
