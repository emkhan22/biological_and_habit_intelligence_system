"""Flat Pydantic v2 Output Schemas for V2.8 Model-Routed AI Synthesis Modules.

Design Rules (Gemini 4 API & AGENTS.md Rule 3):
- Flat schemas ONLY (maximum 1 level of non-recursive list items; zero recursive references).
- No numeric JSON-schema bounds (`ge=`, `le=`) sent to the Gemini API; strict bounds,
  epistemic class membership, and causal-overclaim checks are enforced client-side via
  `@model_validator(mode="after")` inside `Model.model_validate_json()`.
- Every analytical output explicitly carries `epistemic_label` belonging to:
  `Measured` | `Descriptive` | `Correlation` | `Prediction` | `Causal_Hypothesis` | `Screening_Context`.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from health_engine.config_loader import EPISTEMIC_CLASSES

EpistemicLiteral = Literal[
    "Measured",
    "Descriptive",
    "Correlation",
    "Prediction",
    "Causal_Hypothesis",
    "Screening_Context",
]

FORBIDDEN_CAUSAL_PHRASES: tuple[str, ...] = (
    "proves causation",
    "proven to cause",
    "causally proves",
    "definitively causes",
)


def _check_epistemic_and_causality(epistemic_label: str, text_blobs: list[str]) -> None:
    if epistemic_label not in EPISTEMIC_CLASSES:
        raise ValueError(
            f"Invalid epistemic_label '{epistemic_label}'. Expected one of {sorted(EPISTEMIC_CLASSES)}."
        )
    joined = " ".join(t for t in text_blobs if t).lower()
    if epistemic_label in {"Correlation", "Prediction", "Descriptive"}:
        for bad in FORBIDDEN_CAUSAL_PHRASES:
            if bad in joined:
                raise ValueError(
                    f"AGENTS.md Rule 3 violation: phrase '{bad}' forbidden under epistemic_label='{epistemic_label}'."
                )


# ==============================================================================
# Module 1B: Micro Journal Hinglish/English Semantic Enricher
# ==============================================================================
class JournalEnrichmentItem(BaseModel):
    event_id: str = Field(description="Exact Event_ID from input journal/symptom batch.")
    detected_themes_csv: str = Field(
        description="Comma-separated canonical themes (e.g. 'Late_Night_Work, Musculoskeletal_Pain')."
    )
    trigger_context: str = Field(
        description="Concise Hinglish/English trigger context extracted from entry."
    )
    pain_radiation_note: str = Field(
        description="Pain location/radiation detail if musculoskeletal/cramp symptom present, else 'None'."
    )
    sentiment_polarity: Literal["Positive", "Negative", "Mixed", "Neutral"] = Field(
        description="Nuanced sentiment polarity of the entry."
    )
    epistemic_label: EpistemicLiteral = Field(
        description="Epistemic class for semantic tagging (must be 'Descriptive')."
    )

    @model_validator(mode="after")
    def _validate_item(self) -> JournalEnrichmentItem:
        _check_epistemic_and_causality(
            self.epistemic_label, [self.trigger_context, self.pain_radiation_note]
        )
        return self


class JournalEnrichmentBatchSchema(BaseModel):
    items: list[JournalEnrichmentItem]


# ==============================================================================
# Module 1C: Home-Cooked Meal & Bioavailability Auditor
# ==============================================================================
class MealBioavailabilityItem(BaseModel):
    date: str = Field(description="Exact ISO date YYYY-MM-DD audited.")
    fat_cofactor_assessment: str = Field(
        description="Adequacy of dietary fat (>=10g) paired with D-Rise 60K / Omega-3 doses."
    )
    thyronorm_fasting_assessment: str = Field(
        description="Assessment of morning Thyronorm fasting gap before first meal/tea/shake."
    )
    late_spicy_gi_sleep_note: str = Field(
        description="Proximity of late/heavy/spicy Indian home-cooked meals to GI reflux or sleep HR spikes."
    )
    protein_fiber_tweaks: str = Field(
        description="1-2 practical Indian home-cooked menu tweaks for protein/fiber density."
    )
    bioavailability_status: Literal["Optimal", "Needs_Timing_Tweak", "Missing_Fat_Cofactor", "Unlogged_Meals"] = Field(
        description="Overall bioavailability audit tag for the date."
    )
    epistemic_label: EpistemicLiteral = Field(
        description="Epistemic class (e.g. 'Descriptive' or 'Causal_Hypothesis')."
    )

    @model_validator(mode="after")
    def _validate_item(self) -> MealBioavailabilityItem:
        _check_epistemic_and_causality(
            self.epistemic_label,
            [
                self.fat_cofactor_assessment,
                self.thyronorm_fasting_assessment,
                self.late_spicy_gi_sleep_note,
                self.protein_fiber_tweaks,
            ],
        )
        return self


class MealBioavailabilityBatchSchema(BaseModel):
    items: list[MealBioavailabilityItem]


# ==============================================================================
# Module 2A: Daily Holistic Brief & Coaching Nudges
# ==============================================================================
class DailyPatternItem(BaseModel):
    pattern_title: str
    epistemic_label: EpistemicLiteral
    supporting_metric: str
    caveat: str

    @model_validator(mode="after")
    def _validate_pattern(self) -> DailyPatternItem:
        _check_epistemic_and_causality(
            self.epistemic_label,
            [self.pattern_title, self.supporting_metric, self.caveat],
        )
        return self


class ActionNudgeItem(BaseModel):
    title: str
    why: str
    your_data: str
    evidence: str
    epistemic_label: EpistemicLiteral

    @model_validator(mode="after")
    def _validate_nudge(self) -> ActionNudgeItem:
        _check_epistemic_and_causality(
            self.epistemic_label,
            [self.title, self.why, self.your_data, self.evidence],
        )
        return self


class DailyBriefSchema(BaseModel):
    overview_one_liner: str
    headline: str
    inside_your_body_summary: str
    top_3_patterns: list[DailyPatternItem]
    action_nudges: list[ActionNudgeItem]
    epistemic_label: EpistemicLiteral

    @model_validator(mode="after")
    def _validate_brief(self) -> DailyBriefSchema:
        if not self.overview_one_liner.strip() or not self.headline.strip():
            raise ValueError("DailyBriefSchema requires non-empty overview_one_liner and headline.")
        if len(self.top_3_patterns) == 0 or len(self.top_3_patterns) > 5:
            raise ValueError("top_3_patterns must contain 1..5 items.")
        if len(self.action_nudges) == 0 or len(self.action_nudges) > 5:
            raise ValueError("action_nudges must contain 1..5 items.")
        _check_epistemic_and_causality(
            self.epistemic_label,
            [self.overview_one_liner, self.headline, self.inside_your_body_summary],
        )
        return self


# ==============================================================================
# Module 2B: Friday Weekly Deep Review (Thu-Wed)
# ==============================================================================
class WeeklyReviewSchema(BaseModel):
    headline: str
    dose_working: str
    week_influencers: str
    habit_review_improved_vs_slipped: str
    next_week_priorities: list[str]
    clinical_and_bioavailability_notes: str
    epistemic_label: EpistemicLiteral
    caveat_note: str

    @model_validator(mode="after")
    def _validate_weekly(self) -> WeeklyReviewSchema:
        if len(self.next_week_priorities) == 0 or len(self.next_week_priorities) > 5:
            raise ValueError("next_week_priorities must contain 1..5 items.")
        _check_epistemic_and_causality(
            self.epistemic_label,
            [
                self.headline,
                self.dose_working,
                self.week_influencers,
                self.habit_review_improved_vs_slipped,
                self.clinical_and_bioavailability_notes,
            ],
        )
        return self


# ==============================================================================
# Module 2C: Monthly Whole-Body Clinical Review
# ==============================================================================
class MonthlyReviewSchema(BaseModel):
    headline: str
    body_comp_fat_vs_lean_summary: str
    masld_and_metabolic_trajectory: str
    menstrual_cycle_and_hematology_summary: str
    treatment_scorecards_synthesis: str
    lab_retest_readiness: str
    epistemic_label: EpistemicLiteral
    caveat_note: str

    @model_validator(mode="after")
    def _validate_monthly(self) -> MonthlyReviewSchema:
        _check_epistemic_and_causality(
            self.epistemic_label,
            [
                self.headline,
                self.body_comp_fat_vs_lean_summary,
                self.masld_and_metabolic_trajectory,
                self.treatment_scorecards_synthesis,
            ],
        )
        return self


# ==============================================================================
# Module 3A: Weekly Peer-Expectedness & Regimen Safety Check (Search Grounded)
# ==============================================================================
class PeerSafetyItem(BaseModel):
    symptom_or_shift: str
    active_therapy_context: str
    safety_tag: Literal["Expected", "Monitor", "Ask_Doctor"]
    clinical_rationale: str
    reference_source_title: str
    reference_url: str
    epistemic_label: EpistemicLiteral

    @model_validator(mode="after")
    def _validate_peer_item(self) -> PeerSafetyItem:
        _check_epistemic_and_causality(self.epistemic_label, [self.clinical_rationale])
        return self


class PeerExpectednessSchema(BaseModel):
    summary_headline: str
    items: list[PeerSafetyItem]
    regimen_interaction_note: str
    epistemic_label: EpistemicLiteral

    @model_validator(mode="after")
    def _validate_peer(self) -> PeerExpectednessSchema:
        if len(self.items) == 0:
            raise ValueError("PeerExpectednessSchema requires at least 1 item.")
        _check_epistemic_and_causality(
            self.epistemic_label, [self.summary_headline, self.regimen_interaction_note]
        )
        return self


# ==============================================================================
# Module 3B: On-Demand Specialist Doctor Visit Prep (Search Grounded)
# ==============================================================================
class SpecialistQuestionItem(BaseModel):
    priority_rank: int
    question: str
    supporting_patient_data: str
    guideline_context: str

    @model_validator(mode="after")
    def _validate_q(self) -> SpecialistQuestionItem:
        if not (1 <= self.priority_rank <= 10):
            raise ValueError("priority_rank must be between 1 and 10.")
        return self


class DoctorVisitPrepSchema(BaseModel):
    specialty: str
    printable_clinical_summary: str
    lab_delta_checklist: list[str]
    treatment_scorecard_highlights: str
    top_5_questions: list[SpecialistQuestionItem]
    epistemic_label: EpistemicLiteral
    caveat_note: str

    @model_validator(mode="after")
    def _validate_doc(self) -> DoctorVisitPrepSchema:
        if len(self.top_5_questions) == 0 or len(self.top_5_questions) > 6:
            raise ValueError("top_5_questions must contain 1..6 items.")
        _check_epistemic_and_causality(
            self.epistemic_label,
            [self.printable_clinical_summary, self.treatment_scorecard_highlights],
        )
        return self


# ==============================================================================
# Module 3C: Weekly GLP-1 & Metabolic Research Pulse (Search Grounded)
# ==============================================================================
class ResearchStudyItem(BaseModel):
    study_or_guideline_title: str
    publication_date_or_year: str
    topic_domain: Literal[
        "Tirzepatide_GLP1_GIP",
        "MASLD_Fatty_Liver",
        "Lean_Mass_Preservation",
        "Iron_Thyroid_Metabolic_Recovery",
    ]
    key_finding_plain_english: str
    why_it_matters_for_profile: str
    source_url: str
    epistemic_label: EpistemicLiteral

    @model_validator(mode="after")
    def _validate_study(self) -> ResearchStudyItem:
        _check_epistemic_and_causality(
            self.epistemic_label,
            [self.key_finding_plain_english, self.why_it_matters_for_profile],
        )
        return self


class ResearchPulseSchema(BaseModel):
    digest_headline: str
    studies: list[ResearchStudyItem]
    epistemic_label: EpistemicLiteral

    @model_validator(mode="after")
    def _validate_pulse(self) -> ResearchPulseSchema:
        if len(self.studies) == 0 or len(self.studies) > 6:
            raise ValueError("ResearchPulseSchema requires 1..6 studies.")
        _check_epistemic_and_causality(self.epistemic_label, [self.digest_headline])
        return self


# ==============================================================================
# Module 4A: Stateless Interactive Wisdom Q&A & Simulator Advisor
# ==============================================================================
class CitationItem(BaseModel):
    title: str
    url: str


class InteractiveAnswerSchema(BaseModel):
    what_your_data_shows: str
    what_research_says: str
    citations: list[CitationItem]
    confidence_tier: Literal["High_Measured", "Moderate_Observational", "Exploratory_Low_N", "Clinical_Screening_Only"]
    epistemic_label: EpistemicLiteral
    ask_your_doctor_note: str

    @model_validator(mode="after")
    def _validate_qa(self) -> InteractiveAnswerSchema:
        _check_epistemic_and_causality(
            self.epistemic_label,
            [self.what_your_data_shows, self.what_research_says],
        )
        return self


class SimulatorAdviceSchema(BaseModel):
    what_your_data_shows: str
    what_research_says: str
    realistic_timeline_and_lean_mass_guardrail: str
    prioritized_lever_adjustments: list[str]
    citations: list[CitationItem]
    confidence_tier: Literal["High_Measured", "Moderate_Observational", "Exploratory_Low_N", "Clinical_Screening_Only"]
    epistemic_label: EpistemicLiteral
    ask_your_doctor_note: str

    @model_validator(mode="after")
    def _validate_sim(self) -> SimulatorAdviceSchema:
        _check_epistemic_and_causality(
            self.epistemic_label,
            [
                self.what_your_data_shows,
                self.what_research_says,
                self.realistic_timeline_and_lean_mass_guardrail,
            ],
        )
        return self
