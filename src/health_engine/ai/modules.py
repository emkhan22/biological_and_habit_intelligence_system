"""Unified Registry & Re-exports for V2.8 Non-Smoking AI Synthesis Modules (1B..4A)."""

from health_engine.ai.brief_daily import DAILY_BRIEF_COLUMNS, run_module_2a_daily_brief
from health_engine.ai.brief_monthly import MONTHLY_BRIEF_COLUMNS, run_module_2c_monthly_review
from health_engine.ai.brief_weekly import (
    WEEKLY_BRIEF_COLUMNS,
    compute_thu_wed_windows,
    run_module_2b_weekly_review,
)
from health_engine.ai.clinical_peer_check import (
    PEER_EXPECTEDNESS_COLUMNS,
    run_module_3a_peer_expectedness,
)
from health_engine.ai.doctor_prep import (
    DOCTOR_BRIEF_COLUMNS,
    VALID_SPECIALTIES,
    run_module_3b_doctor_prep,
)
from health_engine.ai.enricher_journal import JOURNAL_AI_COLUMNS, run_module_1b_journal_enricher
from health_engine.ai.enricher_meals import MEAL_AI_COLUMNS, run_module_1c_meal_auditor
from health_engine.ai.interactive_service import advise_simulator_target, answer_health_question
from health_engine.ai.research_pulse import RESEARCH_PULSE_COLUMNS, run_module_3c_research_pulse

__all__ = [
    "DAILY_BRIEF_COLUMNS",
    "DOCTOR_BRIEF_COLUMNS",
    "JOURNAL_AI_COLUMNS",
    "MEAL_AI_COLUMNS",
    "MONTHLY_BRIEF_COLUMNS",
    "PEER_EXPECTEDNESS_COLUMNS",
    "RESEARCH_PULSE_COLUMNS",
    "VALID_SPECIALTIES",
    "WEEKLY_BRIEF_COLUMNS",
    "advise_simulator_target",
    "answer_health_question",
    "compute_thu_wed_windows",
    "run_module_1b_journal_enricher",
    "run_module_1c_meal_auditor",
    "run_module_2a_daily_brief",
    "run_module_2b_weekly_review",
    "run_module_2c_monthly_review",
    "run_module_3a_peer_expectedness",
    "run_module_3b_doctor_prep",
    "run_module_3c_research_pulse",
]
