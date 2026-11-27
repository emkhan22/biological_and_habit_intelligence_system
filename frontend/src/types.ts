export type EpistemicLabel =
  | "Measured"
  | "Descriptive"
  | "Correlation"
  | "Prediction"
  | "Causal_Hypothesis"
  | "Screening_Context";

export type NavTabId =
  | "today"
  | "body_fuel"
  | "sleep_energy"
  | "therapies_reports"
  | "ask_research"
  | "cravings";

export interface StatsCoverageMeta {
  raw_n?: number | null;
  valid_n?: number | null;
  effective_n?: number | null;
  missingness_pct?: number | null;
  maturity_level?: string | null;
  stability_flag?: string | null;
  caveat_note?: string | null;
}

export interface HealthDataPayload {
  source: string;
  packet?: Record<string, any>;
  tables: Record<string, Array<Record<string, any>>>;
}

export interface PrivateSmokingPayload {
  source: string;
  unlocked_email?: string;
  tables: {
    Private_Smoking_Descriptive?: Array<Record<string, any>>;
    Private_Smoking_Risk_Drivers?: Array<Record<string, any>>;
    Private_Delay_Outcome_Stats?: Array<Record<string, any>>;
    Private_AI_Coach?: Array<Record<string, any>>;
    Private_Model_Audit?: Array<Record<string, any>>;
  };
}

export interface AskHealthResponse {
  what_your_data_shows: string;
  what_research_says: string;
  citations: Array<{ title: string; url: string }>;
  confidence_tier: string;
  epistemic_label: EpistemicLabel;
  ask_your_doctor_note: string;
  engine_badge?: string;
}

export interface SimulatorAdviceResponse {
  what_your_data_shows: string;
  what_research_says: string;
  realistic_timeline_and_lean_mass_guardrail: string;
  prioritized_lever_adjustments: string[];
  citations: Array<{ title: string; url: string }>;
  confidence_tier: string;
  epistemic_label: EpistemicLabel;
  ask_your_doctor_note: string;
  engine_badge?: string;
}
