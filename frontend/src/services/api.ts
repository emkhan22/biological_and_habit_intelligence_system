import type {
  AskHealthResponse,
  HealthDataPayload,
  PrivateSmokingPayload,
  SimulatorAdviceResponse,
} from "../types";

/**
 * Fetch Clean Mode health data from `/api/health-data` (Health Intelligence Store V2).
 * Falls back cleanly to `/mock_data.json` when running locally or on static hosts.
 * Structurally NEVER calls `/api/private-smoking`.
 */
export async function fetchCleanHealthData(): Promise<HealthDataPayload> {
  try {
    const res = await fetch("/api/health-data", {
      headers: { Accept: "application/json" },
    });
    if (res.ok) {
      const contentType = res.headers.get("content-type") || "";
      if (contentType.includes("application/json")) {
        return (await res.json()) as HealthDataPayload;
      }
    }
  } catch {
    // Fall through to local/static mock bundle
  }

  const mockRes = await fetch("/mock_data.json");
  if (!mockRes.ok) {
    throw new Error("Unable to load Health Intelligence Store V2 data or offline mock_data.json.");
  }
  return (await mockRes.json()) as HealthDataPayload;
}

/** Server-side Gemini 4 Hydrogen Q&A handler (never exposes GEMINI_API_KEY to browser). */
export async function askHealthQuestionApi(
  question: string,
  packet: Record<string, any> | undefined,
  useSearch = true
): Promise<AskHealthResponse> {
  try {
    const res = await fetch("/api/ask-health", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, packet, use_search: useSearch }),
    });
    if (res.ok) {
      return (await res.json()) as AskHealthResponse;
    }
  } catch {
    // Deterministic offline fallback
  }

  const meta = packet?.meta ?? {};
  const t7 = packet?.today_and_7d ?? {};
  const wt = packet?.weight_alignment ?? {};
  return {
    what_your_data_shows: `Offline Preview (${meta.report_date_display ?? "Latest"}): Phenotype '${t7.latest_phenotype ?? "N/A"}', EMA velocity ${wt.ema_trend_loss_kg_per_wk ?? "—"} kg/wk. Question received: "${question}".`,
    what_research_says:
      "Connect Cloudflare Pages Functions with GEMINI_API_KEY for live Gemini 4 Hydrogen search-grounded answers. Observational metrics never prove biological causation.",
    citations: [{ title: "Health Intelligence Store V2 Packet", url: "internal://Health_Intelligence_Store_v2" }],
    confidence_tier: "Moderate_Observational",
    epistemic_label: "Descriptive",
    ask_your_doctor_note: "Confirm medication or clinical lab decisions directly with your treating clinician.",
    engine_badge: "Deterministic_Fallback",
  };
}

/** Server-side Simulator Advisor handler. */
export async function requestSimulatorAdviceApi(
  targetWeightKg: number,
  sliders: Record<string, number | string>,
  packet: Record<string, any> | undefined
): Promise<SimulatorAdviceResponse> {
  try {
    const res = await fetch("/api/simulator-advice", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        target_weight_kg: targetWeightKg,
        sliders_dict: sliders,
        packet,
      }),
    });
    if (res.ok) {
      return (await res.json()) as SimulatorAdviceResponse;
    }
  } catch {
    // Offline fallback
  }

  return {
    what_your_data_shows: `Target ${targetWeightKg.toFixed(1)} kg modeled with protein=${sliders.protein_g}g/d, steps=${sliders.steps}/d, bedtime=${sliders.bedtime_ist} IST.`,
    what_research_says:
      "Targeting 0.5–1.0% body weight loss per week with >=1.2 g/kg FFM protein and 2–3 resistance sessions/week supports >=75% fat-share of loss.",
    realistic_timeline_and_lean_mass_guardrail:
      "Protect >=90g protein daily and avoid pushing sustained deficit past ~750–900 kcal/day without Fitdays BIA lean-mass monitoring.",
    prioritized_lever_adjustments: [
      "Keep bedtime before 01:00 IST to protect clean sleep duration and overnight HRV.",
      "Maintain >=90g protein and >=25g fiber on Days 0–2 post-Tirzepatide shot.",
      "Hit 2–3 weekly resistance sessions alongside >=7,500 daily steps.",
    ],
    citations: [{ title: "Gold Weight Alignment & Calorie Balance Audit", url: "internal://Gold_Weight_Intervals" }],
    confidence_tier: "Exploratory_Low_N",
    epistemic_label: "Prediction",
    ask_your_doctor_note: "Review target weight milestones and Tirzepatide titration pace with your doctor.",
    engine_badge: "Deterministic_Fallback",
  };
}

/**
 * Strictly isolated Private Mode fetch — ONLY invoked after explicit user email + PIN unlock.
 */
export async function unlockPrivateSmokingApi(
  email: string,
  pin: string
): Promise<PrivateSmokingPayload> {
  const res = await fetch("/api/private-smoking", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, pin }),
  });
  if (!res.ok) {
    const errBody = await res.json().catch(() => ({}));
    throw new Error(errBody.error || "Private Mode unlock failed (verify email allowlist and PIN).");
  }
  return (await res.json()) as PrivateSmokingPayload;
}
