import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import App from "../App";
import type { HealthDataPayload } from "../types";

function makeBasePayload(withOffGlp1 = false): HealthDataPayload {
  const habitRows: Array<Record<string, any>> = [
    {
      row_type: "On_GLP1_Peak_vs_Trough",
      period_id: "Peak_Suppression_Days_0_2",
      protein_g_mean: 94.0,
      clean_eating_score: 82.0,
      steps_mean: 8400,
      hunger_mean: 1.2,
      food_noise_mean: 0.9,
      valid_n: 12,
      effective_n: 10,
      missingness_pct: 5.0,
      maturity_level: "Level 0 (Baseline)",
    },
    {
      row_type: "On_GLP1_Peak_vs_Trough",
      period_id: "Late_Cycle_Trough_Days_4_6",
      protein_g_mean: 88.0,
      clean_eating_score: 76.5,
      steps_mean: 7900,
      hunger_mean: 2.8,
      food_noise_mean: 2.4,
      valid_n: 12,
      effective_n: 10,
      missingness_pct: 5.0,
      maturity_level: "Level 0 (Baseline)",
    },
  ];

  if (withOffGlp1) {
    habitRows.push({
      row_type: "Post_GLP1_Week",
      period_id: "Post_GLP1_Week_1",
      post_glp1_week_num: 1,
      glp1_regime: "Off_GLP1_Transition",
      transition_trigger_reason: "Missed_Or_Delayed_Shot_Gt_10d",
      protein_g_mean: 84.0,
      delta_protein_vs_on_glp1: -7.0,
      clean_eating_score: 74.0,
      steps_mean: 8100,
      hunger_mean: 3.6,
      delta_hunger_vs_on_glp1: 1.4,
      valid_n: 7,
    });
  }

  return {
    source: "Health_Intelligence_Store_v2_Test",
    packet: {
      meta: {
        report_date_iso: "2026-11-26",
        report_date_display: "26-11-2026",
        sync_timestamp_ist: "27-11-2026 21:00:00 IST",
        total_spine_days: 540,
        coverage_7d: { sleep_present_days: 6, vitals_present_days: 6, nutrition_present_days: 6 },
      },
      today_and_7d: {
        latest_phenotype: "Earlier-Bedtime Recovery",
        readiness_pillars: {
          sleep_circadian_score: 92,
          autonomic_recovery_score: 78,
          activity_movement_score: 88,
          nutrition_metabolic_score: 84,
          regimen_consistency_score: 100,
        },
        latest_z_scores: { hrv_robust_z: 0.45, sleep_dur_robust_z: 0.3 },
        latest_biometrics: { hrv_rmssd_ms: 44.2, asleep_min: 435, sleep_hr_avg: 68.4 },
        latest_sleep_hr_spikes_per_hr: 0.65,
      },
      tirzepatide_and_treatments: { active_tirz_dose_mg: 5.0, shot_day_int: 1, tirz_pk_est: 5.8 },
      weight_alignment: {
        latest_measured_weight_kg: 102.7,
        latest_ema_weight_kg: 103.5,
        ema_trend_loss_kg_per_wk: 0.72,
        masld_progress: [{ pct_weight_lost_from_oct7: 5.17, kg_remaining_to_target: 1.98 }],
        recent_intervals: [{ fat_share_of_loss: 0.75, maturity_level: "Level 0 (Descriptive Ledger)" }],
      },
      clinical_context: {
        clinical_indices: [{ index_name: "HOMA-IR", index_value: 11.43, display_date: "20-09-2026", fasting_verified: true }],
        open_retest_windows: [{ treatment_id: "IV_Iron_FCM_1g", days_on_treatment: 49, clinical_window_status: "Retest_Window_Open" }],
      },
    },
    tables: {
      Gold_Daily_Unified: [
        { date: "2026-11-26", display_date: "26-11-2026", is_period_day: false, asleep_min: 435, deep_min: 80, rem_min: 95, light_min: 260 },
      ],
      Gold_Weight_Trend: [{ date: "2026-11-26", display_date: "26-11-2026", ema_weight_kg: 103.5, measured_weight_kg: 102.7 }],
      Gold_Weight_Intervals: [
        {
          interval_id: "WINT-001",
          display_start_date: "20-11-2026",
          display_end_date: "26-11-2026",
          delta_kg: -0.4,
          loss_rate_kg_per_wk: 0.47,
          fat_share_of_loss: 0.75,
          end_weighin_status: "Fasted",
          is_fasted_interval: true,
          nutrition_coverage_pct: 0.85,
          maturity_level: "Level 0 (Descriptive Ledger)",
        },
      ],
      Gold_Habit_Durability: habitRows,
      Gold_Sweet_Hunger_Analysis: [
        {
          window_alignment: "Next_Day_Post_Sweet",
          outcome_metric: "next_day_hunger",
          sweet_days_n: 4,
          non_sweet_logged_days_n: 16,
          nutrition_logged_days: 20,
          nutrition_coverage_pct: 0.85,
          sweet_window_mean: 2.6,
          non_sweet_window_mean: 1.9,
          reverse_causality_risk: true,
          pct_sweet_preceded_by_hunger_or_sleep_debt: 50.0,
          raw_n: 24,
          valid_n: 20,
          effective_n: 16,
          missingness_pct: 16.7,
          maturity_level: "Level 1 (Exploratory)",
          stability_flag: "Exploratory_Signal",
          epistemic_label: "Correlation",
          caveat_note: "Evaluated strictly on logged nutrition days.",
        },
      ],
      Gold_Calorie_Balance_Audit: [
        {
          window_id: "AUD-14D-20261126",
          window_days: 14,
          bmr_kcal_mean: 1493,
          tdee_est_kcal_mean: 2180,
          logged_kcal_mean: 1460,
          nutrition_coverage_pct: 0.86,
          expected_delta_kg: -1.31,
          actual_delta_kg: -1.25,
          discrepancy_kcal_per_day: 33,
          audit_classification: "Within_Expected_Band",
          maturity_level: "Level 1 (Early Signal)",
        },
      ],
      AI_Insights_Cache: [
        {
          report_date: "2026-11-26",
          headline: "Earlier-Bedtime Recovery & Steady EMA Velocity",
          overview_one_liner: "26-11-2026: Earlier-Bedtime Recovery | Tirzepatide 5.0 mg Day 1.",
          epistemic_label: "Descriptive",
          top_3_patterns_json: "[]",
          action_nudges_json: "[]",
        },
      ],
      AI_Cost_Ledger: [{ estimated_inr: 279.26 }],
    },
  };
}

describe("Health Intelligence V3.0 PWA", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("enforces Clean Mode isolation: zero Cravings tab and zero /api/private-smoking calls", () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch");
    render(<App initialData={makeBasePayload(false)} />);

    expect(screen.getAllByText("Today").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Body & Fuel").length).toBeGreaterThan(0);
    expect(screen.queryByText("Cravings")).not.toBeInTheDocument();
    expect(screen.queryByTestId("private-cravings-view")).not.toBeInTheDocument();
    expect(screen.queryByTestId("off-glp1-week-chip")).not.toBeInTheDocument();

    // Zero calls to /api/private-smoking in Clean Mode
    const privCalls = fetchSpy.mock.calls.filter((c) => String(c[0]).includes("private-smoking"));
    expect(privCalls.length).toBe(0);
  });

  it("toggles collapsible Stats & Coverage drawer showing raw_n, valid_n, effective_n, and caveats", () => {
    render(<App initialData={makeBasePayload(false)} />);
    const drawerBtns = screen.getAllByRole("button", { name: /Stats & Coverage/i });
    expect(drawerBtns.length).toBeGreaterThan(0);

    fireEvent.click(drawerBtns[0]);
    const body = screen.getByTestId("stats-coverage-drawer-body");
    expect(body).toHaveTextContent("raw_n");
    expect(body).toHaveTextContent("valid_n");
    expect(body).toHaveTextContent("effective_n");
    expect(body).toHaveTextContent("missingness_pct");
    expect(body).toHaveTextContent("540");
  });

  it("activates Off-GLP-1 Week chip and expanded Off-GLP-1 Durability panel when Off_GLP1_Transition is present", () => {
    render(<App initialData={makeBasePayload(true)} />);
    expect(screen.getByTestId("off-glp1-week-chip")).toHaveTextContent("Off-GLP-1 Week 1");

    // Navigate to Body & Fuel and confirm Sweet->Hunger reverse causality, Calorie Audit, and expanded Off-GLP-1 table
    fireEvent.click(screen.getAllByText("Body & Fuel")[0]);
    expect(screen.getByTestId("off-glp1-panel-body")).toHaveTextContent("Post_GLP1_Week_1");
    expect(screen.getByTestId("sweet-hunger-section")).toHaveTextContent("Reverse-Causality Caution");
    expect(screen.getByTestId("calorie-audit-section")).toHaveTextContent("Within Expected Band");
  });

  it("unlocks Private Mode Cravings view ONLY after PIN submission to /api/private-smoking", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(
      new Response(
        JSON.stringify({
          source: "Smoking_Private_Store (Test)",
          tables: {
            Private_AI_Coach: [
              {
                trigger_pattern_synthesis: "Evening stress cues.",
                friction_tactics_json: JSON.stringify([
                  { tactic_title: "10-Min Water Delay", when_to_deploy: "Evening", concrete_protocol: "Drink water." },
                ]),
                epistemic_label: "Descriptive",
              },
            ],
            Private_Delay_Outcome_Stats: [{ delay_bucket: "10-20m", events_n: 5, resist_rate: 0.8 }],
            Private_Smoking_Risk_Drivers: [],
          },
        }),
        { status: 200, headers: { "Content-Type": "application/json" } }
      )
    );

    render(<App initialData={makeBasePayload(false)} />);
    fireEvent.click(screen.getByRole("button", { name: /Private Mode/i }));
    fireEvent.change(screen.getByLabelText(/Allowlisted Email/i), { target: { value: "me@example.com" } });
    fireEvent.change(screen.getByLabelText(/Secondary PIN/i), { target: { value: "2468" } });
    fireEvent.click(screen.getByRole("button", { name: /Unlock Private View/i }));

    await waitFor(() => {
      expect(screen.getByTestId("private-cravings-view")).toBeInTheDocument();
    });
    expect(fetchSpy).toHaveBeenCalledWith(
      "/api/private-smoking",
      expect.objectContaining({ method: "POST" })
    );
    expect(screen.getByText("10-Min Water Delay")).toBeInTheDocument();
  });

  it("shows latest-valid fallback with 'as of' date and Regimen Consistency on Today, plus expandable correlations & grouped ML audit on Ask & Research", () => {
    const payload = makeBasePayload(false);
    // Simulate live Cloudflare mode where today's biometrics are unlogged & packet readiness_pillars is omitted
    payload.packet!.today_and_7d = {
      latest_phenotype: "Late-Night Sedentary Drain",
      latest_biometrics: { date: "2026-11-27", display_date: "27-11-2026", hrv_rmssd_ms: null, asleep_min: null },
    };
    payload.tables.Gold_Daily_Unified = [
      { date: "2026-11-26", display_date: "26-11-2026", asleep_min: 420, hrv_rmssd_ms: 48.5, sleep_hr_avg: 66.2, steps: 8200, protein_g: 95 },
      { date: "2026-11-27", display_date: "27-11-2026", asleep_min: null, hrv_rmssd_ms: null, sleep_hr_avg: null, steps: null, protein_g: null },
    ];
    payload.tables.AI_Insights_Cache = [
      {
        report_date: "2026-11-27",
        headline: "Tirzepatide 5.0 mg Day 1: 100.0 Regimen Consistency",
        overview_one_liner: "On 27-11-2026, Regimen Consistency is 100.0 and today's wearable biometrics are unlogged.",
        epistemic_label: "Descriptive",
        top_3_patterns_json: "[]",
        action_nudges_json: "[]",
      },
    ];
    payload.tables.Gold_Stats_Correlations = [
      {
        pair_id: "CORR-1",
        x_variable: "bedtime_hr_rel_mid",
        y_variable: "asleep_min",
        spearman_rho: -0.45,
        partial_rho: -0.38,
        valid_n: 120,
        effective_n: 95,
        stability_flag: "Stable_Signal",
      },
    ];
    payload.tables.Model_Run_Audit = [
      {
        model_id: "ML_SLEEP_2A_CORE",
        algorithm: "RandomForestRegressor_SHAP",
        valid_n: 171,
        effective_n: 140,
        ml_cv_r2: 0.16,
        stability_flag: "Active_Level_3",
        maturity_level: "Level 3",
      },
      {
        model_id: "ML_WEIGHT_3A_INTERVALS",
        algorithm: "Gated",
        valid_n: 12,
        effective_n: 12,
        stability_flag: "Gated_Insufficient_N",
        maturity_level: "Level 0-1",
      },
    ];

    render(<App initialData={payload} />);
    expect(screen.getByText("100.0")).toBeInTheDocument();
    expect(screen.getAllByText(/as of 26-11-2026/i).length).toBeGreaterThan(0);

    fireEvent.click(screen.getAllByText("Ask & Research")[0]);
    expect(screen.getByText("₹279.26 / ₹500")).toBeInTheDocument();
    expect(screen.getByText(/Enable Google Search Grounding/i)).toBeInTheDocument();
    expect(screen.getByText(/Active Models \(1\)/i)).toBeInTheDocument();
    expect(screen.getByText(/Waiting for More Data \(1\)/i)).toBeInTheDocument();

    // Click correlation row to expand plain-English explanation
    const corrCell = screen.getByText(/Bedtime Timing \(vs\. Midnight\) → Clean Sleep Duration/i);
    fireEvent.click(corrCell);
    expect(screen.getByText(/Correlation does not prove causation/i)).toBeInTheDocument();
  });
});

