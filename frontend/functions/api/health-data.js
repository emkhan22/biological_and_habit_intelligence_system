import { batchReadSheetTabs, getGoogleAccessToken } from "./_googleAuth.js";

/**
 * Clean Mode Health Intelligence Store V2 Reader (`/api/health-data`).
 * Structurally NEVER reads `Smoking_Log` or `Smoking_Private_Store`.
 */
const CLEAN_TABS = [
  "Gold_Daily_Unified",
  "Gold_Personal_Baselines",
  "Gold_Sleep_HR_Spikes",
  "Gold_Clinical_Indices",
  "Gold_Weight_Intervals",
  "Gold_Weight_Trend",
  "Gold_Weight_Clinical_Link",
  "Gold_Treatment_Scorecards",
  "Gold_Stats_Correlations",
  "Gold_Tirz_Folded_Wave",
  "Gold_Cardiac_Efficiency_Buckets",
  "Gold_Habit_Durability",
  "Gold_Sweet_Hunger_Analysis",
  "Gold_Calorie_Balance_Audit",
  "Gold_ML_Model_Summary",
  "Gold_ML_Feature_Drivers",
  "Gold_Daily_Phenotypes",
  "Gold_Phenotype_Profiles",
  "Gold_Journal_Themes_AI",
  "Gold_Meal_Bioavailability_AI",
  "Model_Run_Audit",
  "AI_Insights_Cache",
  "AI_Weekly_Briefs",
  "AI_Monthly_Briefs",
  "AI_Peer_Expectedness",
  "AI_Doctor_Briefs",
  "AI_Research_Pulse",
  "AI_Cost_Ledger",
  "Silver_Lab_Results",
];

export async function onRequestGet({ env }) {
  try {
    const saJson = env.HEALTH_READER_SA_JSON;
    const sheetId = env.SHEET_ID_STORE_V2;

    if (!saJson || !sheetId) {
      return new Response(
        JSON.stringify({
          error: "HEALTH_READER_SA_JSON or SHEET_ID_STORE_V2 not configured; client should fall back to /mock_data.json.",
        }),
        { status: 503, headers: { "Content-Type": "application/json" } }
      );
    }

    const token = await getGoogleAccessToken(saJson);
    const tables = await batchReadSheetTabs(token, sheetId, CLEAN_TABS);

    // Build lightweight packet header for Q&A / TodayView
    const daily = tables.Gold_Daily_Unified || [];
    const latest = daily[daily.length - 1] || {};
    const trend = tables.Gold_Weight_Trend || [];
    const latestTrend = trend[trend.length - 1] || {};
    const pheno = tables.Gold_Daily_Phenotypes || [];
    const latestPheno = pheno[pheno.length - 1] || {};

    const payload = {
      source: "Health_Intelligence_Store_v2 (Live)",
      packet: {
        meta: {
          report_date_iso: latest.date || null,
          report_date_display: latest.display_date || null,
          total_spine_days: daily.length,
        },
        today_and_7d: {
          latest_phenotype: latestPheno.phenotype_label || null,
          latest_biometrics: latest,
        },
        tirzepatide_and_treatments: {
          shot_day_int: latest.shot_day_int ?? null,
          active_tirz_dose_mg: latest.active_tirz_dose_mg ?? null,
          tirz_pk_est: latest.tirz_pk_est ?? null,
        },
        weight_alignment: {
          latest_measured_weight_kg: latest.latest_weight_kg ?? latestTrend.measured_weight_kg ?? null,
          latest_ema_weight_kg: latestTrend.ema_weight_kg ?? null,
          ema_trend_loss_kg_per_wk: latestTrend.trend_loss_kg_per_wk ?? null,
          recent_intervals: (tables.Gold_Weight_Intervals || []).slice(-4),
          masld_progress: (tables.Gold_Weight_Clinical_Link || []).slice(0, 4),
        },
        clinical_context: {
          clinical_indices: (tables.Gold_Clinical_Indices || []).slice(0, 10),
          open_retest_windows: [],
        },
      },
      tables,
    };

    return new Response(JSON.stringify(payload), {
      status: 200,
      headers: {
        "Content-Type": "application/json",
        "Cache-Control": "private, max-age=300",
      },
    });
  } catch (err) {
    return new Response(JSON.stringify({ error: err?.message || "Failed to read Health Intelligence Store V2." }), {
      status: 500,
      headers: { "Content-Type": "application/json" },
    });
  }
}
