/**
 * Human-readable labels, clinical text cleanup, and latest-valid biometric helpers
 * for Health-OS PWA. Preserves epistemic honesty and never zero-fills missing data.
 */

const FEATURE_LABELS: Record<string, string> = {
  bedtime_hr_rel_mid: "Bedtime Timing (vs. Midnight)",
  tirz_pk_clean: "Tirzepatide Level (Modeled PK)",
  tirz_pk_est: "Tirzepatide Level (Modeled PK)",
  prev_day_avg_hr: "Prior-Day Avg Heart Rate",
  prev_day_active_hours: "Prior-Day Active Hours",
  prev_day_steps: "Prior-Day Step Count",
  prev_day_zone_mins: "Prior-Day Active Zone Mins",
  cycle_mid_extended: "Mid-Cycle Phase Window",
  prev_day_workout_mins: "Prior-Day Workout Minutes",
  prev_day_sleep_min: "Prior-Night Sleep Duration",
  prev_day_asleep_min: "Prior-Night Sleep Duration",
  prev_day_hrv: "Prior-Night HRV (RMSSD)",
  prev_day_calories: "Prior-Day Calorie Intake",
  prev_day_protein_g: "Prior-Day Protein Intake",
  prev_day_late_kcal_after_21: "Prior-Night Eating After 21:00",
  late_kcal_after_21: "Late-Night Calories (After 21:00)",
  evening_kcal_after_18: "Evening Calories (After 18:00)",
  last_meal_to_bed_min: "Dinner-to-Bedtime Gap",
  hrv_rmssd_ms: "Overnight HRV (RMSSD)",
  asleep_min: "Clean Sleep Duration",
  asleep_hr: "Clean Sleep Hours",
  deep_min: "Deep Sleep (min)",
  rem_min: "REM Sleep (min)",
  light_min: "Light Sleep (min)",
  sleep_hr_avg: "Overnight Avg HR",
  sleep_eff_pct: "Sleep Efficiency (%)",
  spikes_per_asleep_hr: "Sleep HR Spikes / Hr",
  sleep_fragmentation_per_hr: "Sleep Fragmentation / Hr",
  avg_hr: "Daily Avg Heart Rate",
  daytime_avg_hr: "Daytime Avg Heart Rate",
  resting_hr: "Resting Heart Rate",
  steps: "Daily Steps",
  active_hours: "Active Hours",
  zone_mins: "Active Zone Minutes",
  workout_mins: "Workout Minutes",
  weight_kg: "Weight (kg)",
  calories_kcal: "Calories Logged (kcal)",
  protein_g: "Daily Protein (g)",
  fiber_g: "Daily Fiber (g)",
  sugar_g: "Daily Sugar (g)",
  hydration_liters: "Hydration (L)",
  hunger: "Hunger Rating",
  food_noise: "Food Noise Rating",
  glapp_gi_score: "GI Symptom Score",
  glapp_side_effect_score: "Side-Effect Score",
  is_period_day: "Menstrual Phase Day",
  cycle_day: "Menstrual Cycle Day",
  shot_day_int: "Days Since Tirzepatide Shot",
};

const STATUS_LABELS: Record<string, string> = {
  Fasted_Assumed_Morning: "Morning (Fasted)",
  Fasted_Confirmed: "Morning (Fasted)",
  Fasted_Verified: "Morning (Fasted)",
  Post_Meal: "Post-Meal",
  Unknown_No_Meal_Log: "Unverified (No Food Log)",
  Retest_Window_Open: "Retest Window Open",
  In_Equilibration: "In Equilibration",
  Baseline_Window: "Baseline Window",
  Completed_Course: "Completed Course",
  Within_Expected_Band: "Within Expected Band",
  Potential_Under_Logging: "Potential Under-Logging",
  Potential_Over_Logging: "Potential Over-Logging",
  Unclear_Water_Cycle_Or_Log_Gap: "Unclear (Water / Cycle / Log Gap)",
  Gated_Insufficient_N: "Waiting for More Data",
  Gated_Below_N100: "Waiting for 100+ Days",
  Gated_Below_N30: "Waiting for 30+ Days",
  Active_Level_2: "Active (Linear Baseline)",
  Active_Level_3: "Active (Tree + Linear)",
  Exploratory_Signal: "Exploratory Signal",
  Stable_Signal: "Stable Signal",
  Moderate_Signal: "Moderate Signal",
  Weak_Or_Unstable: "Weak / Unstable",
  Insufficient_N: "More Data Needed",
  Descriptive_Only: "Descriptive Only",
};

const MODEL_LABELS: Record<string, string> = {
  ML_SLEEP_2A_CORE: "Clean Sleep Duration Predictor",
  ML_SLEEP_2B_NUTRITION: "Sleep Duration + Nutrition Predictor",
  ML_HRV_1A_ALL: "Overnight HRV Predictor (All Vitals)",
  ML_HRV_1B_MODIFIABLE: "Modifiable Overnight HRV Predictor",
  ML_WEIGHT_3A_INTERVALS: "Weight-Loss Interval Predictor",
  ML_WEIGHT_3B_DAILY_EMA: "Daily Weight Velocity Predictor",
  ML_NLP_JOURNAL_TAXONOMY: "Journal Theme & Episode Rules",
  ML_PHENOTYPE_KMEANS: "Daily Recovery Phenotype Clustering",
};

export function humanizeFeature(raw: string | null | undefined): string {
  if (!raw) return "—";
  const key = String(raw).trim();
  if (FEATURE_LABELS[key]) return FEATURE_LABELS[key];
  return key
    .replace(/_/g, " ")
    .replace(/\b(hrv|hr|pk|bmr|tdee|ema|gi|rem|ffm|masld|bmi|kcal|mg|mcg|kg)\b/gi, (m) => m.toUpperCase())
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

export function humanizeStatus(raw: string | null | undefined): string {
  if (!raw) return "—";
  const key = String(raw).trim();
  if (STATUS_LABELS[key]) return STATUS_LABELS[key];
  return key.replace(/_/g, " ");
}

export function humanizeModelId(raw: string | null | undefined): string {
  if (!raw) return "—";
  const key = String(raw).trim();
  if (MODEL_LABELS[key]) return MODEL_LABELS[key];
  return key.replace(/^ML_/, "").replace(/_/g, " ");
}

export function humanizeTreatmentId(raw: string | null | undefined): string {
  if (!raw) return "—";
  const s = String(raw).trim();
  const map: Record<string, string> = {
    Tirzepatide_2_5mg: "Tirzepatide 2.5 mg",
    Tirzepatide_5mg: "Tirzepatide 5.0 mg",
    Tirzepatide_5_0mg: "Tirzepatide 5.0 mg",
    Thyronorm_100mcg: "Thyronorm 100 mcg",
    Thyronorm_88mcg: "Thyronorm 88 mcg",
    Vit_D_600K_IM: "Vitamin D3 600K IM",
    IV_Iron_FCM_1g: "IV Iron (FCM 1g)",
    Matilda_Forte: "Matilda Forte",
    Trazer_F_Forte: "Trazer F Forte",
  };
  if (map[s]) return map[s];
  return s.replace(/_/g, " ");
}

/**
 * Translate raw pipeline identifiers and over-precise decimals inside AI prose
 * into clean, clinical-grade text.
 */
export function humanizeClinicalText(raw: string | null | undefined): string {
  if (!raw) return "";
  let out = String(raw);

  // Replace known internal IDs & snake_case tokens
  const replacements: Array<[RegExp, string]> = [
    [/\bWINT-\d{3}(?:\s*(?:to|–|-|\.\.)\s*WINT-\d{3})?\b/g, "recent short weigh-in spans"],
    [/\bAUD-28D-\d+\b/g, "28-day rolling audit"],
    [/\bAUD-14D-\d+\b/g, "14-day rolling audit"],
    [/\bAUD-7D-\d+\b/g, "7-day snapshot"],
    [/\bRolling_28d_Audit\b/g, "28-day calorie audit"],
    [/\bRolling_14d_Audit\b/g, "14-day calorie audit"],
    [/\bRetest_Window_Open\b/g, "Retest Window Open"],
    [/\bIn_Equilibration\b/g, "In Equilibration"],
    [/\bWithin_Expected_Band\b/g, "within the expected range"],
    [/\bUnclear_Water_Cycle_Or_Log_Gap\b/g, "unclear (water/cycle/log gap)"],
    [/\bPotential_Under_Logging\b/g, "potential under-logging"],
    [/\bPotential_Over_Logging\b/g, "potential over-logging"],
    [/\bML_SLEEP_2A_CORE\b/g, "Clean Sleep Duration Model"],
    [/\bML_HRV_1B_MODIFIABLE\b/g, "Modifiable HRV Model"],
    [/\bML_NLP_JOURNAL_TAXONOMY\b/g, "Journal Pattern Rules"],
    [/\bMICRO_JOURNAL_EPISODE_CHAINS\b/g, "Episode Chain Analysis"],
    [/\bRULE_MISSEDWALK_BLOATING\b/g, "Missed-Walk → Evening Bloating pattern"],
    [/\bRandomForestRegressor_SHAP\b/g, "Random Forest ML"],
    [/\bbedtime_hr_rel_mid\b/g, "bedtime timing"],
    [/\btirz_pk_clean\b/g, "modeled Tirzepatide level"],
    [/\bprev_day_avg_hr\b/g, "prior-day avg heart rate"],
    [/\bprev_day_active_hours\b/g, "prior-day active hours"],
    [/\bprev_day_steps\b/g, "prior-day steps"],
    [/\bprev_day_zone_mins\b/g, "prior-day active zone minutes"],
    [/\bprev_day_workout_mins\b/g, "prior-day workout minutes"],
    [/\bcycle_mid_extended\b/g, "mid-cycle phase"],
    [/\bhrv_rmssd_ms\b/g, "overnight HRV (RMSSD)"],
    [/\bregimen_consistency_score\b/g, "Regimen Consistency"],
    [/\bexpected_delta_kg\b/g, "expected weight change"],
    [/\bactual_delta_kg\b/g, "actual weight change"],
    [/\boverlap_warning=true\b/gi, "concurrent treatment overlap noted"],
    [/\boverlap_warning\b/gi, "concurrent treatment overlap"],
    [/\bThyronorm_100mcg\b/g, "Thyronorm 100 mcg"],
    [/\bVit_D_600K_IM\b/g, "Vitamin D3 600K IM"],
    [/\bIV_Iron_FCM_1g\b/g, "IV Iron (FCM 1g)"],
    [/\bUnlogged_Subsequent_Meal\b/g, "no subsequent meal logged yet"],
    [/\bafter_lunch_med_skip_rate_on_shot_days\b/g, "post-lunch supplement skip rate on shot days"],
    [/\bmed_skip_rate_when_meal_skipped\b/g, "supplement skip rate when a meal is skipped"],
    [/\bfeature_weight\b/g, "model weight"],
  ];

  for (const [pattern, replacement] of replacements) {
    out = out.replace(pattern, replacement);
  }

  // Round overly precise 3–4 decimal floats (e.g., 103.8613 -> 103.9, 0.7113 -> 0.71)
  out = out.replace(/(-?\d+)\.(\d{3,})/g, (match, intPart) => {
    const n = Number(match);
    if (!Number.isFinite(n)) return match;
    if (Math.abs(n) >= 100) return n.toFixed(1);
    if (Math.abs(n) >= 10) return n.toFixed(1);
    return n.toFixed(2);
  });

  return out;
}

/**
 * Split long clinical prose into clean, scannable bullet sentences.
 */
export function splitIntoScannablePoints(text: string | null | undefined, maxPoints = 6): string[] {
  const cleaned = humanizeClinicalText(text).trim();
  if (!cleaned) return [];
  const rawParts = cleaned
    .split(/(?:\n+|(?<=[.!?])\s+(?=[A-Z0-9]))/)
    .map((s) => s.replace(/^[-•*]\s*/, "").trim())
    .filter((s) => s.length > 4);
  if (rawParts.length <= maxPoints) return rawParts;
  // Group excess sentences gently so nothing is lost
  const grouped: string[] = [];
  const perGroup = Math.ceil(rawParts.length / maxPoints);
  for (let i = 0; i < rawParts.length; i += perGroup) {
    grouped.push(rawParts.slice(i, i + perGroup).join(" "));
  }
  return grouped;
}

/** Format hour relative to midnight (e.g., -0.5 -> "23:30", 1.25 -> "01:15", 8.5 -> "08:30"). */
export function formatClockHour(hr: number | string | null | undefined): string {
  if (hr === null || hr === undefined || hr === "") return "—";
  const n = Number(hr);
  if (!Number.isFinite(n)) return "—";
  let wrapped = ((n % 24) + 24) % 24;
  const h = Math.floor(wrapped);
  let m = Math.round((wrapped - h) * 60);
  let finalH = h;
  if (m === 60) {
    finalH = (h + 1) % 24;
    m = 0;
  }
  return `${String(finalH).padStart(2, "0")}:${String(m).padStart(2, "0")}`;
}

export interface LatestValidMetric {
  value: number | null;
  date: string | null;
  isFallback: boolean;
}

/**
 * Find the latest finite numeric value for `field` across chronological rows.
 * Marks `isFallback = true` when the latest row itself was null/blank.
 */
export function findLatestValid(
  rows: Array<Record<string, any>>,
  field: string
): LatestValidMetric {
  if (!rows || rows.length === 0) return { value: null, date: null, isFallback: false };
  const latestRow = rows[rows.length - 1];
  const latestDate = String(latestRow?.display_date || latestRow?.date || "");
  for (let i = rows.length - 1; i >= 0; i--) {
    const raw = rows[i]?.[field];
    if (raw !== null && raw !== undefined && raw !== "") {
      const n = Number(raw);
      if (Number.isFinite(n)) {
        const rowDate = String(rows[i]?.display_date || rows[i]?.date || "");
        return {
          value: n,
          date: rowDate || null,
          isFallback: i < rows.length - 1 && rowDate !== latestDate,
        };
      }
    }
  }
  return { value: null, date: null, isFallback: false };
}

/**
 * Compute deterministic 5-pillar readiness scores with latest-valid fallback
 * so live Cloudflare Pages mode never renders blank dashes when today's sync hasn't run.
 */
export function resolveReadinessPillars(
  packetPillars: Record<string, any> | undefined,
  dailyRows: Array<Record<string, any>>,
  latestBriefText?: string
): {
  sleep: LatestValidMetric;
  autonomic: LatestValidMetric;
  activity: LatestValidMetric;
  nutrition: LatestValidMetric;
  regimen: LatestValidMetric;
} {
  const latestRow = dailyRows[dailyRows.length - 1] ?? {};
  const latestDate = String(latestRow.display_date || latestRow.date || "") || null;

  const numOrNull = (v: any): number | null => {
    if (v === null || v === undefined || v === "") return null;
    const n = Number(v);
    return Number.isFinite(n) ? n : null;
  };

  // 1. Sleep & Circadian
  const pktSleep = numOrNull(packetPillars?.sleep_circadian_score);
  let sleep: LatestValidMetric;
  if (pktSleep !== null) {
    sleep = { value: pktSleep, date: latestDate, isFallback: false };
  } else {
    const lv = findLatestValid(dailyRows, "asleep_min");
    const score = lv.value !== null ? Math.min(100, Math.max(0, (lv.value / 420) * 100)) : null;
    sleep = { value: score, date: lv.date, isFallback: lv.isFallback };
  }

  // 2. Autonomic Recovery
  const pktAuto = numOrNull(packetPillars?.autonomic_recovery_score);
  let autonomic: LatestValidMetric;
  if (pktAuto !== null) {
    autonomic = { value: pktAuto, date: latestDate, isFallback: false };
  } else {
    const lv = findLatestValid(dailyRows, "hrv_rmssd_ms");
    const score = lv.value !== null ? Math.min(100, Math.max(0, (lv.value / 55) * 100)) : null;
    autonomic = { value: score, date: lv.date, isFallback: lv.isFallback };
  }

  // 3. Activity & Steps
  const pktAct = numOrNull(packetPillars?.activity_movement_score);
  let activity: LatestValidMetric;
  if (pktAct !== null) {
    activity = { value: pktAct, date: latestDate, isFallback: false };
  } else {
    const lv = findLatestValid(dailyRows, "steps");
    const score = lv.value !== null ? Math.min(100, Math.max(0, (lv.value / 8000) * 100)) : null;
    activity = { value: score, date: lv.date, isFallback: lv.isFallback };
  }

  // 4. Nutrition & Fuel
  const pktNutr = numOrNull(packetPillars?.nutrition_metabolic_score);
  let nutrition: LatestValidMetric;
  if (pktNutr !== null) {
    nutrition = { value: pktNutr, date: latestDate, isFallback: false };
  } else {
    const lv = findLatestValid(dailyRows, "protein_g");
    const score = lv.value !== null ? Math.min(100, Math.max(0, (lv.value / 95) * 100)) : null;
    nutrition = { value: score, date: lv.date, isFallback: lv.isFallback };
  }

  // 5. Regimen Consistency (check packet -> brief text -> meds_taken/skipped in dailyRows)
  const pktReg = numOrNull(packetPillars?.regimen_consistency_score);
  let regimen: LatestValidMetric;
  if (pktReg !== null) {
    regimen = { value: pktReg, date: latestDate, isFallback: false };
  } else {
    // Check if the daily brief explicitly reports Regimen Consistency (e.g. "Regimen Consistency is 100.0")
    const match = latestBriefText?.match(/Regimen Consistency(?:\s+is|\s*[:=])?\s*(\d+(?:\.\d+)?)/i);
    if (match && Number.isFinite(Number(match[1]))) {
      regimen = { value: Number(match[1]), date: latestDate, isFallback: false };
    } else {
      // Fallback: compute from latest row with meds_taken_cnt or meds_skipped_cnt > 0
      let foundVal: number | null = null;
      let foundDate: string | null = null;
      let fallback = false;
      for (let i = dailyRows.length - 1; i >= 0; i--) {
        const taken = Number(dailyRows[i]?.meds_taken_cnt || 0);
        const skipped = Number(dailyRows[i]?.meds_skipped_cnt || 0);
        if (taken + skipped > 0) {
          foundVal = (taken / (taken + skipped)) * 100;
          foundDate = String(dailyRows[i]?.display_date || dailyRows[i]?.date || "") || null;
          fallback = i < dailyRows.length - 1 && foundDate !== latestDate;
          break;
        }
      }
      regimen = { value: foundVal, date: foundDate, isFallback: fallback };
    }
  }

  return { sleep, autonomic, activity, nutrition, regimen };
}
