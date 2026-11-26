/**
 * WebApp.gs — Phase 2 Mobile-First Glapp+ Web App Server (Apps Script HtmlService).
 *
 * Strictly private: deployed as Execute as: Me / Access: Only myself.
 * Reads ONLY from derived Health_Intelligence_Store (zero private smoking access).
 * Reuses Phase 1 helpers from Config.gs, Engine.gs, and DoctorBrief.gs.
 */

function doGet(e) {
  const tpl = HtmlService.createTemplateFromFile('Index');
  return tpl.evaluate()
    .setTitle('Personal Health Intelligence Center')
    .addMetaTag('viewport', 'width=device-width, initial-scale=1, maximum-scale=1, viewport-fit=cover')
    .setXFrameOptionsMode(HtmlService.XFrameOptionsMode.ALLOWALL);
}

/**
 * Modular HTML include helper for Styles.html, Charts.html, and App.html.
 */
function include(filename) {
  return HtmlService.createHtmlOutputFromFile(filename).getContent();
}

/**
 * Returns a single RPC-safe JSON payload of all 10 Health_Intelligence_Store tabs
 * plus per-domain coverage stats and rolling habit benchmarks.
 * Strictly clean-mode: never accesses private smoking logs.
 */
function getDashboardPayload() {
  const cfg = getConfig_();
  const tz = cfg.TIMEZONE;
  const ssStore = SpreadsheetApp.openById(cfg.SOURCES.STORE);

  const daily        = readSheetObjects_(ssStore, 'Daily_Unified');
  const weekly       = readSheetObjects_(ssStore, 'Weekly_Fri_Reports');
  const monthly      = readSheetObjects_(ssStore, 'Monthly_Summaries');
  const labs         = readSheetObjects_(ssStore, 'Lab_Trends');
  const timeline     = readSheetObjects_(ssStore, 'Treatment_Timeline');
  const bioAudit     = readSheetObjects_(ssStore, 'Bioavailability_Audit');
  const menstrual    = readSheetObjects_(ssStore, 'Menstrual_Longitudinal');
  const qaLog        = readSheetObjects_(ssStore, 'QA_Freshness_Log');
  const briefingPkt  = readSheetObjects_(ssStore, 'AI_Briefing_Packet');
  const aiOutputs    = readSheetObjects_(ssStore, 'AI_Outputs');

  const cleanDaily = sanitizeForRpc_(daily, tz);
  const coverage = computeDomainCoverage_(cleanDaily);
  const habitSummary = computeHabitWindowSummary_(cleanDaily);

  return sanitizeForRpc_({
    meta: {
      loadedAt: Utilities.formatDate(new Date(), tz, 'yyyy-MM-dd HH:mm:ss') + ' IST',
      calcVersion: cfg.CALC_VERSION,
      oct7BaselineKg: cfg.OCT7_BASELINE_KG,
      glp1StartDate: cfg.GLP1_START_DATE
    },
    daily: cleanDaily,
    weekly: weekly,
    monthly: monthly,
    labs: labs,
    timeline: timeline,
    bioAudit: bioAudit,
    menstrual: menstrual,
    qaLog: qaLog,
    briefingPacket: briefingPkt,
    aiOutputs: aiOutputs,
    coverage: coverage,
    habitSummary: habitSummary
  }, tz);
}

/**
 * Exposes Phase 1 buildWholeBodyDoctorBrief_() cleanly to the client Reports tab.
 */
function getDoctorBriefPayload() {
  const cfg = getConfig_();
  return sanitizeForRpc_(buildWholeBodyDoctorBrief_(), cfg.TIMEZONE);
}

/**
 * Recursively converts Date instances into ISO strings so google.script.run
 * never silently drops the payload.
 */
function sanitizeForRpc_(val, tz) {
  if (val === null || val === undefined) return '';
  if (val instanceof Date) {
    const hhmm = Utilities.formatDate(val, tz || 'Asia/Kolkata', 'HH:mm:ss');
    return hhmm === '00:00:00'
      ? Utilities.formatDate(val, tz || 'Asia/Kolkata', 'yyyy-MM-dd')
      : Utilities.formatDate(val, tz || 'Asia/Kolkata', 'yyyy-MM-dd HH:mm:ss');
  }
  if (Array.isArray(val)) {
    return val.map(function (item) { return sanitizeForRpc_(item, tz); });
  }
  if (typeof val === 'object') {
    const out = {};
    Object.keys(val).forEach(function (k) {
      out[k] = sanitizeForRpc_(val[k], tz);
    });
    return out;
  }
  return val;
}

/**
 * Computes per-domain data coverage for the 4-part detail screen footers.
 */
function computeDomainCoverage_(dailyRows) {
  const total = dailyRows.length || 1;
  const specs = {
    weight: 'weight_kg',
    bodyComp: 'fat_mass_kg',
    nutrition: 'calories_kcal',
    sleep: 'sleep_hrs',
    vitals: 'hrv_rmssd_ms',
    steps: 'steps',
    symptoms: 'side_effect_count',
    mood: 'glapp_mood',
    meds: 'meds_taken'
  };
  const out = {};
  Object.keys(specs).forEach(function (domain) {
    const col = specs[domain];
    const valid = dailyRows.filter(function (r) {
      return r[col] !== '' && r[col] !== null && r[col] !== undefined;
    });
    out[domain] = {
      validDays: valid.length,
      totalDays: total,
      coveragePct: Math.round((valid.length / total) * 100),
      firstDate: valid.length ? String(valid[0].date || '') : '',
      latestDate: valid.length ? String(valid[valid.length - 1].date || '') : ''
    };
  });
  return out;
}

/**
 * Computes last 7 days vs prior 7 days habit stats + personal Good-Day P90 benchmarks
 * and Sleep -> Workout co-occurrence summary.
 */
function computeHabitWindowSummary_(dailyRows) {
  const n = dailyRows.length;
  const cur7 = dailyRows.slice(Math.max(0, n - 7));
  const prev7 = dailyRows.slice(Math.max(0, n - 14), Math.max(0, n - 7));

  function avgField(arr, f) {
    const vals = arr.map(function (r) { return num_(r[f]); }).filter(function (v) { return v !== null; });
    return avgNums_(vals);
  }
  function sumField(arr, f) {
    return arr.reduce(function (s, r) { return s + (num_(r[f]) || 0); }, 0);
  }

  const allSteps = dailyRows.map(function (r) { return num_(r.steps); }).filter(function (v) { return v !== null; }).sort(function (a, b) { return a - b; });
  const stepsP90 = allSteps.length ? Math.round(allSteps[Math.floor(allSteps.length * 0.9)]) : '';

  const goodSleepWk = [], shortSleepWk = [];
  for (let i = 1; i < dailyRows.length; i++) {
    const prevSl = num_(dailyRows[i - 1].sleep_hrs);
    const nextWk = num_(dailyRows[i].workout_mins) || 0;
    if (prevSl === null) continue;
    if (prevSl >= 7.0) goodSleepWk.push(nextWk);
    else if (prevSl < 6.0) shortSleepWk.push(nextWk);
  }

  return {
    cur7: {
      avgSteps: avgField(cur7, 'steps'),
      avgProtein: avgField(cur7, 'protein_g'),
      avgFiber: avgField(cur7, 'fiber_g'),
      avgCalories: avgField(cur7, 'calories_kcal'),
      avgWater: avgField(cur7, 'water_l'),
      avgSleep: avgField(cur7, 'sleep_hrs'),
      avgHrv: avgField(cur7, 'hrv_rmssd_ms'),
      avgRhr: avgField(cur7, 'resting_hr'),
      avgWellbeing: avgField(cur7, 'wellbeing_score_5pillar'),
      strengthSessions: sumField(cur7, 'strength_sessions'),
      nutritionLoggedDays: cur7.filter(function (r) { return r.calories_kcal !== ''; }).length
    },
    prev7: {
      avgSteps: avgField(prev7, 'steps'),
      avgProtein: avgField(prev7, 'protein_g'),
      avgFiber: avgField(prev7, 'fiber_g'),
      avgSleep: avgField(prev7, 'sleep_hrs'),
      avgWellbeing: avgField(prev7, 'wellbeing_score_5pillar'),
      strengthSessions: sumField(prev7, 'strength_sessions')
    },
    benchmarks: {
      stepsP90: stepsP90,
      goodSleepAvgWorkoutMins: avgNums_(goodSleepWk),
      goodSleepN: goodSleepWk.length,
      shortSleepAvgWorkoutMins: avgNums_(shortSleepWk),
      shortSleepN: shortSleepWk.length
    }
  };
}
