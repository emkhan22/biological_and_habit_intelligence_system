/**
 * WebApp.gs — V2.2 Mobile-First Glapp+ Web App Server (Apps Script HtmlService).
 *
 * Strictly private: deployed as Execute as: Me / Access: Only myself.
 * Clean-mode payload never accesses private smoking logs.
 * Strips all UTC-midnight 05:30:00 / 00:00:00 timezone artifacts before RPC return.
 */

function doGet(e) {
  const tpl = HtmlService.createTemplateFromFile('Index');
  return tpl.evaluate()
    .setTitle('Personal Health Intelligence Center')
    .addMetaTag('viewport', 'width=device-width, initial-scale=1, maximum-scale=1, viewport-fit=cover')
    .setXFrameOptionsMode(HtmlService.XFrameOptionsMode.ALLOWALL);
}

function include(filename) {
  return HtmlService.createHtmlOutputFromFile(filename).getContent();
}

/**
 * Returns a single RPC-safe JSON payload of all 10 Health_Intelligence_Store tabs,
 * plus shots[] (for smooth hourly Bateman PK curve) and blended granular symptomsEvents[].
 * Strictly clean-mode: never accesses private smoking logs.
 */
function getDashboardPayload() {
  const cfg = getConfig_();
  const tz = cfg.TIMEZONE;
  const ssStore = SpreadsheetApp.openById(cfg.SOURCES.STORE);
  const ssGlapp = SpreadsheetApp.openById(cfg.SOURCES.GLAPP);
  const ssMJ    = SpreadsheetApp.openById(cfg.SOURCES.JOURNAL);

  const daily       = readSheetObjects_(ssStore, 'Daily_Unified');
  const weekly      = readSheetObjects_(ssStore, 'Weekly_Fri_Reports');
  const monthly     = readSheetObjects_(ssStore, 'Monthly_Summaries');
  const labs        = readSheetObjects_(ssStore, 'Lab_Trends');
  const timeline    = readSheetObjects_(ssStore, 'Treatment_Timeline');
  const bioAudit    = readSheetObjects_(ssStore, 'Bioavailability_Audit');
  const menstrual   = readSheetObjects_(ssStore, 'Menstrual_Longitudinal');
  const qaLog       = readSheetObjects_(ssStore, 'QA_Freshness_Log');
  const briefingPkt = readSheetObjects_(ssStore, 'AI_Briefing_Packet');
  const aiOutputs   = readSheetObjects_(ssStore, 'AI_Outputs');

  // Exact shots for smooth hourly Bateman PK curve
  const shots = readSheetObjects_(ssGlapp, 'shots')
    .map(function (r) {
      return {
        date: isoDate_(r.date, tz),
        time: isoTime_(r.time, tz) || '12:00',
        dose_mg: num_(r.dose_mg)
      };
    })
    .filter(function (s) { return s.date && s.dose_mg !== null; })
    .sort(function (a, b) { return a.date.localeCompare(b.date); });

  // Granular blended symptoms (Glapp side_effects + Micro Journal Symptoms_Log)
  const symptomsEvents = [];
  readSheetObjects_(ssGlapp, 'side_effects').forEach(function (r) {
    const d = isoDate_(r.date, tz);
    if (!d || !r.side_effect) return;
    symptomsEvents.push({
      date: d,
      time: '',
      symptom: String(r.side_effect).trim(),
      severity: num_(r.level) !== null ? num_(r.level) : 1,
      source: 'Glapp',
      context: ''
    });
  });
  readSheetObjects_(ssMJ, 'Symptoms_Log').forEach(function (r) {
    const d = isoDate_(r.Date, tz);
    if (!d || !r.Symptom) return;
    const sevRaw = String(r.Severity || '').toLowerCase();
    const sevNum = sevRaw.indexOf('sev') !== -1 ? 3 : (sevRaw.indexOf('mod') !== -1 ? 2 : 1);
    symptomsEvents.push({
      date: d,
      time: isoTime_(r.Onset_Time || r.Time, tz),
      symptom: String(r.Symptom).trim(),
      severity: sevNum,
      source: 'Micro Journal',
      context: String(r.Context_Tag || r.Notes || '')
    });
  });
  symptomsEvents.sort(function (a, b) { return a.date.localeCompare(b.date); });

  const cleanDaily = sanitizeForRpc_(daily, tz);
  const coverage = computeDomainCoverage_(cleanDaily);
  const habitSummary = computeHabitWindowSummary_(cleanDaily);

  return sanitizeForRpc_({
    meta: {
      loadedAt: Utilities.formatDate(new Date(), tz, 'dd-MM-yyyy, hh:mm a') + ' IST',
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
    shots: shots,
    symptomsEvents: symptomsEvents,
    coverage: coverage,
    habitSummary: habitSummary
  }, tz);
}

function getDoctorBriefPayload() {
  const cfg = getConfig_();
  return sanitizeForRpc_(buildWholeBodyDoctorBrief_(), cfg.TIMEZONE);
}

/**
 * Recursively converts Date instances & strips UTC-midnight 05:30:00 / 00:00:00 artifacts.
 */
function sanitizeForRpc_(val, tz) {
  if (val === null || val === undefined) return '';
  if (val instanceof Date) {
    const yr = val.getUTCFullYear();
    // Google Sheets time-only values use 1899-12-30 epoch -> return exact HH:mm
    if (yr <= 1900) {
      return Utilities.formatDate(val, tz || 'Asia/Kolkata', 'HH:mm');
    }
    const hhmm = Utilities.formatDate(val, tz || 'Asia/Kolkata', 'HH:mm:ss');
    if (hhmm === '00:00:00' || hhmm === '05:30:00') {
      return Utilities.formatDate(val, tz || 'Asia/Kolkata', 'yyyy-MM-dd');
    }
    return Utilities.formatDate(val, tz || 'Asia/Kolkata', 'yyyy-MM-dd HH:mm');
  }
  if (typeof val === 'string') {
    return val
      .replace(/\s+(00:00:00|05:30:00)(\s*IST)?$/i, '')
      .replace(/T00:00:00(\.000)?Z$/i, '');
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
      avgCarbs: avgField(cur7, 'carbs_g'),
      avgFat: avgField(cur7, 'fat_g'),
      avgFiber: avgField(cur7, 'fiber_g'),
      avgSugar: avgField(cur7, 'sugar_g'),
      avgCalories: avgField(cur7, 'calories_kcal'),
      avgWater: avgField(cur7, 'water_l'),
      avgSleep: avgField(cur7, 'sleep_hrs'),
      avgRem: avgField(cur7, 'rem_mins'),
      avgHrv: avgField(cur7, 'hrv_rmssd_ms'),
      avgRhr: avgField(cur7, 'resting_hr'),
      avgWellbeing: avgField(cur7, 'wellbeing_score_5pillar'),
      avgCleanEating: avgField(cur7, 'clean_eating_score'),
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
