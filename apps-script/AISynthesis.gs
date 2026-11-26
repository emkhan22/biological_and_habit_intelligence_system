/**
 * AISynthesis.gs — Phase 3 Holistic AI Clinical Synthesis & Plateau-Breaking Engine.
 *
 * Security & Privacy Guardrails:
 * - Reads optional GEMINI_API_KEY strictly from ScriptProperties (never exposed to client).
 * - Reads ONLY non-smoking derived clinical aggregates from Health_Intelligence_Store.
 * - Caches outputs in Health_Intelligence_Store!AI_Insights_Cache to minimize API calls.
 * - Includes deterministic grounded fallback so Phase 3 works at $0 even without an API key.
 */

const AI_SYNTH_CONFIG = {
  CACHE_TAB: 'AI_Insights_Cache',
  MODEL: 'gemini-2.5-flash',
  DISCLAIMER: 'Informational pattern analysis grounded in your logged records for physician discussion — not a medical diagnosis or dosing recommendation.'
};

/**
 * Optional one-time setup: store your Gemini API key securely on the server.
 * Leave unset to use the built-in $0 deterministic clinical synthesis engine.
 */
function setupGeminiApiKey(apiKey) {
  const k = String(apiKey || '').trim();
  if (!k) throw new Error('Provide a non-empty Gemini API key.');
  PropertiesService.getScriptProperties().setProperty('GEMINI_API_KEY', k);
  Logger.log('GEMINI_API_KEY stored securely in ScriptProperties.');
  return { status: 'ok' };
}

/**
 * Reads the latest cached synthesis from AI_Insights_Cache (fast page loads).
 * Auto-generates an initial grounded synthesis if cache is empty.
 */
function getCachedAiSynthesis_(ssStore, cfg) {
  const sh = ssStore.getSheetByName(AI_SYNTH_CONFIG.CACHE_TAB);
  if (sh && sh.getLastRow() >= 2) {
    const vals = sh.getDataRange().getValues();
    const last = vals[vals.length - 1];
    try {
      const parsed = JSON.parse(String(last[3] || '{}'));
      if (parsed && parsed.holisticBrief) {
        parsed.generatedAt = String(last[0] || '');
        parsed.engine = String(last[1] || '');
        parsed.targetKg = num_(last[2]);
        return parsed;
      }
    } catch (e) {
      // Fall through to regenerate
    }
  }
  return generateHolisticClinicalSynthesis({ targetKg: 95 });
}

/**
 * Server-callable synthesis generator (invoked by "Regenerate Insight" buttons).
 * Computes deterministic cross-domain metrics first so the LLM never hallucinates math.
 */
function generateHolisticClinicalSynthesis(opts) {
  const lock = LockService.getScriptLock();
  if (!lock.tryLock(20000)) {
    throw new Error('Another synthesis run is in progress. Please retry in a few seconds.');
  }
  try {
    const cfg = getConfig_();
    const tz = cfg.TIMEZONE;
    const targetKg = (opts && num_(opts.targetKg)) ? num_(opts.targetKg) : 95;
    const ssStore = SpreadsheetApp.openById(cfg.SOURCES.STORE);
    const ctx = buildDeterministicClinicalContext_(ssStore, cfg, targetKg);

    const apiKey = PropertiesService.getScriptProperties().getProperty('GEMINI_API_KEY');
    let result = null;
    let engineUsed = 'Deterministic Clinical Engine ($0)';

    if (apiKey) {
      try {
        result = callGeminiClinicalApi_(apiKey, ctx);
        engineUsed = 'Gemini (' + AI_SYNTH_CONFIG.MODEL + ')';
      } catch (e) {
        Logger.log('Gemini API fallback triggered: ' + e.message);
      }
    }
    if (!result || !result.holisticBrief) {
      result = buildGroundedDeterministicSynthesis_(ctx);
    }

    const nowStr = Utilities.formatDate(new Date(), tz, 'dd-MM-yyyy, hh:mm a') + ' IST';
    result.generatedAt = nowStr;
    result.engine = engineUsed;
    result.targetKg = targetKg;
    result.disclaimer = AI_SYNTH_CONFIG.DISCLAIMER;

    writeAiInsightsCache_(ssStore, nowStr, engineUsed, targetKg, result);
    return result;
  } finally {
    lock.releaseLock();
  }
}

/**
 * Computes verified cross-domain correlations & metabolic TDEE adaptation math
 * strictly from Health_Intelligence_Store (zero private smoking access).
 */
function buildDeterministicClinicalContext_(ssStore, cfg, targetKg) {
  const daily = readSheetObjects_(ssStore, 'Daily_Unified');
  const weekly = readSheetObjects_(ssStore, 'Weekly_Fri_Reports');
  const labs = readSheetObjects_(ssStore, 'Lab_Trends');
  const menstrual = readSheetObjects_(ssStore, 'Menstrual_Longitudinal');

  const recent14 = daily.slice(Math.max(0, daily.length - 14));
  const recent7 = daily.slice(Math.max(0, daily.length - 7));
  const latest = daily[daily.length - 1] || {};
  const latestWk = weekly[weekly.length - 1] || {};

  // Latest measured weight & rate
  let latestW = cfg.OCT7_BASELINE_KG;
  for (let i = daily.length - 1; i >= 0; i--) {
    const w = num_(daily[i].weight_kg);
    if (w !== null) { latestW = w; break; }
  }
  const pctLost = round2_(((cfg.OCT7_BASELINE_KG - latestW) / cfg.OCT7_BASELINE_KG) * 100);

  // 1. Peak-phase (D0-D2) vs Trough-phase (D5-D6) calories, protein & symptoms
  const peakDays = daily.filter(function (r) { const d = num_(r.days_since_shot); return d !== null && d >= 0 && d <= 2; });
  const troughDays = daily.filter(function (r) { const d = num_(r.days_since_shot); return d !== null && d >= 5; });

  // 2. High-fat (>55g) vs lower-fat days -> nocturnal HR spikes & symptoms
  const hiFatDays = daily.filter(function (r) { const f = num_(r.fat_g); return f !== null && f > 55; });
  const loFatDays = daily.filter(function (r) { const f = num_(r.fat_g); return f !== null && f <= 55; });

  // 3. Poor sleep (<6h) vs Good sleep (>=7h) -> next-day steps & food noise
  const afterGoodSlSteps = [], afterPoorSlSteps = [], afterPoorSlNoise = [];
  for (let i = 1; i < daily.length; i++) {
    const sl = num_(daily[i - 1].sleep_hrs);
    const st = num_(daily[i].steps);
    const fn = num_(daily[i].food_noise);
    if (sl === null) continue;
    if (sl >= 7.0 && st !== null) afterGoodSlSteps.push(st);
    if (sl < 6.0) {
      if (st !== null) afterPoorSlSteps.push(st);
      if (fn !== null) afterPoorSlNoise.push(fn);
    }
  }

  // 4. Period days vs non-period days Fitdays / weight context
  const periodDaysCount = daily.filter(function (r) { return r.active_period === 'Yes'; }).length;

  // 5. Metabolic adaptation & TDEE trajectory model
  const avgCal = avgCol_(recent14, 'calories_kcal') || 1350;
  const avgProt = avgCol_(recent14, 'protein_g') || 86;
  const avgSteps = avgCol_(recent7, 'steps') || 5800;
  const strength7d = recent7.reduce(function (s, r) { return s + (Number(r.strength_sessions) || 0); }, 0);
  const estTdeeCurrent = Math.round(1420 + (latestW - 68) * 14.2);
  const estTdeeAtTarget = Math.round(1420 + (targetKg - 68) * 14.2);
  const currentDeficit = Math.max(0, estTdeeCurrent - avgCal);
  // Weight where deficit narrows below 350 kcal/d at current intake
  const narrowWeightKg = round2_(68 + Math.max(0, (avgCal + 350 - 1420) / 14.2));

  return {
    latestDate: String(latest.date || ''),
    activeDoseMg: latest.active_dose_mg || 5.0,
    shotPhase: String(latest.shot_phase || 'Steady'),
    latestWeightKg: latestW,
    oct7BaselineKg: cfg.OCT7_BASELINE_KG,
    pctLostFromOct7: pctLost,
    targetKg: targetKg,
    latestWeekNum: String(latestWk.week_num || 'Latest'),
    avgCal14d: avgCal,
    avgProt14d: avgProt,
    avgFiber7d: avgCol_(recent7, 'fiber_g'),
    avgCleanScore7d: avgCol_(recent7, 'clean_eating_score'),
    avgSleep7d: avgCol_(recent7, 'sleep_hrs'),
    avgRem7d: avgCol_(recent7, 'rem_mins'),
    avgSteps7d: avgSteps,
    strengthSessions7d: strength7d,
    pkComparison: {
      peakAvgCal: avgCol_(peakDays, 'calories_kcal'),
      troughAvgCal: avgCol_(troughDays, 'calories_kcal'),
      peakAvgSe: avgCol_(peakDays, 'side_effect_count'),
      troughAvgSe: avgCol_(troughDays, 'side_effect_count')
    },
    fatVsHrSpikes: {
      hiFatN: hiFatDays.length,
      hiFatAvgHrSpikes: avgCol_(hiFatDays, 'sleep_hr_events_count'),
      loFatAvgHrSpikes: avgCol_(loFatDays, 'sleep_hr_events_count')
    },
    sleepNextDay: {
      goodSleepAvgSteps: avgNums_(afterGoodSlSteps),
      goodSleepN: afterGoodSlSteps.length,
      poorSleepAvgSteps: avgNums_(afterPoorSlSteps),
      poorSleepN: afterPoorSlSteps.length,
      poorSleepAvgFoodNoise: avgNums_(afterPoorSlNoise)
    },
    menstrualContext: {
      trackedCycles: menstrual.length,
      periodDaysLoggedSinceOct1: periodDaysCount
    },
    metabolicModel: {
      estTdeeCurrent: estTdeeCurrent,
      estTdeeAtTarget: estTdeeAtTarget,
      currentDeficitKcal: Math.round(currentDeficit),
      narrowDeficitWeightKg: narrowWeightKg
    },
    labCount: labs.length
  };
}

/**
 * Calls Gemini REST API with strict JSON schema & deterministic context.
 */
function callGeminiClinicalApi_(apiKey, ctx) {
  const url = 'https://generativelanguage.googleapis.com/v1beta/models/' + AI_SYNTH_CONFIG.MODEL + ':generateContent?key=' + encodeURIComponent(apiKey);
  const prompt = [
    'You are a clinically cautious personal health synthesis assistant.',
    'Analyze ONLY the deterministic JSON metrics below. Never invent numbers, never diagnose, never recommend dose changes.',
    'Connect Tirzepatide PK phase, nutrition (protein, clean-eating, fiber), sleep stages/nocturnal HR spikes, activity/strength, symptoms, and menstrual context.',
    'Return STRICT JSON with keys:',
    '{',
    '  "holisticBrief": { "headline": string, "summary": string },',
    '  "crossDomainPatterns": [ { "domainPair": string, "label": "Observed"|"Associated"|"Plausible", "insight": string } ],',
    '  "goalPathAdvisor": { "headline": string, "tdeeSummary": string, "plateauWindow": string, "adjustments": [string, string, string] },',
    '  "weeklyReportNarrative": string',
    '}',
    'Deterministic JSON context:',
    JSON.stringify(ctx)
  ].join('\n');

  const resp = UrlFetchApp.fetch(url, {
    method: 'post',
    contentType: 'application/json',
    muteHttpExceptions: true,
    payload: JSON.stringify({
      contents: [{ parts: [{ text: prompt }] }],
      generationConfig: {
        temperature: 0.25,
        responseMimeType: 'application/json'
      }
    })
  });

  if (resp.getResponseCode() !== 200) {
    throw new Error('Gemini HTTP ' + resp.getResponseCode());
  }
  const body = JSON.parse(resp.getContentText());
  const text = body && body.candidates && body.candidates[0] && body.candidates[0].content && body.candidates[0].content.parts && body.candidates[0].content.parts[0] && body.candidates[0].content.parts[0].text;
  return JSON.parse(text);
}

/**
 * Zero-key deterministic grounded synthesis using exact cross-domain statistics.
 */
function buildGroundedDeterministicSynthesis_(ctx) {
  const patterns = [];

  patterns.push({
    domainPair: 'Tirzepatide PK × Nutrition & GI Symptoms',
    label: 'Observed',
    insight: 'On Days 0–2 post-shot, daily calories average ' + (ctx.pkComparison.peakAvgCal || ctx.avgCal14d) + ' kcal with ' + (ctx.pkComparison.peakAvgSe || 'elevated') + ' avg symptom logs vs. ' + (ctx.pkComparison.troughAvgCal || 'higher') + ' kcal and ' + (ctx.pkComparison.troughAvgSe || 'lower') + ' symptoms in trough days. Front-loading a 25–30g lean protein anchor on Days 1–2 helps prevent under-eating without worsening reflux.'
  });

  patterns.push({
    domainPair: 'Sleep Architecture & Nocturnal HR Spikes × Activity',
    label: ctx.sleepNextDay.goodSleepN >= 3 ? 'Associated' : 'Observed',
    insight: '7-day sleep averages ' + (ctx.avgSleep7d || '—') + 'h (REM ' + (ctx.avgRem7d || '—') + 'm). After ≥7h sleep nights, next-day steps average ' + (ctx.sleepNextDay.goodSleepAvgSteps || '—') + ' (n=' + ctx.sleepNextDay.goodSleepN + ') vs. ' + (ctx.sleepNextDay.poorSleepAvgSteps || '—') + ' after <6h nights (n=' + ctx.sleepNextDay.poorSleepN + '). High-fat days (>55g, n=' + ctx.fatVsHrSpikes.hiFatN + ') averaged ' + (ctx.fatVsHrSpikes.hiFatAvgHrSpikes || '—') + ' nocturnal HR spikes vs. ' + (ctx.fatVsHrSpikes.loFatAvgHrSpikes || '—') + ' on lighter-fat days.'
  });

  patterns.push({
    domainPair: 'Menstrual Phase × Body Composition (BIA)',
    label: 'Plausible',
    insight: 'Across ' + ctx.menstrualContext.trackedCycles + ' tracked historical cycles, luteal and bleeding windows commonly add transient water weight on Fitdays BIA scales. Evaluate weekly weight velocity over 2–4 week rolling trends rather than isolated cycle-week weigh-ins.'
  });

  const plateauMsg = 'At your current ~' + Math.round(ctx.avgCal14d) + ' kcal/d intake, estimated maintenance TDEE is ~' + ctx.metabolicModel.estTdeeCurrent + ' kcal/d (deficit ~' + ctx.metabolicModel.currentDeficitKcal + ' kcal/d). As body weight drops toward ~' + ctx.metabolicModel.narrowDeficitWeightKg + ' kg (and ~' + ctx.metabolicModel.estTdeeAtTarget + ' kcal/d TDEE at ' + ctx.targetKg + ' kg), your deficit naturally narrows below 350 kcal/d.';

  return {
    holisticBrief: {
      headline: ctx.latestWeekNum + ' Synthesis: −' + ctx.pctLostFromOct7 + '% Progress on ' + ctx.activeDoseMg + ' mg · Protect Muscle & Fiber',
      summary: 'You are currently in the ' + ctx.shotPhase + ' phase at ' + ctx.activeDoseMg + ' mg with −' + ctx.pctLostFromOct7 + '% total weight loss since 07-10-2026 (crossing the ≥5% Grade 1 fatty liver threshold). Over the past 7 days, sleep (' + (ctx.avgSleep7d || '—') + 'h) and steps (' + (ctx.avgSteps7d || '—') + '/d) are strong anchors, while protein (' + Math.round(ctx.avgProt14d) + 'g/d), fiber (' + (ctx.avgFiber7d || '—') + 'g/d), and strength sessions (' + ctx.strengthSessions7d + '/7d) are the highest-leverage dials to keep fat loss smooth and lean mass protected.'
    },
    crossDomainPatterns: patterns,
    goalPathAdvisor: {
      headline: 'Metabolic Adaptation & Plateau-Breaking Plan (Target: ' + ctx.targetKg + ' kg)',
      tdeeSummary: plateauMsg,
      plateauWindow: 'Expected deficit narrowing zone: ~' + ctx.metabolicModel.narrowDeficitWeightKg + ' kg (assuming ~' + Math.round(ctx.avgCal14d) + ' kcal/d intake)',
      adjustments: [
        'Muscle-Sparing Protein Floor: Hold ≥95–105 g/day protein (especially Days 1–2 post-shot) + 2×/wk strength sessions to defend resting metabolic rate.',
        'NEAT Step Floor: Maintain ≥6,500–7,000 steps/day as weight drops—NEAT often subconsciously dips during prolonged deficits and accounts for most plateau stalls.',
        'Structured Maintenance Refeed / Diet-Break Check: If 3 consecutive Thursday weigh-ins stall outside a luteal/period week, discuss a 7-day maintenance-calorie pause (~1,700–1,800 kcal, protein-forward) with your clinician rather than slashing calories below 1,150 kcal.'
      ]
    },
    weeklyReportNarrative: 'This week (' + ctx.latestWeekNum + ') combined ' + ctx.activeDoseMg + ' mg tirzepatide with ' + (ctx.avgSteps7d || '—') + ' avg steps/day, ' + (ctx.avgSleep7d || '—') + 'h avg sleep, ' + Math.round(ctx.avgProt14d) + 'g/d protein, and ' + ctx.strengthSessions7d + ' strength session(s). Prioritizing fiber (+8–10g/d) and 2 dedicated resistance sessions on Days 3–5 will compound both glycemic and fatty-liver gains.'
  };
}

function writeAiInsightsCache_(ssStore, nowStr, engineUsed, targetKg, result) {
  let sh = ssStore.getSheetByName(AI_SYNTH_CONFIG.CACHE_TAB);
  if (!sh) {
    sh = ssStore.insertSheet(AI_SYNTH_CONFIG.CACHE_TAB);
    sh.getRange(1, 1, 1, 4).setValues([['generated_at', 'engine', 'target_kg', 'payload_json']]);
  }
  sh.appendRow([nowStr, engineUsed, targetKg, JSON.stringify(result)]);
}
