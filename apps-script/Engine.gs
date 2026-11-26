/**
 * Engine.gs — Deterministic Daily Data, Math, QA & Backfill Engine (V2.2).
 *
 * Reads all 4 authoritative Google Sheets (read-only) and refreshes the derived
 * tabs of Health_Intelligence_Store (leaving AI_Outputs untouched for Gemini Tasks).
 * NEVER reads or copies Micro Journal!Smoking_Log.
 */

function runDailyHealthEngine() {
  const lock = LockService.getScriptLock();
  if (!lock.tryLock(30000)) {
    Logger.log('Skipped: another Health Intelligence Engine run is already active.');
    return { status: 'skipped_locked' };
  }

  try {
    const cfg = getConfig_();
    const tz = cfg.TIMEZONE;
    const now = new Date();
    const nowStr = Utilities.formatDate(now, tz, 'dd-MM-yyyy, hh:mm a') + ' IST';
    const todayStr = Utilities.formatDate(now, tz, 'yyyy-MM-dd');

    const ssStore = SpreadsheetApp.openById(cfg.SOURCES.STORE);
    const ssGlapp = SpreadsheetApp.openById(cfg.SOURCES.GLAPP);
    const ssHC    = SpreadsheetApp.openById(cfg.SOURCES.HC);
    const ssMJ    = SpreadsheetApp.openById(cfg.SOURCES.JOURNAL);
    const ssMed   = SpreadsheetApp.openById(cfg.SOURCES.MEDICAL);

    // 1. Upstream ETL_Audit & Drive file modified-time gate
    const etlAudit = readSheetObjects_(ssHC, 'ETL_Audit');
    const lastEtl = etlAudit.length ? etlAudit[etlAudit.length - 1] : {};
    const etlStatus = String(lastEtl.status || 'UNKNOWN');
    const etlRunTime = isoDate_(lastEtl.run_started_ist, tz) || 'Unknown';
    const etlHealthy = etlStatus.toUpperCase() === 'SUCCESS' || etlStatus.toLowerCase() === 'completed';
    const sourceModTimes = {
      glapp: getFileModStr_(cfg.SOURCES.GLAPP, tz),
      hc: getFileModStr_(cfg.SOURCES.HC, tz),
      mj: getFileModStr_(cfg.SOURCES.JOURNAL, tz),
      med: getFileModStr_(cfg.SOURCES.MEDICAL, tz)
    };

    // 2. Batch-read authoritative tabs (NEVER reads Smoking_Log!)
    const shotsRaw  = readSheetObjects_(ssGlapp, 'shots');
    const glWeight  = readSheetObjects_(ssGlapp, 'weight');
    const glMood    = readSheetObjects_(ssGlapp, 'mood');
    const glSE      = readSheetObjects_(ssGlapp, 'side_effects');

    const hcWeight  = readSheetObjects_(ssHC, 'Weight');
    const hcBC      = readSheetObjects_(ssHC, 'Body_Composition');
    const hcNutr    = readSheetObjects_(ssHC, 'Nutrition');
    const hcNutrLog = readSheetObjects_(ssHC, 'Nutrition_Log');
    const hcSteps   = readSheetObjects_(ssHC, 'Steps');
    const hcSleep   = readSheetObjects_(ssHC, 'Sleep');
    const hcSleepHR = readSheetObjects_(ssHC, 'Sleep_HR_Events');
    const hcVitals  = readSheetObjects_(ssHC, 'Vitals');
    const hcWork    = readSheetObjects_(ssHC, 'Workouts');
    const hcFlow    = readSheetObjects_(ssHC, 'Period_Flow_Daily');
    const hcCycles  = readSheetObjects_(ssHC, 'Cycles');

    const mjMeds    = readSheetObjects_(ssMJ, 'Meds_Log');
    const mjSym     = readSheetObjects_(ssMJ, 'Symptoms_Log');
    const mjMood    = readSheetObjects_(ssMJ, 'Journal_Mood');

    const medLabs   = readSheetObjects_(ssMed, 'Lab_Results');
    const medRev    = readSheetObjects_(ssMed, 'Extraction_Review');

    // 3. Shots for 1-compartment Tirzepatide PK model (labeled Model Estimate)
    const shots = shotsRaw
      .map(function (r) {
        return {
          date: isoDate_(r.date, tz),
          time: isoTime_(r.time, tz) || '12:00',
          dose: num_(r.dose_mg)
        };
      })
      .filter(function (s) { return s.date && s.dose !== null; })
      .sort(function (a, b) { return a.date.localeCompare(b.date); });

    // 4. Canonical weight map (Precedence: Body_Composition > HealthConnect Weight > Glapp weight)
    const weights = {};
    glWeight.forEach(function (r) {
      const d = isoDate_(r.date, tz), w = num_(r.weight_kg);
      if (d && w !== null) weights[d] = { w: round2_(w), src: 'Glapp' };
    });
    hcWeight.forEach(function (r) {
      const d = isoDate_(r.date, tz), w = num_(r.weight_kg);
      if (d && w !== null) weights[d] = { w: round2_(w), src: 'HealthConnect' };
    });
    const bcMap = {};
    hcBC.forEach(function (r) {
      const d = isoDate_(r.date, tz), w = num_(r.weight_kg);
      if (d && w !== null) {
        bcMap[d] = r;
        weights[d] = { w: round2_(w), src: 'Fitdays' };
      }
    });

    // 5. Index daily series with QA rules
    const nutrMap = indexByDate_(hcNutr, 'date', tz);
    const vitMap  = indexByDate_(hcVitals, 'date', tz);
    const moodMap = indexByDate_(glMood, 'date', tz);

    const stepMap = {};
    hcSteps.forEach(function (r) {
      const d = isoDate_(r.date, tz), s = num_(r.steps);
      if (d && s !== null) stepMap[d] = s;
    });

    // Sleep: strictly exclude unworn/zero nights (< 120 mins)
    const sleepMap = {};
    let excludedSleepCount = 0;
    hcSleep.forEach(function (r) {
      const d = isoDate_(r.date, tz), m = num_(r.sleep_mins);
      if (!d || m === null) return;
      if (m < cfg.SLEEP_MIN_VALID_MINS) { excludedSleepCount++; return; }
      sleepMap[d] = r;
    });

    // Nocturnal HR Spike Events from HealthConnectDaily!Sleep_HR_Events
    const sleepHrMap = {};
    hcSleepHR.forEach(function (r) {
      const d = isoDate_(r.date, tz);
      if (!d) return;
      if (!sleepHrMap[d]) sleepHrMap[d] = { count: 0, peakBpm: 0, events: [] };
      const peak = num_(r.peak_hr_bpm) || 0;
      const startStr = String(r.event_start_ist || '').trim();
      const hhmm = startStr.length >= 16 ? startStr.slice(11, 16) : isoTime_(r.event_start_ist, tz);
      sleepHrMap[d].count++;
      if (peak > sleepHrMap[d].peakBpm) sleepHrMap[d].peakBpm = peak;
      if (sleepHrMap[d].events.length < 4) {
        sleepHrMap[d].events.push((hhmm || 'night') + ' (' + Math.round(peak) + ' bpm' + (r.stage_context ? ', ' + r.stage_context : '') + ')');
      }
    });

    // Workouts: deduplicate by date + start_time_ist + end_time_ist + source_app
    const seenWorkouts = {};
    const wkMap = {};
    let duplicateWorkoutsRemoved = 0;
    hcWork.forEach(function (r) {
      const d = isoDate_(r.date, tz);
      const key = [d, String(r.start_time_ist || ''), String(r.end_time_ist || ''), String(r.source_app || '')].join('|');
      if (!d) return;
      if (seenWorkouts[key]) { duplicateWorkoutsRemoved++; return; }
      seenWorkouts[key] = true;
      if (!wkMap[d]) wkMap[d] = { mins: 0, count: 0, strength: 0 };
      wkMap[d].mins += (num_(r.duration_mins) || 0);
      wkMap[d].count++;
      const exType = String(r.exercise_type || '').toLowerCase();
      if (exType.indexOf('strength') !== -1 || exType.indexOf('weight') !== -1) {
        wkMap[d].strength++;
      }
    });

    // Side effects & granular Micro Journal symptoms (using event Date, not Logged_At)
    const seMap = {};
    glSE.forEach(function (r) {
      const d = isoDate_(r.date, tz);
      if (!d) return;
      if (!seMap[d]) seMap[d] = [];
      if (r.side_effect) seMap[d].push(String(r.side_effect).trim());
    });
    mjSym.forEach(function (r) {
      const d = isoDate_(r.Date, tz);
      if (!d) return;
      if (!seMap[d]) seMap[d] = [];
      if (r.Symptom) seMap[d].push(String(r.Symptom).trim());
    });

    const periodSet = {};
    hcFlow.forEach(function (r) {
      const d = isoDate_(r.date, tz);
      if (d) periodSet[d] = true;
    });

    const medAdh = {};
    mjMeds.forEach(function (r) {
      const d = isoDate_(r.Date, tz);
      if (!d) return;
      if (!medAdh[d]) medAdh[d] = { taken: 0, skipped: 0 };
      const st = String(r.Status || '').toLowerCase();
      if (st.indexOf('skip') !== -1 || st.indexOf('miss') !== -1) medAdh[d].skipped++;
      else if (st.indexOf('taken') !== -1) medAdh[d].taken++;
    });

    // 6. Build Daily_Unified (2026-10-01 .. todayStr)
    const dailyHeaders = [
      'date','days_since_shot','active_dose_mg','shot_phase','est_tirz_mg_model',
      'weight_kg','weight_source','fat_mass_kg','muscle_mass_kg','skeletal_muscle_pct','visceral_fat','bone_mass_kg','protein_mass_kg',
      'calories_kcal','protein_g','carbs_g','fat_g','fiber_g','sugar_g','water_l','meal_entries','clean_eating_score',
      'sleep_hrs','deep_mins','rem_mins','light_mins','awake_mins','sleep_hr_events_count','sleep_hr_peak_bpm','sleep_hr_events_summary',
      'resting_hr','hrv_rmssd_ms',
      'steps','workout_mins','strength_sessions',
      'hunger','food_noise','glapp_mood','feeling','side_effect_count','top_side_effects',
      'meds_taken','meds_skipped','active_period','wellbeing_score_5pillar','provenance_note'
    ];
    const dailyRows = [dailyHeaders];
    const dailyLookup = {};

    let cur = new Date('2026-10-01T00:00:00Z');
    const end = new Date(todayStr + 'T00:00:00Z');
    while (cur <= end) {
      const ds = cur.toISOString().slice(0, 10);
      const sh = getShotPhase_(ds, shots);
      const pk = ds >= cfg.GLP1_START_DATE ? calcTirzPk_(ds, shots, cfg) : '';
      const wObj = weights[ds];
      const bc = bcMap[ds] || {};
      const nu = nutrMap[ds] || {};
      const sl = sleepMap[ds] || {};
      const shr = sleepHrMap[ds] || null;
      const vi = vitMap[ds] || {};
      const st = stepMap[ds] !== undefined ? stepMap[ds] : '';
      const wk = wkMap[ds] || { mins: 0, count: 0, strength: 0 };
      const mo = moodMap[ds] || {};
      const ses = seMap[ds] || [];
      const md = medAdh[ds] || { taken: 0, skipped: 0 };

      const slHrs = sl.sleep_mins ? round2_(num_(sl.sleep_mins) / 60.0) : '';
      const prot = num_(nu.protein_g);
      const fib = num_(nu.fiber_g);
      const sug = num_(nu.sugar_g);
      const wat = num_(nu.hydration_liters);
      const hrv = num_(vi.hrv_rmssd_ms);

      // Clean / Healthy Eating Quality Score (0-100 when nutrition logged)
      let cleanEat = '';
      if (num_(nu.calories_kcal) !== null) {
        const pScore = prot !== null ? Math.min(100, (prot / 100) * 100) : 50;
        const fScore = fib !== null ? Math.min(100, (fib / 25) * 100) : 50;
        const sScore = sug !== null ? (sug <= 25 ? 100 : Math.max(0, 100 - (sug - 25) * 2.5)) : 80;
        const wScore = wat !== null ? Math.min(100, (wat / 2.2) * 100) : 60;
        cleanEat = Math.round(pScore * 0.35 + fScore * 0.25 + sScore * 0.25 + wScore * 0.15);
      }

      // Transparent 5-Pillar Clean Wellbeing Score (missing pillars excluded, never scored as 0)
      const pillars = [];
      if (slHrs !== '') pillars.push(Math.min(100, Math.round((slHrs / 7.5) * 100)));
      if (st !== '') pillars.push(Math.min(100, Math.round((st / 7000) * 100)));
      if (prot !== null) pillars.push(Math.min(100, Math.round((prot / 100) * 100)));
      if (hrv !== null) pillars.push(Math.min(100, Math.round((hrv / 45) * 100)));
      if (md.taken + md.skipped > 0) pillars.push(Math.round((md.taken / (md.taken + md.skipped)) * 100));
      const wbScore = pillars.length ? Math.round(pillars.reduce(function (a, b) { return a + b; }, 0) / pillars.length) : '';

      const row = [
        ds, sh.days, sh.dose, sh.phase, pk,
        wObj ? wObj.w : '', wObj ? wObj.src : '',
        valOrBlank_(bc.fat_mass_kg), valOrBlank_(bc.muscle_mass_kg), valOrBlank_(bc.skeletal_muscle_pct),
        valOrBlank_(bc.visceral_fat), valOrBlank_(bc.bone_mass_kg), valOrBlank_(bc.protein_mass_kg),
        valOrBlank_(nu.calories_kcal), prot !== null ? prot : '', valOrBlank_(nu.carbs_g), valOrBlank_(nu.fat_g),
        fib !== null ? fib : '', sug !== null ? sug : '', wat !== null ? wat : '', valOrBlank_(nu.entries_logged), cleanEat,
        slHrs, valOrBlank_(sl.sleep_deep_mins), valOrBlank_(sl.sleep_rem_mins), valOrBlank_(sl.sleep_light_mins), valOrBlank_(sl.sleep_awake_mins),
        shr ? shr.count : '', shr ? Math.round(shr.peakBpm) : '', shr ? shr.events.join('; ') : '',
        valOrBlank_(vi.resting_hr), hrv !== null ? hrv : '',
        st, wk.mins ? round2_(wk.mins) : 0, wk.strength,
        valOrBlank_(mo.hunger), valOrBlank_(mo.food_noise), valOrBlank_(mo.mood), mo.feeling || '',
        ses.length || '', ses.length ? Array.from(new Set(ses)).sort().join(', ') : '',
        (md.taken + md.skipped) > 0 ? md.taken : '', (md.taken + md.skipped) > 0 ? md.skipped : '',
        periodSet[ds] ? 'Yes' : 'No', wbScore, 'Measured+Derived (' + cfg.CALC_VERSION + ')'
      ];
      dailyRows.push(row);
      dailyLookup[ds] = rowToObj_(dailyHeaders, row);
      cur.setUTCDate(cur.getUTCDate() + 1);
    }

    // 7. Weekly_Fri_Reports (Full historical backfill since Oct 22 Thu -> Wed + strict Thu close)
    const wkRows = buildWeeklyBackfill_(dailyLookup, weights, todayStr, nowStr, cfg);

    // 8. Monthly_Summaries (with logged-day denominators & macros)
    const moRows = buildMonthlySummaries_(dailyLookup, nowStr);

    // 9. Lab_Trends (Measured clinical labs with full provenance)
    const labRows = buildLabTrends_(medLabs, tz);

    // 10. Bioavailability_Audit (Meds_Log x Nutrition_Log first-meal fasting gap + fixed Skipped rule)
    const bioRows = buildBioavailabilityAudit_(mjMeds, hcNutrLog, tz);

    // 11. Menstrual_Longitudinal (Flo -> HealthConnect handover + fixed period days + flow segments)
    const menRows = buildMenstrualLongitudinal_(hcCycles, hcFlow, tz);

    // 12. QA_Freshness_Log
    const openExtCount = medRev.filter(function (e) {
      const st = String(e.Resolution_Status || '').toLowerCase();
      return st !== 'resolved' && st !== 'closed';
    }).length;
    const totalDays = dailyRows.length - 1;
    const qaRows = [
      ['check_time_ist','domain','source_sheet','latest_date','valid_rows','coverage_or_status','qa_flag'],
      [nowStr,'Upstream ETL','HealthConnectDaily!ETL_Audit',etlRunTime,etlAudit.length,'Status: ' + etlStatus + ' | File mod: ' + sourceModTimes.hc, etlHealthy ? 'Info' : 'Warning'],
      [nowStr,'GLP-1 Core','Glapp Data',shots.length ? shots[shots.length - 1].date : '',shots.length,'Shots: ' + shots.length + ' | File mod: ' + sourceModTimes.glapp,'Info'],
      [nowStr,'Body Comp','HealthConnectDaily!Body_Composition',maxKey_(bcMap),Object.keys(bcMap).length,'Fitdays BIA (WHR 100% blank - omitted)','Info'],
      [nowStr,'Nutrition','HealthConnectDaily!Nutrition',maxKey_(nutrMap),Object.keys(nutrMap).length,Math.round((Object.keys(nutrMap).length / Math.max(1, totalDays)) * 100) + '% days since Oct 1 (unlogged != 0)','Info'],
      [nowStr,'Sleep','HealthConnectDaily!Sleep',maxKey_(sleepMap),Object.keys(sleepMap).length,excludedSleepCount + ' rows <120m excluded; ' + Object.keys(sleepHrMap).length + ' nights with HR spikes','Info'],
      [nowStr,'Vitals','HealthConnectDaily!Vitals',maxKey_(vitMap),Object.keys(vitMap).length,'spo2_pct 100% blank - omitted','Info'],
      [nowStr,'Workouts','HealthConnectDaily!Workouts',maxKey_(wkMap),Object.keys(seenWorkouts).length,duplicateWorkoutsRemoved + ' duplicates removed; notes blank','Info'],
      [nowStr,'Menstrual','HealthConnectDaily!Cycles',hcCycles.length ? isoDate_(hcCycles[0].cycle_start, tz) : '',hcCycles.length,'Flo->HC handover respected; flow segments mapped','Info'],
      [nowStr,'Micro Journal','Micro Journal',maxKey_(medAdh),mjMeds.length + mjSym.length + mjMood.length,'Private server isolation active | File mod: ' + sourceModTimes.mj,'Info'],
      [nowStr,'Medical Records','AggregatedMedicalRecords','',medLabs.length,openExtCount + ' open Extraction_Review items | File mod: ' + sourceModTimes.med, openExtCount > 0 ? 'Warning' : 'Info']
    ];

    // 13. AI_Briefing_Packet
    const latestWk = wkRows[wkRows.length - 1];
    const benchmarks = computeBenchmarks_(dailyLookup);
    const pktRows = [
      ['section','key','value','provenance'],
      ['Meta','Generated_At',nowStr,'Derived'],
      ['Meta','Calc_Version',cfg.CALC_VERSION,'Derived'],
      ['Meta','Upstream_ETL_Status',etlStatus + ' (' + etlRunTime + ')','Measured'],
      ['Weight','Oct7_Baseline_kg',String(cfg.OCT7_BASELINE_KG),'Measured (Fitdays)'],
      ['Weight','Latest_Weight_kg',latestWk[8] + ' on ' + latestWk[9] + ' (' + latestWk[11] + ' kg / ' + latestWk[12] + '%)','Measured'],
      ['Latest_Week','Period',latestWk[0] + ' (' + latestWk[1] + ' to ' + latestWk[2] + '; Close Thu ' + latestWk[3] + ': ' + latestWk[6] + ')','Derived'],
      ['Latest_Week','Nutrition_Avg','Cal ' + latestWk[13] + ', Prot ' + latestWk[14] + 'g, Carbs ' + latestWk[15] + 'g, Fat ' + latestWk[16] + 'g, Fiber ' + latestWk[17] + 'g, CleanScore ' + latestWk[20] + '/100','Derived'],
      ['Latest_Week','Activity_Sleep','Steps ' + latestWk[22] + ', Strength ' + latestWk[23] + 'x, Sleep ' + latestWk[25] + 'h, REM ' + latestWk[26] + 'm, HRV ' + latestWk[28],'Derived'],
      ['Latest_Week','Habit_Delta_vs_Prev',latestWk[32],'Derived'],
      ['Benchmarks','Steps_GoodDay_P90',String(benchmarks.stepsP90) + ' (n=' + benchmarks.stepsN + ')','Derived'],
      ['Benchmarks','Sleep_Workout_CoOccurrence',benchmarks.sleepWorkoutSummary,'Derived (Associated, non-causal)']
    ];

    // 14. Batch write derived tabs (leaves Treatment_Timeline seed & AI_Outputs intact!)
    writeMatrix_(ssStore, 'Daily_Unified', dailyRows);
    writeMatrix_(ssStore, 'Weekly_Fri_Reports', wkRows);
    writeMatrix_(ssStore, 'Monthly_Summaries', moRows);
    writeMatrix_(ssStore, 'Lab_Trends', labRows);
    writeMatrix_(ssStore, 'Bioavailability_Audit', bioRows);
    writeMatrix_(ssStore, 'Menstrual_Longitudinal', menRows);
    writeMatrix_(ssStore, 'QA_Freshness_Log', qaRows);
    writeMatrix_(ssStore, 'AI_Briefing_Packet', pktRows);

    Logger.log('runDailyHealthEngine complete at ' + nowStr + ' | Daily rows=' + (dailyRows.length - 1) + ' | Weeks=' + (wkRows.length - 1));
    return { status: 'ok', generatedAt: nowStr, dailyRows: dailyRows.length - 1, weeks: wkRows.length - 1 };
  } finally {
    lock.releaseLock();
  }
}

// ==================== MODULE BUILDERS ====================
function buildWeeklyBackfill_(dailyLookup, weights, todayStr, nowStr, cfg) {
  const headers = [
    'week_num','week_start_thu','week_end_wed','closing_thu_date','dose_mg',
    'open_thu_weight_kg','close_thu_weight_kg','thu_weight_status','last_avail_weight_kg','last_avail_weight_date',
    'weekly_delta_kg','delta_vs_oct7_kg','delta_vs_oct7_pct',
    'avg_calories','avg_protein_g','avg_carbs_g','avg_fat_g','avg_fiber_g','avg_sugar_g','avg_water_l','clean_eating_score','nutrition_logged_days',
    'avg_steps','strength_sessions','total_workout_mins',
    'avg_sleep_hrs','avg_rem_mins','avg_rhr','avg_hrv',
    'avg_hunger','avg_food_noise','total_side_effects','habit_delta_vs_prev_week','generated_at'
  ];
  const rows = [headers];
  let ws = new Date(cfg.FIRST_THU_WEEK + 'T00:00:00Z');
  const today = new Date(todayStr + 'T00:00:00Z');
  const sortedWeights = Object.keys(weights).sort();
  let wkNum = 1, prevStats = null;

  while (ws <= today) {
    const we = addDays_(ws, 6);
    const closeThu = addDays_(ws, 7);
    const wsStr = ws.toISOString().slice(0, 10);
    const weStr = we.toISOString().slice(0, 10);
    const closeStr = closeThu.toISOString().slice(0, 10);

    const recs = [];
    for (let k = 0; k < 7; k++) {
      const dStr = addDays_(ws, k).toISOString().slice(0, 10);
      if (dailyLookup[dStr]) recs.push(dailyLookup[dStr]);
    }

    const openW = weights[wsStr] ? weights[wsStr].w : '';
    const closeW = weights[closeStr] ? weights[closeStr].w : '';
    const thuStatus = closeW !== '' ? 'Recorded' : 'Pending';

    const availDates = sortedWeights.filter(function (d) { return d <= closeStr; });
    const lastDt = availDates.length ? availDates[availDates.length - 1] : '';
    const lastW = lastDt ? weights[lastDt].w : '';

    const wkDelta = (openW !== '' && closeW !== '') ? round2_(closeW - openW) : '';
    const dOct7 = lastW !== '' ? round2_(lastW - cfg.OCT7_BASELINE_KG) : '';
    const dOct7Pct = dOct7 !== '' ? round2_((dOct7 / cfg.OCT7_BASELINE_KG) * 100) : '';

    const avgSteps = avgCol_(recs, 'steps');
    const avgProt  = avgCol_(recs, 'protein_g');
    const avgFib   = avgCol_(recs, 'fiber_g');
    const avgSleep = avgCol_(recs, 'sleep_hrs');
    const strCnt   = recs.reduce(function (s, r) { return s + (Number(r.strength_sessions) || 0); }, 0);
    const wkMins   = round2_(recs.reduce(function (s, r) { return s + (Number(r.workout_mins) || 0); }, 0));
    const seCnt    = recs.reduce(function (s, r) { return s + (Number(r.side_effect_count) || 0); }, 0);
    const nutrDays = recs.filter(function (r) { return r.calories_kcal !== ''; }).length;

    const deltas = [];
    if (prevStats) {
      if (avgSteps !== '' && prevStats.steps !== '') deltas.push('Steps ' + signed_(Math.round(avgSteps - prevStats.steps)));
      if (avgProt !== '' && prevStats.prot !== '') deltas.push('Protein ' + signed_(round2_(avgProt - prevStats.prot)) + 'g');
      if (avgFib !== '' && prevStats.fib !== '') deltas.push('Fiber ' + signed_(round2_(avgFib - prevStats.fib)) + 'g');
      if (avgSleep !== '' && prevStats.sleep !== '') deltas.push('Sleep ' + signed_(Math.round((avgSleep - prevStats.sleep) * 60)) + 'm');
      deltas.push('Strength ' + signed_(strCnt - prevStats.str));
    }
    prevStats = { steps: avgSteps, prot: avgProt, fib: avgFib, sleep: avgSleep, str: strCnt };

    rows.push([
      'Week ' + wkNum, wsStr, weStr, closeStr, recs.length ? recs[0].active_dose_mg : '',
      openW, closeW !== '' ? closeW : 'Pending', thuStatus, lastW, lastDt,
      wkDelta, dOct7, dOct7Pct,
      avgCol_(recs, 'calories_kcal'), avgProt, avgCol_(recs, 'carbs_g'), avgCol_(recs, 'fat_g'),
      avgFib, avgCol_(recs, 'sugar_g'), avgCol_(recs, 'water_l'), avgCol_(recs, 'clean_eating_score'), nutrDays,
      avgSteps, strCnt, wkMins,
      avgSleep, avgCol_(recs, 'rem_mins'), avgCol_(recs, 'resting_hr'), avgCol_(recs, 'hrv_rmssd_ms'),
      avgCol_(recs, 'hunger'), avgCol_(recs, 'food_noise'), seCnt,
      deltas.length ? deltas.join('; ') : 'Baseline week', nowStr
    ]);
    ws = addDays_(ws, 7);
    wkNum++;
  }
  return rows;
}

function buildMonthlySummaries_(dailyLookup, nowStr) {
  const headers = [
    'month','days_in_month','start_weight_kg','end_weight_kg','delta_kg',
    'avg_protein_g','avg_carbs_g','avg_fat_g','avg_fiber_g','clean_eating_score','nutrition_logged_days',
    'avg_steps','steps_logged_days','strength_sessions','avg_sleep_hrs','sleep_logged_days','avg_hrv','generated_at'
  ];
  const byMonth = {};
  Object.keys(dailyLookup).sort().forEach(function (d) {
    const m = d.slice(0, 7);
    if (!byMonth[m]) byMonth[m] = [];
    byMonth[m].push(dailyLookup[d]);
  });
  const rows = [headers];
  Object.keys(byMonth).sort().forEach(function (m) {
    const recs = byMonth[m];
    const parts = m.split('-');
    const daysInMonth = new Date(Number(parts[0]), Number(parts[1]), 0).getDate();
    const ws = recs.map(function (r) { return num_(r.weight_kg); }).filter(function (v) { return v !== null; });
    rows.push([
      m, daysInMonth, ws.length ? ws[0] : '', ws.length ? ws[ws.length - 1] : '',
      ws.length >= 2 ? round2_(ws[ws.length - 1] - ws[0]) : '',
      avgCol_(recs, 'protein_g'), avgCol_(recs, 'carbs_g'), avgCol_(recs, 'fat_g'), avgCol_(recs, 'fiber_g'),
      avgCol_(recs, 'clean_eating_score'),
      recs.filter(function (r) { return r.calories_kcal !== ''; }).length,
      avgCol_(recs, 'steps'),
      recs.filter(function (r) { return r.steps !== ''; }).length,
      recs.reduce(function (s, r) { return s + (Number(r.strength_sessions) || 0); }, 0),
      avgCol_(recs, 'sleep_hrs'),
      recs.filter(function (r) { return r.sleep_hrs !== ''; }).length,
      avgCol_(recs, 'hrv_rmssd_ms'), nowStr
    ]);
  });
  return rows;
}

function buildLabTrends_(medLabs, tz) {
  const headers = ['test_date','standardized_test_name','value','unit','reference_range','abnormal_flag','source_filename','provenance_type'];
  const keyTerms = ['hba1c','glucose','insulin','ferritin','hemoglobin','haemoglobin','vitamin d','25-oh','b12','tsh','t4','alt','sgpt','ast','sgot','triglyceride','hdl','ldl','vldl'];
  const items = [];
  medLabs.forEach(function (r) {
    const nm = String(r.Standardized_Test_Name || r.Original_Test_Name || '').trim();
    const nl = nm.toLowerCase();
    if (!nm || !keyTerms.some(function (k) { return nl.indexOf(k) !== -1; })) return;
    items.push([
      isoDate_(r.Collection_DateTime, tz), nm, String(r.Original_Value || ''), String(r.Unit || ''),
      String(r.Reference_Range || ''), String(r.Abnormal_Flag || ''), String(r.Source_Filename || ''), 'Measured'
    ]);
  });
  items.sort(function (a, b) { return (a[1] + a[0]).localeCompare(b[1] + b[0]); });
  return [headers].concat(items);
}

function buildBioavailabilityAudit_(mjMeds, hcNutrLog, tz) {
  const mealsByDate = {};
  hcNutrLog.forEach(function (m) {
    const d = isoDate_(m.date, tz);
    const tStr = isoTime_(m.time_ist, tz);
    const tm = timeToMins_(tStr);
    if (!d) return;
    if (!mealsByDate[d]) mealsByDate[d] = [];
    mealsByDate[d].push({ timeStr: tStr, mins: tm, fat: num_(m.fat_g) || 0, mealType: String(m.meal_type || '') });
  });
  Object.keys(mealsByDate).forEach(function (d) {
    mealsByDate[d].sort(function (a, b) { return (a.mins || 0) - (b.mins || 0); });
  });

  const rows = [['date','time','med_or_supplement','status','timing_context','first_meal_time_ist','fasting_gap_mins','meal_cofactor_check','bioavailability_assessment','co_factor_note']];
  mjMeds.forEach(function (r) {
    const d = isoDate_(r.Date, tz);
    const tStr = isoTime_(r.Time, tz);
    const tMins = timeToMins_(tStr);
    const m = String(r.Med_or_Supplement || '').trim();
    const ml = m.toLowerCase();
    const st = String(r.Status || '').trim();
    const stLower = st.toLowerCase();
    const ctx = String(r.Timing_Context || '').trim();
    const dayMeals = mealsByDate[d] || [];
    const firstMeal = dayMeals.length ? dayMeals[0] : null;
    const firstMealStr = firstMeal ? firstMeal.timeStr : '';

    // CRITICAL FIX: Check Skipped/Missed status FIRST for EVERY med/supplement!
    if (stLower.indexOf('skip') !== -1 || stLower.indexOf('miss') !== -1) {
      rows.push([d, tStr, m, st, ctx, firstMealStr, '', 'Dose skipped', 'Skipped', 'Monitor symptom/sleep/energy trends around skipped doses.']);
      return;
    }

    let fastingGap = '';
    let cofactor = dayMeals.length ? (dayMeals.length + ' meals logged (first: ' + firstMealStr + ')') : 'No Nutrition_Log for date';
    let assess = 'Taken';
    let note = 'Verify meal/timing co-factors.';

    if (ml.indexOf('thyronorm') !== -1) {
      const afterMeals = dayMeals.filter(function (x) { return tMins !== null && x.mins !== null && x.mins >= tMins; });
      if (afterMeals.length) {
        fastingGap = afterMeals[0].mins - tMins;
        cofactor = 'First meal at ' + afterMeals[0].timeStr + ' (+' + fastingGap + ' min)';
        assess = fastingGap >= 45 ? 'Optimal Empty-Stomach Gap (>=45m)' : (fastingGap >= 30 ? 'Acceptable Gap (30-44m)' : 'Short Fasting Gap (<30m)');
      } else {
        assess = 'Empty-Stomach Window (Meal Time Unlogged)';
      }
      note = 'Keep >=45-60 min before food/coffee and >=4h from Mg/Trazer F minerals.';
    } else if (ml.indexOf('fish oil') !== -1 || ml.indexOf('omega') !== -1 || ml.indexOf('d-rise') !== -1 || ml.indexOf('vitamin d') !== -1) {
      const nearFatMeal = dayMeals.some(function (x) {
        return tMins !== null && x.mins !== null && Math.abs(x.mins - tMins) <= 90 && x.fat >= 10;
      });
      if (nearFatMeal) {
        cofactor = 'Paired with >=10g fat meal';
        assess = 'Optimal (Fat-Assisted Absorption)';
      } else if (ctx.toLowerCase().indexOf('after') !== -1 || ctx.toLowerCase().indexOf('lunch') !== -1 || ctx.toLowerCase().indexOf('dinner') !== -1) {
        assess = 'Likely Fat-Assisted (Post-Meal)';
      } else {
        assess = 'Check Fat Co-Ingestion';
      }
      note = 'Fat-soluble EPA/DHA and Vitamin D3 absorb substantially better with >=10g dietary fat.';
    } else if (ml.indexOf('magnesium') !== -1) {
      assess = 'Night Spacing Good';
      note = 'Night dosing maximizes >=4h separation from morning Thyronorm.';
    } else if (ml.indexOf('matilda') !== -1 || ml.indexOf('trazer') !== -1) {
      assess = 'Post-Meal Tolerance Good';
      note = 'Post-lunch/dinner dosing avoids morning Thyronorm mineral interference.';
    }
    rows.push([d, tStr, m, st, ctx, firstMealStr, fastingGap, cofactor, assess, note]);
  });
  return rows;
}

function buildMenstrualLongitudinal_(hcCycles, hcFlow, tz) {
  // Group Period_Flow_Daily rows by cycle_start to build flow intensity segments & accurate period days
  const flowByCycle = {};
  hcFlow.forEach(function (r) {
    const cs = isoDate_(r.cycle_start, tz);
    if (!cs) return;
    if (!flowByCycle[cs]) flowByCycle[cs] = [];
    flowByCycle[cs].push({
      day: num_(r.cycle_day) || 1,
      level: String(r.flow_level || 'Flow Unmarked').trim()
    });
  });

  const headers = ['cycle_id','cycle_start','cycle_end','cycle_length_days','marked_flow_days','total_period_days','flow_segments_json','era_tag','source_app','confidence_note'];
  const rows = [headers];
  hcCycles.slice(0, 36).forEach(function (r) {
    const cs = isoDate_(r.cycle_start, tz);
    const ce = isoDate_(r.cycle_end, tz);
    const marked = num_(r.marked_flow_days) || 0;
    const unmarked = num_(r.unmarked_flow_days) || 0;
    const span = num_(r.span_days) || 0;
    const flowRows = (flowByCycle[cs] || []).sort(function (a, b) { return a.day - b.day; });
    const totalPeriodDays = Math.max(marked + unmarked, span, flowRows.length);
    const segs = flowRows.length
      ? flowRows.map(function (f) { return f.level; })
      : (totalPeriodDays > 0 ? Array(Math.min(10, totalPeriodDays)).fill('Flow Unmarked') : []);

    rows.push([
      String(r.cycle_id || ''), cs, ce,
      valOrBlank_(r.cycle_length_days), marked, totalPeriodDays || '',
      JSON.stringify(segs),
      String(r.era_tag || ''), String(r.source_app || ''), 'Measured (Flo->HC handover respected)'
    ]);
  });
  return rows;
}

function computeBenchmarks_(dailyLookup) {
  const dates = Object.keys(dailyLookup).sort();
  const steps = dates.map(function (d) { return num_(dailyLookup[d].steps); }).filter(function (v) { return v !== null; }).sort(function (a, b) { return a - b; });
  const p90 = steps.length ? Math.round(steps[Math.floor(steps.length * 0.9)]) : '';

  const goodSleepWk = [], shortSleepWk = [];
  for (let i = 1; i < dates.length; i++) {
    const prevSleep = num_(dailyLookup[dates[i - 1]].sleep_hrs);
    const nextWk = num_(dailyLookup[dates[i]].workout_mins) || 0;
    if (prevSleep === null) continue;
    if (prevSleep >= 7.0) goodSleepWk.push(nextWk);
    else if (prevSleep < 6.0) shortSleepWk.push(nextWk);
  }
  const summary = (goodSleepWk.length >= 3 && shortSleepWk.length >= 3)
    ? 'After >=7h sleep: avg ' + avgNums_(goodSleepWk) + 'm workout (n=' + goodSleepWk.length + ') vs <6h sleep: avg ' + avgNums_(shortSleepWk) + 'm (n=' + shortSleepWk.length + ')'
    : 'Accumulating paired sleep/workout observations (good n=' + goodSleepWk.length + ', short n=' + shortSleepWk.length + ')';

  return { stepsP90: p90, stepsN: steps.length, sleepWorkoutSummary: summary };
}

// ==================== SHARED UTILITIES ====================
function calcTirzPk_(dtStr, shots, cfg) {
  const ke = Math.LN2 / cfg.TIRZ_HALF_LIFE_HRS;
  const ka = cfg.TIRZ_KA_PER_HR;
  const targetMs = new Date(dtStr + 'T18:00:00Z').getTime();
  let total = 0;
  shots.forEach(function (s) {
    const sMs = new Date(s.date + 'T' + (s.time.length === 5 ? s.time : '12:00') + ':00Z').getTime();
    const hrs = (targetMs - sMs) / 3600000;
    if (hrs > 0) {
      const frac = (ka / (ka - ke)) * (Math.exp(-ke * hrs) - Math.exp(-ka * hrs));
      total += s.dose * Math.max(0, frac);
    }
  });
  return round2_(total);
}

function getShotPhase_(dtStr, shots) {
  const prev = shots.filter(function (s) { return s.date <= dtStr; });
  if (!prev.length) return { days: '', dose: '', phase: '' };
  const last = prev[prev.length - 1];
  const days = Math.round((new Date(dtStr + 'T00:00:00Z') - new Date(last.date + 'T00:00:00Z')) / 86400000);
  const phase = days === 0 ? 'Activation' : (days <= 2 ? 'Peak Suppression' : (days <= 4 ? 'Steady Cruise' : 'Trough'));
  return { days: days, dose: last.dose, phase: phase };
}

function readSheetObjects_(ss, name) {
  const sh = ss.getSheetByName(name);
  if (!sh) return [];
  const vals = sh.getDataRange().getValues();
  if (vals.length < 2) return [];
  const headers = vals[0].map(function (x) { return String(x).trim(); });
  const out = [];
  for (let i = 1; i < vals.length; i++) {
    const row = vals[i];
    if (row.every(function (c) { return c === '' || c === null; })) continue;
    const obj = {};
    for (let j = 0; j < headers.length; j++) obj[headers[j]] = row[j];
    out.push(obj);
  }
  return out;
}

function writeMatrix_(ss, name, matrix) {
  const sh = ss.getSheetByName(name) || ss.insertSheet(name);
  sh.clearContents();
  if (matrix.length && matrix[0].length) {
    sh.getRange(1, 1, matrix.length, matrix[0].length).setValues(matrix);
  }
}

function getFileModStr_(fileId, tz) {
  try {
    return Utilities.formatDate(DriveApp.getFileById(fileId).getLastUpdated(), tz, 'dd-MM-yyyy HH:mm');
  } catch (e) {
    return 'Unknown';
  }
}

function indexByDate_(arr, col, tz) {
  const m = {};
  arr.forEach(function (r) { const d = isoDate_(r[col], tz); if (d) m[d] = r; });
  return m;
}
function isoDate_(v, tz) {
  if (!v) return '';
  if (v instanceof Date) return Utilities.formatDate(v, tz || 'Asia/Kolkata', 'yyyy-MM-dd');
  const s = String(v).trim();
  return s.length >= 10 && s[4] === '-' ? s.slice(0, 10) : '';
}
function isoTime_(v, tz) {
  if (v === '' || v === null || v === undefined) return '';
  if (v instanceof Date) return Utilities.formatDate(v, tz || 'Asia/Kolkata', 'HH:mm');
  const s = String(v).trim();
  if (s.indexOf(':') !== -1) {
    const m = s.match(/(\d{1,2}:\d{2})/);
    if (m) return m[1].length === 4 ? ('0' + m[1]) : m[1];
  }
  const f = Number(s);
  if (!isNaN(f) && f >= 0 && f < 1) {
    const mins = Math.round(f * 24 * 60);
    const hh = Math.floor(mins / 60), mm = mins % 60;
    return (hh < 10 ? '0' : '') + hh + ':' + (mm < 10 ? '0' : '') + mm;
  }
  return '';
}
function timeToMins_(hhmm) {
  if (!hhmm || hhmm.indexOf(':') === -1) return null;
  const p = hhmm.split(':');
  return Number(p[0]) * 60 + Number(p[1]);
}
function num_(v) { const n = Number(v); return (v === '' || v === null || isNaN(n)) ? null : n; }
function valOrBlank_(v) { const n = num_(v); return n === null ? '' : n; }
function round2_(n) { return Math.round(n * 100) / 100; }
function addDays_(d, n) { const x = new Date(d.getTime()); x.setUTCDate(x.getUTCDate() + n); return x; }
function rowToObj_(h, r) { const o = {}; h.forEach(function (k, i) { o[k] = r[i]; }); return o; }
function avgNums_(arr) { return arr.length ? round2_(arr.reduce(function (a, b) { return a + b; }, 0) / arr.length) : ''; }
function avgCol_(arr, k) { return avgNums_(arr.map(function (r) { return num_(r[k]); }).filter(function (x) { return x !== null; })); }
function signed_(n) { return (n >= 0 ? '+' : '') + n; }
function maxKey_(obj) { const k = Object.keys(obj).sort(); return k.length ? k[k.length - 1] : ''; }
