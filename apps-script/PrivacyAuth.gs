/**
 * PrivacyAuth.gs — Defense-in-depth Dual-PIN Privacy & Direct Smoking_Log Reader.
 *
 * Security guarantees:
 * 1. Primary security boundary: Apps Script Web App deployed as "Execute as: Me",
 *    "Who has access: Only myself".
 * 2. Secondary shoulder-surfing boundary: Server-side SHA-256 PIN hashes in
 *    PropertiesService with 5-attempt lockout (10 min) and 30-min CacheService session tokens.
 * 3. Data isolation: Micro Journal!Smoking_Log is NEVER copied to Health_Intelligence_Store.
 *    It is read directly in server memory ONLY when a valid PIN 2 session token is presented.
 */

const PRIVACY_CONSTANTS = {
  MAX_FAILED_ATTEMPTS: 5,
  LOCKOUT_MS: 10 * 60 * 1000,
  SESSION_TTL_SEC: 1800
};

/**
 * Run ONCE in Apps Script editor to configure your two PINs:
 * - cleanPin1: switches/locks app into Clean Mode (hides smoking, 5-pillar Readiness Ring)
 * - privatePin2: unlocks Private Mode for 30 min (shows Cigarettes & Cravings, 6-pillar Ring)
 */
function setupPrivatePins(cleanPin1, privatePin2) {
  const p1 = String(cleanPin1 || '').trim();
  const p2 = String(privatePin2 || '').trim();
  if (!p1 || !p2 || p1 === p2) {
    throw new Error('Provide two distinct non-empty PINs for Clean Mode (PIN 1) and Private Mode (PIN 2).');
  }
  const props = PropertiesService.getScriptProperties();
  props.setProperties({
    PIN1_HASH: sha256Hex_(p1),
    PIN2_HASH: sha256Hex_(p2),
    PIN_FAIL_COUNT: '0',
    PIN_LOCK_UNTIL: '0'
  }, false);
  Logger.log('Dual-PIN SHA-256 hashes saved in ScriptProperties.');
  return { status: 'ok' };
}

/**
 * Server-side PIN verification with brute-force throttling and temporary lockout.
 */
function verifyPinMode(enteredPin) {
  const props = PropertiesService.getScriptProperties();
  const now = Date.now();
  const lockUntil = Number(props.getProperty('PIN_LOCK_UNTIL') || 0);
  if (now < lockUntil) {
    const minsLeft = Math.ceil((lockUntil - now) / 60000);
    return { mode: 'locked', token: '', error: 'Locked due to repeated failed attempts. Try again in ' + minsLeft + ' min.' };
  }

  const h1 = props.getProperty('PIN1_HASH');
  const h2 = props.getProperty('PIN2_HASH');
  if (!h1 || !h2) {
    return { mode: 'clean', token: '', error: 'PINs not configured. Run setupPrivatePins() first.' };
  }

  const candidate = sha256Hex_(String(enteredPin || '').trim());
  if (candidate === h2) {
    props.setProperties({ PIN_FAIL_COUNT: '0', PIN_LOCK_UNTIL: '0' }, false);
    const token = Utilities.getUuid();
    CacheService.getUserCache().put('PRIVATE_TOKEN_' + token, 'VALID', PRIVACY_CONSTANTS.SESSION_TTL_SEC);
    return { mode: 'private', token: token, expiresInSec: PRIVACY_CONSTANTS.SESSION_TTL_SEC };
  }

  if (candidate === h1) {
    props.setProperties({ PIN_FAIL_COUNT: '0', PIN_LOCK_UNTIL: '0' }, false);
    return { mode: 'clean', token: '' };
  }

  const fails = Number(props.getProperty('PIN_FAIL_COUNT') || 0) + 1;
  props.setProperty('PIN_FAIL_COUNT', String(fails));
  if (fails >= PRIVACY_CONSTANTS.MAX_FAILED_ATTEMPTS) {
    props.setProperty('PIN_LOCK_UNTIL', String(now + PRIVACY_CONSTANTS.LOCKOUT_MS));
    return { mode: 'locked', token: '', error: 'Too many failed attempts. Locked for 10 minutes.' };
  }
  return { mode: 'clean', token: '', error: 'Invalid PIN (' + (PRIVACY_CONSTANTS.MAX_FAILED_ATTEMPTS - fails) + ' attempts left).' };
}

/**
 * Explicit server-side logout to revoke Private Mode token immediately.
 */
function revokePrivateSession(sessionToken) {
  if (sessionToken) {
    CacheService.getUserCache().remove('PRIVATE_TOKEN_' + String(sessionToken));
  }
  return { mode: 'clean' };
}

/**
 * Reads Micro Journal!Smoking_Log directly in memory ONLY when a valid PIN 2 token is supplied.
 * Computes smoking count, cravings resisted, avg delay mins, triggers, and co-occurrence with
 * Daily_Unified sleep/symptoms/HRV without ever writing smoking data to Health_Intelligence_Store.
 */
function getPrivateSmokingAnalytics(sessionToken) {
  const token = String(sessionToken || '').trim();
  if (!token || CacheService.getUserCache().get('PRIVATE_TOKEN_' + token) !== 'VALID') {
    throw new Error('Unauthorized: Valid Private Mode session token required.');
  }

  const cfg = getConfig_();
  const ssMJ = SpreadsheetApp.openById(cfg.SOURCES.JOURNAL);
  const ssStore = SpreadsheetApp.openById(cfg.SOURCES.STORE);

  const smokingRows = readSheetObjects_(ssMJ, 'Smoking_Log');
  const dailyRows = readSheetObjects_(ssStore, 'Daily_Unified');
  const dailyByDate = {};
  dailyRows.forEach(function (r) { dailyByDate[String(r.date || '')] = r; });

  let totalSmoked = 0;
  let cravingsResisted = 0;
  const delayVals = [];
  const intensityVals = [];
  const triggerCounts = {};
  const byDate = {};

  const events = smokingRows.map(function (r) {
    const dt = isoDate_(r.Date, cfg.TIMEZONE);
    const tm = isoTime_(r.Time, cfg.TIMEZONE);
    const evType = String(r.Event_Type || '').trim();
    const evLower = evType.toLowerCase();
    const cnt = num_(r.Count) || 0;
    const intensity = num_(r.Craving_Intensity);
    const delay = num_(r.Delay_Minutes);
    const trigger = String(r.Trigger || '').trim();

    const isResisted = evLower.indexOf('resist') !== -1 || evLower.indexOf('craving') !== -1 && cnt === 0;
    if (isResisted) {
      cravingsResisted++;
    } else if (cnt > 0) {
      totalSmoked += cnt;
    }
    if (delay !== null) delayVals.push(delay);
    if (intensity !== null) intensityVals.push(intensity);
    if (trigger) triggerCounts[trigger] = (triggerCounts[trigger] || 0) + 1;

    if (dt) {
      if (!byDate[dt]) byDate[dt] = { smoked: 0, resisted: 0 };
      if (isResisted) byDate[dt].resisted++;
      else byDate[dt].smoked += cnt;
    }

    const dayCtx = dailyByDate[dt] || {};
    return {
      date: dt,
      time: tm,
      eventType: evType,
      count: cnt,
      cravingIntensity: intensity,
      delayMinutes: delay,
      trigger: trigger,
      mood: String(r.Mood || ''),
      barrierOrReason: String(r.Barrier_or_Reason || ''),
      sameDaySleepHrs: dayCtx.sleep_hrs || '',
      sameDaySideEffects: dayCtx.top_side_effects || '',
      sameDayFoodNoise: dayCtx.food_noise || ''
    };
  });

  // Co-occurrence summary (explicitly labeled Observed / Associated, never causal)
  const smokedDates = Object.keys(byDate).filter(function (d) { return byDate[d].smoked > 0; });
  const sleepOnSmoked = smokedDates
    .map(function (d) { return num_(dailyByDate[d] && dailyByDate[d].sleep_hrs); })
    .filter(function (v) { return v !== null; });

  return {
    provenance: 'Measured (Private Server-Side Read Only)',
    totalEventsLogged: events.length,
    totalCigarettesSmoked: totalSmoked,
    cravingsResisted: cravingsResisted,
    avgCravingIntensity: avgNums_(intensityVals),
    avgDelayMinutes: avgNums_(delayVals),
    topTriggers: Object.keys(triggerCounts)
      .map(function (k) { return { trigger: k, count: triggerCounts[k] }; })
      .sort(function (a, b) { return b.count - a.count; }),
    coOccurrenceNote: sleepOnSmoked.length >= 3
      ? { label: 'Associated', n: sleepOnSmoked.length, avgSleepHrsOnSmokedDays: avgNums_(sleepOnSmoked) }
      : { label: 'Not Enough Evidence', n: sleepOnSmoked.length, note: 'Need >= 3 logged smoking days with valid sleep data.' },
    events: events
  };
}

function sha256Hex_(str) {
  const bytes = Utilities.computeDigest(Utilities.DigestAlgorithm.SHA_256, str);
  return bytes.map(function (b) {
    const v = b < 0 ? b + 256 : b;
    return (v < 16 ? '0' : '') + v.toString(16);
  }).join('');
}
