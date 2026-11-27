/**
 * Config.gs — Server-side configuration stored in Apps Script PropertiesService.
 * Keeps all Spreadsheet IDs and secrets out of GitHub version control.
 */

const ENGINE_CONSTANTS = {
  CALC_VERSION: 'v1.1.0',
  TIMEZONE: 'Asia/Kolkata',
  OCT7_BASELINE_KG: 108.3,
  GLP1_START_DATE: '2026-10-21',
  FIRST_THU_WEEK: '2026-10-22',
  SLEEP_MIN_VALID_MINS: 120,
  TIRZ_KA_PER_HR: 0.055,
  TIRZ_HALF_LIFE_HRS: 120.0
};

/**
 * Run ONCE in Apps Script editor to store your 5 Spreadsheet IDs securely.
 * Example:
 *   setupConfig({
 *     MEDICAL: '...',
 *     GLAPP:   '...',
 *     HC:      '...',
 *     JOURNAL: '...',
 *     STORE:   '...'
 *   });
 */
function setupConfig(idsObj) {
  const required = ['MEDICAL', 'GLAPP', 'HC', 'JOURNAL', 'STORE'];
  if (!idsObj || typeof idsObj !== 'object') {
    throw new Error('setupConfig requires an object with keys: ' + required.join(', '));
  }
  const props = PropertiesService.getScriptProperties();
  const toSave = {};
  required.forEach(function (k) {
    const val = String(idsObj[k] || '').trim();
    if (!val) throw new Error('Missing required Spreadsheet ID for key: ' + k);
    toSave['SHEET_ID_' + k] = val;
  });
  props.setProperties(toSave, false);
  Logger.log('Stored 5 Spreadsheet IDs securely in ScriptProperties.');
  return { status: 'ok', keysStored: required };
}

/**
 * Internal helper: returns validated Spreadsheet IDs + engine constants.
 */
function getConfig_() {
  const props = PropertiesService.getScriptProperties();
  const keys = ['MEDICAL', 'GLAPP', 'HC', 'JOURNAL', 'STORE'];
  const sources = {};
  keys.forEach(function (k) {
    const val = props.getProperty('SHEET_ID_' + k);
    if (!val) {
      throw new Error('Missing SHEET_ID_' + k + ' in ScriptProperties. Run setupConfig() first.');
    }
    sources[k] = val;
  });
  return {
    CALC_VERSION: ENGINE_CONSTANTS.CALC_VERSION,
    TIMEZONE: ENGINE_CONSTANTS.TIMEZONE,
    OCT7_BASELINE_KG: ENGINE_CONSTANTS.OCT7_BASELINE_KG,
    GLP1_START_DATE: ENGINE_CONSTANTS.GLP1_START_DATE,
    FIRST_THU_WEEK: ENGINE_CONSTANTS.FIRST_THU_WEEK,
    SLEEP_MIN_VALID_MINS: ENGINE_CONSTANTS.SLEEP_MIN_VALID_MINS,
    TIRZ_KA_PER_HR: ENGINE_CONSTANTS.TIRZ_KA_PER_HR,
    TIRZ_HALF_LIFE_HRS: ENGINE_CONSTANTS.TIRZ_HALF_LIFE_HRS,
    SOURCES: sources
  };
}
