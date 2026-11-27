/**
 * Tests.gs — Built-in Phase 1 Automated QA Self-Test Suite.
 * Run runPhase1SelfTest() in the Apps Script editor after setupConfig() and setupPrivatePins().
 */

function runPhase1SelfTest() {
  const results = [];
  function assert(name, cond, detail) {
    results.push({ test: name, status: cond ? 'PASS' : 'FAIL', detail: detail || '' });
  }

  const cfg = getConfig_();

  // 1. Source workbooks reachable
  const ssStore = SpreadsheetApp.openById(cfg.SOURCES.STORE);
  const ssHC = SpreadsheetApp.openById(cfg.SOURCES.HC);
  assert('Sources reachable', !!ssStore && !!ssHC, 'Opened Health_Intelligence_Store & HealthConnectDaily');

  // 2. Sleep < 120 min exclusion check
  const daily = readSheetObjects_(ssStore, 'Daily_Unified');
  const invalidSleep = daily.filter(function (r) {
    const h = num_(r.sleep_hrs);
    return h !== null && h < 2.0;
  });
  assert('Sleep <120m excluded', invalidSleep.length === 0, 'Rows with sleep <2.0h: ' + invalidSleep.length);

  // 3. Strict Thursday Pending rule check
  const weeks = readSheetObjects_(ssStore, 'Weekly_Fri_Reports');
  const badPending = weeks.filter(function (w) {
    return w.thu_weight_status === 'Pending' && w.close_thu_weight_kg !== 'Pending';
  });
  assert('Thursday Pending rule', weeks.length >= 5 && badPending.length === 0, 'Backfilled weeks=' + weeks.length);

  // 4. Zero Smoking_Log data in Health_Intelligence_Store
  let smokingLeak = false;
  ssStore.getSheets().forEach(function (sh) {
    if (sh.getName().toLowerCase().indexOf('smok') !== -1) smokingLeak = true;
  });
  assert('Zero smoking tabs in Store', !smokingLeak, 'Smoking_Log remains strictly isolated in Micro Journal');

  // 5. Bioavailability Audit populated
  const bio = readSheetObjects_(ssStore, 'Bioavailability_Audit');
  assert('Bioavailability Audit populated', bio.length > 0, 'Rows=' + bio.length);

  // 6. Doctor Brief runs cleanly
  const brief = buildWholeBodyDoctorBrief_();
  assert('Whole-Body Doctor Brief builds', brief.diagnoses.length > 0 && brief.latestLabs.length > 0, 'Diagnoses=' + brief.diagnoses.length + ', Labs=' + brief.latestLabs.length);

  results.forEach(function (r) {
    Logger.log('[' + r.status + '] ' + r.test + ' — ' + r.detail);
  });
  const failed = results.filter(function (r) { return r.status === 'FAIL'; });
  if (failed.length) throw new Error('Phase 1 Self-Test FAILED: ' + failed.length + ' assertion(s).');
  Logger.log('ALL ' + results.length + ' PHASE 1 SELF-TESTS PASSED!');
  return results;
}
