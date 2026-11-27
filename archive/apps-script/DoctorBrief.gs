/**
 * DoctorBrief.gs — Whole-Body Clinician Summary Aggregator.
 * Combines AggregatedMedicalRecords (Diagnoses, Medications, Procedures,
 * Imaging_Reports, Lab_Results, Extraction_Review) with derived trends from
 * Health_Intelligence_Store. Never includes Smoking_Log.
 */

function buildWholeBodyDoctorBrief_() {
  const cfg = getConfig_();
  const ssMed = SpreadsheetApp.openById(cfg.SOURCES.MEDICAL);
  const ssStore = SpreadsheetApp.openById(cfg.SOURCES.STORE);

  const diagnoses = readSheetObjects_(ssMed, 'Diagnoses');
  const medications = readSheetObjects_(ssMed, 'Medications');
  const procedures = readSheetObjects_(ssMed, 'Procedures');
  const imaging = readSheetObjects_(ssMed, 'Imaging_Reports');
  const extractionReview = readSheetObjects_(ssMed, 'Extraction_Review');

  const labTrends = readSheetObjects_(ssStore, 'Lab_Trends');
  const weeklyReports = readSheetObjects_(ssStore, 'Weekly_Fri_Reports');
  const dailyUnified = readSheetObjects_(ssStore, 'Daily_Unified');
  const menstrual = readSheetObjects_(ssStore, 'Menstrual_Longitudinal');

  // Active medications (preserving documented status verbatim)
  const activeMeds = medications.filter(function (m) {
    const st = String(m.Status || '').toLowerCase();
    return st.indexOf('currently') !== -1 || st.indexOf('active') !== -1 || st.indexOf('prescribed') !== -1;
  }).map(function (m) {
    return {
      generic: String(m.Drug_Name_Generic || ''),
      brand: String(m.Brand_Name || ''),
      strength: String(m.Strength || ''),
      dosage: String(m.Prescribed_Dosage || ''),
      frequency: String(m.Frequency || ''),
      status: String(m.Status || ''),
      prescriptionDate: isoDate_(m.Prescription_Date, cfg.TIMEZONE)
    };
  });

  // Latest value per standardized lab test
  const latestLabsMap = {};
  labTrends.forEach(function (r) {
    const name = String(r.standardized_test_name || '');
    const dt = String(r.test_date || '');
    if (!name) return;
    if (!latestLabsMap[name] || dt >= latestLabsMap[name].testDate) {
      latestLabsMap[name] = {
        testName: name,
        testDate: dt,
        value: String(r.value || ''),
        unit: String(r.unit || ''),
        refRange: String(r.reference_range || ''),
        abnormalFlag: String(r.abnormal_flag || ''),
        sourceFilename: String(r.source_filename || ''),
        provenance: 'Measured'
      };
    }
  });

  // Top side effects across GLP-1 window
  const seCounts = {};
  dailyUnified.forEach(function (d) {
    if (!d.top_side_effects) return;
    String(d.top_side_effects).split(',').forEach(function (s) {
      const k = s.trim();
      if (k) seCounts[k] = (seCounts[k] || 0) + 1;
    });
  });
  const topSideEffects = Object.keys(seCounts)
    .map(function (k) { return { symptom: k, daysLogged: seCounts[k] }; })
    .sort(function (a, b) { return b.daysLogged - a.daysLogged; })
    .slice(0, 8);

  // Latest week summary
  const latestWeek = weeklyReports.length ? weeklyReports[weeklyReports.length - 1] : null;

  // Recent menstrual cycles (latest 6)
  const recentCycles = menstrual.slice(0, 6).map(function (c) {
    return {
      cycleStart: String(c.cycle_start || ''),
      cycleLengthDays: c.cycle_length_days,
      flowDays: c.marked_flow_days,
      sourceApp: String(c.source_app || '')
    };
  });

  // Open Extraction_Review flags
  const openExtractionFlags = extractionReview.filter(function (e) {
    const st = String(e.Resolution_Status || '').toLowerCase();
    return st !== 'resolved' && st !== 'closed';
  }).map(function (e) {
    return {
      severity: String(e.Severity || ''),
      affectedTab: String(e.Affected_Tab || ''),
      description: String(e.Description || ''),
      sourceFile: String(e.Source_Filename || '')
    };
  });

  return {
    generatedAt: Utilities.formatDate(new Date(), cfg.TIMEZONE, 'yyyy-MM-dd HH:mm:ss') + ' IST',
    calcVersion: cfg.CALC_VERSION,
    diagnoses: diagnoses.map(function (d) {
      return {
        condition: String(d.Condition_Or_Finding || ''),
        category: String(d.Category || ''),
        status: String(d.Status || ''),
        dateDocumented: isoDate_(d.Date_Documented, cfg.TIMEZONE),
        sourceFile: String(d.Source_Filename || '')
      };
    }),
    activeMedications: activeMeds,
    procedures: procedures.map(function (p) {
      return {
        name: String(p.Procedure_Name || ''),
        date: isoDate_(p.Date_Performed_Or_Planned, cfg.TIMEZONE),
        status: String(p.Status || ''),
        indication: String(p.Indication || '')
      };
    }),
    imagingHighlights: imaging.map(function (im) {
      return {
        date: isoDate_(im.Exam_Date, cfg.TIMEZONE),
        modality: String(im.Modality || ''),
        region: String(im.Body_Region || ''),
        impression: String(im.Impression_Verbatim || '')
      };
    }),
    latestLabs: Object.keys(latestLabsMap).sort().map(function (k) { return latestLabsMap[k]; }),
    latestWeekSummary: latestWeek,
    topSideEffects: topSideEffects,
    recentCycles: recentCycles,
    openExtractionFlags: openExtractionFlags
  };
}
