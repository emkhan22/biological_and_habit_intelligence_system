import React, { useMemo, useState } from "react";
import { Info, Printer } from "lucide-react";
import type { HealthDataPayload } from "../types";
import { EpistemicPill } from "../components/EpistemicPill";
import { formatDateIN, formatNum, formatSigned, safeJsonParse } from "../utils/formatters";
import {
  humanizeClinicalText,
  humanizeFeature,
  humanizeStatus,
  humanizeTreatmentId,
  splitIntoScannablePoints,
} from "../utils/humanize";

interface Props {
  data: HealthDataPayload;
}

const KEY_VITAL_METRICS = new Set([
  "weight_kg",
  "hrv_rmssd_ms",
  "avg_hr",
  "daytime_avg_hr",
  "sleep_hr_avg",
  "resting_hr",
  "asleep_min",
  "steps",
  "hunger",
  "food_noise",
]);

export const TherapiesReportsView: React.FC<Props> = ({ data }) => {
  const allScorecards = data.tables.Gold_Treatment_Scorecards ?? [];
  const weeklyBriefs = (data.tables.AI_Weekly_Briefs ?? []).slice(-4).reverse();
  const monthlyBriefs = (data.tables.AI_Monthly_Briefs ?? []).slice(-1);
  const doctorBriefs = data.tables.AI_Doctor_Briefs ?? [];
  const latestDoctor = doctorBriefs[doctorBriefs.length - 1];
  const journalThemes = (data.tables.Gold_Journal_Themes_AI ?? []).slice(-6).reverse();
  const labs = (data.tables.Silver_Lab_Results ?? []).slice(-10);

  // Item 9: Treatment filter chips + Key Vitals & Weight Only toggle
  const [selectedTreatment, setSelectedTreatment] = useState<string>("ALL");
  const [keyVitalsOnly, setKeyVitalsOnly] = useState<boolean>(true);

  const uniqueTreatments = useMemo(() => {
    const seen = new Set<string>();
    const list: string[] = [];
    for (const r of allScorecards) {
      const tid = String(r.treatment_id || "").trim();
      if (tid && !seen.has(tid)) {
        seen.add(tid);
        list.push(tid);
      }
    }
    return list;
  }, [allScorecards]);

  const filteredScorecards = useMemo(() => {
    return allScorecards.filter((r) => {
      if (selectedTreatment !== "ALL" && r.treatment_id !== selectedTreatment) return false;
      if (keyVitalsOnly && !KEY_VITAL_METRICS.has(String(r.metric_name || ""))) return false;
      return true;
    });
  }, [allScorecards, selectedTreatment, keyVitalsOnly]);

  const docQuestions = safeJsonParse<Array<Record<string, any>>>(
    latestDoctor?.top_5_questions_json,
    []
  );
  const docChecklist = safeJsonParse<string[]>(latestDoctor?.lab_delta_checklist_json, []);
  const docSummaryPoints = splitIntoScannablePoints(latestDoctor?.printable_clinical_summary, 6);
  const docHighlightPoints = splitIntoScannablePoints(latestDoctor?.treatment_scorecard_highlights, 4);

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <h1 className="text-xl font-semibold tracking-tight">Therapies &amp; Reports</h1>
          <p className="text-xs text-slate-400">
            Dual-Baseline Treatment Scorecards, Journal Themes, Longitudinal Labs, Weekly/Monthly Briefs &amp; Printable Doctor Prep
          </p>
        </div>
        <button
          type="button"
          onClick={() => window.print()}
          className="no-print inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-[#1B202C] border border-[#283042] text-xs font-medium text-slate-200 hover:text-white cursor-pointer"
        >
          <Printer className="w-3.5 h-3.5 text-[#60A5FA]" />
          Print Doctor Brief
        </button>
      </div>

      {/* Item 9: Dual-Baseline Treatment Scorecards */}
      <section className="obsidian-card p-5 space-y-3">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div>
            <h2 className="text-sm font-semibold">Dual-Baseline Treatment Scorecards (Pre-Treatment &amp; Prior-Dose)</h2>
            <p className="text-xs text-slate-400">
              Filter by therapy or focus on key vitals &amp; weight metrics
            </p>
          </div>
          <EpistemicPill label="Descriptive" />
        </div>

        {/* Treatment Filter Chips + Key Vitals Toggle */}
        <div className="flex flex-wrap items-center justify-between gap-2 pt-1">
          <div className="flex flex-wrap items-center gap-1.5 text-[11px]">
            <button
              type="button"
              onClick={() => setSelectedTreatment("ALL")}
              className={`px-2.5 py-1 rounded-md border cursor-pointer ${
                selectedTreatment === "ALL"
                  ? "bg-[#1B202C] border-[#C06C9F]/60 text-slate-100 font-medium"
                  : "bg-[#0B0D12] border-[#283042] text-slate-400 hover:text-slate-200"
              }`}
            >
              All ({allScorecards.length})
            </button>
            {uniqueTreatments.map((tid) => (
              <button
                key={tid}
                type="button"
                onClick={() => setSelectedTreatment(tid)}
                className={`px-2.5 py-1 rounded-md border cursor-pointer ${
                  selectedTreatment === tid
                    ? "bg-[#1B202C] border-[#60A5FA]/60 text-slate-100 font-medium"
                    : "bg-[#0B0D12] border-[#283042] text-slate-400 hover:text-slate-200"
                }`}
              >
                {humanizeTreatmentId(tid)}
              </button>
            ))}
          </div>

          <label className="inline-flex items-center gap-2 text-xs text-slate-300 cursor-pointer">
            <input
              type="checkbox"
              checked={keyVitalsOnly}
              onChange={(e) => setKeyVitalsOnly(e.target.checked)}
              className="accent-[#34D399]"
            />
            Key Vitals &amp; Weight Only
          </label>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-xs text-left border-collapse">
            <thead>
              <tr className="border-b border-[#283042] text-slate-400">
                <th className="py-2 pr-3">Treatment</th>
                <th className="py-2 pr-3">Metric</th>
                <th className="py-2 pr-3">Window Avg</th>
                <th className="py-2 pr-3">vs Pre-Treatment</th>
                <th className="py-2 pr-3">vs Prior Dose</th>
                <th className="py-2 pr-3">Overlap</th>
                <th className="py-2">Window Status</th>
              </tr>
            </thead>
            <tbody className="font-mono tabular-nums">
              {filteredScorecards.map((r, i) => (
                <tr key={i} className="border-b border-[#283042]/60 text-slate-200">
                  <td className="py-2 pr-3 font-sans font-medium">
                    {humanizeTreatmentId(r.treatment_id)}
                  </td>
                  <td className="py-2 pr-3 font-sans text-slate-300">
                    {humanizeFeature(r.metric_name)}
                  </td>
                  <td className="py-2 pr-3">{formatNum(r.current_full_phase_mean, 1)}</td>
                  <td className="py-2 pr-3">{formatSigned(r.delta_vs_pre_treatment, 1)}</td>
                  <td className="py-2 pr-3">{formatSigned(r.delta_vs_prior_dose, 1)}</td>
                  <td className="py-2 pr-3 font-sans">
                    {String(r.overlap_warning).toLowerCase() === "true" ? (
                      <span className="text-[#FBBF24]">Concurrent Overlap</span>
                    ) : (
                      <span className="text-slate-500">Clean</span>
                    )}
                  </td>
                  <td className="py-2 font-sans text-slate-300">
                    {humanizeStatus(r.clinical_window_status)}
                  </td>
                </tr>
              ))}
              {filteredScorecards.length === 0 && (
                <tr>
                  <td colSpan={7} className="py-4 text-center font-sans text-slate-500">
                    No metrics match the active filter. Uncheck &ldquo;Key Vitals &amp; Weight Only&rdquo; to view all tracked metrics.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>

        <p className="text-[11px] text-slate-500 flex items-start gap-1.5">
          <Info className="w-3.5 h-3.5 shrink-0 mt-0.5 text-slate-400" />
          <span>
            Some treatments or dose steps only display metrics that have enough valid days inside the active phase window (minimum sample threshold met). Concurrent October/November starts are marked as Concurrent Overlap so changes are interpreted in full regimen context.
          </span>
        </p>
      </section>

      {/* Item 10: Printable Specialist Doctor Visit Brief */}
      {latestDoctor && (
        <section className="obsidian-card p-6 space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-2 pb-3 border-b border-[#283042]">
            <div>
              <h2 className="text-base font-semibold text-slate-100">
                Printable Specialist Doctor Brief ({String(latestDoctor.specialty || "General").replace(/_/g, " ")}) ·{" "}
                {formatDateIN(latestDoctor.display_date || latestDoctor.report_date)}
              </h2>
              <p className="text-xs text-slate-400 mt-0.5">
                Prepared for physician review · Engine: {latestDoctor.engine_badge || latestDoctor.model_used}
              </p>
            </div>
            <EpistemicPill label="Screening_Context" />
          </div>

          {/* Structured Clinical Summary */}
          <div className="space-y-2.5">
            <h3 className="text-xs uppercase tracking-wider font-semibold text-[#60A5FA]">
              Clinical Executive Summary
            </h3>
            <ul className="space-y-2 text-sm text-slate-200 leading-relaxed list-disc pl-5">
              {docSummaryPoints.map((pt, idx) => (
                <li key={idx}>{pt}</li>
              ))}
            </ul>
          </div>

          {docHighlightPoints.length > 0 && (
            <div className="obsidian-subcard p-4 space-y-2">
              <h3 className="text-xs uppercase tracking-wider font-semibold text-[#34D399]">
                Key Regimen &amp; Biomarker Highlights
              </h3>
              <ul className="space-y-1.5 text-sm text-slate-200 leading-relaxed list-disc pl-5">
                {docHighlightPoints.map((pt, idx) => (
                  <li key={idx}>{pt}</li>
                ))}
              </ul>
            </div>
          )}

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
            <div className="obsidian-subcard p-4 space-y-2">
              <h3 className="text-xs uppercase tracking-wider font-semibold text-[#FBBF24]">
                Lab-Delta &amp; Retest Checklist
              </h3>
              <ul className="list-disc pl-5 space-y-1.5 text-slate-200 leading-relaxed text-xs sm:text-sm">
                {docChecklist.map((item, idx) => (
                  <li key={idx}>{humanizeClinicalText(item)}</li>
                ))}
              </ul>
            </div>
            <div className="obsidian-subcard p-4 space-y-2">
              <h3 className="text-xs uppercase tracking-wider font-semibold text-[#A78BFA]">
                Top Prioritized Specialist Questions
              </h3>
              <ol className="list-decimal pl-5 space-y-2.5 text-slate-200 leading-relaxed text-xs sm:text-sm">
                {docQuestions.map((q, idx) => (
                  <li key={idx}>
                    <span className="font-medium text-slate-100">{humanizeClinicalText(q.question)}</span>
                    {q.supporting_patient_data ? (
                      <span className="block text-xs text-slate-400 mt-0.5">
                        {humanizeClinicalText(q.supporting_patient_data)}
                      </span>
                    ) : null}
                  </li>
                ))}
              </ol>
            </div>
          </div>
        </section>
      )}

      {/* Item 11: Weekly & Monthly AI Reviews */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <section className="obsidian-card p-5 space-y-3.5">
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-semibold">Friday Weekly Deep Reviews (Thu–Wed)</h2>
            <EpistemicPill label="Descriptive" />
          </div>
          <div className="space-y-3">
            {weeklyBriefs.map((w) => {
              const priorities = safeJsonParse<string[]>(w.next_week_priorities_json, []);
              const dosePts = splitIntoScannablePoints(w.dose_working, 3);
              const habitPts = splitIntoScannablePoints(w.habit_review_improved_vs_slipped, 3);
              return (
                <div key={w.week_id} className="obsidian-subcard p-4 space-y-2.5">
                  <div className="flex flex-wrap items-center justify-between gap-2 pb-1.5 border-b border-[#283042]/80">
                    <p className="text-sm font-semibold text-slate-100 leading-snug">
                      {humanizeClinicalText(w.headline)}
                    </p>
                    <span className="text-[11px] font-mono text-slate-400">
                      {formatDateIN(w.display_start_date || w.week_start_date)} → {formatDateIN(w.display_end_date || w.week_end_date)}
                    </span>
                  </div>

                  {dosePts.length > 0 && (
                    <div className="space-y-1">
                      <p className="text-[11px] uppercase tracking-wider font-semibold text-[#34D399]">
                        Key Takeaways &amp; Dose Response
                      </p>
                      <ul className="list-disc pl-4 space-y-1 text-xs text-slate-200 leading-relaxed">
                        {dosePts.map((pt, i) => (
                          <li key={i}>{pt}</li>
                        ))}
                      </ul>
                    </div>
                  )}

                  {habitPts.length > 0 && (
                    <div className="space-y-1">
                      <p className="text-[11px] uppercase tracking-wider font-semibold text-[#FBBF24]">
                        Wins &amp; Watch-Outs
                      </p>
                      <ul className="list-disc pl-4 space-y-1 text-xs text-slate-300 leading-relaxed">
                        {habitPts.map((pt, i) => (
                          <li key={i}>{pt}</li>
                        ))}
                      </ul>
                    </div>
                  )}

                  {priorities.length > 0 && (
                    <div className="space-y-1">
                      <p className="text-[11px] uppercase tracking-wider font-semibold text-[#60A5FA]">
                        Next-Week Focus
                      </p>
                      <ul className="list-disc pl-4 space-y-0.5 text-xs text-slate-300 leading-relaxed">
                        {priorities.slice(0, 3).map((pr, i) => (
                          <li key={i}>{humanizeClinicalText(typeof pr === "string" ? pr : JSON.stringify(pr))}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </section>

        <section className="obsidian-card p-5 space-y-3.5">
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-semibold">Monthly Review, Journal Themes &amp; Labs</h2>
            <EpistemicPill label="Descriptive" />
          </div>
          {monthlyBriefs[0] && (
            <div className="obsidian-subcard p-4 space-y-3">
              <p className="text-sm font-semibold text-slate-100 pb-1.5 border-b border-[#283042]/80">
                {humanizeClinicalText(monthlyBriefs[0].headline)}
              </p>

              <div className="space-y-1">
                <p className="text-[11px] uppercase tracking-wider font-semibold text-[#34D399]">
                  Body Composition (Fat vs. Lean)
                </p>
                <ul className="list-disc pl-4 space-y-1 text-xs text-slate-200 leading-relaxed">
                  {splitIntoScannablePoints(monthlyBriefs[0].body_comp_fat_vs_lean_summary, 3).map((pt, i) => (
                    <li key={i}>{pt}</li>
                  ))}
                </ul>
              </div>

              {monthlyBriefs[0].masld_and_metabolic_trajectory && (
                <div className="space-y-1">
                  <p className="text-[11px] uppercase tracking-wider font-semibold text-[#60A5FA]">
                    Metabolic &amp; MASLD Trajectory
                  </p>
                  <ul className="list-disc pl-4 space-y-1 text-xs text-slate-300 leading-relaxed">
                    {splitIntoScannablePoints(monthlyBriefs[0].masld_and_metabolic_trajectory, 2).map((pt, i) => (
                      <li key={i}>{pt}</li>
                    ))}
                  </ul>
                </div>
              )}

              <div className="space-y-1">
                <p className="text-[11px] uppercase tracking-wider font-semibold text-[#FBBF24]">
                  Labs &amp; Retest Readiness
                </p>
                <ul className="list-disc pl-4 space-y-1 text-xs text-slate-300 leading-relaxed">
                  {splitIntoScannablePoints(monthlyBriefs[0].lab_retest_readiness, 3).map((pt, i) => (
                    <li key={i}>{pt}</li>
                  ))}
                </ul>
              </div>
            </div>
          )}

          <div className="space-y-1.5 text-xs">
            <p className="text-[11px] uppercase tracking-wider text-slate-400">Recent AI-Enriched Journal Themes</p>
            {journalThemes.map((j) => (
              <div key={j.event_id} className="obsidian-subcard px-3 py-2 flex items-center justify-between gap-2">
                <span className="text-slate-200">
                  {String(j.detected_themes_ai || "").replace(/_/g, " ")}
                </span>
                <span className="font-mono text-[11px] text-slate-400 shrink-0">
                  {formatDateIN(j.display_date || j.date)}
                </span>
              </div>
            ))}
          </div>
          {labs.length > 0 && (
            <p className="text-[11px] text-slate-500">
              Latest normalized labs synced: {labs.length} records (immutable original clinical values preserved).
            </p>
          )}
        </section>
      </div>
    </div>
  );
};
