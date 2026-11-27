import React from "react";
import { Printer } from "lucide-react";
import type { HealthDataPayload } from "../types";
import { EpistemicPill } from "../components/EpistemicPill";
import { formatDateIN, formatNum, formatSigned, safeJsonParse } from "../utils/formatters";

interface Props {
  data: HealthDataPayload;
}

export const TherapiesReportsView: React.FC<Props> = ({ data }) => {
  const scorecards = (data.tables.Gold_Treatment_Scorecards ?? []).slice(0, 14);
  const weeklyBriefs = (data.tables.AI_Weekly_Briefs ?? []).slice(-4).reverse();
  const monthlyBriefs = (data.tables.AI_Monthly_Briefs ?? []).slice(-1);
  const doctorBriefs = data.tables.AI_Doctor_Briefs ?? [];
  const latestDoctor = doctorBriefs[doctorBriefs.length - 1];
  const journalThemes = (data.tables.Gold_Journal_Themes_AI ?? []).slice(-6).reverse();
  const labs = (data.tables.Silver_Lab_Results ?? []).slice(-10);

  const docQuestions = safeJsonParse<Array<Record<string, any>>>(
    latestDoctor?.top_5_questions_json,
    []
  );
  const docChecklist = safeJsonParse<string[]>(latestDoctor?.lab_delta_checklist_json, []);

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

      {/* Dual-Baseline Treatment Scorecards */}
      <section className="obsidian-card p-5 space-y-3">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold">Dual-Baseline Treatment Scorecards (Pre-Treatment &amp; Prior-Dose)</h2>
          <EpistemicPill label="Descriptive" />
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-xs text-left border-collapse">
            <thead>
              <tr className="border-b border-[#283042] text-slate-400">
                <th className="py-2 pr-3">Treatment</th>
                <th className="py-2 pr-3">Metric</th>
                <th className="py-2 pr-3">Current</th>
                <th className="py-2 pr-3">Δ vs Pre</th>
                <th className="py-2 pr-3">Δ vs Prior Dose</th>
                <th className="py-2 pr-3">Overlap</th>
                <th className="py-2">Window Status</th>
              </tr>
            </thead>
            <tbody className="font-mono tabular-nums">
              {scorecards.map((r, i) => (
                <tr key={i} className="border-b border-[#283042]/60 text-slate-200">
                  <td className="py-2 pr-3 font-sans font-medium">{r.treatment_id}</td>
                  <td className="py-2 pr-3">{r.metric_name}</td>
                  <td className="py-2 pr-3">{formatNum(r.current_full_phase_mean, 2)}</td>
                  <td className="py-2 pr-3">{formatSigned(r.delta_vs_pre_treatment, 2)}</td>
                  <td className="py-2 pr-3">{formatSigned(r.delta_vs_prior_dose, 2)}</td>
                  <td className="py-2 pr-3 font-sans">
                    {String(r.overlap_warning).toLowerCase() === "true" ? (
                      <span className="text-[#FBBF24]">Overlap</span>
                    ) : (
                      <span className="text-slate-500">Clean</span>
                    )}
                  </td>
                  <td className="py-2 font-sans text-slate-400">{r.clinical_window_status}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {/* Printable Specialist Doctor Visit Brief */}
      {latestDoctor && (
        <section className="obsidian-card p-5 space-y-3">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div>
              <h2 className="text-sm font-semibold">
                Printable Specialist Doctor Brief ({latestDoctor.specialty}) ·{" "}
                {formatDateIN(latestDoctor.display_date || latestDoctor.report_date)}
              </h2>
              <p className="text-[11px] text-slate-400">
                Engine: {latestDoctor.engine_badge || latestDoctor.model_used}
              </p>
            </div>
            <EpistemicPill label="Screening_Context" />
          </div>
          <p className="text-xs text-slate-200 leading-relaxed">{latestDoctor.printable_clinical_summary}</p>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs">
            <div className="obsidian-subcard p-3 space-y-1.5">
              <p className="font-semibold text-slate-100">Lab-Delta &amp; Retest Checklist</p>
              <ul className="list-disc pl-4 space-y-1 text-slate-300">
                {docChecklist.map((item, idx) => (
                  <li key={idx}>{item}</li>
                ))}
              </ul>
            </div>
            <div className="obsidian-subcard p-3 space-y-1.5">
              <p className="font-semibold text-slate-100">Top Specialist Questions</p>
              <ol className="list-decimal pl-4 space-y-1.5 text-slate-300">
                {docQuestions.map((q, idx) => (
                  <li key={idx}>
                    <span className="text-slate-100">{q.question}</span>
                    {q.supporting_patient_data ? (
                      <span className="block text-[11px] text-slate-400">{q.supporting_patient_data}</span>
                    ) : null}
                  </li>
                ))}
              </ol>
            </div>
          </div>
        </section>
      )}

      {/* Weekly & Monthly AI Reviews */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <section className="obsidian-card p-5 space-y-3">
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-semibold">Friday Weekly Deep Reviews (Thu–Wed)</h2>
            <EpistemicPill label="Descriptive" />
          </div>
          <div className="space-y-2.5 text-xs">
            {weeklyBriefs.map((w) => (
              <div key={w.week_id} className="obsidian-subcard p-3 space-y-1">
                <p className="font-semibold text-slate-100">{w.headline}</p>
                <p className="text-slate-300">{w.dose_working}</p>
                <p className="text-slate-400">{w.habit_review_improved_vs_slipped}</p>
              </div>
            ))}
          </div>
        </section>

        <section className="obsidian-card p-5 space-y-3">
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-semibold">Monthly Review, Journal Themes &amp; Labs</h2>
            <EpistemicPill label="Descriptive" />
          </div>
          {monthlyBriefs[0] && (
            <div className="obsidian-subcard p-3 space-y-1 text-xs">
              <p className="font-semibold text-slate-100">{monthlyBriefs[0].headline}</p>
              <p className="text-slate-300">{monthlyBriefs[0].body_comp_fat_vs_lean_summary}</p>
              <p className="text-slate-400">{monthlyBriefs[0].lab_retest_readiness}</p>
            </div>
          )}
          <div className="space-y-1.5 text-xs">
            <p className="text-[11px] uppercase tracking-wider text-slate-400">Recent AI-Enriched Journal Themes</p>
            {journalThemes.map((j) => (
              <div key={j.event_id} className="obsidian-subcard px-3 py-2 flex items-center justify-between gap-2">
                <span className="text-slate-200">{j.detected_themes_ai}</span>
                <span className="font-mono text-[11px] text-slate-400">{formatDateIN(j.display_date || j.date)}</span>
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
