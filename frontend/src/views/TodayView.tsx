import React, { useState } from "react";
import { ChevronDown, ChevronUp, Droplets, HeartPulse, Scale, Sparkles, Syringe } from "lucide-react";
import type { HealthDataPayload } from "../types";
import { EpistemicPill } from "../components/EpistemicPill";
import { StatsCoverageDrawer } from "../components/StatsCoverageDrawer";
import {
  formatDateIN,
  formatNum,
  formatPct,
  formatSigned,
  formatSleepHm,
  safeJsonParse,
} from "../utils/formatters";
import {
  findLatestValid,
  humanizeClinicalText,
  humanizeStatus,
  humanizeTreatmentId,
  resolveReadinessPillars,
} from "../utils/humanize";

interface Props {
  data: HealthDataPayload;
}

export const TodayView: React.FC<Props> = ({ data }) => {
  const [nudgesOpen, setNudgesOpen] = useState(false);

  const packet = data.packet ?? {};
  const meta = packet.meta ?? {};
  const t7 = packet.today_and_7d ?? {};
  const tirz = packet.tirzepatide_and_treatments ?? {};
  const wt = packet.weight_alignment ?? {};
  const clin = packet.clinical_context ?? {};

  const dailyRows = data.tables.Gold_Daily_Unified ?? [];
  const baselineRows = data.tables.Gold_Personal_Baselines ?? [];
  const spikeRows = data.tables.Gold_Sleep_HR_Spikes ?? [];
  const latestDaily = dailyRows[dailyRows.length - 1] ?? {};
  const aiInsights = data.tables.AI_Insights_Cache ?? [];
  const latestBrief = aiInsights[aiInsights.length - 1] ?? {};
  const habitRows = data.tables.Gold_Habit_Durability ?? [];
  const postGlp1Weeks = habitRows.filter((r) => r.row_type === "Post_GLP1_Week");
  const latestPostWeek = postGlp1Weeks[postGlp1Weeks.length - 1];
  const peakVsTrough = habitRows.filter((r) => r.row_type === "On_GLP1_Peak_vs_Trough");

  const isPeriodActive =
    latestDaily.is_period_day === true ||
    String(latestDaily.is_period_day).toLowerCase() === "true";

  // Item 1: 5-Pillar Readiness Strip with latest-valid fallback & Regimen Consistency sync
  const briefCombinedText = `${latestBrief.headline || ""} ${latestBrief.overview_one_liner || ""} ${latestBrief.inside_your_body_summary || ""}`;
  const resolvedPillars = resolveReadinessPillars(t7.readiness_pillars, dailyRows, briefCombinedText);

  const pillarItems = [
    { label: "Sleep & Circadian", metric: resolvedPillars.sleep, color: "text-[#A78BFA]" },
    { label: "Autonomic Recovery", metric: resolvedPillars.autonomic, color: "text-[#34D399]" },
    { label: "Activity & Steps", metric: resolvedPillars.activity, color: "text-[#60A5FA]" },
    { label: "Nutrition & Fuel", metric: resolvedPillars.nutrition, color: "text-[#FBBF24]" },
    { label: "Regimen Consistency", metric: resolvedPillars.regimen, color: "text-[#C06C9F]" },
  ];

  // Item 1: Latest-valid fallback for Autonomic & Sleep Recovery card
  const latestHrv =
    t7.latest_biometrics?.hrv_rmssd_ms != null && t7.latest_biometrics?.hrv_rmssd_ms !== ""
      ? { value: Number(t7.latest_biometrics.hrv_rmssd_ms), date: meta.report_date_display || latestDaily.display_date, isFallback: false }
      : findLatestValid(dailyRows, "hrv_rmssd_ms");

  const latestSleep =
    t7.latest_biometrics?.asleep_min != null && t7.latest_biometrics?.asleep_min !== ""
      ? { value: Number(t7.latest_biometrics.asleep_min), date: meta.report_date_display || latestDaily.display_date, isFallback: false }
      : findLatestValid(dailyRows, "asleep_min");

  const latestSleepHr =
    t7.latest_biometrics?.sleep_hr_avg != null && t7.latest_biometrics?.sleep_hr_avg !== ""
      ? { value: Number(t7.latest_biometrics.sleep_hr_avg), date: meta.report_date_display || latestDaily.display_date, isFallback: false }
      : findLatestValid(dailyRows, "sleep_hr_avg");

  const latestSpikes =
    t7.latest_sleep_hr_spikes_per_hr != null && t7.latest_sleep_hr_spikes_per_hr !== ""
      ? { value: Number(t7.latest_sleep_hr_spikes_per_hr), date: meta.report_date_display || latestDaily.display_date, isFallback: false }
      : findLatestValid(spikeRows, "spikes_per_asleep_hr");

  const zScores = t7.latest_z_scores ?? {};
  const hrvZ =
    zScores.hrv_robust_z != null
      ? Number(zScores.hrv_robust_z)
      : findLatestValid(baselineRows, "hrv_z_robust_glp1").value ??
        findLatestValid(baselineRows, "hrv_z_robust_pre_oct").value;
  const sleepZ =
    zScores.sleep_dur_robust_z != null
      ? Number(zScores.sleep_dur_robust_z)
      : findLatestValid(baselineRows, "sleep_dur_z_robust_glp1").value ??
        findLatestValid(baselineRows, "sleep_dur_z_robust_pre_oct").value;

  const nudges = safeJsonParse<Array<Record<string, any>>>(latestBrief.action_nudges_json, []);
  const patterns = safeJsonParse<Array<Record<string, any>>>(latestBrief.top_3_patterns_json, []);
  const openWindows: Array<Record<string, any>> = clin.open_retest_windows ?? [];
  const latestMasld = (wt.masld_progress ?? [])[0];
  const latestInterval = (wt.recent_intervals ?? [])[wt.recent_intervals?.length - 1];

  return (
    <div className="space-y-5">
      {/* Conditional Active Period Banner */}
      {isPeriodActive && (
        <div
          className="rounded-xl border border-[#C06C9F]/40 bg-[#C06C9F]/10 px-4 py-3 flex items-center justify-between text-xs"
          data-testid="active-period-banner"
        >
          <span className="inline-flex items-center gap-2 text-[#F1F5F9]">
            <Droplets className="w-4 h-4 text-[#C06C9F]" />
            <strong>Active Menstrual Phase (Cycle Day {latestDaily.cycle_day ?? "—"}):</strong>{" "}
            Short-term scale weight &amp; resting HR shifts may reflect luteal/menstrual fluid retention.
          </span>
          <EpistemicPill label="Measured" />
        </div>
      )}

      {/* Executive Overview Band */}
      <section className="obsidian-card p-5 space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <p className="text-[11px] uppercase tracking-wider text-slate-400">
              Executive Overview · {formatDateIN(meta.report_date_display || latestDaily.display_date)} (IST)
            </p>
            <h1 className="text-xl font-semibold tracking-tight text-slate-100 mt-0.5">
              {humanizeClinicalText(latestBrief.headline) || "Daily Biological & Habit Synthesis"}
            </h1>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <span className="px-2.5 py-1 rounded-md text-xs font-medium bg-[#1B202C] border border-[#283042] text-[#A78BFA]">
              {t7.latest_phenotype || "Wearable Baseline"}
            </span>
            {tirz.active_tirz_dose_mg != null && (
              <span className="px-2.5 py-1 rounded-md text-xs font-medium bg-[#1B202C] border border-[#283042] text-[#34D399]">
                Tirzepatide {tirz.active_tirz_dose_mg} mg · Day {tirz.shot_day_int ?? "—"}
              </span>
            )}
            {latestPostWeek && (
              <span
                className="px-2.5 py-1 rounded-md text-xs font-semibold bg-[#FBBF24]/15 border border-[#FBBF24]/40 text-[#FBBF24]"
                data-testid="off-glp1-week-chip"
              >
                Off-GLP-1 Week {latestPostWeek.post_glp1_week_num}
              </span>
            )}
            {openWindows.slice(0, 2).map((w) => (
              <span
                key={w.treatment_id}
                className="px-2.5 py-1 rounded-md text-xs font-medium bg-[#FB7185]/10 border border-[#FB7185]/30 text-[#FB7185]"
              >
                {humanizeTreatmentId(w.treatment_id)}: {humanizeStatus(w.clinical_window_status)}
              </span>
            ))}
          </div>
        </div>

        {/* 5-Pillar Readiness Strip */}
        <div className="grid grid-cols-2 sm:grid-cols-5 gap-2.5">
          {pillarItems.map((p) => (
            <div key={p.label} className="obsidian-subcard px-3 py-2.5">
              <p className="text-[11px] text-slate-400 truncate">{p.label}</p>
              <p className={`text-lg font-semibold font-mono tabular-nums mt-0.5 ${p.color}`}>
                {p.metric.value !== null ? `${p.metric.value.toFixed(p.label === "Regimen Consistency" ? 1 : 0)}` : "—"}
              </p>
              {p.metric.isFallback && p.metric.date && (
                <p className="text-[10px] text-slate-500 mt-0.5">
                  as of {formatDateIN(p.metric.date)}
                </p>
              )}
            </div>
          ))}
        </div>

        {/* 1-Line AI Brief + Expandable Nudges */}
        <div className="obsidian-subcard p-3.5 space-y-2.5">
          <div className="flex items-start justify-between gap-3">
            <div className="flex items-start gap-2">
              <Sparkles className="w-4 h-4 text-[#C06C9F] shrink-0 mt-0.5" />
              <p className="text-sm text-slate-200 leading-relaxed">
                {humanizeClinicalText(latestBrief.overview_one_liner) ||
                  "Validated Silver/Gold telemetry synced. Expand coaching nudges for actionable priorities."}
              </p>
            </div>
            <EpistemicPill label={latestBrief.epistemic_label || "Descriptive"} className="shrink-0" />
          </div>

          <button
            type="button"
            onClick={() => setNudgesOpen((v) => !v)}
            className="inline-flex items-center gap-1.5 text-xs font-medium text-[#60A5FA] hover:underline cursor-pointer"
          >
            {nudgesOpen
              ? "Hide Patterns & Coaching Nudges"
              : `Expand Top Patterns (${patterns.length}) & Nudges (${nudges.length})`}
            {nudgesOpen ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
          </button>

          {nudgesOpen && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3 pt-2 border-t border-[#283042]">
              <div className="space-y-2">
                <p className="text-[11px] uppercase tracking-wider text-slate-400">Top Patterns</p>
                {patterns.map((pt, i) => (
                  <div key={i} className="p-2.5 rounded-lg bg-[#0B0D12] border border-[#283042] space-y-1">
                    <div className="flex items-center justify-between gap-2">
                      <span className="text-xs font-semibold text-slate-100">
                        {humanizeClinicalText(pt.pattern_title)}
                      </span>
                      <EpistemicPill label={pt.epistemic_label} />
                    </div>
                    <p className="text-xs text-slate-300">{humanizeClinicalText(pt.supporting_metric)}</p>
                    <p className="text-[11px] text-slate-500">{humanizeClinicalText(pt.caveat)}</p>
                  </div>
                ))}
              </div>
              <div className="space-y-2">
                <p className="text-[11px] uppercase tracking-wider text-slate-400">Action Nudges</p>
                {nudges.map((n, i) => (
                  <div key={i} className="p-2.5 rounded-lg bg-[#0B0D12] border border-[#283042] space-y-1">
                    <div className="flex items-center justify-between gap-2">
                      <span className="text-xs font-semibold text-slate-100">{humanizeClinicalText(n.title)}</span>
                      <EpistemicPill label={n.epistemic_label} />
                    </div>
                    <p className="text-xs text-slate-300">{humanizeClinicalText(n.why)}</p>
                    <p className="text-[11px] text-slate-400">Data: {humanizeClinicalText(n.your_data)}</p>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </section>

      {/* 4 Core Bento Cards */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Bento 1: Autonomic & Sleep Recovery */}
        <article className="obsidian-card p-5 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between gap-2 mb-3">
              <div className="flex items-center gap-2">
                <HeartPulse className="w-4 h-4 text-[#34D399]" />
                <h2 className="text-sm font-semibold">Autonomic &amp; Sleep Recovery</h2>
              </div>
              <EpistemicPill label="Measured" />
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-3 gap-2.5 font-mono tabular-nums">
              <div className="obsidian-subcard p-3">
                <span className="text-[11px] font-sans text-slate-400 block">Overnight HRV</span>
                <span className="text-base font-semibold text-[#34D399]">
                  {formatNum(latestHrv.value, 1, " ms")}
                </span>
                <span className="text-[11px] text-slate-500 block">
                  Z: {formatSigned(hrvZ)}
                </span>
                {latestHrv.isFallback && latestHrv.date && (
                  <span className="text-[10px] font-sans text-slate-500 block mt-0.5">
                    as of {formatDateIN(latestHrv.date)}
                  </span>
                )}
              </div>
              <div className="obsidian-subcard p-3">
                <span className="text-[11px] font-sans text-slate-400 block">Clean Sleep</span>
                <span className="text-base font-semibold text-[#A78BFA]">
                  {formatSleepHm(latestSleep.value)}
                </span>
                <span className="text-[11px] text-slate-500 block">
                  Z: {formatSigned(sleepZ)}
                </span>
                {latestSleep.isFallback && latestSleep.date && (
                  <span className="text-[10px] font-sans text-slate-500 block mt-0.5">
                    as of {formatDateIN(latestSleep.date)}
                  </span>
                )}
              </div>
              <div className="obsidian-subcard p-3">
                <span className="text-[11px] font-sans text-slate-400 block">HR Spikes / Hr</span>
                <span className="text-base font-semibold text-[#60A5FA]">
                  {formatNum(latestSpikes.value, 2)}
                </span>
                <span className="text-[11px] text-slate-500 block">
                  Sleep HR: {formatNum(latestSleepHr.value, 1, " bpm")}
                </span>
                {(latestSpikes.isFallback || latestSleepHr.isFallback) && (latestSpikes.date || latestSleepHr.date) && (
                  <span className="text-[10px] font-sans text-slate-500 block mt-0.5">
                    as of {formatDateIN(latestSpikes.date || latestSleepHr.date)}
                  </span>
                )}
              </div>
            </div>
          </div>

          <StatsCoverageDrawer
            meta={{
              raw_n: meta.total_spine_days,
              valid_n: meta.coverage_7d?.sleep_present_days ?? dailyRows.slice(-7).filter((r) => r.asleep_min != null).length,
              effective_n: meta.coverage_7d?.vitals_present_days ?? dailyRows.slice(-7).filter((r) => r.hrv_rmssd_ms != null).length,
              missingness_pct:
                meta.coverage_7d?.sleep_present_days != null
                  ? ((7 - meta.coverage_7d.sleep_present_days) / 7) * 100
                  : ((7 - dailyRows.slice(-7).filter((r) => r.asleep_min != null).length) / 7) * 100,
              maturity_level: "7-Day Wearable Coverage",
              stability_flag: "Measured_Wear_Verified",
              caveat_note:
                "Only wear-verified Fitbit staged nights >=120m are averaged; unlogged/unworn nights remain NaN and display the latest valid synced date.",
            }}
          />
        </article>

        {/* Bento 2: Weight Velocity & Lean Shield */}
        <article className="obsidian-card p-5 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between gap-2 mb-3">
              <div className="flex items-center gap-2">
                <Scale className="w-4 h-4 text-[#60A5FA]" />
                <h2 className="text-sm font-semibold">Weight Velocity &amp; Lean Shield</h2>
              </div>
              <EpistemicPill label="Descriptive" />
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-3 gap-2.5 font-mono tabular-nums">
              <div className="obsidian-subcard p-3">
                <span className="text-[11px] font-sans text-slate-400 block">Latest / EMA</span>
                <span className="text-base font-semibold text-slate-100">
                  {formatNum(wt.latest_measured_weight_kg, 1, " kg")}
                </span>
                <span className="text-[11px] text-slate-500 block">
                  EMA: {formatNum(wt.latest_ema_weight_kg, 1, " kg")}
                </span>
              </div>
              <div className="obsidian-subcard p-3">
                <span className="text-[11px] font-sans text-slate-400 block">EMA Velocity</span>
                <span className="text-base font-semibold text-[#34D399]">
                  {formatNum(wt.ema_trend_loss_kg_per_wk, 2, " kg/wk")}
                </span>
                <span className="text-[11px] text-slate-500 block">
                  Fat Share: {formatPct(latestInterval?.fat_share_of_loss, true)}
                </span>
              </div>
              <div className="obsidian-subcard p-3">
                <span className="text-[11px] font-sans text-slate-400 block">MASLD 7% Goal</span>
                <span className="text-base font-semibold text-[#FBBF24]">
                  {formatNum(latestMasld?.pct_weight_lost_from_oct7, 1, "%")}
                </span>
                <span className="text-[11px] text-slate-500 block">
                  Left: {formatNum(latestMasld?.kg_remaining_to_target, 1, " kg")}
                </span>
              </div>
            </div>
          </div>

          <StatsCoverageDrawer
            meta={{
              raw_n: (data.tables.Gold_Weight_Intervals ?? []).length,
              valid_n: (data.tables.Gold_Weight_Intervals ?? []).filter((r) => r.is_fasted_interval).length,
              effective_n: (data.tables.Gold_Weight_Intervals ?? []).length,
              missingness_pct: 0,
              maturity_level: latestInterval?.maturity_level || "Level 0 (Descriptive Ledger)",
              stability_flag: "Fasted_Anchor_Priority",
              caveat_note:
                "MASLD 7% & 10% milestones track cumulative reduction from Oct 7 (108.3 kg) alongside Fitdays BIA fat-share.",
            }}
          />
        </article>

        {/* Bento 3: Tirzepatide Wave, Scorecards & Habit Carryover */}
        <article className="obsidian-card p-5 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between gap-2 mb-3">
              <div className="flex items-center gap-2">
                <Syringe className="w-4 h-4 text-[#C06C9F]" />
                <h2 className="text-sm font-semibold">Tirzepatide Wave, Scorecards &amp; Habit Carryover</h2>
              </div>
              <EpistemicPill label="Descriptive" />
            </div>

            <div className="grid grid-cols-2 gap-2.5 text-xs">
              {peakVsTrough.length > 0 ? (
                peakVsTrough.map((row) => (
                  <div key={row.period_id} className="obsidian-subcard p-3 space-y-1">
                    <p className="font-semibold text-slate-200">
                      {row.period_id === "Peak_Suppression_Days_0_2" ? "Peak Days 0–2" : "Trough Days 4–6"}
                    </p>
                    <p className="font-mono text-slate-300 tabular-nums">
                      Protein: {formatNum(row.protein_g_mean, 0, "g")} · Clean: {formatNum(row.clean_eating_score, 0)}
                    </p>
                    <p className="font-mono text-slate-400 tabular-nums">
                      Steps: {formatNum(row.steps_mean, 0)} · Sleep: {formatSleepHm(row.asleep_hr_mean != null ? Number(row.asleep_hr_mean) * 60 : null)}
                    </p>
                  </div>
                ))
              ) : (
                <div className="col-span-2 obsidian-subcard p-3 text-slate-400">
                  Collecting GLP-1 shot-day peak vs trough carryover metrics.
                </div>
              )}
            </div>
          </div>

          <StatsCoverageDrawer
            meta={{
              raw_n: habitRows.length,
              valid_n: peakVsTrough.reduce((acc, r) => acc + Number(r.valid_n || 0), 0),
              effective_n: peakVsTrough.reduce((acc, r) => acc + Number(r.effective_n || 0), 0),
              missingness_pct: peakVsTrough[0]?.missingness_pct ?? null,
              maturity_level: peakVsTrough[0]?.maturity_level || "Level 0 (Baseline)",
              stability_flag: "Descriptive_Only",
              caveat_note:
                "PK concentration is a modeled half-life curve (~5 days); treatment scorecards note concurrent Oct/Nov therapy starts.",
            }}
          />
        </article>

        {/* Bento 4: Clinical Labs, Cycle & Episode Chains */}
        <article className="obsidian-card p-5 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between gap-2 mb-3">
              <h2 className="text-sm font-semibold">Clinical Labs, Cycle &amp; Episode Chains</h2>
              <EpistemicPill label="Screening_Context" />
            </div>

            <div className="space-y-2 text-xs">
              {(clin.clinical_indices ?? []).slice(0, 3).map((idxRow: any, i: number) => (
                <div key={i} className="obsidian-subcard px-3 py-2 flex items-center justify-between gap-2">
                  <div>
                    <span className="font-medium text-slate-200">{idxRow.index_name}</span>
                    <span className="text-[11px] text-slate-500 block">
                      {formatDateIN(idxRow.display_date)} · Fasting: {String(idxRow.fasting_verified)}
                    </span>
                  </div>
                  <span className="font-mono font-semibold text-[#FB7185] tabular-nums">
                    {formatNum(idxRow.index_value, 2)} {idxRow.unit || ""}
                  </span>
                </div>
              ))}
              {openWindows.length > 0 && (
                <p className="text-[11px] text-slate-400 pt-1">
                  Open windows:{" "}
                  {openWindows
                    .map((w) => `${humanizeTreatmentId(w.treatment_id)} (Day ${w.days_on_treatment})`)
                    .join(" · ")}
                </p>
              )}
            </div>
          </div>

          <StatsCoverageDrawer
            meta={{
              raw_n: (data.tables.Gold_Clinical_Indices ?? []).length,
              valid_n: (data.tables.Gold_Clinical_Indices ?? []).length,
              effective_n: (data.tables.Gold_Clinical_Indices ?? []).length,
              missingness_pct: 0,
              maturity_level: "Clinical Screening Context",
              stability_flag: "Screening_Only",
              caveat_note:
                "FIB-4, HOMA-IR, and TG/HDL carry mandatory age & fasting caveats and are for physician discussion only.",
            }}
          />
        </article>
      </div>
    </div>
  );
};
