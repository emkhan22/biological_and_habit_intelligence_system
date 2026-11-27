import React, { useMemo, useState } from "react";
import type { EChartsOption } from "echarts";
import { AlertTriangle, ChevronDown, ChevronUp, Sliders } from "lucide-react";
import type { HealthDataPayload, SimulatorAdviceResponse } from "../types";
import { EChartBox } from "../components/EChartBox";
import { EpistemicPill } from "../components/EpistemicPill";
import { StatsCoverageDrawer } from "../components/StatsCoverageDrawer";
import { requestSimulatorAdviceApi } from "../services/api";
import { formatDateIN, formatNum, formatPct, formatSigned } from "../utils/formatters";

interface Props {
  data: HealthDataPayload;
}

const AUDIT_BADGE: Record<string, string> = {
  Within_Expected_Band: "bg-[#34D399]/10 text-[#34D399] border-[#34D399]/30",
  Potential_Under_Logging: "bg-[#FBBF24]/10 text-[#FBBF24] border-[#FBBF24]/30",
  Potential_Over_Logging: "bg-[#60A5FA]/10 text-[#60A5FA] border-[#60A5FA]/30",
  Unclear_Water_Cycle_Or_Log_Gap: "bg-slate-700/30 text-slate-300 border-[#283042]",
};

export const BodyFuelView: React.FC<Props> = ({ data }) => {
  const weightTrend = data.tables.Gold_Weight_Trend ?? [];
  const weightIntervals = data.tables.Gold_Weight_Intervals ?? [];
  const habitRows = data.tables.Gold_Habit_Durability ?? [];
  const sweetRows = data.tables.Gold_Sweet_Hunger_Analysis ?? [];
  const calAuditRows = data.tables.Gold_Calorie_Balance_Audit ?? [];
  const mealBioRows = data.tables.Gold_Meal_Bioavailability_AI ?? [];

  const peakVsTrough = habitRows.filter((r) => r.row_type === "On_GLP1_Peak_vs_Trough");
  const postGlp1Weeks = habitRows.filter((r) => r.row_type === "Post_GLP1_Week");
  const weeklyHabits = habitRows.filter((r) => r.row_type === "Weekly_Durability").slice(-8);
  const offGlp1Active = postGlp1Weeks.length > 0;
  const [offPanelOpen, setOffPanelOpen] = useState(offGlp1Active);

  // Simulator State
  const [targetKg, setTargetKg] = useState(95.0);
  const [proteinSlider, setProteinSlider] = useState(100);
  const [stepsSlider, setStepsSlider] = useState(8500);
  const [bedtimeSlider, setBedtimeSlider] = useState("00:30");
  const [simResult, setSimResult] = useState<SimulatorAdviceResponse | null>(null);
  const [simLoading, setSimLoading] = useState(false);

  const weightTrendOption = useMemo<EChartsOption>(() => {
    const pts = weightTrend.filter((r) => r.ema_weight_kg != null || r.measured_weight_kg != null).slice(-45);
    return {
      tooltip: { trigger: "axis", backgroundColor: "#141821", borderColor: "#283042", textStyle: { color: "#F1F5F9" } },
      grid: { left: 44, right: 20, top: 24, bottom: 28 },
      xAxis: {
        type: "category",
        data: pts.map((p) => formatDateIN(p.display_date || p.date)),
        axisLine: { lineStyle: { color: "#283042" } },
      },
      yAxis: {
        type: "value",
        scale: true,
        splitLine: { lineStyle: { color: "#283042", type: "dashed" } },
      },
      series: [
        {
          name: "EMA Weight (kg)",
          type: "line",
          smooth: true,
          showSymbol: false,
          data: pts.map((p) => p.ema_weight_kg),
          lineStyle: { color: "#60A5FA", width: 2 },
        },
        {
          name: "Measured (kg)",
          type: "scatter",
          symbolSize: 7,
          data: pts.map((p) => p.measured_weight_kg),
          itemStyle: { color: "#34D399" },
        },
      ],
    };
  }, [weightTrend]);

  const sweetChartOption = useMemo<EChartsOption>(() => {
    const hunRows = sweetRows.filter((r) => r.outcome_metric !== "subsequent_meal_kcal");
    return {
      tooltip: { trigger: "axis", backgroundColor: "#141821", borderColor: "#283042", textStyle: { color: "#F1F5F9" } },
      legend: { top: 0, textStyle: { color: "#94A3B8", fontSize: 11 } },
      grid: { left: 40, right: 16, top: 32, bottom: 28 },
      xAxis: {
        type: "category",
        data: hunRows.map((r) => String(r.outcome_metric).replace(/_/g, " ")),
        axisLine: { lineStyle: { color: "#283042" } },
      },
      yAxis: {
        type: "value",
        splitLine: { lineStyle: { color: "#283042", type: "dashed" } },
      },
      series: [
        {
          name: "Sweet Days",
          type: "bar",
          data: hunRows.map((r) => r.sweet_window_mean),
          itemStyle: { color: "#C06C9F", borderRadius: [4, 4, 0, 0] },
        },
        {
          name: "Non-Sweet Logged",
          type: "bar",
          data: hunRows.map((r) => r.non_sweet_window_mean),
          itemStyle: { color: "#34D399", borderRadius: [4, 4, 0, 0] },
        },
      ],
    };
  }, [sweetRows]);

  const calAuditOption = useMemo<EChartsOption>(() => {
    const items = calAuditRows.slice(-8);
    return {
      tooltip: { trigger: "axis", backgroundColor: "#141821", borderColor: "#283042", textStyle: { color: "#F1F5F9" } },
      legend: { top: 0, textStyle: { color: "#94A3B8", fontSize: 11 } },
      grid: { left: 48, right: 16, top: 32, bottom: 32 },
      xAxis: {
        type: "category",
        data: items.map((r) => r.window_id),
        axisLine: { lineStyle: { color: "#283042" } },
      },
      yAxis: {
        type: "value",
        name: "Delta kg",
        splitLine: { lineStyle: { color: "#283042", type: "dashed" } },
      },
      series: [
        {
          name: "Expected Delta (kg)",
          type: "bar",
          data: items.map((r) => r.expected_delta_kg),
          itemStyle: { color: "#60A5FA" },
        },
        {
          name: "Actual Delta (kg)",
          type: "bar",
          data: items.map((r) => r.actual_delta_kg),
          itemStyle: { color: "#34D399" },
        },
      ],
    };
  }, [calAuditRows]);

  async function handleRunSimulator(e: React.FormEvent) {
    e.preventDefault();
    setSimLoading(true);
    try {
      const resp = await requestSimulatorAdviceApi(
        targetKg,
        { protein_g: proteinSlider, steps: stepsSlider, bedtime_ist: bedtimeSlider },
        data.packet
      );
      setSimResult(resp);
    } finally {
      setSimLoading(false);
    }
  }

  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-xl font-semibold tracking-tight">Body &amp; Fuel</h1>
        <p className="text-xs text-slate-400">
          3-Grain Weight Alignment, GLP-1 Habit Durability, Sweet → Hunger Audit, and BMR/TDEE Consistency
        </p>
      </div>

      {/* 1. 3-Grain Weight Alignment */}
      <section className="obsidian-card p-5 space-y-3">
        <div className="flex items-center justify-between gap-2">
          <h2 className="text-sm font-semibold">3-Grain Weight Trend &amp; Fasted Intervals</h2>
          <EpistemicPill label="Descriptive" />
        </div>
        <EChartBox option={weightTrendOption} height={240} isEmpty={weightTrend.length === 0} />
        <div className="overflow-x-auto">
          <table className="w-full text-xs text-left border-collapse">
            <thead>
              <tr className="border-b border-[#283042] text-slate-400">
                <th className="py-2 pr-3">Interval</th>
                <th className="py-2 pr-3">Dates (IST)</th>
                <th className="py-2 pr-3">Delta</th>
                <th className="py-2 pr-3">Rate</th>
                <th className="py-2 pr-3">Fat Share</th>
                <th className="py-2 pr-3">End Status</th>
                <th className="py-2">Nutr Cov</th>
              </tr>
            </thead>
            <tbody className="font-mono tabular-nums">
              {weightIntervals.slice(-6).map((row) => (
                <tr key={row.interval_id} className="border-b border-[#283042]/60 text-slate-200">
                  <td className="py-2 pr-3">{row.interval_id}</td>
                  <td className="py-2 pr-3 font-sans">
                    {formatDateIN(row.display_start_date)} → {formatDateIN(row.display_end_date)}
                  </td>
                  <td className="py-2 pr-3">{formatSigned(row.delta_kg, 2, " kg")}</td>
                  <td className="py-2 pr-3 text-[#34D399]">{formatNum(row.loss_rate_kg_per_wk, 2, " kg/wk")}</td>
                  <td className="py-2 pr-3">{formatPct(row.fat_share_of_loss, true)}</td>
                  <td className="py-2 pr-3 font-sans text-slate-400">{row.end_weighin_status}</td>
                  <td className="py-2">{formatPct(row.nutrition_coverage_pct, true)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <StatsCoverageDrawer
          meta={{
            raw_n: weightIntervals.length,
            valid_n: weightIntervals.filter((r) => r.is_fasted_interval).length,
            effective_n: weightIntervals.length,
            missingness_pct: 0,
            maturity_level: weightIntervals[0]?.maturity_level || "Level 0 (Descriptive Ledger)",
            stability_flag: "Fasted_Anchor_Priority",
            caveat_note: "Fasted weigh-ins are classified dynamically against first logged meal time (no naive clock cutoff).",
          }}
        />
      </section>

      {/* 2. Habit Carryover & Off-GLP-1 Durability */}
      <section className="obsidian-card p-5 space-y-3" data-testid="habit-durability-section">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div>
            <h2 className="text-sm font-semibold">Habit Carryover &amp; Off-GLP-1 Durability</h2>
            <p className="text-xs text-slate-400">
              Peak Suppression (Days 0–2) vs. Late-Cycle Trough (Days 4–6) + Automatic Taper / Post-GLP-1 Tracking
            </p>
          </div>
          <EpistemicPill label="Descriptive" />
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          {peakVsTrough.map((r) => (
            <div key={r.period_id} className="obsidian-subcard p-3.5 space-y-1.5">
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold text-slate-100">
                  {r.period_id === "Peak_Suppression_Days_0_2"
                    ? "Peak Suppression (Days 0–2)"
                    : "Late-Cycle Trough (Days 4–6)"}
                </span>
                <span className="text-[11px] font-mono text-slate-400">n={r.valid_n}</span>
              </div>
              <div className="grid grid-cols-3 gap-2 text-xs font-mono tabular-nums">
                <div>
                  <span className="text-[10px] font-sans text-slate-500 block">Protein</span>
                  <span className="text-[#34D399]">{formatNum(r.protein_g_mean, 0, "g")}</span>
                </div>
                <div>
                  <span className="text-[10px] font-sans text-slate-500 block">Clean Score</span>
                  <span className="text-[#60A5FA]">{formatNum(r.clean_eating_score, 1)}</span>
                </div>
                <div>
                  <span className="text-[10px] font-sans text-slate-500 block">Hunger / Noise</span>
                  <span className="text-[#FBBF24]">
                    {formatNum(r.hunger_mean, 1)} / {formatNum(r.food_noise_mean, 1)}
                  </span>
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Quiet/Collapsed Off-GLP-1 Durability Panel unless Off_GLP1_Transition is active */}
        <div className="obsidian-subcard p-3">
          <button
            type="button"
            onClick={() => setOffPanelOpen((v) => !v)}
            aria-expanded={offPanelOpen}
            className="w-full flex items-center justify-between text-xs font-medium text-slate-300 cursor-pointer"
            data-testid="off-glp1-toggle"
          >
            <span>
              {offGlp1Active
                ? `Off-GLP-1 Transition Active (${postGlp1Weeks.length} Post-GLP-1 Week${postGlp1Weeks.length > 1 ? "s" : ""})`
                : "Off-GLP-1 / Taper Durability Tracker (Standby — Currently On Active GLP-1)"}
            </span>
            {offPanelOpen ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
          </button>

          {offPanelOpen && (
            <div className="mt-3 space-y-2" data-testid="off-glp1-panel-body">
              {offGlp1Active ? (
                <div className="overflow-x-auto">
                  <table className="w-full text-xs text-left border-collapse">
                    <thead>
                      <tr className="border-b border-[#283042] text-slate-400">
                        <th className="py-1.5 pr-3">Post-GLP-1 Week</th>
                        <th className="py-1.5 pr-3">Trigger</th>
                        <th className="py-1.5 pr-3">Protein (Δ vs On)</th>
                        <th className="py-1.5 pr-3">Clean Score</th>
                        <th className="py-1.5 pr-3">Steps</th>
                        <th className="py-1.5">Hunger (Δ vs On)</th>
                      </tr>
                    </thead>
                    <tbody className="font-mono tabular-nums">
                      {postGlp1Weeks.map((w) => (
                        <tr key={w.period_id} className="border-b border-[#283042]/50">
                          <td className="py-1.5 pr-3 text-[#FBBF24] font-semibold">{w.period_id}</td>
                          <td className="py-1.5 pr-3 font-sans text-slate-400">{w.transition_trigger_reason}</td>
                          <td className="py-1.5 pr-3">
                            {formatNum(w.protein_g_mean, 0, "g")} ({formatSigned(w.delta_protein_vs_on_glp1, 1)})
                          </td>
                          <td className="py-1.5 pr-3">{formatNum(w.clean_eating_score, 1)}</td>
                          <td className="py-1.5 pr-3">{formatNum(w.steps_mean, 0)}</td>
                          <td className="py-1.5">
                            {formatNum(w.hunger_mean, 1)} ({formatSigned(w.delta_hunger_vs_on_glp1, 1)})
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <p className="text-xs text-slate-400">
                  No sustained taper or &gt;10-day shot gap detected. Post_GLP1_Week_1..N rows activate automatically when you taper or pause Tirzepatide.
                  Recent weekly clean-eating average across {weeklyHabits.length} weeks is tracked above.
                </p>
              )}
            </div>
          )}
        </div>
      </section>

      {/* 3. Sweet Intake -> Hunger Window Analysis */}
      <section className="obsidian-card p-5 space-y-3" data-testid="sweet-hunger-section">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div>
            <h2 className="text-sm font-semibold">Sweet Intake → Hunger &amp; Food-Noise Window Analysis</h2>
            <p className="text-xs text-slate-400">
              Evaluated strictly on logged nutrition days (unlogged days remain NaN, never zero sweets)
            </p>
          </div>
          <EpistemicPill label={sweetRows[0]?.epistemic_label || "Descriptive"} />
        </div>

        <div className="rounded-lg border border-[#FBBF24]/30 bg-[#FBBF24]/10 px-3.5 py-2.5 flex items-start gap-2 text-xs text-slate-200">
          <AlertTriangle className="w-4 h-4 text-[#FBBF24] shrink-0 mt-0.5" />
          <span>
            <strong>Reverse-Causality Caution:</strong> Elevated prior hunger, cravings, or sleep debt (&lt;6h / post-02:00 bedtime) frequently precede sweet intake rather than sweets causing hunger.
            {sweetRows[0]?.pct_sweet_preceded_by_hunger_or_sleep_debt != null
              ? ` Preceding hunger/sleep debt occurred on ${formatNum(sweetRows[0].pct_sweet_preceded_by_hunger_or_sleep_debt, 1, "%")} of sweet days.`
              : ""}
          </span>
        </div>

        <EChartBox option={sweetChartOption} height={220} isEmpty={sweetRows.length === 0} />

        <StatsCoverageDrawer
          meta={{
            raw_n: sweetRows[0]?.raw_n,
            valid_n: sweetRows[0]?.valid_n,
            effective_n: sweetRows[0]?.effective_n,
            missingness_pct: sweetRows[0]?.missingness_pct,
            maturity_level: sweetRows[0]?.maturity_level || "Level 0 (Baseline)",
            stability_flag: sweetRows[0]?.stability_flag || "Descriptive_Only",
            caveat_note: sweetRows[0]?.caveat_note,
          }}
        />
      </section>

      {/* 4. BMR/TDEE Calorie Deficit vs. KGs Lost Audit */}
      <section className="obsidian-card p-5 space-y-3" data-testid="calorie-audit-section">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div>
            <h2 className="text-sm font-semibold">BMR / TDEE Calorie Deficit vs. KGs Lost Audit</h2>
            <p className="text-xs text-slate-400">
              Fitdays Katch-McArdle FFM BMR + Fitbit TDEE vs. fasted weight change across 14d / 28d / interval windows
            </p>
          </div>
          <EpistemicPill label="Descriptive" />
        </div>

        <EChartBox option={calAuditOption} height={220} isEmpty={calAuditRows.length === 0} />

        <div className="overflow-x-auto">
          <table className="w-full text-xs text-left border-collapse">
            <thead>
              <tr className="border-b border-[#283042] text-slate-400">
                <th className="py-2 pr-3">Window</th>
                <th className="py-2 pr-3">BMR / TDEE</th>
                <th className="py-2 pr-3">Logged kcal</th>
                <th className="py-2 pr-3">Exp vs Actual</th>
                <th className="py-2 pr-3">Discrepancy</th>
                <th className="py-2">Classification</th>
              </tr>
            </thead>
            <tbody className="font-mono tabular-nums">
              {calAuditRows.slice(-6).map((row) => (
                <tr key={row.window_id} className="border-b border-[#283042]/60 text-slate-200">
                  <td className="py-2 pr-3">
                    {row.window_id}
                    <span className="block text-[10px] font-sans text-slate-500">
                      {row.window_days}d · cov {formatPct(row.nutrition_coverage_pct, true)}
                    </span>
                  </td>
                  <td className="py-2 pr-3">
                    {formatNum(row.bmr_kcal_mean, 0)} / {formatNum(row.tdee_est_kcal_mean, 0)}
                  </td>
                  <td className="py-2 pr-3">{formatNum(row.logged_kcal_mean, 0)}</td>
                  <td className="py-2 pr-3">
                    {formatSigned(row.expected_delta_kg, 2)} vs {formatSigned(row.actual_delta_kg, 2)} kg
                  </td>
                  <td className="py-2 pr-3">{formatSigned(row.discrepancy_kcal_per_day, 0, " kcal/d")}</td>
                  <td className="py-2 font-sans">
                    <span
                      className={`inline-block px-2 py-0.5 rounded border text-[11px] ${
                        AUDIT_BADGE[row.audit_classification] || AUDIT_BADGE.Unclear_Water_Cycle_Or_Log_Gap
                      }`}
                    >
                      {String(row.audit_classification || "Unclear").replace(/_/g, " ")}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <StatsCoverageDrawer
          meta={{
            raw_n: calAuditRows.length,
            valid_n: calAuditRows.filter((r) => r.audit_classification !== "Unclear_Water_Cycle_Or_Log_Gap").length,
            effective_n: calAuditRows.length,
            missingness_pct: 0,
            maturity_level: calAuditRows[0]?.maturity_level || "Level 0 (Descriptive Ledger)",
            stability_flag: "Multi_Week_BMR_Audit",
            caveat_note:
              "Short windows (<14d) or non-fasted weigh-ins are classified as Unclear_Water_Cycle_Or_Log_Gap rather than logging error.",
          }}
        />
      </section>

      {/* 5. Home-Cooked Meal Bioavailability & Interactive Target Simulator */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <section className="obsidian-card p-5 space-y-3">
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-semibold">Recent Home-Cooked Meal &amp; Bioavailability Audit</h2>
            <EpistemicPill label="Descriptive" />
          </div>
          <div className="space-y-2 text-xs">
            {mealBioRows.slice(-4).map((m) => (
              <div key={m.date} className="obsidian-subcard p-3 space-y-1">
                <div className="flex items-center justify-between">
                  <span className="font-mono text-slate-200">{formatDateIN(m.display_date || m.date)}</span>
                  <span className="text-[11px] text-[#34D399]">{m.bioavailability_status}</span>
                </div>
                <p className="text-slate-300">{m.fat_cofactor_assessment}</p>
                <p className="text-slate-400">{m.protein_fiber_tweaks}</p>
              </div>
            ))}
          </div>
        </section>

        <section className="obsidian-card p-5 space-y-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Sliders className="w-4 h-4 text-[#A78BFA]" />
              <h2 className="text-sm font-semibold">Target Weight &amp; Lean-Mass Simulator</h2>
            </div>
            <EpistemicPill label="Prediction" />
          </div>
          <form onSubmit={handleRunSimulator} className="space-y-3 text-xs">
            <div className="grid grid-cols-2 gap-3">
              <label className="block">
                <span className="text-slate-400 block mb-1">Target Weight ({targetKg} kg)</span>
                <input
                  type="range"
                  min={75}
                  max={105}
                  step={0.5}
                  value={targetKg}
                  onChange={(e) => setTargetKg(Number(e.target.value))}
                  className="w-full accent-[#C06C9F]"
                />
              </label>
              <label className="block">
                <span className="text-slate-400 block mb-1">Daily Protein ({proteinSlider} g)</span>
                <input
                  type="range"
                  min={70}
                  max={130}
                  step={5}
                  value={proteinSlider}
                  onChange={(e) => setProteinSlider(Number(e.target.value))}
                  className="w-full accent-[#34D399]"
                />
              </label>
              <label className="block">
                <span className="text-slate-400 block mb-1">Daily Steps ({stepsSlider})</span>
                <input
                  type="range"
                  min={4000}
                  max={12000}
                  step={500}
                  value={stepsSlider}
                  onChange={(e) => setStepsSlider(Number(e.target.value))}
                  className="w-full accent-[#60A5FA]"
                />
              </label>
              <label className="block">
                <span className="text-slate-400 block mb-1">Target Bedtime (IST)</span>
                <select
                  value={bedtimeSlider}
                  onChange={(e) => setBedtimeSlider(e.target.value)}
                  className="w-full rounded bg-[#0B0D12] border border-[#283042] px-2 py-1.5 text-slate-200"
                >
                  <option value="23:30">23:30 IST</option>
                  <option value="00:00">00:00 IST</option>
                  <option value="00:30">00:30 IST</option>
                  <option value="01:30">01:30 IST</option>
                </select>
              </label>
            </div>
            <button
              type="submit"
              disabled={simLoading}
              className="px-4 py-2 rounded-lg bg-[#A78BFA] text-[#0B0D12] font-semibold text-xs hover:opacity-90 cursor-pointer"
            >
              {simLoading ? "Simulating..." : "Run Lean-Mass Guardrail Advisor"}
            </button>
          </form>

          {simResult && (
            <div className="obsidian-subcard p-3 space-y-1.5 text-xs">
              <p className="text-slate-200">{simResult.what_your_data_shows}</p>
              <p className="text-slate-400">{simResult.realistic_timeline_and_lean_mass_guardrail}</p>
              <p className="text-[11px] text-[#FB7185]">{simResult.ask_your_doctor_note}</p>
            </div>
          )}
        </section>
      </div>
    </div>
  );
};
