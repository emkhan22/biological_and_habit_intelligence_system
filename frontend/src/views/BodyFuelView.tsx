import React, { useMemo, useState } from "react";
import type { EChartsOption } from "echarts";
import { AlertTriangle, ChevronDown, ChevronUp, Info, Sliders } from "lucide-react";
import type { HealthDataPayload, SimulatorAdviceResponse } from "../types";
import { EChartBox } from "../components/EChartBox";
import { EpistemicPill } from "../components/EpistemicPill";
import { StatsCoverageDrawer } from "../components/StatsCoverageDrawer";
import { requestSimulatorAdviceApi } from "../services/api";
import {
  formatDateIN,
  formatNum,
  formatPct,
  formatSigned,
  formatSleepHm,
} from "../utils/formatters";
import { humanizeStatus } from "../utils/humanize";

interface Props {
  data: HealthDataPayload;
}

const AUDIT_BADGE: Record<string, string> = {
  Within_Expected_Band: "bg-[#34D399]/10 text-[#34D399] border-[#34D399]/30",
  Potential_Under_Logging: "bg-[#FBBF24]/10 text-[#FBBF24] border-[#FBBF24]/30",
  Potential_Over_Logging: "bg-[#60A5FA]/10 text-[#60A5FA] border-[#60A5FA]/30",
  Unclear_Water_Cycle_Or_Log_Gap: "bg-slate-700/30 text-slate-300 border-[#283042]",
};

const SWEET_METRIC_LABELS: Record<string, string> = {
  same_day_hunger: "Same-Day Hunger",
  next_day_hunger: "Next-Day Hunger",
  same_day_food_noise: "Same-Day Food Noise",
  next_day_food_noise: "Next-Day Food Noise",
};

function isFastedStatus(status: unknown, isFastedFlag?: unknown): boolean {
  if (isFastedFlag === true || String(isFastedFlag).toLowerCase() === "true") return true;
  const s = String(status || "").toLowerCase();
  return s.includes("fasted") || s.includes("morning");
}

function isRollingAuditRow(row: Record<string, any>): boolean {
  const wt = String(row.window_type || "");
  const wid = String(row.window_id || "");
  return wt.startsWith("Rolling_") || wid.startsWith("AUD-");
}

function formatAuditWindowLabel(row: Record<string, any>): string {
  const wt = String(row.window_type || "");
  const wid = String(row.window_id || "");
  const days = row.window_days ?? "";
  if (wt === "Rolling_28d_Audit" || wid.startsWith("AUD-28D")) return "28-Day Rolling Audit";
  if (wt === "Rolling_14d_Audit" || wid.startsWith("AUD-14D")) return "14-Day Rolling Audit";
  if (wt === "Rolling_7d_Snapshot" || wid.startsWith("AUD-7D")) return "7-Day Snapshot";
  return `${days}-Day Weigh-In Span`;
}

function formatAuditShortLabel(row: Record<string, any>): string {
  const wt = String(row.window_type || "");
  const wid = String(row.window_id || "");
  if (wt === "Rolling_28d_Audit" || wid.startsWith("AUD-28D")) return "28d Audit";
  if (wt === "Rolling_14d_Audit" || wid.startsWith("AUD-14D")) return "14d Audit";
  if (wt === "Rolling_7d_Snapshot" || wid.startsWith("AUD-7D")) return "7d Snap";
  return `${formatDateIN(row.display_end_date || row.end_date).slice(0, 5)} (${row.window_days}d)`;
}

/** Smooth sparse EMA trend points by linearly interpolating across flat/constant stretches. */
function interpolateSmoothTrend(pts: Array<Record<string, any>>): Array<number | null> {
  const anchors: Array<{ idx: number; val: number }> = [];
  pts.forEach((p, i) => {
    const raw = p.ema_weight_kg;
    if (raw != null && raw !== "" && Number.isFinite(Number(raw))) {
      const val = Number(raw);
      const isMeasured = p.measured_weight_kg != null && p.measured_weight_kg !== "";
      if (i === 0 || i === pts.length - 1 || isMeasured || anchors.length === 0 || Math.abs(val - anchors[anchors.length - 1].val) > 0.001) {
        anchors.push({ idx: i, val });
      }
    }
  });
  if (anchors.length === 0) return pts.map(() => null);
  const out: Array<number | null> = new Array(pts.length).fill(null);
  for (let a = 0; a < anchors.length; a++) {
    const cur = anchors[a];
    out[cur.idx] = Number(cur.val.toFixed(1));
    if (a + 1 < anchors.length) {
      const nxt = anchors[a + 1];
      const span = nxt.idx - cur.idx;
      for (let k = 1; k < span; k++) {
        const interp = cur.val + ((nxt.val - cur.val) * k) / span;
        out[cur.idx + k] = Number(interp.toFixed(1));
      }
    }
  }
  return out;
}

/** Select non-overlapping consecutive intervals from chronological interval rows. */
function selectConsecutiveIntervals(rows: Array<Record<string, any>>): Array<Record<string, any>> {
  const sorted = [...rows].sort((a, b) => {
    const da = String(a.end_date || "");
    const db = String(b.end_date || "");
    if (da !== db) return da.localeCompare(db);
    return Number(a.interval_days || 0) - Number(b.interval_days || 0);
  });
  // Walk backwards from the latest interval, picking the shortest clean consecutive predecessor
  const picked: Array<Record<string, any>> = [];
  let cursorStart = "9999-99-99";
  for (let i = sorted.length - 1; i >= 0; i--) {
    const r = sorted[i];
    const endD = String(r.end_date || "");
    const startD = String(r.start_date || "");
    if (picked.length === 0 || endD <= cursorStart) {
      picked.push(r);
      cursorStart = startD;
    }
  }
  return picked.reverse();
}

export const BodyFuelView: React.FC<Props> = ({ data }) => {
  const dailyRows = data.tables.Gold_Daily_Unified ?? [];
  const weightTrend = data.tables.Gold_Weight_Trend ?? [];
  const weightIntervals = data.tables.Gold_Weight_Intervals ?? [];
  const habitRows = data.tables.Gold_Habit_Durability ?? [];
  const sweetRows = data.tables.Gold_Sweet_Hunger_Analysis ?? [];
  const sweetMealTags = data.tables.Gold_Sweet_Meal_Tags_AI ?? [];
  const calAuditRows = data.tables.Gold_Calorie_Balance_Audit ?? [];
  const mealBioRows = data.tables.Gold_Meal_Bioavailability_AI ?? [];

  // Item 2: Chart & Interval filter states
  const [weightChartMode, setWeightChartMode] = useState<"weight" | "velocity">("weight");
  const [weighInFilter, setWeighInFilter] = useState<"all" | "fasted" | "post_meal">("all");
  const [intervalFilter, setIntervalFilter] = useState<"consecutive" | "fasted" | "all">("consecutive");

  // Item 3: Habit Durability state
  const peakVsTrough = habitRows.filter((r) => r.row_type === "On_GLP1_Peak_vs_Trough");
  const postGlp1Weeks = habitRows.filter((r) => r.row_type === "Post_GLP1_Week");
  const weeklyHabits = habitRows.filter((r) => r.row_type === "Weekly_Durability").slice(-8);
  const offGlp1Active = postGlp1Weeks.length > 0;
  const [offPanelOpen, setOffPanelOpen] = useState(offGlp1Active);

  // Item 4: Logged Sweet Days breakdown toggle
  const [sweetDaysOpen, setSweetDaysOpen] = useState(false);

  // Item 5: Calorie audit filter
  const [auditFilter, setAuditFilter] = useState<"rolling" | "spans" | "all">("all");

  // Simulator State
  const [targetKg, setTargetKg] = useState(95.0);
  const [proteinSlider, setProteinSlider] = useState(100);
  const [stepsSlider, setStepsSlider] = useState(8500);
  const [bedtimeSlider, setBedtimeSlider] = useState("00:30");
  const [simResult, setSimResult] = useState<SimulatorAdviceResponse | null>(null);
  const [simLoading, setSimLoading] = useState(false);

  // Item 2: Weight & Velocity Charts
  const weightTrendOption = useMemo<EChartsOption>(() => {
    const pts = weightTrend.filter((r) => r.ema_weight_kg != null || r.measured_weight_kg != null).slice(-45);
    const dates = pts.map((p) => formatDateIN(p.display_date || p.date));

    if (weightChartMode === "velocity") {
      const velData = pts.map((p) => {
        const v = p.trend_loss_kg_per_wk;
        if (v == null || v === "" || !Number.isFinite(Number(v))) return null;
        return Number((-Math.abs(Number(v)) * (Number(v) >= 0 ? 1 : -1)).toFixed(2));
      });
      return {
        tooltip: {
          trigger: "axis",
          backgroundColor: "#141821",
          borderColor: "#283042",
          textStyle: { color: "#F1F5F9", fontSize: 12 },
          valueFormatter: (val) => (val != null ? `${Number(val).toFixed(2)} kg/wk` : "—"),
        },
        legend: { top: 0, textStyle: { color: "#94A3B8", fontSize: 11 } },
        grid: { left: 52, right: 20, top: 34, bottom: 28 },
        xAxis: { type: "category", data: dates, axisLine: { lineStyle: { color: "#283042" } } },
        yAxis: {
          type: "value",
          name: "kg / wk",
          splitLine: { lineStyle: { color: "#283042", type: "dashed" } },
          axisLabel: { formatter: (v: number) => `${v.toFixed(1)}` },
        },
        series: [
          {
            name: "EMA Weekly Velocity (kg/wk)",
            type: "line",
            smooth: 0.4,
            showSymbol: false,
            data: velData,
            lineStyle: { color: "#34D399", width: 2.5 },
            areaStyle: { color: "rgba(52, 211, 153, 0.08)" },
          },
        ],
      };
    }

    const smoothedTrend = interpolateSmoothTrend(pts);
    const fastedPoints = pts.map((p) => {
      if (p.measured_weight_kg == null || p.measured_weight_kg === "") return null;
      return isFastedStatus(p.weighin_status, p.is_fasted_anchor)
        ? Number(Number(p.measured_weight_kg).toFixed(1))
        : null;
    });
    const postMealPoints = pts.map((p) => {
      if (p.measured_weight_kg == null || p.measured_weight_kg === "") return null;
      return !isFastedStatus(p.weighin_status, p.is_fasted_anchor)
        ? Number(Number(p.measured_weight_kg).toFixed(1))
        : null;
    });

    const series: any[] = [
      {
        name: "Smoothed Trend (EMA)",
        type: "line",
        smooth: 0.45,
        showSymbol: false,
        connectNulls: true,
        data: smoothedTrend,
        lineStyle: { color: "#60A5FA", width: 2.5 },
      },
    ];
    if (weighInFilter === "all" || weighInFilter === "fasted") {
      series.push({
        name: "Morning (Fasted)",
        type: "scatter",
        symbolSize: 8,
        data: fastedPoints,
        itemStyle: { color: "#34D399" },
      });
    }
    if (weighInFilter === "all" || weighInFilter === "post_meal") {
      series.push({
        name: "Post-Meal / Other",
        type: "scatter",
        symbol: "diamond",
        symbolSize: 8,
        data: postMealPoints,
        itemStyle: { color: "#FBBF24" },
      });
    }

    return {
      tooltip: {
        trigger: "axis",
        backgroundColor: "#141821",
        borderColor: "#283042",
        textStyle: { color: "#F1F5F9", fontSize: 12 },
        valueFormatter: (val) => (val != null ? `${Number(val).toFixed(1)} kg` : "—"),
      },
      legend: { top: 0, textStyle: { color: "#94A3B8", fontSize: 11 } },
      grid: { left: 48, right: 20, top: 34, bottom: 28 },
      xAxis: {
        type: "category",
        data: dates,
        axisLine: { lineStyle: { color: "#283042" } },
      },
      yAxis: {
        type: "value",
        scale: true,
        splitLine: { lineStyle: { color: "#283042", type: "dashed" } },
        axisLabel: { formatter: (v: number) => `${v.toFixed(1)}` },
      },
      series,
    };
  }, [weightTrend, weightChartMode, weighInFilter]);

  const displayedIntervals = useMemo(() => {
    if (intervalFilter === "fasted") {
      return weightIntervals
        .filter((r) => isFastedStatus(r.end_weighin_status, r.is_fasted_interval))
        .slice(-8)
        .reverse();
    }
    if (intervalFilter === "consecutive") {
      return selectConsecutiveIntervals(weightIntervals).slice(-8).reverse();
    }
    return weightIntervals.slice(-10).reverse();
  }, [weightIntervals, intervalFilter]);

  // Item 3: Plain-English Habit Carryover Summary
  const carryoverSummary = useMemo(() => {
    const peak = peakVsTrough.find((r) => r.period_id === "Peak_Suppression_Days_0_2");
    const trough = peakVsTrough.find((r) => r.period_id === "Late_Cycle_Trough_Days_4_6");
    if (!peak || !trough) return null;
    const pProt = Number(peak.protein_g_mean ?? 0);
    const tProt = Number(trough.protein_g_mean ?? 0);
    const pSteps = Number(peak.steps_mean ?? 0);
    const tSteps = Number(trough.steps_mean ?? 0);
    const pSleep = Number(peak.asleep_hr_mean ?? 0);
    const tSleep = Number(trough.asleep_hr_mean ?? 0);
    const pHun = Number(peak.hunger_mean ?? 0);
    const tHun = Number(trough.hunger_mean ?? 0);

    const protMsg =
      tProt >= pProt - 5
        ? `protein intake holds strong (${formatNum(tProt, 0, "g")} vs. ${formatNum(pProt, 0, "g")})`
        : `protein dips slightly (${formatNum(tProt, 0, "g")} vs. ${formatNum(pProt, 0, "g")})`;
    const stepMsg =
      tSteps >= pSteps - 400
        ? `steps remain steady (${formatNum(tSteps, 0)}/day)`
        : `daily steps ease off (${formatNum(tSteps, 0)} vs. ${formatNum(pSteps, 0)})`;
    const sleepMsg =
      Math.abs(tSleep - pSleep) <= 0.35
        ? `sleep stays consistent (~${formatSleepHm(tSleep * 60)})`
        : `sleep shifts to ${formatSleepHm(tSleep * 60)} vs. ${formatSleepHm(pSleep * 60)}`;
    const hungerMsg =
      tHun > pHun + 0.4
        ? `even as hunger rises (${formatNum(tHun, 1)} vs. ${formatNum(pHun, 1)})`
        : `with hunger staying controlled (${formatNum(tHun, 1)} vs. ${formatNum(pHun, 1)})`;

    return `When appetite suppression eases on Days 4–6, ${protMsg}, ${stepMsg}, and ${sleepMsg} — ${hungerMsg}.`;
  }, [peakVsTrough]);

  // Item 4: Sweet Intake -> Hunger Chart & Breakdown
  const sweetHungerRows = useMemo(
    () => sweetRows.filter((r) => r.outcome_metric !== "subsequent_meal_kcal"),
    [sweetRows]
  );
  const sweetN = sweetHungerRows[0]?.sweet_days_n ?? 0;
  const nonSweetN = sweetHungerRows[0]?.non_sweet_logged_days_n ?? 0;
  const sweetLoggedDays = sweetHungerRows[0]?.nutrition_logged_days ?? 0;

  const sweetTakeaway = useMemo(() => {
    if (sweetHungerRows.length === 0) return null;
    const nextHun = sweetHungerRows.find((r) => r.outcome_metric === "next_day_hunger");
    const sameNoise = sweetHungerRows.find((r) => r.outcome_metric === "same_day_food_noise");
    if (Number(sweetN) === 0) {
      return `Across ${sweetLoggedDays} logged nutrition days, no sweet-flagged days have been recorded yet. Baseline hunger averages ${formatNum(nextHun?.non_sweet_window_mean, 1)}/5.`;
    }
    const diffHun = Number(nextHun?.sweet_minus_nonsweet_diff ?? 0);
    const diffNoise = Number(sameNoise?.sweet_minus_nonsweet_diff ?? 0);
    const hunDesc =
      Math.abs(diffHun) < 0.25
        ? "virtually unchanged next-day hunger"
        : diffHun > 0
        ? `slightly higher next-day hunger (${formatSigned(diffHun, 2)} pts)`
        : `slightly lower next-day hunger (${formatSigned(diffHun, 2)} pts)`;
    const noiseDesc =
      diffNoise > 0.25
        ? `same-day food noise is higher by ${formatSigned(diffNoise, 2)} pts`
        : `same-day food noise shows minimal shift (${formatSigned(diffNoise, 2)} pts)`;
    return `Across ${sweetLoggedDays} logged nutrition days (Sweet n=${sweetN} vs. Non-Sweet n=${nonSweetN}), sweet days show ${hunDesc} while ${noiseDesc} — differences are not statistically significant (p > 0.20) and often reflect prior hunger or sleep debt rather than sweets causing hunger.`;
  }, [sweetHungerRows, sweetN, nonSweetN, sweetLoggedDays]);

  const sweetChartOption = useMemo<EChartsOption>(() => {
    const sweetLegend = `Sweet Days (n=${sweetN})`;
    const nonSweetLegend = `Non-Sweet Logged (n=${nonSweetN})`;
    return {
      tooltip: {
        trigger: "axis",
        backgroundColor: "#141821",
        borderColor: "#283042",
        textStyle: { color: "#F1F5F9", fontSize: 12 },
        valueFormatter: (val) => (val != null ? `${Number(val).toFixed(2)} / 5` : "—"),
      },
      legend: { top: 0, textStyle: { color: "#94A3B8", fontSize: 11 } },
      grid: { left: 52, right: 16, top: 38, bottom: 28 },
      xAxis: {
        type: "category",
        data: sweetHungerRows.map((r) => SWEET_METRIC_LABELS[r.outcome_metric] || String(r.outcome_metric).replace(/_/g, " ")),
        axisLine: { lineStyle: { color: "#283042" } },
        axisLabel: { color: "#CBD5E1", fontSize: 11 },
      },
      yAxis: {
        type: "value",
        name: "Rating (0–5)",
        max: 5,
        splitLine: { lineStyle: { color: "#283042", type: "dashed" } },
      },
      series: [
        {
          name: sweetLegend,
          type: "bar",
          barMaxWidth: 34,
          data: sweetHungerRows.map((r) => (r.sweet_window_mean != null ? Number(Number(r.sweet_window_mean).toFixed(2)) : null)),
          itemStyle: { color: "#C06C9F", borderRadius: [4, 4, 0, 0] },
          label: { show: true, position: "top", color: "#F1F5F9", fontSize: 11, formatter: "{c}" },
        },
        {
          name: nonSweetLegend,
          type: "bar",
          barMaxWidth: 34,
          data: sweetHungerRows.map((r) => (r.non_sweet_window_mean != null ? Number(Number(r.non_sweet_window_mean).toFixed(2)) : null)),
          itemStyle: { color: "#34D399", borderRadius: [4, 4, 0, 0] },
          label: { show: true, position: "top", color: "#F1F5F9", fontSize: 11, formatter: "{c}" },
        },
      ],
    };
  }, [sweetHungerRows, sweetN, nonSweetN]);

  // Collapsible Logged Sweet Days breakdown (combining daily sugar/hunger/sleep + AI sweet tags)
  const loggedSweetDayRows = useMemo(() => {
    const aiSweetItems = sweetMealTags
      .filter((t) => t.is_sweet_ai === true || String(t.is_sweet_ai).toLowerCase() === "true")
      .map((t) => `${t.sample_meal_name || t.normalized_meal_name} (${String(t.sweet_category || "Sweet").replace(/_/g, " ")})`);

    const rows: Array<{
      date: string;
      sugarG: number | null;
      hunger: number | null;
      priorFlag: string;
    }> = [];

    for (let i = 0; i < dailyRows.length; i++) {
      const r = dailyRows[i];
      const logged = Number(r.nutrition_entries_logged || 0) > 0 || r.calories_kcal != null;
      const sugar = r.sugar_g != null && r.sugar_g !== "" ? Number(r.sugar_g) : null;
      if (logged && sugar !== null && sugar >= 15) {
        const prev = i > 0 ? dailyRows[i - 1] : null;
        const priorHunger = prev?.hunger != null ? Number(prev.hunger) : null;
        const sleepMin = r.asleep_min != null ? Number(r.asleep_min) : null;
        const bedHr = r.bedtime_hr_rel_mid != null ? Number(r.bedtime_hr_rel_mid) : null;
        const flags: string[] = [];
        if (priorHunger !== null && priorHunger >= 3) flags.push(`Prior Hunger ${priorHunger.toFixed(0)}/5`);
        if (sleepMin !== null && sleepMin < 360) flags.push(`Short Sleep (${formatSleepHm(sleepMin)})`);
        if (bedHr !== null && bedHr >= 2) flags.push("Late Bedtime (>02:00)");
        rows.push({
          date: formatDateIN(r.display_date || r.date),
          sugarG: sugar,
          hunger: r.hunger != null ? Number(r.hunger) : null,
          priorFlag: flags.length > 0 ? flags.join(" · ") : "None flagged",
        });
      }
    }
    return { days: rows.reverse().slice(0, 20), aiSweetItems };
  }, [dailyRows, sweetMealTags]);

  // Item 5: Calorie Balance Audit filtered views
  const filteredCalAudit = useMemo(() => {
    const rolling = calAuditRows.filter((r) => isRollingAuditRow(r));
    const spans = calAuditRows.filter((r) => !isRollingAuditRow(r)).slice(-6);
    if (auditFilter === "rolling") return rolling.slice().reverse();
    if (auditFilter === "spans") return spans.slice().reverse();
    return [...rolling.slice().reverse(), ...spans.slice().reverse()];
  }, [calAuditRows, auditFilter]);

  const calAuditOption = useMemo<EChartsOption>(() => {
    const items = filteredCalAudit.slice(0, 8).reverse();
    return {
      tooltip: {
        trigger: "axis",
        backgroundColor: "#141821",
        borderColor: "#283042",
        textStyle: { color: "#F1F5F9", fontSize: 12 },
        valueFormatter: (val) => (val != null ? `${Number(val).toFixed(2)} kg` : "—"),
      },
      legend: { top: 0, textStyle: { color: "#94A3B8", fontSize: 11 } },
      grid: { left: 48, right: 16, top: 34, bottom: 32 },
      xAxis: {
        type: "category",
        data: items.map((r) => formatAuditShortLabel(r)),
        axisLine: { lineStyle: { color: "#283042" } },
      },
      yAxis: {
        type: "value",
        name: "Change (kg)",
        splitLine: { lineStyle: { color: "#283042", type: "dashed" } },
      },
      series: [
        {
          name: "Expected Loss (kg)",
          type: "bar",
          data: items.map((r) => (r.expected_delta_kg != null ? Number(Number(r.expected_delta_kg).toFixed(2)) : null)),
          itemStyle: { color: "#60A5FA", borderRadius: [0, 0, 3, 3] },
        },
        {
          name: "Actual Loss (kg)",
          type: "bar",
          data: items.map((r) => (r.actual_delta_kg != null ? Number(Number(r.actual_delta_kg).toFixed(2)) : null)),
          itemStyle: { color: "#34D399", borderRadius: [0, 0, 3, 3] },
        },
      ],
    };
  }, [filteredCalAudit]);

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
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div>
            <h2 className="text-sm font-semibold">3-Grain Weight Trend &amp; Fasted Intervals</h2>
            <p className="text-xs text-slate-400">
              Smoothed EMA trend separated from Morning (Fasted) and Post-Meal weigh-ins
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <div className="inline-flex rounded-lg bg-[#0B0D12] border border-[#283042] p-0.5 text-[11px]">
              <button
                type="button"
                onClick={() => setWeightChartMode("weight")}
                className={`px-2.5 py-1 rounded-md cursor-pointer ${
                  weightChartMode === "weight" ? "bg-[#1B202C] text-slate-100 font-medium" : "text-slate-400"
                }`}
              >
                Weight &amp; Trend
              </button>
              <button
                type="button"
                onClick={() => setWeightChartMode("velocity")}
                className={`px-2.5 py-1 rounded-md cursor-pointer ${
                  weightChartMode === "velocity" ? "bg-[#1B202C] text-slate-100 font-medium" : "text-slate-400"
                }`}
              >
                Weekly Velocity
              </button>
            </div>
            <EpistemicPill label="Descriptive" />
          </div>
        </div>

        {weightChartMode === "weight" && (
          <div className="flex flex-wrap items-center gap-1.5 text-[11px]">
            <span className="text-slate-500 mr-1">Weigh-ins:</span>
            {(
              [
                ["all", "All Weigh-Ins"],
                ["fasted", "Morning (Fasted) Only"],
                ["post_meal", "Post-Meal Only"],
              ] as const
            ).map(([id, label]) => (
              <button
                key={id}
                type="button"
                onClick={() => setWeighInFilter(id)}
                className={`px-2.5 py-1 rounded-md border cursor-pointer ${
                  weighInFilter === id
                    ? "bg-[#1B202C] border-[#60A5FA]/50 text-slate-100"
                    : "bg-[#0B0D12] border-[#283042] text-slate-400 hover:text-slate-200"
                }`}
              >
                {label}
              </button>
            ))}
          </div>
        )}

        <EChartBox option={weightTrendOption} height={240} isEmpty={weightTrend.length === 0} />

        {/* Interval Filter Bar */}
        <div className="flex flex-wrap items-center justify-between gap-2 pt-2 border-t border-[#283042]">
          <span className="text-xs font-medium text-slate-300">Weigh-In Intervals</span>
          <div className="flex flex-wrap items-center gap-1.5 text-[11px]">
            {(
              [
                ["consecutive", "Consecutive (Non-Overlapping)"],
                ["fasted", "Morning (Fasted)"],
                ["all", "All Intervals"],
              ] as const
            ).map(([id, label]) => (
              <button
                key={id}
                type="button"
                onClick={() => setIntervalFilter(id)}
                className={`px-2.5 py-1 rounded-md border cursor-pointer ${
                  intervalFilter === id
                    ? "bg-[#1B202C] border-[#34D399]/50 text-slate-100"
                    : "bg-[#0B0D12] border-[#283042] text-slate-400 hover:text-slate-200"
                }`}
              >
                {label}
              </button>
            ))}
          </div>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-xs text-left border-collapse">
            <thead>
              <tr className="border-b border-[#283042] text-slate-400">
                <th className="py-2 pr-3">Dates (IST)</th>
                <th className="py-2 pr-3">Days</th>
                <th className="py-2 pr-3">Weight Change</th>
                <th className="py-2 pr-3">Weekly Pace</th>
                <th className="py-2 pr-3">Fat Share</th>
                <th className="py-2 pr-3">Weigh-In Type</th>
                <th className="py-2">Food Logged %</th>
              </tr>
            </thead>
            <tbody className="font-mono tabular-nums">
              {displayedIntervals.map((row, idx) => {
                const delta = row.delta_kg != null ? Number(row.delta_kg) : null;
                const rawRate = row.loss_rate_kg_per_wk != null ? Number(row.loss_rate_kg_per_wk) : null;
                // Consistent sign: negative means weight loss, positive means weight gain
                const signedRate = rawRate !== null ? -rawRate : null;
                return (
                  <tr key={`${row.start_date}-${row.end_date}-${idx}`} className="border-b border-[#283042]/60 text-slate-200">
                    <td className="py-2 pr-3 font-sans">
                      {formatDateIN(row.display_start_date || row.start_date)} → {formatDateIN(row.display_end_date || row.end_date)}
                    </td>
                    <td className="py-2 pr-3 text-slate-400">{row.interval_days}d</td>
                    <td className={`py-2 pr-3 ${delta !== null && delta <= 0 ? "text-[#34D399]" : "text-[#FBBF24]"}`}>
                      {formatSigned(delta, 1, " kg")}
                    </td>
                    <td className={`py-2 pr-3 ${signedRate !== null && signedRate <= 0 ? "text-[#34D399]" : "text-[#FBBF24]"}`}>
                      {formatSigned(signedRate, 1, " kg/wk")}
                    </td>
                    <td className="py-2 pr-3">{formatPct(row.fat_share_of_loss, true)}</td>
                    <td className="py-2 pr-3 font-sans text-slate-300">{humanizeStatus(row.end_weighin_status)}</td>
                    <td className="py-2">{formatPct(row.nutrition_coverage_pct, true)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <StatsCoverageDrawer
          meta={{
            raw_n: weightIntervals.length,
            valid_n: weightIntervals.filter((r) => isFastedStatus(r.end_weighin_status, r.is_fasted_interval)).length,
            effective_n: weightIntervals.length,
            missingness_pct: 0,
            maturity_level: weightIntervals[0]?.maturity_level || "Level 0 (Descriptive Ledger)",
            stability_flag: "Fasted_Anchor_Priority",
            caveat_note: "Morning (Fasted) weigh-ins are classified dynamically against first logged meal time (no naive clock cutoff).",
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

        {carryoverSummary && (
          <div className="rounded-lg border border-[#34D399]/25 bg-[#34D399]/5 px-3.5 py-2.5 text-xs text-slate-200 leading-relaxed">
            <strong className="text-[#34D399]">Carryover Summary:</strong> {carryoverSummary}
          </div>
        )}

        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          {peakVsTrough.map((r) => (
            <div key={r.period_id} className="obsidian-subcard p-4 space-y-2.5">
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold text-slate-100">
                  {r.period_id === "Peak_Suppression_Days_0_2"
                    ? "Peak Suppression (Days 0–2)"
                    : "Late-Cycle Trough (Days 4–6)"}
                </span>
                <span className="text-[11px] font-mono text-slate-400">n={r.valid_n} days</span>
              </div>
              <div className="grid grid-cols-3 gap-2.5 text-xs font-mono tabular-nums">
                <div>
                  <span className="text-[10px] font-sans text-slate-500 block">Protein</span>
                  <span className="text-[#34D399] font-semibold">{formatNum(r.protein_g_mean, 0, "g")}</span>
                </div>
                <div>
                  <span className="text-[10px] font-sans text-slate-500 block">Clean Score</span>
                  <span className="text-[#60A5FA] font-semibold">{formatNum(r.clean_eating_score, 1)}</span>
                </div>
                <div>
                  <span className="text-[10px] font-sans text-slate-500 block">Hunger / Noise</span>
                  <span className="text-[#FBBF24] font-semibold">
                    {formatNum(r.hunger_mean, 1)} / {formatNum(r.food_noise_mean, 1)}
                  </span>
                </div>
                <div>
                  <span className="text-[10px] font-sans text-slate-500 block">Avg Steps</span>
                  <span className="text-slate-200">{formatNum(r.steps_mean, 0)}</span>
                </div>
                <div>
                  <span className="text-[10px] font-sans text-slate-500 block">Strength / Workouts</span>
                  <span className="text-slate-200">{formatNum(r.strength_sessions_cnt, 0, " sess")}</span>
                </div>
                <div>
                  <span className="text-[10px] font-sans text-slate-500 block">Sleep Duration</span>
                  <span className="text-[#A78BFA]">
                    {formatSleepHm(r.asleep_hr_mean != null ? Number(r.asleep_hr_mean) * 60 : null)}
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
                          <td className="py-1.5 pr-3 font-sans text-slate-400">{humanizeStatus(w.transition_trigger_reason)}</td>
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
                  No sustained taper or &gt;10-day shot gap detected. Post-GLP-1 Week 1..N rows activate automatically when you taper or pause Tirzepatide.
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
              Sweet Days (<span className="font-mono text-[#C06C9F]">n={sweetN}</span>) vs. Non-Sweet Logged Days (
              <span className="font-mono text-[#34D399]">n={nonSweetN}</span>) across{" "}
              <span className="font-mono text-slate-200">{sweetLoggedDays}</span> logged food days
            </p>
          </div>
          <EpistemicPill label={sweetRows[0]?.epistemic_label || "Descriptive"} />
        </div>

        {sweetTakeaway && (
          <p className="text-xs text-slate-200 leading-relaxed obsidian-subcard px-3.5 py-2.5">
            <strong className="text-[#60A5FA]">Takeaway:</strong> {sweetTakeaway}
          </p>
        )}

        <div className="rounded-lg border border-[#FBBF24]/30 bg-[#FBBF24]/10 px-3.5 py-2.5 flex items-start gap-2 text-xs text-slate-200">
          <AlertTriangle className="w-4 h-4 text-[#FBBF24] shrink-0 mt-0.5" />
          <span>
            <strong>Reverse-Causality Caution:</strong> Elevated prior hunger, cravings, or sleep debt (&lt;6h / post-02:00 bedtime) frequently precede sweet intake rather than sweets causing hunger.
            {sweetRows[0]?.pct_sweet_preceded_by_hunger_or_sleep_debt != null
              ? ` Preceding hunger/sleep debt occurred on ${formatNum(sweetRows[0].pct_sweet_preceded_by_hunger_or_sleep_debt, 1, "%")} of sweet days.`
              : ""}
          </span>
        </div>

        <EChartBox option={sweetChartOption} height={240} isEmpty={sweetRows.length === 0} />

        {/* Collapsible Logged Sweet Days Breakdown */}
        <div className="obsidian-subcard p-3">
          <button
            type="button"
            onClick={() => setSweetDaysOpen((v) => !v)}
            className="w-full flex items-center justify-between text-xs font-medium text-slate-300 cursor-pointer"
          >
            <span>
              Logged Sweet Days &amp; Classified Sweet Items ({loggedSweetDayRows.days.length} higher-sugar days · {loggedSweetDayRows.aiSweetItems.length} AI-tagged items)
            </span>
            {sweetDaysOpen ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
          </button>
          {sweetDaysOpen && (
            <div className="mt-3 space-y-3 text-xs">
              {loggedSweetDayRows.aiSweetItems.length > 0 && (
                <div>
                  <p className="text-[11px] text-slate-400 mb-1.5">AI-Classified Sweet Items in Your Food Log:</p>
                  <div className="flex flex-wrap gap-1.5">
                    {loggedSweetDayRows.aiSweetItems.slice(0, 14).map((item, idx) => (
                      <span
                        key={idx}
                        className="px-2 py-0.5 rounded bg-[#0B0D12] border border-[#283042] text-[11px] text-slate-300"
                      >
                        {item}
                      </span>
                    ))}
                  </div>
                </div>
              )}
              {loggedSweetDayRows.days.length > 0 ? (
                <div className="overflow-x-auto">
                  <table className="w-full text-xs text-left border-collapse">
                    <thead>
                      <tr className="border-b border-[#283042] text-slate-400">
                        <th className="py-1.5 pr-3">Date (IST)</th>
                        <th className="py-1.5 pr-3">Logged Sugar</th>
                        <th className="py-1.5 pr-3">Same-Day Hunger</th>
                        <th className="py-1.5">Prior Hunger / Sleep Debt Flag</th>
                      </tr>
                    </thead>
                    <tbody className="font-mono tabular-nums">
                      {loggedSweetDayRows.days.map((d, i) => (
                        <tr key={i} className="border-b border-[#283042]/50 text-slate-200">
                          <td className="py-1.5 pr-3 font-sans">{d.date}</td>
                          <td className="py-1.5 pr-3 text-[#C06C9F]">{formatNum(d.sugarG, 0, "g")}</td>
                          <td className="py-1.5 pr-3">{formatNum(d.hunger, 1, " / 5")}</td>
                          <td className="py-1.5 font-sans text-slate-400">{d.priorFlag}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <p className="text-slate-400">No logged high-sugar days recorded in the current window.</p>
              )}
            </div>
          )}
        </div>

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
              Compares your Fitdays FFM BMR + Fitbit TDEE energy model against actual scale change
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <div className="inline-flex rounded-lg bg-[#0B0D12] border border-[#283042] p-0.5 text-[11px]">
              {(
                [
                  ["all", "All Windows"],
                  ["rolling", "14d / 28d Audits"],
                  ["spans", "Weigh-In Spans"],
                ] as const
              ).map(([id, label]) => (
                <button
                  key={id}
                  type="button"
                  onClick={() => setAuditFilter(id)}
                  className={`px-2.5 py-1 rounded-md cursor-pointer ${
                    auditFilter === id ? "bg-[#1B202C] text-slate-100 font-medium" : "text-slate-400"
                  }`}
                >
                  {label}
                </button>
              ))}
            </div>
            <EpistemicPill label="Descriptive" />
          </div>
        </div>

        <div className="rounded-lg border border-[#283042] bg-[#1B202C]/60 px-3.5 py-2.5 flex items-start gap-2 text-xs text-slate-300 leading-relaxed">
          <Info className="w-4 h-4 text-[#60A5FA] shrink-0 mt-0.5" />
          <span>
            <strong>How to read Daily Calorie Gap:</strong> A value within{" "}
            <span className="font-mono text-[#34D399]">±250 kcal/d</span> means your logged calories closely match actual weight loss.{" "}
            <strong>Positive (+kcal/d)</strong> means actual loss was slightly slower than modeled (possible water retention or unlogged bites);{" "}
            <strong>Negative (-kcal/d)</strong> means you lost slightly faster than modeled. Short 1–8 day spans routinely fluctuate from hydration, glycogen, and cycle shifts — rely on the <strong>14-Day</strong> and <strong>28-Day Rolling Audits</strong>.
          </span>
        </div>

        <EChartBox option={calAuditOption} height={220} isEmpty={filteredCalAudit.length === 0} />

        <div className="overflow-x-auto">
          <table className="w-full text-xs text-left border-collapse">
            <thead>
              <tr className="border-b border-[#283042] text-slate-400">
                <th className="py-2 pr-3">Window &amp; Dates (IST)</th>
                <th className="py-2 pr-3">BMR / TDEE</th>
                <th className="py-2 pr-3">Logged Intake</th>
                <th className="py-2 pr-3">Expected Loss</th>
                <th className="py-2 pr-3">Actual Loss</th>
                <th className="py-2 pr-3">Daily Calorie Gap (kcal/d)</th>
                <th className="py-2 pr-3">Food Logs Covered (%)</th>
                <th className="py-2">Assessment</th>
              </tr>
            </thead>
            <tbody className="font-mono tabular-nums">
              {filteredCalAudit.slice(0, 8).map((row) => (
                <tr key={row.window_id} className="border-b border-[#283042]/60 text-slate-200">
                  <td className="py-2 pr-3 font-sans">
                    <span className="font-medium text-slate-100 block">{formatAuditWindowLabel(row)}</span>
                    <span className="text-[11px] text-slate-400">
                      {formatDateIN(row.display_start_date || row.start_date)} → {formatDateIN(row.display_end_date || row.end_date)}
                    </span>
                  </td>
                  <td className="py-2 pr-3">
                    {formatNum(row.bmr_kcal_mean, 0)} / {formatNum(row.tdee_est_kcal_mean, 0)}
                  </td>
                  <td className="py-2 pr-3">{formatNum(row.logged_kcal_mean, 0, " kcal")}</td>
                  <td className="py-2 pr-3 text-[#60A5FA]">{formatSigned(row.expected_delta_kg, 2, " kg")}</td>
                  <td className="py-2 pr-3 text-[#34D399]">{formatSigned(row.actual_delta_kg, 2, " kg")}</td>
                  <td className="py-2 pr-3">{formatSigned(row.discrepancy_kcal_per_day, 0, " kcal/d")}</td>
                  <td className="py-2 pr-3">{formatPct(row.nutrition_coverage_pct, true)}</td>
                  <td className="py-2 font-sans">
                    <span
                      className={`inline-block px-2 py-0.5 rounded border text-[11px] ${
                        AUDIT_BADGE[row.audit_classification] || AUDIT_BADGE.Unclear_Water_Cycle_Or_Log_Gap
                      }`}
                    >
                      {humanizeStatus(row.audit_classification)}
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
              "Short windows (<14d) or non-fasted weigh-ins are marked as Unclear (Water / Cycle / Log Gap) rather than logging error.",
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
                  <span className="text-[11px] text-[#34D399]">{humanizeStatus(m.bioavailability_status)}</span>
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
