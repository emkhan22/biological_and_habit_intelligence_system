import React, { useMemo } from "react";
import type { EChartsOption } from "echarts";
import { ArrowDownRight, ArrowUpRight, Info } from "lucide-react";
import type { HealthDataPayload } from "../types";
import { EChartBox } from "../components/EChartBox";
import { EpistemicPill } from "../components/EpistemicPill";
import { StatsCoverageDrawer } from "../components/StatsCoverageDrawer";
import { formatDateIN, formatNum, formatSleepHm } from "../utils/formatters";
import { formatClockHour, humanizeFeature, humanizeStatus } from "../utils/humanize";

interface Props {
  data: HealthDataPayload;
}

/** Shift hours relative to 21:00 IST so all bedtime-to-wake bars stay positive on the Y-axis. */
const CLOCK_BASE_HR = -3; // -3h relative to midnight = 21:00 IST

function normalizeDirection(dir: unknown): { label: string; isPositive: boolean } {
  const s = String(dir || "").toLowerCase();
  const isPos = s.includes("pos") || s.includes("higher") || s.includes("+");
  return {
    label: isPos ? "↑ Higher" : "↓ Lower",
    isPositive: isPos,
  };
}

export const SleepEnergyView: React.FC<Props> = ({ data }) => {
  const daily = (data.tables.Gold_Daily_Unified ?? []).slice(-21);
  const dailyPhenotypes = (data.tables.Gold_Daily_Phenotypes ?? []).slice(-7);
  const mlDrivers = data.tables.Gold_ML_Feature_Drivers ?? [];
  const mlModels = data.tables.Gold_ML_Model_Summary ?? [];
  const profiles = data.tables.Gold_Phenotype_Profiles ?? [];
  const cardiacBuckets = (data.tables.Gold_Cardiac_Efficiency_Buckets ?? []).filter(
    (r) =>
      r.aggregation_scope === "Overall" &&
      r.mean_hr_bpm != null &&
      r.mean_hr_bpm !== "" &&
      Number.isFinite(Number(r.mean_hr_bpm))
  );

  // Item 6: Nocturnal Sleep Architecture — Bedtime-to-Wake Window with Xh Ym on top + Deep/REM/Light in tooltip
  const sleepPts = useMemo(
    () => daily.filter((r) => r.asleep_min != null && r.asleep_min !== "" && Number.isFinite(Number(r.asleep_min))),
    [daily]
  );

  const sleepOption = useMemo<EChartsOption>(() => {
    const dates = sleepPts.map((p) => formatDateIN(p.display_date || p.date).slice(0, 5));
    const offsets: number[] = [];
    const durations: Array<{ value: number; raw: Record<string, any> }> = [];

    for (const p of sleepPts) {
      const asleepHr = Number(p.asleep_min) / 60;
      const bedRel =
        p.bedtime_hr_rel_mid != null && p.bedtime_hr_rel_mid !== "" && Number.isFinite(Number(p.bedtime_hr_rel_mid))
          ? Number(p.bedtime_hr_rel_mid)
          : 0.5; // fallback 00:30 if bedtime timestamp absent
      const startShifted = Math.max(0, bedRel - CLOCK_BASE_HR);
      offsets.push(Number(startShifted.toFixed(2)));
      durations.push({ value: Number(asleepHr.toFixed(2)), raw: p });
    }

    return {
      tooltip: {
        trigger: "axis",
        axisPointer: { type: "shadow" },
        backgroundColor: "#141821",
        borderColor: "#283042",
        textStyle: { color: "#F1F5F9", fontSize: 12 },
        formatter: (params: any) => {
          const list = Array.isArray(params) ? params : [params];
          const durItem = list.find((item: any) => item.seriesName === "Sleep Window") || list[0];
          const p = durItem?.data?.raw;
          if (!p) return "";
          const dateLabel = formatDateIN(p.display_date || p.date);
          const bedStr = formatClockHour(p.bedtime_hr_rel_mid);
          const wakeStr = formatClockHour(p.wake_hr);
          const totalStr = formatSleepHm(p.asleep_min);
          return [
            `<div style="font-weight:600;margin-bottom:4px">${dateLabel}</div>`,
            `<div>Window: <b>${bedStr} → ${wakeStr} IST</b></div>`,
            `<div>Total Sleep: <b style="color:#A78BFA">${totalStr}</b></div>`,
            `<div style="margin-top:4px;color:#94A3B8">Deep: <b style="color:#A78BFA">${formatNum(p.deep_min, 0, "m")}</b> · REM: <b style="color:#60A5FA">${formatNum(p.rem_min, 0, "m")}</b> · Light: <b style="color:#CBD5E1">${formatNum(p.light_min, 0, "m")}</b></div>`,
          ].join("");
        },
      },
      grid: { left: 54, right: 16, top: 32, bottom: 28 },
      xAxis: {
        type: "category",
        data: dates,
        axisLine: { lineStyle: { color: "#283042" } },
        axisLabel: { color: "#94A3B8", fontSize: 11 },
      },
      yAxis: {
        type: "value",
        min: 0,
        max: 15,
        interval: 2,
        axisLabel: {
          color: "#94A3B8",
          fontSize: 11,
          formatter: (val: number) => formatClockHour(val + CLOCK_BASE_HR),
        },
        splitLine: { lineStyle: { color: "#283042", type: "dashed" } },
      },
      series: [
        {
          name: "Bedtime Offset",
          type: "bar",
          stack: "sleepWindow",
          silent: true,
          itemStyle: { color: "transparent" },
          emphasis: { itemStyle: { color: "transparent" } },
          data: offsets,
        },
        {
          name: "Sleep Window",
          type: "bar",
          stack: "sleepWindow",
          barMaxWidth: 26,
          itemStyle: { color: "#A78BFA", borderRadius: [4, 4, 4, 4] },
          label: {
            show: true,
            position: "top",
            color: "#E2E8F0",
            fontSize: 10,
            formatter: (params: any) => {
              const mins = params?.data?.raw?.asleep_min;
              return mins != null ? formatSleepHm(mins) : "";
            },
          },
          data: durations,
        },
      ],
    };
  }, [sleepPts]);

  // Item 7: Recent 7-day phenotype mix
  const recentPhenoMix = useMemo(() => {
    const counts: Record<string, number> = {};
    for (const r of dailyPhenotypes) {
      const lbl = String(r.phenotype_label || "").trim();
      if (lbl) counts[lbl] = (counts[lbl] || 0) + 1;
    }
    return Object.entries(counts).sort((a, b) => b[1] - a[1]);
  }, [dailyPhenotypes]);

  // Item 8: Cardiac efficiency chart (valid finite buckets only)
  const cardiacOption = useMemo<EChartsOption>(() => {
    return {
      tooltip: {
        trigger: "axis",
        backgroundColor: "#141821",
        borderColor: "#283042",
        textStyle: { color: "#F1F5F9" },
        valueFormatter: (val) => (val != null ? `${Number(val).toFixed(1)} bpm` : "—"),
      },
      grid: { left: 48, right: 16, top: 24, bottom: 28 },
      xAxis: {
        type: "category",
        data: cardiacBuckets.map((b) => b.step_bucket),
        axisLine: { lineStyle: { color: "#283042" } },
      },
      yAxis: {
        type: "value",
        name: "bpm",
        scale: true,
        splitLine: { lineStyle: { color: "#283042", type: "dashed" } },
      },
      series: [
        {
          name: "Mean HR (bpm)",
          type: "line",
          smooth: true,
          data: cardiacBuckets.map((b) => Number(Number(b.mean_hr_bpm).toFixed(1))),
          lineStyle: { color: "#FB7185", width: 2 },
          itemStyle: { color: "#FB7185" },
        },
      ],
    };
  }, [cardiacBuckets]);

  const topSleepDrivers = mlDrivers.filter((d) => d.model_id === "ML_SLEEP_2A_CORE").slice(0, 5);
  const topHrvDrivers = mlDrivers.filter((d) => d.model_id === "ML_HRV_1B_MODIFIABLE").slice(0, 5);
  const sleepModelMeta = mlModels.find((m) => m.model_id === "ML_SLEEP_2A_CORE");

  const sleepDriverTakeaway = useMemo(() => {
    if (topSleepDrivers.length === 0) return null;
    const top1 = humanizeFeature(topSleepDrivers[0].feature_name);
    const dir1 = normalizeDirection(topSleepDrivers[0].direction).isPositive ? "longer" : "shorter";
    const top2 = topSleepDrivers[1] ? humanizeFeature(topSleepDrivers[1].feature_name) : null;
    return `In out-of-sample cross-validation, ${top1} is the strongest predictor of clean sleep duration (higher values link to ${dir1} sleep)${top2 ? `, followed by ${top2}` : ""}.`;
  }, [topSleepDrivers]);

  const hrvDriverTakeaway = useMemo(() => {
    if (topHrvDrivers.length === 0) return null;
    const top1 = humanizeFeature(topHrvDrivers[0].feature_name);
    const dir1 = normalizeDirection(topHrvDrivers[0].direction).isPositive ? "higher" : "lower";
    const top2 = topHrvDrivers[1] ? humanizeFeature(topHrvDrivers[1].feature_name) : null;
    return `Excluding overnight heart rate itself, ${top1} carries the largest predictive weight for next-night HRV (higher values associate with ${dir1} HRV)${top2 ? `, alongside ${top2}` : ""}.`;
  }, [topHrvDrivers]);

  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-xl font-semibold tracking-tight">Sleep &amp; Energy</h1>
        <p className="text-xs text-slate-400">
          Bedtime-to-Wake Architecture, Daily Recovery Phenotypes, Predictive HRV/Sleep Drivers (Non-Causal) &amp; Cardiac Efficiency
        </p>
      </div>

      {/* Item 6: Nocturnal Sleep Architecture */}
      <section className="obsidian-card p-5 space-y-3">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div>
            <h2 className="text-sm font-semibold">Clean Nocturnal Sleep Window &amp; Duration (Fitbit &ge;120m)</h2>
            <p className="text-xs text-slate-400">
              Each bar spans bedtime to wake-up (IST) with total sleep printed on top; hover any night for Deep, REM &amp; Light minutes
            </p>
          </div>
          <EpistemicPill label="Measured" />
        </div>
        <EChartBox option={sleepOption} height={260} isEmpty={sleepPts.length === 0} />
      </section>

      {/* Item 7: K-Means Phenotype Profiles */}
      <section className="obsidian-card p-5 space-y-3">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div>
            <h2 className="text-sm font-semibold">K-Means Daily Behavioral &amp; Recovery Phenotypes</h2>
            <p className="text-xs text-slate-400">
              Groups your days into recurring sleep-and-activity patterns; percentages show each pattern&apos;s share of all analyzed days (summing to 100%).
            </p>
          </div>
          <EpistemicPill label="Descriptive" />
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          {profiles.map((p) => (
            <div key={p.cluster_id} className="obsidian-subcard p-3.5 space-y-2">
              <div className="flex items-start justify-between gap-2">
                <span className="text-xs font-semibold text-slate-100">{p.phenotype_label}</span>
                <span className="text-[11px] font-mono text-[#A78BFA] shrink-0">
                  {formatNum(p.share_pct, 1, "%")} of days
                </span>
              </div>
              <div className="grid grid-cols-2 gap-1.5 text-xs font-mono tabular-nums text-slate-300">
                <div>
                  <span className="text-[10px] font-sans text-slate-500 block">Avg Sleep</span>
                  {formatSleepHm(p.asleep_min_mean)}
                </div>
                <div>
                  <span className="text-[10px] font-sans text-slate-500 block">Avg Steps</span>
                  {formatNum(p.steps_mean, 0)}
                </div>
                <div>
                  <span className="text-[10px] font-sans text-slate-500 block">Avg HRV (RMSSD)</span>
                  <span className="text-[#34D399]">{formatNum(p.ext_hrv_rmssd_mean, 1, " ms")}</span>
                </div>
                <div>
                  <span className="text-[10px] font-sans text-slate-500 block">Avg Deep Sleep</span>
                  <span className="text-[#60A5FA]">{formatNum(p.ext_deep_min_mean, 0, " min")}</span>
                </div>
              </div>
            </div>
          ))}
        </div>

        <div className="flex flex-wrap items-center justify-between gap-2 text-[11px] text-slate-400 pt-1">
          <span>
            <strong className="text-slate-300">Recovery direction:</strong> Higher Avg HRV (RMSSD) and Deep Sleep indicate stronger overnight autonomic recovery.
          </span>
          {recentPhenoMix.length > 0 && (
            <div className="flex flex-wrap items-center gap-1.5">
              <span className="text-slate-500">Last 7d mix:</span>
              {recentPhenoMix.map(([label, cnt]) => (
                <span
                  key={label}
                  className="px-2 py-0.5 rounded bg-[#0B0D12] border border-[#283042] text-slate-300"
                >
                  {label} ({cnt}d)
                </span>
              ))}
            </div>
          )}
        </div>

        <StatsCoverageDrawer
          meta={{
            raw_n: profiles.reduce((a, b) => a + Number(b.days_count || 0), 0),
            valid_n: profiles.reduce((a, b) => a + Number(b.days_count || 0), 0),
            effective_n: profiles.reduce((a, b) => a + Number(b.days_count || 0), 0),
            missingness_pct: 0,
            maturity_level: profiles[0]?.maturity_level || "Level 2 (Exploratory Phenotypes)",
            stability_flag: `ARI=${formatNum(profiles[0]?.median_ari, 3)} (${profiles[0]?.stability_flag || "Stable"})`,
            caveat_note: "Held-out recovery markers (HRV, deep sleep) are never used to form clusters — they validate recovery separation after clustering.",
          }}
        />
      </section>

      {/* Item 8: Predictive Drivers (Sleep + Modifiable HRV) */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <section className="obsidian-card p-5 space-y-3">
          <div className="flex items-center justify-between gap-2">
            <div>
              <h2 className="text-sm font-semibold">
                Clean Sleep Duration Drivers ({sleepModelMeta?.selected_algorithm?.replace(/_/g, " ") || "ML"})
              </h2>
              <p className="text-[11px] text-slate-400">
                Predictive ML feature importance (explains past variance, not a future forecast)
              </p>
            </div>
            <EpistemicPill label="Prediction" />
          </div>
          {sleepDriverTakeaway && (
            <p className="text-xs text-slate-200 obsidian-subcard px-3 py-2 leading-relaxed">
              {sleepDriverTakeaway}
            </p>
          )}
          <div className="space-y-2 text-xs">
            {topSleepDrivers.map((d) => {
              const dir = normalizeDirection(d.direction);
              return (
                <div key={d.feature_name} className="obsidian-subcard px-3 py-2 flex items-center justify-between gap-2">
                  <div>
                    <span className="font-medium text-slate-100 block">{humanizeFeature(d.feature_name)}</span>
                    <div className="flex items-center gap-2 mt-0.5">
                      <span
                        className={`inline-flex items-center gap-0.5 text-[10px] font-medium ${
                          dir.isPositive ? "text-[#34D399]" : "text-[#FB7185]"
                        }`}
                      >
                        {dir.isPositive ? <ArrowUpRight className="w-3 h-3" /> : <ArrowDownRight className="w-3 h-3" />}
                        {dir.label}
                      </span>
                      <span className="text-[10px] text-slate-500">· {humanizeStatus(d.feature_stability_flag)}</span>
                    </div>
                  </div>
                  <span className="font-mono font-semibold text-[#A78BFA] tabular-nums">
                    {formatNum(d.feature_weight, 2)}
                  </span>
                </div>
              );
            })}
          </div>
          <p className="text-[11px] text-slate-500">
            Predictive weights &amp; SHAP reflect out-of-sample CV utility and do not prove biological causation.
          </p>
        </section>

        <section className="obsidian-card p-5 space-y-3">
          <div className="flex items-center justify-between gap-2">
            <div>
              <h2 className="text-sm font-semibold">Modifiable Overnight HRV Drivers (Excl. Sleep HR)</h2>
              <p className="text-[11px] text-slate-400">
                Predictive ML feature importance (explains past variance, not a future forecast)
              </p>
            </div>
            <EpistemicPill label="Prediction" />
          </div>
          {hrvDriverTakeaway && (
            <p className="text-xs text-slate-200 obsidian-subcard px-3 py-2 leading-relaxed">
              {hrvDriverTakeaway}
            </p>
          )}
          <div className="space-y-2 text-xs">
            {topHrvDrivers.map((d) => {
              const dir = normalizeDirection(d.direction);
              return (
                <div key={d.feature_name} className="obsidian-subcard px-3 py-2 flex items-center justify-between gap-2">
                  <div>
                    <span className="font-medium text-slate-100 block">{humanizeFeature(d.feature_name)}</span>
                    <div className="flex items-center gap-2 mt-0.5">
                      <span
                        className={`inline-flex items-center gap-0.5 text-[10px] font-medium ${
                          dir.isPositive ? "text-[#34D399]" : "text-[#FB7185]"
                        }`}
                      >
                        {dir.isPositive ? <ArrowUpRight className="w-3 h-3" /> : <ArrowDownRight className="w-3 h-3" />}
                        {dir.label}
                      </span>
                      <span className="text-[10px] text-slate-500">· {humanizeStatus(d.feature_stability_flag)}</span>
                    </div>
                  </div>
                  <span className="font-mono font-semibold text-[#34D399] tabular-nums">
                    {formatNum(d.feature_weight, 2)}
                  </span>
                </div>
              );
            })}
          </div>
          <p className="text-[11px] text-slate-500">
            Predictive weights &amp; SHAP reflect out-of-sample CV utility and do not prove biological causation.
          </p>
        </section>
      </div>

      {/* Item 8: Hourly Cardiac Step Efficiency with Friendly Empty State */}
      <section className="obsidian-card p-5 space-y-3">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold">Exertion-Adjusted Hourly Cardiac Efficiency (HR by Step Bucket)</h2>
          <EpistemicPill label="Descriptive" />
        </div>
        {cardiacBuckets.length >= 2 ? (
          <EChartBox option={cardiacOption} height={220} />
        ) : (
          <div className="rounded-lg border border-[#283042] bg-[#1B202C]/50 px-4 py-6 flex items-center justify-center gap-2.5 text-xs text-slate-400">
            <Info className="w-4 h-4 text-[#60A5FA] shrink-0" />
            <span>
              More matched hourly step and heart-rate data is needed before this cardiac efficiency curve can render.
              Once enough hourly Fitbit pairs accumulate across activity tiers, your HR-per-step curve will appear here automatically.
            </span>
          </div>
        )}
      </section>
    </div>
  );
};
