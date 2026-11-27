import React, { useMemo } from "react";
import type { EChartsOption } from "echarts";
import type { HealthDataPayload } from "../types";
import { EChartBox } from "../components/EChartBox";
import { EpistemicPill } from "../components/EpistemicPill";
import { StatsCoverageDrawer } from "../components/StatsCoverageDrawer";
import { formatDateIN, formatNum, formatSleepHm } from "../utils/formatters";

interface Props {
  data: HealthDataPayload;
}

export const SleepEnergyView: React.FC<Props> = ({ data }) => {
  const daily = (data.tables.Gold_Daily_Unified ?? []).slice(-30);
  const mlDrivers = data.tables.Gold_ML_Feature_Drivers ?? [];
  const mlModels = data.tables.Gold_ML_Model_Summary ?? [];
  const profiles = data.tables.Gold_Phenotype_Profiles ?? [];
  const cardiacBuckets = (data.tables.Gold_Cardiac_Efficiency_Buckets ?? []).filter(
    (r) => r.aggregation_scope === "Overall"
  );

  const sleepOption = useMemo<EChartsOption>(() => {
    const pts = daily.filter((r) => r.asleep_min != null);
    return {
      tooltip: { trigger: "axis", backgroundColor: "#141821", borderColor: "#283042", textStyle: { color: "#F1F5F9" } },
      legend: { top: 0, textStyle: { color: "#94A3B8", fontSize: 11 } },
      grid: { left: 44, right: 16, top: 30, bottom: 28 },
      xAxis: {
        type: "category",
        data: pts.map((p) => formatDateIN(p.display_date || p.date)),
        axisLine: { lineStyle: { color: "#283042" } },
      },
      yAxis: {
        type: "value",
        name: "min",
        splitLine: { lineStyle: { color: "#283042", type: "dashed" } },
      },
      series: [
        { name: "Deep", type: "bar", stack: "sleep", data: pts.map((p) => p.deep_min), itemStyle: { color: "#A78BFA" } },
        { name: "REM", type: "bar", stack: "sleep", data: pts.map((p) => p.rem_min), itemStyle: { color: "#60A5FA" } },
        { name: "Light", type: "bar", stack: "sleep", data: pts.map((p) => p.light_min), itemStyle: { color: "#334155" } },
      ],
    };
  }, [daily]);

  const cardiacOption = useMemo<EChartsOption>(() => {
    return {
      tooltip: { trigger: "axis", backgroundColor: "#141821", borderColor: "#283042", textStyle: { color: "#F1F5F9" } },
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
          data: cardiacBuckets.map((b) => b.mean_hr_bpm),
          lineStyle: { color: "#FB7185", width: 2 },
          itemStyle: { color: "#FB7185" },
        },
      ],
    };
  }, [cardiacBuckets]);

  const topSleepDrivers = mlDrivers.filter((d) => d.model_id === "ML_SLEEP_2A_CORE").slice(0, 5);
  const topHrvDrivers = mlDrivers.filter((d) => d.model_id === "ML_HRV_1B_MODIFIABLE").slice(0, 5);
  const sleepModelMeta = mlModels.find((m) => m.model_id === "ML_SLEEP_2A_CORE");

  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-xl font-semibold tracking-tight">Sleep &amp; Energy</h1>
        <p className="text-xs text-slate-400">
          Sleep Architecture, Predictive HRV/Sleep Drivers (Non-Causal), K-Means Regimes &amp; Cardiac Step Efficiency
        </p>
      </div>

      {/* Sleep Architecture */}
      <section className="obsidian-card p-5 space-y-3">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold">Clean Nocturnal Sleep Architecture (Fitbit &ge;120m)</h2>
          <EpistemicPill label="Measured" />
        </div>
        <EChartBox option={sleepOption} height={240} isEmpty={daily.filter((r) => r.asleep_min != null).length === 0} />
      </section>

      {/* K-Means Phenotype Profiles */}
      <section className="obsidian-card p-5 space-y-3">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold">K-Means Daily Behavioral &amp; Recovery Phenotypes</h2>
          <EpistemicPill label="Descriptive" />
        </div>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          {profiles.map((p) => (
            <div key={p.cluster_id} className="obsidian-subcard p-3.5 space-y-1.5">
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold text-slate-100">{p.phenotype_label}</span>
                <span className="text-[11px] font-mono text-[#A78BFA]">{formatNum(p.share_pct, 1, "%")}</span>
              </div>
              <div className="grid grid-cols-2 gap-1.5 text-xs font-mono tabular-nums text-slate-300">
                <div>Sleep: {formatSleepHm(p.asleep_min_mean)}</div>
                <div>Steps: {formatNum(p.steps_mean, 0)}</div>
                <div className="text-[#34D399]">Ext HRV: {formatNum(p.ext_hrv_rmssd_mean, 1, " ms")}</div>
                <div className="text-[#60A5FA]">Deep: {formatNum(p.ext_deep_min_mean, 0, "m")}</div>
              </div>
            </div>
          ))}
        </div>
        <StatsCoverageDrawer
          meta={{
            raw_n: profiles.reduce((a, b) => a + Number(b.days_count || 0), 0),
            valid_n: profiles.reduce((a, b) => a + Number(b.days_count || 0), 0),
            effective_n: profiles.reduce((a, b) => a + Number(b.days_count || 0), 0),
            missingness_pct: 0,
            maturity_level: profiles[0]?.maturity_level || "Level 2 (Exploratory Phenotypes)",
            stability_flag: `ARI=${formatNum(profiles[0]?.median_ari, 3)} (${profiles[0]?.stability_flag || "Stable"})`,
            caveat_note: "External held-out recovery markers (HRV, deep sleep) are never used during K-Means clustering.",
          }}
        />
      </section>

      {/* Predictive Drivers (Sleep + Modifiable HRV) */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <section className="obsidian-card p-5 space-y-3">
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-semibold">Clean Sleep Duration Drivers ({sleepModelMeta?.selected_algorithm || "ML"})</h2>
            <EpistemicPill label="Prediction" />
          </div>
          <div className="space-y-2 text-xs">
            {topSleepDrivers.map((d) => (
              <div key={d.feature_name} className="obsidian-subcard px-3 py-2 flex items-center justify-between">
                <div>
                  <span className="font-mono text-slate-200">{d.feature_name}</span>
                  <span className="block text-[10px] text-slate-500">
                    {d.direction} · {d.feature_stability_flag}
                  </span>
                </div>
                <span className="font-mono font-semibold text-[#A78BFA] tabular-nums">
                  {formatNum(d.feature_weight, 2)}
                </span>
              </div>
            ))}
          </div>
          <p className="text-[11px] text-slate-500">
            Predictive weights &amp; SHAP reflect out-of-sample CV utility and do not prove biological causation.
          </p>
        </section>

        <section className="obsidian-card p-5 space-y-3">
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-semibold">Modifiable Overnight HRV Drivers (Excl. Sleep HR)</h2>
            <EpistemicPill label="Prediction" />
          </div>
          <div className="space-y-2 text-xs">
            {topHrvDrivers.map((d) => (
              <div key={d.feature_name} className="obsidian-subcard px-3 py-2 flex items-center justify-between">
                <div>
                  <span className="font-mono text-slate-200">{d.feature_name}</span>
                  <span className="block text-[10px] text-slate-500">
                    {d.direction} · {d.feature_stability_flag}
                  </span>
                </div>
                <span className="font-mono font-semibold text-[#34D399] tabular-nums">
                  {formatNum(d.feature_weight, 2)}
                </span>
              </div>
            ))}
          </div>
        </section>
      </div>

      {/* Hourly Cardiac Step Efficiency */}
      <section className="obsidian-card p-5 space-y-3">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold">Exertion-Adjusted Hourly Cardiac Efficiency (HR by Step Bucket)</h2>
          <EpistemicPill label="Descriptive" />
        </div>
        <EChartBox option={cardiacOption} height={220} isEmpty={cardiacBuckets.length === 0} />
      </section>
    </div>
  );
};
