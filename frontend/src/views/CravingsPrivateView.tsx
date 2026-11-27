import React from "react";
import { ShieldCheck } from "lucide-react";
import type { PrivateSmokingPayload } from "../types";
import { EpistemicPill } from "../components/EpistemicPill";
import { formatPct, safeJsonParse } from "../utils/formatters";

interface Props {
  privateData: PrivateSmokingPayload | null;
}

export const CravingsPrivateView: React.FC<Props> = ({ privateData }) => {
  if (!privateData) {
    return (
      <div className="obsidian-card p-6 text-sm text-slate-400">
        Private Mode data not loaded.
      </div>
    );
  }

  const coachRows = privateData.tables.Private_AI_Coach ?? [];
  const latestCoach = coachRows[coachRows.length - 1];
  const tactics = safeJsonParse<Array<Record<string, any>>>(
    latestCoach?.friction_tactics_json,
    []
  );
  const delayRows = privateData.tables.Private_Delay_Outcome_Stats ?? [];
  const driverRows = (privateData.tables.Private_Smoking_Risk_Drivers ?? []).slice(0, 8);

  return (
    <div className="space-y-5" data-testid="private-cravings-view">
      <div className="rounded-xl border border-[#FB7185]/30 bg-[#FB7185]/10 px-4 py-3 flex items-center justify-between text-xs">
        <span className="inline-flex items-center gap-2 text-slate-100">
          <ShieldCheck className="w-4 h-4 text-[#FB7185]" />
          <strong>Air-Gapped Private Mode:</strong> Reading strictly from{" "}
          <code className="font-mono">Smoking_Private_Store</code> ({privateData.source}). Zero rows shared with Clean Mode.
        </span>
        <EpistemicPill label="Descriptive" />
      </div>

      {/* Private AI Behavioral Friction Coach */}
      {latestCoach && (
        <section className="obsidian-card p-5 space-y-3">
          <div className="flex items-center justify-between">
            <h1 className="text-base font-semibold">Private Behavioral Friction Coach</h1>
            <EpistemicPill label={latestCoach.epistemic_label || "Descriptive"} />
          </div>
          <p className="text-xs text-slate-200 leading-relaxed">{latestCoach.trigger_pattern_synthesis}</p>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs">
            {tactics.map((t, i) => (
              <div key={i} className="obsidian-subcard p-3 space-y-1">
                <p className="font-semibold text-[#34D399]">{t.tactic_title}</p>
                <p className="text-[11px] text-slate-400">When: {t.when_to_deploy}</p>
                <p className="text-slate-200">{t.concrete_protocol}</p>
              </div>
            ))}
          </div>
        </section>
      )}

      {/* Onset Drivers & Delay-Bucket Resist Rates */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <section className="obsidian-card p-5 space-y-2.5">
          <h2 className="text-sm font-semibold">Onset Trigger &amp; Circadian Risk Drivers</h2>
          <div className="space-y-1.5 text-xs">
            {driverRows.map((d, i) => (
              <div key={i} className="obsidian-subcard px-3 py-2 flex items-center justify-between font-mono">
                <span className="font-sans text-slate-200">
                  {d.dimension_or_feature}: {d.stratum_value}
                </span>
                <span className="text-[#FB7185]">
                  n={d.event_count} · {formatPct(d.smoked_rate, true)}
                </span>
              </div>
            ))}
          </div>
        </section>

        <section className="obsidian-card p-5 space-y-2.5">
          <h2 className="text-sm font-semibold">Post-Event Delay Leverage (Strictly Isolated from ML X)</h2>
          <div className="space-y-1.5 text-xs">
            {delayRows.map((d, i) => (
              <div key={i} className="obsidian-subcard px-3 py-2 flex items-center justify-between font-mono">
                <span className="font-sans text-slate-200">{d.delay_bucket}</span>
                <span className="text-[#34D399]">
                  Resist: {formatPct(d.resist_rate, true)} (n={d.events_n})
                </span>
              </div>
            ))}
          </div>
        </section>
      </div>
    </div>
  );
};
