import React, { useMemo, useState } from "react";
import { ArrowDownRight, ArrowUpRight, ChevronDown, ChevronRight, ExternalLink, Search, Send } from "lucide-react";
import type { AskHealthResponse, HealthDataPayload } from "../types";
import { EpistemicPill } from "../components/EpistemicPill";
import { askHealthQuestionApi } from "../services/api";
import { formatNum } from "../utils/formatters";
import { humanizeClinicalText, humanizeFeature, humanizeModelId, humanizeStatus } from "../utils/humanize";

interface Props {
  data: HealthDataPayload;
}

function isGatedModel(m: Record<string, any>): boolean {
  const sf = String(m.stability_flag || "").toLowerCase();
  const ml = String(m.maturity_level || "").toLowerCase();
  const algo = String(m.algorithm || "").toLowerCase();
  return (
    sf.includes("gated") ||
    sf.includes("insufficient") ||
    ml.includes("level 0") ||
    ml.includes("level 1 (gated") ||
    algo.includes("gated")
  );
}

function explainGatedReason(m: Record<string, any>): string {
  const sf = String(m.stability_flag || "");
  const n = m.valid_n ?? 0;
  const eff = m.effective_n ?? 0;
  if (sf.includes("N100")) {
    return `Waiting for 100+ matched days (currently n=${n}, n_eff=${eff}) before unlocking tree ensembles & SHAP.`;
  }
  return `Waiting for more matched data (currently n=${n}, n_eff=${eff}) to meet minimum out-of-sample stability thresholds.`;
}

function explainCorrelationPlainEnglish(c: Record<string, any>): string {
  const xLabel = humanizeFeature(c.x_variable);
  const yLabel = humanizeFeature(c.y_variable);
  const rho = Number(c.spearman_rho);
  const absRho = Math.abs(rho);
  const strength =
    absRho >= 0.5 ? "moderate-to-strong" : absRho >= 0.25 ? "modest" : "weak";
  const dirText =
    rho >= 0
      ? `higher ${xLabel} tends to coincide with higher ${yLabel}`
      : `higher ${xLabel} tends to coincide with lower ${yLabel}`;
  const partialStr =
    c.partial_rho != null && Number.isFinite(Number(c.partial_rho))
      ? ` After adjusting for covariates (${humanizeClinicalText(c.covariates_adjusted || "sleep/cycle/dose")}), partial ρ is ${formatNum(c.partial_rho, 2)}.`
      : "";
  return `Shows a ${strength} ${rho >= 0 ? "positive" : "negative"} association (ρ = ${formatNum(rho, 2)}): ${dirText}.${partialStr} Correlation does not prove causation.`;
}

export const AskResearchView: React.FC<Props> = ({ data }) => {
  const [question, setQuestion] = useState("");
  const [useSearch, setUseSearch] = useState(true);
  const [loading, setLoading] = useState(false);
  const [answer, setAnswer] = useState<AskHealthResponse | null>(null);
  const [expandedCorrId, setExpandedCorrId] = useState<string | null>(null);

  const correlations = (data.tables.Gold_Stats_Correlations ?? [])
    .filter((c) => c.spearman_rho != null && c.spearman_rho !== "")
    .slice(0, 10);
  const modelAudit = data.tables.Model_Run_Audit ?? [];
  const researchPulse = data.tables.AI_Research_Pulse ?? [];
  const peerChecks = data.tables.AI_Peer_Expectedness ?? [];
  const costLedger = data.tables.AI_Cost_Ledger ?? [];

  const totalInr = costLedger.reduce((acc, r) => acc + Number(r.estimated_inr || 0), 0);
  const pctBudget = Math.min(100, (totalInr / 500) * 100);

  const { activeModels, gatedModels } = useMemo(() => {
    const active: Array<Record<string, any>> = [];
    const gated: Array<Record<string, any>> = [];
    for (const m of modelAudit) {
      if (isGatedModel(m)) gated.push(m);
      else active.push(m);
    }
    return { activeModels: active, gatedModels: gated };
  }, [modelAudit]);

  async function handleAsk(e: React.FormEvent) {
    e.preventDefault();
    if (!question.trim()) return;
    setLoading(true);
    try {
      const resp = await askHealthQuestionApi(question.trim(), data.packet, useSearch);
      setAnswer(resp);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="text-xl font-semibold tracking-tight">Ask &amp; Research</h1>
          <p className="text-xs text-slate-400">
            Live Wisdom Q&amp;A (/api/ask-health), Verified Correlations, Model Run Audit, Research Pulse &amp; ₹500 AI Budget Meter
          </p>
        </div>

        {/* ₹500 Monthly Budget Circuit Breaker Meter */}
        <div className="obsidian-subcard px-3.5 py-2 text-xs min-w-[210px]">
          <div className="flex items-center justify-between font-mono tabular-nums">
            <span className="text-slate-400 font-sans">AI Spend MTD</span>
            <span className="text-[#34D399]">₹{totalInr.toFixed(2)} / ₹500</span>
          </div>
          <div className="mt-1.5 h-1.5 rounded-full bg-[#0B0D12] overflow-hidden">
            <div
              className="h-full bg-[#C06C9F]"
              style={{ width: `${pctBudget.toFixed(1)}%` }}
            />
          </div>
        </div>
      </div>

      {/* Live Wisdom Q&A */}
      <section className="obsidian-card p-5 space-y-3">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold">Ask Your Health Intelligence Store</h2>
          <EpistemicPill label="Screening_Context" />
        </div>
        <form onSubmit={handleAsk} className="space-y-2.5">
          <div className="flex flex-col sm:flex-row gap-2">
            <input
              type="text"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="Ask about your Tirzepatide wave, bedtime vs HRV, sweet→hunger patterns, or BMR calorie audit..."
              className="flex-1 rounded-lg bg-[#0B0D12] border border-[#283042] px-3 py-2 text-sm text-slate-100 focus:outline-none focus:border-[#C06C9F]"
            />
            <button
              type="submit"
              disabled={loading}
              className="inline-flex items-center justify-center gap-1.5 px-4 py-2 rounded-lg bg-[#C06C9F] text-[#0B0D12] font-semibold text-xs hover:opacity-90 disabled:opacity-50 cursor-pointer"
            >
              <Send className="w-3.5 h-3.5" />
              {loading ? "Synthesizing..." : "Ask"}
            </button>
          </div>
          <label className="inline-flex items-center gap-2 text-xs text-slate-400 cursor-pointer">
            <input
              type="checkbox"
              checked={useSearch}
              onChange={(e) => setUseSearch(e.target.checked)}
              className="accent-[#60A5FA]"
            />
            <Search className="w-3.5 h-3.5 text-[#60A5FA]" />
            Enable Google Search Grounding
          </label>
        </form>

        {answer && (
          <div className="obsidian-subcard p-4 space-y-2 text-xs">
            <div className="flex items-center justify-between">
              <span className="font-semibold text-slate-100">Synthesis Result ({answer.confidence_tier})</span>
              <EpistemicPill label={answer.epistemic_label} />
            </div>
            <p className="text-slate-200 leading-relaxed">
              <strong className="text-[#34D399]">What your data shows:</strong> {humanizeClinicalText(answer.what_your_data_shows)}
            </p>
            <p className="text-slate-300 leading-relaxed">
              <strong className="text-[#60A5FA]">What research says:</strong> {humanizeClinicalText(answer.what_research_says)}
            </p>
            <p className="text-[11px] text-[#FB7185]">{answer.ask_your_doctor_note}</p>
          </div>
        )}
      </section>

      {/* Item 12: Verified Correlations & Model Run Audit */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <section className="obsidian-card p-5 space-y-3">
          <div className="flex items-center justify-between gap-2">
            <div>
              <h2 className="text-sm font-semibold">Verified Correlations (Autocorrelation-Adjusted)</h2>
              <p className="text-[11px] text-slate-400 mt-0.5">
                Day-to-day biometrics repeat over time, so we compute effective independent days (n_eff) to avoid false positives. Click any row to inspect details.
              </p>
            </div>
            <EpistemicPill label="Correlation" />
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-xs text-left border-collapse">
              <thead>
                <tr className="border-b border-[#283042] text-slate-400">
                  <th className="py-1.5 pr-2">Relationship</th>
                  <th className="py-1.5 pr-2">Direction &amp; ρ</th>
                  <th className="py-1.5 pr-2">n / n_eff</th>
                  <th className="py-1.5">Stability</th>
                </tr>
              </thead>
              <tbody className="font-mono tabular-nums">
                {correlations.map((c) => {
                  const rowKey = String(c.pair_id || `${c.x_variable}-${c.y_variable}`);
                  const isOpen = expandedCorrId === rowKey;
                  const rho = Number(c.spearman_rho);
                  const isPos = rho >= 0;
                  return (
                    <React.Fragment key={rowKey}>
                      <tr
                        onClick={() => setExpandedCorrId(isOpen ? null : rowKey)}
                        className="border-b border-[#283042]/50 text-slate-200 hover:bg-[#1B202C]/70 cursor-pointer"
                      >
                        <td className="py-2 pr-2 font-sans">
                          <span className="inline-flex items-center gap-1">
                            {isOpen ? (
                              <ChevronDown className="w-3.5 h-3.5 text-[#60A5FA] shrink-0" />
                            ) : (
                              <ChevronRight className="w-3.5 h-3.5 text-slate-500 shrink-0" />
                            )}
                            <span>
                              {humanizeFeature(c.x_variable)} → {humanizeFeature(c.y_variable)}
                            </span>
                          </span>
                        </td>
                        <td className="py-2 pr-2">
                          <span
                            className={`inline-flex items-center gap-0.5 ${
                              isPos ? "text-[#34D399]" : "text-[#FB7185]"
                            }`}
                          >
                            {isPos ? <ArrowUpRight className="w-3 h-3" /> : <ArrowDownRight className="w-3 h-3" />}
                            {formatNum(rho, 2)}
                          </span>
                        </td>
                        <td className="py-2 pr-2 text-slate-300">
                          {c.valid_n} / {c.effective_n}
                        </td>
                        <td className="py-2 font-sans text-slate-400">
                          {humanizeStatus(c.stability_flag)}
                        </td>
                      </tr>
                      {isOpen && (
                        <tr className="bg-[#0B0D12]/80 border-b border-[#283042]">
                          <td colSpan={4} className="p-3 font-sans text-xs text-slate-300 space-y-1.5">
                            <p className="leading-relaxed text-slate-200">
                              {explainCorrelationPlainEnglish(c)}
                            </p>
                            <div className="flex flex-wrap gap-3 text-[11px] font-mono text-slate-400">
                              <span>Days (n): {c.valid_n}</span>
                              <span>Effective Days (n_eff): {c.effective_n}</span>
                              {c.ci_95_low != null && c.ci_95_high != null && (
                                <span>95% CI: [{formatNum(c.ci_95_low, 2)}, {formatNum(c.ci_95_high, 2)}]</span>
                              )}
                              {c.partial_rho != null && <span>Partial ρ: {formatNum(c.partial_rho, 2)}</span>}
                            </div>
                            {c.caveat_note && (
                              <p className="text-[11px] text-slate-500">{humanizeClinicalText(c.caveat_note)}</p>
                            )}
                          </td>
                        </tr>
                      )}
                    </React.Fragment>
                  );
                })}
              </tbody>
            </table>
          </div>
        </section>

        <section className="obsidian-card p-5 space-y-3">
          <div className="flex items-center justify-between gap-2">
            <div>
              <h2 className="text-sm font-semibold">Model Run Audit (Progressive ML Gates)</h2>
              <p className="text-[11px] text-slate-400 mt-0.5">
                Shows which predictive models had enough clean data to run vs. models intentionally held back until more days accumulate.
              </p>
            </div>
            <EpistemicPill label="Prediction" />
          </div>

          {/* Active Models */}
          {activeModels.length > 0 && (
            <div className="space-y-2">
              <p className="text-[11px] uppercase tracking-wider font-semibold text-[#34D399]">
                Active Models ({activeModels.length})
              </p>
              {activeModels.map((m) => (
                <div key={m.model_id} className="obsidian-subcard px-3 py-2.5 flex items-center justify-between gap-2 text-xs">
                  <div>
                    <span className="font-medium text-slate-100 block">{humanizeModelId(m.model_id)}</span>
                    <span className="block text-[11px] text-slate-400">
                      {String(m.algorithm || "ML").replace(/_/g, " ")} · {humanizeStatus(m.stability_flag)}
                    </span>
                  </div>
                  <div className="text-right font-mono tabular-nums text-[11px] shrink-0">
                    <span className="block text-[#34D399]">
                      n={m.valid_n} (n_eff={m.effective_n})
                    </span>
                    <span className="text-slate-400">
                      CV R²={formatNum(m.ml_cv_r2 ?? m.linear_cv_r2, 2)}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* Waiting for More Data (Gated) */}
          {gatedModels.length > 0 && (
            <div className="space-y-2 pt-1">
              <p className="text-[11px] uppercase tracking-wider font-semibold text-[#FBBF24]">
                Waiting for More Data ({gatedModels.length})
              </p>
              {gatedModels.map((m) => (
                <div key={m.model_id} className="obsidian-subcard px-3 py-2.5 space-y-1 text-xs">
                  <div className="flex items-center justify-between gap-2">
                    <span className="font-medium text-slate-200">{humanizeModelId(m.model_id)}</span>
                    <span className="px-2 py-0.5 rounded bg-[#FBBF24]/10 border border-[#FBBF24]/30 text-[10px] text-[#FBBF24] shrink-0">
                      {humanizeStatus(m.stability_flag)}
                    </span>
                  </div>
                  <p className="text-[11px] text-slate-400">{explainGatedReason(m)}</p>
                </div>
              ))}
            </div>
          )}
        </section>
      </div>

      {/* Research Pulse & Peer Expectedness */}
      <section className="obsidian-card p-5 space-y-3">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold">Search-Grounded Research Pulse &amp; Regimen Safety</h2>
          <EpistemicPill label="Screening_Context" />
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs">
          {researchPulse.map((st, idx) => (
            <div key={idx} className="obsidian-subcard p-3.5 space-y-1.5">
              <div className="flex items-start justify-between gap-2">
                <span className="font-semibold text-slate-100">{st.study_or_guideline_title}</span>
                <span className="text-[10px] font-mono text-slate-400 shrink-0">{st.publication_date_or_year}</span>
              </div>
              <p className="text-slate-300 leading-relaxed">{st.key_finding_plain_english}</p>
              <p className="text-slate-400 leading-relaxed">{st.why_it_matters_for_profile}</p>
              {st.source_url && (
                <a
                  href={st.source_url}
                  target="_blank"
                  rel="noreferrer"
                  className="inline-flex items-center gap-1 text-[#60A5FA] hover:underline"
                >
                  Verified Source <ExternalLink className="w-3 h-3" />
                </a>
              )}
            </div>
          ))}
        </div>
        {peerChecks.length > 0 && (
          <div className="pt-2 border-t border-[#283042] space-y-1.5 text-xs">
            <p className="text-[11px] uppercase tracking-wider text-slate-400">Peer-Expectedness Safety Tags</p>
            {peerChecks.map((pc, i) => (
              <div key={i} className="obsidian-subcard px-3 py-2 flex items-center justify-between gap-2">
                <span className="text-slate-200">{pc.symptom_or_shift}</span>
                <span className="px-2 py-0.5 rounded bg-[#141821] border border-[#283042] text-[#FBBF24] font-mono">
                  {humanizeStatus(pc.safety_tag)}
                </span>
              </div>
            ))}
          </div>
        )}
      </section>
    </div>
  );
};
