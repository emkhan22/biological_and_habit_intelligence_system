import React, { useState } from "react";
import { ExternalLink, Search, Send } from "lucide-react";
import type { AskHealthResponse, HealthDataPayload } from "../types";
import { EpistemicPill } from "../components/EpistemicPill";
import { askHealthQuestionApi } from "../services/api";
import { formatNum } from "../utils/formatters";

interface Props {
  data: HealthDataPayload;
}

export const AskResearchView: React.FC<Props> = ({ data }) => {
  const [question, setQuestion] = useState("");
  const [useSearch, setUseSearch] = useState(true);
  const [loading, setLoading] = useState(false);
  const [answer, setAnswer] = useState<AskHealthResponse | null>(null);

  const correlations = (data.tables.Gold_Stats_Correlations ?? [])
    .filter((c) => c.spearman_rho != null)
    .slice(0, 10);
  const modelAudit = data.tables.Model_Run_Audit ?? [];
  const researchPulse = data.tables.AI_Research_Pulse ?? [];
  const peerChecks = data.tables.AI_Peer_Expectedness ?? [];
  const costLedger = data.tables.AI_Cost_Ledger ?? [];

  const totalInr = costLedger.reduce((acc, r) => acc + Number(r.estimated_inr || 0), 0);
  const pctBudget = Math.min(100, (totalInr / 500) * 100);

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
            Live Wisdom Q&amp;A (/api/ask-health), Verified Correlations, Model_Run_Audit, Research Pulse &amp; ₹500 AI Budget Meter
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
            <p className="text-slate-200">
              <strong className="text-[#34D399]">What your data shows:</strong> {answer.what_your_data_shows}
            </p>
            <p className="text-slate-300">
              <strong className="text-[#60A5FA]">What research says:</strong> {answer.what_research_says}
            </p>
            <p className="text-[11px] text-[#FB7185]">{answer.ask_your_doctor_note}</p>
          </div>
        )}
      </section>

      {/* Correlations & Model_Run_Audit */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <section className="obsidian-card p-5 space-y-3">
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-semibold">Verified Correlations (Autocorrelation-Adjusted)</h2>
            <EpistemicPill label="Correlation" />
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-xs text-left border-collapse">
              <thead>
                <tr className="border-b border-[#283042] text-slate-400">
                  <th className="py-1.5 pr-2">Pair</th>
                  <th className="py-1.5 pr-2">ρ / Partial</th>
                  <th className="py-1.5 pr-2">n / n_eff</th>
                  <th className="py-1.5">Stability</th>
                </tr>
              </thead>
              <tbody className="font-mono tabular-nums">
                {correlations.map((c) => (
                  <tr key={c.pair_id} className="border-b border-[#283042]/50 text-slate-200">
                    <td className="py-1.5 pr-2 font-sans">
                      {c.x_variable} → {c.y_variable}
                    </td>
                    <td className="py-1.5 pr-2 text-[#FBBF24]">
                      {formatNum(c.spearman_rho, 2)} / {formatNum(c.partial_rho, 2)}
                    </td>
                    <td className="py-1.5 pr-2">
                      {c.valid_n} / {c.effective_n}
                    </td>
                    <td className="py-1.5 font-sans text-slate-400">{c.stability_flag}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>

        <section className="obsidian-card p-5 space-y-3">
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-semibold">Model_Run_Audit (Progressive ML Gates)</h2>
            <EpistemicPill label="Prediction" />
          </div>
          <div className="space-y-2 text-xs">
            {modelAudit.map((m) => (
              <div key={m.model_id} className="obsidian-subcard px-3 py-2 flex items-center justify-between gap-2">
                <div>
                  <span className="font-mono font-semibold text-slate-100">{m.model_id}</span>
                  <span className="block text-[11px] text-slate-400">
                    {m.algorithm} · {m.maturity_level}
                  </span>
                </div>
                <div className="text-right font-mono tabular-nums text-[11px]">
                  <span className="block text-[#A78BFA]">
                    n={m.valid_n} (eff={m.effective_n})
                  </span>
                  <span className="text-slate-400">
                    R²={formatNum(m.ml_cv_r2 ?? m.linear_cv_r2, 2)}
                  </span>
                </div>
              </div>
            ))}
          </div>
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
              <p className="text-slate-300">{st.key_finding_plain_english}</p>
              <p className="text-slate-400">{st.why_it_matters_for_profile}</p>
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
                  {pc.safety_tag}
                </span>
              </div>
            ))}
          </div>
        )}
      </section>
    </div>
  );
};
