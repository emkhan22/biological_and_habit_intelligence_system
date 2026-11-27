import React, { useState } from "react";
import { ChevronDown, ChevronUp, Info } from "lucide-react";
import type { StatsCoverageMeta } from "../types";
import { formatNum } from "../utils/formatters";

interface Props {
  title?: string;
  meta: StatsCoverageMeta;
}

export const StatsCoverageDrawer: React.FC<Props> = ({
  title = "Stats & Coverage",
  meta,
}) => {
  const [open, setOpen] = useState(false);

  return (
    <div className="mt-3 border-t border-[#283042] pt-2.5">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        className="flex w-full items-center justify-between text-xs text-slate-400 hover:text-slate-200 transition-colors cursor-pointer"
      >
        <span className="inline-flex items-center gap-1.5 font-medium">
          <Info className="w-3.5 h-3.5 text-[#60A5FA]" />
          {title}
          {meta.maturity_level ? (
            <span className="text-[11px] text-slate-500">· {meta.maturity_level}</span>
          ) : null}
        </span>
        {open ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
      </button>

      {open && (
        <div
          className="mt-2.5 obsidian-subcard p-3 text-xs text-slate-300 space-y-2"
          data-testid="stats-coverage-drawer-body"
        >
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 font-mono tabular-nums">
            <div>
              <span className="text-slate-500 block text-[10px] uppercase">raw_n</span>
              <span>{meta.raw_n ?? "—"}</span>
            </div>
            <div>
              <span className="text-slate-500 block text-[10px] uppercase">valid_n</span>
              <span>{meta.valid_n ?? "—"}</span>
            </div>
            <div>
              <span className="text-slate-500 block text-[10px] uppercase">effective_n</span>
              <span>{meta.effective_n ?? "—"}</span>
            </div>
            <div>
              <span className="text-slate-500 block text-[10px] uppercase">missingness_pct</span>
              <span>{formatNum(meta.missingness_pct, 1, "%")}</span>
            </div>
            <div>
              <span className="text-slate-500 block text-[10px] uppercase">stability_flag</span>
              <span>{meta.stability_flag ?? "—"}</span>
            </div>
            <div>
              <span className="text-slate-500 block text-[10px] uppercase">maturity_level</span>
              <span className="font-sans">{meta.maturity_level ?? "—"}</span>
            </div>
          </div>
          {meta.caveat_note ? (
            <p className="text-[11px] leading-relaxed text-slate-400 border-t border-[#283042] pt-2">
              {meta.caveat_note}
            </p>
          ) : null}
        </div>
      )}
    </div>
  );
};
