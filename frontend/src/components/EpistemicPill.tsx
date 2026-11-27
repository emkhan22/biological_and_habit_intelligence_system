import React from "react";
import type { EpistemicLabel } from "../types";
import { epistemicDisplayText, normalizeEpistemicLabel } from "../utils/formatters";

const STYLE_MAP: Record<EpistemicLabel, string> = {
  Measured: "bg-[#34D399]/10 text-[#34D399] border-[#34D399]/30",
  Descriptive: "bg-[#60A5FA]/10 text-[#60A5FA] border-[#60A5FA]/30",
  Correlation: "bg-[#FBBF24]/10 text-[#FBBF24] border-[#FBBF24]/30",
  Prediction: "bg-[#A78BFA]/10 text-[#A78BFA] border-[#A78BFA]/30",
  Causal_Hypothesis: "bg-[#C06C9F]/10 text-[#C06C9F] border-[#C06C9F]/30",
  Screening_Context: "bg-[#FB7185]/10 text-[#FB7185] border-[#FB7185]/30",
};

interface Props {
  label?: string | null;
  className?: string;
}

export const EpistemicPill: React.FC<Props> = ({ label, className = "" }) => {
  const norm = normalizeEpistemicLabel(label);
  return (
    <span
      className={`inline-flex items-center px-2 py-0.5 text-[11px] font-medium tracking-wide rounded border ${STYLE_MAP[norm]} ${className}`}
      data-epistemic={norm}
    >
      {epistemicDisplayText(norm)}
    </span>
  );
};
