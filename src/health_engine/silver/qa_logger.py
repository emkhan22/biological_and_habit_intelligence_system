"""Structured QA logger materializing the Silver_QA_Log audit table."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from health_engine.config_loader import now_ist_str, validate_epistemic_class


@dataclass
class SilverQALogger:
    """Collects deterministic row-count audit records across all Silver cleaning gates."""

    run_ts_ist: str = field(default_factory=now_ist_str)
    records: list[dict[str, object]] = field(default_factory=list)

    def log(
        self,
        domain: str,
        source_tab: str,
        rule_id: str,
        total_input_rows: int,
        flagged_or_excluded_rows: int,
        valid_output_rows: int,
        detail_note: str = "",
        epistemic_class: str = "Descriptive",
    ) -> None:
        validate_epistemic_class(epistemic_class)
        self.records.append(
            {
                "run_ts_ist": self.run_ts_ist,
                "domain": domain,
                "source_tab": source_tab,
                "rule_id": rule_id,
                "total_input_rows": int(total_input_rows),
                "flagged_or_excluded_rows": int(flagged_or_excluded_rows),
                "valid_output_rows": int(valid_output_rows),
                "detail_note": detail_note,
                "epistemic_class": epistemic_class,
            }
        )

    def to_dataframe(self) -> pd.DataFrame:
        cols = [
            "run_ts_ist",
            "domain",
            "source_tab",
            "rule_id",
            "total_input_rows",
            "flagged_or_excluded_rows",
            "valid_output_rows",
            "detail_note",
            "epistemic_class",
        ]
        if not self.records:
            return pd.DataFrame(columns=cols)
        return pd.DataFrame(self.records, columns=cols)
