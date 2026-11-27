"""Silver Gate 3: Analyte-Specific Lab Unit Normalization & Auto-Date Corruption Quarantine.

Rules (AGENTS.md Rule 1 & Section 3.3):
- NEVER overwrite `Original_Value`, `Numeric_Value`, `Unit`, or `Standardized_Unit`.
- Append `Normalized_Value`, `Normalized_Unit`, `Normalization_Rule_ID`, `Normalization_Valid_Flag`,
  and `data_corruption_flag`.
- Quarantine spreadsheet auto-date corruption (e.g., `2026-05-02` in `Original_Value` where `"2-5"`
  was intended, or date strings in `Source_Page`) with `data_corruption_flag = True`.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from health_engine.config_loader import add_display_date_col, to_iso_date_str
from health_engine.silver.qa_logger import SilverQALogger


def _canon_unit_token(u: str) -> str:
    """Normalize micro symbols and whitespace for deterministic unit lookup."""
    s = str(u or "").strip()
    s = s.replace("μ", "µ")
    s = re.sub(r"\s+", " ", s)
    return s.lower()


def _build_corruption_detector(lab_norm_cfg: dict[str, Any]) -> list[re.Pattern[str]]:
    pats = lab_norm_cfg.get("corruption_patterns", {})
    compiled: list[re.Pattern[str]] = []
    for key in ("iso_date_regex", "slash_date_regex", "month_name_date_regex"):
        expr = pats.get(key)
        if expr:
            compiled.append(re.compile(expr))
    return compiled


def _is_autodate_corrupted(val: Any, patterns: list[re.Pattern[str]]) -> bool:
    if val is None or pd.isna(val):
        return False
    s = str(val).strip()
    if not s:
        return False
    return any(p.match(s) is not None for p in patterns)


def _extract_scalar_float(num_val: Any, orig_val: Any) -> float:
    for candidate in (num_val, orig_val):
        if candidate is None or pd.isna(candidate):
            continue
        s = str(candidate).strip().replace(",", "")
        if not s:
            continue
        # Reject censored (<, >) or range strings (e.g. "2-5")
        if re.match(r"^[+-]?(\d+(\.\d+)?|\.\d+)$", s):
            try:
                return float(s)
            except ValueError:
                continue
    return np.nan


def clean_lab_results(
    labs_raw: pd.DataFrame,
    lab_norm_cfg: dict[str, Any],
    qa: SilverQALogger,
) -> pd.DataFrame:
    """Produce Silver_Lab_Results preserving all original clinical columns."""
    if labs_raw.empty:
        qa.log(
            "Labs",
            "AggregatedMedicalRecords!Lab_Results",
            "LAB_UNIT_NORMALIZATION",
            0,
            0,
            0,
            "Empty Lab_Results input",
        )
        return pd.DataFrame()

    df = labs_raw.copy()
    immutable_cols = ["Original_Value", "Numeric_Value", "Unit", "Standardized_Unit"]
    snapshots = {c: df[c].copy() for c in immutable_cols if c in df.columns}

    patterns = _build_corruption_detector(lab_norm_cfg)
    analytes_map: dict[str, Any] = lab_norm_cfg.get("analytes", {})

    # Pre-index analyte unit rules
    lookup: dict[str, dict[str, Any]] = {}
    for analyte_name, spec in analytes_map.items():
        unit_rules: dict[str, tuple[float, str]] = {}
        for conv in spec.get("conversions", []):
            mult = float(conv["multiplier"])
            rid = str(conv["rule_id"])
            for su in conv.get("source_units", []):
                unit_rules[_canon_unit_token(su)] = (mult, rid)
        lookup[analyte_name.strip().lower()] = {
            "canonical_unit": str(spec["canonical_unit"]),
            "plausible_min": spec.get("plausible_min"),
            "plausible_max": spec.get("plausible_max"),
            "unit_rules": unit_rules,
        }

    norm_vals: list[float] = []
    norm_units: list[str] = []
    rule_ids: list[str] = []
    valid_flags: list[bool] = []
    corrupt_flags: list[bool] = []

    converted_non_identity_cnt = 0
    plausibility_fail_cnt = 0

    for row in df.to_dict(orient="records"):
        orig_v = row.get("Original_Value", "")
        num_v = row.get("Numeric_Value", "")
        raw_u = str(row.get("Unit", "") or "").strip()
        std_u = str(row.get("Standardized_Unit", "") or "").strip()
        src_page = row.get("Source_Page", "")

        # 1. Detect spreadsheet auto-date corruption in Original_Value or Source_Page
        if _is_autodate_corrupted(orig_v, patterns) or _is_autodate_corrupted(src_page, patterns):
            corrupt_flags.append(True)
            norm_vals.append(np.nan)
            norm_units.append("")
            rule_ids.append("QUARANTINED_AUTODATE_CORRUPTION")
            valid_flags.append(False)
            continue

        corrupt_flags.append(False)
        scalar = _extract_scalar_float(num_v, orig_v)
        if np.isnan(scalar):
            norm_vals.append(np.nan)
            norm_units.append(std_u or raw_u)
            rule_ids.append("NON_NUMERIC_QUALITATIVE")
            valid_flags.append(False)
            continue

        test_name = str(
            row.get("Standardized_Test_Name") or row.get("Original_Test_Name") or ""
        ).strip()
        spec = lookup.get(test_name.lower())

        if spec is not None:
            matched: tuple[float, str] | None = None
            for u_candidate in (std_u, raw_u):
                tok = _canon_unit_token(u_candidate)
                if tok and tok in spec["unit_rules"]:
                    matched = spec["unit_rules"][tok]
                    break
            if matched is None:
                norm_vals.append(np.nan)
                norm_units.append(spec["canonical_unit"])
                rule_ids.append("UNRECOGNIZED_SOURCE_UNIT")
                valid_flags.append(False)
                plausibility_fail_cnt += 1
                continue

            mult, rid = matched
            converted = float(scalar * mult)
            if not np.isclose(mult, 1.0):
                converted_non_identity_cnt += 1
            pmin = spec["plausible_min"]
            pmax = spec["plausible_max"]
            in_bounds = True
            if pmin is not None and converted < float(pmin):
                in_bounds = False
            if pmax is not None and converted > float(pmax):
                in_bounds = False
            if not in_bounds:
                plausibility_fail_cnt += 1
            norm_vals.append(converted)
            norm_units.append(spec["canonical_unit"])
            rule_ids.append(rid)
            valid_flags.append(in_bounds)
        else:
            norm_vals.append(float(scalar))
            norm_units.append(std_u or raw_u)
            rule_ids.append("PASSTHROUGH_STANDARDIZED")
            valid_flags.append(True)

    df["Collection_Date"] = (
        df["Collection_DateTime"].map(to_iso_date_str)
        if "Collection_DateTime" in df.columns
        else None
    )
    df = add_display_date_col(df, "Collection_Date")
    df["Normalized_Value"] = norm_vals
    df["Normalized_Unit"] = norm_units
    df["Normalization_Rule_ID"] = rule_ids
    df["Normalization_Valid_Flag"] = valid_flags
    df["data_corruption_flag"] = corrupt_flags
    df["epistemic_class"] = "Measured"

    # Verify AGENTS.md Rule 1 immutability guarantee
    for col, orig_series in snapshots.items():
        if not df[col].equals(orig_series):
            raise RuntimeError(f"Immutable clinical column '{col}' was altered during normalization!")

    corrupt_cnt = int(sum(corrupt_flags))
    valid_cnt = int(sum(valid_flags))
    qa.log(
        domain="Labs",
        source_tab="AggregatedMedicalRecords!Lab_Results",
        rule_id="LAB_AUTODATE_CORRUPTION_QUARANTINED",
        total_input_rows=len(df),
        flagged_or_excluded_rows=corrupt_cnt,
        valid_output_rows=len(df) - corrupt_cnt,
        detail_note="Quarantined rows with spreadsheet auto-date strings in Original_Value or Source_Page",
    )
    qa.log(
        domain="Labs",
        source_tab="AggregatedMedicalRecords!Lab_Results",
        rule_id="LAB_UNIT_NORMALIZATION",
        total_input_rows=len(df),
        flagged_or_excluded_rows=plausibility_fail_cnt,
        valid_output_rows=valid_cnt,
        detail_note=(
            f"Normalized quantitative rows={valid_cnt}; active unit rescalings={converted_non_identity_cnt}"
        ),
    )

    return df
