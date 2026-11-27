"""Gold Table 5: Gold_Clinical_Indices (`epistemic_class = 'Screening_Context'`).

Rules (Prompt Section 4.5 & AGENTS.md Rules 1, 3):
- Compute longitudinal derived indices from `Silver_Lab_Results` (`Normalized_Value`) +
  `Silver_Body_Measurements`:
  - `HOMA_IR` and `TyG_Index`: compute STRICTLY ONLY when verified fasting glucose
    (`Fasting Blood Glucose (FBS)`) and fasting insulin / triglycerides share the exact
    same fasting `Collection_Date`; flag provenance and reject non-fasting or cross-date pairs.
  - `TG_HDL_Ratio` and `Non_HDL_Cholesterol`.
  - `Mentzer_Index` (`MCV / RBC_Count`) with mandatory caveat:
    "Microcytosis screening context only; not a direct iron-repletion measure."
  - `FIB4` and `APRI` (exact age from DOB `1994-09-22` at each lab date and `AST_ULN = 40`)
    with mandatory caveat:
    "Reduced diagnostic accuracy under age 35; screening context only, not a fibrosis clearance."
  - `NLR` (`Neutrophils (%) / Lymphocytes (%)`).
  - `WHR` (`waist_cm / hip_cm`) and `WHtR` (`waist_cm / 163.0`) from `Body_Measurements`.
"""

from __future__ import annotations

from datetime import date
import math
from typing import Any

import numpy as np
import pandas as pd

from health_engine.config_loader import add_display_date_col, validate_epistemic_class


FASTING_GLUCOSE_NAMES: frozenset[str] = frozenset(
    {
        "fasting blood glucose (fbs)",
        "fasting plasma glucose",
        "blood glucose fasting",
        "fbs",
    }
)

FASTING_INSULIN_NAMES: frozenset[str] = frozenset(
    {
        "fasting insulin",
        "serum insulin (fasting)",
        "insulin fasting",
    }
)

NON_FASTING_DISQUALIFIERS: tuple[str, ...] = (
    "random",
    "postprandial",
    "post prandial",
    "ppbs",
    "estimated average",
    "eag",
    "non-fasting",
    "non fasting",
    "2 hr",
    "2-hr",
)


def _is_verified_fasting_row(row: dict[str, Any], allowed_std_names: frozenset[str]) -> bool:
    std_nm = str(row.get("Standardized_Test_Name") or "").strip().lower()
    orig_nm = str(row.get("Original_Test_Name") or "").strip().lower()
    spec_nm = str(row.get("Specimen_Type") or "").strip().lower()
    if std_nm not in allowed_std_names:
        return False
    combined = f"{std_nm} | {orig_nm} | {spec_nm}"
    if any(tok in combined for tok in NON_FASTING_DISQUALIFIERS):
        return False
    return True


def _exact_age_years(dob_iso: str, target_iso: str) -> float:
    d0 = date.fromisoformat(dob_iso)
    d1 = date.fromisoformat(target_iso)
    return (d1 - d0).days / 365.2425


def build_gold_clinical_indices(
    silver_labs: pd.DataFrame,
    silver_body_meas: pd.DataFrame,
    thresholds_cfg: dict[str, Any],
) -> pd.DataFrame:
    """Compute longitudinal Gold_Clinical_Indices with strict fasting & date-matching provenance."""
    validate_epistemic_class("Screening_Context")
    dob_iso = str(thresholds_cfg["patient_profile"]["dob"])
    height_cm = float(thresholds_cfg["patient_profile"]["height_cm"])
    clin_cfg = thresholds_cfg["clinical_screening"]
    ast_uln = float(clin_cfg["ast_uln_u_l"])
    caveat_mentzer = str(clin_cfg["caveats"]["mentzer"])
    caveat_fib4_apri = str(clin_cfg["caveats"]["fib4_apri"])

    cols = [
        "date",
        "display_date",
        "index_name",
        "index_value",
        "unit",
        "fasting_verified",
        "age_at_test_years",
        "source_record_ids",
        "provenance_detail",
        "clinical_caveat",
        "epistemic_class",
    ]
    records: list[dict[str, Any]] = []

    # 1. Lab-derived longitudinal indices grouped strictly by exact Collection_Date
    if not silver_labs.empty:
        valid_labs = silver_labs.loc[
            (~silver_labs["data_corruption_flag"].fillna(False))
            & (silver_labs["Normalization_Valid_Flag"].fillna(False))
            & (silver_labs["Normalized_Value"].notna())
            & (silver_labs["Collection_Date"].notna())
        ].copy()

        for coll_date, grp in valid_labs.groupby("Collection_Date", sort=True):
            ds = str(coll_date)
            rows_list = grp.to_dict(orient="records")

            # Index normalized values + Record_IDs per standardized analyte on this exact date
            by_test: dict[str, dict[str, Any]] = {}
            fasting_gluc: dict[str, Any] | None = None
            fasting_ins: dict[str, Any] | None = None

            for r in rows_list:
                tname = str(r.get("Standardized_Test_Name") or "").strip().lower()
                val = float(r["Normalized_Value"])
                if val <= 0 and tname not in {"neutrophils (%)", "lymphocytes (%)"}:
                    continue
                by_test[tname] = r
                if _is_verified_fasting_row(r, FASTING_GLUCOSE_NAMES):
                    fasting_gluc = r
                if _is_verified_fasting_row(r, FASTING_INSULIN_NAMES):
                    fasting_ins = r

            age_yrs = round(_exact_age_years(dob_iso, ds), 3)

            # HOMA-IR: strictly same-date verified Fasting Glucose + Fasting Insulin
            if fasting_gluc is not None and fasting_ins is not None:
                fg_v = float(fasting_gluc["Normalized_Value"])
                fi_v = float(fasting_ins["Normalized_Value"])
                homa = (fg_v * fi_v) / 405.0
                records.append(
                    {
                        "date": ds,
                        "index_name": "HOMA_IR",
                        "index_value": round(homa, 4),
                        "unit": "index",
                        "fasting_verified": True,
                        "age_at_test_years": age_yrs,
                        "source_record_ids": f"{fasting_gluc.get('Record_ID','')}+{fasting_ins.get('Record_ID','')}",
                        "provenance_detail": f"Same-day fasting draw ({ds}): FBS={fg_v} mg/dL, Insulin={fi_v} µU/mL",
                        "clinical_caveat": "Insulin resistance screening index; interpret alongside HbA1c and clinical context.",
                        "epistemic_class": "Screening_Context",
                    }
                )

            # TyG Index: strictly same-date verified Fasting Glucose + Triglycerides
            tg_row = by_test.get("triglycerides")
            if fasting_gluc is not None and tg_row is not None:
                fg_v = float(fasting_gluc["Normalized_Value"])
                tg_v = float(tg_row["Normalized_Value"])
                if fg_v > 0 and tg_v > 0:
                    tyg = math.log((tg_v * fg_v) / 2.0)
                    records.append(
                        {
                            "date": ds,
                            "index_name": "TyG_Index",
                            "index_value": round(tyg, 4),
                            "unit": "ln(mg^2/dL^2)",
                            "fasting_verified": True,
                            "age_at_test_years": age_yrs,
                            "source_record_ids": f"{tg_row.get('Record_ID','')}+{fasting_gluc.get('Record_ID','')}",
                            "provenance_detail": f"Same-day fasting draw ({ds}): TG={tg_v} mg/dL, FBS={fg_v} mg/dL",
                            "clinical_caveat": "Surrogate insulin-sensitivity screening context; requires true fasting draw.",
                            "epistemic_class": "Screening_Context",
                        }
                    )

            # TG / HDL Ratio
            hdl_row = by_test.get("hdl cholesterol")
            if tg_row is not None and hdl_row is not None:
                tg_v = float(tg_row["Normalized_Value"])
                hdl_v = float(hdl_row["Normalized_Value"])
                if hdl_v > 0:
                    records.append(
                        {
                            "date": ds,
                            "index_name": "TG_HDL_Ratio",
                            "index_value": round(tg_v / hdl_v, 4),
                            "unit": "Ratio",
                            "fasting_verified": bool(fasting_gluc is not None),
                            "age_at_test_years": age_yrs,
                            "source_record_ids": f"{tg_row.get('Record_ID','')}+{hdl_row.get('Record_ID','')}",
                            "provenance_detail": f"TG={tg_v} mg/dL / HDL={hdl_v} mg/dL",
                            "clinical_caveat": "Atherogenic dyslipidemia screening ratio.",
                            "epistemic_class": "Screening_Context",
                        }
                    )

            # Non-HDL Cholesterol
            tchol_row = by_test.get("total cholesterol")
            if tchol_row is not None and hdl_row is not None:
                tc_v = float(tchol_row["Normalized_Value"])
                hdl_v = float(hdl_row["Normalized_Value"])
                records.append(
                    {
                        "date": ds,
                        "index_name": "Non_HDL_Cholesterol",
                        "index_value": round(tc_v - hdl_v, 2),
                        "unit": "mg/dL",
                        "fasting_verified": bool(fasting_gluc is not None),
                        "age_at_test_years": age_yrs,
                        "source_record_ids": f"{tchol_row.get('Record_ID','')}+{hdl_row.get('Record_ID','')}",
                        "provenance_detail": f"Total_Chol={tc_v} - HDL={hdl_v} mg/dL",
                        "clinical_caveat": "Total atherogenic lipoprotein cholesterol screening context.",
                        "epistemic_class": "Screening_Context",
                    }
                )

            # Mentzer Index (MCV / RBC Count)
            mcv_row = by_test.get("mcv")
            rbc_row = by_test.get("rbc count")
            if mcv_row is not None and rbc_row is not None:
                mcv_v = float(mcv_row["Normalized_Value"])
                rbc_v = float(rbc_row["Normalized_Value"])
                if rbc_v > 0:
                    records.append(
                        {
                            "date": ds,
                            "index_name": "Mentzer_Index",
                            "index_value": round(mcv_v / rbc_v, 4),
                            "unit": "fL / (10^6/µL)",
                            "fasting_verified": False,
                            "age_at_test_years": age_yrs,
                            "source_record_ids": f"{mcv_row.get('Record_ID','')}+{rbc_row.get('Record_ID','')}",
                            "provenance_detail": f"MCV={mcv_v} fL / RBC={rbc_v} 10^6/µL",
                            "clinical_caveat": caveat_mentzer,
                            "epistemic_class": "Screening_Context",
                        }
                    )

            # FIB-4 & APRI (using normalized Platelet Count in 10^3/µL and exact DOB age)
            ast_row = by_test.get("ast (sgot)")
            alt_row = by_test.get("alt (sgpt)")
            plt_row = by_test.get("platelet count")
            if ast_row is not None and plt_row is not None:
                ast_v = float(ast_row["Normalized_Value"])
                plt_v = float(plt_row["Normalized_Value"])  # guaranteed 10^3/µL via Silver Gate 3
                if plt_v > 0:
                    apri = ((ast_v / ast_uln) / plt_v) * 100.0
                    records.append(
                        {
                            "date": ds,
                            "index_name": "APRI",
                            "index_value": round(apri, 4),
                            "unit": "score",
                            "fasting_verified": False,
                            "age_at_test_years": age_yrs,
                            "source_record_ids": f"{ast_row.get('Record_ID','')}+{plt_row.get('Record_ID','')}",
                            "provenance_detail": f"AST={ast_v} U/L (ULN={ast_uln}), Platelets={plt_v} 10^3/µL",
                            "clinical_caveat": caveat_fib4_apri,
                            "epistemic_class": "Screening_Context",
                        }
                    )
                    if alt_row is not None:
                        alt_v = float(alt_row["Normalized_Value"])
                        if alt_v > 0:
                            fib4 = (age_yrs * ast_v) / (plt_v * math.sqrt(alt_v))
                            records.append(
                                {
                                    "date": ds,
                                    "index_name": "FIB4",
                                    "index_value": round(fib4, 4),
                                    "unit": "score",
                                    "fasting_verified": False,
                                    "age_at_test_years": age_yrs,
                                    "source_record_ids": (
                                        f"{ast_row.get('Record_ID','')}+{alt_row.get('Record_ID','')}+{plt_row.get('Record_ID','')}"
                                    ),
                                    "provenance_detail": (
                                        f"Age={age_yrs}y, AST={ast_v} U/L, ALT={alt_v} U/L, Platelets={plt_v} 10^3/µL"
                                    ),
                                    "clinical_caveat": caveat_fib4_apri,
                                    "epistemic_class": "Screening_Context",
                                }
                            )

            # NLR (Neutrophils % / Lymphocytes %)
            neut_row = by_test.get("neutrophils (%)")
            lymph_row = by_test.get("lymphocytes (%)")
            if neut_row is not None and lymph_row is not None:
                neut_v = float(neut_row["Normalized_Value"])
                lymph_v = float(lymph_row["Normalized_Value"])
                if lymph_v > 0:
                    records.append(
                        {
                            "date": ds,
                            "index_name": "NLR",
                            "index_value": round(neut_v / lymph_v, 4),
                            "unit": "Ratio",
                            "fasting_verified": False,
                            "age_at_test_years": age_yrs,
                            "source_record_ids": f"{neut_row.get('Record_ID','')}+{lymph_row.get('Record_ID','')}",
                            "provenance_detail": f"Neutrophils={neut_v}% / Lymphocytes={lymph_v}%",
                            "clinical_caveat": "Systemic inflammatory screening ratio; sensitive to acute infection or stress.",
                            "epistemic_class": "Screening_Context",
                        }
                    )

    # 2. Anthropometric Indices from Body_Measurements (WHR & WHtR with height = 163.0 cm)
    if not silver_body_meas.empty:
        bm = silver_body_meas.copy()
        bm["_waist"] = pd.to_numeric(bm.get("waist_cm", pd.Series(dtype=str)).replace("", np.nan), errors="coerce")
        bm["_hip"] = pd.to_numeric(bm.get("hip_cm", pd.Series(dtype=str)).replace("", np.nan), errors="coerce")
        for r in bm.sort_values("date").to_dict(orient="records"):
            ds = str(r.get("date") or "")
            if not ds:
                continue
            age_yrs = round(_exact_age_years(dob_iso, ds), 3)
            w_cm = r["_waist"]
            h_cm = r["_hip"]
            if pd.notna(w_cm) and float(w_cm) > 0:
                if pd.notna(h_cm) and float(h_cm) > 0:
                    records.append(
                        {
                            "date": ds,
                            "index_name": "WHR",
                            "index_value": round(float(w_cm) / float(h_cm), 4),
                            "unit": "Ratio",
                            "fasting_verified": False,
                            "age_at_test_years": age_yrs,
                            "source_record_ids": f"Body_Measurements:{ds}",
                            "provenance_detail": f"waist_cm={w_cm} / hip_cm={h_cm}",
                            "clinical_caveat": "Central adiposity tape-measurement screening ratio.",
                            "epistemic_class": "Screening_Context",
                        }
                    )
                records.append(
                    {
                        "date": ds,
                        "index_name": "WHtR",
                        "index_value": round(float(w_cm) / height_cm, 4),
                        "unit": "Ratio",
                        "fasting_verified": False,
                        "age_at_test_years": age_yrs,
                        "source_record_ids": f"Body_Measurements:{ds}",
                        "provenance_detail": f"waist_cm={w_cm} / height_cm={height_cm}",
                        "clinical_caveat": "Waist-to-height central cardiometabolic screening ratio.",
                        "epistemic_class": "Screening_Context",
                    }
                )

    if not records:
        return pd.DataFrame(columns=cols)

    out_df = pd.DataFrame(records).sort_values(["date", "index_name"]).reset_index(drop=True)
    out_df = add_display_date_col(out_df, "date")
    return out_df[cols]
