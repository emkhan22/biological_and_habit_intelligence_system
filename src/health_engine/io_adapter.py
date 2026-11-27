"""Dual CSV + Google Sheets I/O Adapter with strict Smoking Privacy Boundary.

AGENTS.md Rule 2 enforcement:
- `Micro Journal!Smoking_Log` and `Smoking_Private_Store` are structurally blocked
  across ALL methods in this module. Any attempt to read or write a tab or workbook
  referencing smoking raises `PrivacyBoundaryViolationError` immediately.
"""

from __future__ import annotations

import json
import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import gspread
from gspread.exceptions import APIError, WorksheetNotFound
from gspread.utils import ValueInputOption, ValueRenderOption
import numpy as np
import pandas as pd


class PrivacyBoundaryViolationError(PermissionError):
    """Raised when the main health pipeline attempts to touch isolated smoking data."""


# Explicit allowlist of source workbooks and tabs permitted in the main health pipeline
ALLOWED_SOURCE_TABS: dict[str, frozenset[str]] = {
    "HealthConnectDaily": frozenset(
        {
            "Steps",
            "Sleep",
            "Sleep_Sessions",
            "Sleep_Stage_Timeline",
            "Sleep_HR_Events",
            "Vitals",
            "Heart_Rate_Hourly",
            "Steps_Hourly",
            "Activity_Daily",
            "Weekly_Cardio",
            "Workouts",
            "Nutrition",
            "Nutrition_Log",
            "Weight",
            "Body_Composition",
            "Body_Measurements",
            "Cycles",
            "Menstrual_Cycle",
            "Period_Flow_Daily",
            "Archived_Menstrual_Cycle",
            "ETL_Audit",
            "Data_Quality",
        }
    ),
    "Glapp Data": frozenset({"shots", "weight", "mood", "side_effects"}),
    "AggregatedMedicalRecords": frozenset(
        {
            "Lab_Results",
            "Lab_Reports",
            "Diagnoses",
            "Medications",
            "Prescriptions",
            "Procedures",
            "Extraction_Review",
        }
    ),
    "Micro Journal": frozenset({"Meds_Log", "Symptoms_Log", "Journal_Mood"}),
}

FORBIDDEN_SUBSTRINGS: tuple[str, ...] = ("smoking", "smoke_log", "smoking_private")


def enforce_non_smoking_boundary(workbook: str, tab: str) -> None:
    """Raise PrivacyBoundaryViolationError if workbook or tab touches smoking data."""
    wb_norm = str(workbook or "").strip()
    tab_norm = str(tab or "").strip()
    combined = f"{wb_norm}!{tab_norm}".lower()
    for token in FORBIDDEN_SUBSTRINGS:
        if token in combined:
            raise PrivacyBoundaryViolationError(
                f"AGENTS.md Rule 2 violation: '{wb_norm}!{tab_norm}' is strictly isolated "
                "from health_engine and may only be accessed by private_smoking."
            )


def sanitize_df_for_sheets(df: pd.DataFrame) -> list[list[Any]]:
    """Serialize a DataFrame to a 2D list preserving NaN as empty string (never 0)."""
    headers = [str(c) for c in df.columns]
    if df.empty:
        return [headers]
    rows: list[list[Any]] = [headers]
    for record in df.itertuples(index=False, name=None):
        row_out: list[Any] = []
        for val in record:
            if val is None or (isinstance(val, (float, np.floating)) and np.isnan(val)):
                row_out.append("")
            elif pd.isna(val):
                row_out.append("")
            elif isinstance(val, (np.integer, int)) and not isinstance(val, bool):
                row_out.append(int(val))
            elif isinstance(val, (np.floating, float)):
                row_out.append(round(float(val), 6))
            elif isinstance(val, (np.bool_, bool)):
                row_out.append(bool(val))
            else:
                row_out.append(str(val))
        rows.append(row_out)
    return rows


class HealthIOAdapter(ABC):
    """Abstract Dual I/O Adapter for reading Bronze sources and writing Silver/Gold tables."""

    @abstractmethod
    def read_source_tab(self, workbook: str, tab: str, required: bool = False) -> pd.DataFrame:
        """Read a source tab after enforcing the non-smoking boundary."""

    @abstractmethod
    def write_derived_table(self, tab_name: str, df: pd.DataFrame) -> None:
        """Write a validated Silver, Gold, or QA table."""

    def read_derived_table(self, tab_name: str) -> pd.DataFrame:
        """Read an existing derived Silver, Gold, AI, or Ledger table if present (else empty DataFrame)."""
        enforce_non_smoking_boundary("Health_Intelligence_Store_v2", tab_name)
        return pd.DataFrame()


class LocalCSVAdapter(HealthIOAdapter):
    """Offline CSV/Parquet adapter reading from --data-dir and exporting to --output-dir."""

    def __init__(self, data_dir: Path, output_dir: Path | None = None) -> None:
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir) if output_dir else (self.data_dir / "out")

    def _resolve_csv_path(self, workbook: str, tab: str) -> Path | None:
        wb_slug = workbook.replace(" ", "_")
        candidates = [
            self.data_dir / workbook / f"{tab}.csv",
            self.data_dir / wb_slug / f"{tab}.csv",
            self.data_dir / f"{workbook}__{tab}.csv",
            self.data_dir / f"{wb_slug}__{tab}.csv",
            self.data_dir / f"{tab}.csv",
        ]
        for path in candidates:
            if path.exists():
                return path
        return None

    def read_source_tab(self, workbook: str, tab: str, required: bool = False) -> pd.DataFrame:
        enforce_non_smoking_boundary(workbook, tab)
        allowed = ALLOWED_SOURCE_TABS.get(workbook)
        if allowed is None or tab not in allowed:
            raise PrivacyBoundaryViolationError(
                f"Tab '{workbook}!{tab}' is not on the health_engine source allowlist."
            )
        csv_path = self._resolve_csv_path(workbook, tab)
        if csv_path is None:
            if required:
                raise FileNotFoundError(
                    f"Required CSV fixture not found for {workbook}!{tab} in {self.data_dir}"
                )
            return pd.DataFrame()
        # dtype=str prevents pandas from silently coercing "2-5" or leading zeros;
        # downstream Silver modules parse exact numeric/timestamp columns explicitly.
        return pd.read_csv(csv_path, dtype=str, keep_default_na=False)

    def read_derived_table(self, tab_name: str) -> pd.DataFrame:
        enforce_non_smoking_boundary("Health_Intelligence_Store_v2", tab_name)
        pq_path = self.output_dir / f"{tab_name}.parquet"
        csv_path = self.output_dir / f"{tab_name}.csv"
        if pq_path.exists():
            return pd.read_parquet(pq_path)
        if csv_path.exists():
            return pd.read_csv(csv_path)
        return pd.DataFrame()

    def write_derived_table(self, tab_name: str, df: pd.DataFrame) -> None:
        enforce_non_smoking_boundary("Health_Intelligence_Store_v2", tab_name)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        csv_out = self.output_dir / f"{tab_name}.csv"
        parquet_out = self.output_dir / f"{tab_name}.parquet"
        df.to_csv(csv_out, index=False)
        # Normalize object columns cleanly for Arrow Parquet serialization
        pq_df = df.copy()
        for col in pq_df.columns:
            if pq_df[col].dtype == object:
                pq_df[col] = pq_df[col].astype("string")
        pq_df.to_parquet(parquet_out, index=False)


class GoogleSheetsAdapter(HealthIOAdapter):
    """Live Google Sheets I/O adapter via gspread 6.x with rate-limit exponential backoff."""

    DEFAULT_SOURCE_ENV_KEYS: dict[str, str] = {
        "HealthConnectDaily": "SHEET_ID_HC",
        "Glapp Data": "SHEET_ID_GLAPP",
        "AggregatedMedicalRecords": "SHEET_ID_MEDICAL",
        "Micro Journal": "SHEET_ID_JOURNAL",
    }

    def __init__(
        self,
        sa_json_str: str | None = None,
        source_sheet_ids: dict[str, str] | None = None,
        target_store_id: str | None = None,
        local_mirror_dir: Path | None = None,
        max_retries: int = 7,
        base_delay_sec: float = 2.5,
    ) -> None:
        if os.environ.get("GCP_SMOKING_SA_JSON") and not sa_json_str and not os.environ.get("GCP_HEALTH_SA_JSON"):
            raise PrivacyBoundaryViolationError(
                "health_engine refuses to fall back to GCP_SMOKING_SA_JSON."
            )
        raw_creds = sa_json_str or os.environ.get("GCP_HEALTH_SA_JSON", "")
        if not raw_creds:
            raise ValueError(
                "Missing GCP_HEALTH_SA_JSON service account credentials for GoogleSheetsAdapter."
            )
        creds_dict = json.loads(raw_creds)
        self.client = gspread.service_account_from_dict(creds_dict)
        self.source_ids = source_sheet_ids or {
            wb: os.environ.get(env_key, "").strip()
            for wb, env_key in self.DEFAULT_SOURCE_ENV_KEYS.items()
        }
        self.target_store_id = (
            target_store_id
            or os.environ.get("SHEET_ID_STORE_V2", "").strip()
            or os.environ.get("SHEET_ID_STORE", "").strip()
        )
        self.local_mirror_dir = Path(local_mirror_dir) if local_mirror_dir else None
        self.max_retries = max_retries
        self.base_delay_sec = base_delay_sec
        self._spreadsheet_cache: dict[str, gspread.Spreadsheet] = {}
        self._worksheet_cache: dict[str, dict[str, gspread.Worksheet]] = {}

    def _with_backoff(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        for attempt in range(self.max_retries):
            try:
                return fn(*args, **kwargs)
            except APIError as exc:
                status = getattr(getattr(exc, "response", None), "status_code", None)
                if status in {429, 500, 502, 503, 504} and attempt < self.max_retries - 1:
                    sleep_s = min(65.0, self.base_delay_sec * (2**attempt))
                    time.sleep(sleep_s)
                    continue
                raise

    def _open_by_id(self, sheet_id: str) -> gspread.Spreadsheet:
        if not sheet_id:
            raise ValueError("Spreadsheet ID is empty.")
        if sheet_id not in self._spreadsheet_cache:
            self._spreadsheet_cache[sheet_id] = self._with_backoff(
                self.client.open_by_key, sheet_id
            )
        return self._spreadsheet_cache[sheet_id]

    def _get_worksheet_map(self, sheet_id: str) -> dict[str, gspread.Worksheet]:
        if sheet_id not in self._worksheet_cache:
            sh = self._open_by_id(sheet_id)
            ws_list = self._with_backoff(sh.worksheets)
            self._worksheet_cache[sheet_id] = {ws.title: ws for ws in ws_list}
        return self._worksheet_cache[sheet_id]

    def read_source_tab(self, workbook: str, tab: str, required: bool = False) -> pd.DataFrame:
        enforce_non_smoking_boundary(workbook, tab)
        allowed = ALLOWED_SOURCE_TABS.get(workbook)
        if allowed is None or tab not in allowed:
            raise PrivacyBoundaryViolationError(
                f"Tab '{workbook}!{tab}' is not on the health_engine source allowlist."
            )
        sheet_id = self.source_ids.get(workbook, "")
        if not sheet_id:
            if required:
                raise ValueError(f"Missing Sheet ID for source workbook '{workbook}'.")
            return pd.DataFrame()
        ws_map = self._get_worksheet_map(sheet_id)
        ws = ws_map.get(tab)
        if ws is None:
            if required:
                raise WorksheetNotFound(tab)
            return pd.DataFrame()
        values = self._with_backoff(
            ws.get_all_values,
            value_render_option=ValueRenderOption.formatted,
        )
        if not values:
            return pd.DataFrame()
        headers = [str(h).strip() for h in values[0]]
        data_rows = [
            r + [""] * max(0, len(headers) - len(r))
            for r in values[1:]
            if any(str(cell).strip() != "" for cell in r)
        ]
        return pd.DataFrame([r[: len(headers)] for r in data_rows], columns=headers, dtype=str)

    def read_derived_table(self, tab_name: str) -> pd.DataFrame:
        enforce_non_smoking_boundary("Health_Intelligence_Store_v2", tab_name)
        if not self.target_store_id:
            if self.local_mirror_dir is not None:
                return LocalCSVAdapter(self.local_mirror_dir, self.local_mirror_dir).read_derived_table(
                    tab_name
                )
            return pd.DataFrame()
        ws_map = self._get_worksheet_map(self.target_store_id)
        ws = ws_map.get(tab_name)
        if ws is None:
            if self.local_mirror_dir is not None:
                return LocalCSVAdapter(self.local_mirror_dir, self.local_mirror_dir).read_derived_table(
                    tab_name
                )
            return pd.DataFrame()
        values = self._with_backoff(
            ws.get_all_values,
            value_render_option=ValueRenderOption.formatted,
        )
        if not values:
            return pd.DataFrame()
        headers = [str(h).strip() for h in values[0]]
        data_rows = [
            r + [""] * max(0, len(headers) - len(r))
            for r in values[1:]
            if any(str(cell).strip() != "" for cell in r)
        ]
        return pd.DataFrame([r[: len(headers)] for r in data_rows], columns=headers)

    def write_derived_table(self, tab_name: str, df: pd.DataFrame) -> None:
        enforce_non_smoking_boundary("Health_Intelligence_Store_v2", tab_name)
        if self.local_mirror_dir is not None:
            LocalCSVAdapter(self.local_mirror_dir, self.local_mirror_dir).write_derived_table(
                tab_name, df
            )
        if not self.target_store_id:
            raise ValueError("Missing SHEET_ID_STORE_V2 for writing derived tables.")
        sh = self._open_by_id(self.target_store_id)
        ws_map = self._get_worksheet_map(self.target_store_id)
        matrix = sanitize_df_for_sheets(df)
        rows_needed = max(len(matrix) + 10, 50)
        cols_needed = max(len(matrix[0]) + 2, 10)
        ws = ws_map.get(tab_name)
        if ws is not None:
            self._with_backoff(ws.clear)
            if ws.row_count < len(matrix) or ws.col_count < len(matrix[0]):
                self._with_backoff(ws.resize, rows=rows_needed, cols=cols_needed)
        else:
            ws = self._with_backoff(
                sh.add_worksheet,
                title=tab_name,
                rows=rows_needed,
                cols=cols_needed,
            )
            ws_map[tab_name] = ws
        self._with_backoff(
            ws.update,
            values=matrix,
            range_name="A1",
            value_input_option=ValueInputOption.raw,
        )
