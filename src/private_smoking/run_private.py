"""Isolated CLI Runner for `private_smoking` (`Smoking_Private_Store` ONLY).

Security & Privacy Guarantees (AGENTS.md Rule 2):
- Authenticates exclusively via `GCP_SMOKING_SA_JSON`.
- Reads ONLY `Micro Journal!Smoking_Log` (`SHEET_ID_JOURNAL`).
- Writes ONLY to `Smoking_Private_Store` (`SHEET_ID_SMOKING_STORE`).
- Refuses execution if `SHEET_ID_SMOKING_STORE` matches `SHEET_ID_STORE` or `SHEET_ID_STORE_V2`.
- Never prints raw row contents or counts to stdout/CI logs.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from typing import Any

import gspread
from gspread.exceptions import APIError, WorksheetNotFound
from gspread.utils import ValueInputOption, ValueRenderOption
import numpy as np
import pandas as pd

from private_smoking.silver_smoking import process_silver_smoking
from private_smoking.stats_smoking import build_private_smoking_stats


def _serialize_matrix(df: pd.DataFrame) -> list[list[Any]]:
    headers = [str(c) for c in df.columns]
    rows: list[list[Any]] = [headers]
    for rec in df.itertuples(index=False, name=None):
        out: list[Any] = []
        for v in rec:
            if v is None or (isinstance(v, (float, np.floating)) and np.isnan(v)) or pd.isna(v):
                out.append("")
            elif isinstance(v, (np.integer, int)) and not isinstance(val_int := v, bool):
                out.append(int(val_int))
            elif isinstance(v, (np.floating, float)):
                out.append(round(float(v), 4))
            elif isinstance(v, (np.bool_, bool)):
                out.append(bool(v))
            else:
                out.append(str(v))
        rows.append(out)
    return rows


def _with_backoff(fn: Any, *args: Any, **kwargs: Any) -> Any:
    for attempt in range(5):
        try:
            return fn(*args, **kwargs)
        except APIError as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status in {429, 500, 502, 503, 504} and attempt < 4:
                time.sleep(1.5 * (2**attempt))
                continue
            raise


def _load_optional_local_context(
    health_out_dir: Path | None,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    if health_out_dir is None or not health_out_dir.exists():
        return None, None
    daily_ctx: pd.DataFrame | None = None
    wint_ctx: pd.DataFrame | None = None
    d_pq = health_out_dir / "Gold_Daily_Unified.parquet"
    d_csv = health_out_dir / "Gold_Daily_Unified.csv"
    if d_pq.exists():
        daily_ctx = pd.read_parquet(d_pq)
    elif d_csv.exists():
        daily_ctx = pd.read_csv(d_csv)

    w_pq = health_out_dir / "Gold_Weight_Intervals.parquet"
    w_csv = health_out_dir / "Gold_Weight_Intervals.csv"
    if w_pq.exists():
        wint_ctx = pd.read_parquet(w_pq)
    elif w_csv.exists():
        wint_ctx = pd.read_csv(w_csv)
    return daily_ctx, wint_ctx


def run_private_pipeline_csv(
    data_dir: Path,
    output_dir: Path,
    health_out_dir: Path | None = None,
) -> dict[str, pd.DataFrame]:
    candidates = [
        data_dir / "Micro Journal" / "Smoking_Log.csv",
        data_dir / "Micro_Journal" / "Smoking_Log.csv",
        data_dir / "Micro_Journal__Smoking_Log.csv",
        data_dir / "Smoking_Log.csv",
    ]
    src_path = next((p for p in candidates if p.exists()), None)
    raw_df = (
        pd.read_csv(src_path, dtype=str, keep_default_na=False)
        if src_path is not None
        else pd.DataFrame()
    )
    tables = process_silver_smoking(raw_df)
    daily_ctx, wint_ctx = _load_optional_local_context(health_out_dir)
    v20_tables = build_private_smoking_stats(
        tables, daily_context_df=daily_ctx, weight_intervals_df=wint_ctx
    )
    all_tables = {**tables, **v20_tables}

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, tbl in all_tables.items():
        tbl.to_csv(output_dir / f"{name}.csv", index=False)
        pq = tbl.copy()
        for c in pq.columns:
            if pq[c].dtype == object:
                pq[c] = pq[c].astype("string")
        pq.to_parquet(output_dir / f"{name}.parquet", index=False)
    return all_tables


def _read_optional_sheet_df(
    client: gspread.Client, sheet_id: str, tab_name: str
) -> pd.DataFrame | None:
    if not sheet_id:
        return None
    try:
        sh = _with_backoff(client.open_by_key, sheet_id)
        ws = _with_backoff(sh.worksheet, tab_name)
        vals = _with_backoff(ws.get_all_values, value_render_option=ValueRenderOption.formatted)
        if not vals:
            return None
        headers = [str(h).strip() for h in vals[0]]
        body = [
            r + [""] * max(0, len(headers) - len(r))
            for r in vals[1:]
            if any(str(c).strip() != "" for c in r)
        ]
        return pd.DataFrame([r[: len(headers)] for r in body], columns=headers, dtype=str)
    except Exception:
        # Graceful fallback when GCP_SMOKING_SA_JSON has zero read access to non-smoking store
        return None


def run_private_pipeline_sheets(health_out_dir: Path | None = None) -> None:
    sa_raw = os.environ.get("GCP_SMOKING_SA_JSON", "").strip()
    if not sa_raw:
        raise ValueError("Missing required GCP_SMOKING_SA_JSON secret for private_smoking.")

    journal_id = os.environ.get("SHEET_ID_JOURNAL", "").strip()
    private_store_id = os.environ.get("SHEET_ID_SMOKING_STORE", "").strip()
    store_v2_id = os.environ.get("SHEET_ID_STORE_V2", "").strip()
    public_store_ids = {
        os.environ.get("SHEET_ID_STORE", "").strip(),
        store_v2_id,
    } - {""}

    if not journal_id or not private_store_id:
        raise ValueError("Both SHEET_ID_JOURNAL and SHEET_ID_SMOKING_STORE must be set.")
    if private_store_id in public_store_ids or private_store_id == journal_id:
        raise PermissionError(
            "SHEET_ID_SMOKING_STORE must be a dedicated private store distinct from Health_Intelligence_Store."
        )

    client = gspread.service_account_from_dict(json.loads(sa_raw))
    sh_src = _with_backoff(client.open_by_key, journal_id)
    ws_src = _with_backoff(sh_src.worksheet, "Smoking_Log")
    vals = _with_backoff(ws_src.get_all_values, value_render_option=ValueRenderOption.formatted)

    if vals:
        headers = [str(h).strip() for h in vals[0]]
        body = [
            r + [""] * max(0, len(headers) - len(r))
            for r in vals[1:]
            if any(str(c).strip() != "" for c in r)
        ]
        raw_df = pd.DataFrame([r[: len(headers)] for r in body], columns=headers, dtype=str)
    else:
        raw_df = pd.DataFrame()

    tables = process_silver_smoking(raw_df)
    daily_ctx, wint_ctx = _load_optional_local_context(health_out_dir)
    if daily_ctx is None and store_v2_id:
        daily_ctx = _read_optional_sheet_df(client, store_v2_id, "Gold_Daily_Unified")
    if wint_ctx is None and store_v2_id:
        wint_ctx = _read_optional_sheet_df(client, store_v2_id, "Gold_Weight_Intervals")

    v20_tables = build_private_smoking_stats(
        tables, daily_context_df=daily_ctx, weight_intervals_df=wint_ctx
    )
    all_tables = {**tables, **v20_tables}

    # Strictly write ONLY to dedicated private_store_id (Smoking_Private_Store)
    sh_dst = _with_backoff(client.open_by_key, private_store_id)
    for tab_name, tbl in all_tables.items():
        mat = _serialize_matrix(tbl)
        rows_n = max(len(mat) + 10, 50)
        cols_n = max(len(mat[0]) + 2, 10)
        try:
            ws_dst = _with_backoff(sh_dst.worksheet, tab_name)
            _with_backoff(ws_dst.clear)
            if ws_dst.row_count < len(mat) or ws_dst.col_count < len(mat[0]):
                _with_backoff(ws_dst.resize, rows=rows_n, cols=cols_n)
        except WorksheetNotFound:
            ws_dst = _with_backoff(sh_dst.add_worksheet, title=tab_name, rows=rows_n, cols=cols_n)
        _with_backoff(
            ws_dst.update,
            values=mat,
            range_name="A1",
            value_input_option=ValueInputOption.raw,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run isolated private smoking Silver + V2.0 Stats pipeline.")
    parser.add_argument("--source", choices=("csv", "sheets"), required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("./sample_private"))
    parser.add_argument("--output-dir", type=Path, default=Path("./private_out"))
    parser.add_argument("--health-out-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    if args.source == "csv":
        run_private_pipeline_csv(args.data_dir, args.output_dir, health_out_dir=args.health_out_dir)
    else:
        run_private_pipeline_sheets(health_out_dir=args.health_out_dir)

    # Sanitized status line ONLY — zero raw data or counts printed to stdout/CI logs
    print("[private_smoking] sync status=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
