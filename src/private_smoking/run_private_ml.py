"""Isolated CLI Runner for V2.5 Private Smoking ML (`Smoking_Private_Store` ONLY).

Security & Privacy Guarantees (AGENTS.md Rule 2):
- Authenticates exclusively via `GCP_SMOKING_SA_JSON`.
- Writes ONLY to `Smoking_Private_Store` (`SHEET_ID_SMOKING_STORE`) or local private `--output-dir`.
- Refuses execution if `SHEET_ID_SMOKING_STORE` matches `SHEET_ID_STORE` or `SHEET_ID_STORE_V2`.
- Never prints raw row contents, triggers, timestamps, or counts to stdout/CI logs.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import gspread
from gspread.exceptions import WorksheetNotFound
from gspread.utils import ValueInputOption, ValueRenderOption
import pandas as pd

from private_smoking.ml_smoking import build_private_smoking_ml
from private_smoking.run_private import (
    _load_optional_local_context,
    _read_optional_sheet_df,
    _serialize_matrix,
    _with_backoff,
)
from private_smoking.silver_smoking import process_silver_smoking


def run_private_ml_csv(
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
    silver_tables = process_silver_smoking(raw_df)
    daily_ctx, _ = _load_optional_local_context(health_out_dir)
    ml_tables = build_private_smoking_ml(silver_tables, daily_context_df=daily_ctx)

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, tbl in ml_tables.items():
        tbl.to_csv(output_dir / f"{name}.csv", index=False)
        pq = tbl.copy()
        for c in pq.columns:
            if pq[c].dtype == object:
                pq[c] = pq[c].astype("string")
        pq.to_parquet(output_dir / f"{name}.parquet", index=False)
    return ml_tables


def run_private_ml_sheets(health_out_dir: Path | None = None) -> None:
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

    silver_tables = process_silver_smoking(raw_df)
    daily_ctx, _ = _load_optional_local_context(health_out_dir)
    if daily_ctx is None and store_v2_id:
        daily_ctx = _read_optional_sheet_df(client, store_v2_id, "Gold_Daily_Unified")

    ml_tables = build_private_smoking_ml(silver_tables, daily_context_df=daily_ctx)

    sh_dst = _with_backoff(client.open_by_key, private_store_id)
    ws_list = _with_backoff(sh_dst.worksheets)
    ws_map = {ws.title: ws for ws in ws_list}

    for tab_name, tbl in ml_tables.items():
        mat = _serialize_matrix(tbl)
        rows_n = max(len(mat) + 10, 50)
        cols_n = max(len(mat[0]) + 2, 10)
        ws_dst = ws_map.get(tab_name)
        if ws_dst is not None:
            _with_backoff(ws_dst.clear)
            if ws_dst.row_count < len(mat) or ws_dst.col_count < len(mat[0]):
                _with_backoff(ws_dst.resize, rows=rows_n, cols=cols_n)
        else:
            ws_dst = _with_backoff(sh_dst.add_worksheet, title=tab_name, rows=rows_n, cols=cols_n)
            ws_map[tab_name] = ws_dst
        _with_backoff(
            ws_dst.update,
            values=mat,
            range_name="A1",
            value_input_option=ValueInputOption.raw,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run isolated V2.5 Private Smoking ML pipeline.")
    parser.add_argument("--source", choices=("csv", "sheets"), required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("./sample_private"))
    parser.add_argument("--output-dir", type=Path, default=Path("./private_out"))
    parser.add_argument("--health-out-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    if args.source == "csv":
        run_private_ml_csv(args.data_dir, args.output_dir, health_out_dir=args.health_out_dir)
    else:
        run_private_ml_sheets(health_out_dir=args.health_out_dir)

    # Sanitized status line ONLY — zero raw data or counts printed to stdout/CI logs
    print("[private_smoking ml] sync status=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
