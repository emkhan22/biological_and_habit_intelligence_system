"""Privacy boundary & zero-leakage tests (AGENTS.md Rule 2)."""

from __future__ import annotations

import ast
from pathlib import Path
import pandas as pd
import pytest

from health_engine.io_adapter import (
    LocalCSVAdapter,
    PrivacyBoundaryViolationError,
    enforce_non_smoking_boundary,
)
from private_smoking.run_private import main as run_private_main
from private_smoking.silver_smoking import process_silver_smoking


def test_health_engine_structurally_blocks_smoking_log(tmp_path: Path) -> None:
    # Even if a Smoking_Log.csv file exists on disk, LocalCSVAdapter MUST raise PrivacyBoundaryViolationError
    mj_dir = tmp_path / "Micro Journal"
    mj_dir.mkdir(parents=True)
    (mj_dir / "Smoking_Log.csv").write_text("Date,Time,Event_Type,Count\n2026-11-22,10:00,Smoked,1\n")

    adapter = LocalCSVAdapter(tmp_path, tmp_path / "out")

    with pytest.raises(PrivacyBoundaryViolationError):
        adapter.read_source_tab("Micro Journal", "Smoking_Log")

    with pytest.raises(PrivacyBoundaryViolationError):
        adapter.write_derived_table("Silver_Smoking_Summary", pd.DataFrame({"a": [1]}))

    with pytest.raises(PrivacyBoundaryViolationError):
        enforce_non_smoking_boundary("Smoking_Private_Store", "Any_Tab")


def test_health_engine_static_ast_has_zero_private_smoking_imports() -> None:
    src_root = Path(__file__).resolve().parents[1] / "src" / "health_engine"
    py_files = sorted(src_root.rglob("*.py"))
    assert py_files, "Expected health_engine source files"

    for py_path in py_files:
        source_txt = py_path.read_text(encoding="utf-8")
        tree = ast.parse(source_txt, filename=str(py_path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith(
                        "private_smoking"
                    ), f"Forbidden import in {py_path}"
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                assert not mod.startswith(
                    "private_smoking"
                ), f"Forbidden from-import in {py_path}"


def test_private_smoking_zero_leakage_onset_vs_post_outcome_split_and_silent_stdout(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    priv_in = tmp_path / "priv_in" / "Micro Journal"
    priv_in.mkdir(parents=True)
    raw = pd.DataFrame(
        [
            {
                "Date": "2026-11-22",
                "Time": "16:15",
                "Event_Type": "Smoked",
                "Count": "1",
                "Craving_Intensity": "4",
                "Delay_Minutes": "18",
                "Trigger": "Work_Stress_Unique_Marker",
                "Mood": "Tense",
                "Barrier_or_Reason": "Stepped_Outside_Marker",
            },
            # Exact duplicate -> deduplicated
            {
                "Date": "2026-11-22",
                "Time": "16:15",
                "Event_Type": "Smoked",
                "Count": "1",
                "Craving_Intensity": "4",
                "Delay_Minutes": "18",
                "Trigger": "Work_Stress_Unique_Marker",
                "Mood": "Tense",
                "Barrier_or_Reason": "Stepped_Outside_Marker",
            },
            # Resisted craving (True_Zero)
            {
                "Date": "2026-11-23",
                "Time": "11:00",
                "Event_Type": "Craving Resisted",
                "Count": "0",
                "Craving_Intensity": "3",
                "Delay_Minutes": "25",
                "Trigger": "Coffee",
                "Mood": "Calm",
                "Barrier_or_Reason": "Deep breathing",
            },
        ]
    )
    raw.to_csv(priv_in / "Smoking_Log.csv", index=False)

    tables = process_silver_smoking(raw)
    onset = tables["Silver_Smoking_Onset_Context"]
    post = tables["Silver_Smoking_Post_Outcomes"]
    miss = tables["Silver_Smoking_Daily_Missingness"].set_index("date")

    assert len(onset) == 2
    assert len(post) == 2

    # Strictly zero post-event leakage columns in Onset Context
    forbidden_in_onset = {
        "delay_minutes_post",
        "Delay_Minutes",
        "count_smoked_post",
        "Count",
        "is_resisted_outcome",
        "barrier_or_reason_post",
        "Barrier_or_Reason",
    }
    assert forbidden_in_onset.isdisjoint(set(onset.columns))
    assert "delay_minutes_post" in post.columns
    assert miss.loc["2026-11-23", "smoking_log_status"] == "True_Zero"

    # CLI execution prints ZERO raw triggers, notes, or timestamps to stdout
    rc = run_private_main(
        ["--source", "csv", "--data-dir", str(tmp_path / "priv_in"), "--output-dir", str(tmp_path / "priv_out")]
    )
    assert rc == 0
    captured = capsys.readouterr()
    assert "Work_Stress_Unique_Marker" not in captured.out
    assert "Stepped_Outside_Marker" not in captured.out
    assert "16:15" not in captured.out
