"""Export validated non-smoking ./build_out tables into frontend/public/mock_data.json.

Strictly enforces `assert_packet_smoking_free` so zero smoking data can ever enter
the public Clean Mode frontend mock bundle.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from health_engine.ai.packet_builder import assert_packet_smoking_free

FRONTEND_TABLES: tuple[str, ...] = (
    "Gold_Daily_Unified",
    "Gold_Personal_Baselines",
    "Gold_Sleep_HR_Spikes",
    "Gold_Clinical_Indices",
    "Gold_Med_Bioavailability",
    "Gold_Weight_Intervals",
    "Gold_Weight_Trend",
    "Gold_Weight_Clinical_Link",
    "Gold_Treatment_Scorecards",
    "Gold_Stats_Correlations",
    "Gold_Tirz_Folded_Wave",
    "Gold_Cardiac_Efficiency",
    "Gold_Cardiac_Efficiency_Buckets",
    "Gold_Cycle_Phase_Stats",
    "Gold_Episode_Chains",
    "Gold_Habit_Durability",
    "Gold_Sweet_Hunger_Analysis",
    "Gold_Calorie_Balance_Audit",
    "Gold_ML_Model_Summary",
    "Gold_ML_Feature_Drivers",
    "Gold_Daily_Phenotypes",
    "Gold_Phenotype_Profiles",
    "Gold_Journal_Themes",
    "Gold_Journal_Themes_AI",
    "Gold_Meal_Bioavailability_AI",
    "Gold_Episode_Rules",
    "Stats_Run_Audit",
    "Model_Run_Audit",
    "AI_Insights_Cache",
    "AI_Weekly_Briefs",
    "AI_Monthly_Briefs",
    "AI_Peer_Expectedness",
    "AI_Doctor_Briefs",
    "AI_Research_Pulse",
    "AI_Cost_Ledger",
    "Silver_Lab_Results",
)

ROW_LIMITS: dict[str, int] = {
    "Gold_Daily_Unified": 90,
    "Gold_Personal_Baselines": 90,
    "Gold_Sleep_HR_Spikes": 90,
    "Gold_Weight_Trend": 90,
    "Gold_Cardiac_Efficiency": 90,
    "Gold_Daily_Phenotypes": 90,
}


def _clean_val(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, (float, np.floating)):
        if math.isnan(float(v)) or math.isinf(float(v)):
            return None
        return round(float(v), 4)
    if isinstance(v, (np.integer, int)) and not isinstance(v, bool):
        return int(v)
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if pd.isna(v):
        return None
    s = str(v)
    if s.strip().lower() in {"nan", "nat", "<na>"}:
        return None
    return s


def export_mock_bundle(build_dir: Path, out_file: Path) -> None:
    payload: dict[str, Any] = {"source": "Health_Intelligence_Store_v2_Local_Mock", "tables": {}}
    packet_path = build_dir / "AI_Briefing_Packet.json"
    if packet_path.exists():
        payload["packet"] = json.loads(packet_path.read_text(encoding="utf-8"))

    for tname in FRONTEND_TABLES:
        csv_p = build_dir / f"{tname}.csv"
        if not csv_p.exists():
            payload["tables"][tname] = []
            continue
        df = pd.read_csv(csv_p)
        lim = ROW_LIMITS.get(tname)
        if lim is not None and len(df) > lim:
            df = df.tail(lim)
        payload["tables"][tname] = [
            {str(k): _clean_val(v) for k, v in rec.items()}
            for rec in df.to_dict(orient="records")
        ]

    assert_packet_smoking_free(payload, path="frontend_mock_data")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[export_frontend_mock] Wrote {out_file} ({out_file.stat().st_size:,} bytes)")


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    export_mock_bundle(root / "build_out", root / "frontend" / "public" / "mock_data.json")
