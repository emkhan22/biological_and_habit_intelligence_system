"""Configuration loader, IST localization helpers, and epistemic contract constants."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import yaml

IST = ZoneInfo("Asia/Kolkata")

EPISTEMIC_CLASSES: frozenset[str] = frozenset(
    {
        "Measured",
        "Descriptive",
        "Correlation",
        "Prediction",
        "Causal_Hypothesis",
        "Screening_Context",
    }
)

MISSING_REASONS: frozenset[str] = frozenset(
    {
        "Not_Worn",
        "Not_Logged",
        "Source_Gap",
        "True_Zero",
    }
)

STREAM_STATUSES: frozenset[str] = MISSING_REASONS | frozenset({"Present"})


def repo_root() -> Path:
    """Return repository root containing config/."""
    return Path(__file__).resolve().parents[2]


def load_yaml_config(path: Path) -> dict[str, Any]:
    """Load a YAML file safely into a dictionary."""
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in YAML config: {path}")
    return data


def load_engine_configs(config_dir: Path | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load (thresholds_cfg, lab_norm_cfg) from config directory."""
    base = config_dir or (repo_root() / "config")
    thresholds = load_yaml_config(base / "thresholds.yaml")
    lab_norm = load_yaml_config(base / "lab_unit_normalization.yaml")
    return thresholds, lab_norm


def load_treatments_config(config_dir: Path | None = None) -> dict[str, Any]:
    """Load treatments.yaml registry from config directory."""
    base = config_dir or (repo_root() / "config")
    tpath = base / "treatments.yaml"
    if not tpath.exists():
        tpath = repo_root() / "config" / "treatments.yaml"
    return load_yaml_config(tpath)


def load_journal_themes_config(config_dir: Path | None = None) -> dict[str, Any]:
    """Load journal_themes.yaml deterministic NLP taxonomy from config directory."""
    base = config_dir or (repo_root() / "config")
    jpath = base / "journal_themes.yaml"
    if not jpath.exists():
        jpath = repo_root() / "config" / "journal_themes.yaml"
    return load_yaml_config(jpath)


def load_ai_routing_config(config_dir: Path | None = None) -> dict[str, Any]:
    """Load ai_routing.yaml model routing & ₹500 budget circuit breaker configuration."""
    base = config_dir or (repo_root() / "config")
    apath = base / "ai_routing.yaml"
    if not apath.exists():
        apath = repo_root() / "config" / "ai_routing.yaml"
    return load_yaml_config(apath)


def now_ist() -> datetime:
    """Return current timezone-aware timestamp in Asia/Kolkata."""
    return datetime.now(tz=IST)


def now_ist_str() -> str:
    """Return formatted IST audit timestamp (DD-MM-YYYY HH:MM:SS IST)."""
    return now_ist().strftime("%d-%m-%Y %H:%M:%S IST")


def to_iso_date_str(val: Any) -> str | None:
    """Normalize a date/timestamp string or object into ISO YYYY-MM-DD (IST)."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    if isinstance(val, datetime):
        dt = val.astimezone(IST) if val.tzinfo else val
        return dt.strftime("%Y-%m-%d")
    if isinstance(val, date):
        return val.strftime("%Y-%m-%d")
    s = str(val).strip()
    if not s or s.lower() in {"nan", "none", "nat"}:
        return None
    # Fast path for YYYY-MM-DD(...)
    if len(s) >= 10 and s[4] == "-" and s[7] == "-":
        candidate = s[:10]
        try:
            date.fromisoformat(candidate)
            return candidate
        except ValueError:
            pass
    # Handle DD-MM-YYYY
    if len(s) >= 10 and s[2] == "-" and s[5] == "-":
        try:
            d = datetime.strptime(s[:10], "%d-%m-%Y").date()
            return d.isoformat()
        except ValueError:
            pass
    parsed = pd.to_datetime(s, errors="coerce")
    if pd.isna(parsed):
        return None
    if parsed.tzinfo is not None:
        parsed = parsed.tz_convert(IST)
    return parsed.strftime("%Y-%m-%d")


def iso_to_display_date(iso_date: str | None) -> str | None:
    """Convert internal ISO YYYY-MM-DD date to Indian display format DD-MM-YYYY."""
    if not iso_date or pd.isna(iso_date):
        return None
    s = str(iso_date).strip()[:10]
    if len(s) == 10 and s[4] == "-" and s[7] == "-":
        return f"{s[8:10]}-{s[5:7]}-{s[0:4]}"
    return None


def add_display_date_col(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Ensure a DD-MM-YYYY display_date column sits immediately beside ISO date."""
    out = df.copy()
    if date_col not in out.columns:
        return out
    disp = out[date_col].map(iso_to_display_date)
    if "display_date" in out.columns:
        out["display_date"] = disp
        return out
    idx = list(out.columns).index(date_col) + 1
    out.insert(idx, "display_date", disp)
    return out


def validate_epistemic_class(label: str) -> str:
    """Validate that an epistemic honesty label belongs to AGENTS.md Rule 3 enum."""
    if label not in EPISTEMIC_CLASSES:
        raise ValueError(
            f"Invalid epistemic_class '{label}'. Must be one of {sorted(EPISTEMIC_CLASSES)}"
        )
    return label
