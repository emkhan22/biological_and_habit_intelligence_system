"""Physically isolated private smoking pipeline (`Smoking_Private_Store` only).

AGENTS.md Rule 2:
- Uses a separate Google Service Account (`GCP_SMOKING_SA_JSON`).
- Reads ONLY `Micro Journal!Smoking_Log`.
- Writes ONLY to `Smoking_Private_Store` (never `Health_Intelligence_Store` or v2).
- Never prints raw rows or text fields to stdout/CI logs.
"""

__version__ = "1.5.0"
