# Personal Health Intelligence System (V1.5 Silver + Gold Engine)

Deterministic Python 3.11+ Silver Cleaning Gate and V1.5 Gold Statistical Foundations engine governed by [AGENTS.md](./AGENTS.md).

## Architecture & Data Contracts
- **Bronze Sources (Read-Only):** `HealthConnectDaily`, `Glapp Data`, `AggregatedMedicalRecords`, and non-smoking tabs of `Micro Journal` (`Meds_Log`, `Symptoms_Log`, `Journal_Mood`).
- **Target Store:** `Health_Intelligence_Store_v2` (`SHEET_ID_STORE_V2`) + local CSV/Parquet mirrors.
- **Air-Gapped Private Pipeline (`src/private_smoking/`):** Physically isolated package and workflow (`.github/workflows/private_smoking_pipeline.yml`) using `GCP_SMOKING_SA_JSON` to read `Micro Journal!Smoking_Log` and write exclusively to `Smoking_Private_Store`.

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Run unit & privacy-boundary test suite
pytest -v

# Local offline CSV execution
python -m health_engine.run_v15 --source csv --data-dir ./sample_data --output-dir ./build_out

# Live Google Sheets execution
python -m health_engine.run_v15 --source sheets --output-dir ./build_out
```