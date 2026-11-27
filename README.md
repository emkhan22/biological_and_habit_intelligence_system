# Personal Health Intelligence System (V1.5 Silver/Gold + V2.0 Stats + V2.5 ML + V2.8 AI Synthesis)

Deterministic Python 3.11+ Silver Cleaning Gate, V1.5 Gold Feature Store, V2.0 Statistical Inference Engine, V2.5 Progressive ML Engine, and V2.8 Model-Routed AI Synthesis Layer (`google-genai` Gemini 4 / 3.5 with ₹500/month INR circuit breaker) governed by [AGENTS.md](./AGENTS.md).

## Architecture & Data Contracts
- **Bronze Sources (Read-Only):** `HealthConnectDaily`, `Glapp Data`, `AggregatedMedicalRecords`, and non-smoking tabs of `Micro Journal` (`Meds_Log`, `Symptoms_Log`, `Journal_Mood`).
- **Target Store:** `Health_Intelligence_Store_v2` (`SHEET_ID_STORE_V2`) + local CSV/Parquet mirrors + smoking-free `AI_Briefing_Packet.json`.
- **Air-Gapped Private Pipeline (`src/private_smoking/`):** Physically isolated package and workflow (`.github/workflows/private_smoking_pipeline.yml`) using `GCP_SMOKING_SA_JSON` to read `Micro Journal!Smoking_Log` and write Silver, V2.0 statistical summaries, V2.5 onset-only behavioral ML, and V2.8 weekly behavioral friction coaching exclusively to `Smoking_Private_Store`.

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Run unit, statistical, ML, AI, & privacy-boundary test suite
pytest -v

# Local offline CSV execution (V1.5 -> V2.0 -> V2.5 -> V2.8 AI Dry-Run)
python -m health_engine.run_v25 --source csv --data-dir ./sample_data --output-dir ./build_out
python -m health_engine.run_v28_ai --source csv --data-dir ./sample_data --output-dir ./build_out --reuse-local-mirror --daily --weekly --monthly --doctor-prep Endocrinology --dry-run
python -m private_smoking.run_private_ml --source csv --data-dir ./sample_data --output-dir ./private_out --health-out-dir ./build_out
python -m private_smoking.run_private_ai --source csv --data-dir ./sample_data --output-dir ./private_out --dry-run

# Live Google Sheets execution (single Bronze read via local Silver mirror)
python -m health_engine.run_v15 --source sheets --output-dir ./build_out
python -m health_engine.run_v20 --source sheets --output-dir ./build_out --skip-v15-persist --reuse-local-mirror
python -m health_engine.run_v25 --source sheets --output-dir ./build_out --skip-upstream-persist --reuse-local-mirror
python -m health_engine.run_v28_ai --source sheets --output-dir ./build_out --reuse-local-mirror --daily
```