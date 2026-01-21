# Wind Turbine Inspection Assistant

Prototype pipeline + web UI for wind turbine blade inspection support:
- manual retrieval (RAG) over maintenance manuals
- SCADA case selection
- (optional) image description + defect prompting
- one-click run in a FastAPI-backed webapp

## Quickstart (Webapp)

Windows (PowerShell) from the repo root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Required for LLM-backed endpoints (no fallback)
$env:OPENAI_API_KEY = "YOUR_KEY_HERE"

python apps/wind_rag_webapp/backend/app.py
```

Then open: http://127.0.0.1:8000

If `OPENAI_API_KEY` is not set, the server still starts, but LLM endpoints return an explicit error and the web UI shows a warning.

## Repo layout

- `apps/wind_rag_webapp/` — FastAPI backend + static UI
- `packages/manuals/` — manual indexing + retrieval baseline (RAG)
- `packages/Agents/` — query builder (LLM-backed)
- `packages/Scada/` — SCADA processing + generated cards
- `packages/Faulty_Image_Describtion/` — optional vision prompting helpers
- `experiments/` — evaluation + scripts (not required for running the webapp)
- `docs/` — project docs

## Configuration

The webapp backend supports overriding paths via env vars:
- `RAG_INDEX_DIR`, `MANUALS_DIR`, `SCADA_DIR`, `FAULTY_DIR`, `EVAL_DIR`

---

## Legacy: Wind Turbine Blade Few-Shot Pipeline

Local, offline-friendly few-shot prompting workflow for wind turbine blade defect checks with retrieval-based example selection.

## Project layout

```
.
├─ main.py
├─ wind_fewshot/
│  ├─ __init__.py
│  ├─ config.py
│  ├─ embedding.py
│  ├─ indexer.py
│  ├─ prompt.py
│  ├─ retrieval.py
│  ├─ schema.py
│  ├─ tiling.py
│  └─ utils.py
├─ fewshot_bank/
│  ├─ README.md
│  └─ metadata.jsonl
├─ indices/
│  └─ .gitkeep
└─ runs/
   └─ .gitkeep
```

## Installation

Create a virtual environment and install requirements:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Optional (better embeddings): install PyTorch + OpenCLIP and ensure weights are cached locally.

## OpenAI API key (for LLM features)

Some parts of this repo (e.g. `apps/wind_rag_webapp/` and `packages/manuals/rag_cli.py --generate`) call the OpenAI API. Set `OPENAI_API_KEY` in your environment before using them.

PowerShell:

```powershell
$env:OPENAI_API_KEY = "YOUR_KEY_HERE"
```

If `OPENAI_API_KEY` is not set, LLM-backed features return an explicit error (no fallback).

## Populate few-shot bank

Use `fewshot_bank/metadata.jsonl` (JSONL, one object per line). See `fewshot_bank/README.md` for schema.

Recommended:
- Provide `path_full` and optionally `path_crop`.
- Provide `gold_json` as the expected output for that example.
- Include some `no_damage`, `look_alike`, and `uncertain` examples.

## Build index

```bash
python main.py build-index --metadata fewshot_bank/metadata.jsonl --index-dir indices
```

Backends:
- `auto` (default): tries OpenCLIP, falls back to histogram.
- `open_clip`: use OpenCLIP only (requires local weights).
- `histogram`: lightweight fallback.

## Prepare a run

```bash
python main.py prepare-run --query path/to/query.jpg --metadata fewshot_bank/metadata.jsonl --index-dir indices --runs-dir runs
```

Artifacts under `runs/run_YYYYMMDD_HHMMSS/`:
- `prompt.txt`
- `attachments.json`
- `retrieval.json`
- `query_tiles/`

## Model invocation

Use `prompt.txt` and `attachments.json` with your vision model of choice.
The prompt enforces JSON-only output and includes few-shot examples.

## Validate output

```bash
python main.py validate-output --run runs/run_YYYYMMDD_HHMMSS --json path/to/model_output.json
```

Produces `validation_report.txt`.

## Notes

- The schema requires `damage_present=false` to have empty findings.
- If the image is ambiguous, the model must output `damage_present="uncertain"`.
- Retrieval enforces diversity and minimum counts for no-damage and uncertain examples (if available).
