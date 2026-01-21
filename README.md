# Wind Turbine Inspection Assistant

A lightweight end-to-end prototype that turns a turbine incident (notes + optional image descriptions + SCADA case) into:
- a composed retrieval query (LLM-assisted)
- Top-K retrieved manual chunks (local FAISS + BM25 hybrid)
- a short diagnosis + recommendation (LLM)

## Quickstart (Webapp)

Prerequisites: Python 3.10+ and an OpenAI API key.

PowerShell (from the repo root):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

$env:OPENAI_API_KEY = "YOUR_KEY_HERE"
python apps/wind_rag_webapp/backend/app.py
```

Open: http://127.0.0.1:8000

Notes:
- If `OPENAI_API_KEY` is not set, the server still starts, but LLM endpoints return a clear error and the UI shows a warning.

## Repository Layout

- `apps/wind_rag_webapp/` – FastAPI backend + static UI (`apps/wind_rag_webapp/backend/static/index.html`)
- `packages/manuals/` – manual indexing and retrieval (FAISS + BM25)
- `packages/Agents/` – query builder (LLM-assisted term extraction + query pack)
- `packages/Scada/` – SCADA processing and generated cards
- `packages/Faulty_Image_Describtion/` – optional image description / prompting helpers
- `experiments/` – evaluation and dataset utilities (not required to run the webapp)
- `docs/` – project docs

## Manual Retrieval (RAG)

The webapp reads the index from `packages/manuals/manuals_index/` by default.

Build / rebuild the index:

```powershell
python packages/manuals/rag_cli.py build --manuals-dir "packages/manuals/manuals_text" --index-dir "packages/manuals/manuals_index" --section-chunks
```

Optional: export PDFs to text first (recommended):

```powershell
python packages/manuals/rag_cli.py export-txt --manuals-dir "packages/manuals/manuals_files" --out-dir "packages/manuals/manuals_text"
```

Quick query test:

```powershell
python packages/manuals/rag_cli.py query --index-dir "packages/manuals/manuals_index" --query "Leading edge erosion inspection" --top-k 5
```

## SCADA Data

The webapp prefers generated cards under:
- `packages/Scada/scada_cards_out/`
- `packages/Scada/scada_windows_meta/`

If those are missing, it falls back to sample windows in `apps/wind_rag_webapp/backend/data/scada_samples.json`.

## Configuration (env vars)

- `OPENAI_API_KEY` – required for LLM endpoints
- `PORT` – default `8000`
- `RAG_INDEX_DIR` – default `packages/manuals/manuals_index`
- `MANUALS_DIR`, `SCADA_DIR`, `FAULTY_DIR`, `EVAL_DIR` – override package locations
- `RAG_MODEL` – sentence-transformers model (optional)
- `RAG_ALPHA` – blend weight between embeddings and BM25 (default `0.7`)

## Troubleshooting

- `OPENAI_API_KEY is not set`: set the env var before starting the app.
- `manuals directory not found` / `Missing index file`: (re)build the manual index and/or set `RAG_INDEX_DIR`.
- If imports fail after moving folders: run from the repo root and keep the `packages/` folder intact.
