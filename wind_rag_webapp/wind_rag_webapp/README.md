# Wind Manual Assistant (Prototype Webapp)

A small ChatGPT-like web UI for your pipeline:
- free-text incident description
- (simulated) image upload via drag & drop -> you only send image *descriptions*
- SCADA case selection by time window (predefined samples)
- one-click run -> query composer -> retrieval -> action recommendation
- result table with Top-K retrieved manual chunks (with source + page range)

## 1) Backend setup

### Windows (PowerShell)

```powershell
cd wind_rag_webapp\backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# IMPORTANT: set your OpenAI API key as an env var (do NOT hardcode, do NOT commit)
$env:OPENAI_API_KEY = "YOUR_KEY_HERE"

python app.py
```

Then open: http://127.0.0.1:8000

### Notes
- If `OPENAI_API_KEY` is not set, the app still runs (fallback query + heuristic recommendation).
- Replace `backend/data/scada_samples.json` with your own predefined time windows.
- Replace `backend/data/chunks.json` with your own chunk export.

## 2) Integrate your real vector DB

In `backend/app.py`, search for `retrieve_bm25`.
- That function is a simple local BM25.
- Swap it with your vector search (e.g., embeddings + cosine similarity, FAISS, Qdrant, Pinecone, etc.).
- Keep returning: source, page, page_end, section, chunk_id, score, snippet.

## 3) Security

If you ever pasted your API key into chat or git history, **rotate/revoke it immediately** in your OpenAI dashboard.
