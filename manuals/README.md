# Text RAG Baseline (Manuals)

![Project visual](../Gemini_Generated_Image_bi9r96bi9r96bi9r.png)

Minimal text-first retrieval pipeline over manuals with optional grounded report generation.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Build an index

```bash
python rag_cli.py build --manuals-dir "manuals_files" --index-dir "./manuals_index"
```

Optional knobs:
- `--model` sentence-transformers model (default: `all-MiniLM-L6-v2`)
- `--chunk-size` target chunk size in whitespace tokens (default: 450)
- `--chunk-overlap` overlap in tokens (default: 70)
- `--section-chunks` add full-section chunks between headings
- `--section-manuals` base filenames that use outline/heading sectioning; use `"*"` for all (default: `"*"`)
- `--skip-first-pages` leading pages to skip for manuals in `--skip-page-manuals` (default: 2)
- `--skip-page-manuals` base filenames that skip leading pages (default: `manual-02.pdf`)
- `--min-page-tokens` drop pages with too little text (default: 20)
- `--min-alpha-ratio` drop low-text/number-heavy sections (default: 0.6)

Artifacts written to `./manuals_index`:
- `index.faiss`
- `metadata.json`
- `config.json`

## Query the index

```bash
python rag_cli.py query --index-dir "./manuals_index" --query "Gearbox vibration spike after shutdown" --top-k 5
```

With grounded JSON report (requires `OPENAI_API_KEY`):

```bash
python rag_cli.py query --index-dir "./manuals_index" --query "Gearbox vibration spike after shutdown" --top-k 5 --generate
```

Hybrid retrieval blends embeddings and BM25 by default:

```
score = alpha * cosine_similarity + (1 - alpha) * bm25
```

Set `--alpha` (0-1) to tune the mix.

You can filter or boost chunk kinds at query time:

```bash
python rag_cli.py query --index-dir "./manuals_index" --query "leading edge erosion inspection and repair" --top-k 5 --allowed-kinds procedure checklist troubleshooting safety
```

```bash
python rag_cli.py query --index-dir "./manuals_index" --query "leading edge erosion inspection and repair" --top-k 5 --kind-boost procedure=1.2 checklist=1.1 safety=1.1
```

## Evaluate retrieval

`gold_example.jsonl` is a dummy template. Replace `relevant_chunk_ids` with actual chunk ids from `metadata.json`.

```bash
python rag_cli.py eval --index-dir "./manuals_index" --gold "gold_example.jsonl" --top-k 5
```

## Notes

- PDFs are processed page-by-page; failed pages are skipped with a warning.
- Text files are treated as a single document with no page index.
- Chunk ids are deterministic per manual path and chunk index.
- Chunking hard-splits at headings like "ELEMENT", "Lesson", "Tab.", "Figure", or numbered section lines.
- Retrieved results include `page`, `page_end`, and `section` when available to help locate content in the manuals.
- Date-only headings and Figure/Table captions are ignored as section splits.
