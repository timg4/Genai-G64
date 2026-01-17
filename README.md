# Wind Turbine Inspection Assistant

![Project visual](Gemini_Generated_Image_bi9r96bi9r96bi9r.png)

This repo contains the MVP pipeline for a wind turbine inspection assistant, including:
- Text-first manuals RAG baseline (chunking, hybrid retrieval, grounded JSON reports).
- SCADA preprocessing and card generation used in the broader pipeline.

See `manuals/README.md` for the text RAG setup and usage.

## SCADA data (Wind Farm A)
We use the CARE To Compare SCADA dataset under `Scada/CARE_To_Compare/`.
Each dataset file is 10‑minute sensor data with an `event_info.csv` descriptor.

## SCADA pipeline
Two steps, intentionally separated:

1) **Data → window** (simulation only)  
   Extracts 6h windows and includes 7 days of history in a single JSON payload.
   This does not produce cards; it only prepares inputs for step 2.

2) **Window → card** (real use)  
   Takes a single JSON payload and produces a SCADA card (stats, tags, summary).

**Window selection (simulation):** within each dataset we pick the 6h slice with the highest anomaly score (mean absolute z‑score vs. training baseline). For “Nix” we pick the most stable 6h slice from normal events.

**Card contents:** source metadata, window times, summary stats, derived metrics (e.g., power residuals, status ratios), tags, and a short LLM summary. Class labels are kept in a separate `scada_windows_by_class.json` for evaluation.

## Usage
Install:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Generate window payloads (simulation):
```bash
python3 Scada/scada_card_builder.py --per-class 5 --nix-count 5 --baseline-hours 168 --window-hours 6
```
Outputs `Scada/scada_windows/*.json` plus `Scada/scada_windows/scada_windows_by_class.json`.

Generate cards from windows:
```bash
export OPENAI_API_KEY="your_key_here"
python3 Scada/scada_card_from_window.py --input-directory Scada/scada_windows --output-path Scada/scada_cards_out
```
Or a single file:
```bash
python3 Scada/scada_card_from_window.py --input-json Scada/scada_windows/WF-A-10-53591.json --output-path Scada/scada_card.json
```
