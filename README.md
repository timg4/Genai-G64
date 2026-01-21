# Wind Turbine Inspection Assistant

AI-powered prototype that supports wind turbine technicians during inspection and diagnosis. The system combines RAG retrieval over maintenance manuals, SCADA data analysis with anomaly detection, and optional vision-based defect detection into a one-click pipeline accessible via a web UI.

Given an incident (mechanic notes, SCADA data, wind turbine images), the assistant retrieves relevant manual passages, summarizes operational patterns, and generates actionable recommendations with risk assessment.

## Quickstart (Webapp)

### Windows (PowerShell)

```powershell
cd apps/wind_rag_webapp/backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:OPENAI_API_KEY = "YOUR_KEY_HERE"
python app.py
```

### macOS

```bash
cd apps/wind_rag_webapp/backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY="YOUR_KEY_HERE"
python app.py
```

Then open: **http://127.0.0.1:8000**

> **Note:** Linux should work with the same bash commands as macOS but is untested.

If `OPENAI_API_KEY` is not set, the server still starts but LLM endpoints return an error and the web UI shows a warning.

## Sample Data

### SCADA Data

**Source:** CARE to Compare Dataset

**Citation:** Guck, S. (2024). *Wind Turbine SCADA Data for Early Fault Detection.* [Kaggle](https://www.kaggle.com/datasets/sguck/wind-turbine-scada-data-for-early-fault-detection)

**Description:** 95 datasets containing 89 years of SCADA time series from 36 wind turbines across 3 wind farms:
- Wind Farm A: 5 onshore turbines in Portugal (based on [EDP Open Data](https://www.edp.com/en/innovation/open-data/data))
- Wind Farms B & C: Offshore turbines in Germany (anonymized)

The dataset is balanced with 44 labeled anomaly events and 51 normal behavior datasets. Features include 10-minute averages of wind speed, power output, rotor/generator RPM, temperatures, pitch angles, and operational status.

**Preprocessing:**
- Extract 6-hour sliding windows with 168-hour (7-day) baseline history
- Compute statistical features (mean, std, max) for key sensors
- Z-score normalization against baseline statistics
- Anomaly classification into 5 fault types: VG;MT, LE;ER, LR;DA, LE;CR, SF;PO
- Automated tagging (high wind, yaw misalignment, power deficit, elevated temps, etc.)
- LLM-generated natural language summaries for each window

### Image Data

**Source:** DTU-Wind Turbine Blade Drone Inspection Images

**Citations:**
- Shihavuddin, A. (2018). *DTU - Drone Inspection Images of Wind Turbine.* [Mendeley Data](https://data.mendeley.com/datasets/hd96prn3nc/2)
- Gohar, I. et al. (2023). *Automatic Defect Detection in Wind Turbine Blade Images: Model Benchmarks and Re-Annotations.* [GitHub](https://github.com/imadgohar/DTU-annotations)

**Description:** 324 high-resolution drone images (5280 x 2890 px) of wind turbine blades with annotated defects.

**Defect Classes:**
| Code | Description | Count |
|------|-------------|-------|
| VG;MT | Vortex Generator - Missing Teeth | 264 |
| LE;ER | Leading Edge - Erosion | 338 |
| LR;DA | Lightning Receptor - Damage | 20 |
| LE;CR | Leading Edge - Crack | 82 |
| SF;PO | Surface - Paint-Off | 92 |

**Preprocessing:**
- Few-shot prompting with vision model (GPT-5.2) to generate text descriptions
- Image tiling (1024px patches) for high-resolution inputs
- Descriptions passed to query builder for retrieval context

### Technical Manuals

**Sources:**
- Wind Empowerment Maintenance Manual
- Vestas V90 Manual
- Synthetic manuals based on public technical documents

**Preprocessing:**
- PDF text extraction using pypdf with page tracking
- Section-aware chunking at heading boundaries (450 tokens, 70-token overlap)
- Chunk classification: procedure, checklist, troubleshooting, safety, inspection, definition
- Dense semantic embeddings are generated using the all-MiniLM-L6-v2 sentence embedding model [https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2]
- Hybrid retrieval index: FAISS (sentence-transformers embeddings) + BM25 keyword search
- Configurable blend: `score = alpha * embedding_score + (1 - alpha) * bm25_score`

## Repository Layout

```
├── apps/
│   └── wind_rag_webapp/           # FastAPI backend + static UI
├── docs/                          # Project documentation
├── experiments/                   # Evaluation scripts and test cases
├── packages/
│   ├── Agents/                    # Query builder - extracts search terms from context
│   ├── Faulty_Image_Describtion/  # Vision prompting for blade defect detection
│   ├── manuals/                   # RAG indexing & retrieval
│   └── Scada/                     # SCADA processing, card generation, summaries
├── README.md
└── requirements.txt
```
