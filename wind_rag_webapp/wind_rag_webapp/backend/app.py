from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import re
import sys
import time
import uuid
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# OpenAI is optional: the app runs without it (fallback mode).
try:
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
SCADA_SAMPLES_PATH = DATA_DIR / "scada_samples.json"
UPLOADS_DIR = DATA_DIR / "uploads"
FAULTY_RUNS_DIR = DATA_DIR / "faulty_runs"

# SCADA cards paths (for new all-cards endpoint)
def _find_repo_root(start: Path) -> Path:
    for parent in [start] + list(start.parents):
        if (parent / "manuals").is_dir():
            return parent
    return start.parents[3]


REPO_ROOT = _find_repo_root(APP_DIR)
MANUALS_DIR = REPO_ROOT / "manuals"
RAG_INDEX_DIR = Path(
    os.environ.get("RAG_INDEX_DIR", str(MANUALS_DIR / "manuals_index"))
)

# SCADA cards directories
SCADA_CARDS_DIR = REPO_ROOT / "Scada" / "scada_cards_out"
SCADA_WINDOWS_META_DIR = REPO_ROOT / "Scada" / "scada_windows_meta"
SCADA_BY_CLASS_PATH = SCADA_WINDOWS_META_DIR / "scada_windows_by_class.json"
EVAL_CASES_PATH = REPO_ROOT / "evalution" / "cases.json"

# Class code to display name mapping
CLASS_DISPLAY_NAMES = {
    "VG;MT": "Vortex Generating Panel – Missing Teeth",
    "LE;ER": "Leading Edge – Erosion",
    "LR;DA": "Lightning Receptor – Damage",
    "LE;CR": "Leading Edge – Crack",
    "SF;PO": "Surface – Paint-Off",
    "Nix": "Normalzustand",
}

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")

# Querybuilder integration (optional)
QUERY_BUILDER_ERROR: Optional[str] = None
sys.path.insert(0, str(REPO_ROOT))
try:
    from Agents.Querybuilder import compose_query_pack  # type: ignore
except Exception as exc:  # pragma: no cover
    QUERY_BUILDER_ERROR = str(exc)
    compose_query_pack = None  # type: ignore


class RunRequest(BaseModel):
    mechanic_notes: str = ""
    image_descriptions: List[str] = Field(default_factory=list)
    image_files: List[Dict[str, str]] = Field(default_factory=list)
    scada_id: Optional[str] = None
    scada_case: Optional[Dict[str, Any]] = None  # Direct case data from frontend
    top_k: int = 10


class DiagnosisRequest(BaseModel):
    mechanic_notes: str = ""
    image_descriptions: List[str] = Field(default_factory=list)
    scada_id: Optional[str] = None
    scada_case: Optional[Dict[str, Any]] = None  # Direct case data from frontend


class RiskCode(str, Enum):
    high_risk = "high_risk"
    medium_risk = "medium_risk"
    low_risk = "low_risk"
    no_risk = "no_risk"
    not_classified = "not_classified"


class DiagnosisResponse(BaseModel):
    diagnosis: str
    risk_code: RiskCode


class ImageFile(BaseModel):
    name: str = "upload.jpg"
    data_url: str


class DescribeRequest(BaseModel):
    image_files: List[ImageFile] = Field(default_factory=list)


class DescribeResponse(BaseModel):
    descriptions: List[Dict[str, str]]


class RetrievedChunk(BaseModel):
    chunk_id: str
    source: str
    page: int
    page_end: int
    section: Optional[str] = None
    score: float
    snippet: str


class RunResponse(BaseModel):
    query_pack: Dict[str, Any]
    recommendation_markdown: str
    retrieved: List[RetrievedChunk]



# -----------------------------
# Query composer + recommender
# -----------------------------
SYSTEM_PROMPT_RECOMMEND_OLD = """

You will receive:
1) incident context (SCADA summary + optional notes)
2) top retrieved manual chunks, each with (source, page range, section, text)

Task:
- Write a practical action recommendation for a technician.
- Use short numbered steps.
- Try to reference from chunks, cite the manual name and pages, e.g. [ManualName pX-Y].
- If the chunks are irrelevant or only training/admin, say so and ask for the right manual.
- Prioritize safety: if there are any safety warnings, mention them early. First priority is safety for technician.
- If there is no issue indicated, start by saying everything is ok, then add optional maintenance steps (e.g., "Everything is OK, but for maintenance you can still do: ...").
"""

SYSTEM_PROMPT_RECOMMEND = """
System: # Role and Objective
Give concise, actionable recommendations for technicians, using incident context and manual content. Prioritize safety and relevance.

# Instructions
- Input: 
  1. Incident context (SCADA summary and/or Technician notes and/or image descriptions).
  2. Top manual chunks (source, page range, section, and text).

- Task:
  - Recommend practical actions for the technician.
  - Use short, numbered steps.
  - Reference manual chunks with full manual source name and page numbers (e.g., `[manual-01 pX-Y]`).
  - If the chunks are irrelevant or only training/admin, say so and ask for the right manual.
  - If there are any, list safety warnings first.
  - If no issue is found, state that all is OK, then suggest optional maintenance steps (e.g., "Everything is OK, but maintenance could include: ...").

# Output Format
- Numbered, concise action steps.
- Include manual citations where relevant.
- Address safety at the start of each recommendation.

# Verbosity
- Remain concise and direct.
- Number each step.

# Stop Conditions
- End output after detailing action steps, safety notices, and (if needed) maintenance suggestions.
"""

SYSTEM_PROMPT_DIAGNOSIS = """
You will receive incident input used for RAG (SCADA summary, mechanic notes, image descriptions).
Return strict JSON with keys:
- diagnosis: string (concise English diagnosis, 2-3 sentences, observations only, no recommendations/steps)
- risk_code: one of ["high_risk","medium_risk","low_risk","no_risk","not_classified"]

Risk guidance:
- high_risk: strong indication of severe damage or immediate danger (or strong stop-criteria signals).
- medium_risk: clear issue/anomaly but not obviously immediate danger.
- low_risk: minor anomaly / early indicator / low severity.
- no_risk: inputs indicate normal state / no anomaly.
- not_classified: insufficient or ambiguous information; default to this if unsure.
"""


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def recommend_actions(context: Dict[str, Any], retrieved: List[Dict[str, Any]]) -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key or OpenAI is None:
        # Simple offline summary
        lines = [
            "**(Offline Modus)** Kein OPENAI_API_KEY gesetzt.\n",
            "**Kontext:**",
            f"- SCADA: {context.get('scada_case', {}).get('summary', '')}",
            f"- Notizen: {context.get('mechanic_notes', '')}",
            f"- Bilder: {context.get('fault_images_description', '')}",
            "\n**Gefundene Manual-Stellen:**",
        ]
        for c in retrieved[:5]:
            lines.append(f"- {c.get('source')} p{c.get('page')}-{c.get('page_end')}: {c.get('section')}")
        return "\n".join(lines)

    client = OpenAI()

    chunk_blocks = []
    for i, c in enumerate(retrieved[:10], start=1):
        src = c.get("source")
        p0 = c.get("page")
        p1 = c.get("page_end")
        sec = c.get("section") or ""
        text = (c.get("text") or "")
        text = text[:1800]
        chunk_blocks.append(
            f"CHUNK {i}: [{src} p{p0}-{p1}] {sec}\n{text}\n"
        )

    user_msg = {
        "context": context,
        "retrieved_chunks": chunk_blocks,
    }

    resp = client.responses.create(
        model=os.environ.get("OPENAI_MODEL", "gpt-5.2"),
        input=[
            {"role": "system", "content": SYSTEM_PROMPT_RECOMMEND},
            {"role": "user", "content": json.dumps(user_msg, ensure_ascii=False)},
        ],
        max_output_tokens=700,
    )

    return resp.output_text.strip()


def _extract_scada_summary(scada_case: Optional[Dict[str, Any]]) -> str:
    if not scada_case or not isinstance(scada_case, dict):
        return ""
    candidates: List[Optional[str]] = [
        scada_case.get("summary"),
    ]
    case_block = scada_case.get("case")
    if isinstance(case_block, dict):
        candidates.append(case_block.get("summary"))
    candidates.extend(
        [
            scada_case.get("event_description"),
            scada_case.get("event_label_display"),
            scada_case.get("event_label"),
        ]
    )
    for item in candidates:
        if isinstance(item, str) and item.strip():
            return item.strip()
    return ""


def _ensure_sentence(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    if text[-1] in ".!?":
        return text
    return f"{text}."


def _diagnosis_offline(context: Dict[str, Any]) -> str:
    sentences: List[str] = []
    scada_summary = str(context.get("scada_summary") or "").strip()
    scada_id = str(context.get("scada_id") or "").strip()
    if scada_summary:
        sentences.append(_ensure_sentence(f"SCADA summary: {scada_summary}"))
    elif scada_id:
        sentences.append(_ensure_sentence(f"SCADA source: {scada_id}"))

    notes = str(context.get("mechanic_notes") or "").strip()
    if notes:
        sentences.append(_ensure_sentence(f"Mechanic notes report: {notes}"))

    images = context.get("image_descriptions") or []
    if isinstance(images, list):
        cleaned = [str(x).strip() for x in images if isinstance(x, str) and str(x).strip()]
        if cleaned:
            joined = "; ".join(cleaned)
            sentences.append(_ensure_sentence(f"Image observations include {joined}"))

    if not sentences:
        sentences = ["No diagnostic input was provided."]
    if len(sentences) < 2:
        sentences.append("Available information is limited, so this summary is tentative.")
    return " ".join(sentences[:3]).strip()


class DiagnosisLLMOutput(BaseModel):
    diagnosis: str
    risk_code: RiskCode = RiskCode.not_classified


def generate_diagnosis(context: Dict[str, Any]) -> Tuple[str, RiskCode]:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key or OpenAI is None:
        return _diagnosis_offline(context), RiskCode.not_classified

    client = OpenAI()
    try:
        resp = client.responses.parse(
            model=os.environ.get("OPENAI_MODEL", "gpt-5.2"),
            input=[
                {"role": "system", "content": SYSTEM_PROMPT_DIAGNOSIS},
                {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
            ],
            text_format=DiagnosisLLMOutput,
            max_output_tokens=240,
        )
        parsed: DiagnosisLLMOutput = resp.output_parsed
        diagnosis_text = (parsed.diagnosis or "").strip() or "(no diagnosis)"
        risk_code = parsed.risk_code or RiskCode.not_classified
        return diagnosis_text, risk_code
    except Exception:
        # Fallback: keep diagnosis, but do not guess risk
        resp = client.responses.create(
            model=os.environ.get("OPENAI_MODEL", "gpt-5.2"),
            input=[
                {"role": "system", "content": "Write a concise English diagnosis in 2-3 sentences. Return plain text only."},
                {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
            ],
            max_output_tokens=180,
        )
        return resp.output_text.strip(), RiskCode.not_classified


# -----------------------------
# RAG integration
# -----------------------------

RAG_INIT_ERROR: Optional[str] = None
RETRIEVER = None
FIXED_KIND_BOOST = {
    "procedure": 1.3,
    "checklist": 1.3,
    "troubleshooting": 1.3,
    "safety": 1.2,
    "inspection": 1.1,
    "training_admin": 0.7,
    "definition": 0.5,
    "changelog": 0.5,
}
if MANUALS_DIR.exists():
    sys.path.insert(0, str(REPO_ROOT))
    try:
        from manuals.rag_core import Retriever  # type: ignore
    except Exception as exc:
        RAG_INIT_ERROR = f"Failed to import manuals.rag_core: {exc}"
    else:
        try:
            RETRIEVER = Retriever(
                str(RAG_INDEX_DIR),
                model_name=os.environ.get("RAG_MODEL"),
                alpha=float(os.environ.get("RAG_ALPHA", "0.7")),
                synthetic_weight=1.0,
                kind_boost=FIXED_KIND_BOOST,
            )
        except Exception as exc:
            RAG_INIT_ERROR = f"Failed to initialize Retriever: {exc}"
else:
    RAG_INIT_ERROR = "manuals/ directory not found; cannot load RAG index."


# -----------------------------
# Faulty image description integration (optional)
# -----------------------------

FAULTY_IMPORT_ERROR: Optional[str] = None
FAULTY_EXAMPLES_PATH = REPO_ROOT / "Faulty_Image_Describtion" / "examples.json"
FAULTY_MODEL_FIXED = "gpt-5.2"
try:
    from Faulty_Image_Describtion.io_utils import read_examples as read_faulty_examples  # type: ignore
    from Faulty_Image_Describtion.openai_adapter import run_openai_vision as run_faulty_openai  # type: ignore
    from Faulty_Image_Describtion.prompt import build_prompt as build_faulty_prompt  # type: ignore
except Exception as exc:  # pragma: no cover
    FAULTY_IMPORT_ERROR = str(exc)


def _decode_data_url(data_url: str) -> bytes:
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    return base64.b64decode(data_url)


def _save_upload(data_url: str, filename: str) -> Path:
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = Path(filename).suffix or ".jpg"
    out_path = UPLOADS_DIR / f"{uuid.uuid4().hex}{suffix}"
    out_path.write_bytes(_decode_data_url(data_url))
    return out_path


def describe_faulty_image(image_path: Path) -> Optional[str]:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key or OpenAI is None:
        return None
    if FAULTY_IMPORT_ERROR or not FAULTY_EXAMPLES_PATH.exists():
        return None

    examples = read_faulty_examples(FAULTY_EXAMPLES_PATH)
    max_examples = int(os.environ.get("FAULTY_FEWSHOT_K", "4"))
    prompt, attachments = build_faulty_prompt(
        examples,
        image_path,
        max_examples=max_examples,
        base_dir=REPO_ROOT,
    )
    run_dir = _create_faulty_run_dir()
    if run_dir:
        (run_dir / "prompt.txt").write_text(prompt, encoding="utf-8-sig")
        (run_dir / "attachments.json").write_text(
            json.dumps({"attachments": attachments}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        _write_upload_debug(run_dir, image_path)
    result = run_faulty_openai(
        prompt,
        attachments,
        model=FAULTY_MODEL_FIXED,
        api_key=api_key,
    )
    if run_dir:
        raw_path = run_dir / "model_output_raw.json"
        raw_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        parsed = result.get("parsed_json") or {}
        if parsed:
            (run_dir / "model_output.json").write_text(
                json.dumps(parsed, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
    parsed = result.get("parsed_json") or {}
    if isinstance(parsed, dict) and parsed.get("description"):
        return str(parsed["description"])
    raw = str(result.get("raw_text", "")).strip()
    return raw or None


def _create_faulty_run_dir() -> Optional[Path]:
    try:
        FAULTY_RUNS_DIR.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        run_dir = FAULTY_RUNS_DIR / f"run_{ts}_{uuid.uuid4().hex[:6]}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir
    except Exception:
        return None


def _write_upload_debug(run_dir: Path, image_path: Path) -> None:
    try:
        data = image_path.read_bytes()
        info = {
            "saved_path": str(image_path),
            "size_bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
        (run_dir / f"query_image{image_path.suffix}").write_bytes(data)
        (run_dir / "upload_info.json").write_text(
            json.dumps(info, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        return


# -----------------------------
# App setup
# -----------------------------

app = FastAPI(title="Wind Manual RAG Webapp")

app.mount("/static", StaticFiles(directory=str(APP_DIR / "static")), name="static")


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    html = (APP_DIR / "static" / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(content=html)


@app.get("/api/scada/options")
def scada_options() -> List[Dict[str, str]]:
    samples = _load_json(SCADA_SAMPLES_PATH)
    return [{"id": s["id"], "label": s.get("label", s["id"])} for s in samples]


@app.get("/api/scada/{scada_id}")
def get_scada(scada_id: str) -> Dict[str, Any]:
    samples = _load_json(SCADA_SAMPLES_PATH)
    for s in samples:
        if s["id"] == scada_id:
            return s["case"]
    raise HTTPException(status_code=404, detail="Unknown scada_id")


@app.get("/api/scada-cards/all")
def get_all_scada_cards() -> Dict[str, Any]:
    """Load all SCADA cards from scada_cards_out with class mappings."""
    # Load class mappings
    class_to_windows: Dict[str, List[str]] = {}
    if SCADA_BY_CLASS_PATH.exists():
        class_to_windows = _load_json(SCADA_BY_CLASS_PATH)
    
    # Build reverse mapping: window_id -> class_code
    window_to_class: Dict[str, str] = {}
    for class_code, window_ids in class_to_windows.items():
        for wid in window_ids:
            window_to_class[wid] = class_code
    
    # Load all cards
    cards = []
    all_tags = set()
    
    if SCADA_CARDS_DIR.exists():
        for card_file in sorted(SCADA_CARDS_DIR.glob("*_card.json")):
            try:
                card_data = _load_json(card_file)
                window_id = card_data.get("window_id", "")
                source_info = card_data.get("source", {})
                event_label = source_info.get("event_label", "normal")
                event_description = source_info.get("event_description", "")
                tags = card_data.get("tags", [])
                
                # Add tags to global set
                all_tags.update(tags)
                
                # Get class code (try exact match first, then base window_id for variants like WF-A-25-52366-1)
                class_code = window_to_class.get(window_id)
                if not class_code:
                    # Try base window_id (strip trailing -1, -2, etc.)
                    base_id = "-".join(window_id.rsplit("-", 1)[:-1]) if window_id.count("-") > 3 else window_id
                    class_code = window_to_class.get(base_id, "")
                
                # Display label for event
                event_label_display = "no recorded anomaly" if event_label == "normal" else event_label
                
                cards.append({
                    "id": window_id,
                    "class_code": class_code,
                    "class_name": CLASS_DISPLAY_NAMES.get(class_code, class_code),
                    "tags": tags,
                    "event_label": event_label,
                    "event_label_display": event_label_display,
                    "event_description": event_description if event_description and event_description != "nan" else "",
                    "case": card_data,  # Full card data for use
                })
            except Exception:
                continue
    
    return {
        "cards": cards,
        "all_tags": sorted(all_tags),
        "all_classes": [
            {"code": code, "name": name}
            for code, name in CLASS_DISPLAY_NAMES.items()
        ],
    }


@app.get("/api/eval/cases")
def get_eval_cases() -> List[Dict[str, Any]]:
    if not EVAL_CASES_PATH.exists():
        raise HTTPException(status_code=404, detail="Eval cases not found")
    return _load_json(EVAL_CASES_PATH)


def retrieve(
    qp: Dict[str, Any],
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    if RAG_INIT_ERROR or RETRIEVER is None:
        raise HTTPException(status_code=500, detail=RAG_INIT_ERROR or "RAG unavailable")

    q = str(qp.get("query") or "")
    must = [m for m in (qp.get("must_terms") or []) if isinstance(m, str) and m.strip()]
    should = [s for s in (qp.get("should_terms") or []) if isinstance(s, str) and s.strip()]
    exclude = [
        e.lower().strip()
        for e in (qp.get("exclude_terms") or [])
        if isinstance(e, str) and e.strip()
    ]

    query_text = " ".join([q] + must + should).strip()
    results = RETRIEVER.search(query_text, top_k=top_k)

    if not (must or exclude):
        return results

    filtered = []
    for item in results:
        text_l = ((item.get("section") or "") + " " + (item.get("text") or "")).lower()
        if exclude and any(term in text_l for term in exclude):
            continue
        if must and any(term.lower() not in text_l for term in must):
            continue
        filtered.append(item)

    return filtered[:top_k]


@app.post("/api/diagnosis", response_model=DiagnosisResponse)
def diagnosis(req: DiagnosisRequest) -> DiagnosisResponse:
    scada_case: Dict[str, Any] = {}
    if req.scada_case:
        scada_case = req.scada_case
    elif req.scada_id:
        card_path = SCADA_CARDS_DIR / f"{req.scada_id}_card.json"
        if card_path.exists():
            scada_case = _load_json(card_path)
        else:
            samples = _load_json(SCADA_SAMPLES_PATH)
            for s in samples:
                if s["id"] == req.scada_id:
                    scada_case = s["case"]
                    break

    image_descriptions = [
        x for x in (req.image_descriptions or []) if isinstance(x, str) and x.strip()
    ]
    scada_summary = _extract_scada_summary(scada_case)

    context = {
        "scada_id": req.scada_id,
        "scada_summary": scada_summary,
        "mechanic_notes": req.mechanic_notes,
        "image_descriptions": image_descriptions,
    }

    diagnosis_text, risk_code = generate_diagnosis(context=context)
    return DiagnosisResponse(diagnosis=diagnosis_text, risk_code=risk_code)


@app.post("/api/run", response_model=RunResponse)
def run(req: RunRequest) -> RunResponse:
    # 1) assemble context
    scada_case: Dict[str, Any] = {}
    
    # Use direct case data if provided, otherwise fall back to scada_id lookup
    if req.scada_case:
        scada_case = req.scada_case
    elif req.scada_id:
        # Try loading from scada_cards_out first
        card_path = SCADA_CARDS_DIR / f"{req.scada_id}_card.json"
        if card_path.exists():
            scada_case = _load_json(card_path)
        else:
            # Fall back to old samples file
            samples = _load_json(SCADA_SAMPLES_PATH)
            for s in samples:
                if s["id"] == req.scada_id:
                    scada_case = s["case"]
                    break

    image_descriptions = [x for x in req.image_descriptions if x.strip()] if req.image_descriptions else []
    if req.image_files and not image_descriptions:
        for item in req.image_files:
            data_url = item.get("data_url") or ""
            if not data_url:
                continue
            filename = item.get("name") or "upload.jpg"
            try:
                image_path = _save_upload(data_url, filename)
            except Exception:
                continue
            desc = describe_faulty_image(image_path)
            if desc:
                image_descriptions.append(desc)

    context = {
        "scada_case": scada_case,
        "fault_images_description": "\n".join(image_descriptions) if image_descriptions else "",
        "mechanic_notes": req.mechanic_notes,
    }

    # 2) QueryComposer (LLM or fallback)
    if QUERY_BUILDER_ERROR or compose_query_pack is None:
        raise HTTPException(status_code=500, detail=f"Querybuilder import failed: {QUERY_BUILDER_ERROR}")
    qp = compose_query_pack(context, model=os.environ.get("OPENAI_MODEL", "gpt-5.2"))

    # 3) Retrieve top-k chunks
    top_k = max(3, min(30, int(req.top_k or 10)))
    retrieved_raw = retrieve(qp, top_k=top_k)

    retrieved = []
    for d in retrieved_raw:
        snippet = (d.get("text") or "").strip().replace("\n", " ")
        snippet = snippet[:240] + ("..." if len(snippet) > 240 else "")
        retrieved.append(
            RetrievedChunk(
                chunk_id=d.get("chunk_id", ""),
                source=d.get("source", ""),
                page=int(d.get("page", 0) or 0),
                page_end=int(d.get("page_end", d.get("page", 0)) or 0),
                section=d.get("section"),
                score=float(d.get("score", 0.0) or 0.0),
                snippet=snippet,
            )
        )

    # 4) Recommendation (LLM or offline)
    recommendation = recommend_actions(context=context, retrieved=retrieved_raw)

    return RunResponse(
        query_pack=qp,
        recommendation_markdown=recommendation,
        retrieved=retrieved,
    )


@app.post("/api/faulty-describe", response_model=DescribeResponse)
def faulty_describe(req: DescribeRequest) -> DescribeResponse:
    if not req.image_files:
        return DescribeResponse(descriptions=[])

    descriptions: List[Dict[str, str]] = []
    for item in req.image_files:
        if not item.data_url:
            descriptions.append({"name": item.name, "description": ""})
            continue
        try:
            image_path = _save_upload(item.data_url, item.name or "upload.jpg")
        except Exception:
            descriptions.append({"name": item.name, "description": ""})
            continue
        desc = describe_faulty_image(image_path) or ""
        descriptions.append({"name": item.name, "description": desc})

    return DescribeResponse(descriptions=descriptions)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=int(os.environ.get("PORT", "8000")), reload=True)
