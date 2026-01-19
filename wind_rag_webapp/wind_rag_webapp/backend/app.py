from __future__ import annotations

import base64
import json
import math
import os
import re
import sys
import uuid
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
    free_text: str = ""
    image_descriptions: List[str] = Field(default_factory=list)
    image_files: List[Dict[str, str]] = Field(default_factory=list)
    scada_id: Optional[str] = None
    top_k: int = 10


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
SYSTEM_PROMPT_RECOMMEND = """You are a wind-turbine maintenance assistant.

You will receive:
1) incident context (SCADA summary + optional notes)
2) top retrieved manual chunks, each with (source, page range, section, text)

Task:
- Write a practical action recommendation (in German) for a technician.
- Use short numbered steps.
- When you reference something from a chunk, cite it like: [source pX-Y].
- If the chunks are irrelevant or only training/admin, say so and ask for the right manual.
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
        model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        input=[
            {"role": "system", "content": SYSTEM_PROMPT_RECOMMEND},
            {"role": "user", "content": json.dumps(user_msg, ensure_ascii=False)},
        ],
        max_output_tokens=700,
    )

    return resp.output_text.strip()


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
    result = run_faulty_openai(
        prompt,
        attachments,
        model=os.environ.get("FAULTY_MODEL", os.environ.get("OPENAI_MODEL", "gpt-4o-mini")),
        api_key=api_key,
    )
    parsed = result.get("parsed_json") or {}
    if isinstance(parsed, dict) and parsed.get("description"):
        return str(parsed["description"])
    raw = str(result.get("raw_text", "")).strip()
    return raw or None


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


@app.post("/api/run", response_model=RunResponse)
def run(req: RunRequest) -> RunResponse:
    # 1) assemble context
    scada_case: Dict[str, Any] = {}
    if req.scada_id:
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
        "mechanic_notes": req.free_text,
    }

    # 2) QueryComposer (LLM or fallback)
    if QUERY_BUILDER_ERROR or compose_query_pack is None:
        raise HTTPException(status_code=500, detail=f"Querybuilder import failed: {QUERY_BUILDER_ERROR}")
    qp = compose_query_pack(context, model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"))

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
