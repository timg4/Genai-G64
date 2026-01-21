from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field
from openai import OpenAI
import json
import os
import re

# -------------------------
# PROMPT: EXTRACT TERMS (NOT FINAL QUERY)
# -------------------------

SYSTEM_PROMPT = """You are QueryComposer for a wind-turbine manuals RAG system.

## Role and Objective
Extract balanced, reliable retrieval terms from the incident context for downstream retrieval. Output only the structured JSON—never full sentences.

## Instructions
- Extract 2-6 keyword-style terms (2-5 words each) per source
- Use exact labels/codes when present (case_id, class_label, tags)
- Do not invent components not in the input
- Add a few synonyms if helpful (not required for every term)
- Add exclude_terms for admin/training noise if detected

## Instructions
- Separate terms by input source: SCADA, Visual, Notes.
- For each source:
  - Extract up to 5 relevant, short keyword-style terms (2–5 words each). Fewer is acceptable if necessary.
  - Do not invent components: use only input-provided elements.
  - Add a few synonyms if helpful (not required for every term)
  - Add exclude_terms for admin/training noise if detected (omit if none).
- For sources with no input, always return an empty array.

## Output Format
{
  "scada_terms": ["term1", "term2"],
  "image_terms": ["term1"],
  "notes_terms": ["term1", "term2"],
  "synonyms": ["alt_phrase1", "alt_phrase2"],
  "exclude_terms": ["training", "changelog"],
  "key_metrics": []
}

Rules:
- Each term is a concise keyword or phrase (not a sentence)
- Use empty arrays for sources with no input
- Synonyms are optional alternate phrasings for any terms
- key_metrics can be empty

## Conciseness
Keep terms brief and focused. No sentences or extra detail.

## Stop Criteria
Finish once reliable, relevant terms for each source are extracted and the JSON matches the schema exactly.

"""

# -------------------------
# STRICT / CLOSED SCHEMA
# -------------------------

class MetricItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(...)
    value: float = Field(...)

class TermPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # Codes/labels to always include if present
    codes: List[str] = Field(...)
    # Signals extracted from each input source
    scada_terms: List[str] = Field(...)
    image_terms: List[str] = Field(...)
    notes_terms: List[str] = Field(...)
    # Helpful synonyms / alternate phrasing
    synonyms: List[str] = Field(...)
    # Negative keywords to avoid admin/training sections
    exclude_terms: List[str] = Field(...)
    # Key metrics (only if present)
    key_metrics: List[MetricItem] = Field(...)

class ExtractedFacts(BaseModel):
    model_config = ConfigDict(extra="forbid")
    codes: List[str] = Field(...)
    scada_terms: List[str] = Field(...)
    image_terms: List[str] = Field(...)
    notes_terms: List[str] = Field(...)
    key_metrics: List[MetricItem] = Field(...)

class QueryPack(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(...)
    must_terms: List[str] = Field(...)
    should_terms: List[str] = Field(...)
    exclude_terms: List[str] = Field(...)
    extracted_facts: ExtractedFacts = Field(...)

# -------------------------
# FALLBACK TERM EXTRACTION (NO LLM)
# -------------------------

_STOP = set(["the","a","an","and","or","of","to","in","on","with","for","is","are","was","were"])

def _simple_terms_from_text(text: str, max_terms: int = 12) -> List[str]:
    # very lightweight keyword extraction: keep phrases like "leading edge", "yaw misalignment"
    text = (text or "").strip()
    if not text:
        return []
    # keep words + degree symbol, numbers, hyphen; split by separators
    tokens = re.findall(r"[A-Za-z]+(?:-[A-Za-z]+)?|\d+(?:\.\d+)?|°", text.lower())
    tokens = [t for t in tokens if t not in _STOP and len(t) > 2]
    # join common pairs (naive)
    phrases = []
    for i in range(len(tokens)-1):
        a,b = tokens[i], tokens[i+1]
        if a.isalpha() and b.isalpha():
            phrases.append(f"{a} {b}")
    # prefer domain-ish phrases
    priority = []
    for p in phrases:
        if any(k in p for k in ["leading edge","trailing edge","yaw misalignment","pitch angle","power deficit","power residual","blade crack"]):
            priority.append(p)
    out = list(dict.fromkeys(priority + phrases + tokens))  # unique keep order
    return out[:max_terms]

def _fallback_query_pack(payload: Dict[str, Any]) -> Dict[str, Any]:
    scada = payload.get("scada_case") or {}
    tags = scada.get("tags") or []
    stats = scada.get("stats") or {}
    derived = scada.get("derived") or {}

    codes = []
    if scada.get("case_id"): codes.append(str(scada["case_id"]))
    if scada.get("class_label"): codes.append(str(scada["class_label"]))
    if scada.get("source", {}).get("event_description") not in (None, "", "nan"):
        codes.append(str(scada["source"]["event_description"]))
    codes.extend([str(t) for t in tags])

    scada_terms = []
    if scada.get("summary"):
        scada_terms += _simple_terms_from_text(scada["summary"], 12)
    # add a few hard synonyms that usually help
    synonyms = [
        "underperformance", "power deficit", "power residual", "below expected power",
        "yaw misalignment", "yaw error", "yaw offset",
        "overheating", "elevated temperature"
    ]

    image_desc = payload.get("fault_images_description") or ""
    notes = payload.get("mechanic_notes") or ""
    image_terms = _simple_terms_from_text(image_desc, 10)
    notes_terms = _simple_terms_from_text(notes, 10)

    key_metrics = []
    def add_metric(name, v):
        if isinstance(v, (int, float)):
            key_metrics.append({"name": name, "value": float(v)})

    add_metric("wind_speed_mean", stats.get("wind_speed_mean"))
    add_metric("wind_speed_max", stats.get("wind_speed_max"))
    add_metric("power_mean", stats.get("power_mean"))
    add_metric("yaw_misalignment_mean", stats.get("yaw_misalignment_mean"))
    add_metric("temp_mean", stats.get("temp_mean"))
    add_metric("power_residual_mean", derived.get("power_residual_mean"))

    exclude = [
        "change log", "abbreviations", "terms and definitions",
        "course", "timetable", "assessment", "training standard"
    ]

    qp = _assemble_balanced_query(
        codes=codes,
        scada_terms=scada_terms,
        image_terms=image_terms,
        notes_terms=notes_terms,
        synonyms=synonyms,
        exclude_terms=exclude,
    )

    qp["extracted_facts"] = {
        "codes": codes,
        "scada_terms": scada_terms,
        "image_terms": image_terms,
        "notes_terms": notes_terms,
        "key_metrics": key_metrics,
    }
    return qp

# -------------------------
# BALANCED QUERY ASSEMBLY (THE IMPORTANT PART)
# -------------------------

def _truncate(lst: List[str], n: int) -> List[str]:
    out = []
    for x in lst:
        x = (x or "").strip()
        if not x:
            continue
        if x not in out:
            out.append(x)
        if len(out) >= n:
            break
    return out

def _ensure_min_terms(lst: List[str], min_n: int, backup_text: str) -> List[str]:
    if len(lst) >= min_n:
        return lst
    # add simple terms from backup text to reach min_n
    extra = _simple_terms_from_text(backup_text, max_terms=20)
    merged = list(dict.fromkeys(lst + extra))
    return merged[:max(min_n, len(merged))]

def _assemble_balanced_query(
    codes: List[str],
    scada_terms: List[str],
    image_terms: List[str],
    notes_terms: List[str],
    synonyms: List[str],
    exclude_terms: List[str],
) -> Dict[str, Any]:
    # quotas (tune)
    SCADA_Q = 10
    IMAGE_Q = 10
    NOTES_Q = 10
    SYN_Q = 8

    codes = _truncate(codes, 10)
    scada_terms = _truncate(scada_terms, SCADA_Q)
    image_terms = _truncate(image_terms, IMAGE_Q)
    notes_terms = _truncate(notes_terms, NOTES_Q)
    synonyms = _truncate(synonyms, SYN_Q)

    # Build query as “fielded” text so vector search sees every signal explicitly
    parts = []
    if codes:
        parts.append("CODES: " + " | ".join(codes))
    if scada_terms:
        parts.append("SCADA: " + " ; ".join(scada_terms))
    if image_terms:
        parts.append("VISUAL: " + " ; ".join(image_terms))
    if notes_terms:
        parts.append("NOTES: " + " ; ".join(notes_terms))
    if synonyms:
        parts.append("SYNONYMS: " + " ; ".join(synonyms))

    query = "  ||  ".join(parts)

    # must_terms: keep only stable identifiers (class_label, tags) if present in codes
    must_terms = []
    for c in codes:
        if ";" in c or c.startswith("SCADA-") or c in (codes[:]):  # keep identifiers broadly
            # don't explode must_terms; use should for most
            pass
    # A safer must: class_label if it exists as token like "VG;MT"
    for c in codes:
        if ";" in c and len(c) <= 10:
            must_terms.append(c)

    should_terms = _truncate(scada_terms + image_terms + notes_terms + synonyms, 20)

    return {
        "query": query,
        "must_terms": _truncate(must_terms, 3),
        "should_terms": should_terms,
        "exclude_terms": exclude_terms,
    }

# -------------------------
# LLM TERM EXTRACTION + BALANCED BUILD
# -------------------------

def build_query_pack(
    scada_case: Dict[str, Any],
    fault_images_description: Optional[str] = None,
    mechanic_notes: Optional[str] = None,
    model: str = "gpt-5.2",
) -> QueryPack:
    client = OpenAI()

    user_payload = {
        "scada": {
            "summary": (scada_case or {}).get("summary", ""),
            "tags": (scada_case or {}).get("tags", []),
        },
        "images": fault_images_description or "",
        "notes": mechanic_notes or "",
    }

    resp = client.responses.parse(
        model=model,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(user_payload, indent=2)},
        ],
        text_format=TermPlan,
    )
    plan: TermPlan = resp.output_parsed

    # guarantee “equal importance”: if input exists, enforce minimum coverage
    img_text = fault_images_description or ""
    notes_text = mechanic_notes or ""
    scada_text = (scada_case or {}).get("summary") or ""

    if img_text.strip():
        plan.image_terms = _ensure_min_terms(plan.image_terms, 4, img_text)
    if notes_text.strip():
        plan.notes_terms = _ensure_min_terms(plan.notes_terms, 4, notes_text)
    if scada_text.strip():
        plan.scada_terms = _ensure_min_terms(plan.scada_terms, 4, scada_text)

    qp_core = _assemble_balanced_query(
        codes=plan.codes,
        scada_terms=plan.scada_terms,
        image_terms=plan.image_terms,
        notes_terms=plan.notes_terms,
        synonyms=plan.synonyms,
        exclude_terms=plan.exclude_terms,
    )

    extracted = ExtractedFacts(
        codes=plan.codes,
        scada_terms=plan.scada_terms,
        image_terms=plan.image_terms,
        notes_terms=plan.notes_terms,
        key_metrics=plan.key_metrics,
    )

    return QueryPack(
        query=qp_core["query"],
        must_terms=qp_core["must_terms"],
        should_terms=qp_core["should_terms"],
        exclude_terms=qp_core["exclude_terms"],
        extracted_facts=extracted,
    )

def compose_query_pack(payload: Dict[str, Any], model: Optional[str] = None) -> Dict[str, Any]:
    if model is None:
        return _fallback_query_pack(payload)

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Please set OPENAI_API_KEY (see README.md in the repository root)."
        )

    try:
        qp = build_query_pack(
            scada_case=payload.get("scada_case") or {},
            fault_images_description=payload.get("fault_images_description"),
            mechanic_notes=payload.get("mechanic_notes"),
            model=model or "gpt-5.2",
        )
        return qp.model_dump()
    except Exception as e:
        qp = _fallback_query_pack(payload)
        qp["extracted_facts"]["note"] = f"LLM term extraction failed; used fallback. ({type(e).__name__})"
        return qp

# -------------------------
# EXAMPLE
# -------------------------

if __name__ == "__main__":
    example_scada = {
        "case_id": "SCADA-A-25-52331-VG_MT",
        "class_label": "VG;MT",
        "tags": ["high_wind", "elevated_temps"],
        "stats": {
            "wind_speed_mean": 18.0333,
            "wind_speed_max": 20.3,
            "power_mean": 0.9755,
            "yaw_misalignment_mean": 8.0583,
            "temp_mean": 67.4861,
        },
        "derived": {"power_residual_mean": -0.1293},
        "summary": "produced power below expected in strong wind; elevated temperature; yaw offset ~8°; status normal."
    }

    qp = build_query_pack(
        scada_case=example_scada,
        fault_images_description="crack near leading edge; erosion visible",
        mechanic_notes="customer reports blade issue and underperformance in high wind, no alarms"
    )

    print(qp.model_dump_json(indent=2))
