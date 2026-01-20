from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field
from openai import OpenAI
import json
import os

SYSTEM_PROMPT = """You are QueryComposer for a wind-turbine manuals RAG system.

Goal: Convert the provided incident context into ONE high-quality retrieval query for a vector database of manual chunks. Include all relevant details; do not leave out important signals from the inputs. Ensure the main signals also appear explicitly in the query string (not only in must/should/exclude), since retrieval uses the query text.

Inputs may include:
- scada_case (JSON): stats, derived fields, tags, class_label, summary, event description
- fault_images_description (free text, optional)
- mechanic_notes (free text, optional)

Output requirements:
1) Produce a single retrieval query string that is compact but information-dense.
2) Extract the most important signals from the inputs and reflect them in the query:
   - component(s) likely involved (only if supported by input text; otherwise keep generic)
   - symptoms (power below expected, elevated temperature, yaw misalignment, vibration, etc.)
   - operating conditions (wind speed, high load, rated operation)
   - any codes/labels exactly as given
3) Expand with a few domain synonyms to improve recall.
5) Do NOT hallucinate components or alarms not present.
6) Return JSON strictly matching the schema.
"""

# -------------------------
# FALLBACK QUERY PACK
# -------------------------

def _fallback_query_pack(payload: Dict[str, Any]) -> Dict[str, Any]:
    scada = payload.get("scada_case") or {}
    tags = scada.get("tags") or []
    stats = scada.get("stats") or {}
    derived = scada.get("derived") or {}

    parts = []
    if scada.get("case_id"):
        parts.append(str(scada.get("case_id")))
    if scada.get("class_label"):
        parts.append(f"class_label {scada.get('class_label')}")
    if tags:
        parts.append("tags " + ",".join(tags))
    if "summary" in scada:
        parts.append(str(scada.get("summary")))
    if payload.get("mechanic_notes"):
        parts.append(str(payload.get("mechanic_notes")))
    if payload.get("fault_images_description"):
        parts.append(str(payload.get("fault_images_description")))

    parts.append("power deficit underperformance power residual")
    parts.append("yaw misalignment yaw error")
    parts.append("temperature elevated temps overheating")

    q = " | ".join([p for p in parts if p])

    return {
        "query": q,
        "must_terms": [str(scada.get("class_label"))] if scada.get("class_label") else [],
        "should_terms": ["power residual", "underperformance", "yaw misalignment", "temperature"],
        "exclude_terms": [
            "change log",
            "abbreviations",
            "terms and definitions",
            "course",
            "timetable",
            "assessment",
            "training standard",
        ],
        "extracted_facts": {
            "case_id": scada.get("case_id"),
            "class_label": scada.get("class_label"),
            "tags": tags,
            "key_metrics": {
                "wind_speed_mean": stats.get("wind_speed_mean"),
                "wind_speed_max": stats.get("wind_speed_max"),
                "power_mean": stats.get("power_mean"),
                "yaw_misalignment_mean": stats.get("yaw_misalignment_mean"),
                "temp_mean": stats.get("temp_mean"),
                "power_residual_mean": derived.get("power_residual_mean"),
            },
        },
    }

# -------------------------
# STRICT FACT EXTRACTION
# -------------------------


class MetricItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(..., description="Metric name, e.g. wind_speed_mean")
    value: float = Field(..., description="Metric value")

class ExtractedFacts(BaseModel):
    model_config = ConfigDict(extra="forbid")
    component: Optional[str] = Field(..., description="Likely component or null if unknown")
    symptoms: List[str] = Field(..., description="List of symptoms; can be empty")
    operating_conditions: List[str] = Field(..., description="List of operating conditions; can be empty")
    key_metrics: List[MetricItem] = Field(..., description="List of key metrics; can be empty")
    visual_observations: List[str] = Field(..., description="Observations from images; can be empty")
    labels: List[str] = Field(..., description="Labels/tags/codes; can be empty")

class QueryPack(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(..., description="Single retrieval query to send to the vector DB")
    must_terms: List[str] = Field(..., description="Hard constraints; can be empty")
    should_terms: List[str] = Field(..., description="Boost terms; can be empty")
    exclude_terms: List[str] = Field(..., description="Negative keywords; can be empty")
    extracted_facts: ExtractedFacts = Field(..., description="Signals used to build the query")

def build_query_pack(
    scada_case: Dict[str, Any],
    fault_images_description: Optional[str] = None,
    mechanic_notes: Optional[str] = None,
    model: str = "gpt-4o-mini",
) -> QueryPack:

    client = OpenAI()
    user_payload = {
        "scada_case": scada_case,
        "fault_images_description": fault_images_description,
        "mechanic_notes": mechanic_notes,
        "instruction": "Create the best retrieval query pack for manual-chunk search."
    }

    resp = client.responses.parse(
        model=model,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(user_payload, indent=2)},
        ],
        text_format=QueryPack,
    )

    return resp.output_parsed


def compose_query_pack(payload: Dict[str, Any], model: Optional[str] = None) -> Dict[str, Any]:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return _fallback_query_pack(payload)

    try:
        qp = build_query_pack(
            scada_case=payload.get("scada_case") or {},
            fault_images_description=payload.get("fault_images_description"),
            mechanic_notes=payload.get("mechanic_notes"),
            model=model or "gpt-4o-mini",
        )
        return qp.model_dump()
    except Exception:
        qp = _fallback_query_pack(payload)
        qp["extracted_facts"]["note"] = "Querybuilder failed; used fallback."
        return qp

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
        "summary": (
            "produced power was consistently below expected in strong wind; "
            "elevated temperature; yaw offset ~8°; status normal."
        )
    }

    qp = build_query_pack(
        scada_case=example_scada,
        fault_images_description="crack near leading edge; erosion visible",
        mechanic_notes="customer reports underperformance in high wind, no alarms"
    )

    print(qp.model_dump_json(indent=2))
