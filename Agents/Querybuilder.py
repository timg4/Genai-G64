from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field
from openai import OpenAI
import json

client = OpenAI()

SYSTEM_PROMPT = """You are QueryComposer for a wind-turbine manuals RAG system.

Goal: Convert the provided incident context into ONE high-quality retrieval query for a vector database of manual chunks.

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
4) Avoid admin/training/metadata content.
   Add negative keywords for: change log, abbreviations, terms and definitions, course, timetable, assessment, training standard.
5) Do NOT hallucinate components or alarms not present.
6) Return JSON strictly matching the schema.
"""

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