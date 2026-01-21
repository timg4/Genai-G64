import base64
import json
import os
import re
from pathlib import Path
from typing import Dict, List


def _image_to_base64(path: Path) -> str:
    data = path.read_bytes()
    return base64.b64encode(data).decode("ascii")


def _extract_json(text: str) -> Dict[str, str]:
    # Try to extract JSON inside ```json ... ```
    match = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.S)
    if match:
        text = match.group(1)
    text = text.strip()

    # If model returned extra text, attempt to isolate the first JSON object
    if not text.startswith("{"):
        m2 = re.search(r"(\{.*\})", text, flags=re.S)
        if m2:
            text = m2.group(1).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


def run_openai_vision(
    prompt: str,
    attachments: List[Dict[str, str]],
    model: str = "gpt-5.2",
    api_key: str = "",
) -> Dict[str, str]:
    try:
        from openai import OpenAI
    except Exception as exc:
        raise RuntimeError("openai package not installed") from exc

    if not api_key:
        api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    client = OpenAI(api_key=api_key)

    content = [{"type": "input_text", "text": prompt}]
    for att in attachments:
        if att.get("type") != "image":
            continue
        label = att.get("label", "unlabeled")
        path = Path(att["path"])
        encoded = _image_to_base64(path)
        # Bind the label to the next image in the multi-modal stream.
        content.append({"type": "input_text", "text": f"<{label}> (image follows)"})
        content.append(
            {
                "type": "input_image",
                "image_url": f"data:image/jpeg;base64,{encoded}",
                "detail": "high",
            }
        )

    content.append(
        {
            "type": "input_text",
            "text": (
                "IMPORTANT: Only the image labeled <query_full> influences the final answer. "
                "Ignore all other images for the final decision. Return ONLY the JSON output."
            ),
        }
    )

    request_args = {
        "model": model,
        "input": [{"role": "user", "content": content}],
    }
    if model.startswith("gpt-5"):
        request_args["reasoning"] = {"effort": "high"}

    response = client.responses.create(**request_args)

    text = response.output_text
    parsed = _extract_json(text)
    return {"raw_text": text, "parsed_json": parsed}
