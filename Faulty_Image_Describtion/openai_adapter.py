import base64
import json
import os
import re
from pathlib import Path
from typing import Dict, List


def _image_to_base64(path: Path) -> str:
    data = path.read_bytes()
    return base64.b64encode(data).decode("ascii")


def run_openai_vision(
    prompt: str,
    attachments: List[Dict[str, str]],
    model: str = "gpt-4o-mini",
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
        path = Path(att["path"])
        encoded = _image_to_base64(path)
        content.append(
            {
                "type": "input_image",
                "image_url": f"data:image/jpeg;base64,{encoded}",
            }
        )

    response = client.responses.create(
        model=model,
        input=[{"role": "user", "content": content}],
    )

    text = response.output_text
    parsed = _extract_json(text)
    return {"raw_text": text, "parsed_json": parsed}


def _extract_json(text: str) -> Dict[str, str]:
    match = re.search(r"```json\\s*(\\{.*?\\})\\s*```", text, flags=re.S)
    if not match:
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            return {}
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return {}
