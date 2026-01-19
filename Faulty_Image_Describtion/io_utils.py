import json
from pathlib import Path
from typing import Any, Dict, List


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def read_examples(path: Path) -> List[dict]:
    data = read_json(path)
    if isinstance(data, dict) and "examples" in data:
        return list(data["examples"])
    if isinstance(data, list):
        return list(data)
    raise ValueError("examples json must be a list or {\"examples\": [...]} ")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
