import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set


CATEGORY_MAP = {
    0: "VG;MT",
    1: "LE;ER",
    2: "LR;DA",
    3: "LE;CR",
    4: "SF;PO",
    5: "No Damage",
    6: "Invalid_Image"
}


def _portable_rel_path(raw: str) -> Path:
    # Allow Windows-style rel paths (backslashes) to work on macOS/Linux.
    # JSON example files often store rel_path with "\" separators.
    return Path((raw or "").replace("\\", "/"))


def _schema_block() -> str:
    return (
        "Output JSON schema:\n"
        "{\n"
        '  \"description\": str\n'
        "}\n"
    )


def build_prompt(
    examples: List[dict],
    query_path: Path,
    max_examples: int = 8,
    query_tiles: Optional[List[Path]] = None,
    base_dir: Optional[Path] = None,
    few_shot_spec: Optional[dict] = None,
) -> Tuple[str, List[dict]]:
    attachments: List[dict] = []
    lines: List[str] = []
    if base_dir is None:
        base_dir = Path.cwd()

    lines.append(
        "You are a cautious wind turbine blade inspection assistant. "
        "Your task is to describe what is seen at the input image and if there a damages present."
        "Examples below are for calibration only. Do NOT include any example image findings "
        "or categories in the final output. Only analyze the QUERY image(s) at the end. "
        "If the image quality is poor, distant, or ambiguous, say you are unsure. "
    )
    lines.append("")
    lines.append("Categories:")
    for k, v in CATEGORY_MAP.items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append(_schema_block())
    lines.append("Rules:")
    lines.append("- Describe only the QUERY image.")
    lines.append("- The description must state whether defects are present, absent, or uncertain.")
    lines.append("- If no wind turbine or relevant component is visible, say so and state that damage cannot be assessed.")
    lines.append("- If a turbine component is visible but no defects are seen, state that no defects are present.")
    lines.append("- If unsure, say so explicitly instead of guessing.")
    if isinstance(few_shot_spec, dict):
        constraints = few_shot_spec.get("output_constraints") or []
        if constraints:
            lines.append("- Additional constraints:")
            for c in constraints:
                lines.append(f"  - {c}")
    lines.append("")

    if isinstance(few_shot_spec, dict):
        triggers = few_shot_spec.get("class_triggers") or {}
        if triggers:
            lines.append("Class triggers (reference only):")
            for name, desc in triggers.items():
                lines.append(f"- {name}: {desc}")
            lines.append("")

    lines.append("Few-shot examples (do not analyze for final answer):")
    lines.append("")

    selected = _select_balanced_examples(examples, max_examples=max_examples)
    for idx, ex in enumerate(selected, start=1):
        label = f"example_{idx}"
        path_full = _resolve_example_path(ex, base_dir)
        attachments.append({"type": "image", "label": label, "path": str(path_full)})
        lines.append(f"Example {idx}:")
        lines.append(f"- Image: <{label}>")
        if ex.get("findings"):
            lines.append("Ground-truth findings (text descriptions):")
            lines.append(_format_findings(ex["findings"]))
        example_text = _example_text(ex)
        if example_text:
            lines.append("Expected output:")
            lines.append(json.dumps({"description": example_text}, ensure_ascii=False))
        lines.append("")

    query_label = "query_full"
    attachments.append({"type": "image", "label": query_label, "path": str(query_path)})
    lines.append("Query (ONLY this image influences the final answer):")
    lines.append(f"- Image: <{query_label}>")
    if query_tiles:
        lines.append("- Tiles:")
        for idx, tile in enumerate(query_tiles, start=1):
            label = f"query_tile_{idx}"
            attachments.append({"type": "image", "label": label, "path": str(tile)})
            lines.append(f"  - <{label}>")
    lines.append("")
    lines.append("Return ONLY the JSON output, with no extra text.")

    return "\n".join(lines), attachments


def _example_categories(example: dict) -> Set[str]:
    categories = set()
    for f in example.get("findings", []) or []:
        name = f.get("category_name")
        if name:
            categories.add(str(name))
    return categories


def _select_balanced_examples(examples: List[dict], max_examples: int) -> List[dict]:
    no_damage = [e for e in examples if e.get("example_type") == "no_damage"]
    damage = [e for e in examples if e.get("example_type") == "damage"]
    invalid = [e for e in examples if e.get("example_type") == "invalid"]

    selected: List[dict] = []
    selected.extend(no_damage[:2])
    if invalid:
        selected.extend(invalid[:1])

    damage_sorted = sorted(damage, key=lambda e: len(_example_categories(e)))
    used_categories: Set[str] = set()

    for ex in damage_sorted:
        if len(selected) >= max_examples:
            break
        cats = _example_categories(ex)
        if not cats:
            continue
        if len(used_categories | cats) > 2:
            continue
        selected.append(ex)
        used_categories |= cats

    for ex in damage_sorted:
        if len(selected) >= max_examples:
            break
        if ex in selected:
            continue
        selected.append(ex)

    return selected[:max_examples]


def _resolve_example_path(example: dict, base_dir: Path) -> Path:
    if example.get("path_full"):
        raw = str(example["path_full"])
        return Path(raw.replace("\\", "/")).expanduser().resolve()
    if example.get("rel_path"):
        rel = _portable_rel_path(str(example["rel_path"]))
        candidate = (base_dir / rel)
        if candidate.exists():
            return candidate.resolve()
        fallback = (base_dir.parent / rel)
        return fallback.resolve()
    raise ValueError("Example is missing path_full or rel_path")


def _format_findings(findings: List[dict]) -> str:
    lines = []
    for f in findings:
        cat = f.get("category_name", "unknown")
        desc = f.get("image_description", "")
        if desc:
            lines.append(f"- {cat}: {desc}")
        else:
            lines.append(f"- {cat}")
    return "\n".join(lines)


def _example_text(example: dict) -> str:
    if example.get("gold_text"):
        return str(example["gold_text"])
    if example.get("image_description"):
        return str(example["image_description"])
    if example.get("example_type") == "invalid":
        return "Kein Windrad bzw. kein relevanter Windradbereich ist sichtbar; eine Schadensbewertung ist nicht möglich."
    if example.get("example_type") == "no_damage":
        return "Windradbereich sichtbar; keine Schäden erkennbar."
    findings = example.get("findings") or []
    if not findings:
        return "No visible defects are present."
    texts = []
    for f in findings:
        desc = f.get("image_description")
        if desc:
            texts.append(desc)
    return " ".join(texts).strip()
