import time
from pathlib import Path
from typing import List, Tuple, Optional

from Faulty_Image_Describtion.io_utils import ensure_dir, read_json, write_json
from Faulty_Image_Describtion.prompt import build_prompt
from Faulty_Image_Describtion.schema import validate_output
from Faulty_Image_Describtion.tiling import tile_image


def prepare_run(
    query_path: Path,
    examples_json: Path,
    run_dir: Path,
    max_examples: int = 8,
    tiles_enabled: bool = False,
    tile_size: int = 1024,
    tile_rows: int = 3,
    tile_cols: int = 3,
    tile_overlap: float = 0.2,
) -> Tuple[Path, Path]:
    payload = read_json(examples_json)
    if isinstance(payload, dict) and "examples" in payload:
        examples = payload.get("examples", [])
        few_shot_spec = payload.get("few_shot_spec")
    elif isinstance(payload, list):
        examples = payload
        few_shot_spec = None
    else:
        raise ValueError("examples json must be a list or {\"examples\": [...]} ")
    query_tiles: Optional[List[Path]] = None
    if tiles_enabled:
        tiles_dir = run_dir / "query_tiles"
        tiles = tile_image(
            image_path=query_path,
            output_dir=tiles_dir,
            tile_size=tile_size,
            rows=tile_rows,
            cols=tile_cols,
            overlap=tile_overlap,
        )
        query_tiles = [t.path for t in tiles]

    prompt_text, attachments = build_prompt(
        examples,
        query_path,
        max_examples=max_examples,
        query_tiles=query_tiles,
        base_dir=Path.cwd(),
        few_shot_spec=few_shot_spec,
    )

    ensure_dir(run_dir)
    prompt_path = run_dir / "prompt.txt"
    attachments_path = run_dir / "attachments.json"

    prompt_path.write_text(prompt_text, encoding="utf-8-sig")
    write_json(attachments_path, {"attachments": attachments})
    return prompt_path, attachments_path


def validate_output_file(run_dir: Path, json_path: Path) -> Path:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    errors = validate_output(payload)
    report_path = run_dir / "validation_report.txt"
    if errors:
        report = "Validation FAILED\n" + "\n".join(f"- {e}" for e in errors)
    else:
        report = "Validation OK\n"
    report_path.write_text(report, encoding="utf-8")
    return report_path


def default_run_dir(base_dir: Path) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    candidate = base_dir / f"run_{ts}"
    if not candidate.exists():
        return candidate
    for i in range(1, 1000):
        alt = base_dir / f"run_{ts}_{i:02d}"
        if not alt.exists():
            return alt
    return base_dir / f"run_{ts}_{int(time.time() * 1000)}"
