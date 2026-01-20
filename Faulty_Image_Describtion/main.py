import argparse
import sys
from pathlib import Path

if __package__ is None:
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from Faulty_Image_Describtion.openai_adapter import run_openai_vision
from Faulty_Image_Describtion.pipeline import (
    default_run_dir,
    prepare_run,
    validate_output_file,
)
from Faulty_Image_Describtion.io_utils import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Faulty image description pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    prep = sub.add_parser("prepare-run", help="Prepare prompt + attachments")
    prep.add_argument("--query", required=True)
    prep.add_argument("--examples", required=True, help="Examples JSON file")
    prep.add_argument("--runs-dir", default="evalution/runs")
    prep.add_argument("--max-examples", type=int, default=8)
    prep.add_argument("--tiles", action="store_true", default=False)
    prep.add_argument("--tile-size", type=int, default=1024)
    prep.add_argument("--tile-rows", type=int, default=3)
    prep.add_argument("--tile-cols", type=int, default=3)
    prep.add_argument("--tile-overlap", type=float, default=0.2)

    val = sub.add_parser("validate-output", help="Validate model output JSON")
    val.add_argument("--run", required=True)
    val.add_argument("--json", required=True)

    run = sub.add_parser("run-openai", help="Call OpenAI Vision and save output")
    run.add_argument("--run", required=True)
    run.add_argument("--model", default="gpt-5.2")
    run.add_argument("--api-key", default="")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "prepare-run":
        query_path = Path(args.query).expanduser().resolve()
        examples_json = Path(args.examples).expanduser().resolve()
        runs_dir = Path(args.runs_dir).expanduser().resolve()
        run_dir = default_run_dir(runs_dir)
        prepare_run(
            query_path,
            examples_json,
            run_dir,
            max_examples=args.max_examples,
            tiles_enabled=args.tiles,
            tile_size=args.tile_size,
            tile_rows=args.tile_rows,
            tile_cols=args.tile_cols,
            tile_overlap=args.tile_overlap,
        )
        print(run_dir)
    elif args.command == "validate-output":
        run_dir = Path(args.run).expanduser().resolve()
        json_path = Path(args.json).expanduser().resolve()
        report_path = validate_output_file(run_dir, json_path)
        print(report_path)
    elif args.command == "run-openai":
        run_dir = Path(args.run).expanduser().resolve()
        prompt_path = run_dir / "prompt.txt"
        attachments_path = run_dir / "attachments.json"
        prompt = prompt_path.read_text(encoding="utf-8")
        attachments = __import__("json").loads(
            attachments_path.read_text(encoding="utf-8")
        ).get("attachments", [])
        result = run_openai_vision(prompt, attachments, model=args.model, api_key=args.api_key)
        parsed = result.get("parsed_json") or {}
        if parsed:
            output_path = run_dir / "model_output.json"
            write_json(output_path, parsed)
        else:
            output_path = run_dir / "model_output_raw.json"
            write_json(output_path, result)
        print(output_path)


if __name__ == "__main__":
    main()
