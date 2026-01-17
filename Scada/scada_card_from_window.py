#!/usr/bin/env python3
"""
Build a SCADA card from a JSON payload that embeds CSV history and window data.
This file is the "real use" path: given recent SCADA data, output a card JSON.
"""

from __future__ import annotations

import argparse
import json
from io import StringIO
from pathlib import Path
from typing import Dict, List

import pandas as pd

from scada_processing import (
    build_tags,
    compute_baseline,
    compute_window_metrics,
    fit_power_model,
    generate_summary,
)

WIND_COL = "wind_speed_3_avg"
WIND_STD_COL = "wind_speed_3_std"
WIND_MAX_COL = "wind_speed_3_max"
POWER_COL = "power_30_avg"
POWER_STD_COL = "power_30_std"
ROTOR_COL = "sensor_52_avg"
GEN_COL = "sensor_18_avg"
PITCH_COL = "sensor_5_avg"
WIND_DIR_COL = "sensor_1_avg"
NACELLE_DIR_COL = "sensor_42_avg"
TEMP_COLS = ["sensor_11_avg", "sensor_12_avg", "sensor_13_avg", "sensor_14_avg"]

BASE_DIR = Path(__file__).resolve().parent

ANOMALY_FEATURES = [
    "wind_speed_3_avg",
    "wind_speed_3_std",
    "wind_speed_3_max",
    "wind_speed_4_avg",
    "power_30_avg",
    "power_30_std",
    "sensor_52_avg",
    "sensor_52_std",
    "sensor_18_avg",
    "sensor_18_std",
    "sensor_5_avg",
    "sensor_5_std",
    "sensor_11_avg",
    "sensor_12_avg",
    "sensor_13_avg",
    "sensor_14_avg",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build SCADA cards from JSON window payload(s).")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input-json")
    group.add_argument("--input-directory")
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--baseline-hours", type=int, default=None)
    parser.add_argument("--window-hours", type=int, default=None)
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--no-llm", action="store_true")
    return parser.parse_args()


def load_payload(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text())


def parse_csv_payload(csv_text: str, csv_sep: str) -> pd.DataFrame:
    df = pd.read_csv(StringIO(csv_text), sep=csv_sep)
    for col in df.columns:
        if col not in {"time_stamp", "train_test"}:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def slice_windows(
    df: pd.DataFrame, baseline_hours: int, window_hours: int
) -> Dict[str, pd.DataFrame]:
    dt = pd.to_datetime(df["time_stamp"], errors="coerce")
    if dt.isna().all():
        raise ValueError("time_stamp column is not parseable")
    end_time = dt.iloc[-1]
    baseline_start = end_time - pd.Timedelta(hours=baseline_hours)
    window_start = end_time - pd.Timedelta(hours=window_hours)

    baseline_df = df[(dt >= baseline_start) & (dt <= end_time)].copy()
    window_df = df[(dt >= window_start) & (dt <= end_time)].copy()

    if len(window_df) == 0:
        raise ValueError("window_df is empty; check time_stamp and window_hours")
    if len(baseline_df) == 0:
        raise ValueError("baseline_df is empty; check time_stamp and baseline_hours")

    return {
        "baseline_df": baseline_df,
        "window_df": window_df,
        "end_time": end_time,
    }


def build_card(
    payload: Dict[str, object],
    baseline_df: pd.DataFrame,
    window_df: pd.DataFrame,
    end_time: pd.Timestamp,
    baseline_hours: int,
    window_hours: int,
    model: str,
    no_llm: bool,
) -> Dict[str, object]:
    slope, intercept = fit_power_model(baseline_df, WIND_COL, POWER_COL)
    baseline = compute_baseline(
        baseline_df,
        anomaly_features=ANOMALY_FEATURES,
        wind_col=WIND_COL,
        wind_std_col=WIND_STD_COL,
        power_col=POWER_COL,
        power_std_col=POWER_STD_COL,
        pitch_col=PITCH_COL,
        rotor_col=ROTOR_COL,
        gen_col=GEN_COL,
        wind_dir_col=WIND_DIR_COL,
        nacelle_dir_col=NACELLE_DIR_COL,
        temp_cols=TEMP_COLS,
        slope=slope,
        intercept=intercept,
    )

    metrics = compute_window_metrics(
        window_df,
        baseline,
        slope,
        intercept,
        wind_col=WIND_COL,
        wind_std_col=WIND_STD_COL,
        wind_max_col=WIND_MAX_COL,
        power_col=POWER_COL,
        power_std_col=POWER_STD_COL,
        rotor_col=ROTOR_COL,
        gen_col=GEN_COL,
        pitch_col=PITCH_COL,
        wind_dir_col=WIND_DIR_COL,
        nacelle_dir_col=NACELLE_DIR_COL,
        temp_cols=TEMP_COLS,
    )
    tags = build_tags(metrics["z"])

    card = {
        "window_id": payload.get("window_id", ""),
        "source": payload.get("meta", {}),
        "window": {
            "end_time": str(end_time),
            "start_time": str(window_df["time_stamp"].iloc[0]),
            "row_count": int(len(window_df)),
            "baseline_hours": int(baseline_hours),
            "window_hours": int(window_hours),
        },
        "stats": metrics["stats"],
        "derived": metrics["derived"],
        "tags": tags,
    }

    if not no_llm:
        card["summary"] = generate_summary(card, model)

    return card


def resolve_output_path(output_path: str | None, input_path: Path, is_dir_mode: bool) -> Path:
    if is_dir_mode:
        if output_path is None:
            return BASE_DIR / "scada_cards_out"
        return Path(output_path)

    if output_path is None:
        return BASE_DIR / "scada_card.json"
    out = Path(output_path)
    if out.suffix.lower() == ".json":
        return out
    return out / f"{input_path.stem}_card.json"


def process_payload_file(
    input_path: Path,
    output_path: Path,
    baseline_hours_override: int | None,
    window_hours_override: int | None,
    model: str,
    no_llm: bool,
) -> None:
    payload = load_payload(input_path)
    csv_text = payload.get("csv")
    if not isinstance(csv_text, str) or not csv_text.strip():
        raise ValueError(f"payload is missing 'csv' in {input_path}")
    csv_sep = payload.get("csv_sep", ";")
    baseline_hours = baseline_hours_override or int(payload.get("baseline_hours", 168))
    window_hours = window_hours_override or int(payload.get("window_hours", 6))

    df = parse_csv_payload(csv_text, csv_sep)
    slices = slice_windows(df, baseline_hours, window_hours)

    card = build_card(
        payload,
        slices["baseline_df"],
        slices["window_df"],
        slices["end_time"],
        baseline_hours,
        window_hours,
        model=model,
        no_llm=no_llm,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(card, indent=2, ensure_ascii=True))


def main() -> None:
    args = parse_args()

    if args.input_json:
        input_path = Path(args.input_json)
        output_path = resolve_output_path(args.output_path, input_path, is_dir_mode=False)
        process_payload_file(
            input_path,
            output_path,
            args.baseline_hours,
            args.window_hours,
            model=args.model,
            no_llm=args.no_llm,
        )
        print(f"Saved card to {output_path}")
        return

    input_dir = Path(args.input_directory)
    if not input_dir.is_dir():
        raise ValueError(f"--input-directory is not a directory: {input_dir}")
    output_dir = resolve_output_path(args.output_path, input_dir, is_dir_mode=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob("*.json"))
    if not files:
        raise ValueError(f"No .json files found in {input_dir}")

    for input_path in files:
        output_path = output_dir / f"{input_path.stem}_card.json"
        process_payload_file(
            input_path,
            output_path,
            args.baseline_hours,
            args.window_hours,
            model=args.model,
            no_llm=args.no_llm,
        )

    print(f"Saved {len(files)} cards to {output_dir}")


if __name__ == "__main__":
    main()
