#!/usr/bin/env python3
"""
Extract 6h SCADA windows (with preceding baseline history) from the Wind Farm A
SCADA dataset and emit JSON payloads for simulation/testing.

Each output JSON embeds the CSV (history + window) and metadata required by
scada_card_from_window.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from scada_processing import (
    compute_baseline,
    compute_window_metrics,
    fit_power_model,
    score_classes,
)

CLASSES = ["VG;MT", "LE;ER", "LR;DA", "LE;CR", "SF;PO"]

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "CARE_To_Compare" / "Wind Farm A" / "datasets"
EVENT_INFO_PATH = BASE_DIR / "CARE_To_Compare" / "Wind Farm A" / "event_info.csv"

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

FEATURE_COLUMNS = [
    "wind_speed_3_avg",
    "wind_speed_3_max",
    "wind_speed_3_std",
    "wind_speed_4_avg",
    "power_30_avg",
    "power_30_std",
    "sensor_52_avg",
    "sensor_52_std",
    "sensor_18_avg",
    "sensor_18_std",
    "sensor_5_avg",
    "sensor_5_std",
    "sensor_1_avg",
    "sensor_42_avg",
    "sensor_11_avg",
    "sensor_12_avg",
    "sensor_13_avg",
    "sensor_14_avg",
]

CORE_COLUMNS = [
    "time_stamp",
    "asset_id",
    "id",
    "train_test",
    "status_type_id",
]

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
    parser = argparse.ArgumentParser(description="Sample SCADA windows for simulation.")
    parser.add_argument("--output-dir", default=str(BASE_DIR / "scada_windows"))
    parser.add_argument("--baseline-hours", type=int, default=168)
    parser.add_argument("--window-hours", type=int, default=6)
    parser.add_argument("--per-class", type=int, default=2)
    parser.add_argument("--nix-count", type=int, default=1)
    parser.add_argument("--csv-sep", default=";")
    return parser.parse_args()


def load_event_info(path: Path) -> Dict[int, Dict[str, str]]:
    df = pd.read_csv(path, sep=";")
    info = {}
    for _, row in df.iterrows():
        event_id = int(row["event_id"])
        info[event_id] = {
            "asset": str(row.get("asset", "")).strip(),
            "event_label": str(row.get("event_label", "")).strip(),
            "event_start": str(row.get("event_start", "")).strip(),
            "event_end": str(row.get("event_end", "")).strip(),
            "event_start_id": str(row.get("event_start_id", "")).strip(),
            "event_end_id": str(row.get("event_end_id", "")).strip(),
            "event_description": str(row.get("event_description", "")).strip(),
        }
    return info


def to_int(value: str) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except Exception:
        try:
            return int(float(value))
        except Exception:
            return None


def load_dataset(path: Path) -> pd.DataFrame:
    usecols = CORE_COLUMNS + FEATURE_COLUMNS
    df = pd.read_csv(path, sep=";", usecols=usecols)
    for col in FEATURE_COLUMNS + ["id", "status_type_id", "asset_id"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def select_candidate_df(df: pd.DataFrame, event_meta: Dict[str, str], window_len: int) -> pd.DataFrame:
    pred_df = df[df["train_test"] == "prediction"]
    if not event_meta:
        return pred_df
    start_id = to_int(event_meta.get("event_start_id", ""))
    end_id = to_int(event_meta.get("event_end_id", ""))
    if start_id is None or end_id is None:
        return pred_df
    subset = pred_df[(pred_df["id"] >= start_id) & (pred_df["id"] <= end_id)]
    if len(subset) >= window_len:
        return subset
    return pred_df


def compute_row_anomaly_scores(df: pd.DataFrame, baseline: Dict[str, float]) -> pd.Series:
    means = baseline["means"]
    stds = baseline["stds"]
    z = (df[ANOMALY_FEATURES] - means) / stds
    z = z.replace([float("inf"), float("-inf")], 0).fillna(0)
    return z.abs().mean(axis=1)


def select_window(row_scores: pd.Series, window_len: int, mode: str) -> Tuple[int, float]:
    scores = row_scores.rolling(window=window_len).mean()
    if scores.isna().all():
        return -1, float("nan")
    if mode == "min":
        idx = scores.idxmin()
    else:
        idx = scores.idxmax()
    return int(idx), float(scores.loc[idx])


def select_candidates_by_class(
    candidates: List[Dict[str, object]], per_class: int
) -> List[Tuple[Dict[str, object], str]]:
    selected: List[Tuple[Dict[str, object], str]] = []
    used_dataset_ids = set()
    for class_label in CLASSES:
        ranked = sorted(candidates, key=lambda c: c["scores"][class_label], reverse=True)
        picks: List[Dict[str, object]] = []
        for cand in ranked:
            if cand["dataset_id"] in used_dataset_ids:
                continue
            picks.append(cand)
            used_dataset_ids.add(cand["dataset_id"])
            if len(picks) >= per_class:
                break
        if len(picks) < per_class:
            for cand in ranked:
                if cand in picks:
                    continue
                picks.append(cand)
                if len(picks) >= per_class:
                    break
        for cand in picks:
            selected.append((cand, class_label))
    return selected


def select_nix_candidates(nix_candidates: List[Dict[str, object]], nix_count: int) -> List[Dict[str, object]]:
    ranked = sorted(nix_candidates, key=lambda c: c["anomaly_score"])
    selected = []
    used_dataset_ids = set()
    for cand in ranked:
        if cand["dataset_id"] in used_dataset_ids:
            continue
        selected.append(cand)
        used_dataset_ids.add(cand["dataset_id"])
        if len(selected) >= nix_count:
            break
    if len(selected) < nix_count:
        for cand in ranked:
            if cand in selected:
                continue
            selected.append(cand)
            if len(selected) >= nix_count:
                break
    return selected


def make_window_payload(
    df: pd.DataFrame,
    window_end: pd.Timestamp,
    baseline_hours: int,
    window_hours: int,
    window_id: str,
    meta: Dict[str, object],
    csv_sep: str,
) -> Dict[str, object]:
    dt = pd.to_datetime(df["time_stamp"], errors="coerce")
    baseline_start = window_end - pd.Timedelta(hours=baseline_hours)
    history_mask = (dt >= baseline_start) & (dt <= window_end)
    history_df = df[history_mask].copy()
    history_df = history_df.dropna(subset=["time_stamp"])
    csv_text = history_df.to_csv(sep=csv_sep, index=False)

    return {
        "window_id": window_id,
        "baseline_hours": baseline_hours,
        "window_hours": window_hours,
        "csv_sep": csv_sep,
        "csv": csv_text,
        "meta": meta,
    }


def main() -> None:
    args = parse_args()
    window_len = int(round(args.window_hours * 6))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    event_info = load_event_info(EVENT_INFO_PATH)
    candidates = []
    nix_candidates = []

    for dataset_path in sorted(DATASET_DIR.glob("*.csv"), key=lambda p: int(p.stem)):
        dataset_id = int(dataset_path.stem)
        event_meta = event_info.get(dataset_id, {})

        df = load_dataset(dataset_path)
        train_df = df[df["train_test"] == "train"]
        if len(train_df) < window_len:
            continue

        slope, intercept = fit_power_model(train_df, WIND_COL, POWER_COL)
        baseline = compute_baseline(
            train_df,
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
        candidate_df = select_candidate_df(df, event_meta, window_len)
        if len(candidate_df) < window_len:
            continue

        cand = candidate_df.reset_index(drop=True)
        row_scores = compute_row_anomaly_scores(cand, baseline)
        idx_max, max_score = select_window(row_scores, window_len, mode="max")
        if idx_max < 0:
            continue
        start = idx_max - window_len + 1
        end = idx_max
        window_df = cand.iloc[start : end + 1]

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
        scores = score_classes(metrics["z"])

        window_info = {
            "start_time": str(window_df["time_stamp"].iloc[0]),
            "end_time": str(window_df["time_stamp"].iloc[-1]),
            "start_id": int(window_df["id"].iloc[0]),
            "end_id": int(window_df["id"].iloc[-1]),
            "row_count": int(len(window_df)),
        }

        candidates.append(
            {
                "dataset_id": dataset_id,
                "asset_id": int(window_df["asset_id"].iloc[0]),
                "event_label": event_meta.get("event_label", ""),
                "event_description": event_meta.get("event_description", ""),
                "window": window_info,
                "stats": metrics["stats"],
                "derived": metrics["derived"],
                "scores": scores,
                "anomaly_score": max_score,
            }
        )

        if event_meta.get("event_label") == "normal":
            idx_min, min_score = select_window(row_scores, window_len, mode="min")
            if idx_min >= 0:
                start_min = idx_min - window_len + 1
                end_min = idx_min
                window_min = cand.iloc[start_min : end_min + 1]
                metrics_min = compute_window_metrics(
                    window_min,
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
                window_info_min = {
                    "start_time": str(window_min["time_stamp"].iloc[0]),
                    "end_time": str(window_min["time_stamp"].iloc[-1]),
                    "start_id": int(window_min["id"].iloc[0]),
                    "end_id": int(window_min["id"].iloc[-1]),
                    "row_count": int(len(window_min)),
                }
                nix_candidates.append(
                    {
                        "dataset_id": dataset_id,
                        "asset_id": int(window_min["asset_id"].iloc[0]),
                        "event_label": event_meta.get("event_label", ""),
                        "event_description": event_meta.get("event_description", ""),
                        "window": window_info_min,
                        "stats": metrics_min["stats"],
                        "derived": metrics_min["derived"],
                        "scores": score_classes(metrics_min["z"]),
                        "anomaly_score": min_score,
                    }
                )

    selected = select_candidates_by_class(candidates, args.per_class)
    nix_selected = [(cand, "Nix") for cand in select_nix_candidates(nix_candidates, args.nix_count)]
    selected.extend(nix_selected)

    by_class: Dict[str, List[str]] = {}
    manifest = []
    used_ids = {}

    for cand, class_label in selected:
        window_end = pd.to_datetime(cand["window"]["end_time"], errors="coerce")
        if pd.isna(window_end):
            continue
        base_id = f"WF-A-{cand['dataset_id']}-{cand['window']['end_id']}"
        count = used_ids.get(base_id, 0)
        used_ids[base_id] = count + 1
        window_id = base_id if count == 0 else f"{base_id}-{count}"
        file_name = f"{window_id}.json"

        meta = {
            "source": "Wind Farm A",
            "dataset_id": cand["dataset_id"],
            "asset_id": cand["asset_id"],
            "event_id": cand["dataset_id"],
            "event_label": cand["event_label"],
            "event_description": cand["event_description"],
            "window_end": cand["window"]["end_time"],
        }

        payload = make_window_payload(
            df=load_dataset(DATASET_DIR / f"{cand['dataset_id']}.csv"),
            window_end=window_end,
            baseline_hours=args.baseline_hours,
            window_hours=args.window_hours,
            window_id=window_id,
            meta=meta,
            csv_sep=args.csv_sep,
        )

        (output_dir / file_name).write_text(json.dumps(payload, indent=2, ensure_ascii=True))
        by_class.setdefault(class_label, []).append(window_id)
        manifest.append({"window_id": window_id, "file": file_name, "meta": meta})

    (output_dir / "scada_windows_manifest.json").write_text(
        json.dumps({"windows": manifest}, indent=2, ensure_ascii=True)
    )
    (output_dir / "scada_windows_by_class.json").write_text(
        json.dumps(by_class, indent=2, ensure_ascii=True)
    )

    print(f"Saved {len(manifest)} windows to {output_dir}")


if __name__ == "__main__":
    main()
