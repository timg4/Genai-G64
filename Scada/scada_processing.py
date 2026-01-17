#!/usr/bin/env python3
"""
Processing utilities for SCADA windows.
Compute baseline stats, derive metrics, score class fit, and generate summaries.
"""

from __future__ import annotations

import json
import math
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


def angle_diff_deg(a: pd.Series, b: pd.Series) -> pd.Series:
    diff = (a - b + 180.0) % 360.0 - 180.0
    return diff.abs()


def safe_float(value: float, ndigits: int = 4) -> float:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return float("nan")
    return float(round(value, ndigits))


def z_score(value: float, mean: float, std: float) -> float:
    if std is None or std == 0 or (isinstance(std, float) and math.isnan(std)):
        return 0.0
    return (value - mean) / std


def fit_power_model(train_df: pd.DataFrame, wind_col: str, power_col: str) -> Tuple[float, float]:
    x = train_df[wind_col].to_numpy()
    y = train_df[power_col].to_numpy()
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        intercept = float(np.nanmean(y)) if mask.sum() > 0 else 0.0
        return 0.0, intercept
    slope, intercept = np.polyfit(x[mask], y[mask], 1)
    return float(slope), float(intercept)


def power_residuals(
    df: pd.DataFrame, slope: float, intercept: float, wind_col: str, power_col: str
) -> np.ndarray:
    wind = df[wind_col].to_numpy()
    power = df[power_col].to_numpy()
    expected = slope * wind + intercept
    denom = np.maximum(np.abs(expected), 1.0)
    return (power - expected) / denom


def compute_baseline(
    train_df: pd.DataFrame,
    anomaly_features: List[str],
    wind_col: str,
    wind_std_col: str,
    power_col: str,
    power_std_col: str,
    pitch_col: str,
    rotor_col: str,
    gen_col: str,
    wind_dir_col: str,
    nacelle_dir_col: str,
    temp_cols: List[str],
    slope: float,
    intercept: float,
) -> Dict[str, float]:
    baseline = {}
    means = train_df[anomaly_features].mean()
    stds = train_df[anomaly_features].std().replace(0, 1.0).fillna(1.0)
    baseline["means"] = means
    baseline["stds"] = stds

    baseline["wind_mean"] = float(train_df[wind_col].mean())
    baseline["wind_std"] = float(train_df[wind_col].std() or 1.0)
    baseline["power_mean"] = float(train_df[power_col].mean())
    baseline["power_std"] = float(train_df[power_col].std() or 1.0)
    baseline["pitch_mean"] = float(train_df[pitch_col].mean())
    baseline["pitch_std"] = float(train_df[pitch_col].std() or 1.0)
    baseline["rotor_std"] = float(train_df[rotor_col].std() or 1.0)
    baseline["gen_std"] = float(train_df[gen_col].std() or 1.0)
    baseline["power_std_mean"] = float(train_df[power_std_col].mean())
    baseline["power_std_std"] = float(train_df[power_std_col].std() or 1.0)
    baseline["wind_std_mean"] = float(train_df[wind_std_col].mean())
    baseline["wind_std_std"] = float(train_df[wind_std_col].std() or 1.0)

    yaw = angle_diff_deg(train_df[wind_dir_col], train_df[nacelle_dir_col])
    baseline["yaw_mean"] = float(yaw.mean())
    baseline["yaw_std"] = float(yaw.std() or 1.0)

    temps = train_df[temp_cols].mean(axis=1)
    baseline["temp_mean"] = float(temps.mean())
    baseline["temp_std"] = float(temps.std() or 1.0)

    residuals = power_residuals(train_df, slope, intercept, wind_col, power_col)
    baseline["residual_mean"] = float(np.nanmean(residuals))
    baseline["residual_std"] = float(np.nanstd(residuals) or 1.0)

    status = train_df["status_type_id"].dropna()
    if len(status) == 0:
        baseline["status_ratio"] = 0.0
    else:
        non_normal = status[~status.isin([0, 2])]
        baseline["status_ratio"] = float(len(non_normal) / len(status))

    return baseline


def compute_window_metrics(
    window_df: pd.DataFrame,
    baseline: Dict[str, float],
    slope: float,
    intercept: float,
    wind_col: str,
    wind_std_col: str,
    wind_max_col: str,
    power_col: str,
    power_std_col: str,
    rotor_col: str,
    gen_col: str,
    pitch_col: str,
    wind_dir_col: str,
    nacelle_dir_col: str,
    temp_cols: List[str],
) -> Dict[str, Dict[str, float]]:
    wind = window_df[wind_col]
    power = window_df[power_col]
    rotor = window_df[rotor_col]
    gen = window_df[gen_col]
    pitch = window_df[pitch_col]
    yaw = angle_diff_deg(window_df[wind_dir_col], window_df[nacelle_dir_col])
    temps = window_df[temp_cols].mean(axis=1)

    residuals = power_residuals(window_df, slope, intercept, wind_col, power_col)

    status = window_df["status_type_id"].dropna()
    status_counts = status.value_counts().to_dict()
    if len(status) == 0:
        non_normal_ratio = 0.0
    else:
        non_normal_ratio = float(len(status[~status.isin([0, 2])]) / len(status))

    stats = {
        "wind_speed_mean": safe_float(wind.mean()),
        "wind_speed_std": safe_float(wind.std()),
        "wind_speed_max": safe_float(window_df[wind_max_col].max()),
        "power_mean": safe_float(power.mean()),
        "power_std": safe_float(power.std()),
        "rotor_rpm_mean": safe_float(rotor.mean()),
        "rotor_rpm_std": safe_float(rotor.std()),
        "generator_rpm_mean": safe_float(gen.mean()),
        "generator_rpm_std": safe_float(gen.std()),
        "pitch_mean": safe_float(pitch.mean()),
        "pitch_std": safe_float(pitch.std()),
        "yaw_misalignment_mean": safe_float(yaw.mean()),
        "yaw_misalignment_std": safe_float(yaw.std()),
        "temp_mean": safe_float(temps.mean()),
    }

    derived = {
        "power_residual_mean": safe_float(float(np.nanmean(residuals))),
        "power_residual_std": safe_float(float(np.nanstd(residuals))),
        "status_non_normal_ratio": safe_float(non_normal_ratio),
        "status_counts": {str(k): int(v) for k, v in status_counts.items()},
    }

    z = {
        "power_residual_mean": z_score(
            float(np.nanmean(residuals)), baseline["residual_mean"], baseline["residual_std"]
        ),
        "wind_speed_mean": z_score(stats["wind_speed_mean"], baseline["wind_mean"], baseline["wind_std"]),
        "wind_speed_std": z_score(stats["wind_speed_std"], baseline["wind_std_mean"], baseline["wind_std_std"]),
        "yaw_misalignment_mean": z_score(stats["yaw_misalignment_mean"], baseline["yaw_mean"], baseline["yaw_std"]),
        "pitch_mean": z_score(stats["pitch_mean"], baseline["pitch_mean"], baseline["pitch_std"]),
        "rotor_rpm_std": z_score(stats["rotor_rpm_std"], 0.0, baseline["rotor_std"]),
        "generator_rpm_std": z_score(stats["generator_rpm_std"], 0.0, baseline["gen_std"]),
        "power_std": z_score(stats["power_std"], baseline["power_std_mean"], baseline["power_std_std"]),
        "temp_mean": z_score(stats["temp_mean"], baseline["temp_mean"], baseline["temp_std"]),
    }

    p = baseline["status_ratio"]
    denom = max(math.sqrt(p * (1.0 - p) / max(len(window_df), 1)), 0.05)
    z["status_non_normal_ratio"] = (non_normal_ratio - p) / denom

    return {"stats": stats, "derived": derived, "z": z}


def score_classes(z: Dict[str, float]) -> Dict[str, float]:
    residual_z = z["power_residual_mean"]
    mild_neg = max(0.0, 1.0 - abs(residual_z + 0.5))
    strong_neg = max(-residual_z, 0.0)

    scores = {}
    scores["VG;MT"] = (
        mild_neg * 1.0
        + max(z["pitch_mean"], 0.0) * 0.3
        - abs(z["status_non_normal_ratio"]) * 0.2
    )
    scores["LE;ER"] = (
        strong_neg * 1.0
        + max(z["pitch_mean"], 0.0) * 0.5
        + max(z["power_std"], 0.0) * 0.3
    )
    scores["LR;DA"] = (
        max(z["wind_speed_mean"], 0.0) * 0.5
        + max(z["wind_speed_std"], 0.0) * 0.7
        + max(z["yaw_misalignment_mean"], 0.0) * 0.6
        + max(z["status_non_normal_ratio"], 0.0) * 0.4
    )
    scores["LE;CR"] = (
        max(z["rotor_rpm_std"], 0.0) * 0.6
        + max(z["generator_rpm_std"], 0.0) * 0.6
        + max(z["temp_mean"], 0.0) * 0.5
        + max(z["status_non_normal_ratio"], 0.0) * 0.3
    )
    scores["SF;PO"] = (
        mild_neg * 0.9
        - max(abs(z["pitch_mean"]) - 1.0, 0.0) * 0.2
        - max(z["status_non_normal_ratio"], 0.0) * 0.2
    )
    return scores


def build_tags(z: Dict[str, float]) -> List[str]:
    tags = []
    if z["wind_speed_mean"] > 1.5:
        tags.append("high_wind")
    if z["wind_speed_std"] > 1.5:
        tags.append("high_wind_variability")
    if z["yaw_misalignment_mean"] > 1.5:
        tags.append("yaw_misalignment")
    if z["power_residual_mean"] < -1.0:
        tags.append("power_deficit")
    if z["power_std"] > 1.5:
        tags.append("power_variability")
    if z["rotor_rpm_std"] > 1.5 or z["generator_rpm_std"] > 1.5:
        tags.append("rpm_instability")
    if z["temp_mean"] > 1.5:
        tags.append("elevated_temps")
    if z["status_non_normal_ratio"] > 1.5:
        tags.append("derated_or_downtime")
    return tags


def build_summary_prompt(card: Dict[str, object]) -> str:
    stats = card["stats"]
    derived = card["derived"]
    tags = ", ".join(card.get("tags", [])) or "none"
    prompt = (
        "Summarize this 6h SCADA window in 2-3 sentences. "
        "Focus on operational patterns and anomalies. "
        "Do not diagnose faults or reference labels. "
        "Stats: wind_speed_mean={wind_speed_mean}, wind_speed_std={wind_speed_std}, "
        "wind_speed_max={wind_speed_max}, power_mean={power_mean}, power_std={power_std}, "
        "rotor_rpm_std={rotor_rpm_std}, generator_rpm_std={generator_rpm_std}, "
        "pitch_mean={pitch_mean}, yaw_misalignment_mean={yaw_misalignment_mean}, "
        "power_residual_mean={power_residual_mean}, status_non_normal_ratio={status_non_normal_ratio}, "
        "temp_mean={temp_mean}. Tags: {tags}."
    )
    return prompt.format(
        wind_speed_mean=stats["wind_speed_mean"],
        wind_speed_std=stats["wind_speed_std"],
        wind_speed_max=stats["wind_speed_max"],
        power_mean=stats["power_mean"],
        power_std=stats["power_std"],
        rotor_rpm_std=stats["rotor_rpm_std"],
        generator_rpm_std=stats["generator_rpm_std"],
        pitch_mean=stats["pitch_mean"],
        yaw_misalignment_mean=stats["yaw_misalignment_mean"],
        power_residual_mean=derived["power_residual_mean"],
        status_non_normal_ratio=derived["status_non_normal_ratio"],
        temp_mean=stats["temp_mean"],
        tags=tags,
    )


def generate_summary(card: Dict[str, object], model: str) -> str:
    prompt = build_summary_prompt(card)

    if OpenAI is None:
        raise RuntimeError("openai package is not installed")

    client = OpenAI()
    response = client.responses.create(model=model, input=prompt)
    text = getattr(response, "output_text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()

    data = response.model_dump() if hasattr(response, "model_dump") else response.__dict__
    return json.dumps(data, ensure_ascii=True, indent=2)
