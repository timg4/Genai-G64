import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

UNCERTAIN_LABEL = "Uncertain"
NONE_LABEL = "<none>"
CANON_LABELS = [
    "VG;MT",
    "LE;ER",
    "LR;DA",
    "LE;CR",
    "SF;PO",
    "No Damage",
    "Invalid_Image",
]


def _as_label_list(value: object) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, str):
        val = value.strip()
        return [val] if val else []
    return [str(value).strip()]


def load_results(path: Path) -> List[Dict[str, Set[str]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "results" in payload:
        items = payload.get("results", [])
    elif isinstance(payload, list):
        items = payload
    else:
        raise ValueError("eval_results.json must be a list or {\"results\": [...]} ")

    rows: List[Dict[str, Set[str]]] = []
    for item in items:
        filename = str(item.get("filename", "")).strip()
        gold = set(_as_label_list(item.get("gold_labels")))
        pred = set(_as_label_list(item.get("pred_labels")))
        rows.append({"filename": filename, "gold": gold, "pred": pred})
    return rows


def _collect_labels(rows: Iterable[Dict[str, Set[str]]], include_uncertain: bool) -> List[str]:
    labels: Set[str] = set()
    for row in rows:
        labels.update(row["gold"])
        labels.update(row["pred"])
    if not include_uncertain:
        labels.discard(UNCERTAIN_LABEL)
    return sorted(labels)


def _prepare_rows(
    rows: List[Dict[str, Set[str]]], mode: str
) -> Tuple[List[Dict[str, Set[str]]], int]:
    prepared: List[Dict[str, Set[str]]] = []
    skipped = 0
    for row in rows:
        gold = set(row["gold"])
        pred = set(row["pred"])
        has_uncertain = UNCERTAIN_LABEL in pred
        pred_no_uncertain = pred - {UNCERTAIN_LABEL}

        if mode == "strict":
            pred_use = pred
        elif mode == "abstain_penalize":
            pred_use = pred_no_uncertain
        elif mode == "abstain_skip":
            if has_uncertain and not pred_no_uncertain:
                skipped += 1
                continue
            pred_use = pred_no_uncertain
        else:
            raise ValueError(f"Unknown mode: {mode}")

        prepared.append({"filename": row["filename"], "gold": gold, "pred": pred_use})
    return prepared, skipped


def compute_metrics(rows: List[Dict[str, Set[str]]], labels: List[str]) -> Dict[str, object]:
    totals = {label: {"tp": 0, "fp": 0, "fn": 0} for label in labels}
    exact = 0
    for row in rows:
        gold = row["gold"]
        pred = row["pred"]
        if gold == pred:
            exact += 1
        for label in labels:
            if label in pred and label in gold:
                totals[label]["tp"] += 1
            elif label in pred and label not in gold:
                totals[label]["fp"] += 1
            elif label not in pred and label in gold:
                totals[label]["fn"] += 1

    def safe_div(a: float, b: float) -> float:
        return a / b if b else 0.0

    per_class = {}
    micro_tp = micro_fp = micro_fn = 0
    for label, c in totals.items():
        tp, fp, fn = c["tp"], c["fp"], c["fn"]
        micro_tp += tp
        micro_fp += fp
        micro_fn += fn
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall)
        per_class[label] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    micro_precision = safe_div(micro_tp, micro_tp + micro_fp)
    micro_recall = safe_div(micro_tp, micro_tp + micro_fn)
    micro_f1 = safe_div(2 * micro_precision * micro_recall, micro_precision + micro_recall)

    macro_f1 = safe_div(sum(v["f1"] for v in per_class.values()), len(per_class))

    return {
        "num_samples": len(rows),
        "exact_match_accuracy": safe_div(exact, len(rows)),
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "labels": labels,
        "per_class": per_class,
    }


def write_predicted_labels(path: Path, rows: List[Dict[str, Set[str]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "pred_labels"])
        writer.writeheader()
        for row in rows:
            pred_text = "|".join(sorted(row["pred"])) if row["pred"] else ""
            writer.writerow({"filename": row["filename"], "pred_labels": pred_text})


def _label_set_key(labels: Set[str]) -> str:
    if not labels:
        return "<none>"
    return "|".join(sorted(labels))


def compute_confusion_matrix(rows: List[Dict[str, Set[str]]]) -> Tuple[List[str], List[List[int]]]:
    keys: Set[str] = set()
    for row in rows:
        keys.add(_label_set_key(row["gold"]))
        keys.add(_label_set_key(row["pred"]))
    labels = sorted(keys)
    index = {label: i for i, label in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]
    for row in rows:
        gold_key = _label_set_key(row["gold"])
        pred_key = _label_set_key(row["pred"])
        matrix[index[gold_key]][index[pred_key]] += 1
    return labels, matrix


def compute_label_confusion_matrix(
    rows: List[Dict[str, Set[str]]],
    label_order: List[str],
    none_label: str = NONE_LABEL,
) -> Tuple[List[str], List[List[int]]]:
    labels = list(label_order)
    if none_label not in labels:
        labels.append(none_label)
    index = {label: i for i, label in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]
    for row in rows:
        gold = row["gold"]
        pred = row["pred"] or {none_label}
        for g in gold:
            if g not in index:
                continue
            for p in pred:
                if p not in index:
                    continue
                matrix[index[g]][index[p]] += 1
    return labels, matrix


def compute_label_stats(
    rows: List[Dict[str, Set[str]]], labels: List[str]
) -> Dict[str, Dict[str, int]]:
    stats: Dict[str, Dict[str, int]] = {}
    for label in labels:
        tp = fp = fn = tn = 0
        for row in rows:
            gold_has = label in row["gold"]
            pred_has = label in row["pred"]
            if gold_has and pred_has:
                tp += 1
            elif not gold_has and pred_has:
                fp += 1
            elif gold_has and not pred_has:
                fn += 1
            else:
                tn += 1
        stats[label] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn}
    return stats


def write_confusion_matrix_csv(path: Path, labels: List[str], matrix: List[List[int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["gold\\pred"] + labels)
        for label, row in zip(labels, matrix):
            writer.writerow([label] + row)


def write_confusion_matrix_csv_with_stats(
    path: Path, labels: List[str], matrix: List[List[int]], stats: Dict[str, Dict[str, int]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["gold\\pred"] + labels + ["TP", "FP", "FN", "TN"])
        for label, row in zip(labels, matrix):
            s = stats.get(label, {"tp": 0, "fp": 0, "fn": 0, "tn": 0})
            writer.writerow([label] + row + [s["tp"], s["fp"], s["fn"], s["tn"]])


def write_confusion_matrix_plot(path: Path, labels: List[str], matrix: List[List[int]]) -> str:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return "matplotlib_missing"

    size = max(6.0, 0.6 * len(labels))
    fig, ax = plt.subplots(figsize=(size, size))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Gold")
    for i, row in enumerate(matrix):
        for j, val in enumerate(row):
            if val:
                ax.text(j, i, str(val), ha="center", va="center", fontsize=8, color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return "ok"


def write_confusion_matrix_html(path: Path, labels: List[str], matrix: List[List[int]]) -> None:
    max_val = max((max(row) for row in matrix), default=0)
    def cell_color(val: int) -> str:
        if max_val <= 0:
            return "#ffffff"
        t = val / max_val
        shade = int(255 - (120 * t))
        return f"#{shade:02x}{shade:02x}ff"

    rows = []
    header = "".join(f"<th>{label}</th>" for label in labels)
    rows.append(f"<tr><th>gold/pred</th>{header}</tr>")
    for label, row in zip(labels, matrix):
        cells = []
        for val in row:
            color = cell_color(val)
            cells.append(f"<td style='background:{color}'>{val}</td>")
        rows.append(f"<tr><th>{label}</th>{''.join(cells)}</tr>")

    html = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<style>table{border-collapse:collapse;font-family:Arial,sans-serif;font-size:12px}"
        "th,td{border:1px solid #ccc;padding:4px 6px;text-align:center}</style>"
        "</head><body>"
        "<h3>Confusion Matrix (abstain_skip)</h3>"
        "<table>" + "".join(rows) + "</table>"
        "</body></html>"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute metrics from eval_results.json")
    parser.add_argument(
        "--results",
        default="Faulty_Image_Describtion_runs_eval/eval_results.json",
        help="Path to eval_results.json",
    )
    parser.add_argument(
        "--out-dir",
        default="",
        help="Output directory (defaults to eval_results.json folder)",
    )
    parser.add_argument("--write-predicted", action="store_true")
    parser.add_argument("--write-summary", action="store_true")
    parser.add_argument("--write-confusion", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_path = Path(args.results).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else results_path.parent
    rows = load_results(results_path)

    uncertain_only = sum(1 for r in rows if r["pred"] == {UNCERTAIN_LABEL})
    uncertain_with_other = sum(
        1 for r in rows if UNCERTAIN_LABEL in r["pred"] and len(r["pred"]) > 1
    )

    strict_rows, strict_skipped = _prepare_rows(rows, "strict")
    abstain_rows, abstain_skipped = _prepare_rows(rows, "abstain_penalize")
    skip_rows, skip_skipped = _prepare_rows(rows, "abstain_skip")

    strict_labels = _collect_labels(strict_rows, include_uncertain=True)
    abstain_labels = _collect_labels(abstain_rows, include_uncertain=False)
    skip_labels = _collect_labels(skip_rows, include_uncertain=False)

    summary = {
        "num_samples": len(rows),
        "uncertain_only": uncertain_only,
        "uncertain_with_other": uncertain_with_other,
        "modes": {
            "strict": {
                "skipped": strict_skipped,
                **compute_metrics(strict_rows, strict_labels),
            },
            "abstain_penalize": {
                "skipped": abstain_skipped,
                **compute_metrics(abstain_rows, abstain_labels),
            },
            "abstain_skip": {
                "skipped": skip_skipped,
                "coverage": (len(skip_rows) / len(rows)) if rows else 0.0,
                **compute_metrics(skip_rows, skip_labels),
            },
        },
    }

    if args.write_predicted:
        write_predicted_labels(out_dir / "predicted_labels.csv", rows)
    if args.write_summary:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "metrics_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    if args.write_confusion:
        labels_sets, matrix_sets = compute_confusion_matrix(skip_rows)
        write_confusion_matrix_csv(
            out_dir / "confusion_matrix_abstain_skip.csv", labels_sets, matrix_sets
        )
        labels_found: Set[str] = set()
        for row in skip_rows:
            labels_found.update(row["gold"])
            labels_found.update(row["pred"])
        labels_found.discard(UNCERTAIN_LABEL)
        label_order = [l for l in CANON_LABELS if l in labels_found]
        label_order += sorted(labels_found - set(label_order))
        labels_flat, matrix_flat = compute_label_confusion_matrix(skip_rows, label_order)
        stats_flat = compute_label_stats(skip_rows, labels_flat)
        write_confusion_matrix_csv(
            out_dir / "confusion_matrix_abstain_skip_labels.csv",
            labels_flat,
            matrix_flat,
        )
        write_confusion_matrix_csv_with_stats(
            out_dir / "confusion_matrix_abstain_skip_labels_with_stats.csv",
            labels_flat,
            matrix_flat,
            stats_flat,
        )
        png_status = write_confusion_matrix_plot(
            out_dir / "confusion_matrix_abstain_skip_labels.png",
            labels_flat,
            matrix_flat,
        )
        if png_status != "ok":
            write_confusion_matrix_html(
                out_dir / "confusion_matrix_abstain_skip_labels.html",
                labels_flat,
                matrix_flat,
            )
        if args.write_summary:
            summary["modes"]["abstain_skip"]["confusion_matrix_sets"] = {
                "labels": labels_sets,
                "matrix": matrix_sets,
            }
            summary["modes"]["abstain_skip"]["confusion_matrix_labels"] = {
                "labels": labels_flat,
                "matrix": matrix_flat,
                "png_status": png_status,
                "stats": stats_flat,
            }

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
