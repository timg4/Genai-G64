#!/usr/bin/env python3
"""
Evaluation script for RAG retrieval pipeline.
Runs test cases through retrieval and outputs results for metric computation.

Improvements vs. original:
- Adds robust "relaxed" metrics besides strict chunk_id matching:
  * source-level (manual match)
  * section-level (source+section match)
  * page-proximity match (same source and near page)
- Fixes NDCG: uses binary relevance (gold is a set, not an ordered ranking)
- Adds optional weighted recall (weights by chunk_kind)
- Adds per-class breakdown
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from Agents.Querybuilder import compose_query_pack
from manuals.rag_core import Retriever


# ---------------------------------------------------------------------------
# Config: kind weights (should mirror your retrieval biasing)
# ---------------------------------------------------------------------------

KIND_WEIGHTS: Dict[str, float] = {
    "procedure": 1.3,
    "checklist": 1.3,
    "troubleshooting": 1.3,
    "safety": 1.2,
    "inspection": 1.1,
    "training_admin": 0.7,
    "definition": 0.5,
    "changelog": 0.5,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def unique_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def safe_log2(x: float) -> float:
    return math.log(x, 2)


# ---------------------------------------------------------------------------
# Metric computation functions
# ---------------------------------------------------------------------------


def compute_recall(retrieved: List[str], gold: List[str]) -> float:
    if not gold:
        return 0.0
    return len(set(retrieved) & set(gold)) / len(set(gold))


def compute_precision(retrieved: List[str], gold: List[str]) -> float:
    if not retrieved:
        return 0.0
    return len(set(retrieved) & set(gold)) / len(set(retrieved))


def compute_mrr(retrieved: List[str], gold: List[str]) -> float:
    gold_set = set(gold)
    for rank, item in enumerate(retrieved, start=1):
        if item in gold_set:
            return 1.0 / rank
    return 0.0


def compute_hit(retrieved: List[str], gold: List[str]) -> float:
    return 1.0 if (set(retrieved) & set(gold)) else 0.0


def compute_ndcg_binary(retrieved: List[str], gold: List[str]) -> float:
    """
    Binary NDCG for a gold SET.
    DCG gives 1/log2(rank+1) for each relevant hit.
    IDCG assumes all relevant items are ranked at the top.
    """
    gold_set = set(gold)
    if not gold_set:
        return 0.0

    dcg = 0.0
    for rank, item in enumerate(retrieved, start=1):
        if item in gold_set:
            dcg += 1.0 / safe_log2(rank + 1)

    ideal_hits = min(len(gold_set), len(retrieved))
    idcg = sum(1.0 / safe_log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return (dcg / idcg) if idcg > 0 else 0.0


def compute_weighted_recall(
    retrieved: List[str],
    gold: List[str],
    kind_by_id: Dict[str, str],
) -> float:
    """
    Weighted recall: sum(weights of gold items retrieved) / sum(weights of gold items)
    Useful when gold mixes inspection+safety+procedure and you want to reward retrieving
    the important ones.
    """
    gold_set = set(gold)
    if not gold_set:
        return 0.0

    def w(cid: str) -> float:
        return KIND_WEIGHTS.get(kind_by_id.get(cid, ""), 1.0)

    denom = sum(w(cid) for cid in gold_set)
    if denom <= 0:
        return 0.0

    retrieved_set = set(retrieved)
    num = sum(w(cid) for cid in gold_set if cid in retrieved_set)
    return num / denom


def compute_all_metrics(
    retrieved: List[str],
    gold: List[str],
    kind_by_id: Optional[Dict[str, str]] = None,
) -> Dict[str, float]:
    metrics = {
        "recall": compute_recall(retrieved, gold),
        "precision": compute_precision(retrieved, gold),
        "mrr": compute_mrr(retrieved, gold),
        "hit": compute_hit(retrieved, gold),
        "ndcg": compute_ndcg_binary(retrieved, gold),
    }
    if kind_by_id is not None:
        metrics["weighted_recall"] = compute_weighted_recall(retrieved, gold, kind_by_id)
    return metrics


# ---------------------------------------------------------------------------
# IO / loading
# ---------------------------------------------------------------------------


def load_test_cases(path: str) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Retrieval runner
# ---------------------------------------------------------------------------


def run_retrieval(
    case: Dict[str, Any],
    retriever: Retriever,
    meta_by_id: Dict[str, Dict[str, Any]],
    top_k: int = 10,
    use_llm: bool = True,
    model: str = "gpt-5.2",
    page_tolerance: int = 1,
) -> Dict[str, Any]:
    """
    Runs retrieval and computes:
    - strict metrics on chunk_id
    - relaxed metrics on source
    - relaxed metrics on source+section
    - relaxed metrics on page proximity (same source, page within +/- tolerance)
    """
    context = {
        "scada_case": case.get("scada_case"),
        "fault_images_description": case.get("fault_images_description", ""),
        "mechanic_notes": case.get("mechanic_notes", ""),
    }

    if use_llm:
        query_pack = compose_query_pack(context, model=model)
    else:
        old_key = os.environ.pop("OPENAI_API_KEY", None)
        query_pack = compose_query_pack(context, model=None)
        if old_key:
            os.environ["OPENAI_API_KEY"] = old_key

    results = retriever.search(query_text=query_pack["query"], top_k=top_k)

    for i, chunk in enumerate(results, start=1):
        chunk["rank"] = i

    gold_chunk_ids: List[str] = case.get("gold_chunk_ids", []) or []
    retrieved_chunk_ids = [c.get("chunk_id", "") for c in results if c.get("chunk_id")]

    # Build mapping chunk_id -> kind/source/section/page
    kind_by_id = {cid: meta_by_id.get(cid, {}).get("chunk_kind", "") for cid in gold_chunk_ids}
    kind_by_id.update({cid: meta_by_id.get(cid, {}).get("chunk_kind", "") for cid in retrieved_chunk_ids})

    def chunk_to_source(cid: str) -> str:
        return meta_by_id.get(cid, {}).get("source", "")

    def chunk_to_section_key(cid: str) -> str:
        m = meta_by_id.get(cid, {})
        src = m.get("source", "")
        sec = m.get("section", "")
        return f"{src}::{sec}" if src or sec else ""

    def chunk_to_page(cid: str) -> Optional[int]:
        p = meta_by_id.get(cid, {}).get("page", None)
        return int(p) if isinstance(p, (int, float)) else None

    # Strict metrics (chunk_id)
    metrics_strict = compute_all_metrics(retrieved_chunk_ids, gold_chunk_ids, kind_by_id=kind_by_id)

    # Source-level (dedup order)
    gold_sources = unique_preserve_order([chunk_to_source(cid) for cid in gold_chunk_ids])
    retrieved_sources = unique_preserve_order([chunk_to_source(cid) for cid in retrieved_chunk_ids])
    metrics_source = compute_all_metrics(retrieved_sources, gold_sources)

    # Section-level (source+section)
    gold_sections = unique_preserve_order([chunk_to_section_key(cid) for cid in gold_chunk_ids])
    retrieved_sections = unique_preserve_order([chunk_to_section_key(cid) for cid in retrieved_chunk_ids])
    metrics_section = compute_all_metrics(retrieved_sections, gold_sections)

    # Page-proximity match:
    # A retrieved chunk counts as hit if:
    # - same source as any gold AND
    # - its page within +/- tolerance of any gold page for that source
    gold_pages_by_source: Dict[str, List[int]] = defaultdict(list)
    for cid in gold_chunk_ids:
        src = chunk_to_source(cid)
        p = chunk_to_page(cid)
        if src and p is not None:
            gold_pages_by_source[src].append(p)

    def page_match_key(rcid: str) -> str:
        src = chunk_to_source(rcid)
        p = chunk_to_page(rcid)
        if not src or p is None:
            return ""
        for gp in gold_pages_by_source.get(src, []):
            if abs(p - gp) <= page_tolerance:
                # return a stable key so we can score set-metrics
                return f"{src}::p{gp}"
        return ""

    # Build a gold "page keys" set (one per gold page)
    gold_page_keys = []
    for src, pages in gold_pages_by_source.items():
        for p in pages:
            gold_page_keys.append(f"{src}::p{p}")
    gold_page_keys = unique_preserve_order(gold_page_keys)

    retrieved_page_keys = []
    for cid in retrieved_chunk_ids:
        k = page_match_key(cid)
        if k:
            retrieved_page_keys.append(k)
    retrieved_page_keys = unique_preserve_order(retrieved_page_keys)

    metrics_page = compute_all_metrics(retrieved_page_keys, gold_page_keys)

    return {
        "case_id": case.get("id", ""),
        "class_label": case.get("scada_case", {}).get("class_label", ""),
        "inputs": {
            "mechanic_notes": case.get("mechanic_notes", ""),
            "fault_images_description": case.get("fault_images_description", ""),
        },
        "query_pack": query_pack,
        "retrieved_chunks": results,
        "gold_chunk_ids": gold_chunk_ids,
        "derived_gold": {
            "gold_sources": gold_sources,
            "gold_sections": gold_sections,
            "gold_page_keys": gold_page_keys,
        },
        "derived_retrieved": {
            "retrieved_sources": retrieved_sources,
            "retrieved_sections": retrieved_sections,
            "retrieved_page_keys": retrieved_page_keys,
        },
        "metrics": {
            "strict_chunk_id": metrics_strict,
            "source": metrics_source,
            "section": metrics_section,
            "page_proximity": metrics_page,
        },
    }


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------


def evaluate_all(
    cases_path: str,
    index_dir: str,
    top_k: int = 10,
    use_llm: bool = True,
    model: str = "gpt-5.2",
    output_path: Optional[str] = None,
    page_tolerance: int = 1,
) -> Dict[str, Any]:
    cases = load_test_cases(cases_path)
    print(f"Loaded {len(cases)} test cases from {cases_path}")

    print(f"Loading retriever from {index_dir}...")
    retriever = Retriever(index_dir)
    metadata_list = retriever.metadata if isinstance(retriever.metadata, list) else []
    print(f"Retriever loaded with {len(metadata_list)} chunks")

    meta_by_id: Dict[str, Dict[str, Any]] = {
        m.get("chunk_id", ""): m for m in metadata_list if m.get("chunk_id")
    }

    results: List[Dict[str, Any]] = []
    for i, case in enumerate(cases, start=1):
        case_id = case.get("id", f"case_{i}")
        print(f"[{i}/{len(cases)}] Processing {case_id}...")

        try:
            result = run_retrieval(
                case=case,
                retriever=retriever,
                meta_by_id=meta_by_id,
                top_k=top_k,
                use_llm=use_llm,
                model=model,
                page_tolerance=page_tolerance,
            )
            results.append(result)
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                "case_id": case_id,
                "class_label": case.get("scada_case", {}).get("class_label", ""),
                "error": str(e),
                "retrieved_chunks": [],
                "gold_chunk_ids": case.get("gold_chunk_ids", []),
                "metrics": None,
            })

    # Aggregate metrics per metric-group
    metric_groups = ["strict_chunk_id", "source", "section", "page_proximity"]
    metric_names = ["recall", "precision", "mrr", "hit", "ndcg"]

    valid_results = [
        r for r in results
        if r.get("metrics") is not None and (r.get("gold_chunk_ids") or r.get("derived_gold", {}).get("gold_sources"))
    ]

    aggregate_metrics: Dict[str, Any] = {"num_evaluated": len(valid_results), "groups": {}}

    for group in metric_groups:
        group_vals: Dict[str, List[float]] = {m: [] for m in metric_names}
        group_vals["weighted_recall"] = []

        for r in valid_results:
            m = (r.get("metrics") or {}).get(group, {})
            for name in metric_names:
                if name in m:
                    group_vals[name].append(float(m[name]))
            if "weighted_recall" in m:
                group_vals["weighted_recall"].append(float(m["weighted_recall"]))

        aggregate_metrics["groups"][group] = {
            k: (sum(v) / len(v) if v else 0.0) for k, v in group_vals.items()
        }

    # Per-class breakdown (use strict + section as the most informative)
    by_class: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in valid_results:
        by_class[r.get("class_label", "")].append(r)

    class_breakdown: Dict[str, Any] = {}
    for cl, items in by_class.items():
        def avg(group: str, metric: str) -> float:
            vals = []
            for it in items:
                m = it["metrics"].get(group, {})
                if metric in m:
                    vals.append(float(m[metric]))
            return sum(vals) / len(vals) if vals else 0.0

        class_breakdown[cl] = {
            "n": len(items),
            "strict_hit": avg("strict_chunk_id", "hit"),
            "strict_mrr": avg("strict_chunk_id", "mrr"),
            "section_hit": avg("section", "hit"),
            "section_mrr": avg("section", "mrr"),
        }

    output = {
        "config": {
            "top_k": top_k,
            "use_llm": use_llm,
            "model": model if use_llm else None,
            "num_cases": len(cases),
            "index_dir": index_dir,
            "page_tolerance": page_tolerance,
        },
        "aggregate_metrics": aggregate_metrics,
        "class_breakdown": class_breakdown,
        "results": results,
    }

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nSaved results to {output_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("AGGREGATE METRICS (mean over evaluated cases)")
    print("=" * 70)
    print(f"  Evaluated cases: {aggregate_metrics['num_evaluated']} / {len(cases)}")

    for group in metric_groups:
        g = aggregate_metrics["groups"].get(group, {})
        print("\n" + "-" * 70)
        print(f"  {group.upper()}")
        print("-" * 70)
        print(f"  Recall@{top_k}:    {g.get('recall', 0.0):.4f}")
        print(f"  Precision@{top_k}: {g.get('precision', 0.0):.4f}")
        print(f"  MRR:              {g.get('mrr', 0.0):.4f}")
        print(f"  Hit@{top_k}:       {g.get('hit', 0.0):.4f}")
        print(f"  NDCG@{top_k}:      {g.get('ndcg', 0.0):.4f}")
        if "weighted_recall" in g:
            print(f"  Weighted Recall:   {g.get('weighted_recall', 0.0):.4f}")

    print("\n" + "=" * 70)
    print("PER-CLASS (quick view)")
    print("=" * 70)
    for cl, stats in sorted(class_breakdown.items(), key=lambda x: x[0]):
        print(
            f"  {cl or 'UNKNOWN'} | n={stats['n']:>2} | strict_hit={stats['strict_hit']:.3f} "
            f"| section_hit={stats['section_hit']:.3f}"
        )
    print("=" * 70)

    return output


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RAG retrieval evaluation on test cases")
    parser.add_argument(
        "--cases",
        default=str(PROJECT_ROOT / "evalution" / "cases_with_gold.json"),
        help="Path to test cases JSON with gold_chunk_ids",
    )
    parser.add_argument(
        "--index-dir",
        default=str(PROJECT_ROOT / "manuals" / "manuals_index"),
        help="Path to RAG index directory",
    )
    parser.add_argument("--top-k", type=int, default=10, help="Number of chunks to retrieve (default: 10)")
    parser.add_argument("--no-llm", action="store_true", help="Use fallback query builder (no LLM)")
    parser.add_argument("--model", default="gpt-5.2", help="OpenAI model for query composition (default: gpt-5.2)")
    parser.add_argument(
        "--page-tolerance",
        type=int,
        default=1,
        help="Page tolerance for relaxed page-proximity matching (default: 1)",
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "evalution" / "results.json"),
        help="Output path for results JSON",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate_all(
        cases_path=args.cases,
        index_dir=args.index_dir,
        top_k=args.top_k,
        use_llm=not args.no_llm,
        model=args.model,
        output_path=args.output,
        page_tolerance=args.page_tolerance,
    )


if __name__ == "__main__":
    main()
