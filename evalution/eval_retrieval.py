#!/usr/bin/env python3
"""
Evaluation script for RAG retrieval pipeline.
Runs test cases through retrieval and outputs results for metric computation.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from Agents.Querybuilder import compose_query_pack
from manuals.rag_core import Retriever


# ---------------------------------------------------------------------------
# Metric computation functions
# ---------------------------------------------------------------------------


def compute_recall(retrieved_ids: List[str], gold_ids: List[str]) -> float:
    """Compute Recall: |retrieved ∩ gold| / |gold|."""
    if not gold_ids:
        return 0.0
    retrieved_set = set(retrieved_ids)
    gold_set = set(gold_ids)
    return len(retrieved_set & gold_set) / len(gold_set)


def compute_precision(retrieved_ids: List[str], gold_ids: List[str]) -> float:
    """Compute Precision: |retrieved ∩ gold| / |retrieved|."""
    if not retrieved_ids:
        return 0.0
    retrieved_set = set(retrieved_ids)
    gold_set = set(gold_ids)
    return len(retrieved_set & gold_set) / len(retrieved_set)


def compute_mrr(retrieved_ids: List[str], gold_ids: List[str]) -> float:
    """Compute Mean Reciprocal Rank: 1/rank of first gold hit."""
    gold_set = set(gold_ids)
    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in gold_set:
            return 1.0 / rank
    return 0.0


def compute_hit(retrieved_ids: List[str], gold_ids: List[str]) -> float:
    """Compute Hit: 1 if any gold in retrieved, else 0."""
    retrieved_set = set(retrieved_ids)
    gold_set = set(gold_ids)
    return 1.0 if retrieved_set & gold_set else 0.0


def compute_ndcg(retrieved_ids: List[str], gold_ids: List[str]) -> float:
    """
    Compute NDCG using gold list order as relevance grades.
    First gold chunk gets relevance n, second gets n-1, etc.
    """
    if not gold_ids:
        return 0.0

    # Build relevance map: first gold = len(gold_ids), second = len(gold_ids)-1, ...
    relevance = {chunk_id: len(gold_ids) - i for i, chunk_id in enumerate(gold_ids)}

    # Compute DCG
    dcg = 0.0
    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        rel = relevance.get(chunk_id, 0)
        if rel > 0:
            dcg += rel / math.log2(rank + 1)

    # Compute ideal DCG (gold chunks in perfect order)
    ideal_rels = sorted(relevance.values(), reverse=True)
    idcg = 0.0
    for rank, rel in enumerate(ideal_rels, start=1):
        idcg += rel / math.log2(rank + 1)

    if idcg == 0:
        return 0.0
    return dcg / idcg


def compute_all_metrics(
    retrieved_ids: List[str], gold_ids: List[str]
) -> Dict[str, float]:
    """Compute all metrics for a single case."""
    return {
        "recall": compute_recall(retrieved_ids, gold_ids),
        "precision": compute_precision(retrieved_ids, gold_ids),
        "mrr": compute_mrr(retrieved_ids, gold_ids),
        "hit": compute_hit(retrieved_ids, gold_ids),
        "ndcg": compute_ndcg(retrieved_ids, gold_ids),
    }


def load_test_cases(path: str) -> List[Dict[str, Any]]:
    """Load test cases from JSON file."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def run_retrieval(
    case: Dict[str, Any],
    retriever: Retriever,
    top_k: int = 10,
    use_llm: bool = True,
    model: str = "gpt-5.2",
) -> Dict[str, Any]:
    """
    Run retrieval for a single test case.

    Args:
        case: Test case dict with scada_case, mechanic_notes, fault_images_description
        retriever: Initialized Retriever instance
        top_k: Number of chunks to retrieve
        use_llm: Whether to use LLM for query composition
        model: OpenAI model for query composition

    Returns:
        Dict with case_id, class_label, inputs, query_pack, retrieved_chunks,
        gold_chunk_ids, and metrics
    """
    # Build context matching webapp format
    # Note: Querybuilder expects fault_images_description as a string, not a list
    context = {
        "scada_case": case.get("scada_case"),
        "fault_images_description": case.get("fault_images_description", ""),
        "mechanic_notes": case.get("mechanic_notes", ""),
    }

    # Compose query (with or without LLM)
    if use_llm:
        query_pack = compose_query_pack(context, model=model)
    else:
        # Force fallback by temporarily unsetting API key
        old_key = os.environ.pop("OPENAI_API_KEY", None)
        query_pack = compose_query_pack(context, model=None)
        if old_key:
            os.environ["OPENAI_API_KEY"] = old_key

    # Run retrieval
    results = retriever.search(
        query_text=query_pack["query"],
        top_k=top_k,
    )

    # Add rank to each result
    for i, chunk in enumerate(results, start=1):
        chunk["rank"] = i

    # Extract gold chunk IDs and compute metrics
    gold_chunk_ids = case.get("gold_chunk_ids", [])
    retrieved_ids = [chunk.get("chunk_id", "") for chunk in results]
    metrics = compute_all_metrics(retrieved_ids, gold_chunk_ids)

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
        "metrics": metrics,
    }


def evaluate_all(
    cases_path: str,
    index_dir: str,
    top_k: int = 10,
    use_llm: bool = True,
    model: str = "gpt-5.2",
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run evaluation on all test cases.

    Args:
        cases_path: Path to test cases JSON
        index_dir: Path to RAG index directory
        top_k: Number of chunks to retrieve per case
        use_llm: Whether to use LLM for query composition
        model: OpenAI model for query composition
        output_path: Path to save results JSON (optional)

    Returns:
        Dict with config and results
    """
    cases = load_test_cases(cases_path)
    print(f"Loaded {len(cases)} test cases from {cases_path}")

    print(f"Loading retriever from {index_dir}...")
    retriever = Retriever(index_dir)
    print(f"Retriever loaded with {len(retriever.metadata)} chunks")

    results = []
    for i, case in enumerate(cases, start=1):
        case_id = case.get("id", f"case_{i}")
        print(f"[{i}/{len(cases)}] Processing {case_id}...")

        try:
            result = run_retrieval(
                case, retriever, top_k=top_k, use_llm=use_llm, model=model
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

    # Compute aggregate metrics (skip cases with empty gold or errors)
    valid_results = [
        r for r in results
        if r.get("metrics") is not None and r.get("gold_chunk_ids")
    ]

    aggregate_metrics = {}
    if valid_results:
        metric_names = ["recall", "precision", "mrr", "hit", "ndcg"]
        for metric in metric_names:
            values = [r["metrics"][metric] for r in valid_results]
            aggregate_metrics[metric] = sum(values) / len(values)
        aggregate_metrics["num_evaluated"] = len(valid_results)
    else:
        aggregate_metrics = {
            "recall": 0.0,
            "precision": 0.0,
            "mrr": 0.0,
            "hit": 0.0,
            "ndcg": 0.0,
            "num_evaluated": 0,
        }

    output = {
        "config": {
            "top_k": top_k,
            "use_llm": use_llm,
            "model": model if use_llm else None,
            "num_cases": len(cases),
            "index_dir": index_dir,
        },
        "aggregate_metrics": aggregate_metrics,
        "results": results,
    }

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(json.dumps(output, indent=2, ensure_ascii=False))
        print(f"\nSaved results to {output_path}")

    # Print summary table
    print("\n" + "=" * 60)
    print("AGGREGATE METRICS")
    print("=" * 60)
    print(f"  Evaluated cases: {aggregate_metrics['num_evaluated']} / {len(cases)}")
    print(f"  Recall@{top_k}:    {aggregate_metrics['recall']:.4f}")
    print(f"  Precision@{top_k}: {aggregate_metrics['precision']:.4f}")
    print(f"  MRR:              {aggregate_metrics['mrr']:.4f}")
    print(f"  Hit@{top_k}:       {aggregate_metrics['hit']:.4f}")
    print(f"  NDCG@{top_k}:      {aggregate_metrics['ndcg']:.4f}")
    print("=" * 60)

    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RAG retrieval evaluation on test cases"
    )
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
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of chunks to retrieve (default: 10)",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Use fallback query builder (no LLM)",
    )
    parser.add_argument(
        "--model",
        default="gpt-5.2",
        help="OpenAI model for query composition (default: gpt-5.2)",
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
    )


if __name__ == "__main__":
    main()
