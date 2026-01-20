#!/usr/bin/env python3
"""
Evaluation script for RAG retrieval pipeline.
Runs test cases through retrieval and outputs results for metric computation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from Agents.Querybuilder import compose_query_pack
from manuals.rag_core import Retriever


def load_test_cases(path: str) -> List[Dict[str, Any]]:
    """Load test cases from JSON file."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def run_retrieval(
    case: Dict[str, Any],
    retriever: Retriever,
    top_k: int = 10,
    use_llm: bool = True,
    model: str = "gpt-4o-mini",
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
        Dict with case_id, class_label, inputs, query_pack, retrieved_chunks
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

    return {
        "case_id": case.get("id", ""),
        "class_label": case.get("class_label", ""),
        "inputs": {
            "mechanic_notes": case.get("mechanic_notes", ""),
            "fault_images_description": case.get("fault_images_description", ""),
        },
        "query_pack": query_pack,
        "retrieved_chunks": results,
    }


def evaluate_all(
    cases_path: str,
    index_dir: str,
    top_k: int = 10,
    use_llm: bool = True,
    model: str = "gpt-4o-mini",
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
                "class_label": case.get("class_label", ""),
                "error": str(e),
                "retrieved_chunks": [],
            })

    output = {
        "config": {
            "top_k": top_k,
            "use_llm": use_llm,
            "model": model if use_llm else None,
            "num_cases": len(cases),
            "index_dir": index_dir,
        },
        "results": results,
    }

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(json.dumps(output, indent=2, ensure_ascii=False))
        print(f"\nSaved results to {output_path}")

    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RAG retrieval evaluation on test cases"
    )
    parser.add_argument(
        "--cases",
        default=str(PROJECT_ROOT / "evalution" / "cases.json"),
        help="Path to test cases JSON",
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
        default="gpt-4o-mini",
        help="OpenAI model for query composition (default: gpt-4o-mini)",
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
