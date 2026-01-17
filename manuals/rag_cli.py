import argparse
import json
import os

from rag_core import (
    build_index,
    evaluate_retrieval,
    generate_grounded_report,
    setup_logging,
    Retriever,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Text RAG baseline for turbine manuals."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="Build a FAISS index.")
    build_parser.add_argument(
        "--manuals",
        nargs="+",
        required=False,
        help="Paths to manuals (PDF or text).",
    )
    build_parser.add_argument(
        "--manuals-dir",
        default=None,
        help="Directory containing manuals to index.",
    )
    build_parser.add_argument(
        "--index-dir",
        required=True,
        help="Output directory for FAISS index and metadata.",
    )
    build_parser.add_argument(
        "--model",
        default="all-MiniLM-L6-v2",
        help="Sentence-transformers model name.",
    )
    build_parser.add_argument(
        "--chunk-size",
        type=int,
        default=450,
        help="Chunk size in whitespace tokens.",
    )
    build_parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=70,
        help="Chunk overlap in whitespace tokens.",
    )
    build_parser.add_argument(
        "--section-chunks",
        action="store_true",
        help="Add extra chunks for full sections between headings.",
    )
    build_parser.add_argument(
        "--max-section-tokens",
        type=int,
        default=800,
        help="Max tokens for a section chunk when --section-chunks is set.",
    )
    build_parser.add_argument(
        "--section-manuals",
        nargs="*",
        default=["*"],
        help="Manual base filenames to use outline/heading sectioning; use \"*\" for all.",
    )
    build_parser.add_argument(
        "--skip-first-pages",
        type=int,
        default=2,
        help="Number of leading pages to skip for manuals in --skip-page-manuals.",
    )
    build_parser.add_argument(
        "--skip-page-manuals",
        nargs="*",
        default=["manual-02.pdf"],
        help="Manual base filenames to skip leading pages for.",
    )
    build_parser.add_argument(
        "--min-page-tokens",
        type=int,
        default=20,
        help="Minimum tokens per page to keep (filters image/table pages).",
    )
    build_parser.add_argument(
        "--min-alpha-ratio",
        type=float,
        default=0.6,
        help="Minimum letter/alpha ratio for section content.",
    )
    build_parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for embeddings.",
    )
    build_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic runs.",
    )

    query_parser = subparsers.add_parser("query", help="Query the index.")
    query_parser.add_argument(
        "--index-dir",
        required=True,
        help="Directory containing index.faiss and metadata.",
    )
    query_parser.add_argument("--query", required=True, help="Query text.")
    query_parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve.",
    )
    query_parser.add_argument(
        "--model",
        default=None,
        help="Embedding model name (defaults to config.json).",
    )
    query_parser.add_argument(
        "--alpha",
        type=float,
        default=0.7,
        help="Hybrid weight for embeddings vs. BM25 (0-1).",
    )
    query_parser.add_argument(
        "--allowed-kinds",
        nargs="*",
        default=None,
        help="Only return chunks with these kinds (e.g. procedure checklist safety).",
    )
    query_parser.add_argument(
        "--kind-boost",
        nargs="*",
        default=None,
        help="Boost kinds like procedure=1.2 checklist=1.1 safety=1.1",
    )
    query_parser.add_argument(
        "--generate",
        action="store_true",
        help="Generate grounded JSON report if OPENAI_API_KEY is set.",
    )
    query_parser.add_argument(
        "--openai-model",
        default="gpt-5.2",
        help="OpenAI model name for report generation.",
    )
    query_parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Timeout in seconds for OpenAI call.",
    )

    eval_parser = subparsers.add_parser("eval", help="Evaluate retrieval.")
    eval_parser.add_argument(
        "--index-dir",
        required=True,
        help="Directory containing index.faiss and metadata.",
    )
    eval_parser.add_argument(
        "--gold",
        required=True,
        help="Gold JSONL file with relevant chunk ids.",
    )
    eval_parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve for evaluation.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging()

    if args.command == "build":
        manual_paths = []
        if args.manuals:
            manual_paths.extend(args.manuals)
        if args.manuals_dir:
            if not os.path.isdir(args.manuals_dir):
                raise ValueError(f"Manuals directory not found: {args.manuals_dir}")
            entries = sorted(os.listdir(args.manuals_dir))
            for name in entries:
                ext = os.path.splitext(name)[1].lower()
                if ext in {".pdf", ".txt"}:
                    manual_paths.append(os.path.join(args.manuals_dir, name))
        if not manual_paths:
            raise ValueError("No manuals provided. Use --manuals or --manuals-dir.")
        build_index(
            manual_paths=manual_paths,
            index_dir=args.index_dir,
            model_name=args.model,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            batch_size=args.batch_size,
            seed=args.seed,
            add_section_chunks=args.section_chunks,
            max_section_tokens=args.max_section_tokens,
            section_manuals=args.section_manuals,
            skip_first_pages=args.skip_first_pages,
            skip_page_manuals=args.skip_page_manuals,
            min_page_tokens=args.min_page_tokens,
            min_alpha_ratio=args.min_alpha_ratio,
        )
        return

    if args.command == "query":
        kind_boost = {}
        if args.kind_boost:
            for item in args.kind_boost:
                if "=" not in item:
                    continue
                key, value = item.split("=", 1)
                try:
                    kind_boost[key] = float(value)
                except ValueError:
                    continue
        retriever = Retriever(
            args.index_dir,
            model_name=args.model,
            alpha=args.alpha,
            kind_boost=kind_boost,
            allowed_kinds=args.allowed_kinds,
        )
        results = retriever.search(args.query, top_k=args.top_k)
        output = {"query": args.query, "results": results}

        if args.generate:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                report = generate_grounded_report(
                    query_text=args.query,
                    retrieved_chunks=results,
                    model_name=args.openai_model,
                    api_key=api_key,
                    timeout=args.timeout,
                )
                output["report"] = report
            else:
                output["report"] = None
                output["warning"] = "OPENAI_API_KEY not set; skipping generation."

        print(json.dumps(output, ensure_ascii=True, indent=2))
        return

    if args.command == "eval":
        metrics = evaluate_retrieval(
            index_dir=args.index_dir,
            gold_path=args.gold,
            top_k=args.top_k,
        )
        print(json.dumps(metrics, ensure_ascii=True, indent=2))
        return

    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
