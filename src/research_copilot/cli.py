"""Command line entrypoint."""

from __future__ import annotations

import argparse
import json

from research_copilot.service import ResearchCopilotService


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Agentic Research Copilot")
    subcommands = parser.add_subparsers(dest="command", required=True)

    subcommands.add_parser("ingest-sample", help="Ingest committed RAGBench techqa fixture")

    query = subcommands.add_parser("query", help="Ask a research question")
    query.add_argument("question")
    query.add_argument(
        "--mode",
        default="hybrid_rerank",
        choices=["lexical", "vector", "hybrid", "hybrid_rerank"],
    )

    evaluate = subcommands.add_parser("evaluate", help="Run retrieval evaluation")
    evaluate.add_argument(
        "--modes",
        nargs="*",
        default=["lexical", "vector", "hybrid", "hybrid_rerank"],
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    service = ResearchCopilotService.create_demo()

    if args.command == "ingest-sample":
        print(json.dumps(service.ingest_ragbench(sample_only=True), indent=2))
    elif args.command == "query":
        service.ingest_sample()
        response = service.query(question=args.question, retrieval_mode=args.mode)
        print(response.model_dump_json(indent=2))
    elif args.command == "evaluate":
        service.ingest_sample()
        report = service.run_evaluation(modes=args.modes)
        print(report.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
