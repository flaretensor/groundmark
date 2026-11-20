import argparse
import json
import sys
from pathlib import Path

from pydantic import ValidationError

from groundmark.analytics import report
from groundmark.config import Settings
from groundmark.models import ResearchResult
from groundmark.research import Options, ResearchFailure, ResearchRunner
from groundmark.sources import private_json

FAILURES = {
    "configuration": "Research could not start: set NEBIUS_API_KEY and TAVILY_API_KEY in .env.",
    "invalid_question": "Research could not start: the question must be 1 to 500 characters.",
}


def _ask_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("question")
    parser.add_argument("--max-rounds", type=int)
    parser.add_argument("--max-credits", type=int)
    parser.add_argument("--max-cost", type=float)
    parser.add_argument("--use-cache", action="store_true", default=None)
    parser.add_argument("--trace-out", type=Path)
    parser.add_argument("--json", action="store_true")


def _write_text(result: ResearchResult) -> None:
    if result.summary:
        print(result.summary)
    if result.trace.stop_reason == "unanswerable":
        print("This can't be known yet as of today's date.")
    elif result.insufficient_evidence:
        print("The retrieved sources don't fully answer this question.")
    for item in result.missing_information:
        print(f"- {item}")
    for conflict in result.conflicts:
        references = " ".join(f"[{source_id}]" for source_id in conflict.source_ids)
        print(f"Sources disagree: {conflict.description} {references}".rstrip())
    sources = {source.source_id: source.url for source in result.sources}
    for row in result.claims:
        print(f"{row.claim.claim_id} {row.status}  {row.claim.text}")
        if row.reason:
            print(f"  {row.reason}")
        for pair in row.pairs:
            print(
                f"  [{pair.source_id}] {pair.verdict}, {pair.quote_match}  {sources[pair.source_id]}"
            )
            if pair.evidence_quote:
                print(f'    "{pair.evidence_quote}"')
    trace = result.trace
    print(
        f"Run {trace.run_id}  {trace.stop_reason} · {len(trace.rounds)} round(s) · "
        f"{len(result.sources)} sources · {trace.total_latency_ms / 1000:.2f}s · "
        f"{trace.tavily_credits} credits · USD {trace.estimated_usd:.6f} estimated"
    )
    for error in trace.errors:
        print(f"  {error.stage}: {error.kind}")


def _ask(args, runner=None) -> int:
    try:
        options = Options(
            max_rounds=args.max_rounds,
            max_credits=args.max_credits,
            max_cost=args.max_cost,
            use_cache=args.use_cache,
        )
        result = (runner or ResearchRunner()).run(args.question, options)
    except ValidationError:
        print("Invalid question or options.", file=sys.stderr)
        return 2
    except ResearchFailure as error:
        print(FAILURES.get(str(error), f"Research failed ({error})."), file=sys.stderr)
        return 2
    if args.trace_out:
        private_json(args.trace_out, result.trace.model_dump(mode="json"))
    if args.json:
        print(result.model_dump_json(indent=2))
    else:
        _write_text(result)
    return 0


def organizer_main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Groundmark live-web research")
    _ask_arguments(parser)
    return _ask(parser.parse_args(argv))


def main(argv=None, runner=None) -> int:
    parser = argparse.ArgumentParser(description="Groundmark live-web research")
    subcommands = parser.add_subparsers(dest="command", required=True)
    _ask_arguments(subcommands.add_parser("ask"))
    subcommands.add_parser("report")
    arguments = parser.parse_args(argv)
    if arguments.command == "report":
        print(json.dumps(report(Settings().artifacts_dir / "runs"), indent=2))
        return 0
    return _ask(arguments, runner)


if __name__ == "__main__":
    raise SystemExit(main())
