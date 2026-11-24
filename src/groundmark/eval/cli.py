import argparse
import json

from groundmark.eval.datasets import DATASETS
from groundmark.eval.splits import freeze

STAGES = ("smoke", "dev", "test")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="groundmark-eval")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("fetch")
    commands.add_parser("split")
    checker = commands.add_parser("checker")
    checker.add_argument("--mlflow-uri", default="")
    checker.add_argument("--model", action="append")
    run = commands.add_parser("run")
    run.add_argument("--stage", choices=STAGES, required=True)
    run.add_argument("--live", action="store_true")
    run.add_argument("--confirm-test", action="store_true")
    judge = commands.add_parser("judge")
    judge.add_argument("--stage", choices=STAGES, required=True)
    judge.add_argument("--confirm-test", action="store_true")
    validate = commands.add_parser("judge-validate")
    validate.add_argument("--dataset", choices=("evouna",), required=True)
    validate.add_argument("--stage", choices=("smoke", "dev"), default="dev")
    validate.add_argument("--limit", type=int)
    report = commands.add_parser("report")
    report.add_argument("--stage", choices=STAGES, required=True)
    report.add_argument("--mlflow-uri")
    args = parser.parse_args(argv)

    if args.command == "fetch":
        for dataset in DATASETS:
            dataset.fetch()
    elif args.command == "split":
        freeze()
    elif args.command == "checker":
        from groundmark.eval.checkers import run_candidates

        failures = run_candidates(args.mlflow_uri, args.model)
        for failure in failures:
            print(json.dumps(failure))
        if failures:
            raise SystemExit(1)
    else:
        if args.command == "run" and not args.live:
            parser.error("run requires --live")
        if args.command in ("run", "judge") and args.stage == "test" and not args.confirm_test:
            parser.error("TEST requires --confirm-test")
        try:
            _evaluate(args)
        except (ValueError, OSError) as error:
            parser.exit(1, f"{error}\n")


def _evaluate(args) -> None:
    from groundmark.config import Settings
    from groundmark.eval.runner import ROOT, live_runner, run_stage, stage_ledger, stage_questions
    from groundmark.research import ResearchFailure

    if args.command == "run":
        questions = stage_questions(args.stage)
        ledger = stage_ledger(ROOT, args.stage)
        try:
            runner = live_runner(Settings(), ledger)
        except ResearchFailure as error:
            raise ValueError(str(error)) from None
        print(
            json.dumps(
                {
                    "stage": args.stage,
                    "completed": run_stage(
                        args.stage,
                        questions,
                        runner,
                        ledger,
                        confirm_test=args.confirm_test,
                    ),
                    "eligible": len(questions),
                    "budget_blocked": ledger.scope_blocked,
                }
            )
        )
    elif args.command == "judge":
        from groundmark.eval.judge import grade_live

        print(json.dumps({"graded_arms": grade_live(args.stage, Settings())}))
    elif args.command == "judge-validate":
        from groundmark.eval.judge import live_judge, validate_evouna

        if args.limit is not None and args.limit < 1:
            raise ValueError("--limit must be positive")
        with live_judge(Settings(), stage_ledger(ROOT, args.stage)) as judge:
            print(json.dumps(validate_evouna(judge, limit=args.limit)))
    elif args.command == "report":
        from groundmark.eval.report import report

        uri = args.mlflow_uri if args.mlflow_uri is not None else Settings().mlflow_tracking_uri
        print(json.dumps(report(args.stage, mlflow_uri=uri), indent=2))


if __name__ == "__main__":
    main()
