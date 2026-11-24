import argparse

from groundmark.eval.datasets import DATASETS
from groundmark.eval.splits import freeze


def main() -> None:
    parser = argparse.ArgumentParser(prog="groundmark-eval")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("fetch")
    commands.add_parser("split")
    checker = commands.add_parser("checker")
    checker.add_argument("--mlflow-uri", default="")
    checker.add_argument("--model", action="append")
    args = parser.parse_args()
    if args.command == "fetch":
        for dataset in DATASETS:
            dataset.fetch()
    elif args.command == "split":
        freeze()
    elif args.command == "checker":
        from groundmark.eval.checkers import run_candidates

        failures = run_candidates(args.mlflow_uri, args.model)
        for failure in failures:
            print(failure)
        if failures:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
