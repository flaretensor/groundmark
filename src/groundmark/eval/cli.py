import argparse

from groundmark.eval.datasets import DATASETS
from groundmark.eval.splits import freeze


def main() -> None:
    parser = argparse.ArgumentParser(prog="groundmark-eval")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("fetch")
    commands.add_parser("split")
    args = parser.parse_args()
    if args.command == "fetch":
        for dataset in DATASETS:
            dataset.fetch()
    elif args.command == "split":
        freeze()


if __name__ == "__main__":
    main()
