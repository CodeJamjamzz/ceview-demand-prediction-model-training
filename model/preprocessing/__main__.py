"""Run with python -m model.preprocessing COMMAND."""

import argparse
import json
from pathlib import Path

from .analysis import approve, run_analysis
from .common import load_config, load_weekly, metadata_status
from .dataset import build


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["audit", "analyze", "approve", "build"])
    parser.add_argument("--config", default="model/preprocessing/config.json")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--reviewer", default="")
    parser.add_argument("--note", default="")
    parser.add_argument("--acknowledge-stable-leader", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    try:
        config = load_config(root / args.config)
        if args.command == "approve":
            approve(config, root, args.reviewer, args.note, args.acknowledge_stable_leader)
            print("Diagnostic review recorded.")
            return
        weekly, audit, coverage = load_weekly(config, root)
        if args.command == "audit":
            _, problems, _ = metadata_status(config, root)
            print(json.dumps({**audit, "metadata_issues": problems}, indent=2))
            print(coverage.to_string(index=False))
        elif args.command == "analyze":
            destination, report = run_analysis(weekly, audit, coverage, config, root)
            print(destination)
            print(json.dumps({k: report[k] for k in ["missing_weeks", "blocking_issues", "review_reasons"]}, indent=2))
        else:
            print(build(weekly, audit, coverage, config, root))
    except (ValueError, KeyError, FileNotFoundError) as error:
        parser.exit(2, f"Preprocessing stopped: {error}\n")


if __name__ == "__main__":
    main()
