#!/usr/bin/env python
"""CLI entrypoint for baseline workflow preflight checks."""

from __future__ import annotations

import argparse
import pathlib
import sys

from preflight_checks import render_report, run_preflight, write_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run preflight checks for the baseline workflow")
    parser.add_argument("--config", dest="config", type=pathlib.Path, required=True, help="Path to the baseline YAML config")
    parser.add_argument(
        "--json-out",
        dest="json_out",
        type=pathlib.Path,
        default=None,
        help="Optional path to write structured JSON results",
    )
    parser.add_argument(
        "--report-out",
        dest="report_out",
        type=pathlib.Path,
        default=None,
        help="Optional path to write a human-readable markdown report",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root_dir = pathlib.Path(__file__).resolve().parents[1]
    result = run_preflight(config_path=args.config.resolve(), root_dir=root_dir)
    write_outputs(result, json_out=args.json_out, report_out=args.report_out)
    print(render_report(result))
    return 0 if result.ok else 1


if __name__ == "__main__":
    sys.exit(main())
