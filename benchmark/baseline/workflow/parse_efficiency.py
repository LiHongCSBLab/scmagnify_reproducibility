"""
Parse method efficiency metrics from baseline logs.
"""

from __future__ import annotations

import argparse
import pathlib
import re

import pandas as pd


ELAPSED_PATTERN = re.compile(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\):\s*(.+)")
MAX_RSS_PATTERN = re.compile(r"Maximum resident set size \(kbytes\):\s*(\d+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse baseline efficiency logs")
    parser.add_argument("-p", "--home", dest="dirPjtHome", type=pathlib.Path, required=True,
                        help="Path to the project home directory")
    parser.add_argument("-v", "--version", dest="version", type=str, required=True,
                        help="Benchmark version")
    parser.add_argument("-o", "--output", dest="output", type=pathlib.Path, default=None,
                        help="Optional output CSV path")
    return parser.parse_args()


def _parse_elapsed_to_hours(value: str) -> float:
    parts = value.strip().split(":")
    if len(parts) == 2:
        minutes, seconds = parts
        total_seconds = int(minutes) * 60 + float(seconds)
    elif len(parts) == 3:
        hours, minutes, seconds = parts
        total_seconds = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    else:
        raise ValueError(f"Unsupported elapsed time format: {value}")
    return total_seconds / 3600


def _parse_log(log_path: pathlib.Path) -> dict[str, object]:
    elapsed = None
    max_rss_kb = None

    with log_path.open() as handle:
        for line in handle:
            elapsed_match = ELAPSED_PATTERN.search(line)
            if elapsed_match:
                elapsed = elapsed_match.group(1).strip()
                continue

            max_rss_match = MAX_RSS_PATTERN.search(line)
            if max_rss_match:
                max_rss_kb = int(max_rss_match.group(1))

    return {
        "method": log_path.stem,
        "log": str(log_path),
        "elapsed": elapsed,
        "time_h": _parse_elapsed_to_hours(elapsed) if elapsed is not None else None,
        "max_rss_kb": max_rss_kb,
        "memory_gb": (max_rss_kb / 1024 / 1024) if max_rss_kb is not None else None,
    }


def main(args: argparse.Namespace) -> None:
    log_dir = args.dirPjtHome / "benchmark" / args.version / "log"
    if not log_dir.exists():
        raise FileNotFoundError(f"Log directory does not exist: {log_dir}")

    output_path = args.output or (args.dirPjtHome / "benchmark" / args.version / "efficiency.csv")
    log_files = sorted(log_dir.glob("*.log"))
    if not log_files:
        raise FileNotFoundError(f"No log files found in {log_dir}")

    rows = [_parse_log(log_path) for log_path in log_files]
    df = pd.DataFrame(rows).sort_values("method").reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved efficiency summary to {output_path}")


if __name__ == "__main__":
    main(parse_args())
