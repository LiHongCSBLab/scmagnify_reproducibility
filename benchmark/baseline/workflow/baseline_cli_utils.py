"""Shared helpers for benchmark baseline workflow scripts."""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
from pathlib import Path
from typing import Optional

import psutil


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def log_memory_usage() -> None:
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    logging.info("Memory usage: %.2f MB", memory_info.rss / 1024 ** 2)


def run_logged_command(cmd: list[str], *, cwd: Optional[Path] = None) -> None:
    logging.info("Running command: %s", " ".join(map(str, cmd)))
    result = subprocess.run(cmd, cwd=str(cwd) if cwd else None, text=True, capture_output=True)
    if result.stdout:
        logging.info(result.stdout)
    if result.stderr:
        logging.info(result.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {' '.join(map(str, cmd))}")
