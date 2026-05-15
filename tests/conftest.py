from __future__ import annotations

import sys
from pathlib import Path

import pytest

WORKFLOW_DIR = Path(__file__).resolve().parents[1] / "benchmark" / "baseline" / "workflow"
if str(WORKFLOW_DIR) not in sys.path:
    sys.path.insert(0, str(WORKFLOW_DIR))

DATA_DIR = Path(__file__).resolve().parent / "data"
if str(DATA_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_DIR))


@pytest.fixture
def workflow_script_dir() -> Path:
    return WORKFLOW_DIR


@pytest.fixture
def dictys_fixture_root() -> Path:
    return Path(__file__).resolve().parent / "data" / "dictys_naiveb_100"


@pytest.fixture
def scenicplus_fixture_root() -> Path:
    return Path(__file__).resolve().parent / "data" / "dictys_naiveb_100"
