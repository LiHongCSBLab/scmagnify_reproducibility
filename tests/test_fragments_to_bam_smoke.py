from __future__ import annotations

import importlib
import shutil

import pytest

pytest.importorskip("mudata")

smoke_mod = importlib.import_module("test_fragments_to_bam")


@pytest.mark.requires_samtools
def test_fragments_to_bam_smoke() -> None:
    if shutil.which("samtools") is None:
        pytest.skip("samtools is not available on PATH")
    smoke_mod.run_test()
