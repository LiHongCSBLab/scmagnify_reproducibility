from __future__ import annotations

import importlib
import os
import sys
import types

import pytest

pytest.importorskip("mudata")

resume_mod = importlib.import_module("run_dictys_fragments_resume")


def test_normalize_force_rebuild_accepts_csv_and_repeated_values() -> None:
    assert resume_mod.normalize_force_rebuild(["prepare,p2g", "mdl"]) == {"prepare", "p2g", "mdl"}


def test_normalize_force_rebuild_all_expands_every_stage() -> None:
    assert resume_mod.normalize_force_rebuild(["all"]) == set(resume_mod.ALL_STAGES)


def test_normalize_force_rebuild_rejects_unknown_stage() -> None:
    with pytest.raises(ValueError, match="Unsupported stage"):
        resume_mod.normalize_force_rebuild(["bad-stage"])


@pytest.mark.parametrize(
    ("outputs_exist", "resume", "upstream_changed", "force_rebuild", "expected"),
    [
        (True, True, False, set(), "skip"),
        (False, True, False, set(), "run"),
        (True, False, False, set(), "rebuild"),
        (True, True, True, set(), "rebuild"),
        (False, True, True, set(), "run"),
        (True, True, False, {"p2g"}, "rebuild"),
    ],
)
def test_decide_stage_action(
    outputs_exist: bool,
    resume: bool,
    upstream_changed: bool,
    force_rebuild: set[str],
    expected: str,
) -> None:
    action = resume_mod.decide_stage_action(
        stage="p2g",
        outputs_exist=outputs_exist,
        resume=resume,
        upstream_changed=upstream_changed,
        force_rebuild=force_rebuild,
    )
    assert action == expected


def test_should_skip_lineage_only_in_lineage_mode(tmp_path) -> None:
    artifacts = resume_mod.build_lineage_artifacts(tmp_path / "tmp", tmp_path / "net", "NaiveB")
    artifacts.net_dir.mkdir(parents=True)
    artifacts.final_csv.write_text("TF,Target,Score\n", encoding="utf-8")

    assert resume_mod.should_skip_lineage(
        artifacts=artifacts,
        resume=True,
        resume_mode="lineage",
        force_rebuild=set(),
    )
    assert not resume_mod.should_skip_lineage(
        artifacts=artifacts,
        resume=True,
        resume_mode="stage",
        force_rebuild=set(),
    )
    assert not resume_mod.should_skip_lineage(
        artifacts=artifacts,
        resume=True,
        resume_mode="lineage",
        force_rebuild={"final"},
    )


def test_ensure_wellington_on_path_discovers_dictys_bin(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    fake_env = tmp_path / "fake_env"
    site_packages = fake_env / "lib" / "python3.9" / "site-packages" / "dictys"
    bin_dir = fake_env / "bin"
    site_packages.mkdir(parents=True)
    bin_dir.mkdir(parents=True)
    (site_packages / "__init__.py").write_text("", encoding="utf-8")
    wellington = bin_dir / "wellington_footprints.py"
    wellington.write_text("#!/usr/bin/env python\n", encoding="utf-8")

    fake_dictys = types.SimpleNamespace(__file__=str(site_packages / "__init__.py"))
    monkeypatch.setitem(sys.modules, "dictys", fake_dictys)
    monkeypatch.setattr(resume_mod.shutil, "which", lambda name: None)
    monkeypatch.setenv("PATH", "/usr/bin")

    discovered = resume_mod.ensure_wellington_on_path()

    assert discovered == bin_dir
    assert os.environ["PATH"].split(os.pathsep)[0] == str(bin_dir)
