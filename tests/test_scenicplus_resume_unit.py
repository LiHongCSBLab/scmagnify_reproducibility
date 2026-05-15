from __future__ import annotations

import importlib
import logging

import pandas as pd
import pytest

pytest.importorskip("mudata")
pytest.importorskip("pyranges")

resume_mod = importlib.import_module("run_scenicplus_resume")


def test_normalize_force_rebuild_accepts_csv_and_repeated_values() -> None:
    assert resume_mod.normalize_force_rebuild(["prepare,motif", "egrn"]) == {"prepare", "motif", "egrn"}


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
        (True, True, False, {"menr"}, "rebuild"),
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
        stage="menr",
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

    assert resume_mod.should_skip_lineage(artifacts=artifacts, resume=True, resume_mode="lineage", force_rebuild=set())
    assert not resume_mod.should_skip_lineage(artifacts=artifacts, resume=True, resume_mode="stage", force_rebuild=set())
    assert not resume_mod.should_skip_lineage(artifacts=artifacts, resume=True, resume_mode="lineage", force_rebuild={"final"})


def test_validate_region_set_dir_requires_nested_bed_files(tmp_path) -> None:
    region_dir = tmp_path / "regions" / "NaiveB"
    region_dir.mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="does not contain any nested .bed files"):
        resume_mod.validate_region_set_dir(region_dir, "NaiveB")
    nested_dir = region_dir / "set_01"
    nested_dir.mkdir()
    (nested_dir / "set_01.bed").write_text("chr1\t1\t2\n", encoding="utf-8")
    assert resume_mod.validate_region_set_dir(region_dir, "NaiveB") == region_dir


def test_validate_lineage_cells_warns_on_missing_and_nan(caplog: pytest.LogCaptureFixture) -> None:
    cell_selected = pd.DataFrame({"NaiveB": [True, None, "true"]}, index=pd.Index(["cell1", "cell2", "cell3"]))
    input_obs_names = pd.Index(["cell1", "cell3"])

    with caplog.at_level(logging.WARNING):
        overlap = resume_mod.validate_lineage_cells("NaiveB", cell_selected, input_obs_names)

    assert list(overlap) == ["cell1", "cell3"]
    assert "NA values" in caplog.text


def test_validate_lineage_cells_errors_without_overlap() -> None:
    cell_selected = pd.DataFrame({"NaiveB": [True, False]}, index=pd.Index(["cell1", "cell2"]))
    with pytest.raises(ValueError, match="overlap"):
        resume_mod.validate_lineage_cells("NaiveB", cell_selected, pd.Index(["other"]))
