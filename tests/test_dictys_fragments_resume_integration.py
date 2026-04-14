from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import anndata as ad
import mudata as mu
import numpy as np
import pandas as pd
import pytest

resume_mod = pytest.importorskip("run_dictys_fragments_resume")


def _mini_mdata() -> mu.MuData:
    obs = pd.DataFrame(index=pd.Index(["cell1", "cell2"], name="cell"))
    rna = ad.AnnData(
        X=np.array([[1.0, 2.0], [3.0, 4.0]]),
        obs=obs.copy(),
        var=pd.DataFrame(index=pd.Index(["GeneA", "GeneB"], name="gene")),
    )
    atac = ad.AnnData(
        X=np.array([[1.0, 0.0], [0.0, 1.0]]),
        obs=obs.copy(),
        var=pd.DataFrame(index=pd.Index(["chr1-100-220", "chr1-300-450"], name="peak")),
    )
    out = mu.MuData({"rna": rna, "atac": atac})
    out.obs["celltype"] = ["NaiveB", "NaiveB"]
    return out


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _make_args(home: Path) -> object:
    return type(
        "Args",
        (),
        {
            "dataset": "mini_dataset",
            "dirPjtHome": home,
            "celllist": home / "benchmark" / "cell_list" / "cell_state_masks.csv",
            "genelist": home / "benchmark" / "gene_list" / "genes.csv",
            "version": "test_run",
            "save": True,
            "seed": 0,
            "refGenome": "hg38",
            "fragments": None,
            "fragment_samples": None,
            "barcode_map": None,
            "barcode_transform": "auto",
            "sample_separator": "#",
            "threads": 1,
            "device": "cpu",
            "distance": 50000,
            "n_p2g_links": 5,
            "thr_score": 0.0,
            "motifs": None,
            "homer_genome": None,
            "gene_bed": None,
            "resume": True,
            "resume_mode": "stage",
            "force_rebuild": None,
        },
    )()


def test_resume_main_skips_completed_stages(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    home = tmp_path / "fixture"
    _write_text(home / "benchmark" / "cell_list" / "cell_state_masks.csv", ",NaiveB\ncell1,1\ncell2,1\n")
    _write_text(home / "benchmark" / "gene_list" / "genes.csv", "GeneA\nGeneB\n")
    _write_text(home / "benchmark" / "data" / "mini_dataset.h5mu", "placeholder")

    calls = {"prepare": 0, "p2g": 0, "tfb": 0, "mdl_inputs": 0, "mdl": 0}

    monkeypatch.setattr(resume_mod, "_resolve_resources", lambda args: type("R", (), {"gene_bed": Path("gene.bed")})())
    monkeypatch.setattr(resume_mod, "_resolve_fragment_specs", lambda args: [])
    monkeypatch.setattr(resume_mod, "_load_barcode_map", lambda path: {})
    monkeypatch.setattr(resume_mod, "_resolve_input_data", lambda home_dir, dataset: (home_dir / "benchmark" / "data" / f"{dataset}.h5mu", "h5mu"))
    monkeypatch.setattr(resume_mod, "_load_lineage_mdata", lambda *args, **kwargs: _mini_mdata())
    monkeypatch.setattr(resume_mod, "log_memory_usage", lambda: None)

    def fake_prepare(lineage_mdata: mu.MuData, work_dir: Path) -> mu.MuData:
        calls["prepare"] += 1
        return lineage_mdata

    def fake_build_obs_lookup(*args, **kwargs) -> dict[tuple[str | None, str], str]:
        return {}

    def fake_p2g(prepared_mdata_path: Path, output_path: Path, temp_dir: Path, gene_bed: Path, distance: int) -> pd.DataFrame:
        calls["p2g"] += 1
        df = pd.DataFrame({"cre": ["chr1-100-220"], "gene": ["GeneA"], "score": [1.0]})
        df.to_csv(output_path, index=False)
        return df

    def fake_tfb(**kwargs) -> tuple[pd.DataFrame, pd.DataFrame]:
        calls["tfb"] += 1
        lineage_dir = kwargs["lineage_dir"]
        tfb = pd.DataFrame({"cre": ["chr1-100-220"], "tf": ["GeneA"], "score": [0.5]})
        qc = pd.DataFrame(
            {"fragment_path": ["dummy"], "sample": ["S1"], "total_fragment_rows": [1], "kept_fragment_counts": [1], "matched_cells": [1]}
        )
        tfb.to_csv(lineage_dir / "tfb.csv", index=False)
        qc.to_csv(lineage_dir / "barcode_match_qc.csv", index=False)
        return tfb, qc

    def fake_mdl_inputs(prepared_mdata_path: Path, p2g_df: pd.DataFrame, tfb_df: pd.DataFrame, lineage_dir: Path) -> tuple[Path, Path, Path]:
        calls["mdl_inputs"] += 1
        expr_path = lineage_dir / "mdl_expr.tsv.gz"
        peaks_path = lineage_dir / "mdl_peaks.tsv.gz"
        tfb_path = lineage_dir / "mdl_tfb.tsv.gz"
        expr_path.write_text("expr\n", encoding="utf-8")
        peaks_path.write_text("peaks\n", encoding="utf-8")
        tfb_df.rename(columns={"tf": "TF", "cre": "loc"}).to_csv(tfb_path, sep="\t", index=False)
        return expr_path, peaks_path, tfb_path

    def fake_run_mdl_with_resume(**kwargs) -> pd.DataFrame:
        calls["mdl"] += 1
        artifacts = kwargs["artifacts"]
        for path in resume_mod.stage_outputs(artifacts, resume_mod.STAGE_MDL):
            path.write_text("stage\n", encoding="utf-8")
        pd.DataFrame([[1]], index=["GeneA"], columns=["GeneB"]).to_csv(artifacts.net_nweight, sep="\t")
        pd.DataFrame([[1]], index=["GeneA"], columns=["GeneB"]).to_csv(artifacts.binlinking, sep="\t")
        return pd.DataFrame({"TF": ["GeneA"], "Target": ["GeneB"], "Score": [1.0]})

    monkeypatch.setattr(resume_mod, "_prepare_preprocessed_mdata", fake_prepare)
    monkeypatch.setattr(resume_mod, "_build_obs_lookup", fake_build_obs_lookup)
    monkeypatch.setattr(resume_mod, "_build_p2g", fake_p2g)
    monkeypatch.setattr(resume_mod, "_build_tfb", fake_tfb)
    monkeypatch.setattr(resume_mod, "_prepare_mdl_inputs", fake_mdl_inputs)
    monkeypatch.setattr(resume_mod, "run_mdl_with_resume", fake_run_mdl_with_resume)

    args = _make_args(home)
    resume_mod.main(args)
    assert calls == {"prepare": 1, "p2g": 1, "tfb": 1, "mdl_inputs": 1, "mdl": 1}

    resume_mod.main(args)
    assert calls == {"prepare": 1, "p2g": 1, "tfb": 1, "mdl_inputs": 1, "mdl": 1}

    p2g_path = home / "tmp" / "dictys_fragments_wd" / "test_run" / "NaiveB" / "p2g.csv"
    p2g_path.unlink()
    resume_mod.main(args)
    assert calls == {"prepare": 1, "p2g": 2, "tfb": 2, "mdl_inputs": 2, "mdl": 2}


@pytest.mark.integration
def test_real_fixture_can_run_twice_with_resume(dictys_fixture_root: Path, workflow_script_dir: Path, tmp_path: Path) -> None:
    if os.environ.get("DICTYS_RUN_REAL_INTEGRATION") != "1":
        pytest.skip("Set DICTYS_RUN_REAL_INTEGRATION=1 to run the real Dictys integration test")

    home = dictys_fixture_root
    dataset = "t-cell-depleted-bm_naiveb_100"
    fragments_manifest = home / "benchmark" / "data" / f"{dataset}.fragments.csv"
    if not fragments_manifest.exists():
        pytest.skip(f"Missing generated fixture: {fragments_manifest}")

    version = f"pytest_{os.getpid()}"
    cmd = [
        sys.executable,
        str(workflow_script_dir / "run_dictys_fragments_resume.py"),
        "--home",
        str(home),
        "--dataset",
        dataset,
        "--cell",
        str(home / "benchmark" / "cell_list" / "cell_state_masks.csv"),
        "--gene",
        str(home / "benchmark" / "gene_list" / "test_assoc_fdr1e-3_A0.3.csv"),
        "--version",
        version,
        "--tmp-save",
        "true",
        "--seed",
        "0",
        "--ref-genome",
        "hg38",
        "--threads",
        "2",
        "--device",
        "cpu",
        "--distance",
        "50000",
        "--n-p2g-links",
        "5",
    ]
    env = os.environ.copy()

    first = subprocess.run(cmd, cwd=str(workflow_script_dir), capture_output=True, text=True, check=False, env=env)
    assert first.returncode == 0, first.stderr

    second = subprocess.run(cmd, cwd=str(workflow_script_dir), capture_output=True, text=True, check=False, env=env)
    assert second.returncode == 0, second.stderr

    final_csv = home / "benchmark" / version / "net" / "Dictys_NaiveB.csv"
    assert final_csv.exists()
