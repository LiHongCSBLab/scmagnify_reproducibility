from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import anndata as ad
import mudata as mu
import numpy as np
import pandas as pd
import pytest

resume_mod = pytest.importorskip("run_scenicplus_resume")


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
    return mu.MuData({"rna": rna, "atac": atac})


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
            "regionSetRoot": home / "benchmark" / "data" / "scenicplus_region_sets" / "mini_dataset",
            "dbRoot": None,
            "threads": 1,
            "ext": 1000,
            "region_set_mode": "manual",
            "region_fdr": 0.05,
            "region_min_logfc": 0.0,
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
    _write_text(
        home / "benchmark" / "data" / "scenicplus_region_sets" / "mini_dataset" / "NaiveB" / "set_01" / "set_01.bed",
        "chr1\t100\t220\n",
    )

    calls: list[str] = []

    monkeypatch.setattr(
        resume_mod,
        "_resolve_reference_resources",
        lambda ref_genome, db_root, home_dir: type(
            "R",
            (),
            {
                "annotation_tsv": Path("annotation.tsv"),
                "chromsizes_tsv": Path("chromsizes.tsv"),
                "rankings_feather": Path("rankings.feather"),
                "motif_annotation_tsv": Path("motif.tbl"),
                "species": "homo_sapiens",
            },
        )(),
    )
    monkeypatch.setattr(resume_mod, "_resolve_input_data", lambda home_dir, dataset: (home_dir / "benchmark" / "data" / f"{dataset}.h5mu", "h5mu"))
    monkeypatch.setattr(resume_mod, "load_input_obs_names", lambda input_path, input_type: pd.Index(["cell1", "cell2"]))
    monkeypatch.setattr(resume_mod, "_resolve_region_set_root", lambda home_dir, dataset, region_set_root: region_set_root)
    monkeypatch.setattr(resume_mod, "_discover_search_space_subcommand", lambda: "search_space")
    monkeypatch.setattr(resume_mod, "_load_lineage_mdata", lambda *args, **kwargs: _mini_mdata())
    monkeypatch.setattr(resume_mod, "log_memory_usage", lambda: None)

    def fake_run_logged_command(cmd: list[str], *, cwd=None) -> None:
        calls.append(" ".join(cmd[:3]))
        if cmd[:3] == ["scenicplus", "prepare_data", "search_space"]:
            Path(cmd[-1]).write_text("space\n", encoding="utf-8")
        elif cmd[:3] == ["scenicplus", "grn_inference", "region_to_gene"]:
            pd.DataFrame({"region": ["chr1:100-220"], "target": ["GeneA"], "importance_x_rho": [0.5]}).to_csv(
                cmd[cmd.index("--out_region_to_gene_adjacencies") + 1], sep="\t", index=False
            )
        elif cmd[:3] == ["scenicplus", "grn_inference", "motif_enrichment_cistarget"]:
            Path(cmd[cmd.index("--output_fname_cistarget_result") + 1]).write_text("cistarget\n", encoding="utf-8")
        elif cmd[:3] == ["scenicplus", "prepare_data", "prepare_menr"]:
            Path(cmd[cmd.index("--out_file_tf_names") + 1]).write_text("GeneA\n", encoding="utf-8")
            ad.AnnData(
                X=np.array([[1.0]]),
                obs=pd.DataFrame(index=pd.Index(["chr1:100-220"])),
                var=pd.DataFrame(index=pd.Index(["GeneA"])),
            ).write_h5ad(cmd[cmd.index("--out_file_direct_annotation") + 1])
            ad.AnnData(
                X=np.array([[1.0]]),
                obs=pd.DataFrame(index=pd.Index(["chr1:100-220"])),
                var=pd.DataFrame(index=pd.Index(["GeneA"])),
            ).write_h5ad(cmd[cmd.index("--out_file_extended_annotation") + 1])
        elif cmd[:3] == ["scenicplus", "grn_inference", "TF_to_gene"]:
            pd.DataFrame(
                {
                    "TF": ["GeneA"],
                    "target": ["GeneA"],
                    "importance": [0.75],
                    "regulation": [1],
                    "rho": [0.3],
                }
            ).to_csv(cmd[cmd.index("--out_tf_to_gene_adjacencies") + 1], sep="\t", index=False)

    monkeypatch.setattr(resume_mod, "run_logged_command", fake_run_logged_command)
    monkeypatch.setattr(
        resume_mod,
        "_derive_tfb_from_direct_h5ad",
        lambda prepared_mdata_path, p2g_df, direct_h5ad_path: pd.DataFrame({"cre": ["chr1-100-220"], "tf": ["GeneA"], "score": [5.0]}),
    )

    args = _make_args(home)
    resume_mod.main(args)
    assert len(calls) == 5

    resume_mod.main(args)
    assert len(calls) == 5

    tg_adj = home / "tmp" / "scenicplus_wd" / "test_run" / "NaiveB" / "tg_adj.tsv"
    tg_adj.unlink()
    resume_mod.main(args)
    assert len(calls) == 6


def test_resume_main_can_generate_lineage_vs_rest_region_set(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    home = tmp_path / "fixture"
    _write_text(home / "benchmark" / "cell_list" / "cell_state_masks.csv", ",NaiveB\ncell1,1\ncell2,1\n")
    _write_text(home / "benchmark" / "gene_list" / "genes.csv", "GeneA\nGeneB\n")
    _write_text(home / "benchmark" / "data" / "mini_dataset.h5mu", "placeholder")
    (home / "benchmark" / "data" / "scenicplus_region_sets" / "mini_dataset").mkdir(parents=True, exist_ok=True)

    calls: list[str] = []
    generated: list[str] = []

    monkeypatch.setattr(
        resume_mod,
        "_resolve_reference_resources",
        lambda ref_genome, db_root, home_dir: type(
            "R",
            (),
            {
                "annotation_tsv": Path("annotation.tsv"),
                "chromsizes_tsv": Path("chromsizes.tsv"),
                "rankings_feather": Path("rankings.feather"),
                "motif_annotation_tsv": Path("motif.tbl"),
                "species": "homo_sapiens",
            },
        )(),
    )
    monkeypatch.setattr(resume_mod, "_resolve_input_data", lambda home_dir, dataset: (home_dir / "benchmark" / "data" / f"{dataset}.h5mu", "h5mu"))
    monkeypatch.setattr(resume_mod, "load_input_obs_names", lambda input_path, input_type: pd.Index(["cell1", "cell2"]))
    monkeypatch.setattr(resume_mod, "_resolve_region_set_root", lambda home_dir, dataset, region_set_root: region_set_root)
    monkeypatch.setattr(resume_mod, "_discover_search_space_subcommand", lambda: "search_space")
    monkeypatch.setattr(resume_mod, "_load_lineage_mdata", lambda *args, **kwargs: _mini_mdata())
    monkeypatch.setattr(resume_mod, "log_memory_usage", lambda: None)

    def fake_region_set_builder(**kwargs):
        generated.append(kwargs["lineage"])
        bed_path = kwargs["region_set_root"] / kwargs["lineage"] / "lineage_vs_rest" / "lineage_vs_rest.bed"
        _write_text(bed_path, "chr1\t100\t220\n")
        return kwargs["region_set_root"] / kwargs["lineage"]

    monkeypatch.setattr(resume_mod, "build_lineage_vs_rest_region_set", fake_region_set_builder)

    def fake_run_logged_command(cmd: list[str], *, cwd=None) -> None:
        calls.append(" ".join(cmd[:3]))
        if cmd[:3] == ["scenicplus", "prepare_data", "search_space"]:
            Path(cmd[-1]).write_text("space\n", encoding="utf-8")
        elif cmd[:3] == ["scenicplus", "grn_inference", "region_to_gene"]:
            pd.DataFrame({"region": ["chr1:100-220"], "target": ["GeneA"], "importance_x_rho": [0.5]}).to_csv(
                cmd[cmd.index("--out_region_to_gene_adjacencies") + 1], sep="\t", index=False
            )
        elif cmd[:3] == ["scenicplus", "grn_inference", "motif_enrichment_cistarget"]:
            Path(cmd[cmd.index("--output_fname_cistarget_result") + 1]).write_text("cistarget\n", encoding="utf-8")
        elif cmd[:3] == ["scenicplus", "prepare_data", "prepare_menr"]:
            Path(cmd[cmd.index("--out_file_tf_names") + 1]).write_text("GeneA\n", encoding="utf-8")
            ad.AnnData(
                X=np.array([[1.0]]),
                obs=pd.DataFrame(index=pd.Index(["chr1:100-220"])),
                var=pd.DataFrame(index=pd.Index(["GeneA"])),
            ).write_h5ad(cmd[cmd.index("--out_file_direct_annotation") + 1])
            ad.AnnData(
                X=np.array([[1.0]]),
                obs=pd.DataFrame(index=pd.Index(["chr1:100-220"])),
                var=pd.DataFrame(index=pd.Index(["GeneA"])),
            ).write_h5ad(cmd[cmd.index("--out_file_extended_annotation") + 1])
        elif cmd[:3] == ["scenicplus", "grn_inference", "TF_to_gene"]:
            pd.DataFrame(
                {
                    "TF": ["GeneA"],
                    "target": ["GeneA"],
                    "importance": [0.75],
                    "regulation": [1],
                    "rho": [0.3],
                }
            ).to_csv(cmd[cmd.index("--out_tf_to_gene_adjacencies") + 1], sep="\t", index=False)

    monkeypatch.setattr(resume_mod, "run_logged_command", fake_run_logged_command)
    monkeypatch.setattr(
        resume_mod,
        "_derive_tfb_from_direct_h5ad",
        lambda prepared_mdata_path, p2g_df, direct_h5ad_path: pd.DataFrame({"cre": ["chr1-100-220"], "tf": ["GeneA"], "score": [5.0]}),
    )

    args = _make_args(home)
    args.region_set_mode = "lineage_vs_rest"
    resume_mod.main(args)

    assert generated == ["NaiveB"]
    assert len(calls) == 5
    final_csv = home / "benchmark" / "test_run" / "net" / "SCENICPLUS_NaiveB.csv"
    assert final_csv.exists()


@pytest.mark.integration
def test_real_fixture_can_run_twice_with_resume(scenicplus_fixture_root: Path, workflow_script_dir: Path) -> None:
    if os.environ.get("SCENICPLUS_RUN_REAL_INTEGRATION") != "1":
        pytest.skip("Set SCENICPLUS_RUN_REAL_INTEGRATION=1 to run the real ScenicPlus integration test")
    if shutil.which("scenicplus") is None:
        pytest.skip("scenicplus CLI is not available on PATH")
    if os.environ.get("SCENICPLUS_DB_ROOT") is None:
        pytest.skip("SCENICPLUS_DB_ROOT must be set for the real ScenicPlus integration test")

    home = scenicplus_fixture_root
    dataset = "t-cell-depleted-bm_naiveb_100"
    region_root = home / "benchmark" / "data" / "scenicplus_region_sets" / dataset
    if not region_root.exists():
        pytest.skip(f"Missing generated ScenicPlus region sets: {region_root}")

    version = f"pytest_scenicplus_{os.getpid()}"
    cmd = [
        sys.executable,
        str(workflow_script_dir / "run_scenicplus_resume.py"),
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
        "--region-set-root",
        str(region_root),
        "--threads",
        "2",
        "--ext",
        "50000",
    ]
    env = os.environ.copy()

    first = subprocess.run(cmd, cwd=str(workflow_script_dir), capture_output=True, text=True, check=False, env=env)
    assert first.returncode == 0, first.stderr

    second = subprocess.run(cmd, cwd=str(workflow_script_dir), capture_output=True, text=True, check=False, env=env)
    assert second.returncode == 0, second.stderr

    final_csv = home / "benchmark" / version / "net" / "SCENICPLUS_NaiveB.csv"
    assert final_csv.exists()
