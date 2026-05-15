from __future__ import annotations

import importlib
from pathlib import Path

import anndata as ad
import mudata as mu
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("mudata")
pytest.importorskip("pyranges")

resume_mod = importlib.import_module("run_scenicplus_resume")
region_mod = importlib.import_module("prepare_scenicplus_region_sets")


def test_format_region_to_gene_output_filters_zero_scores(tmp_path: Path) -> None:
    path = tmp_path / "rg.tsv"
    pd.DataFrame(
        {
            "region": ["chr1:100-200", "chr1:300-400"],
            "target": ["GeneA", "GeneB"],
            "importance_x_rho": [0.5, 0.0],
        }
    ).to_csv(path, sep="\t", index=False)

    df = resume_mod._format_region_to_gene_output(path)
    assert df.to_dict(orient="records") == [{"cre": "chr1-100-200", "gene": "GeneA", "score": 0.5}]


def test_format_egrn_output_sorts_and_scores(tmp_path: Path) -> None:
    path = tmp_path / "egrn.tsv"
    pd.DataFrame(
        {
            "TF": ["TF1", "TF2", "TF3"],
            "Region": ["chr1:1-2", "chr1:2-3", "chr1:3-4"],
            "Gene": ["GeneA", "GeneB", "GeneC"],
            "regulation": [1.0, -1.0, 0.0],
            "triplet_rank": [2, 1, 3],
        }
    ).to_csv(path, sep="\t", index=False)

    df = resume_mod._format_egrn_output(path)
    assert list(df["TF"]) == ["TF1", "TF2"]
    assert list(df["Target"]) == ["GeneA", "GeneB"]


def test_format_tf_to_gene_output_filters_zero_regulation_and_signs_score(tmp_path: Path) -> None:
    path = tmp_path / "tg.tsv"
    pd.DataFrame(
        {
            "TF": ["TF1", "TF2", "TF3"],
            "target": ["GeneA", "GeneB", "GeneC"],
            "importance": [3.0, 2.0, 5.0],
            "regulation": [1, -1, 0],
            "rho": [0.2, -0.3, 0.0],
        }
    ).to_csv(path, sep="\t", index=False)

    df = resume_mod._format_tf_to_gene_output(path)
    assert df.to_dict(orient="records") == [
        {"TF": "TF1", "Target": "GeneA", "Score": 3.0},
        {"TF": "TF2", "Target": "GeneB", "Score": -2.0},
    ]


def test_filter_lineage_vs_rest_markers_keeps_significant_positive_hits() -> None:
    df = pd.DataFrame(
        {
            "names": ["chr1:1-2", "chr1:2-3", "chr1:3-4"],
            "pvals_adj": [0.01, 0.2, 0.03],
            "logfoldchanges": [1.5, 3.0, -0.5],
            "pct_nz_group": [0.5, 0.6, 0.7],
            "pct_nz_reference": [0.1, 0.1, 0.2],
            "scores": [5.0, 4.0, 3.0],
        }
    )

    filtered = resume_mod.filter_lineage_vs_rest_markers(df, fdr=0.05, min_logfc=0.0)
    assert filtered["names"].tolist() == ["chr1:1-2"]


def test_build_lineage_vs_rest_region_set_writes_single_nested_bed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    atac = ad.AnnData(
        X=np.array([[3.0, 0.0], [2.0, 0.0], [0.0, 2.0], [0.0, 3.0]]),
        obs=pd.DataFrame(index=pd.Index(["cell1", "cell2", "cell3", "cell4"], name="cell")),
        var=pd.DataFrame(index=pd.Index(["chr1-100-200", "chr1-300-400"], name="peak")),
    )
    mdata = mu.MuData({"atac": atac})
    input_path = tmp_path / "mini.h5mu"
    mdata.write(input_path)

    monkeypatch.setattr(resume_mod.sc.tl, "rank_genes_groups", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        resume_mod.sc.get,
        "rank_genes_groups_df",
        lambda *args, **kwargs: pd.DataFrame(
            {
                "names": ["chr1-100-200", "chr1-300-400"],
                "pvals_adj": [0.01, 0.2],
                "logfoldchanges": [1.0, 0.5],
                "pct_nz_group": [1.0, 0.0],
                "pct_nz_reference": [0.0, 1.0],
                "scores": [4.0, 1.0],
            }
        ),
    )

    lineage_root = resume_mod.build_lineage_vs_rest_region_set(
        input_data=input_path,
        input_type="h5mu",
        lineage="NaiveB",
        cell_selected=pd.DataFrame({"NaiveB": [True, True, False, False]}, index=["cell1", "cell2", "cell3", "cell4"]),
        region_set_root=tmp_path / "region_sets" / "dataset",
        fdr=0.05,
        min_logfc=0.0,
        overwrite=True,
    )

    assert lineage_root == tmp_path / "region_sets" / "dataset" / "NaiveB"
    bed_path = lineage_root / "lineage_vs_rest" / "lineage_vs_rest.bed"
    metadata_path = lineage_root / "lineage_vs_rest" / "metadata.csv"
    assert bed_path.exists()
    assert metadata_path.exists()
    assert "chr1\t100\t200" in bed_path.read_text(encoding="utf-8")


def test_parse_region_name_accepts_colon_and_hyphen() -> None:
    assert region_mod.parse_region_name("chr1:100-200") == ("chr1", 100, 200)
    assert region_mod.parse_region_name("chr1-100-200") == ("chr1", 100, 200)


def test_parse_region_name_rejects_invalid_region() -> None:
    with pytest.raises(ValueError, match="Unsupported region format"):
        region_mod.parse_region_name("chr1")


def test_build_region_sets_is_deterministic() -> None:
    index = pd.Index([f"chr1-{start}-{start + 100}" for start in range(0, 1000, 100)])
    first = region_mod.build_region_sets(index, [3, 4], seed=7)
    second = region_mod.build_region_sets(index, [3, 4], seed=7)

    assert [name for name, _ in first] == [name for name, _ in second]
    assert [name for name, _ in first] == ["set_01", "set_02"]
    for (_, first_df), (_, second_df) in zip(first, second):
        pd.testing.assert_frame_equal(first_df, second_df)


def test_derive_tfb_from_direct_h5ad_handles_pandas_23_index_split(tmp_path: Path) -> None:
    obs = pd.DataFrame(index=pd.Index(["cell1", "cell2"], name="cell"))
    prepared = mu.MuData(
        {
            "scRNA": ad.AnnData(
                X=np.array([[1.0], [2.0]]),
                obs=obs.copy(),
                var=pd.DataFrame(index=pd.Index(["GeneA"], name="gene")),
            ),
            "scATAC": ad.AnnData(
                X=np.array([[1.0], [0.0]]),
                obs=obs.copy(),
                var=pd.DataFrame(index=pd.Index(["chr1:100-200"], name="peak")),
            ),
        }
    )
    prepared_path = tmp_path / "prepared.h5mu"
    prepared.write(prepared_path)

    direct = ad.AnnData(
        X=np.array([[1.0]]),
        obs=pd.DataFrame(index=pd.Index(["chr1:100-200"], name="region")),
        var=pd.DataFrame(index=pd.Index(["GeneA"], name="tf")),
    )
    direct_path = tmp_path / "direct.h5ad"
    direct.write_h5ad(direct_path)

    p2g_df = pd.DataFrame({"cre": ["chr1-100-200"], "gene": ["GeneA"], "score": [0.5]})
    tfb = resume_mod._derive_tfb_from_direct_h5ad(prepared_path, p2g_df, direct_path)

    assert tfb.to_dict(orient="records") == [{"cre": "chr1-100-200", "tf": "GeneA", "score": 5.0}]


def test_parse_set_sizes_rejects_empty_or_nonpositive() -> None:
    with pytest.raises(ValueError, match="At least one region-set size"):
        region_mod.parse_set_sizes("")
    with pytest.raises(ValueError, match="positive integers"):
        region_mod.parse_set_sizes("1,0")
