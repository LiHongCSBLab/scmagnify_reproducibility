"""
SCENIC+: benchmark script

A standalone baseline runner inspired by GRETA's decoupled SCENIC+ logic,
but implemented locally without importing or calling GRETA scripts at runtime.

Key differences from the original SCENIC+ workflow:
- skips pycisTopic topic modelling
- skips DAR generation
- expects user-provided region sets for motif enrichment
- prefers paired multiome input in benchmark/data/<dataset>.h5mu
"""

from __future__ import annotations

import argparse
import logging
import os
import pathlib
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Iterable

import anndata as ad
import mudata as mu
import numpy as np
import pandas as pd
import psutil
import pyranges as pr
import scipy.sparse as sp
import session_info


@dataclass
class ScenicPlusResources:
    annotation_tsv: pathlib.Path
    chromsizes_tsv: pathlib.Path
    rankings_feather: pathlib.Path
    motif_annotation_tsv: pathlib.Path
    species: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SCENIC+ baseline")
    parser.add_argument("-p", "--home", dest="dirPjtHome", type=pathlib.Path, required=True,
                        help="Path to the project home directory")
    parser.add_argument("-d", "--dataset", dest="dataset", type=str, required=True,
                        help="Dataset key")
    parser.add_argument("-c", "--cell", dest="celllist", type=pathlib.Path, required=True,
                        help="Path to cell list file (.csv)")
    parser.add_argument("-g", "--gene", dest="genelist", type=pathlib.Path, required=True,
                        help="Path to gene list file (.csv)")
    parser.add_argument("-v", "--version", dest="version", type=str, required=True,
                        help="Benchmark version")
    parser.add_argument("-t", "--tmp-save", dest="save", type=bool, default=False,
                        help="Temporary flag")
    parser.add_argument("-s", "--seed", dest="seed", type=int, default=0,
                        help="Random seed")
    parser.add_argument("-r", "--ref-genome", dest="refGenome", type=str, default="hg38",
                        help="Reference genome")
    parser.add_argument("--region-set-root", dest="regionSetRoot", type=pathlib.Path, default=None,
                        help="Root directory containing user-supplied region sets")
    parser.add_argument("--scenicplus-db-root", dest="dbRoot", type=pathlib.Path, default=None,
                        help="Root directory containing SCENIC+ reference resources")
    parser.add_argument("--threads", dest="threads", type=int, default=16,
                        help="Number of threads for SCENIC+ CLI steps")
    parser.add_argument("--ext", dest="ext", type=int, default=250000,
                        help="Search-space extension passed to SCENIC+")
    return parser.parse_args()


def log_memory_usage() -> None:
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    logging.info("Memory usage: %.2f MB", memory_info.rss / 1024 ** 2)


def _run_command(cmd: list[str], *, cwd: pathlib.Path | None = None) -> None:
    logging.info("Running command: %s", " ".join(map(str, cmd)))
    result = subprocess.run(cmd, cwd=str(cwd) if cwd else None, text=True, capture_output=True)
    if result.stdout:
        logging.info(result.stdout)
    if result.stderr:
        logging.info(result.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {' '.join(map(str, cmd))}")


def _truthy_mask(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    lowered = series.astype(str).str.strip().str.lower()
    return lowered.isin(["true", "1", "yes", "y"])


def _resolve_input_data(home_dir: pathlib.Path, dataset: str) -> tuple[pathlib.Path, str]:
    h5mu_path = home_dir / "benchmark" / "data" / f"{dataset}.h5mu"
    h5ad_path = home_dir / "benchmark" / "data" / f"{dataset}.h5ad"
    if h5mu_path.exists():
        return h5mu_path, "h5mu"
    if h5ad_path.exists():
        return h5ad_path, "h5ad"
    raise FileNotFoundError(f"Neither {h5mu_path} nor {h5ad_path} exists")


def _resolve_reference_resources(ref_genome: str, db_root: pathlib.Path | None, home_dir: pathlib.Path) -> ScenicPlusResources:
    if db_root is None:
        env_root = os.environ.get("SCENICPLUS_DB_ROOT")
        if env_root:
            db_root = pathlib.Path(env_root)
        else:
            db_root = home_dir / "benchmark" / "reference" / "scenicplus"

    species_map = {
        "hg38": ScenicPlusResources(
            annotation_tsv=db_root / "hg38" / "genome" / "annotation.tsv",
            chromsizes_tsv=db_root / "hg38" / "genome" / "chromsizes.tsv",
            rankings_feather=db_root / "hg38" / "motif" / "human_motif_SCREEN.regions_vs_motifs.rankings.feather",
            motif_annotation_tsv=db_root / "hg38" / "motif" / "motifs-v10nr_clust" / "nr.hgnc-m0.001-o0.0.tbl",
            species="homo_sapiens",
        ),
        "mm10": ScenicPlusResources(
            annotation_tsv=db_root / "mm10" / "genome" / "annotation.tsv",
            chromsizes_tsv=db_root / "mm10" / "genome" / "chromsizes.tsv",
            rankings_feather=db_root / "mm10" / "motif" / "mouse_motif_SCREEN.regions_vs_motifs.rankings.feather",
            motif_annotation_tsv=db_root / "mm10" / "motif" / "motifs-v10nr_clust" / "nr.mgi-m0.001-o0.0.tbl",
            species="mus_musculus",
        ),
    }
    if ref_genome not in species_map:
        raise ValueError(f"Unsupported reference genome: {ref_genome}")

    resources = species_map[ref_genome]
    missing = [
        str(p) for p in [
            resources.annotation_tsv,
            resources.chromsizes_tsv,
            resources.rankings_feather,
            resources.motif_annotation_tsv,
        ] if not p.exists()
    ]
    if missing:
        raise FileNotFoundError("Missing SCENIC+ reference resources:\n" + "\n".join(missing))
    return resources


def _resolve_region_set_root(home_dir: pathlib.Path, dataset: str, region_set_root: pathlib.Path | None) -> pathlib.Path:
    if region_set_root is None:
        env_root = os.environ.get("SCENICPLUS_REGION_SET_ROOT")
        if env_root:
            region_set_root = pathlib.Path(env_root)
        else:
            region_set_root = home_dir / "benchmark" / "data" / "scenicplus_region_sets" / dataset
    if not region_set_root.exists():
        raise FileNotFoundError(f"Region set root does not exist: {region_set_root}")
    return region_set_root


def _find_modality_keys(mod: dict[str, ad.AnnData], candidates: Iterable[str]) -> str | None:
    for key in candidates:
        if key in mod:
            return key
    return None


def _load_lineage_mdata(
    input_path: pathlib.Path,
    input_type: str,
    lineage: str,
    cell_selected: pd.DataFrame,
    gene_selected: pd.Index,
) -> mu.MuData:
    lineage_mask = _truthy_mask(cell_selected[lineage])
    lineage_cells = pd.Index(cell_selected.index[lineage_mask])
    if lineage_cells.empty:
        raise ValueError(f"No cells selected for lineage {lineage}")

    if input_type == "h5mu":
        mdata = mu.read(input_path)
        rna_key = _find_modality_keys(mdata.mod, ["scRNA", "RNA", "rna"])
        atac_key = _find_modality_keys(mdata.mod, ["scATAC", "ATAC", "atac"])
        if rna_key is None or atac_key is None:
            raise ValueError("Input h5mu must contain RNA and ATAC modalities")
        rna = mdata.mod[rna_key][lineage_cells].copy()
        atac = mdata.mod[atac_key][lineage_cells].copy()
    else:
        adata = ad.read_h5ad(input_path)
        obsm_key = next((k for k in ["ATAC", "atac", "Peaks", "peaks"] if k in adata.obsm), None)
        var_key = next((k for k in ["ATAC_var_names", "atac_var_names", "peak_names", "Peaks_var_names"] if k in adata.uns), None)
        if obsm_key is None or var_key is None:
            raise ValueError(
                "Input h5ad does not expose paired ATAC information in a supported obsm/uns layout; provide benchmark/data/<dataset>.h5mu instead."
            )
        rna = adata[lineage_cells].copy()
        atac_x = adata[lineage_cells].obsm[obsm_key]
        if sp.issparse(atac_x):
            atac_x = atac_x.tocsr()
        atac = ad.AnnData(
            X=atac_x,
            obs=adata.obs.loc[lineage_cells].copy(),
            var=pd.DataFrame(index=pd.Index(adata.uns[var_key]).astype(str)),
        )

    rna = rna[:, rna.var_names.intersection(gene_selected)].copy()
    if rna.n_vars == 0:
        raise ValueError(f"No selected genes remain for lineage {lineage}")

    if "counts" in rna.layers:
        rna.X = rna.layers["counts"].copy()
    if "counts" in atac.layers:
        atac.X = atac.layers["counts"].copy()

    shared_cells = rna.obs_names.intersection(atac.obs_names)
    if shared_cells.empty:
        raise ValueError(f"RNA and ATAC cells do not overlap for lineage {lineage}")

    return mu.MuData({"rna": rna[shared_cells].copy(), "atac": atac[shared_cells].copy()})


def _prepare_scenicplus_mudata(mdata: mu.MuData) -> mu.MuData:
    rna = mdata.mod["rna"].copy()
    atac = mdata.mod["atac"].copy()
    atac.var_names = atac.var_names.astype(str).str.replace("-", ":", n=1, regex=False)
    return mu.MuData({"scRNA": rna, "scATAC": atac})


def _discover_search_space_subcommand() -> str:
    for name in ["search_spance", "search_space"]:
        result = subprocess.run(["scenicplus", "prepare_data", name, "--help"], capture_output=True, text=True)
        if result.returncode == 0:
            return name
    raise RuntimeError("Could not find scenicplus prepare_data search_spance/search_space subcommand")


def _format_region_to_gene_output(rg_adj_path: pathlib.Path) -> pd.DataFrame:
    rg = pd.read_table(rg_adj_path)
    if rg.shape[0] == 0:
        return pd.DataFrame(columns=["cre", "gene", "score"])
    rg = rg.loc[rg["importance_x_rho"].abs() > 1e-16, ["region", "target", "importance_x_rho"]].copy()
    rg["region"] = rg["region"].astype(str).str.replace(":", "-", n=1, regex=False)
    rg.columns = ["cre", "gene", "score"]
    return rg


def _get_pr(index: pd.Index) -> pr.PyRanges:
    df = index.astype(str).str.replace(":", "-", regex=False).str.split("-", expand=True)
    df.columns = ["Chromosome", "Start", "End"]
    return pr.PyRanges(df)


def _get_vars(ranges: pr.PyRanges) -> pd.Index:
    rdf = ranges.df
    return pd.Index(rdf["Chromosome"].astype(str) + ":" + rdf["Start"].astype(str) + "-" + rdf["End"].astype(str))


def _derive_tfb_from_direct_h5ad(prepared_mdata_path: pathlib.Path, p2g_df: pd.DataFrame, direct_h5ad_path: pathlib.Path) -> pd.DataFrame:
    if p2g_df.shape[0] == 0:
        return pd.DataFrame(columns=["cre", "tf", "score"])

    motifs = ad.read_h5ad(direct_h5ad_path)
    prepared = mu.read(prepared_mdata_path)
    genes = prepared.mod["scRNA"].var_names
    motifs = motifs[:, motifs.var_names.isin(genes)].copy()

    motifs_pr = _get_pr(pd.Index(motifs.obs_names))
    p2g_pr = _get_pr(pd.Index(p2g_df["cre"].unique()))
    inter = motifs_pr.join(p2g_pr)
    if inter.df.shape[0] == 0:
        return pd.DataFrame(columns=["cre", "tf", "score"])

    inter_motifs = _get_vars(pr.PyRanges(inter.df[["Chromosome", "Start", "End"]]))
    inter_p2g = _get_vars(pr.PyRanges(inter.df[["Chromosome", "Start_b", "End_b"]].rename(columns={"Start_b": "Start", "End_b": "End"})))

    new_motifs = ad.AnnData(
        X=sp.csr_matrix((inter_p2g.size, motifs.var_names.size)),
        obs=pd.DataFrame(index=inter_p2g),
        var=pd.DataFrame(index=motifs.var_names),
    )
    new_motifs[inter_p2g, :].X = motifs[inter_motifs, :].X
    coo = new_motifs.X.tocoo()
    tfb = pd.DataFrame({
        "cre": new_motifs.obs_names[coo.row],
        "tf": new_motifs.var_names[coo.col],
        "score": 5.0,
    })
    tfb["cre"] = tfb["cre"].str.replace(":", "-", n=1, regex=False)
    return tfb


def _format_egrn_output(egrn_path: pathlib.Path) -> pd.DataFrame:
    df = pd.read_table(egrn_path)
    if df.shape[0] == 0:
        return pd.DataFrame(columns=["TF", "Target", "Score"])
    df = df[df["regulation"] != 0.0].copy()
    if df.shape[0] == 0:
        return pd.DataFrame(columns=["TF", "Target", "Score"])
    df = df[["TF", "Region", "Gene", "regulation", "triplet_rank"]].sort_values("triplet_rank").reset_index(drop=True)
    df = df.reset_index(names="rank")
    rank_max = max(df["rank"].max(), 1)
    df["score"] = (1 - (df["rank"] / rank_max)) * df["regulation"]
    edge_df = df[["TF", "Gene", "score"]].copy()
    edge_df.columns = ["TF", "Target", "Score"]
    edge_df = edge_df.sort_values("Score", ascending=False).reset_index(drop=True)
    return edge_df


def _lineage_region_set_dir(region_set_root: pathlib.Path, lineage: str) -> pathlib.Path:
    candidate = region_set_root / lineage
    return candidate if candidate.exists() else region_set_root


def main(args: argparse.Namespace) -> None:
    benchmark_dir = args.dirPjtHome / "benchmark" / args.version
    tmp_save_dir = args.dirPjtHome / "tmp" / "scenicplus_wd" / args.version
    net_dir = benchmark_dir / "net"
    log_dir = benchmark_dir / "log"
    fig_dir = benchmark_dir / "fig"
    for path in [benchmark_dir, tmp_save_dir, net_dir, log_dir, fig_dir]:
        path.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        filename=log_dir / "SCENICPLUS.log",
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        filemode="w",
    )

    np.random.seed(args.seed)
    logging.info("Benchmark Version: %s with seed %s", args.version, args.seed)
    logging.info("Packages Version: %s", session_info.show())
    log_memory_usage()

    resources = _resolve_reference_resources(args.refGenome, args.dbRoot, args.dirPjtHome)
    region_set_root = _resolve_region_set_root(args.dirPjtHome, args.dataset, args.regionSetRoot)
    input_data, input_type = _resolve_input_data(args.dirPjtHome, args.dataset)
    search_space_subcommand = _discover_search_space_subcommand()

    logging.info("Input data: %s (%s)", input_data, input_type)
    logging.info("Region set root: %s", region_set_root)
    logging.info("Skipping pycisTopic, MALLET, DAR generation, and DEM motif enrichment")

    gene_selected = pd.read_csv(args.genelist, header=None)[0].astype(str)
    cell_selected = pd.read_csv(args.celllist, index_col=0)
    logging.info("Gene list: %s", args.genelist)
    logging.info("Selected %d genes to benchmark", len(gene_selected))
    logging.info("Cell list: %s", args.celllist)
    log_memory_usage()

    for i, lin in enumerate(cell_selected.columns):
        logging.info("(%d/%d) Cell State %s...", i + 1, len(cell_selected.columns), lin)
        step_start = time.time()
        lin_dir = tmp_save_dir / lin
        lin_dir.mkdir(parents=True, exist_ok=True)

        try:
            lineage_mdata = _load_lineage_mdata(input_data, input_type, lin, cell_selected, pd.Index(gene_selected))
            scenic_mdata = _prepare_scenicplus_mudata(lineage_mdata)
            prepared_mdata_path = lin_dir / "mdata.h5mu"
            scenic_mdata.write(prepared_mdata_path)

            region_set_dir = _lineage_region_set_dir(region_set_root, lin)
            if not region_set_dir.exists():
                raise FileNotFoundError(f"Region set directory does not exist for lineage {lin}: {region_set_dir}")

            _run_command([
                "scenicplus", "prepare_data", search_space_subcommand,
                "--multiome_mudata_fname", str(prepared_mdata_path),
                "--gene_annotation_fname", str(resources.annotation_tsv),
                "--chromsizes_fname", str(resources.chromsizes_tsv),
                "--upstream", "1000", str(args.ext),
                "--downstream", "1000", str(args.ext),
                "--out_fname", str(lin_dir / "space.tsv"),
            ])

            _run_command([
                "scenicplus", "grn_inference", "region_to_gene",
                "--multiome_mudata_fname", str(prepared_mdata_path),
                "--search_space_fname", str(lin_dir / "space.tsv"),
                "--temp_dir", os.environ.get("TMPDIR", "/tmp"),
                "--out_region_to_gene_adjacencies", str(lin_dir / "rg_adj.tsv"),
                "--n_cpu", str(args.threads),
            ])

            p2g_df = _format_region_to_gene_output(lin_dir / "rg_adj.tsv")
            p2g_df.to_csv(lin_dir / "p2g.csv", index=False)

            _run_command([
                "scenicplus", "grn_inference", "motif_enrichment_cistarget",
                "--region_set_folder", str(region_set_dir),
                "--cistarget_db_fname", str(resources.rankings_feather),
                "--output_fname_cistarget_result", str(lin_dir / "cistarget.hdf5"),
                "--path_to_motif_annotations", str(resources.motif_annotation_tsv),
                "--annotations_to_use", "Direct_annot", "Orthology_annot",
                "--temp_dir", os.environ.get("TMPDIR", "/tmp"),
                "--species", resources.species,
                "--n_cpu", str(args.threads),
            ])

            _run_command([
                "scenicplus", "prepare_data", "prepare_menr",
                "--paths_to_motif_enrichment_results", str(lin_dir / "cistarget.hdf5"),
                "--multiome_mudata_fname", str(prepared_mdata_path),
                "--out_file_tf_names", str(lin_dir / "tfs.txt"),
                "--out_file_direct_annotation", str(lin_dir / "direct.h5ad"),
                "--out_file_extended_annotation", str(lin_dir / "extended.h5ad"),
                "--direct_annotation", "Direct_annot", "Orthology_annot",
                "--extended_annotation", "Orthology_annot",
            ])

            tfb_df = _derive_tfb_from_direct_h5ad(prepared_mdata_path, p2g_df, lin_dir / "direct.h5ad")
            tfb_df.to_csv(lin_dir / "tfb.csv", index=False)

            _run_command([
                "scenicplus", "grn_inference", "TF_to_gene",
                "--multiome_mudata_fname", str(prepared_mdata_path),
                "--tf_names", str(lin_dir / "tfs.txt"),
                "--temp_dir", os.environ.get("TMPDIR", "/tmp"),
                "--out_tf_to_gene_adjacencies", str(lin_dir / "tg_adj.tsv"),
                "--method", "GBM",
                "--n_cpu", str(args.threads),
            ])

            egrn_cmd = [
                "scenicplus", "grn_inference", "eGRN",
                "--TF_to_gene_adj_fname", str(lin_dir / "tg_adj.tsv"),
                "--region_to_gene_adj_fname", str(lin_dir / "rg_adj.tsv"),
                "--cistromes_fname", str(lin_dir / "direct.h5ad"),
                "--ranking_db_fname", str(resources.rankings_feather),
                "--eRegulon_out_fname", str(lin_dir / "egrn.tsv"),
                "--temp_dir", os.environ.get("TMPDIR", "/tmp"),
                "--min_target_genes", "10",
                "--n_cpu", str(args.threads),
            ]
            if p2g_df.shape[0] > 0 and not (p2g_df["score"] < 0).any():
                egrn_cmd.extend(["--do_not_rho_dichotomize_r2g", "--do_not_rho_dichotomize_eRegulon"])
            _run_command(egrn_cmd)

            edge_df = _format_egrn_output(lin_dir / "egrn.tsv")
            out_csv = net_dir / f"SCENICPLUS_{lin}.csv"
            edge_df.to_csv(out_csv, index=False)
            logging.info("Saved network to %s", out_csv)
            logging.info("Finished %s in %.2f seconds", lin, time.time() - step_start)
            log_memory_usage()
        finally:
            if not args.save and lin_dir.exists():
                shutil.rmtree(lin_dir)

    logging.info("SCENIC+ baseline finished!")
    log_memory_usage()


if __name__ == "__main__":
    main(parse_args())
