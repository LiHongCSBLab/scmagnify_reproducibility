"""
SCENIC+ benchmark script with checkpoint-aware resume support.

Current output behavior:
- The final benchmark network is written directly from `tg_adj.tsv`
  (the `scenicplus grn_inference TF_to_gene` output).
- The final CSV has columns `TF`, `Target`, `Score`, where
  `Score = importance * regulation`.

Required region-set layout:
- `--region-set-root` should point to the dataset-level root.
- Under that root, each lineage must contain nested set folders with BED files, e.g.
  `<region-set-root>/<lineage>/set_01/set_01.bed`
  `<region-set-root>/<lineage>/set_02/set_02.bed`
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pathlib
import re
import shutil
import tempfile
import time
from dataclasses import dataclass
from typing import Iterable

import anndata as ad
import mudata as mu
import numpy as np
import pandas as pd
import pyranges as pr
import scanpy as sc
import scipy.sparse as sp
import session_info

from baseline_cli_utils import log_memory_usage, run_logged_command, str2bool
from run_scenicplus import (
    ScenicPlusResources,
    _derive_tfb_from_direct_h5ad,
    _discover_search_space_subcommand,
    _find_modality_keys,
    _format_egrn_output,
    _format_region_to_gene_output,
    _get_vars,
    _lineage_region_set_dir,
    _load_lineage_mdata,
    _resolve_input_data,
    _resolve_reference_resources,
    _resolve_region_set_root,
    _truthy_mask,
)

STAGE_PREPARE = "prepare"
STAGE_SEARCH_SPACE = "search_space"
STAGE_REGION_TO_GENE = "region_to_gene"
STAGE_MOTIF = "motif"
STAGE_MENR = "menr"
STAGE_TF_TO_GENE = "tf_to_gene"
STAGE_EGRN = "egrn"
STAGE_FINAL = "final"
REGION_SET_MODE_MANUAL = "manual"
REGION_SET_MODE_LINEAGE_VS_REST = "lineage_vs_rest"
REGION_SET_MODE_LINEAGE_OPEN = "lineage_open"
MOTIF_PARAMS_FILENAME = "motif_params.json"
ALL_STAGES = (
    STAGE_PREPARE,
    STAGE_SEARCH_SPACE,
    STAGE_REGION_TO_GENE,
    STAGE_MOTIF,
    STAGE_MENR,
    STAGE_TF_TO_GENE,
    STAGE_EGRN,
    STAGE_FINAL,
)


@dataclass(frozen=True)
class LineageArtifacts:
    lineage: str
    lineage_dir: pathlib.Path
    net_dir: pathlib.Path

    @property
    def prepared_mdata(self) -> pathlib.Path:
        return self.lineage_dir / "mdata.h5mu"

    @property
    def search_space(self) -> pathlib.Path:
        return self.lineage_dir / "space.tsv"

    @property
    def rg_adj(self) -> pathlib.Path:
        return self.lineage_dir / "rg_adj.tsv"

    @property
    def p2g_csv(self) -> pathlib.Path:
        return self.lineage_dir / "p2g.csv"

    @property
    def cistarget(self) -> pathlib.Path:
        return self.lineage_dir / "cistarget.hdf5"

    @property
    def tfs_txt(self) -> pathlib.Path:
        return self.lineage_dir / "tfs.txt"

    @property
    def direct_h5ad(self) -> pathlib.Path:
        return self.lineage_dir / "direct.h5ad"

    @property
    def extended_h5ad(self) -> pathlib.Path:
        return self.lineage_dir / "extended.h5ad"

    @property
    def tfb_csv(self) -> pathlib.Path:
        return self.lineage_dir / "tfb.csv"

    @property
    def tg_adj(self) -> pathlib.Path:
        return self.lineage_dir / "tg_adj.tsv"

    @property
    def egrn(self) -> pathlib.Path:
        return self.lineage_dir / "egrn.tsv"

    @property
    def final_csv(self) -> pathlib.Path:
        return self.net_dir / f"SCENICPLUS_{self.lineage}.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SCENIC+ baseline with checkpoint resume")
    parser.add_argument("-p", "--home", dest="dirPjtHome", type=pathlib.Path, required=True, help="Path to the project home directory")
    parser.add_argument("-d", "--dataset", dest="dataset", type=str, required=True, help="Dataset key")
    parser.add_argument("-c", "--cell", dest="celllist", type=pathlib.Path, required=True, help="Path to cell list file (.csv)")
    parser.add_argument("-g", "--gene", dest="genelist", type=pathlib.Path, required=True, help="Path to gene list file (.csv)")
    parser.add_argument("-v", "--version", dest="version", type=str, required=True, help="Benchmark version")
    parser.add_argument(
        "-t",
        "--tmp-save",
        dest="save",
        type=str2bool,
        default=True,
        help="Whether to preserve temporary working files (defaults to true for resume support)",
    )
    parser.add_argument("-s", "--seed", dest="seed", type=int, default=0, help="Random seed")
    parser.add_argument("-r", "--ref-genome", dest="refGenome", type=str, default="hg38", help="Reference genome")
    parser.add_argument(
        "--region-set-root",
        dest="regionSetRoot",
        type=pathlib.Path,
        default=None,
        help="Dataset-level root containing lineage subdirectories with nested BED sets, e.g. <root>/NaiveB/set_01/set_01.bed",
    )
    parser.add_argument(
        "--region-set-mode",
        dest="region_set_mode",
        choices=[REGION_SET_MODE_MANUAL, REGION_SET_MODE_LINEAGE_VS_REST, REGION_SET_MODE_LINEAGE_OPEN],
        default=REGION_SET_MODE_MANUAL,
        help=(
            "How region sets are supplied: use existing nested BED files (`manual`), "
            "generate differential lineage-vs-rest BED with scanpy (`lineage_vs_rest`), "
            "or keep lineage-open peaks (`lineage_open`)."
        ),
    )
    parser.add_argument(
        "--region-fdr",
        dest="region_fdr",
        type=float,
        default=0.05,
        help="Adjusted p-value cutoff for lineage-vs-rest peak selection (default: 0.05); used only when --region-set-mode lineage_vs_rest",
    )
    parser.add_argument(
        "--region-min-logfc",
        dest="region_min_logfc",
        type=float,
        default=0.0,
        help="Minimum log fold-change for lineage-vs-rest peak selection (default: 0.0); used only when --region-set-mode lineage_vs_rest",
    )
    parser.add_argument(
        "--scenicplus-db-root",
        dest="dbRoot",
        type=pathlib.Path,
        default=None,
        help="Root directory containing SCENIC+ reference resources such as genome_annotation.tsv, chromsizes.tsv, rankings feather, and motif annotations",
    )
    parser.add_argument(
        "--auc-threshold",
        dest="auc_threshold",
        type=float,
        default=0.005,
        help=(
            "cisTarget AUC window as a fraction of ranked regions (0, 1]; "
            "default 0.005 = top 0.5%% of the database"
        ),
    )
    parser.add_argument(
        "--nes-threshold",
        dest="nes_threshold",
        type=float,
        default=3.0,
        help="Minimum NES for calling enriched motifs in motif_enrichment_cistarget (default: 3.0)",
    )
    parser.add_argument("--threads", dest="threads", type=int, default=16, help="Number of threads for SCENIC+ CLI steps")
    parser.add_argument("--ext", dest="ext", type=int, default=250000, help="Search-space extension passed to SCENIC+")
    parser.add_argument("--resume", dest="resume", type=str2bool, default=True, help="Resume from existing checkpoints when possible")
    parser.add_argument("--resume-mode", dest="resume_mode", choices=["lineage", "stage"], default="stage", help="Skip either complete lineages or individual stages")
    parser.add_argument(
        "--force-rebuild",
        dest="force_rebuild",
        nargs="*",
        default=None,
        help="Stages to rebuild even if outputs exist. Accepts repeated values or comma-separated values from: "
        + ", ".join(ALL_STAGES)
        + ", all",
    )
    return parser.parse_args()


def build_lineage_artifacts(tmp_save_dir: pathlib.Path, net_dir: pathlib.Path, lineage: str) -> LineageArtifacts:
    return LineageArtifacts(lineage=lineage, lineage_dir=tmp_save_dir / lineage, net_dir=net_dir)


def normalize_force_rebuild(values: Iterable[str] | None) -> set[str]:
    normalized: set[str] = set()
    if values is None:
        return normalized
    for value in values:
        for piece in str(value).split(","):
            stage = piece.strip().lower()
            if not stage:
                continue
            if stage == "all":
                return set(ALL_STAGES)
            if stage not in ALL_STAGES:
                raise ValueError(f"Unsupported stage for --force-rebuild: {stage}")
            normalized.add(stage)
    return normalized


def stage_outputs(artifacts: LineageArtifacts, stage: str) -> tuple[pathlib.Path, ...]:
    mapping = {
        STAGE_PREPARE: (artifacts.prepared_mdata,),
        STAGE_SEARCH_SPACE: (artifacts.search_space,),
        STAGE_REGION_TO_GENE: (artifacts.rg_adj, artifacts.p2g_csv),
        STAGE_MOTIF: (artifacts.cistarget,),
        STAGE_MENR: (artifacts.tfs_txt, artifacts.direct_h5ad, artifacts.extended_h5ad, artifacts.tfb_csv),
        STAGE_TF_TO_GENE: (artifacts.tg_adj,),
        STAGE_EGRN: (artifacts.egrn,),
        STAGE_FINAL: (artifacts.final_csv,),
    }
    return mapping[stage]


def outputs_ready(paths: Iterable[pathlib.Path]) -> bool:
    return all(path.exists() for path in paths)


def decide_stage_action(
    *,
    stage: str,
    outputs_exist: bool,
    resume: bool,
    upstream_changed: bool,
    force_rebuild: set[str],
) -> str:
    if stage in force_rebuild:
        return "rebuild" if outputs_exist else "run"
    if not resume:
        return "rebuild" if outputs_exist else "run"
    if upstream_changed:
        return "rebuild" if outputs_exist else "run"
    if outputs_exist:
        return "skip"
    return "run"


def should_skip_lineage(*, artifacts: LineageArtifacts, resume: bool, resume_mode: str, force_rebuild: set[str]) -> bool:
    if not resume or resume_mode != "lineage":
        return False
    if force_rebuild:
        return False
    return artifacts.final_csv.exists()


def log_stage_decision(lineage: str, stage: str, action: str, reason: str) -> None:
    logging.info("[%s] %s %s (%s)", lineage, stage.upper(), action.upper(), reason)


def validate_motif_thresholds(auc_threshold: float, nes_threshold: float) -> None:
    if not 0.0 < auc_threshold <= 1.0:
        raise ValueError(f"--auc-threshold must be within (0, 1], got {auc_threshold}")
    if nes_threshold <= 0:
        raise ValueError(f"--nes-threshold must be > 0, got {nes_threshold}")


def motif_run_params(args: argparse.Namespace) -> dict[str, float]:
    return {
        "auc_threshold": float(args.auc_threshold),
        "nes_threshold": float(args.nes_threshold),
    }


def read_stage_params(path: pathlib.Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        logging.warning("Ignoring invalid stage params file: %s", path)
        return None


def write_stage_params(path: pathlib.Path, params: dict) -> None:
    path.write_text(json.dumps(params, indent=2, sort_keys=True) + "\n")


def stage_params_stale(path: pathlib.Path, expected: dict, *, outputs_exist: bool) -> bool:
    if not outputs_exist:
        return False
    saved = read_stage_params(path)
    if saved is None:
        return True
    return saved != expected


def read_optional_csv(path: pathlib.Path, columns: list[str], **kwargs) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path, **kwargs)
    return pd.DataFrame(columns=columns)


def load_input_obs_names(input_path: pathlib.Path, input_type: str) -> pd.Index:
    if input_type == "h5mu":
        mdata = mu.read(input_path)
        rna_key = _find_modality_keys(mdata.mod, ["scRNA", "RNA", "rna"])
        atac_key = _find_modality_keys(mdata.mod, ["scATAC", "ATAC", "atac"])
        if rna_key is None or atac_key is None:
            return pd.Index([])
        return mdata.mod[rna_key].obs_names.intersection(mdata.mod[atac_key].obs_names)

    adata = ad.read_h5ad(input_path)
    return pd.Index(adata.obs_names)


def warn_mask_issues(cell_selected: pd.DataFrame, lineage: str) -> None:
    series = cell_selected[lineage]
    missing = int(series.isna().sum())
    if missing > 0:
        logging.warning("[%s] Cell mask contains %d NA values; treating them as unselected", lineage, missing)


def validate_lineage_cells(lineage: str, cell_selected: pd.DataFrame, input_obs_names: pd.Index) -> pd.Index:
    warn_mask_issues(cell_selected, lineage)
    lineage_mask = _truthy_mask(cell_selected[lineage].fillna(False))
    lineage_cells = pd.Index(cell_selected.index[lineage_mask])
    overlap = lineage_cells.intersection(input_obs_names)
    if lineage_cells.empty:
        raise ValueError(f"No cells selected for lineage {lineage}")
    if overlap.empty:
        raise ValueError(f"No selected cells from {lineage} overlap the input object's obs_names")
    missing = lineage_cells.difference(input_obs_names)
    if len(missing) > 0:
        logging.warning("[%s] %d selected cells are absent from the input object", lineage, len(missing))
    return overlap


def validate_region_set_dir(region_set_dir: pathlib.Path, lineage: str) -> pathlib.Path:
    if not region_set_dir.exists():
        raise FileNotFoundError(f"Region set directory does not exist for lineage {lineage}: {region_set_dir}")
    bed_files = sorted(region_set_dir.glob("*/*.bed"))
    if not bed_files:
        raise FileNotFoundError(
            f"Region set directory for lineage {lineage} does not contain any nested .bed files: {region_set_dir}"
        )
    return region_set_dir


def resolve_motif_region_set_folder(
    *,
    lineage_root: pathlib.Path,
    lineage: str,
    region_set_mode: str,
    staging_parent: pathlib.Path,
) -> pathlib.Path:
    """Return a region-set folder for cisTarget containing only the active mode.

    scenicplus scans every subfolder under ``region_set_folder``. When both
    ``lineage_open`` and ``lineage_vs_rest`` exist under the lineage root, it
    would run both and append twice to the same HDF5. For auto-generated modes
    we stage a directory with a single symlinked subfolder.
    """
    if region_set_mode == REGION_SET_MODE_MANUAL:
        return validate_region_set_dir(lineage_root, lineage)

    mode_dir = lineage_root / region_set_mode
    expected_bed = mode_dir / f"{region_set_mode}.bed"
    if not expected_bed.exists():
        raise FileNotFoundError(
            f"Region set BED does not exist for lineage {lineage} ({region_set_mode}): {expected_bed}"
        )

    staging = staging_parent / f"region_sets_motif_{region_set_mode}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)
    link_path = staging / region_set_mode
    if link_path.exists() or link_path.is_symlink():
        link_path.unlink()
    link_path.symlink_to(mode_dir, target_is_directory=True)
    logging.info("[%s] Motif region-set folder (active mode only): %s", lineage, staging)
    return validate_region_set_dir(staging, lineage)


def _subprocess_env_with_hdf5_nfs_compat() -> dict[str, str]:
    env = os.environ.copy()
    # TrueNas/NFS often fails HDF5 flock on append ("Resource temporarily unavailable").
    env["HDF5_USE_FILE_LOCKING"] = "FALSE"
    return env


def _local_cistarget_output_path(lineage: str) -> pathlib.Path:
    return pathlib.Path(tempfile.gettempdir()) / f"scenicplus_{lineage}_{os.getpid()}_cistarget.hdf5"


def _install_cistarget_output(local_h5: pathlib.Path, destination: pathlib.Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    shutil.move(str(local_h5), str(destination))


def parse_region_name(region: str) -> tuple[str, int, int]:
    text = str(region).strip().replace(":", "-", 1)
    match = re.fullmatch(r"(.+)-(\d+)-(\d+)", text)
    if match is None:
        raise ValueError(f"Unsupported region format: {region!r}")
    chrom, start, end = match.groups()
    return chrom, int(start), int(end)


def region_index_to_bed(region_index: pd.Index) -> pd.DataFrame:
    rows = [parse_region_name(region) for region in region_index.astype(str)]
    return pd.DataFrame(rows, columns=["chr", "start", "end"]).sort_values(["chr", "start", "end"]).reset_index(drop=True)


def _lineage_region_root(region_set_root: pathlib.Path, lineage: str) -> pathlib.Path:
    return region_set_root if region_set_root.name == lineage else region_set_root / lineage


def _load_full_atac_adata(input_path: pathlib.Path, input_type: str) -> ad.AnnData:
    if input_type == "h5mu":
        mdata = mu.read(input_path)
        atac_key = _find_modality_keys(mdata.mod, ["scATAC", "ATAC", "atac"])
        if atac_key is None:
            raise ValueError("Input h5mu must contain an ATAC modality")
        atac = mdata.mod[atac_key].copy()
    else:
        adata = ad.read_h5ad(input_path)
        obsm_key = next((k for k in ["ATAC", "atac", "Peaks", "peaks"] if k in adata.obsm), None)
        var_key = next((k for k in ["ATAC_var_names", "atac_var_names", "peak_names", "Peaks_var_names"] if k in adata.uns), None)
        if obsm_key is None or var_key is None:
            raise ValueError(
                "Input h5ad does not expose paired ATAC information in a supported obsm/uns layout; provide benchmark/data/<dataset>.h5mu instead."
            )
        atac_x = adata.obsm[obsm_key]
        if sp.issparse(atac_x):
            atac_x = atac_x.tocsr()
        atac = ad.AnnData(
            X=atac_x,
            obs=adata.obs.copy(),
            var=pd.DataFrame(index=pd.Index(adata.uns[var_key]).astype(str)),
        )

    if "counts" in atac.layers:
        atac.X = atac.layers["counts"].copy()
    if sp.issparse(atac.X):
        atac.X = atac.X.tocsr()
    return atac


def _lineage_open_peak_names(atac_lineage: ad.AnnData) -> pd.Index:
    x = atac_lineage.X
    if sp.issparse(x):
        # sparse matrix keeps this fast for large peak spaces
        open_mask = np.asarray((x > 0).sum(axis=0)).ravel() > 0
    else:
        open_mask = np.asarray(x > 0).sum(axis=0).ravel() > 0
    return pd.Index(atac_lineage.var_names[open_mask].astype(str))


def filter_lineage_vs_rest_markers(rank_df: pd.DataFrame, *, fdr: float, min_logfc: float) -> pd.DataFrame:
    if not 0 < fdr <= 1:
        raise ValueError(f"--region-fdr must be within (0, 1], got {fdr}")
    required = {"names", "pvals_adj"}
    missing = sorted(required.difference(rank_df.columns))
    if missing:
        raise ValueError(f"rank_genes_groups output is missing required columns: {', '.join(missing)}")

    mask = rank_df["pvals_adj"].fillna(np.inf) <= fdr
    if "logfoldchanges" in rank_df.columns:
        mask &= rank_df["logfoldchanges"].fillna(-np.inf) > min_logfc
    if {"pct_nz_group", "pct_nz_reference"}.issubset(rank_df.columns):
        mask &= rank_df["pct_nz_group"].fillna(0) > rank_df["pct_nz_reference"].fillna(0)

    filtered = rank_df.loc[mask].copy()
    if filtered.shape[0] == 0:
        return filtered
    filtered = filtered.sort_values(["pvals_adj", "scores"], ascending=[True, False], na_position="last").reset_index(drop=True)
    return filtered


def build_lineage_vs_rest_region_set(
    *,
    input_data: pathlib.Path,
    input_type: str,
    lineage: str,
    cell_selected: pd.DataFrame,
    region_set_root: pathlib.Path,
    fdr: float,
    min_logfc: float,
    overwrite: bool,
) -> pathlib.Path:
    lineage_root = _lineage_region_root(region_set_root, lineage)
    set_dir = lineage_root / REGION_SET_MODE_LINEAGE_VS_REST
    bed_path = set_dir / f"{REGION_SET_MODE_LINEAGE_VS_REST}.bed"
    metadata_path = set_dir / "metadata.csv"

    if bed_path.exists() and not overwrite:
        return lineage_root
    if set_dir.exists() and overwrite:
        shutil.rmtree(set_dir)
    set_dir.mkdir(parents=True, exist_ok=True)

    atac = _load_full_atac_adata(input_data, input_type)
    shared_cells = atac.obs_names.intersection(cell_selected.index)
    if shared_cells.empty:
        raise ValueError(f"No cells in the ATAC object overlap the cell mask for lineage {lineage}")

    warn_mask_issues(cell_selected, lineage)
    lineage_mask = _truthy_mask(cell_selected.reindex(shared_cells)[lineage].fillna(False))
    n_lineage = int(lineage_mask.sum())
    n_rest = int((~lineage_mask).sum())
    if n_lineage == 0:
        raise ValueError(f"No lineage cells are available to build a lineage-vs-rest region set for {lineage}")
    if n_rest == 0:
        raise ValueError(f"No rest cells are available to build a lineage-vs-rest region set for {lineage}")

    atac = atac[shared_cells].copy()
    atac.obs["__lineage_vs_rest__"] = pd.Categorical(np.where(lineage_mask, lineage, "rest"), categories=[lineage, "rest"])
    sc.tl.rank_genes_groups(
        atac,
        groupby="__lineage_vs_rest__",
        groups=[lineage],
        reference="rest",
        method="wilcoxon",
        use_raw=False,
        pts=True,
    )
    rank_df = sc.get.rank_genes_groups_df(atac, group=lineage)
    filtered = filter_lineage_vs_rest_markers(rank_df, fdr=fdr, min_logfc=min_logfc)
    if filtered.shape[0] == 0:
        raise ValueError(
            f"No peaks passed lineage-vs-rest filtering for {lineage}. "
            f"Recommended starting thresholds are FDR <= 0.05 and logFC > 0; current values are FDR <= {fdr} and logFC > {min_logfc}."
        )

    bed_df = region_index_to_bed(pd.Index(filtered["names"].astype(str).unique()))
    bed_df.to_csv(bed_path, sep="\t", header=False, index=False)
    pd.DataFrame(
        [
            {"key": "mode", "value": REGION_SET_MODE_LINEAGE_VS_REST},
            {"key": "lineage", "value": lineage},
            {"key": "n_lineage_cells", "value": str(n_lineage)},
            {"key": "n_rest_cells", "value": str(n_rest)},
            {"key": "fdr_threshold", "value": str(fdr)},
            {"key": "min_logfc", "value": str(min_logfc)},
            {"key": "recommended_filter", "value": "pvals_adj <= 0.05 and logfoldchanges > 0"},
            {"key": "n_selected_regions", "value": str(bed_df.shape[0])},
        ]
    ).to_csv(metadata_path, index=False)
    logging.info(
        "[%s] Generated lineage-vs-rest region set with %d regions using recommended filters: FDR <= %.3g, logFC > %.3g",
        lineage,
        bed_df.shape[0],
        fdr,
        min_logfc,
    )
    return lineage_root


def build_lineage_open_region_set(
    *,
    input_data: pathlib.Path,
    input_type: str,
    lineage: str,
    cell_selected: pd.DataFrame,
    region_set_root: pathlib.Path,
    overwrite: bool,
) -> pathlib.Path:
    lineage_root = _lineage_region_root(region_set_root, lineage)
    set_dir = lineage_root / REGION_SET_MODE_LINEAGE_OPEN
    bed_path = set_dir / f"{REGION_SET_MODE_LINEAGE_OPEN}.bed"
    metadata_path = set_dir / "metadata.csv"

    if bed_path.exists() and not overwrite:
        return lineage_root
    if set_dir.exists() and overwrite:
        shutil.rmtree(set_dir)
    set_dir.mkdir(parents=True, exist_ok=True)

    atac = _load_full_atac_adata(input_data, input_type)
    shared_cells = atac.obs_names.intersection(cell_selected.index)
    if shared_cells.empty:
        raise ValueError(f"No cells in the ATAC object overlap the cell mask for lineage {lineage}")

    warn_mask_issues(cell_selected, lineage)
    lineage_mask = _truthy_mask(cell_selected.reindex(shared_cells)[lineage].fillna(False))
    n_lineage = int(lineage_mask.sum())
    n_rest = int((~lineage_mask).sum())
    if n_lineage == 0:
        raise ValueError(f"No lineage cells are available to build a lineage-open region set for {lineage}")

    atac = atac[shared_cells].copy()
    atac_lineage = atac[lineage_mask.values].copy()
    if atac_lineage.n_obs == 0:
        raise ValueError(f"No lineage cells remain after alignment for {lineage}")

    open_peak_names = _lineage_open_peak_names(atac_lineage)
    if open_peak_names.empty:
        raise ValueError(f"No open peaks found within lineage cells for {lineage}")

    bed_df = region_index_to_bed(open_peak_names.unique())
    bed_df.to_csv(bed_path, sep="\t", header=False, index=False)
    pd.DataFrame(
        [
            {"key": "mode", "value": REGION_SET_MODE_LINEAGE_OPEN},
            {"key": "lineage", "value": lineage},
            {"key": "n_lineage_cells", "value": str(n_lineage)},
            {"key": "n_rest_cells", "value": str(n_rest)},
            {"key": "selection_logic", "value": "keep peaks with non-zero accessibility in lineage cells"},
            {"key": "n_selected_regions", "value": str(bed_df.shape[0])},
        ]
    ).to_csv(metadata_path, index=False)
    logging.info(
        "[%s] Generated lineage-open region set with %d regions (kept peaks with non-zero accessibility in lineage cells)",
        lineage,
        bed_df.shape[0],
    )
    return lineage_root


def ensure_outputs_exist(paths: Iterable[pathlib.Path], stage_name: str) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise RuntimeError(
            f"{stage_name} finished without producing required outputs:\n" + "\n".join(missing)
        )


def _get_pr_compat(index: pd.Index) -> pr.PyRanges:
    split = (
        pd.Series(pd.Index(index).astype(str), dtype="string")
        .str.replace(":", "-", n=1, regex=False)
        .str.split("-", expand=True)
    )
    if split.shape[1] != 3:
        raise ValueError("Expected genomic regions in chr:start-end or chr-start-end format")
    split.columns = ["Chromosome", "Start", "End"]
    split["Start"] = split["Start"].astype(int)
    split["End"] = split["End"].astype(int)
    return pr.PyRanges(split)


def _derive_tfb_from_direct_h5ad(prepared_mdata_path: pathlib.Path, p2g_df: pd.DataFrame, direct_h5ad_path: pathlib.Path) -> pd.DataFrame:
    if p2g_df.shape[0] == 0:
        return pd.DataFrame(columns=["cre", "tf", "score"])

    motifs = ad.read_h5ad(direct_h5ad_path)
    prepared = mu.read(prepared_mdata_path)
    genes = prepared.mod["scRNA"].var_names
    motifs = motifs[:, motifs.var_names.isin(genes)].copy()

    motifs_pr = _get_pr_compat(pd.Index(motifs.obs_names))
    p2g_pr = _get_pr_compat(pd.Index(p2g_df["cre"].unique()))
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


def _format_tf_to_gene_output(tg_adj_path: pathlib.Path) -> pd.DataFrame:
    df = pd.read_table(tg_adj_path)
    if df.shape[0] == 0:
        return pd.DataFrame(columns=["TF", "Target", "Score"])

    required = {"TF", "target", "importance", "regulation"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"TF_to_gene output is missing required columns: {', '.join(missing)}")

    df = df.loc[df["regulation"] != 0, ["TF", "target", "importance", "regulation"]].copy()
    if df.shape[0] == 0:
        return pd.DataFrame(columns=["TF", "Target", "Score"])

    df["Score"] = df["importance"] * df["regulation"]
    edge_df = df[["TF", "target", "Score"]].copy()
    edge_df.columns = ["TF", "Target", "Score"]
    edge_df = edge_df.sort_values("Score", ascending=False).reset_index(drop=True)
    return edge_df


def normalize_peak_name_for_scenicplus(peak: str) -> str:
    text = str(peak).strip()
    match = re.fullmatch(r"(.+?)[-:](\d+)[-:](\d+)", text)
    if match is None:
        raise ValueError(f"Unsupported ATAC peak format for ScenicPlus: {peak!r}")
    chrom, start, end = match.groups()
    return f"{chrom}:{start}-{end}"


def prepare_scenicplus_mudata(mdata: mu.MuData) -> mu.MuData:
    rna = mdata.mod["rna"].copy()
    atac = mdata.mod["atac"].copy()
    atac.var_names = pd.Index([normalize_peak_name_for_scenicplus(peak) for peak in atac.var_names.astype(str)])
    return mu.MuData({"scRNA": rna, "scATAC": atac})


def resolve_and_prepare_region_set_root(args: argparse.Namespace) -> pathlib.Path:
    region_set_root = args.regionSetRoot
    if region_set_root is None:
        env_root = os.environ.get("SCENICPLUS_REGION_SET_ROOT")
        if env_root:
            region_set_root = pathlib.Path(env_root)
        else:
            region_set_root = args.dirPjtHome / "benchmark" / "data" / "scenicplus_region_sets" / args.dataset

    auto_build_modes = {REGION_SET_MODE_LINEAGE_VS_REST, REGION_SET_MODE_LINEAGE_OPEN}
    if args.region_set_mode in auto_build_modes and not region_set_root.exists():
        region_set_root.mkdir(parents=True, exist_ok=True)
        logging.info(
            "Created missing region-set root for %s mode: %s",
            args.region_set_mode,
            region_set_root,
        )

    return _resolve_region_set_root(args.dirPjtHome, args.dataset, region_set_root)


def main(args: argparse.Namespace) -> None:
    validate_motif_thresholds(args.auc_threshold, args.nes_threshold)
    force_rebuild = normalize_force_rebuild(args.force_rebuild)
    current_motif_params = motif_run_params(args)

    benchmark_dir = args.dirPjtHome / "benchmark" / args.version
    tmp_save_dir = args.dirPjtHome / "tmp" / "scenicplus_wd" / args.version
    net_dir = benchmark_dir / "net"
    log_dir = benchmark_dir / "log"
    fig_dir = benchmark_dir / "fig"
    for path in [benchmark_dir, tmp_save_dir, net_dir, log_dir, fig_dir]:
        path.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        filename=log_dir / "SCENICPLUSResume.log",
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        filemode="a",
    )

    np.random.seed(args.seed)
    logging.info("Benchmark Version: %s with seed %s", args.version, args.seed)
    logging.info("Resume enabled: %s (%s)", args.resume, args.resume_mode)
    logging.info("Force rebuild: %s", sorted(force_rebuild))
    logging.info("Packages Version: %s", session_info.show())
    log_memory_usage()

    resources: ScenicPlusResources = _resolve_reference_resources(args.refGenome, args.dbRoot, args.dirPjtHome)
    region_set_root = resolve_and_prepare_region_set_root(args)
    input_data, input_type = _resolve_input_data(args.dirPjtHome, args.dataset)
    input_obs_names = load_input_obs_names(input_data, input_type)
    search_space_subcommand = _discover_search_space_subcommand()

    logging.info("Input data: %s (%s)", input_data, input_type)
    logging.info("Region set root: %s", region_set_root)
    logging.info("Skipping pycisTopic, MALLET, DAR generation, and DEM motif enrichment")
    logging.info(
        "Region-set mode: %s (lineage_vs_rest uses differential filters FDR <= %.3g and logFC > %.3g; lineage_open ignores these thresholds)",
        args.region_set_mode,
        args.region_fdr,
        args.region_min_logfc,
    )
    logging.info(
        "cisTarget motif thresholds: auc_threshold=%.6g, nes_threshold=%.6g",
        args.auc_threshold,
        args.nes_threshold,
    )

    gene_selected = pd.read_csv(args.genelist, header=None)[0].astype(str)
    cell_selected = pd.read_csv(args.celllist, index_col=0)
    logging.info("Gene list: %s", args.genelist)
    logging.info("Selected %d genes to benchmark", len(gene_selected))
    logging.info("Cell list: %s", args.celllist)
    log_memory_usage()

    for i, lineage in enumerate(cell_selected.columns):
        logging.info("(%d/%d) Cell State %s...", i + 1, len(cell_selected.columns), lineage)
        step_start = time.time()
        artifacts = build_lineage_artifacts(tmp_save_dir, net_dir, lineage)

        if should_skip_lineage(artifacts=artifacts, resume=args.resume, resume_mode=args.resume_mode, force_rebuild=force_rebuild):
            logging.info("[%s] LINEAGE SKIP (final output exists: %s)", lineage, artifacts.final_csv)
            continue

        artifacts.lineage_dir.mkdir(parents=True, exist_ok=True)
        lineage_resume = args.resume and args.resume_mode == "stage"
        upstream_changed = False

        try:
            prepare_action = decide_stage_action(
                stage=STAGE_PREPARE,
                outputs_exist=outputs_ready(stage_outputs(artifacts, STAGE_PREPARE)),
                resume=lineage_resume,
                upstream_changed=upstream_changed,
                force_rebuild=force_rebuild,
            )
            log_stage_decision(lineage, STAGE_PREPARE, prepare_action, "checkpoint inspection")
            if prepare_action == "skip":
                scenic_mdata = mu.read(artifacts.prepared_mdata)
            else:
                validate_lineage_cells(lineage, cell_selected, input_obs_names)
                lineage_mdata = _load_lineage_mdata(input_data, input_type, lineage, cell_selected, pd.Index(gene_selected))
                scenic_mdata = prepare_scenicplus_mudata(lineage_mdata)
                scenic_mdata.write(artifacts.prepared_mdata)
                upstream_changed = True

            if args.region_set_mode == REGION_SET_MODE_LINEAGE_VS_REST:
                build_lineage_vs_rest_region_set(
                    input_data=input_data,
                    input_type=input_type,
                    lineage=lineage,
                    cell_selected=cell_selected,
                    region_set_root=region_set_root,
                    fdr=args.region_fdr,
                    min_logfc=args.region_min_logfc,
                    overwrite=(not args.resume) or (STAGE_MOTIF in force_rebuild),
                )
            elif args.region_set_mode == REGION_SET_MODE_LINEAGE_OPEN:
                build_lineage_open_region_set(
                    input_data=input_data,
                    input_type=input_type,
                    lineage=lineage,
                    cell_selected=cell_selected,
                    region_set_root=region_set_root,
                    overwrite=(not args.resume) or (STAGE_MOTIF in force_rebuild),
                )
            lineage_region_root = _lineage_region_set_dir(region_set_root, lineage)
            region_set_dir = resolve_motif_region_set_folder(
                lineage_root=lineage_region_root,
                lineage=lineage,
                region_set_mode=args.region_set_mode,
                staging_parent=artifacts.lineage_dir,
            )

            search_space_action = decide_stage_action(
                stage=STAGE_SEARCH_SPACE,
                outputs_exist=outputs_ready(stage_outputs(artifacts, STAGE_SEARCH_SPACE)),
                resume=lineage_resume,
                upstream_changed=upstream_changed,
                force_rebuild=force_rebuild,
            )
            log_stage_decision(lineage, STAGE_SEARCH_SPACE, search_space_action, "checkpoint inspection")
            if search_space_action != "skip":
                run_logged_command(
                    [
                        "scenicplus",
                        "prepare_data",
                        search_space_subcommand,
                        "--multiome_mudata_fname",
                        str(artifacts.prepared_mdata),
                        "--gene_annotation_fname",
                        str(resources.annotation_tsv),
                        "--chromsizes_fname",
                        str(resources.chromsizes_tsv),
                        "--upstream",
                        "1000",
                        str(args.ext),
                        "--downstream",
                        "1000",
                        str(args.ext),
                        "--out_fname",
                        str(artifacts.search_space),
                    ]
                )
                ensure_outputs_exist(stage_outputs(artifacts, STAGE_SEARCH_SPACE), STAGE_SEARCH_SPACE)
                upstream_changed = True

            region_to_gene_action = decide_stage_action(
                stage=STAGE_REGION_TO_GENE,
                outputs_exist=outputs_ready(stage_outputs(artifacts, STAGE_REGION_TO_GENE)),
                resume=lineage_resume,
                upstream_changed=upstream_changed,
                force_rebuild=force_rebuild,
            )
            log_stage_decision(lineage, STAGE_REGION_TO_GENE, region_to_gene_action, "checkpoint inspection")
            if region_to_gene_action == "skip":
                p2g_df = pd.read_csv(artifacts.p2g_csv)
            else:
                run_logged_command(
                    [
                        "scenicplus",
                        "grn_inference",
                        "region_to_gene",
                        "--multiome_mudata_fname",
                        str(artifacts.prepared_mdata),
                        "--search_space_fname",
                        str(artifacts.search_space),
                        "--temp_dir",
                        os.environ.get("TMPDIR", "/tmp"),
                        "--out_region_to_gene_adjacencies",
                        str(artifacts.rg_adj),
                        "--n_cpu",
                        str(args.threads),
                    ]
                )
                ensure_outputs_exist([artifacts.rg_adj], STAGE_REGION_TO_GENE)
                p2g_df = _format_region_to_gene_output(artifacts.rg_adj)
                p2g_df.to_csv(artifacts.p2g_csv, index=False)
                ensure_outputs_exist(stage_outputs(artifacts, STAGE_REGION_TO_GENE), STAGE_REGION_TO_GENE)
                upstream_changed = True

            motif_params_path = artifacts.lineage_dir / MOTIF_PARAMS_FILENAME
            motif_outputs_exist = outputs_ready(stage_outputs(artifacts, STAGE_MOTIF))
            motif_params_stale = lineage_resume and stage_params_stale(
                motif_params_path,
                current_motif_params,
                outputs_exist=motif_outputs_exist,
            )
            if motif_params_stale:
                saved = read_stage_params(motif_params_path)
                logging.info(
                    "[%s] MOTIF params changed (saved=%s, current=%s); rebuilding motif and downstream stages",
                    lineage,
                    saved,
                    current_motif_params,
                )
            motif_action = decide_stage_action(
                stage=STAGE_MOTIF,
                outputs_exist=outputs_ready(stage_outputs(artifacts, STAGE_MOTIF)),
                resume=lineage_resume,
                upstream_changed=upstream_changed or motif_params_stale,
                force_rebuild=force_rebuild,
            )
            log_stage_decision(
                lineage,
                STAGE_MOTIF,
                motif_action,
                "checkpoint inspection" if not motif_params_stale else "motif threshold params changed",
            )
            if motif_action != "skip":
                local_cistarget = _local_cistarget_output_path(lineage)
                if local_cistarget.exists():
                    local_cistarget.unlink()
                if artifacts.cistarget.exists():
                    artifacts.cistarget.unlink()
                try:
                    run_logged_command(
                        [
                            "scenicplus",
                            "grn_inference",
                            "motif_enrichment_cistarget",
                            "--region_set_folder",
                            str(region_set_dir),
                            "--cistarget_db_fname",
                            str(resources.rankings_feather),
                            "--output_fname_cistarget_result",
                            str(local_cistarget),
                            "--path_to_motif_annotations",
                            str(resources.motif_annotation_tsv),
                            "--auc_threshold",
                            str(args.auc_threshold),
                            "--nes_threshold",
                            str(args.nes_threshold),
                            "--annotations_to_use",
                            "Direct_annot",
                            "Orthology_annot",
                            "--temp_dir",
                            os.environ.get("TMPDIR", "/tmp"),
                            "--species",
                            resources.species,
                            "--n_cpu",
                            str(args.threads),
                        ],
                        env=_subprocess_env_with_hdf5_nfs_compat(),
                    )
                    _install_cistarget_output(local_cistarget, artifacts.cistarget)
                finally:
                    if local_cistarget.exists():
                        local_cistarget.unlink()
                ensure_outputs_exist(stage_outputs(artifacts, STAGE_MOTIF), STAGE_MOTIF)
                write_stage_params(motif_params_path, current_motif_params)
                upstream_changed = True

            menr_action = decide_stage_action(
                stage=STAGE_MENR,
                outputs_exist=outputs_ready(stage_outputs(artifacts, STAGE_MENR)),
                resume=lineage_resume,
                upstream_changed=upstream_changed,
                force_rebuild=force_rebuild,
            )
            log_stage_decision(lineage, STAGE_MENR, menr_action, "checkpoint inspection")
            if menr_action == "skip":
                tfb_df = read_optional_csv(artifacts.tfb_csv, ["cre", "tf", "score"])
            else:
                run_logged_command(
                    [
                        "scenicplus",
                        "prepare_data",
                        "prepare_menr",
                        "--paths_to_motif_enrichment_results",
                        str(artifacts.cistarget),
                        "--multiome_mudata_fname",
                        str(artifacts.prepared_mdata),
                        "--out_file_tf_names",
                        str(artifacts.tfs_txt),
                        "--out_file_direct_annotation",
                        str(artifacts.direct_h5ad),
                        "--out_file_extended_annotation",
                        str(artifacts.extended_h5ad),
                        "--direct_annotation",
                        "Direct_annot",
                        "Orthology_annot",
                        "--extended_annotation",
                        "Orthology_annot",
                    ]
                )
                ensure_outputs_exist(
                    [artifacts.cistarget, artifacts.tfs_txt, artifacts.direct_h5ad, artifacts.extended_h5ad],
                    STAGE_MENR,
                )
                tfb_df = _derive_tfb_from_direct_h5ad(artifacts.prepared_mdata, p2g_df, artifacts.direct_h5ad)
                tfb_df.to_csv(artifacts.tfb_csv, index=False)
                ensure_outputs_exist(stage_outputs(artifacts, STAGE_MENR), STAGE_MENR)
                upstream_changed = True

            tf_to_gene_action = decide_stage_action(
                stage=STAGE_TF_TO_GENE,
                outputs_exist=outputs_ready(stage_outputs(artifacts, STAGE_TF_TO_GENE)),
                resume=lineage_resume,
                upstream_changed=upstream_changed,
                force_rebuild=force_rebuild,
            )
            log_stage_decision(lineage, STAGE_TF_TO_GENE, tf_to_gene_action, "checkpoint inspection")
            if tf_to_gene_action != "skip":
                run_logged_command(
                    [
                        "scenicplus",
                        "grn_inference",
                        "TF_to_gene",
                        "--multiome_mudata_fname",
                        str(artifacts.prepared_mdata),
                        "--tf_names",
                        str(artifacts.tfs_txt),
                        "--temp_dir",
                        os.environ.get("TMPDIR", "/tmp"),
                        "--out_tf_to_gene_adjacencies",
                        str(artifacts.tg_adj),
                        "--method",
                        "GBM",
                        "--n_cpu",
                        str(args.threads),
                    ]
                )
                ensure_outputs_exist(stage_outputs(artifacts, STAGE_TF_TO_GENE), STAGE_TF_TO_GENE)
                upstream_changed = True

            logging.info("[%s] %s SKIP (final network will be derived directly from TF_to_gene output)", lineage, STAGE_EGRN.upper())

            final_action = decide_stage_action(
                stage=STAGE_FINAL,
                outputs_exist=outputs_ready(stage_outputs(artifacts, STAGE_FINAL)),
                resume=lineage_resume,
                upstream_changed=upstream_changed,
                force_rebuild=force_rebuild,
            )
            log_stage_decision(lineage, STAGE_FINAL, final_action, "checkpoint inspection")
            ensure_outputs_exist([artifacts.tg_adj], STAGE_FINAL)
            edge_df = _format_tf_to_gene_output(artifacts.tg_adj)
            if final_action != "skip":
                edge_df.to_csv(artifacts.final_csv, index=False)
                ensure_outputs_exist(stage_outputs(artifacts, STAGE_FINAL), STAGE_FINAL)

            logging.info("Saved network to %s", artifacts.final_csv)
            logging.info("Finished %s in %.2f seconds", lineage, time.time() - step_start)
            log_memory_usage()
        finally:
            if not args.save and artifacts.lineage_dir.exists():
                shutil.rmtree(artifacts.lineage_dir)

    logging.info("SCENIC+ resume baseline finished!")
    log_memory_usage()


if __name__ == "__main__":
    main(parse_args())
