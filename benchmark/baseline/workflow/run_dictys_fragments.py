"""
Dictys fragments-first benchmark script.
"""

from __future__ import annotations

import argparse
import gzip
import logging
import os
import re
import pathlib
import shutil
import subprocess
import time
from dataclasses import dataclass
from typing import Iterable

import anndata as ad
import dictys
import mudata as mu
import numpy as np
import pandas as pd
import scipy.sparse as sp
import session_info

from baseline_cli_utils import log_memory_usage, run_logged_command, str2bool


@dataclass(frozen=True)
class FragmentSpec:
    path: pathlib.Path
    sample: str | None = None


@dataclass(frozen=True)
class DictysResources:
    homer_genome: pathlib.Path
    motifs: pathlib.Path
    gene_bed: pathlib.Path
    valid_chromosomes: set[str]
    sam_header: str


HG38_CHROM_SIZES = {
    "chr1": 248956422,
    "chr2": 242193529,
    "chr3": 198295559,
    "chr4": 190214555,
    "chr5": 181538259,
    "chr6": 170805979,
    "chr7": 159345973,
    "chr8": 145138636,
    "chr9": 138394717,
    "chr10": 133797422,
    "chr11": 135086622,
    "chr12": 133275309,
    "chr13": 114364328,
    "chr14": 107043718,
    "chr15": 101991189,
    "chr16": 90338345,
    "chr17": 83257441,
    "chr18": 80373285,
    "chr19": 58617616,
    "chr20": 64444167,
    "chr21": 46709983,
    "chr22": 50818468,
    "chrX": 156040895,
    "chrY": 57227415,
}

MM10_CHROM_SIZES = {
    "chr1": 195471971,
    "chr2": 182113224,
    "chr3": 160039680,
    "chr4": 156508116,
    "chr5": 151834684,
    "chr6": 149736546,
    "chr7": 145441459,
    "chr8": 129401213,
    "chr9": 124595110,
    "chr10": 130694993,
    "chr11": 122082543,
    "chr12": 120129022,
    "chr13": 120421639,
    "chr14": 124902244,
    "chr15": 104043685,
    "chr16": 98207768,
    "chr17": 94987271,
    "chr18": 90702639,
    "chr19": 61431566,
    "chrX": 171031299,
    "chrY": 91744698,
}

FWFLAG = 99
BWFLAG = 147
MAPQ = 60
RNEXT = "="
LSHIFT = 4
RSHIFT = -5
SEQLEN = 50
CIGAR = f"{SEQLEN}M"
SEQ = "N" * SEQLEN
QUAL = "F" * SEQLEN
COMMON_SAMPLE_SEPARATORS = ["#", "_", "|", ":", "/"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Dictys from fragment files")
    parser.add_argument("-d", "--dataset", dest="dataset", type=str, required=True,
                        help="Dataset key")
    parser.add_argument("-p", "--home", dest="dirPjtHome", type=pathlib.Path, required=True,
                        help="Path to the project home directory")
    parser.add_argument("-c", "--cell", dest="celllist", type=pathlib.Path, required=True,
                        help="Path to cell list file (.csv)")
    parser.add_argument("-g", "--gene", dest="genelist", type=pathlib.Path, required=True,
                        help="Path to gene list file (.csv)")
    parser.add_argument("-v", "--version", dest="version", type=str, required=True,
                        help="Benchmark version")
    parser.add_argument("-t", "--tmp-save", dest="save", type=str2bool, default=False,
                        help="Temporary flag")
    parser.add_argument("-s", "--seed", dest="seed", type=int, default=0,
                        help="Random seed")
    parser.add_argument("-r", "--ref-genome", dest="refGenome", type=str, default="hg38",
                        help="Reference genome")
    parser.add_argument("--fragments", dest="fragments", nargs="+", default=None,
                        help="One or more fragment files")
    parser.add_argument("--fragment-samples", dest="fragment_samples", nargs="+", default=None,
                        help="Sample label for each fragment file")
    parser.add_argument("--barcode-map", dest="barcode_map", type=pathlib.Path, default=None,
                        help="Optional CSV mapping fragment barcodes to obs names")
    parser.add_argument("--barcode-transform", dest="barcode_transform",
                        choices=["none", "auto", "prefix_suffix"], default="auto",
                        help="How to reconcile fragment barcodes with obs names")
    parser.add_argument("--sample-separator", dest="sample_separator", type=str, default="#",
                        help="Preferred separator between sample and barcode in obs names")
    parser.add_argument("--threads", dest="threads", type=int, default=8,
                        help="Thread count for Dictys CLI steps")
    parser.add_argument("--device", dest="device", type=str, default="cpu",
                        help="Device passed to dictys network reconstruct")
    parser.add_argument("--distance", dest="distance", type=int, default=250000,
                        help="Distance cutoff used for p2g/tssdist")
    parser.add_argument("--n-p2g-links", dest="n_p2g_links", type=int, default=15,
                        help="Links kept by dictys chromatin binlinking")
    parser.add_argument("--thr-score", dest="thr_score", type=float, default=0.0,
                        help="Absolute score threshold for exported edges")
    parser.add_argument("--motifs", dest="motifs", type=pathlib.Path, default=None,
                        help="Optional HOMER motif file")
    parser.add_argument("--homer-genome", dest="homer_genome", type=pathlib.Path, default=None,
                        help="Optional HOMER genome directory")
    parser.add_argument("--gene-bed", dest="gene_bed", type=pathlib.Path, default=None,
                        help="Optional gene annotation BED used by Dictys")
    return parser.parse_args()


def _resolve_input_data(home_dir: pathlib.Path, dataset: str) -> tuple[pathlib.Path, str]:
    h5mu_path = home_dir / "benchmark" / "data" / f"{dataset}.h5mu"
    h5ad_path = home_dir / "benchmark" / "data" / f"{dataset}.h5ad"
    if h5mu_path.exists():
        return h5mu_path, "h5mu"
    if h5ad_path.exists():
        return h5ad_path, "h5ad"
    raise FileNotFoundError(f"Neither {h5mu_path} nor {h5ad_path} exists")


def _find_modality_key(mod: dict[str, ad.AnnData], candidates: Iterable[str]) -> str | None:
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
    lineage_mask = cell_selected[lineage].astype(bool)
    lineage_cells = pd.Index(cell_selected.index[lineage_mask])
    if lineage_cells.empty:
        raise ValueError(f"No cells selected for lineage {lineage}")

    if input_type == "h5mu":
        mdata = mu.read(input_path)
        rna_key = _find_modality_key(mdata.mod, ["RNA", "rna", "scRNA"])
        atac_key = _find_modality_key(mdata.mod, ["ATAC", "atac", "scATAC"])
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

    out = mu.MuData({"rna": rna[shared_cells].copy(), "atac": atac[shared_cells].copy()})
    out.obs["celltype"] = lineage
    return out


def _prepare_preprocessed_mdata(lineage_mdata: mu.MuData, work_dir: pathlib.Path) -> mu.MuData:
    expr_path = work_dir / "pre_qc_expression.tsv.gz"
    rna = lineage_mdata.mod["rna"].copy()
    atac = lineage_mdata.mod["atac"].copy()

    rna_counts = rna.layers["counts"] if "counts" in rna.layers else rna.X
    atac_counts = atac.layers["counts"] if "counts" in atac.layers else atac.X
    if sp.issparse(rna_counts):
        rna_counts = rna_counts.toarray()
    if sp.issparse(atac_counts):
        atac_counts = atac_counts.toarray()

    pd.DataFrame(rna_counts.T, index=rna.var_names, columns=rna.obs_names).to_csv(expr_path, sep="\t", compression="gzip")
    dictys.preproc.qc_reads(str(expr_path), str(expr_path), 50, 10, 0, 200, 100, 0)

    rna_df = pd.read_csv(expr_path, sep="\t", compression="gzip", index_col=0)
    genes = pd.Index(rna_df.index.astype(str))
    barcodes = pd.Index(rna_df.columns.astype(str))

    rna = rna[barcodes, genes].copy()
    atac = atac[atac.obs_names.intersection(barcodes)].copy()
    shared = rna.obs_names.intersection(atac.obs_names)
    if shared.empty:
        raise ValueError("No shared cells remain after Dictys QC")

    rna = rna[shared].copy()
    atac = atac[shared].copy()
    out = mu.MuData({"rna": rna, "atac": atac})
    out.obs["celltype"] = lineage_mdata.obs.loc[shared, "celltype"].astype(str).values
    return out


def _resolve_resources(args: argparse.Namespace) -> DictysResources:
    if args.refGenome == "hg38":
        homer_genome = args.homer_genome or pathlib.Path("/mnt/TrueNas/project/chenxufeng/Ref/human/hg38/homer_genome")
        motifs = args.motifs or pathlib.Path("/mnt/TrueNas/project/chenxufeng/Database/motif_databases/HOCOMOCO/HOCOMOCOv11_full_HUMAN_mono_homer_format_0.0001.motif")
        gene_bed = args.gene_bed or pathlib.Path("/mnt/TrueNas/project/chenxufeng/Ref/human/hg38/annotations/ucsc/gene.bed")
        chrom_sizes = HG38_CHROM_SIZES
    elif args.refGenome == "mm10":
        homer_genome = args.homer_genome or pathlib.Path("/mnt/TrueNas/project/chenxufeng/Ref/mouse/mm10/homer_genome")
        motifs = args.motifs or pathlib.Path("/mnt/TrueNas/project/chenxufeng/Database/motif_databases/HOCOMOCO/HOCOMOCOv11_full_MOUSE_mono_homer_format_0.0001.motif")
        gene_bed = args.gene_bed or pathlib.Path("/mnt/TrueNas/project/chenxufeng/Ref/mouse/mm10/annotations/ucsc/gene.bed")
        chrom_sizes = MM10_CHROM_SIZES
    else:
        raise ValueError(f"Unsupported reference genome: {args.refGenome}")

    missing = [str(path) for path in [homer_genome, motifs, gene_bed] if not pathlib.Path(path).exists()]
    if missing:
        raise FileNotFoundError("Missing Dictys reference resources:\n" + "\n".join(missing))

    header = "@HD\tSO:coordinate\n" + "\n".join(
        f"@SQ\tSN:{chrom}\tLN:{length}" for chrom, length in chrom_sizes.items()
    ) + "\n"
    return DictysResources(
        homer_genome=pathlib.Path(homer_genome),
        motifs=pathlib.Path(motifs),
        gene_bed=pathlib.Path(gene_bed),
        valid_chromosomes=set(chrom_sizes),
        sam_header=header,
    )


def _resolve_fragment_specs(args: argparse.Namespace) -> list[FragmentSpec]:
    if args.fragments:
        paths = [pathlib.Path(path) for path in args.fragments]
        samples = args.fragment_samples
    else:
        manifest_csv = args.dirPjtHome / "benchmark" / "data" / f"{args.dataset}.fragments.csv"
        manifest_txt = args.dirPjtHome / "benchmark" / "data" / f"{args.dataset}.fragments.txt"
        env_paths = os.environ.get("DICTYS_FRAGMENTS")
        env_samples = os.environ.get("DICTYS_FRAGMENT_SAMPLES")
        if manifest_csv.exists():
            manifest = pd.read_csv(manifest_csv)
            if "path" not in manifest.columns:
                raise ValueError(f"Fragment manifest missing 'path' column: {manifest_csv}")
            paths = [pathlib.Path(path) for path in manifest["path"].astype(str)]
            samples = manifest["sample"].astype(str).tolist() if "sample" in manifest.columns else None
        elif manifest_txt.exists():
            paths = [pathlib.Path(line.strip()) for line in manifest_txt.read_text().splitlines() if line.strip()]
            samples = None
        elif env_paths:
            paths = [pathlib.Path(part) for part in env_paths.split(",") if part.strip()]
            samples = [part.strip() for part in env_samples.split(",")] if env_samples else None
        else:
            raise ValueError(
                "No fragment inputs provided. Use --fragments, set DICTYS_FRAGMENTS, or create benchmark/data/<dataset>.fragments.csv"
            )

    if args.fragment_samples is not None and args.fragments is not None:
        samples = args.fragment_samples

    if samples is not None and len(samples) != len(paths):
        raise ValueError("--fragment-samples must have the same length as --fragments")

    specs = [FragmentSpec(path=path, sample=samples[i] if samples is not None else None) for i, path in enumerate(paths)]
    missing = [str(spec.path) for spec in specs if not spec.path.exists()]
    if missing:
        raise FileNotFoundError("Missing fragment files:\n" + "\n".join(missing))
    return specs


def _load_barcode_map(path: pathlib.Path | None) -> dict[tuple[str | None, str], str]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"Barcode map does not exist: {path}")

    df = pd.read_csv(path)
    raw_col = next((col for col in ["fragment_barcode", "raw_barcode", "barcode"] if col in df.columns), None)
    obs_col = next((col for col in ["obs_name", "cell", "cell_name"] if col in df.columns), None)
    sample_col = "sample" if "sample" in df.columns else None
    if raw_col is None or obs_col is None:
        raise ValueError("Barcode map must contain fragment_barcode/raw_barcode/barcode and obs_name/cell columns")

    mapping: dict[tuple[str | None, str], str] = {}
    for row in df.itertuples(index=False):
        raw = str(getattr(row, raw_col))
        obs = str(getattr(row, obs_col))
        sample = str(getattr(row, sample_col)) if sample_col else None
        key = (sample, raw)
        if key in mapping and mapping[key] != obs:
            raise ValueError(f"Conflicting barcode map entries for {key}")
        mapping[key] = obs
    return mapping


def _candidate_keys(obs_name: str, sample: str | None, transform: str, sample_separator: str) -> set[str]:
    keys = {obs_name}
    if transform == "none" or sample is None:
        return keys

    separators = [sample_separator] + [sep for sep in COMMON_SAMPLE_SEPARATORS if sep != sample_separator]
    for sep in separators:
        prefix = f"{sample}{sep}"
        suffix = f"{sep}{sample}"
        if obs_name.startswith(prefix):
            keys.add(obs_name[len(prefix):])
        if obs_name.endswith(suffix):
            keys.add(obs_name[:-len(suffix)])
    return keys


def _build_obs_lookup(
    obs_names: Iterable[str],
    fragment_specs: list[FragmentSpec],
    transform: str,
    sample_separator: str,
) -> dict[tuple[str | None, str], str]:
    lookup: dict[tuple[str | None, str], str] = {}
    ambiguous: set[tuple[str | None, str]] = set()
    samples = sorted({spec.sample for spec in fragment_specs if spec.sample is not None})

    for obs_name in map(str, obs_names):
        for key in _candidate_keys(obs_name, None, "none", sample_separator):
            pair = (None, key)
            if pair in lookup and lookup[pair] != obs_name:
                ambiguous.add(pair)
            else:
                lookup[pair] = obs_name
        if transform != "none":
            for sample in samples:
                for key in _candidate_keys(obs_name, sample, transform, sample_separator):
                    pair = (sample, key)
                    if pair in lookup and lookup[pair] != obs_name:
                        ambiguous.add(pair)
                    else:
                        lookup[pair] = obs_name

    for pair in ambiguous:
        lookup.pop(pair, None)
    if ambiguous:
        examples = ", ".join(f"{sample or 'NA'}::{barcode}" for sample, barcode in list(ambiguous)[:5])
        logging.warning("Dropping %d ambiguous barcode keys: %s", len(ambiguous), examples)
    return lookup


def _resolve_obs_name(
    raw_barcode: str,
    spec: FragmentSpec,
    obs_lookup: dict[tuple[str | None, str], str],
    barcode_map: dict[tuple[str | None, str], str],
) -> str | None:
    explicit = barcode_map.get((spec.sample, raw_barcode))
    if explicit is None:
        explicit = barcode_map.get((None, raw_barcode))
    if explicit is not None:
        return explicit

    observed = obs_lookup.get((spec.sample, raw_barcode))
    if observed is not None:
        return observed
    return obs_lookup.get((None, raw_barcode))


def _p2g_cre_to_bed_rows(cre_series: pd.Series) -> pd.DataFrame:
    """Turn CRE strings from _build_p2g into BED rows.

    ``cre`` is ``region`` with only the first ':' replaced by '-' (see _build_p2g). Dictys may emit
    ``chrom:start-end`` or ``chrom:start:end``; after that step the suffix is ``start-end`` or
    ``start:end`` respectively.
    """
    coord_pair = re.compile(r"^(\d+)\s*[-:]\s*(\d+)$")
    rows: list[tuple[str, int, int]] = []
    for cre in cre_series.drop_duplicates().astype(str):
        first_dash = cre.find("-")
        if first_dash < 0:
            raise ValueError(f"Invalid CRE (no hyphen): {cre!r}")
        chrom, tail = cre[:first_dash], cre[first_dash + 1 :].strip()
        m = coord_pair.fullmatch(tail)
        if m is None:
            raise ValueError(
                f"Invalid CRE (expected 'start-end' or 'start:end' after chrom): {cre!r}"
            )
        rows.append((chrom, int(m.group(1)), int(m.group(2))))
    return pd.DataFrame(rows, columns=["chr", "start", "end"])


def _canonical_mdl_peak_loc(cre: str) -> str:
    """Normalize CRE strings so p2g and tfb peaks use the same ``loc`` key.

    Dictys ``chromatin tssdist`` expects peak names in ``chr:start:end`` format. Upstream
    p2g/tfb intermediates can mix ``chr-start:end`` and ``chr-start-end``. Parse
    chrom/start/end like :func:`_p2g_cre_to_bed_rows`, then normalize to the format expected
    by Dictys MDL subcommands.
    """
    cre = str(cre).strip()
    first_dash = cre.find("-")
    if first_dash < 0:
        return cre
    chrom, tail = cre[:first_dash], cre[first_dash + 1 :].strip()
    coord_pair = re.compile(r"^(\d+)\s*[-:]\s*(\d+)$")
    m = coord_pair.fullmatch(tail)
    if m is None:
        return cre.replace("-", ":", 1)
    return f"{chrom}:{m.group(1)}:{m.group(2)}"


def _write_peaks_bed(atac_var_names: pd.Index, path: pathlib.Path) -> None:
    peaks = pd.Index(atac_var_names.astype(str)).str.replace(":", "-", regex=False)
    peak_df = peaks.str.split("-", expand=True)
    peak_df.columns = ["chr", "start", "end"]
    peak_df["start"] = peak_df["start"].astype(int)
    peak_df["end"] = peak_df["end"].astype(int)
    peak_df = peak_df[(peak_df["end"] - peak_df["start"]) >= 100].sort_values(["chr", "start", "end"])
    peak_df.to_csv(path, sep="\t", header=False, index=False)


def _write_expr_tsv(rna: ad.AnnData, path: pathlib.Path) -> None:
    counts = rna.layers["counts"] if "counts" in rna.layers else rna.X
    if sp.issparse(counts):
        counts = counts.toarray()
    pd.DataFrame(counts.T, index=rna.var_names, columns=rna.obs_names).to_csv(path, sep="\t", compression="gzip")


def _build_p2g(prepared_mdata_path: pathlib.Path, output_path: pathlib.Path, temp_dir: pathlib.Path, gene_bed: pathlib.Path, distance: int) -> pd.DataFrame:
    mdata = mu.read(prepared_mdata_path)
    expr_path = temp_dir / "p2g_expr.tsv.gz"
    peaks_path = temp_dir / "p2g_peaks.tsv.gz"

    _write_expr_tsv(mdata.mod["rna"], expr_path)
    peaks = pd.Index(mdata.mod["atac"].var_names.astype(str)).str.replace("-", ":", n=1, regex=False)
    pd.DataFrame({"placeholder": np.zeros(peaks.size)}, index=peaks).to_csv(peaks_path, sep="\t", compression="gzip")

    dist_path = temp_dir / "p2g_tssdist.tsv.gz"
    run_logged_command([
        "python", "-m", "dictys", "chromatin", "tssdist",
        "--cut", str(distance),
        str(expr_path), str(peaks_path), str(gene_bed), str(dist_path),
    ])

    df = pd.read_csv(dist_path, sep="\t").rename(columns={"region": "cre", "target": "gene", "dist": "score"})
    if df.empty:
        out = pd.DataFrame(columns=["cre", "gene", "score"])
    else:
        df["score"] = -np.abs(df["score"])
        df["cre"] = df["cre"].astype(str).str.replace(":", "-", n=1, regex=False)
        df = df.sort_values("score", ascending=False).reset_index(drop=True).reset_index(names="rank")
        rank_max = max(df["rank"].max(), 1)
        df["score"] = 1 - (df["rank"] / rank_max)
        out = df[["cre", "gene", "score"]]
    out.to_csv(output_path, index=False)
    return out


def _fragments_to_bam(
    fragment_specs: list[FragmentSpec],
    target_obs_names: pd.Index,
    bam_path: pathlib.Path,
    bai_path: pathlib.Path,
    resources: DictysResources,
    obs_lookup: dict[tuple[str | None, str], str],
    barcode_map: dict[tuple[str | None, str], str],
) -> pd.DataFrame:
    target_set = set(map(str, target_obs_names))
    stats_rows: list[dict[str, object]] = []

    view_proc = subprocess.Popen(["samtools", "view", "-b", "-"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    if view_proc.stdin is None or view_proc.stdout is None:
        raise RuntimeError("Failed to start samtools view pipe")
    sort_proc = subprocess.Popen(["samtools", "sort", "-o", str(bam_path)], stdin=view_proc.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    view_proc.stdout.close()

    matched_any = False
    try:
        view_proc.stdin.write(resources.sam_header.encode())
        for spec in fragment_specs:
            total_records = 0
            kept_records = 0
            matched_cells: set[str] = set()
            with gzip.open(spec.path, "rt", encoding="utf-8") as handle:
                for line in handle:
                    total_records += 1
                    fields = line.rstrip("\n").split("\t")
                    if len(fields) < 4:
                        continue
                    chrom, start, end, raw_barcode = fields[:4]
                    count = int(fields[4]) if len(fields) > 4 and fields[4] else 1
                    if chrom not in resources.valid_chromosomes:
                        continue
                    obs_name = _resolve_obs_name(raw_barcode, spec, obs_lookup, barcode_map)
                    if obs_name is None or obs_name not in target_set:
                        continue
                    matched_any = True
                    matched_cells.add(obs_name)
                    kept_records += count

                    fwpos = int(start) - LSHIFT + 1
                    bwpos = int(end) - RSHIFT + 1 - SEQLEN
                    tlen = bwpos + SEQLEN - fwpos
                    qname = f"{chrom}:{start}:{end}:{obs_name}"
                    for rep in range(count):
                        view_proc.stdin.write(
                            f"{qname}:{rep}\t{FWFLAG}\t{chrom}\t{fwpos}\t{MAPQ}\t{CIGAR}\t{RNEXT}\t{bwpos}\t{tlen}\t{SEQ}\t{QUAL}\tCB:Z:{obs_name}\n".encode()
                        )
                        view_proc.stdin.write(
                            f"{qname}:{rep}\t{BWFLAG}\t{chrom}\t{bwpos}\t{MAPQ}\t{CIGAR}\t{RNEXT}\t{fwpos}\t{-tlen}\t{SEQ}\t{QUAL}\tCB:Z:{obs_name}\n".encode()
                        )
            stats_rows.append({
                "fragment_path": str(spec.path),
                "sample": spec.sample,
                "total_fragment_rows": total_records,
                "kept_fragment_counts": kept_records,
                "matched_cells": len(matched_cells),
            })
    finally:
        view_proc.stdin.close()

    view_return = view_proc.wait()
    sort_stdout, sort_stderr = sort_proc.communicate()
    if view_return != 0:
        raise RuntimeError("samtools view failed while building BAM")
    if sort_proc.returncode != 0:
        stderr_text = sort_stderr.decode() if sort_stderr else ""
        raise RuntimeError(f"samtools sort failed while building BAM: {stderr_text}")
    if not matched_any:
        raise ValueError("No selected cells matched the provided fragment barcodes")

    run_logged_command(["samtools", "index", str(bam_path), str(bai_path)])
    return pd.DataFrame(stats_rows)


def _build_tfb(
    prepared_mdata_path: pathlib.Path,
    p2g_df: pd.DataFrame,
    lineage: str,
    lineage_dir: pathlib.Path,
    fragment_specs: list[FragmentSpec],
    resources: DictysResources,
    obs_lookup: dict[tuple[str | None, str], str],
    barcode_map: dict[tuple[str | None, str], str],
    threads: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    mdata = mu.read(prepared_mdata_path)
    expr_path = lineage_dir / "expr.tsv.gz"
    peaks_bed = lineage_dir / "peaks.bed"
    bam_path = lineage_dir / f"reads_{lineage}.bam"
    bai_path = lineage_dir / f"reads_{lineage}.bai"
    foot_path = lineage_dir / f"foot_{lineage}.tsv.gz"
    motif_path = lineage_dir / f"motif_{lineage}.tsv.gz"
    well_path = lineage_dir / f"well_{lineage}.tsv.gz"
    homer_path = lineage_dir / f"homer_{lineage}.tsv.gz"
    bind_path = lineage_dir / f"bind_{lineage}.tsv.gz"
    tfb_bed = lineage_dir / f"tfb_{lineage}.bed"
    tfb_csv = lineage_dir / "tfb.csv"

    _write_expr_tsv(mdata.mod["rna"], expr_path)
    if p2g_df.empty:
        _write_peaks_bed(mdata.mod["atac"].var_names, peaks_bed)
    else:
        peak_df = _p2g_cre_to_bed_rows(p2g_df["cre"])
        peak_df["start"] = peak_df["start"].astype(int)
        peak_df["end"] = peak_df["end"].astype(int)
        peak_df = peak_df[(peak_df["end"] - peak_df["start"]) >= 100].sort_values(["chr", "start", "end"])
        peak_df.to_csv(peaks_bed, sep="\t", header=False, index=False)

    qc_df = _fragments_to_bam(
        fragment_specs=fragment_specs,
        target_obs_names=mdata.mod["rna"].obs_names,
        bam_path=bam_path,
        bai_path=bai_path,
        resources=resources,
        obs_lookup=obs_lookup,
        barcode_map=barcode_map,
    )
    qc_df.to_csv(lineage_dir / "barcode_match_qc.csv", index=False)

    run_logged_command(["python", "-m", "dictys", "chromatin", "wellington", str(bam_path), str(bai_path), str(peaks_bed), str(foot_path), "--nth", str(threads)])
    run_logged_command(["python", "-m", "dictys", "chromatin", "homer", str(foot_path), str(resources.motifs), str(resources.homer_genome), str(expr_path), str(motif_path), str(well_path), str(homer_path)])
    run_logged_command(["python", "-m", "dictys", "chromatin", "binding", str(well_path), str(homer_path), str(bind_path)])

    intersect_cmd = (
        f"zcat {bind_path} | "
        "awk 'BEGIN {FS=\"\\t\"; OFS=\"\\t\"} NR>1 {split($2, coords, \":\"); print coords[1], coords[2], coords[3], $1, $3}' | "
        f"bedtools intersect -a {peaks_bed} -b stdin -wa -wb | "
        "awk 'BEGIN {FS=\"\\t\"; OFS=\"\\t\"} {print $1, $2, $3, $7, $8}' > "
        f"{tfb_bed}"
    )
    run_logged_command(["bash", "-lc", intersect_cmd])

    if tfb_bed.exists() and tfb_bed.stat().st_size > 0:
        tfb = pd.read_csv(tfb_bed, sep="\t", header=None, names=["chr", "start", "end", "tf", "score"])
        tfb["cre"] = tfb["chr"] + "-" + tfb["start"].astype(str) + "-" + tfb["end"].astype(str)
        tfb = tfb[["cre", "tf", "score"]].groupby(["cre", "tf"], as_index=False)["score"].mean()
    else:
        tfb = pd.DataFrame(columns=["cre", "tf", "score"])
    tfb.to_csv(tfb_csv, index=False)
    return tfb, qc_df


def _prepare_mdl_inputs(prepared_mdata_path: pathlib.Path, p2g_df: pd.DataFrame, tfb_df: pd.DataFrame, lineage_dir: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path]:
    mdata = mu.read(prepared_mdata_path)
    expr_path = lineage_dir / "mdl_expr.tsv.gz"
    peaks_path = lineage_dir / "mdl_peaks.tsv.gz"
    tfb_path = lineage_dir / "mdl_tfb.tsv.gz"

    _write_expr_tsv(mdata.mod["rna"], expr_path)

    if p2g_df.empty:
        peaks = pd.Index(mdata.mod["atac"].var_names.astype(str)).str.replace("-", ":", n=1, regex=False)
    else:
        peaks_order: list[str] = []
        peaks_seen: set[str] = set()
        for cre in p2g_df["cre"].drop_duplicates().astype(str):
            loc = _canonical_mdl_peak_loc(str(cre))
            if loc not in peaks_seen:
                peaks_seen.add(loc)
                peaks_order.append(loc)
        peaks = pd.Index(peaks_order)
    pd.DataFrame({"placeholder": np.zeros(peaks.size)}, index=peaks).to_csv(peaks_path, sep="\t", compression="gzip")

    tfb = tfb_df.copy()
    if tfb.empty:
        tfb = pd.DataFrame(columns=["TF", "loc", "score"])
    else:
        tfb["cre"] = tfb["cre"].map(lambda x: _canonical_mdl_peak_loc(str(x)))
        tfb = tfb[tfb["tf"].isin(mdata.mod["rna"].var_names) & tfb["cre"].isin(peaks)]
        tfb = tfb.rename(columns={"tf": "TF", "cre": "loc"})[["TF", "loc", "score"]]
    tfb.to_csv(tfb_path, sep="\t", index=False)
    return expr_path, peaks_path, tfb_path


def _run_mdl(
    lineage_dir: pathlib.Path,
    expr_path: pathlib.Path,
    peaks_path: pathlib.Path,
    tfb_path: pathlib.Path,
    gene_bed: pathlib.Path,
    distance: int,
    n_p2g_links: int,
    threads: int,
    device: str,
    thr_score: float,
) -> pd.DataFrame:
    tssdist_path = lineage_dir / "tssdist.tsv.gz"
    linking_path = lineage_dir / "linking.tsv.gz"
    binlinking_path = lineage_dir / "binlinking.tsv.gz"
    net_weight_path = lineage_dir / "net_weight.tsv.gz"
    net_meanvar_path = lineage_dir / "net_meanvar.tsv.gz"
    net_covfactor_path = lineage_dir / "net_covfactor.tsv.gz"
    net_loss_path = lineage_dir / "net_loss.tsv.gz"
    net_stats_path = lineage_dir / "net_stats.tsv.gz"
    net_nweight_path = lineage_dir / "net_nweight.tsv.gz"

    if tfb_path.stat().st_size == 0:
        return pd.DataFrame(columns=["TF", "Target", "Score"])

    tfb_check = pd.read_csv(tfb_path, sep="\t")
    if tfb_check.empty:
        return pd.DataFrame(columns=["TF", "Target", "Score"])

    run_logged_command(["python", "-m", "dictys", "chromatin", "tssdist", "--cut", str(distance), str(expr_path), str(peaks_path), str(gene_bed), str(tssdist_path)])
    run_logged_command(["python", "-m", "dictys", "chromatin", "linking", str(tfb_path), str(tssdist_path), str(linking_path)])
    run_logged_command(["python", "-m", "dictys", "chromatin", "binlinking", str(linking_path), str(binlinking_path), str(n_p2g_links)])
    run_logged_command(["python", "-m", "dictys", "network", "reconstruct", "--device", device, "--nth", str(threads), str(expr_path), str(binlinking_path), str(net_weight_path), str(net_meanvar_path), str(net_covfactor_path), str(net_loss_path), str(net_stats_path)])
    run_logged_command(["python", "-m", "dictys", "network", "normalize", "--nth", str(threads), str(net_weight_path), str(net_meanvar_path), str(net_covfactor_path), str(net_nweight_path)])

    weights = pd.read_csv(net_nweight_path, sep="\t", index_col=0)
    mask = pd.read_csv(binlinking_path, sep="\t", index_col=0)
    mask = mask.loc[weights.index, weights.columns]
    idx_row, idx_col = np.where(mask.values)
    if idx_row.size == 0:
        return pd.DataFrame(columns=["TF", "Target", "Score"])

    edge_df = pd.DataFrame({
        "TF": weights.index.to_numpy()[idx_row],
        "Target": weights.columns.to_numpy()[idx_col],
        "Score": [weights.iat[row, col] for row, col in zip(idx_row, idx_col)],
    })
    edge_df["Score"] = edge_df["Score"].astype(float)
    edge_df = edge_df[edge_df["Score"].abs() > thr_score].sort_values("Score", ascending=False).reset_index(drop=True)
    return edge_df


def main(args: argparse.Namespace) -> None:
    benchmark_dir = args.dirPjtHome / "benchmark" / args.version
    tmp_save_dir = args.dirPjtHome / "tmp" / "dictys_fragments_wd" / args.version
    net_dir = benchmark_dir / "net"
    log_dir = benchmark_dir / "log"
    fig_dir = benchmark_dir / "fig"
    for path in [benchmark_dir, tmp_save_dir, net_dir, log_dir, fig_dir]:
        path.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        filename=log_dir / "DictysFragments.log",
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        filemode="w",
    )

    np.random.seed(args.seed)
    logging.info("Benchmark Version: %s with seed %s", args.version, args.seed)
    logging.info("Packages Version: %s", session_info.show())
    log_memory_usage()

    resources = _resolve_resources(args)
    fragment_specs = _resolve_fragment_specs(args)
    barcode_map = _load_barcode_map(args.barcode_map)
    input_data, input_type = _resolve_input_data(args.dirPjtHome, args.dataset)
    gene_selected = pd.read_csv(args.genelist, header=None)[0].astype(str)
    cell_selected = pd.read_csv(args.celllist, index_col=0)

    logging.info("Input data: %s (%s)", input_data, input_type)
    logging.info("Fragments: %s", [str(spec.path) for spec in fragment_specs])
    logging.info("Fragment samples: %s", [spec.sample for spec in fragment_specs])
    logging.info("Gene list: %s", args.genelist)
    logging.info("Cell list: %s", args.celllist)
    log_memory_usage()

    for i, lin in enumerate(cell_selected.columns):
        logging.info("(%d/%d) Cell State %s...", i + 1, len(cell_selected.columns), lin)
        step_start = time.time()
        lin_dir = tmp_save_dir / lin
        lin_dir.mkdir(parents=True, exist_ok=True)
        try:
            lineage_mdata = _load_lineage_mdata(input_data, input_type, lin, cell_selected, pd.Index(gene_selected))
            prepared_mdata = _prepare_preprocessed_mdata(lineage_mdata, lin_dir)
            prepared_mdata_path = lin_dir / "pre.h5mu"
            prepared_mdata.write(prepared_mdata_path)

            obs_lookup = _build_obs_lookup(prepared_mdata.mod["rna"].obs_names, fragment_specs, args.barcode_transform, args.sample_separator)
            p2g_df = _build_p2g(prepared_mdata_path, lin_dir / "p2g.csv", lin_dir, resources.gene_bed, args.distance)
            tfb_df, qc_df = _build_tfb(
                prepared_mdata_path=prepared_mdata_path,
                p2g_df=p2g_df,
                lineage=lin,
                lineage_dir=lin_dir,
                fragment_specs=fragment_specs,
                resources=resources,
                obs_lookup=obs_lookup,
                barcode_map=barcode_map,
                threads=args.threads,
            )
            expr_path, peaks_path, tfb_path = _prepare_mdl_inputs(prepared_mdata_path, p2g_df, tfb_df, lin_dir)
            edge_df = _run_mdl(
                lineage_dir=lin_dir,
                expr_path=expr_path,
                peaks_path=peaks_path,
                tfb_path=tfb_path,
                gene_bed=resources.gene_bed,
                distance=args.distance,
                n_p2g_links=args.n_p2g_links,
                threads=args.threads,
                device=args.device,
                thr_score=args.thr_score,
            )
            out_csv = net_dir / f"Dictys_{lin}.csv"
            edge_df.to_csv(out_csv, index=False)
            logging.info("Saved network to %s", out_csv)
            logging.info("Saved barcode QC to %s", lin_dir / "barcode_match_qc.csv")
            logging.info("Barcode QC summary: %s", qc_df.to_dict(orient="records"))
            logging.info("Finished %s in %.2f seconds", lin, time.time() - step_start)
            log_memory_usage()
        finally:
            if not args.save and lin_dir.exists():
                shutil.rmtree(lin_dir)

    logging.info("Dictys fragments baseline finished!")
    log_memory_usage()


if __name__ == "__main__":
    main(parse_args())
