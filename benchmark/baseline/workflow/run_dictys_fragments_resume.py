"""
Dictys fragments workflow with checkpoint-aware resume support.
"""

from __future__ import annotations

import argparse
import logging
import os
import pathlib
import shutil
import time
from dataclasses import dataclass
from typing import Iterable

import mudata as mu
import numpy as np
import pandas as pd
import session_info

from baseline_cli_utils import log_memory_usage, run_logged_command, str2bool
from run_dictys_fragments import (
    DictysResources,
    FragmentSpec,
    _build_obs_lookup,
    _build_p2g,
    _build_tfb,
    _load_barcode_map,
    _load_lineage_mdata,
    _prepare_mdl_inputs,
    _prepare_preprocessed_mdata,
    _resolve_fragment_specs,
    _resolve_input_data,
    _resolve_resources,
)

STAGE_PREPARE = "prepare"
STAGE_P2G = "p2g"
STAGE_TFB = "tfb"
STAGE_MDL_INPUTS = "mdl_inputs"
STAGE_MDL = "mdl"
STAGE_FINAL = "final"
ALL_STAGES = (
    STAGE_PREPARE,
    STAGE_P2G,
    STAGE_TFB,
    STAGE_MDL_INPUTS,
    STAGE_MDL,
    STAGE_FINAL,
)


@dataclass(frozen=True)
class LineageArtifacts:
    lineage: str
    lineage_dir: pathlib.Path
    net_dir: pathlib.Path

    @property
    def prepared_mdata(self) -> pathlib.Path:
        return self.lineage_dir / "pre.h5mu"

    @property
    def p2g_csv(self) -> pathlib.Path:
        return self.lineage_dir / "p2g.csv"

    @property
    def tfb_csv(self) -> pathlib.Path:
        return self.lineage_dir / "tfb.csv"

    @property
    def barcode_qc_csv(self) -> pathlib.Path:
        return self.lineage_dir / "barcode_match_qc.csv"

    @property
    def mdl_expr(self) -> pathlib.Path:
        return self.lineage_dir / "mdl_expr.tsv.gz"

    @property
    def mdl_peaks(self) -> pathlib.Path:
        return self.lineage_dir / "mdl_peaks.tsv.gz"

    @property
    def mdl_tfb(self) -> pathlib.Path:
        return self.lineage_dir / "mdl_tfb.tsv.gz"

    @property
    def tssdist(self) -> pathlib.Path:
        return self.lineage_dir / "tssdist.tsv.gz"

    @property
    def linking(self) -> pathlib.Path:
        return self.lineage_dir / "linking.tsv.gz"

    @property
    def binlinking(self) -> pathlib.Path:
        return self.lineage_dir / "binlinking.tsv.gz"

    @property
    def net_weight(self) -> pathlib.Path:
        return self.lineage_dir / "net_weight.tsv.gz"

    @property
    def net_meanvar(self) -> pathlib.Path:
        return self.lineage_dir / "net_meanvar.tsv.gz"

    @property
    def net_covfactor(self) -> pathlib.Path:
        return self.lineage_dir / "net_covfactor.tsv.gz"

    @property
    def net_loss(self) -> pathlib.Path:
        return self.lineage_dir / "net_loss.tsv.gz"

    @property
    def net_stats(self) -> pathlib.Path:
        return self.lineage_dir / "net_stats.tsv.gz"

    @property
    def net_nweight(self) -> pathlib.Path:
        return self.lineage_dir / "net_nweight.tsv.gz"

    @property
    def final_csv(self) -> pathlib.Path:
        return self.net_dir / f"Dictys_{self.lineage}.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Dictys from fragment files with checkpoint resume")
    parser.add_argument("-d", "--dataset", dest="dataset", type=str, required=True, help="Dataset key")
    parser.add_argument("-p", "--home", dest="dirPjtHome", type=pathlib.Path, required=True, help="Path to the project home directory")
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
    parser.add_argument("--fragments", dest="fragments", nargs="+", default=None, help="One or more fragment files")
    parser.add_argument("--fragment-samples", dest="fragment_samples", nargs="+", default=None, help="Sample label for each fragment file")
    parser.add_argument("--barcode-map", dest="barcode_map", type=pathlib.Path, default=None, help="Optional CSV mapping fragment barcodes to obs names")
    parser.add_argument(
        "--barcode-transform",
        dest="barcode_transform",
        choices=["none", "auto", "prefix_suffix"],
        default="auto",
        help="How to reconcile fragment barcodes with obs names",
    )
    parser.add_argument("--sample-separator", dest="sample_separator", type=str, default="#", help="Preferred separator between sample and barcode in obs names")
    parser.add_argument("--threads", dest="threads", type=int, default=8, help="Thread count for Dictys CLI steps")
    parser.add_argument("--device", dest="device", type=str, default="cpu", help="Device passed to dictys network reconstruct")
    parser.add_argument("--distance", dest="distance", type=int, default=250000, help="Distance cutoff used for p2g/tssdist")
    parser.add_argument("--n-p2g-links", dest="n_p2g_links", type=int, default=15, help="Links kept by dictys chromatin binlinking")
    parser.add_argument("--thr-score", dest="thr_score", type=float, default=0.0, help="Absolute score threshold for exported edges")
    parser.add_argument("--motifs", dest="motifs", type=pathlib.Path, default=None, help="Optional HOMER motif file")
    parser.add_argument("--homer-genome", dest="homer_genome", type=pathlib.Path, default=None, help="Optional HOMER genome directory")
    parser.add_argument("--gene-bed", dest="gene_bed", type=pathlib.Path, default=None, help="Optional gene annotation BED used by Dictys")
    parser.add_argument("--resume", dest="resume", type=str2bool, default=True, help="Resume from existing checkpoints when possible")
    parser.add_argument(
        "--resume-mode",
        dest="resume_mode",
        choices=["lineage", "stage"],
        default="stage",
        help="Skip either complete lineages or individual stages",
    )
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
        STAGE_P2G: (artifacts.p2g_csv,),
        STAGE_TFB: (artifacts.tfb_csv, artifacts.barcode_qc_csv),
        STAGE_MDL_INPUTS: (artifacts.mdl_expr, artifacts.mdl_peaks, artifacts.mdl_tfb),
        STAGE_MDL: (
            artifacts.tssdist,
            artifacts.linking,
            artifacts.binlinking,
            artifacts.net_weight,
            artifacts.net_meanvar,
            artifacts.net_covfactor,
            artifacts.net_loss,
            artifacts.net_stats,
            artifacts.net_nweight,
        ),
        STAGE_FINAL: (artifacts.final_csv,),
    }
    return mapping[stage]


def outputs_ready(paths: Iterable[pathlib.Path]) -> bool:
    return all(path.exists() for path in paths)


def discover_dictys_bin_dir() -> pathlib.Path | None:
    if shutil.which("wellington_footprints.py") is not None:
        return None

    try:
        import dictys
    except ImportError:
        return None

    dictys_path = pathlib.Path(dictys.__file__).resolve()
    for parent in dictys_path.parents:
        candidate = parent / "bin" / "wellington_footprints.py"
        if candidate.exists():
            return candidate.parent
    return None


def ensure_wellington_on_path() -> pathlib.Path | None:
    existing = shutil.which("wellington_footprints.py")
    if existing is not None:
        logging.info("wellington_footprints.py already on PATH: %s", existing)
        return pathlib.Path(existing).parent

    dictys_bin_dir = discover_dictys_bin_dir()
    if dictys_bin_dir is None:
        logging.warning("Could not discover dictys bin directory for wellington_footprints.py")
        return None

    path_parts = os.environ.get("PATH", "").split(os.pathsep) if os.environ.get("PATH") else []
    if str(dictys_bin_dir) not in path_parts:
        os.environ["PATH"] = os.pathsep.join([str(dictys_bin_dir), *path_parts]) if path_parts else str(dictys_bin_dir)
        logging.info("Prepended dictys bin directory to PATH: %s", dictys_bin_dir)
    return dictys_bin_dir


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


def should_skip_lineage(
    *,
    artifacts: LineageArtifacts,
    resume: bool,
    resume_mode: str,
    force_rebuild: set[str],
) -> bool:
    if not resume or resume_mode != "lineage":
        return False
    if force_rebuild:
        return False
    return artifacts.final_csv.exists()


def log_stage_decision(lineage: str, stage: str, action: str, reason: str) -> None:
    logging.info("[%s] %s %s (%s)", lineage, stage.upper(), action.upper(), reason)


def read_optional_csv(path: pathlib.Path, columns: list[str]) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame(columns=columns)


def collect_edge_df(net_nweight_path: pathlib.Path, binlinking_path: pathlib.Path, thr_score: float) -> pd.DataFrame:
    weights = pd.read_csv(net_nweight_path, sep="\t", index_col=0)
    mask = pd.read_csv(binlinking_path, sep="\t", index_col=0)
    mask = mask.loc[weights.index, weights.columns]
    idx_row, idx_col = np.where(mask.values)
    if idx_row.size == 0:
        return pd.DataFrame(columns=["TF", "Target", "Score"])
    edge_df = pd.DataFrame(
        {
            "TF": weights.index.to_numpy()[idx_row],
            "Target": weights.columns.to_numpy()[idx_col],
            "Score": [weights.iat[row, col] for row, col in zip(idx_row, idx_col)],
        }
    )
    edge_df["Score"] = edge_df["Score"].astype(float)
    return edge_df[edge_df["Score"].abs() > thr_score].sort_values("Score", ascending=False).reset_index(drop=True)


def run_mdl_with_resume(
    *,
    artifacts: LineageArtifacts,
    expr_path: pathlib.Path,
    peaks_path: pathlib.Path,
    tfb_path: pathlib.Path,
    gene_bed: pathlib.Path,
    distance: int,
    n_p2g_links: int,
    threads: int,
    device: str,
    thr_score: float,
    resume_substeps: bool,
) -> pd.DataFrame:
    if tfb_path.stat().st_size == 0:
        return pd.DataFrame(columns=["TF", "Target", "Score"])

    tfb_check = pd.read_csv(tfb_path, sep="\t")
    if tfb_check.empty:
        return pd.DataFrame(columns=["TF", "Target", "Score"])

    mdl_steps = [
        (
            "mdl.tssdist",
            (artifacts.tssdist,),
            [
                "python",
                "-m",
                "dictys",
                "chromatin",
                "tssdist",
                "--cut",
                str(distance),
                str(expr_path),
                str(peaks_path),
                str(gene_bed),
                str(artifacts.tssdist),
            ],
        ),
        (
            "mdl.linking",
            (artifacts.linking,),
            ["python", "-m", "dictys", "chromatin", "linking", str(tfb_path), str(artifacts.tssdist), str(artifacts.linking)],
        ),
        (
            "mdl.binlinking",
            (artifacts.binlinking,),
            ["python", "-m", "dictys", "chromatin", "binlinking", str(artifacts.linking), str(artifacts.binlinking), str(n_p2g_links)],
        ),
        (
            "mdl.reconstruct",
            (
                artifacts.net_weight,
                artifacts.net_meanvar,
                artifacts.net_covfactor,
                artifacts.net_loss,
                artifacts.net_stats,
            ),
            [
                "python",
                "-m",
                "dictys",
                "network",
                "reconstruct",
                "--device",
                device,
                "--nth",
                str(threads),
                str(expr_path),
                str(artifacts.binlinking),
                str(artifacts.net_weight),
                str(artifacts.net_meanvar),
                str(artifacts.net_covfactor),
                str(artifacts.net_loss),
                str(artifacts.net_stats),
            ],
        ),
        (
            "mdl.normalize",
            (artifacts.net_nweight,),
            [
                "python",
                "-m",
                "dictys",
                "network",
                "normalize",
                "--nth",
                str(threads),
                str(artifacts.net_weight),
                str(artifacts.net_meanvar),
                str(artifacts.net_covfactor),
                str(artifacts.net_nweight),
            ],
        ),
    ]

    for step_name, outputs, cmd in mdl_steps:
        if resume_substeps and outputs_ready(outputs):
            logging.info("[%s] %s SKIP (existing substep outputs)", artifacts.lineage, step_name)
            continue
        logging.info("[%s] %s RUN", artifacts.lineage, step_name)
        run_logged_command(cmd)

    return collect_edge_df(artifacts.net_nweight, artifacts.binlinking, thr_score)


def main(args: argparse.Namespace) -> None:
    force_rebuild = normalize_force_rebuild(args.force_rebuild)

    benchmark_dir = args.dirPjtHome / "benchmark" / args.version
    tmp_save_dir = args.dirPjtHome / "tmp" / "dictys_fragments_wd" / args.version
    net_dir = benchmark_dir / "net"
    log_dir = benchmark_dir / "log"
    fig_dir = benchmark_dir / "fig"
    for path in [benchmark_dir, tmp_save_dir, net_dir, log_dir, fig_dir]:
        path.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        filename=log_dir / "DictysFragmentsResume.log",
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
    ensure_wellington_on_path()

    resources: DictysResources = _resolve_resources(args)
    fragment_specs: list[FragmentSpec] = _resolve_fragment_specs(args)
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
                prepared_mdata = mu.read(artifacts.prepared_mdata)
            else:
                lineage_mdata = _load_lineage_mdata(input_data, input_type, lineage, cell_selected, pd.Index(gene_selected))
                prepared_mdata = _prepare_preprocessed_mdata(lineage_mdata, artifacts.lineage_dir)
                prepared_mdata.write(artifacts.prepared_mdata)
                upstream_changed = True

            obs_lookup = _build_obs_lookup(
                prepared_mdata.mod["rna"].obs_names,
                fragment_specs,
                args.barcode_transform,
                args.sample_separator,
            )

            p2g_action = decide_stage_action(
                stage=STAGE_P2G,
                outputs_exist=outputs_ready(stage_outputs(artifacts, STAGE_P2G)),
                resume=lineage_resume,
                upstream_changed=upstream_changed,
                force_rebuild=force_rebuild,
            )
            log_stage_decision(lineage, STAGE_P2G, p2g_action, "checkpoint inspection")
            if p2g_action == "skip":
                p2g_df = pd.read_csv(artifacts.p2g_csv)
            else:
                p2g_df = _build_p2g(artifacts.prepared_mdata, artifacts.p2g_csv, artifacts.lineage_dir, resources.gene_bed, args.distance)
                upstream_changed = True

            tfb_action = decide_stage_action(
                stage=STAGE_TFB,
                outputs_exist=outputs_ready(stage_outputs(artifacts, STAGE_TFB)),
                resume=lineage_resume,
                upstream_changed=upstream_changed,
                force_rebuild=force_rebuild,
            )
            log_stage_decision(lineage, STAGE_TFB, tfb_action, "checkpoint inspection")
            if tfb_action == "skip":
                tfb_df = pd.read_csv(artifacts.tfb_csv)
                qc_df = read_optional_csv(artifacts.barcode_qc_csv, ["fragment_path", "sample", "total_fragment_rows", "kept_fragment_counts", "matched_cells"])
            else:
                tfb_df, qc_df = _build_tfb(
                    prepared_mdata_path=artifacts.prepared_mdata,
                    p2g_df=p2g_df,
                    lineage=lineage,
                    lineage_dir=artifacts.lineage_dir,
                    fragment_specs=fragment_specs,
                    resources=resources,
                    obs_lookup=obs_lookup,
                    barcode_map=barcode_map,
                    threads=args.threads,
                )
                upstream_changed = True

            mdl_inputs_action = decide_stage_action(
                stage=STAGE_MDL_INPUTS,
                outputs_exist=outputs_ready(stage_outputs(artifacts, STAGE_MDL_INPUTS)),
                resume=lineage_resume,
                upstream_changed=upstream_changed,
                force_rebuild=force_rebuild,
            )
            log_stage_decision(lineage, STAGE_MDL_INPUTS, mdl_inputs_action, "checkpoint inspection")
            if mdl_inputs_action == "skip":
                expr_path, peaks_path, tfb_path = artifacts.mdl_expr, artifacts.mdl_peaks, artifacts.mdl_tfb
            else:
                expr_path, peaks_path, tfb_path = _prepare_mdl_inputs(artifacts.prepared_mdata, p2g_df, tfb_df, artifacts.lineage_dir)
                upstream_changed = True

            mdl_action = decide_stage_action(
                stage=STAGE_MDL,
                outputs_exist=outputs_ready(stage_outputs(artifacts, STAGE_MDL)),
                resume=lineage_resume,
                upstream_changed=upstream_changed,
                force_rebuild=force_rebuild,
            )
            log_stage_decision(lineage, STAGE_MDL, mdl_action, "checkpoint inspection")
            if mdl_action == "skip":
                edge_df = collect_edge_df(artifacts.net_nweight, artifacts.binlinking, args.thr_score)
            else:
                edge_df = run_mdl_with_resume(
                    artifacts=artifacts,
                    expr_path=expr_path,
                    peaks_path=peaks_path,
                    tfb_path=tfb_path,
                    gene_bed=resources.gene_bed,
                    distance=args.distance,
                    n_p2g_links=args.n_p2g_links,
                    threads=args.threads,
                    device=args.device,
                    thr_score=args.thr_score,
                    resume_substeps=lineage_resume and mdl_action == "run",
                )
                upstream_changed = True

            final_action = decide_stage_action(
                stage=STAGE_FINAL,
                outputs_exist=outputs_ready(stage_outputs(artifacts, STAGE_FINAL)),
                resume=lineage_resume,
                upstream_changed=upstream_changed,
                force_rebuild=force_rebuild,
            )
            log_stage_decision(lineage, STAGE_FINAL, final_action, "checkpoint inspection")
            if final_action != "skip":
                edge_df.to_csv(artifacts.final_csv, index=False)

            logging.info("Saved network to %s", artifacts.final_csv)
            logging.info("Saved barcode QC to %s", artifacts.barcode_qc_csv)
            logging.info("Barcode QC summary: %s", qc_df.to_dict(orient="records"))
            logging.info("Finished %s in %.2f seconds", lineage, time.time() - step_start)
            log_memory_usage()
        finally:
            if not args.save and artifacts.lineage_dir.exists():
                shutil.rmtree(artifacts.lineage_dir)

    logging.info("Dictys fragments resume baseline finished!")
    log_memory_usage()


if __name__ == "__main__":
    main(parse_args())
