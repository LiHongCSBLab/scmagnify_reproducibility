"""Preflight checks for the baseline workflow."""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import subprocess
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Callable

import pandas as pd


def _truthy_mask(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    lowered = series.astype(str).str.strip().str.lower()
    return lowered.isin({"true", "1", "yes", "y"})


def _load_yaml(path: pathlib.Path) -> dict[str, Any]:
    try:
        import yaml

        data = yaml.safe_load(path.read_text())
    except ModuleNotFoundError:
        if shutil.which("yq") is None:
            raise RuntimeError("PyYAML is unavailable and yq is not on PATH")
        result = subprocess.run(
            ["yq", "-o=json", ".", str(path)],
            check=True,
            text=True,
            capture_output=True,
        )
        data = json.loads(result.stdout)
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {path}")
    return data


def _resolve_method_script(root_dir: pathlib.Path, script: str) -> pathlib.Path:
    script_path = pathlib.Path(script)
    if not script_path.is_absolute():
        script_path = root_dir / script_path.as_posix().removeprefix("./")
    return script_path


def _normalize_method_args(args: list[str]) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    idx = 0
    while idx < len(args):
        token = args[idx]
        if not token.startswith("--"):
            idx += 1
            continue
        key = token[2:].replace("-", "_")
        values: list[str] = []
        idx += 1
        while idx < len(args) and not args[idx].startswith("--"):
            values.append(args[idx])
            idx += 1
        value: Any
        if not values:
            value = True
        elif len(values) == 1:
            value = values[0]
        else:
            value = values
        if key in parsed:
            existing = parsed[key]
            if not isinstance(existing, list):
                existing = [existing]
            if isinstance(value, list):
                existing.extend(value)
            else:
                existing.append(value)
            parsed[key] = existing
        else:
            parsed[key] = value
    return parsed


def _as_path(value: Any) -> pathlib.Path | None:
    if value in (None, "", False):
        return None
    return pathlib.Path(str(value))


def _as_path_list(value: Any) -> list[pathlib.Path] | None:
    if value in (None, "", False):
        return None
    if isinstance(value, list):
        return [pathlib.Path(str(item)) for item in value]
    return [pathlib.Path(str(value))]


def _as_str_list(value: Any) -> list[str] | None:
    if value in (None, "", False):
        return None
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _logical_modality_present(keys: set[str], logical_name: str) -> bool:
    lowered = {key.lower() for key in keys}
    if logical_name == "RNA":
        return any(candidate in lowered for candidate in {"rna", "scrna"})
    if logical_name == "ATAC":
        return any(candidate in lowered for candidate in {"atac", "scatac"})
    return logical_name.lower() in lowered


def _is_optional_inspection_error(exc: Exception) -> bool:
    return isinstance(exc, ImportError) or "is required to inspect" in str(exc)


@dataclass(frozen=True)
class MethodConfig:
    name: str
    env: str
    script: pathlib.Path
    args: list[str]


@dataclass(frozen=True)
class BaselineConfig:
    config_path: pathlib.Path
    root_dir: pathlib.Path
    dataset: str
    home_dir: pathlib.Path
    cell_list: pathlib.Path
    gene_list: pathlib.Path
    ref_genome: str
    version: str
    tmp_save: bool
    seed: int
    methods: list[MethodConfig]


@dataclass
class DatasetInspection:
    input_type: str
    path: pathlib.Path
    obs_names: pd.Index
    var_names: pd.Index
    obs_columns: set[str]
    obsm_keys: set[str]
    modality_keys: set[str] = field(default_factory=set)


@dataclass
class MethodCheckResult:
    status: str = "ok"
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    required_inputs: list[str] = field(default_factory=list)
    resolved_inputs: list[str] = field(default_factory=list)

    def skip(self, reason: str) -> None:
        self.status = "skip"
        self.reasons.append(reason)

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reasons": self.reasons,
            "warnings": self.warnings,
            "required_inputs": self.required_inputs,
            "resolved_inputs": self.resolved_inputs,
        }


@dataclass
class PreflightResult:
    config_path: str
    dataset: str
    global_errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    method_status: dict[str, MethodCheckResult] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.global_errors

    def to_dict(self) -> dict[str, Any]:
        runnable = sorted(name for name, info in self.method_status.items() if info.status != "skip")
        skipped = sorted(name for name, info in self.method_status.items() if info.status == "skip")
        return {
            "ok": self.ok,
            "config_path": self.config_path,
            "dataset": self.dataset,
            "global_errors": self.global_errors,
            "warnings": self.warnings,
            "method_status": {name: info.to_dict() for name, info in self.method_status.items()},
            "methods_to_run": runnable,
            "methods_to_skip": skipped,
        }


@dataclass(frozen=True)
class MethodProfile:
    supported_inputs: tuple[str, ...]
    required_obs_columns: tuple[str, ...] = ()
    required_obsm_keys: tuple[str, ...] = ()
    required_modalities: tuple[str, ...] = ()
    checker: Callable[["PreflightContext", MethodConfig, MethodCheckResult], None] | None = None


@dataclass
class PreflightContext:
    config: BaselineConfig
    available_inputs: dict[str, pathlib.Path]
    inspection_cache: dict[str, DatasetInspection] = field(default_factory=dict)

    def find_input(self, supported_inputs: tuple[str, ...]) -> tuple[str, pathlib.Path] | None:
        for input_type in supported_inputs:
            path = self.available_inputs.get(input_type)
            if path is not None:
                return input_type, path
        return None

    def get_inspection(self, input_type: str) -> DatasetInspection:
        if input_type in self.inspection_cache:
            return self.inspection_cache[input_type]
        path = self.available_inputs.get(input_type)
        if path is None:
            raise FileNotFoundError(f"Input file for {input_type} does not exist")
        if input_type == "h5ad":
            try:
                import anndata as ad
            except ModuleNotFoundError as exc:
                raise RuntimeError("anndata is required to inspect .h5ad inputs") from exc
            adata = ad.read_h5ad(path)
            inspection = DatasetInspection(
                input_type=input_type,
                path=path,
                obs_names=pd.Index(adata.obs_names.copy()),
                var_names=pd.Index(adata.var_names.copy()),
                obs_columns=set(adata.obs.columns.astype(str)),
                obsm_keys=set(adata.obsm.keys()),
            )
        elif input_type == "h5mu":
            try:
                import mudata as mu
            except ModuleNotFoundError as exc:
                raise RuntimeError("mudata is required to inspect .h5mu inputs") from exc
            mdata = mu.read(path)
            modality_keys = set(mdata.mod.keys())
            rna_key = next((key for key in ["RNA", "rna", "scRNA"] if key in mdata.mod), None)
            if rna_key is None:
                raise ValueError(f"Unable to find RNA modality in {path}")
            rna = mdata.mod[rna_key]
            inspection = DatasetInspection(
                input_type=input_type,
                path=path,
                obs_names=pd.Index(rna.obs_names.copy()),
                var_names=pd.Index(rna.var_names.copy()),
                obs_columns=set(rna.obs.columns.astype(str)),
                obsm_keys=set(rna.obsm.keys()),
                modality_keys=modality_keys,
            )
        else:
            raise ValueError(f"Unsupported inspection input type: {input_type}")
        self.inspection_cache[input_type] = inspection
        return inspection


def load_baseline_config(config_path: pathlib.Path, root_dir: pathlib.Path) -> BaselineConfig:
    raw = _load_yaml(config_path)
    methods: list[MethodConfig] = []
    for method_raw in raw.get("methods") or []:
        if not isinstance(method_raw, dict):
            raise ValueError("Each item under methods must be a mapping")
        script = str(method_raw.get("script") or "")
        methods.append(
            MethodConfig(
                name=str(method_raw.get("name") or ""),
                env=str(method_raw.get("env") or ""),
                script=_resolve_method_script(root_dir, script) if script else pathlib.Path(),
                args=[str(item) for item in (method_raw.get("args") or [])],
            )
        )
    return BaselineConfig(
        config_path=config_path,
        root_dir=root_dir,
        dataset=str(raw.get("dataset") or ""),
        home_dir=pathlib.Path(str(raw.get("home") or "")),
        cell_list=pathlib.Path(str(raw.get("cell") or "")),
        gene_list=pathlib.Path(str(raw.get("gene") or "")),
        ref_genome=str(raw.get("ref-genome") or ""),
        version=str(raw.get("version") or ""),
        tmp_save=bool(raw.get("tmp-save", False)),
        seed=int(raw.get("seed", 0)),
        methods=methods,
    )


def _check_required_config(config: BaselineConfig, result: PreflightResult) -> None:
    if not config.dataset:
        result.global_errors.append("config key `dataset` is missing or empty")
    if not str(config.home_dir):
        result.global_errors.append("config key `home` is missing or empty")
    if not str(config.cell_list):
        result.global_errors.append("config key `cell` is missing or empty")
    if not str(config.gene_list):
        result.global_errors.append("config key `gene` is missing or empty")
    if not config.ref_genome:
        result.global_errors.append("config key `ref-genome` is missing or empty")
    if not config.methods:
        result.global_errors.append("config key `methods` is empty")
        return

    method_names = [method.name for method in config.methods]
    duplicate_names = pd.Index(method_names)[pd.Index(method_names).duplicated()].unique().tolist()
    if duplicate_names:
        result.global_errors.append("Duplicate method names in config: " + ", ".join(map(str, duplicate_names)))

    if str(config.home_dir) and not config.home_dir.is_dir():
        result.global_errors.append(f"Home directory does not exist: {config.home_dir}")
    if str(config.cell_list) and not config.cell_list.is_file():
        result.global_errors.append(f"Cell list file does not exist: {config.cell_list}")
    if str(config.gene_list) and not config.gene_list.is_file():
        result.global_errors.append(f"Gene list file does not exist: {config.gene_list}")

    for method in config.methods:
        if not method.name:
            result.global_errors.append("A method entry is missing `name`")
        if not method.env:
            result.global_errors.append(f"Method `{method.name or '<unknown>'}` is missing `env`")
        if not str(method.script):
            result.global_errors.append(f"Method `{method.name or '<unknown>'}` is missing `script`")
        elif not method.script.is_file():
            result.global_errors.append(f"Method script does not exist for `{method.name}`: {method.script}")


def _read_cell_list(path: pathlib.Path) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0)


def _read_gene_list(path: pathlib.Path) -> pd.Index:
    genes = pd.read_csv(path, header=None)
    if genes.shape[1] == 0:
        return pd.Index([])
    return pd.Index(genes.iloc[:, 0].astype(str))


def _check_cell_and_gene_lists(config: BaselineConfig, result: PreflightResult) -> tuple[pd.DataFrame | None, pd.Index | None]:
    cell_df: pd.DataFrame | None = None
    gene_idx: pd.Index | None = None
    if config.cell_list.is_file():
        try:
            cell_df = _read_cell_list(config.cell_list)
        except Exception as exc:
            result.global_errors.append(f"Failed to read cell list `{config.cell_list}`: {exc}")
        else:
            if cell_df.empty:
                result.global_errors.append(f"Cell list is empty: {config.cell_list}")
            if cell_df.index.hasnans:
                result.global_errors.append("Cell list contains missing cell IDs in the index")
            if cell_df.index.duplicated().any():
                duplicates = cell_df.index[cell_df.index.duplicated()].unique().tolist()
                result.global_errors.append("Cell list contains duplicated cell IDs: " + ", ".join(map(str, duplicates[:10])))
            if cell_df.columns.duplicated().any():
                duplicates = cell_df.columns[cell_df.columns.duplicated()].unique().tolist()
                result.global_errors.append("Cell list contains duplicated lineage columns: " + ", ".join(map(str, duplicates[:10])))
            if cell_df.shape[1] == 0:
                result.global_errors.append("Cell list does not contain any lineage columns")
            else:
                truthy = cell_df.apply(_truthy_mask)
                if int(truthy.values.sum()) == 0:
                    result.global_errors.append("Cell list does not select any cells")
                for lineage in truthy.columns:
                    if int(truthy[lineage].sum()) == 0:
                        result.global_errors.append(f"Lineage `{lineage}` does not select any cells")

    if config.gene_list.is_file():
        try:
            gene_idx = _read_gene_list(config.gene_list)
        except Exception as exc:
            result.global_errors.append(f"Failed to read gene list `{config.gene_list}`: {exc}")
        else:
            if gene_idx.empty:
                result.global_errors.append(f"Gene list is empty: {config.gene_list}")
            if gene_idx.hasnans:
                result.global_errors.append("Gene list contains missing gene names")
            blank_mask = gene_idx.astype(str).str.strip() == ""
            if bool(blank_mask.any()):
                result.global_errors.append("Gene list contains blank gene names")
            if gene_idx.duplicated().any():
                duplicates = gene_idx[gene_idx.duplicated()].unique().tolist()
                result.global_errors.append("Gene list contains duplicated genes: " + ", ".join(map(str, duplicates[:10])))
    return cell_df, gene_idx


def _check_dataset_consistency(
    context: PreflightContext,
    result: PreflightResult,
    cell_df: pd.DataFrame | None,
    gene_idx: pd.Index | None,
) -> None:
    preferred = context.find_input(("h5mu", "h5ad"))
    if preferred is None or cell_df is None or gene_idx is None:
        return

    try:
        inspection = context.get_inspection(preferred[0])
    except Exception as exc:
        message = f"Failed to inspect dataset file `{preferred[1]}`: {exc}"
        if _is_optional_inspection_error(exc):
            result.warnings.append(message)
        else:
            result.global_errors.append(message)
        return

    cell_index = pd.Index(cell_df.index.astype(str))
    cell_overlap = cell_index.intersection(inspection.obs_names)
    if cell_overlap.empty:
        result.global_errors.append(
            f"Cell list has zero overlap with dataset obs names in `{inspection.path}`"
        )
    elif len(cell_overlap) < len(cell_index):
        missing_count = len(cell_index) - len(cell_overlap)
        result.warnings.append(
            f"Cell list contains {missing_count} cells absent from dataset `{inspection.path.name}`"
        )

    truthy = cell_df.apply(_truthy_mask)
    for lineage in truthy.columns:
        selected_cells = pd.Index(truthy.index[truthy[lineage]].astype(str))
        matched = selected_cells.intersection(inspection.obs_names)
        if selected_cells.empty:
            result.global_errors.append(f"Lineage `{lineage}` has no selected cells")
        elif matched.empty:
            result.global_errors.append(
                f"Lineage `{lineage}` has zero overlap with dataset obs names in `{inspection.path.name}`"
            )
        elif len(matched) < len(selected_cells):
            result.warnings.append(
                f"Lineage `{lineage}` has {len(selected_cells) - len(matched)} selected cells absent from dataset `{inspection.path.name}`"
            )

    gene_overlap = gene_idx.intersection(inspection.var_names)
    if gene_overlap.empty:
        result.global_errors.append(
            f"Gene list has zero overlap with dataset features in `{inspection.path}`"
        )
    elif len(gene_overlap) < len(gene_idx):
        result.warnings.append(
            f"Gene list contains {len(gene_idx) - len(gene_overlap)} genes absent from dataset `{inspection.path.name}`"
        )


def _check_existing_paths(paths: list[pathlib.Path], result: MethodCheckResult, label: str) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        result.skip(f"Missing {label}:\n" + "\n".join(missing))


def _check_scenicplus(context: PreflightContext, method: MethodConfig, result: MethodCheckResult) -> None:
    args = _normalize_method_args(method.args)
    try:
        from run_scenicplus import _resolve_reference_resources, _resolve_region_set_root
    except Exception as exc:
        result.skip(f"Unable to import SCENIC+ resolver helpers: {exc}")
        return

    try:
        resources = _resolve_reference_resources(
            context.config.ref_genome,
            _as_path(args.get("scenicplus_db_root")),
            context.config.home_dir,
        )
        region_root = _resolve_region_set_root(
            context.config.home_dir,
            context.config.dataset,
            _as_path(args.get("region_set_root")),
        )
    except Exception as exc:
        result.skip(str(exc))
        return

    result.resolved_inputs.extend(
        [
            str(resources.annotation_tsv),
            str(resources.chromsizes_tsv),
            str(resources.rankings_feather),
            str(resources.motif_annotation_tsv),
            str(region_root),
        ]
    )


def _check_dictys_fragments(context: PreflightContext, method: MethodConfig, result: MethodCheckResult) -> None:
    args = _normalize_method_args(method.args)
    try:
        from run_dictys_fragments import _load_barcode_map, _resolve_fragment_specs, _resolve_resources
    except Exception as exc:
        result.skip(f"Unable to import DictysFragments resolver helpers: {exc}")
        return

    namespace = SimpleNamespace(
        refGenome=context.config.ref_genome,
        homer_genome=_as_path(args.get("homer_genome")),
        motifs=_as_path(args.get("motifs")),
        gene_bed=_as_path(args.get("gene_bed")),
        fragments=[str(path) for path in (_as_path_list(args.get("fragments")) or [])] or None,
        fragment_samples=_as_str_list(args.get("fragment_samples")),
        dirPjtHome=context.config.home_dir,
        dataset=context.config.dataset,
        barcode_map=_as_path(args.get("barcode_map")),
    )
    try:
        resources = _resolve_resources(namespace)
        fragment_specs = _resolve_fragment_specs(namespace)
        _load_barcode_map(namespace.barcode_map)
    except Exception as exc:
        result.skip(str(exc))
        return

    result.resolved_inputs.extend(
        [
            str(resources.homer_genome),
            str(resources.motifs),
            str(resources.gene_bed),
            *[str(spec.path) for spec in fragment_specs],
        ]
    )
    if namespace.barcode_map is not None:
        result.resolved_inputs.append(str(namespace.barcode_map))


def _check_dictys(context: PreflightContext, method: MethodConfig, result: MethodCheckResult) -> None:
    if context.config.ref_genome == "hg38":
        paths = [
            pathlib.Path("/home/chenxufeng/picb_cxf/Ref/human/hg38/homer_genome"),
            pathlib.Path("/home/chenxufeng/picb_cxf/Database/motif_databases/HOCOMOCOv11_full_HUMAN_mono_homer_format_0.0001.motif"),
            pathlib.Path("/home/chenxufeng/picb_cxf/Ref/human/hg38/annotations/ucsc/gene.bed"),
        ]
    elif context.config.ref_genome == "mm10":
        paths = [
            pathlib.Path("/home/chenxufeng/picb_cxf/Ref/mouse/mm10/homer_genome"),
            pathlib.Path("/home/chenxufeng/picb_cxf/Database/motif_databases/HOCOMOCOv11_full_MOUSE_mono_homer_format_0.0001.motif"),
            pathlib.Path("/home/chenxufeng/picb_cxf/Ref/mouse/mm10/annotations/ucsc/gene.bed"),
        ]
    else:
        result.skip(f"Unsupported reference genome for Dictys: {context.config.ref_genome}")
        return
    _check_existing_paths(paths, result, "Dictys reference resources")
    result.resolved_inputs.extend(str(path) for path in paths)
    result.warn("Per-cell BAM symlink targets under tmp/dictys_wd are runtime-derived and are not exhaustively prechecked.")


def _check_linger(context: PreflightContext, method: MethodConfig, result: MethodCheckResult) -> None:
    grn_dir = pathlib.Path("/home/chenxufeng/picb_cxf/Data/10x_Genomics/PBMCs_10k_scMultiome/data_bulk/")
    if context.config.ref_genome != "hg38":
        result.skip("run_linger.py is only preflight-approved for hg38 because it hardcodes Human training resources.")
        return
    _check_existing_paths([grn_dir], result, "LINGER reference directory")
    result.resolved_inputs.append(str(grn_dir))


def _check_velorama(context: PreflightContext, method: MethodConfig, result: MethodCheckResult) -> None:
    tf_list = pathlib.Path(f"/home/chenxufeng/picb_cxf/Ref/tflists/cistarget/allTFs_{context.config.ref_genome}.txt")
    _check_existing_paths([tf_list], result, "Velorama TF list")
    result.resolved_inputs.append(str(tf_list))


def _check_celloracle(context: PreflightContext, method: MethodConfig, result: MethodCheckResult) -> None:
    base_grn = context.config.home_dir / "tmp" / "celloracle_wd" / "base_GRN_dataframe.parquet"
    _check_existing_paths([base_grn], result, "CellOracle base GRN parquet")
    result.resolved_inputs.append(str(base_grn))


def _check_scenic(context: PreflightContext, method: MethodConfig, result: MethodCheckResult) -> None:
    scenic_driver = pathlib.Path("/home/chenxufeng/WorkSpace/scMagnify/scMagnify-benchmark/baseline/workflow/scenic_grn_ctx.sh")
    db_root = pathlib.Path("/home/chenxufeng/picb_cxf/Ref")
    if context.config.ref_genome == "hg38":
        db_dir = db_root / "human" / "hg38" / "cisTarget_db"
        files = [
            db_dir / "allTFs_hg38.txt",
            db_dir / "hg38_500bp_up_100bp_down_full_tx_v10_clust.genes_vs_motifs.scores.feather",
            db_dir / "hg38_10kbp_up_10kbp_down_full_tx_v10_clust.genes_vs_motifs.rankings.feather",
            db_dir / "hg38_500bp_up_100bp_down_full_tx_v10_clust.genes_vs_motifs.rankings.feather",
            db_dir / "motifs-v10-nr.hgnc-m0.00001-o0.0.tbl",
        ]
    elif context.config.ref_genome == "mm10":
        db_dir = db_root / "mouse" / "mm10" / "cisTarget_db"
        files = [
            db_dir / "allTFs_mm10.txt",
            db_dir / "mm10_500bp_up_100bp_down_full_tx_v10_clust.genes_vs_motifs.scores.feather",
            db_dir / "mm10_10kbp_up_10kbp_down_full_tx_v10_clust.genes_vs_motifs.rankings.feather",
            db_dir / "mm10_500bp_up_100bp_down_full_tx_v10_clust.genes_vs_motifs.rankings.feather",
            db_dir / "motifs-v10-nr.mgi-m0.00001-o0.0.tbl",
        ]
    else:
        result.skip(f"Unsupported reference genome for SCENIC: {context.config.ref_genome}")
        return
    _check_existing_paths([scenic_driver, *files], result, "SCENIC resources")
    result.resolved_inputs.extend([str(scenic_driver), *[str(path) for path in files]])


def _check_sincerities(context: PreflightContext, method: MethodConfig, result: MethodCheckResult) -> None:
    script = pathlib.Path("/home/chenxufeng/WorkSpace/scMagnify/scMagnify-benchmark/baseline/workflow/SINCERITIES.R")
    _check_existing_paths([script], result, "SINCERITIES helper script")
    result.resolved_inputs.append(str(script))
    result.warn("run_sincerities.R also expects `palantir_pseudotime` inside the RDS object, which is not statically inspected here.")


def _check_pando(context: PreflightContext, method: MethodConfig, result: MethodCheckResult) -> None:
    if context.config.ref_genome == "mm10":
        result.skip("run_pando.R hardcodes hg38 motif scanning and is not safe for mm10 datasets.")
        return
    if context.config.ref_genome != "hg38":
        result.skip(f"Unsupported reference genome for Pando: {context.config.ref_genome}")
        return
    result.warn("Pando motif resources come from installed R packages and are not statically checked by file path.")


def _check_figr(context: PreflightContext, method: MethodConfig, result: MethodCheckResult) -> None:
    if context.config.ref_genome == "mm10":
        result.skip("run_figr.R filters peaks with hg38 standard chromosomes and is not safe for mm10 datasets.")
        return
    if context.config.ref_genome != "hg38":
        result.skip(f"Unsupported reference genome for FigR: {context.config.ref_genome}")
        return


METHOD_PROFILES: dict[str, MethodProfile] = {
    "CellOracle": MethodProfile(
        supported_inputs=("h5ad",),
        required_obsm_keys=("X_umap",),
        checker=_check_celloracle,
    ),
    "Pando": MethodProfile(
        supported_inputs=("rds",),
        checker=_check_pando,
    ),
    "FigR": MethodProfile(
        supported_inputs=("rds",),
        checker=_check_figr,
    ),
    "SCENIC": MethodProfile(
        supported_inputs=("h5ad",),
        checker=_check_scenic,
    ),
    "SINCERITIES": MethodProfile(
        supported_inputs=("rds",),
        checker=_check_sincerities,
    ),
    "LINGER": MethodProfile(
        supported_inputs=("h5mu",),
        required_modalities=("RNA", "ATAC"),
        checker=_check_linger,
    ),
    "Dictys": MethodProfile(
        supported_inputs=("h5ad",),
        checker=_check_dictys,
    ),
    "Velorama": MethodProfile(
        supported_inputs=("h5ad",),
        required_obs_columns=("palantir_pseudotime",),
        checker=_check_velorama,
    ),
    "DictysFragments": MethodProfile(
        supported_inputs=("h5mu", "h5ad"),
        required_modalities=("RNA", "ATAC"),
        checker=_check_dictys_fragments,
    ),
    "DictysFragmentsResume": MethodProfile(
        supported_inputs=("h5mu", "h5ad"),
        required_modalities=("RNA", "ATAC"),
        checker=_check_dictys_fragments,
    ),
    "SCENICPLUS": MethodProfile(
        supported_inputs=("h5mu", "h5ad"),
        required_modalities=("RNA", "ATAC"),
        checker=_check_scenicplus,
    ),
    "SCENICPLUSResume": MethodProfile(
        supported_inputs=("h5mu", "h5ad"),
        required_modalities=("RNA", "ATAC"),
        checker=_check_scenicplus,
    ),
}


def _build_available_inputs(config: BaselineConfig) -> dict[str, pathlib.Path]:
    data_dir = config.home_dir / "benchmark" / "data"
    inputs = {
        "h5mu": data_dir / f"{config.dataset}.h5mu",
        "h5ad": data_dir / f"{config.dataset}.h5ad",
        "rds": data_dir / f"{config.dataset}.rds",
    }
    return {name: path for name, path in inputs.items() if path.exists()}


def _check_method_profile(context: PreflightContext, method: MethodConfig) -> MethodCheckResult:
    result = MethodCheckResult()
    profile = METHOD_PROFILES.get(method.name)
    if profile is None:
        result.warn("No dedicated preflight profile is registered for this method.")
        return result

    result.required_inputs.extend(profile.supported_inputs)
    resolved = context.find_input(profile.supported_inputs)
    if resolved is None:
        result.skip(
            "Missing required input file. Expected one of: "
            + ", ".join(str(context.config.home_dir / "benchmark" / "data" / f"{context.config.dataset}.{suffix}") for suffix in profile.supported_inputs)
        )
        return result

    input_type, input_path = resolved
    result.resolved_inputs.append(str(input_path))

    if input_type in {"h5ad", "h5mu"} and (
        profile.required_obs_columns or profile.required_obsm_keys or profile.required_modalities
    ):
        try:
            inspection = context.get_inspection(input_type)
        except Exception as exc:
            if _is_optional_inspection_error(exc):
                result.warn(f"Skipped deep dataset inspection for `{method.name}` because {exc}")
            else:
                result.skip(f"Failed to inspect dataset input `{input_path}`: {exc}")
                return result
        else:
            missing_obs = [column for column in profile.required_obs_columns if column not in inspection.obs_columns]
            if missing_obs:
                result.skip(f"Missing required obs columns for `{method.name}`: {', '.join(missing_obs)}")
                return result

            missing_obsm = [key for key in profile.required_obsm_keys if key not in inspection.obsm_keys]
            if missing_obsm:
                result.skip(f"Missing required obsm keys for `{method.name}`: {', '.join(missing_obsm)}")
                return result

            missing_modalities = [
                name for name in profile.required_modalities if not _logical_modality_present(inspection.modality_keys, name)
            ]
            if missing_modalities and input_type == "h5mu":
                result.skip(
                    f"Missing required modalities for `{method.name}` in `{inspection.path.name}`: {', '.join(missing_modalities)}"
                )
                return result

    if profile.checker is not None:
        profile.checker(context, method, result)
    return result


def run_preflight(config_path: pathlib.Path, root_dir: pathlib.Path) -> PreflightResult:
    config = load_baseline_config(config_path=config_path, root_dir=root_dir)
    result = PreflightResult(config_path=str(config_path), dataset=config.dataset)
    _check_required_config(config, result)
    if result.global_errors:
        return result

    cell_df, gene_idx = _check_cell_and_gene_lists(config, result)
    available_inputs = _build_available_inputs(config)
    context = PreflightContext(config=config, available_inputs=available_inputs)

    _check_dataset_consistency(context, result, cell_df, gene_idx)

    for method in config.methods:
        result.method_status[method.name] = _check_method_profile(context, method)

    if all(info.status == "skip" for info in result.method_status.values()):
        result.global_errors.append("All configured methods would be skipped by preflight; no runnable methods remain.")
    return result


def render_report(result: PreflightResult) -> str:
    lines = [
        "# Baseline Preflight Report",
        "",
        f"- Config: `{result.config_path}`",
        f"- Dataset: `{result.dataset}`",
        f"- Result: `{'PASS' if result.ok else 'FAIL'}`",
        "",
    ]

    lines.append("## Global Errors")
    if result.global_errors:
        lines.extend(f"- {message}" for message in result.global_errors)
    else:
        lines.append("- None")
    lines.append("")

    lines.append("## Warnings")
    if result.warnings:
        lines.extend(f"- {message}" for message in result.warnings)
    else:
        lines.append("- None")
    lines.append("")

    lines.append("## Methods")
    for method_name, info in result.method_status.items():
        lines.append(f"- `{method_name}`: `{info.status}`")
        if info.reasons:
            lines.extend(f"  reason: {reason}" for reason in info.reasons)
        if info.warnings:
            lines.extend(f"  warning: {warning}" for warning in info.warnings)
    lines.append("")
    return "\n".join(lines)


def write_outputs(result: PreflightResult, json_out: pathlib.Path | None, report_out: pathlib.Path | None) -> None:
    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n")
    if report_out is not None:
        report_out.parent.mkdir(parents=True, exist_ok=True)
        report_out.write_text(render_report(result) + "\n")
