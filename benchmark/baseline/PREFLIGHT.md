# Baseline Preflight

`baseline` workflow now provides an optional preflight layer for dataset onboarding and method resource validation.

## What It Checks

- Config integrity: `dataset`, `home`, `cell`, `gene`, `ref-genome`, `methods`
- Cell mask quality: empty file, duplicate cell IDs, duplicate lineage columns, empty lineage selections
- Gene list quality: empty file, blank genes, duplicated genes
- Dataset consistency against `benchmark/data/<dataset>.h5mu` or `.h5ad` when available:
  - cell ID overlap with dataset `obs_names`
  - gene overlap with dataset features
  - per-lineage selected cell overlap
- Method-specific resources:
  - `SCENICPLUS` / `SCENICPLUSResume`
  - `DictysFragments` / `DictysFragmentsResume`
  - `SCENIC`
  - `LINGER`
  - `Velorama`
  - `CellOracle`
  - `Dictys`
  - `SINCERITIES`
  - `Pando`
  - `FigR`

## Trigger Modes

Manual preflight only:

```bash
python workflow/run_preflight.py \
  --config workflow/conf/baseline_cd34_260411.yaml \
  --json-out /tmp/baseline_preflight.json \
  --report-out /tmp/baseline_preflight.md
```

Run through the workflow runner:

```bash
bash BaselineRunner.sh --preflight workflow/conf/baseline_cd34_260411.yaml
```

## Output Semantics

- `global_errors`: fail the preflight and stop the workflow when `--preflight` is enabled
- `warnings`: reported but do not stop execution
- `method_status.<name>.status == skip`: only skip that method, continue running the rest

When `BaselineRunner.sh --preflight` is used, outputs are written to:

- `benchmark/<version>/log/preflight.json`
- `benchmark/<version>/log/preflight_report.md`

## Optional Environment Override

If the default `python` interpreter does not have the required packages for dataset inspection, run preflight in a specific conda environment:

```bash
BASELINE_PREFLIGHT_ENV=scenicplus bash BaselineRunner.sh --preflight workflow/conf/baseline_cd34_260411.yaml
```

## Notes

- Preflight is opt-in and is intended for new dataset acceptance, not mandatory before every run.
- Method-level skips focus on missing method inputs and external reference resources.
- Some R-package-provided resources cannot be validated by static file paths; these are reported as warnings where appropriate.
