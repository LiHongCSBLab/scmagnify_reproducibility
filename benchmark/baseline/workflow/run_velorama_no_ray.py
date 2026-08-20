#!/usr/bin/env python
"""
Single-process Velorama runner (no ray, no model checkpoints).

Input:  preprocessed .h5ad with adata.var['is_reg'] / adata.var['is_target']
Output: edge list CSV with columns: TF,Target,Score
"""

import argparse
import os
from copy import deepcopy

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
from scipy.sparse import issparse

from velorama.models import (
    VeloramaMLP,
    VeloramaMLPTarget,
    prox_update,
    prox_update_target,
    regularize,
    regularize_target,
    restore_parameters,
    ridge_regularize,
)
from velorama.utils import (
    calculate_diffusion_lags,
    construct_dag,
    estimate_interactions,
)


# Fixed parameters (edit here if needed).
SEED = 0
DYNAMICS = "pseudotime_precomputed"
PTLOC = "palantir_pseudotime"
DEVICE = "cpu"
LAG = 5
HIDDEN = 32
MAX_ITER = 1000
LEARNING_RATE = 0.01
LAM_START = -2
LAM_END = 1
NUM_LAMBDAS = 19
LAMBDA_RIDGE = 0.0
PENALTY = "H"
LOOKBACK = 5
CHECK_EVERY = 10
REG_TARGET = 1
N_NEIGHBORS = 30
VELO_MODE = "stochastic"
TIME_SERIES = 0
N_COMPS = 50
PROBA = 1
X_NORM = "zscore"
UPPER_THRESH = 0.95


def _to_dense(matrix):
    if issparse(matrix):
        return matrix.toarray()
    return np.asarray(matrix)


def _zscore_tensor(values: np.ndarray, clip: bool = True) -> torch.FloatTensor:
    std = values.std(0)
    std[std == 0] = 1
    out = torch.FloatTensor(values - values.mean(0)) / std
    if clip:
        out = torch.clip(out, -5, 5)
    return out


def prepare_xy(adata, x_norm: str):
    x_raw = _to_dense(adata[:, adata.var["is_reg"]].X).copy()
    y_raw = _to_dense(adata[:, adata.var["is_target"]].X).copy()

    if x_norm == "zscore":
        x = _zscore_tensor(x_raw, clip=True)
        y = _zscore_tensor(y_raw, clip=True)
    elif x_norm == "none":
        x = torch.FloatTensor(x_raw)
        y = torch.FloatTensor(y_raw)
    else:
        raise ValueError(f"Unsupported x_norm in this script: {x_norm}")
    return x, y


def train_single_lam(
    ax: torch.FloatTensor,
    ay: torch.FloatTensor,
    y: torch.FloatTensor,
    device: str,
    lam: float,
) -> torch.Tensor:
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    num_regs = ax.shape[-1]
    num_targets = y.shape[1]

    ax = ax.to(device)
    y = y.to(device)
    if REG_TARGET:
        ay = ay.to(device)
        model = VeloramaMLPTarget(
            num_targets,
            num_regs,
            lag=LAG,
            hidden=[HIDDEN],
            device=device,
            activation="relu",
        )
    else:
        model = VeloramaMLP(
            num_targets,
            num_regs,
            lag=LAG,
            hidden=[HIDDEN],
            device=device,
            activation="relu",
        )

    model.to(device)
    loss_fn = nn.MSELoss(reduction="none")

    best_it = None
    best_loss = np.inf
    best_model = None

    if REG_TARGET:
        preds = model(ax, ay)
    else:
        preds = model(ax)
    loss = loss_fn(preds, y).mean(0).sum()
    smooth = loss + ridge_regularize(model, LAMBDA_RIDGE)

    for it in range(MAX_ITER):
        smooth.backward()
        for param in model.parameters():
            param.data = param - LEARNING_RATE * param.grad

        if lam > 0:
            if REG_TARGET:
                prox_update_target(model, lam, LEARNING_RATE, PENALTY)
            else:
                prox_update(model, lam, LEARNING_RATE, PENALTY)

        model.zero_grad()

        if REG_TARGET:
            preds = model(ax, ay)
        else:
            preds = model(ax)
        loss = loss_fn(preds, y).mean(0).sum()
        smooth = loss + ridge_regularize(model, LAMBDA_RIDGE)

        if (it + 1) % CHECK_EVERY == 0:
            if REG_TARGET:
                nonsmooth = regularize_target(model, lam, PENALTY)
            else:
                nonsmooth = regularize(model, lam, PENALTY)
            mean_loss = (smooth + nonsmooth).detach() / y.shape[1]

            if mean_loss < best_loss:
                best_loss = mean_loss
                best_it = it
                best_model = deepcopy(model)
            elif (it - best_it) == LOOKBACK * CHECK_EVERY:
                break

    if best_model is not None:
        restore_parameters(model, best_model)

    # target x regulator x lag
    return model.GC(threshold=False, ignore_lag=False).cpu()


def to_edge_csv(gc_mat: np.ndarray, reg_genes: np.ndarray, target_genes: np.ndarray, output_csv: str):
    gc_df = pd.DataFrame(gc_mat, index=target_genes, columns=reg_genes)
    edge_df = (
        gc_df.stack()
        .reset_index()
        .rename(columns={"level_0": "Target", "level_1": "TF", 0: "Score"})
    )
    edge_df = edge_df[["TF", "Target", "Score"]]
    edge_df = edge_df[edge_df["Score"] != 0]
    edge_df = edge_df.sort_values("Score", ascending=False).reset_index(drop=True)
    edge_df.to_csv(output_csv, index=False)


def parse_args():
    parser = argparse.ArgumentParser(description="Run Velorama without ray and save edge CSV only.")
    parser.add_argument("--input_h5ad", required=True, type=str, help="Input h5ad path.")
    parser.add_argument("--output_csv", required=True, type=str, help="Output edge CSV path.")
    parser.add_argument(
        "--device",
        default=DEVICE,
        choices=["cpu", "cuda"],
        help="Compute device (default from fixed config).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)

    adata = sc.read(args.input_h5ad)
    if "is_target" not in adata.var.columns:
        adata.var["is_target"] = True
    if "is_reg" not in adata.var.columns:
        raise ValueError("adata.var['is_reg'] is required in input h5ad.")

    target_genes = adata.var.index.values[adata.var["is_target"]]
    reg_genes = adata.var.index.values[adata.var["is_reg"]]

    x, y = prepare_xy(adata, x_norm=X_NORM)

    sc.pp.scale(adata)
    a = construct_dag(
        adata,
        dynamics=DYNAMICS,
        ptloc=PTLOC,
        proba=PROBA,
        n_neighbors=N_NEIGHBORS,
        velo_mode=VELO_MODE,
        use_time=TIME_SERIES,
        n_comps=N_COMPS,
    )
    a = torch.FloatTensor(a)

    ax = calculate_diffusion_lags(a, x, LAG)
    ay = calculate_diffusion_lags(a, y, LAG) if REG_TARGET else None

    lam_list = np.logspace(LAM_START, LAM_END, num=NUM_LAMBDAS).tolist()
    lam_list = [float(np.round(lam, 4)) for lam in lam_list]
    gc_lags = []
    for lam in lam_list:
        gc_lag = train_single_lam(ax=ax, ay=ay, y=y, device=args.device, lam=lam)
        gc_lags.append(gc_lag.detach())
    all_lags = torch.stack(gc_lags)

    gc_mat = estimate_interactions(all_lags, lag=LAG, upper_thresh=UPPER_THRESH).numpy()
    to_edge_csv(gc_mat=gc_mat, reg_genes=reg_genes, target_genes=target_genes, output_csv=args.output_csv)
    print(f"Saved edge list to: {args.output_csv}")


if __name__ == "__main__":
    main()
