"""
Velorama: benchmark script
"""

import os
import sys
import argparse
import pathlib
import time
import numpy as np
import pandas as pd
import scanpy as sc
import loompy as lp
import logging
import session_info
import subprocess
from typing import List, Optional, Union
from copy import deepcopy

from baseline_cli_utils import log_memory_usage, str2bool


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description="Run CellOracle")
    parser.add_argument("-p", "--home", dest="dirPjtHome", type=pathlib.Path, required=True,
                        help="Path to the project home directory")
    parser.add_argument("-d", "--dataset", dest="dataset", type=pathlib.Path, required=True,
                    help="Dataset Key")
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

    return parser.parse_args()


def _matrix_to_edge(m: Union[np.ndarray, pd.DataFrame],
                   rownames: Optional[List[str]] = None, 
                   colnames: Optional[List[str]] = None) -> pd.DataFrame:
    
    '''
    Convert matrix to edge list
    p.s. row for regulator, column for target
    
    
    Parameters:
    -----------
    m: matrix
    rownames: list of regulator names
    colNames: list of target names

    Return:
    -------
    edge DataFrame [TF, Target, Score]
    '''
    if isinstance(m, pd.DataFrame):
        mat = deepcopy(m)
        rownames = np.array(mat.index)
        colnames = np.array(mat.columns)
        
    elif isinstance(m, np.ndarray):
        mat = deepcopy(m)
        mat = pd.DataFrame(mat)
        if rownames is None or colnames is None:
            raise ValueError('rownames and colnames must be provided if m is numpy.ndarray')
        rownames = np.array(rownames)
        colnames = np.array(colnames)
    
    num_regs = rownames.shape[0]
    num_targets = colnames.shape[0]

    mat_indicator_all = np.zeros([num_regs, num_targets])

    mat_indicator_all[abs(mat) > 0] = 1
    idx_row, idx_col = np.where(mat_indicator_all)

    idx = list(zip(idx_row, idx_col))
    #for row, col in idx:
    #    if row == col:
    #        idx.remove((row, col))
                
    edges_df = pd.DataFrame(
        {'TF': rownames[idx_row], 'Target': colnames[idx_col], 'Score': [mat.iloc[row, col] for row, col in idx]})

    edge = edges_df.sort_values('Score', ascending=False).reset_index(drop=True)
    
    return edge 
    

def main(args: argparse.Namespace) -> None:
    """
    Main function
    """
    # Configurations
    dirPjtHome = args.dirPjtHome
    algoWorkDir = os.path.join(dirPjtHome, "tmp", "velorama_wd")
    tmpSaveDir = os.path.join(dirPjtHome, "tmp", "velorama_wd", args.version)
    if not os.path.exists(tmpSaveDir):
        os.makedirs(tmpSaveDir)
    

    benchmarkDir = os.path.join(dirPjtHome, "benchmark", args.version)
    if not os.path.exists(benchmarkDir):
        os.makedirs(benchmarkDir)
        os.makedirs(os.path.join(benchmarkDir, "net"))
        os.makedirs(os.path.join(benchmarkDir, "log"))
        os.makedirs(os.path.join(benchmarkDir, "fig"))

    # Setting the logger to INFO
    logging.basicConfig(filename=os.path.join(benchmarkDir, "log", "Velorama.log"),
                        level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    np.random.seed(args.seed)
    logging.info(f"Benchmark Version: {args.version} with seed {args.seed}")
    logging.info(f"Packages Version: {session_info.show()}")
    log_memory_usage()

    gene_selected = pd.read_csv(args.genelist, header=None)
    logging.info(f"Gene list: {args.genelist}")
    logging.info(f"Selected {len(gene_selected)} genes to benchmark")
    log_memory_usage()

    cell_selected = pd.read_csv(args.celllist, index_col=0)
    logging.info(f"Cell list: {args.celllist}")
    log_memory_usage()

    logging.info(f"[1/4] Loading the data from {args.dataset}...")
    adata = sc.read_h5ad(os.path.join(dirPjtHome, "benchmark", "data", f"{args.dataset}.h5ad"))
    logging.info(f"adata: {adata}")
    logging.info(f"adata.X: {adata.X.A}")
    log_memory_usage()
    
    logging.info(f"[2/4] Preprocess and Split the data...")
    for i, lin in enumerate(cell_selected.columns):
        if not os.path.exists(os.path.join(tmpSaveDir, lin)):
            os.makedirs(os.path.join(tmpSaveDir, lin))
        logging.info(f"({i+1}/{len(cell_selected.columns)}) Cell State {lin}...")

        step1_start_time = time.time()
        logging.info(f"Prepare data for {lin}")
        adata_lin = adata[cell_selected[lin].values, gene_selected[0].values]
        
        tf_list = pd.read_csv(f"/home/chenxufeng/picb_cxf/Ref/tflists/cistarget/allTFs_{args.refGenome}.txt", header=None)[0].tolist()
        regs = adata_lin.var_names.intersection(tf_list)
        adata_lin.var['is_reg'] = False
        adata_lin.var['is_target'] = True
        adata_lin.var.loc[regs, 'is_reg'] = True
        adata_lin.obs["pseudotime"] = adata_lin.obs["palantir_pseudotime"].copy()
        adata_lin.uns["iroot"] = np.where(adata_lin.obs.index == adata_lin.obs.sort_values("pseudotime").index[0])[0][0]
        
        sc.pp.normalize_total(adata_lin, target_sum=1e4)
        sc.pp.log1p(adata_lin)
        
        adata_lin.write(os.path.join(tmpSaveDir, f"{lin}/{lin}.h5ad"))
        
    logging.info(f"[3/4] Run Velorama...")
    for i, lin in enumerate(cell_selected.columns):
        logging.info(f"({i+1}/{len(cell_selected.columns)}) Cell State {lin}...")
        
        step2_start_time = time.time()
        logging.info(f"Run Velorama for {lin}")
        DYN="pseudotime_precomputed"
        DEV="cuda"
        LAGS="5"
        HIDDEN_DIM="32"
        PTLOC="palantir_pseudotime"
        cmd = f"velorama -ds {os.path.join(tmpSaveDir, f'{lin}/{lin}')} -dyn {DYN} -dev {DEV} -l {LAGS} -hd {HIDDEN_DIM} -rd {tmpSaveDir}/{lin} -sd {tmpSaveDir}/{lin}/ -ptloc {PTLOC}"
        # Write the command to a shell script file
        script_path = os.path.join(tmpSaveDir, f"{lin}/run_velorama_{lin}.sh")
        with open(script_path, 'w') as script_file:
            script_file.write(f"#!/bin/bash\n{cmd}\n")
        # result = subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # logging.info(result.stdout)
        # logging.info(f"Velorama for {lin} finished in {time.time() - step2_start_time:.2f} seconds")
        
    logging.info(f"[4/4] Save the results...")
    for i, lin in enumerate(cell_selected.columns):
        matrix_df = pd.read_csv(os.path.join(tmpSaveDir, lin, "velorama_run.pseudotime_precomputed.velorama.interactions.tsv"),  sep="\t", index_col=0).T
        edge_df = _matrix_to_edge(matrix_df, rownames=matrix_df.index, colnames=matrix_df.columns).reset_index(drop=True)
        edge_df.to_csv(os.path.join(benchmarkDir, "net", f"Velorama_{lin}.csv"), index=False)


    logging.info(f"Results saved to {benchmarkDir}")
    log_memory_usage()

if __name__ == "__main__":
    args = parse_args()
    main(args)

    
        
    
        
            