"""
LINGER: benchmark script
"""

import os
import sys
import argparse
import pathlib
import time
import numpy as np
import pandas as pd
import logging
import session_info
from typing import List, Optional, Union
from copy import deepcopy

from baseline_cli_utils import log_memory_usage, str2bool

import scanpy as sc
import mudata
import LingerGRN
from LingerGRN.preprocess import *
from LingerGRN.pseudo_bulk import *
import LingerGRN.LINGER_tr as LINGER_tr
import LingerGRN.LL_net as LL_net

import warnings
warnings.simplefilter("ignore", FutureWarning)
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", RuntimeWarning)


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
    algoWorkDir = os.path.join(dirPjtHome, "tmp", "linger_wd")
    tmpSaveDir = os.path.join(dirPjtHome, "tmp", "linger_wd", args.version)
    if not os.path.exists(tmpSaveDir):
        os.makedirs(tmpSaveDir, exist_ok=True)

    benchmarkDir = os.path.join(dirPjtHome, "benchmark", args.version)
    if not os.path.exists(benchmarkDir):
        os.makedirs(benchmarkDir)
        os.makedirs(os.path.join(benchmarkDir, "net"))
        os.makedirs(os.path.join(benchmarkDir, "log"))
        os.makedirs(os.path.join(benchmarkDir, "fig"))

    # Setting the logger to INFO
    logging.basicConfig(filename=os.path.join(benchmarkDir, "log", "LINGER.log"),
                        level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        filemode='w')

    np.random.seed(args.seed)
    logging.info(f"Benchmark Version: {args.version} with seed {args.seed}")
    logging.info(f"Packages Version: LINGER 1.105")
    log_memory_usage()

    gene_selected = pd.read_csv(args.genelist, header=None)
    logging.info(f"Gene list: {args.genelist}")
    logging.info(f"Selected {len(gene_selected)} genes to benchmark")
    log_memory_usage()

    cell_selected = pd.read_csv(args.celllist, index_col=0)
    logging.info(f"Cell list: {args.celllist}")
    log_memory_usage()

    # Load data
    logging.info(f"[1/2] Loading the data from {args.dataset}.h5mu...")
    mdata = mudata.read(os.path.join(dirPjtHome, "benchmark", "data", f"{args.dataset}.h5mu"))
    
    adata_RNA = mdata.mod["RNA"]
    adata_ATAC = mdata.mod["ATAC"]
    adata_ATAC.obs["sample"] = adata_RNA.obs["sample"].copy()
    # if "sample" in adata_RNA.obs.columns:
    #     adata_RNA.obs["sample"] = adata_RNA.obs["sample"].copy()
    #     adata_ATAC.obs["sample"] = adata_ATAC.obs["sample"].copy()

    # Run LINGER per cell state
    logging.info(f"[2/2] Running LINGER and saving results...")
    for i, lin in enumerate(cell_selected.columns):
        logging.info(f"({i+1}/{len(cell_selected.columns)}) Cell State {lin}...")
        
        # Step 1: Load data and pre-process
        step1_start_time = time.time()
        logging.info(f"Prepare data for {lin}")
        adata_lin_RNA = adata_RNA[cell_selected[lin].values, gene_selected[0].values].copy()
        adata_lin_RNA.X = adata_lin_RNA.layers["counts"].copy()
        adata_lin_RNA.obs["label"] = lin
        adata_lin_RNA.obs["label"] = adata_lin_RNA.obs["label"].astype("category")
        adata_lin_RNA.obs["barcode"] = adata_lin_RNA.obs_names
        adata_lin_RNA.var["gene_ids"] = adata_lin_RNA.var_names
        
        adata_lin_ATAC = adata_ATAC[cell_selected[lin].values].copy()
        adata_lin_ATAC.obs["label"] = lin
        adata_lin_ATAC.obs["label"] = adata_lin_ATAC.obs["label"].astype("category")
        adata_lin_ATAC.obs["barcode"] = adata_lin_ATAC.obs_names
        adata_lin_ATAC.var["gene_ids"] = adata_lin_ATAC.var_names
        
        # Step 2: Generate the pseudo-bulk/metacell
        samplelist=list(set(adata_lin_RNA.obs["sample"].values)) # sample is generated from cell barcode 
        tempsample=samplelist[0]
        TG_pseudobulk=pd.DataFrame([])
        RE_pseudobulk=pd.DataFrame([])
        singlepseudobulk = (adata_lin_RNA.obs["sample"].unique().shape[0]*adata_lin_RNA.obs["sample"].unique().shape[0]>100)
        for tempsample in samplelist:
            adata_RNAtemp=adata_lin_RNA[adata_lin_RNA.obs["sample"]==tempsample]
            adata_ATACtemp=adata_lin_ATAC[adata_lin_ATAC.obs["sample"]==tempsample]
            TG_pseudobulk_temp,RE_pseudobulk_temp=pseudo_bulk(adata_RNAtemp,adata_ATACtemp,singlepseudobulk)                
            TG_pseudobulk=pd.concat([TG_pseudobulk, TG_pseudobulk_temp], axis=1)
            RE_pseudobulk=pd.concat([RE_pseudobulk, RE_pseudobulk_temp], axis=1)
            RE_pseudobulk[RE_pseudobulk > 100] = 100
        log_memory_usage()
            
        lintmpSaveDir = os.path.join(tmpSaveDir, lin)
        os.makedirs(lintmpSaveDir, exist_ok=True)
        os.makedirs(os.path.join(lintmpSaveDir, "data"), exist_ok=True)
        os.makedirs(os.path.join(lintmpSaveDir, "output"), exist_ok=True)
        
        os.chdir(lintmpSaveDir)
        
        if args.save:
            adata_lin_RNA.write(os.path.join(lintmpSaveDir, "data", "adata_RNA.h5ad"))
            adata_lin_ATAC.write(os.path.join(lintmpSaveDir, "data", "adata_ATAC.h5ad"))
        
        TG_pseudobulk=TG_pseudobulk.fillna(0)
        RE_pseudobulk=RE_pseudobulk.fillna(0)
        
        # Intersect RE_pseudobulk and adata_lin_ATAC.var['gene_ids']
        adata_lin_ATAC = adata_lin_ATAC[:, adata_lin_ATAC.var['gene_ids'].isin(RE_pseudobulk.index)]
        pd.DataFrame(adata_lin_ATAC.var['gene_ids']).to_csv(os.path.join(lintmpSaveDir, "data", "Peaks.txt"),header=None,index=None)
        
        if args.save:
            TG_pseudobulk.to_csv(os.path.join(lintmpSaveDir, "data", "TG_pseudobulk.tsv"))
            RE_pseudobulk.to_csv(os.path.join(lintmpSaveDir, "data", "RE_pseudobulk.tsv"))
        
        # # Step 3: Training the model
        genome = args.refGenome
        outdir = os.path.join(lintmpSaveDir, "output/")
        GRNdir = "/home/chenxufeng/picb_cxf/Data/10x_Genomics/PBMCs_10k_scMultiome/data_bulk/"
        method = "LINGER"
        
        preprocess(TG_pseudobulk,RE_pseudobulk,GRNdir,genome,method,outdir)
        log_memory_usage()
        
        activef='ReLU' # active function chose from 'ReLU','sigmoid','tanh'
        LINGER_tr.training(GRNdir,method,outdir,activef,'Human')
        log_memory_usage()
        
        # Step 4: Cell population gene regulatory network
        LL_net.TF_RE_binding(GRNdir,adata_lin_RNA,adata_lin_ATAC,genome,method,outdir)
        LL_net.cis_reg(GRNdir,adata_lin_RNA,adata_lin_ATAC,genome,method,outdir)
        LL_net.trans_reg(GRNdir,method,outdir,genome)
        log_memory_usage()
        # TODO: Post processing cell_population_trans_regulatory.txt
        
        net_df = pd.read_csv(os.path.join(outdir, "cell_population_trans_regulatory.txt"), sep='\t', index_col=0).T
        edge_df = _matrix_to_edge(net_df.values, rownames=net_df.index, colnames=net_df.columns).reset_index(drop=True)
        
        edge_df.to_csv(os.path.join(benchmarkDir, "net", f"LINGER_{lin}.csv"), index=None)
    
        
if __name__ == "__main__":
    main(parse_args())
    