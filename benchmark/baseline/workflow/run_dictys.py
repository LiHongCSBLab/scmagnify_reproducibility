"""
Dictys: benchmark script
"""

import os
import sys
import argparse
import pathlib
import time
import numpy as np
import pandas as pd
import scanpy as sc
import logging
import session_info
import psutil
import re
import subprocess
from copy import deepcopy
from typing import List, Optional, Union
from tqdm.auto import tqdm


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description="Run Dictys")
    parser.add_argument("-d", "--dataset", dest="dataset", type=pathlib.Path, required=True,
                    help="Dataset Key")
    parser.add_argument("-p", "--home", dest="dirPjtHome", type=pathlib.Path, required=True,
                        help="Path to the project home directory")
    parser.add_argument("-c", "--cell", dest="celllist", type=pathlib.Path, required=True,
                        help="Path to cell list file (.csv)")
    parser.add_argument("-g", "--gene", dest="genelist", type=pathlib.Path, required=True,
                        help="Path to gene list file (.csv)")
    parser.add_argument("-v", "--version", dest="version", type=str, required=True,
                        help="Benchmark version")
    parser.add_argument("-t", "--tmp-save", dest="tmp", type=bool, default=False,
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


def log_memory_usage():
    """
    Log current memory usage
    """
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    logging.info(f"Memory usage: {memory_info.rss / 1024 ** 2:.2f} MB")


def run_command(cmd: str, check: bool = True) -> None:
    """
    Run a shell command using subprocess.run
    """
    subprocess.run(cmd, shell=True, check=check)


def main(args: argparse.Namespace) -> None:
    """
    Main function
    """
    # Configurations
    dirPjtHome = args.dirPjtHome
    algoWorkDir = os.path.join(dirPjtHome, "tmp", "dictys_wd")
    tmpSaveDir = os.path.join(dirPjtHome, "tmp", "dictys_wd", args.version)
    os.makedirs(algoWorkDir, exist_ok=True)

    benchmarkDir = os.path.join(dirPjtHome, "benchmark", args.version)
    os.makedirs(benchmarkDir, exist_ok=True)
    os.makedirs(os.path.join(benchmarkDir, "net"), exist_ok=True)
    os.makedirs(os.path.join(benchmarkDir, "log"), exist_ok=True)
    os.makedirs(os.path.join(benchmarkDir, "fig"), exist_ok=True)

    # Setting the logger to INFO
    logging.basicConfig(filename=os.path.join(benchmarkDir, "log", "Dictys.log"),
                        level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    np.random.seed(args.seed)
    logging.info(f"Benchmark Version: {args.version} with seed {args.seed}")
    logging.info(f"Packages Version: {session_info.show()}")
    log_memory_usage()

    # Load gene and cell lists
    gene_selected = pd.read_csv(args.genelist, header=None)
    logging.info(f"Gene list: {args.genelist}")
    logging.info(f"Selected {len(gene_selected)} genes to benchmark")
    log_memory_usage()

    cell_selected = pd.read_csv(args.celllist, index_col=0)
    logging.info(f"Cell list: {args.celllist}")
    log_memory_usage()

    # Load data
    logging.info(f"[1/8] Loading the data from {args.dataset}...")
    adata = sc.read_h5ad(os.path.join(dirPjtHome, "benchmark", "data", f"{args.dataset}.h5ad"))
    logging.info(f"adata: {adata}")
    logging.info(f"adata.X: {adata.X.A}")
    log_memory_usage()

    # # Process adata.obs_names if needed
    # if re.search(r"IM-1393_BoneMarrow_TcellDep", adata.obs_names[0]):
    #     file_names = adata.obs_names
    #     pattern = r'^IM-1393_BoneMarrow_TcellDep_(\d+)_multiome#'
    #     file_names_replaced = [re.sub(pattern, r'rep\1_', name) for name in file_names]
    #     adata.obs_names = file_names_replaced
    # else:
    #     adata.obs_names = adata.obs_names

    # Preprocess the adata
    logging.info(f"[2/8] Preprocess the adata...")
    for i, lin in enumerate(cell_selected.columns):
        logging.info(f"({i + 1}/{len(cell_selected.columns)}) Cell State {lin}...")
        adata_lin = adata[cell_selected[lin].values, gene_selected[0].values]
        adata_lin.X = adata_lin.layers["counts"].copy()
        adata_lin.obs["lineage"] = lin
        adata_lin.obs["lineage"] = adata_lin.obs["lineage"].astype("category")

        # Save expression data
        expDF = pd.DataFrame(adata_lin.layers["counts"].A.T, index=adata_lin.var_names, columns=adata_lin.obs_names)
        lin_dir = os.path.join(tmpSaveDir, lin, "data")
        os.makedirs(lin_dir, exist_ok=True)
        expDF.to_csv(os.path.join(lin_dir, "expression.tsv.gz"), sep="\t", compression="gzip")

        # Save lineage data
        adata_lin.obs["lineage"].to_csv(os.path.join(lin_dir, "clusters.csv"), header=['Lineages'], index=True, index_label='Barcode')

    # Preprocess Bam files
    logging.info(f"[3/8] Preprocess Bam files...")
    for lin in cell_selected.columns:
        target_dir = os.path.join(tmpSaveDir, lin, "data", "bams")
        os.makedirs(target_dir, exist_ok=True)
        for filename in tqdm(adata[cell_selected[lin].values].obs_names, desc=f"Processing {lin}"):
            # source_file = os.path.join(dirPjtHome, "bams", f"{filename}.bam")
            sample, barcode = filename.split("#")
            source_file = os.path.join(algoWorkDir, sample, "bams", f"{barcode}.bam")
            target_file = os.path.join(target_dir, f"{filename}.bam")
            if not os.path.exists(target_file):
                run_command(f"ln -s {source_file} {target_file}")

    # Preprocess subsets
    logging.info(f"[4/8] Preprocess subsets...")
    for lin in cell_selected.columns:
        lin_dir = os.path.join(tmpSaveDir, lin, "data")
        with open(os.path.join(lin_dir, "subsets.txt"), "w") as f:
            f.write(f"Subset{lin}\n")

        subsets = adata[cell_selected[lin].values].obs_names

        target_dir = os.path.join(lin_dir, "subsets", f"Subset{lin}")
        os.makedirs(target_dir, exist_ok=True)
        names_rna = pd.DataFrame(subsets)
        names_rna.to_csv(os.path.join(target_dir, "names_rna.txt"), header=False, index=False)
        names_atac = names_rna
        names_atac.to_csv(os.path.join(target_dir, "names_atac.txt"), header=False, index=False)

    # Prepare reference genome
    logging.info(f"[5/8] Prepare reference genome...")
    
    if args.refGenome == "hg38":
        homer_geome = "/home/chenxufeng/picb_cxf/Ref/human/hg38/homer_genome"
        motifs = "/home/chenxufeng/picb_cxf/Database/motif_databases/HOCOMOCOv11_full_HUMAN_mono_homer_format_0.0001.motif"
        gene_bed = "/home/chenxufeng/picb_cxf/Ref/human/hg38/annotations/ucsc/gene.bed"
        
    elif args.refGenome == "mm10":
        homer_geome = "/home/chenxufeng/picb_cxf/Ref/mouse/mm10/homer_genome"
        motifs = "/home/chenxufeng/picb_cxf/Database/motif_databases/HOCOMOCOv11_full_MOUSE_mono_homer_format_0.0001.motif"
        gene_bed = "/home/chenxufeng/picb_cxf/Ref/mouse/mm10/annotations/ucsc/gene.bed"
        
    for lin in cell_selected.columns:
        lin_dir = os.path.join(tmpSaveDir, lin, "data")
        run_command(f"ln -sf {homer_geome} {os.path.join(lin_dir, 'genome')}")
        run_command(f"ln -sf {motifs} {os.path.join(lin_dir, 'motifs.motif')}")
        run_command(f"ln -sf {gene_bed} {os.path.join(lin_dir, 'gene.bed')}")

    # Configure the makefile
    logging.info(f"[6/8] Configure the makefile...")
    for lin in cell_selected.columns:
        target_dir = os.path.join(tmpSaveDir, lin, "makefiles")
        os.makedirs(target_dir, exist_ok=True)
        os.chdir(target_dir)
        run_command("dictys_helper makefile_template.sh common.mk config.mk env_none.mk static.mk")
        run_command(f"dictys_helper makefile_update.py {os.path.join(target_dir, 'config.mk')} '{{\"DEVICE\": \"cuda:1\", \"GENOME_MACS2\": \"hs\", \"JOINT\": \"1\"}}'")

    # Run Dictys
    logging.info(f"[7/8] Run Dictys...")
    for lin in cell_selected.columns:
        lin_dir = os.path.join(tmpSaveDir, lin)
        os.chdir(lin_dir)
        run_command(f"dictys_helper makefile_check.py")
        run_command(f"dictys_helper network_inference.sh -j 32 -J 1 static")
        
    # Save the results
    logging.info(f"[8/8] Save the results...")
    from dictys.net import network
    
    for lin in cell_selected.columns:
        # Define paths
        lin_dir = os.path.join(tmpSaveDir, lin)
        output_dir = os.path.join(lin_dir, "output")
        static_h5_path = os.path.join(output_dir, "static.h5")
        static_dir = os.path.join(output_dir, "static")
        full_subset_path = os.path.join(static_dir, "Full", f"Subset{lin}.tsv.gz")
        output_csv_path = os.path.join(benchmarkDir, "net", f"Dictys_{lin}.csv")

        # Load network from file
        d0 = network.from_file(static_h5_path)
        logging.info(f"Loaded network from {static_h5_path} for lineage {lin}")

        # Export network if static directory does not exist
        if not os.path.exists(static_dir):
            logging.info(f"Exporting network to {static_dir} for lineage {lin}")
            d0.export(static_dir, sparsities=None)

        # Read matrix data
        logging.info(f"Reading matrix data from {full_subset_path} for lineage {lin}")
        matrix_df = pd.read_csv(full_subset_path, sep="\t", index_col=0)

        # Convert matrix to edge list
        logging.info(f"Converting matrix to edge list for lineage {lin}")
        edge_df = _matrix_to_edge(matrix_df, rownames=matrix_df.index, colnames=matrix_df.columns).reset_index(drop=True)

        # Save edge list to CSV
        logging.info(f"Saving edge list to {output_csv_path} for lineage {lin}")
        edge_df.to_csv(output_csv_path, index=None)
    

if __name__ == "__main__":
    args = parse_args()
    main(args)