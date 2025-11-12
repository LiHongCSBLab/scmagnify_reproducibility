"""
scMagnify: benchmark script
"""

from typing import List, Optional, Tuple
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

import scmagnify as scm


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
    parser.add_argument("-t", "--tmp-save", dest="save", type=bool, default=False,
                        help="Temporary flag")
    parser.add_argument("-s", "--seed", dest="seed", type=int, default=0,
                        help="Random seed")
    parser.add_argument("-r", "--ref-genome", dest="refGenome", type=str, default="hg38",
                        help="Reference genome")
    
    ## Specific arguments for scMagnify
    # Basal GRN: f"./basal_GRN/basal_grn-{exp}-{lin}.csv"
    parser.add_argument("-b", "--basal-grn", dest="basalGRN", type=str, required=True,
                        help="version of basal GRN")
    
    # Model training
    parser.add_argument("-lr", "--learning-rate", dest="learningRate", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("-e", "--epochs", dest="epochs", type=int, default=100,
                        help="Number of epochs")
    parser.add_argument("-bs", "--batch-size", dest="batchSize", type=int, default=128,
                        help="Batch size")
    parser.add_argument("-p", "--patience", dest="patience", type=int, default=20,
                        help="Patience")
    parser.add_argument("-w", "--weight-decay", dest="weightDecay", type=float, default=0.0,
                        help="Weight decay")
    
    # Model Structure
    parser.add_argument("-hd", "--hidden-dim", dest="hiddenDim", type=List[int], default=[50],
                    help="Hidden dimensions")
    parser.add_argument("-lag", "--time-lag", dest="timeLag", type=int, default=1,
                    help="Time lag")
    
    # Loss function: L = L_MSE + λ * L_sparsity + γ * L_smooth + δ * L_chrom
    parser.add_argument("-lmbd", "--lmbd", dest="lmbd", type=float, default=3.,
                    help="Lambda")
    parser.add_argument("-alpha", "--alpha", dest="alpha", type=float, default=0.5,
                    help="Alpha")
    parser.add_argument("gamma", "--gamma", dest="gamma", type=float, default=0.5,
                    help="Gamma")

    return parser.parse_args()


def log_memory_usage():
    """
    Log current memory usage
    """
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    logging.info(f"Memory usage: {memory_info.rss / 1024 ** 2:.2f} MB")
    

def main(args: argparse.Namespace) -> None:
    """
    Main function
    """
    # Configurations
    dirPjtHome = args.dirPjtHome
    algoWorkDir = os.path.join(dirPjtHome, "tmp", "scmagnify_wd")
    tmpSaveDir = os.path.join(dirPjtHome, "tmp", "scmagnify_wd", args.version)
    if not os.path.exists(algoWorkDir):
        os.makedirs(algoWorkDir)

    benchmarkDir = os.path.join(dirPjtHome, "benchmark", args.version)
    if not os.path.exists(benchmarkDir):
        os.makedirs(benchmarkDir)
        os.makedirs(os.path.join(benchmarkDir, "net"))
        os.makedirs(os.path.join(benchmarkDir, "log"))
        os.makedirs(os.path.join(benchmarkDir, "fig"))
        
    basalGRNDir = os.path.join(dirPjtHome, "basal_GRN")
    if not os.path.exists(basalGRNDir):
        raise FileNotFoundError(f"Basal GRN directory not found: {basalGRNDir}")

    # Setting the logger to INFO
    logging.basicConfig(filename=os.path.join(benchmarkDir, "log", "scMagnify.log"),
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

    # Load data
    logging.info(f"[1/] Loading the data from {args.dataset}...")
    adata = sc.read_h5ad(os.path.join(dirPjtHome, "benchmark", "data", f"{args.dataset}.h5ad"))
    logging.info(f"adata: {adata}")
    logging.info(f"adata.X: {adata.X.A}")
    log_memory_usage()
    
    # Run scMagnify
    logging.info(f"[2/] Running scMagnify and saving the results...")
    for i, lin in enumerate(cell_selected.columns):
        logging.info(f"({i+1}/{len(cell_selected.columns)}) Cell State {lin}...")
        
        # Step1: Load basal GRN
        basalGRN = pd.read_csv(f"{basalGRNDir}/basal_grn-{args.basalGRN}-{lin}.csv")
        
        # Step2: Prepare the data
        adata_lin = adata[cell_selected[lin].values, gene_selected[0].values]
        adata_lin.X = adata_lin.layers["counts"].copy()
        
        # Step3: Train the model
        magni = scm.MAGNI(
            data=adata_lin,
            gene_selected=gene_selected[0],
            basal_grn=basalGRN,
            hidden=args.hiddenDim,
            lag=args.timeLag,
            max_iter=args.epochs,
            batch_size=args.batchSize,
            lr=args.learningRate,
            weight_decay=args.weightDecay,
            patience=args.patience,
            lmbd=args.lmbd,
            alpha=args.alpha,
            gamma=args.gamma,
            seed=args.seed    
        )
        modal=magni.train()
        log_memory_usage()
        
        if args.tmp:
            modal.save(os.path.join(tmpSaveDir, f"scmagnify_{lin}.pt"))
            
        # Step4: Regulation Inference
        gdata = magni.regulation_inference()
        log_memory_usage()
        if args.tmp:
            gdata.write(os.path.join(tmpSaveDir, f"scmagnify_{lin}.h5mu"))
            
        # Step5: Save the results
        edge_df = gdata.uns["network"]["TF", "Target", "score"]
        edge_df.columns = ["TF", "Target", "Score"]
        edge_df.to_csv(os.path.join(benchmarkDir, f"scmagnify_{lin}.csv"), index=False)
         
        # TODO : Add tensor decomposition and visualization
        logging.info(f"Finish Cell State {lin}")
        
    logging.info("scMagnify benchmark finished!")
            
if __name__ == "__main__":
    main(parse_args())
    