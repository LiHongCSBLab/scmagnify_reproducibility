"""
CellOracle: benchmark script
"""

import os
import sys
import argparse
import pathlib
import time
import numpy as np
import pandas as pd
import scanpy as sc
import celloracle as co
import logging
import session_info

from baseline_cli_utils import log_memory_usage, str2bool


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description="Run CellOracle")
    parser.add_argument("-i", "--input", dest="input", type=pathlib.Path, required=True,
                        help="Path to input file (.h5ad)")
    parser.add_argument("-h", "--home", dest="dirPjtHome", type=pathlib.Path, required=True,
                        help="Path to the project home directory")
    parser.add_argument("-c", "--cell", dest="celllist", type=pathlib.Path, required=True,
                        help="Path to cell list file (.csv)")
    parser.add_argument("-g", "--gene", dest="genelist", type=pathlib.Path, required=True,
                        help="Path to gene list file (.csv)")
    parser.add_argument("-v", "--version", dest="version", type=str, required=True,
                        help="Benchmark version")
    parser.add_argument("-t", "--tmp-save", dest="tmp", type=str2bool, default=False,
                        help="Temporary flag")
    parser.add_argument("-s", "--seed", dest="seed", type=int, default=0,
                        help="Random seed")
    parser.add_argument("-r", "--ref-genome", dest="refGenome", type=str, default="hg38",
                        help="Reference genome")

    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    """
    Main function
    """
    # Configurations
    dirPjtHome = args.dirPjtHome
    algoWorkDir = os.path.join(dirPjtHome, "tmp", "celloracle_wd")
    tmpSaveDir = os.path.join(dirPjtHome, "tmp", "celloracle_wd", args.version)
    if not os.path.exists(algoWorkDir):
        os.makedirs(algoWorkDir)

    benchmarkDir = os.path.join(dirPjtHome, "benchmark", args.version)
    if not os.path.exists(benchmarkDir):
        os.makedirs(benchmarkDir)
        os.makedirs(os.path.join(benchmarkDir, "net"))
        os.makedirs(os.path.join(benchmarkDir, "log"))
        os.makedirs(os.path.join(benchmarkDir, "fig"))

    # Setting the logger to INFO
    logging.basicConfig(filename=os.path.join(benchmarkDir, "log", "CellOracle.log"),
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
    logging.info(f"[1/2] Loading the data from {args.input}...")
    adata = sc.read_h5ad(args.input)
    logging.info(f"adata: {adata}")
    logging.info(f"adata.X: {adata.X.A}")
    log_memory_usage()

    base_GRN = pd.read_parquet(os.path.join(algoWorkDir, "base_GRN.parquet"))
    logging.info(f"base_GRN: {base_GRN.head()}")
    log_memory_usage()

    # Run CellOracle per cell state
    logging.info(f"[2/2] Running CellOracle and saving results...")
    

if __name__ == "__main__":
    main(parse_args())
    