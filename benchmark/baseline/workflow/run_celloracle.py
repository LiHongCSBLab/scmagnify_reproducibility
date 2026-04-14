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


def main(args: argparse.Namespace) -> None:
    """
    Main function
    """
    # Configurations
    dirPjtHome = args.dirPjtHome
    algoWorkDir = os.path.join(dirPjtHome, "tmp", "celloracle_wd")
    tmpSaveDir = os.path.join(dirPjtHome, "tmp", "celloracle_wd", args.version)
    if not os.path.exists(tmpSaveDir):
        os.makedirs(tmpSaveDir, exist_ok=True)

    benchmarkDir = os.path.join(dirPjtHome, "benchmark", args.version)
    if not os.path.exists(benchmarkDir):
        os.makedirs(benchmarkDir)
        os.makedirs(os.path.join(benchmarkDir, "net"))
        os.makedirs(os.path.join(benchmarkDir, "log"))
        os.makedirs(os.path.join(benchmarkDir, "fig"))

    # Setting the logger to INFO
    logging.basicConfig(filename=os.path.join(benchmarkDir, "log", "CellOracle.log"),
                        level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        filemode='w')

    np.random.seed(args.seed)
    logging.info(f"Benchmark Version: {args.version} with seed {args.seed}")
    logging.info(f"Packages Version: celloracle {co.__version__}")
    log_memory_usage()

    gene_selected = pd.read_csv(args.genelist, header=None)
    logging.info(f"Gene list: {args.genelist}")
    logging.info(f"Selected {len(gene_selected)} genes to benchmark")
    log_memory_usage()

    cell_selected = pd.read_csv(args.celllist, index_col=0)
    logging.info(f"Cell list: {args.celllist}")
    log_memory_usage()

    # Load data
    logging.info(f"[1/2] Loading the data from {args.dataset}.h5ad...")
    adata = sc.read_h5ad(os.path.join(dirPjtHome, "benchmark", "data", f"{args.dataset}.h5ad"))
    logging.info(f"adata: {adata}")
    logging.info(f"adata.X: {adata.X.A}")
    log_memory_usage()

    base_GRN = pd.read_parquet(os.path.join(algoWorkDir, "base_GRN_dataframe.parquet"))
    logging.info(f"base_GRN: {base_GRN.head()}")
    log_memory_usage()

    # Run CellOracle per cell state
    logging.info(f"[2/2] Running CellOracle and saving results...")
    for i, lin in enumerate(cell_selected.columns):
        logging.info(f"({i+1}/{len(cell_selected.columns)}) Cell State {lin}...")

        # Step 1: Load data and pre-process
        step1_start_time = time.time()
        logging.info(f"Prepare data for {lin}")
        adata_lin = adata[cell_selected[lin].values, gene_selected[0].values].copy()
        adata_lin.X = adata_lin.layers["counts"].copy()
        adata_lin.obs["lineage"] = lin
        adata_lin.obs["lineage"] = adata_lin.obs["lineage"].astype("category")

        oracle = co.Oracle()
        oracle.import_anndata_as_raw_count(adata=adata_lin,
                                           cluster_column_name="lineage",
                                           embedding_name="X_umap")
        oracle.import_TF_data(TF_info_matrix=base_GRN)
        logging.info(oracle)
        log_memory_usage()

        # Perform PCA
        oracle.perform_PCA()

        # Select important PCs
        n_comps = np.where(np.diff(np.diff(np.cumsum(oracle.pca.explained_variance_ratio_)) > 0.002))[0][0]
        n_comps = min(n_comps, 50)
        logging.info(f"Auto-selected n_comps is: {n_comps}")

        n_cell = oracle.adata.shape[0]
        logging.info(f"n_cell is: {n_cell}")

        k = int(0.025 * n_cell)
        logging.info(f"Auto-selected k is: {k}")
        step1_end_time = time.time()
        logging.info(f"Step 1: Load data and pre-process time cost: {step1_end_time - step1_start_time} seconds")
        log_memory_usage()

        # Step 2: KNN imputation
        step2_start_time = time.time()
        oracle.knn_imputation(n_pca_dims=n_comps, k=k, balanced=True, b_sight=k * 8,
                              b_maxl=k * 4, n_jobs=8)
        if args.save:
            oracle.to_hdf5(os.path.join(tmpSaveDir, f"{lin}.celloracle.oracle"))
        step2_end_time = time.time()
        logging.info(f"Step 2: KNN imputation time cost: {step2_end_time - step2_start_time} seconds")
        log_memory_usage()

        # Step 3: Run CellOracle
        step3_start_time = time.time()
        links = oracle.get_links(cluster_name_for_GRN_unit="lineage", alpha=10,
                                 verbose_level=10, n_jobs=8)
        if args.save:
            links.links_dict[lin].to_csv(os.path.join(tmpSaveDir, f"raw_GRN_for_{lin}.csv"), index=None)

        links.filter_links(p=0.001)
        logging.info("Filter links with p=0.001")

        edge_df = links.links_dict[lin]
        edge_df = edge_df.loc[:, ['source', 'target', 'coef_abs']]
        edge_df.columns = ['TF', 'Target', 'Score']

        # Statistics
        logging.info(f"Number of edges: {len(edge_df)}")
        logging.info(f"Number of TFs: {len(edge_df['TF'].unique())}")
        logging.info(f"Number of Targets: {len(edge_df['Target'].unique())}")

        edge_df.to_csv(os.path.join(benchmarkDir, "net", f"CellOracle_{lin}.csv"), index=None)
        logging.info(f"Save edge_df to {os.path.join(benchmarkDir, 'net', f'CellOracle_{lin}.csv')}")
        step3_end_time = time.time()

        logging.info(f"Step 3: Run CellOracle time cost: {step3_end_time - step3_start_time} seconds")
        logging.info(f"Total time cost: {step3_end_time - step1_start_time} seconds")
        log_memory_usage()

        logging.info(f"Finish Cell State {lin}")

    logging.info("CellOracle benchmark finished!")
    log_memory_usage()

if __name__ == "__main__":
    main(parse_args())
    
    
    
    
    
    
    
    
    
