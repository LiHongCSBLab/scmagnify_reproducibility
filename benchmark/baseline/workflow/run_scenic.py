"""
SCENIC: benchmark script
"""

import os
import sys
import ast
import argparse
import pathlib
import time
import numpy as np
import pandas as pd
import scanpy as sc
import loompy as lp
import logging
import session_info
import psutil  
import subprocess
import shutil


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description="Run Pando")
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
    algoWorkDir = os.path.join(dirPjtHome, "tmp", "scenic_wd")
    tmpSaveDir = os.path.join(dirPjtHome, "tmp", "scenic_wd", args.version)
    if not os.path.exists(tmpSaveDir):
        os.makedirs(tmpSaveDir)

    benchmarkDir = os.path.join(dirPjtHome, "benchmark", args.version)
    if not os.path.exists(benchmarkDir):
        os.makedirs(benchmarkDir)
        os.makedirs(os.path.join(benchmarkDir, "net"))
        os.makedirs(os.path.join(benchmarkDir, "log"))
        os.makedirs(os.path.join(benchmarkDir, "fig"))

    # Setting the logger to INFO
    logging.basicConfig(filename=os.path.join(benchmarkDir, "log", "SCENIC.log"),
                        level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        filemode='w')

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
    
    logging.info(f"[2/4] Convert adata to loom...")
    for i, lin in enumerate(cell_selected.columns):
        logging.info(f"({i+1}/{len(cell_selected.columns)}) Cell State {lin}...")

        # Step 1: Load data and pre-process
        step1_start_time = time.time()
        logging.info(f"Prepare data for {lin}")
        adata_lin = adata[cell_selected[lin].values, gene_selected[0].values]
        row_attrs = {
        "Gene": np.array(adata_lin.var_names),
        }
        col_attrs = {
            "CellID": np.array(adata_lin.obs_names),
            "nGene": np.array( np.sum(adata_lin.layers["counts"].transpose()>0 , axis=0)).flatten(),
            "nUMI": np.array( np.sum(adata_lin.layers["counts"].transpose() , axis=0)).flatten(),
        }
        lp.create(os.path.join(tmpSaveDir, f"{lin}.loom"), adata_lin.layers["counts"].transpose(), row_attrs, col_attrs)
        logging.info(f"File {lin}.loom saved.")
        
    logging.info(f"[3/4] Running SCENIC...")
    
    scenic_grn_ctx = "/home/chenxufeng/WorkSpace/scMagnify/scMagnify-benchmark/baseline/workflow/scenic_grn_ctx.sh"
    input_dir = tmpSaveDir
    ref_genome = args.refGenome
    try:
        result = subprocess.run(
            [scenic_grn_ctx, input_dir, ref_genome],  
            check=True,  
            text=True,   
            capture_output=True  
        )
        logging.info(result.stdout)
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to run SCENIC: {e}")
        sys.exit(1)
    logging.info(result.stdout)
    
    logging.info(f"[4/4] Saving results...")
    for i, lin in enumerate(cell_selected.columns):
    
        adj = pd.read_csv(os.path.join(tmpSaveDir, lin, "expr_mat.adjacencies.tsv"), sep="\t", index_col=0)
        adj.columns = ["Target", "Score"]
        adj.to_csv(os.path.join(benchmarkDir, "net", f"GRNBoost2_{lin}.csv"), sep=",")
        
        regulons = pd.read_csv(os.path.join(tmpSaveDir, lin, "regulons.csv"), sep=",", index_col=0, skiprows=1)  
        regulon_df = pd.DataFrame(regulons["TargetGenes"][1:], index=regulons.index[1:])
        
        edge_data = []
        for tf, row in regulon_df.iterrows():
            target_genes_str = row['TargetGenes']
            try:
                target_genes = ast.literal_eval(target_genes_str)
            except (ValueError, SyntaxError) as e:
                print(f"Warning: Failed to parse string for TF {tf}: {e}")
                continue
            
            for item in target_genes:
                if len(item) == 2:
                    gene, score = item
                    edge_data.append([tf, gene, score])
                else:
                    print(f"Warning: Skipping invalid tuple {item} for TF {tf}")

        edgedf = pd.DataFrame(edge_data, columns=["TF", "Target", "Score"])
        edgedf.to_csv(os.path.join(benchmarkDir, "net", f"SCENIC_{lin}.csv"), sep=",", index=False)
        
    if not args.tmp:
        try:
            logging.info(f"Deleting temporary files in {tmpSaveDir}")
            shutil.rmtree(tmpSaveDir)
        except Exception as e:
            logging.error(f"Failed to delete temporary files: {e}")
        
if __name__ == "__main__":
    args = parse_args()
    main(args)
    
        

        
            
