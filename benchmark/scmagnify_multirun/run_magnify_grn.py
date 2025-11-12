import os
import ast
import logging
import psutil
import numpy as np
import pandas as pd
import scanpy as sc
import session_info
import scmagnify as scm
scm.settings.verbosity = 4

import hydra
from omegaconf import DictConfig, OmegaConf

def log_memory_usage():
    """
    Log current memory usage
    """
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    logging.info(f"Memory usage: {memory_info.rss / 1024 ** 2:.2f} MB")

def downsample_cells(adata, proportion, method="random"):
    """
    Downsample cells by a given proportion using the specified method
    
    Parameters:
        cell_selected (pd.DataFrame): The dataframe of selected cells
        adata (AnnData): The AnnData object containing gene expression data
        proportion (float): The proportion of cells to retain
        method (str): Sampling method, either "random" or "geosketch"

    Returns:
        pd.DataFrame: The downsampled cells
    """
    
    cell_index = adata.obs_names
    
    if method == "random":
        cell_sampled = np.random.choice(cell_index, int(len(cell_index) * proportion), replace=False)
    elif method == "geosketch":
        pass
    else:
        raise ValueError("Invalid sampling method {method}")

    return adata[cell_sampled]
    
    
@hydra.main(version_base=None, config_path="conf", config_name="magnify_grn")
def main(cfg: DictConfig) -> None:
    """
    Main function
    """
    
    # Configurations
    dirPjtHome = cfg.home
    algo_work_dir = os.path.join(dirPjtHome, "tmp", "scmagnify_wd")
    tmp_save_dir = os.path.join(dirPjtHome, "tmp", "scmagnify_wd", cfg.version)
    if not os.path.exists(algo_work_dir):
        os.makedirs(algo_work_dir)

    benchmark_dir = os.path.join(dirPjtHome, "benchmark", cfg.version)
    if not os.path.exists(benchmark_dir):
        os.makedirs(benchmark_dir)
        os.makedirs(os.path.join(benchmark_dir, "net"))
        os.makedirs(os.path.join(benchmark_dir, "log"))
        os.makedirs(os.path.join(benchmark_dir, "fig"))
        
    basal_grn_dir = os.path.join(dirPjtHome, "basal_GRN", cfg.version)
    if not os.path.exists(basal_grn_dir):
        raise FileNotFoundError(f"Basal GRN directory not found: {basal_grn_dir}")
    # Setting the logger to INFO
    logging.basicConfig(filename=os.path.join(benchmark_dir, "log", f"scMagnify_{cfg.grn_params.trail}.log"),
                        level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        filemode='w')
    
    logging.info(OmegaConf.to_yaml(cfg))

    np.random.seed(cfg.grn_params.seed)
    logging.info(f"Benchmark Version: {cfg.version} with seed {cfg.grn_params.seed}")
    logging.info(f"Packages Version: {session_info.show()}")
    log_memory_usage()

    gene_selected = pd.read_csv(cfg.gene, header=None)
    logging.info(f"Gene list: {cfg.gene}")
    logging.info(f"Selected {len(gene_selected)} genes to benchmark")
    log_memory_usage()

    cell_selected = pd.read_csv(cfg.cell, index_col=0)
    logging.info(f"Cell list: {cfg.cell}")
    log_memory_usage()

    # Load data
    logging.info(f"[1/] Loading the data from {cfg.dataset}...")
    adata = sc.read_h5ad(os.path.join(dirPjtHome, "benchmark", "data", f"{cfg.dataset}.h5ad"))
    logging.info(f"adata: {adata}")
    logging.info(f"adata.X: {adata.X.A}")
    log_memory_usage()

    # Run scMagnify
    logging.info(f"[2/] Running scMagnify and saving the results...")
    for i, lin in enumerate(cell_selected.columns):
        logging.info(f"({i+1}/{len(cell_selected.columns)}) Cell State {lin} ")
        
        # Step1: Load basal GRN
        basal_grn = pd.read_csv(f"{basal_grn_dir}/basal_grn-{cfg.grn_params.basal_grn}-{lin}.csv")
        
        # Step2: Prepare the data
        adata_lin = adata[cell_selected[lin], gene_selected[0].values]
        adata_lin.X = adata_lin.layers["counts"].copy()
        
        # Downsample cells
        if cfg.grn_params.prop < 1.0:
            adata_lin = downsample_cells(adata_lin, 
                                        proportion=cfg.grn_params.prop, 
                                        method="random")
            logging.info(f"Downsampled cells: {adata_lin.shape}")

        hidden_dim = ast.literal_eval(cfg.grn_params.hidden_dim)
        
        # Step3: Train the model
        magni = scm.MAGNI(
            data=adata_lin,
            gene_selected=gene_selected[0],
            basal_grn=basal_grn,
            hidden=hidden_dim,
            lag=cfg.grn_params.time_lag,
            max_iter=cfg.grn_params.epochs,
            batch_size=cfg.grn_params.batch_size,
            lr=cfg.grn_params.learning_rate,
            weight_decay=cfg.grn_params.weight_decay,
            patience=cfg.grn_params.patience,
            lmbd=cfg.grn_params.lmbd,
            alpha=cfg.grn_params.alpha,
            gamma=cfg.grn_params.gamma,
            seed=cfg.grn_params.seed,
            device=cfg.grn_params.device,
        )
        
        model = magni.train()
        log_memory_usage()
        
        if cfg.tmp_save:
            model.save(os.path.join(tmp_save_dir, f"scmagnify_{lin}.pt"))
            
        # Step4: Regulation Inference
        logging.info(f"Regulation Inference...")
        gdata = magni.regulation_inference()
        log_memory_usage()
        if cfg.tmp_save:
            gdata.write(os.path.join(tmp_save_dir, f"scmagnify_{lin}.h5mu"))
            
        # Step5: Save the results
        edge_df = gdata.uns["network"].iloc[:, 0:3]
        edge_df.columns = ["TF", "Target", "Score"]
        edge_df.to_csv(os.path.join(benchmark_dir, "net", f"scmagnify-{cfg.grn_params.trail}_{lin}.csv"), index=False)
        logging.info(f"Save the network to {os.path.join(benchmark_dir, 'net', f'scmagnify-{cfg.grn_params.trail}_{lin}.csv')}")
        
        # # Step6: Network Score
        # scm.tl.get_network_score(gdata)
        # gdata["GRN"].var["mean_activity"] = gdata["GRN"].X.A.mean(axis=0)
        # scm.pl.stripplot(gdata, 
        #                  theme="darkgrid", 
        #                  sortby="degree_centrality",
        #                  save=os.path.join(benchmark_dir, "fig", f"scmagnify_network_score-{cfg.grn_params.trail}_{lin}.pdf"))
        
        # # Step7: Tensor Decomposition
        # decomper = scm.RegDecomp(gdata)
        # decomper.tucker_decomposition(rank=cfg.grn_params.time_lag)
        # decomper.compute_activity()
        # scm.pl.circosplot(gdata, 
        #           top_tfs=30,
        #           cluster=True,
        #           colorbar=False,
        #           label_kws={"label_size": 14},
        #           figsize=(7, 7),
        #           save=os.path.join(benchmark_dir, "fig", f"_regulon_scmagnify-{cfg.grn_params.trail}_{lin}.pdf")
        # )      
          
        # sc.settings.figdir = os.path.join(benchmark_dir, "fig")
        # sc.pl.umap(gdata["Regulon"], 
        #            color=gdata["Regulon"].var_names, 
        #            ncols=5, 
        #            cmap="Spectral_r", 
        #            save=f"_regulon_scmagnify_{lin}_{cfg.grn_params.trail}.pdf")
        
        # sc.pl.violin(gdata["Regulon"], 
        #              groupby="celltype", 
        #              keys=gdata["Regulon"].var_names, 
        #              rotation=90, 
        #              save=f"_regulon_scmagnify_{lin}_{cfg.grn_params.trail}.pdf")
        
        logging.info(f"Finish Cell State {lin}")
        
    logging.info("scMagnify benchmark finished!")

if __name__ == "__main__":
    main()