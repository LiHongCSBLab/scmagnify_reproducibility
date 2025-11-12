import os
import logging
from rich.progress import Progress
import psutil
import numpy as np
import pandas as pd
import scanpy as sc
import session_info
import scmagnify as scm
from scmagnify import settings
from scmagnify.utils import (_get_data_modal, 
                             _str_to_list,
                             _matrix_to_edge, 
                             _edge_to_matrix)

import hydra
from omegaconf import DictConfig, OmegaConf

def log_memory_usage():
    """
    Log current memory usage
    """
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    logging.info(f"Memory usage: {memory_info.rss / 1024 ** 2:.2f} MB")


def _build_basal_grn(data) -> None:
    """
    Build basal GRN
    """
    data = data.copy()
    data.uns["motif_scan"]["motif_score"]["motif2factors"] = [_str_to_list(x) for x in data.uns["motif_scan"]["motif_score"]["motif2factors"]]
    
    filtered_motif_score = data.uns["motif_scan"]["motif_score"]
    
    peak_list = filtered_motif_score["seqname"].unique()
    multi_index_df = filtered_motif_score.set_index(["seqname", "motif_id"])

    # Create a dictionary mapping motif IDs to transcription factors (TFs)
    motif2factors = filtered_motif_score[["motif_id", "motif2factors"]]
    motif2factors.set_index("motif_id", inplace=True)
    motif_to_tfs = {}

    # Build one-hot encoded matrix for TFs
    with Progress() as progress:
        # Task1：Mapping motifs to TFs
        task1 = progress.add_task("[cyan]Mapping motifs to TFs...", total=len(motif2factors))

        for motif_id, motif2factor in motif2factors.iterrows():
            motif_to_tfs[motif_id] = motif2factor["motif2factors"]
            progress.update(task1, advance=1)  

        # Task2：Building chromatin constraint
        task2 = progress.add_task("[green]Building chromatin constraint...", total=len(peak_list))
        tf_onehot_list = []
        for peak in peak_list:
            motifs = multi_index_df.loc[peak].index
            tfs = []
            for motif in motifs:
                tfs += motif_to_tfs[motif]
            tfs = np.unique(tfs)
            series = pd.Series(np.repeat(1, len(tfs)), index=tfs)
            tf_onehot_list.append(series)
            progress.update(task2, advance=1) 
            
    tf_onehot = pd.concat(tf_onehot_list, axis=1, sort=True).transpose().fillna(0).astype(int)
    tf_onehot.index = peak_list
    del tf_onehot_list

    # Merge TF one-hot matrix with gene-peak correlations
    filtered_peak_gene_corrs = data.uns["peak_gene_corrs"]["filtered_corrs"]
    peak_to_tf = pd.merge(filtered_peak_gene_corrs, tf_onehot, left_on="peak", right_index=True)

    # Aggregate TF bindings by gene
    gene_to_tf = peak_to_tf.groupby("gene").sum().applymap(lambda x: np.where(x > 0, 1, 0))
    gene_to_tf.columns = gene_to_tf.columns.str.upper()

    # Convert gene-to-TF matrix to edges
    gene_to_tf_onehot = gene_to_tf.groupby(level=0, axis=1).sum().applymap(lambda x: np.where(x > 0, 1, 0))
    basal_grn = _matrix_to_edge(gene_to_tf_onehot.T)
    
    return basal_grn
    

@hydra.main(version_base=None, config_path="conf", config_name="magnify_chrom_tcell")
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
        os.makedirs(basal_grn_dir)

    # Setting the logger to INFO
    logging.basicConfig(filename=os.path.join(benchmark_dir, "log", f"scMagnify_chrom.log"),
                        level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        filemode='w')
    
    logging.info(OmegaConf.to_yaml(cfg))

    np.random.seed(cfg.seed)
    logging.info(f"Benchmark Version: {cfg.version} with seed {cfg.seed}")
    logging.info(f"Packages Version: {session_info.show()}")
    log_memory_usage()

    gene_selected = pd.read_csv(cfg.gene, header=None)
    logging.info(f"Gene list: {cfg.gene}")
    logging.info(f"Selected {len(gene_selected)} genes to benchmark")
    log_memory_usage()

    cell_selected = pd.read_csv(cfg.cell, index_col=0)
    logging.info(f"Cell list: {cfg.cell}")
    log_memory_usage()
    
    scm.set_genome(
    version=cfg.ref_genome,
    genomes_dir="/home/chenxufeng/picb_cxf/Ref/human/hg38/"
    )

    # Load data
    logging.info(f"[1/] Loading the data from {cfg.dataset}...")
    mdata = scm.read(os.path.join(dirPjtHome, "benchmark", "data", f"{cfg.dataset}.h5mu"))
    logging.info(f"mdata: {mdata}")
    log_memory_usage()
    
    # Load metacell data
    meta_mdata = scm.read(os.path.join(dirPjtHome, "benchmark", "data", f"{cfg.meta}.h5mu"))
    
    # Run scMagnify
    logging.info(f"[2/] Constructing Basal GRN")
    for i, lin in enumerate(cell_selected.columns):
        logging.info(f"({i+1}/{len(cell_selected.columns)}) Cell State {lin}...")
        
        # Step1: Filter the data
        mdata_fil = mdata[mdata["RNA"].obsm["cell_state_masks"][lin]].copy()
        meta_mdata_fil = meta_mdata[meta_mdata.obsm["cell_state_masks"][lin]].copy()
    
        # Step2: Connect Peak2Gene
        mdata_fil = scm.tl.connect_peaks_genes(mdata_fil, 
                                    meta_mdata_fil, 
                                    gene_selected=gene_selected[0],
                                    span=cfg.span,
                                    cor_cutoff=cfg.cor_cutoff,
                                    pval_cutoff=cfg.pval_cutoff,
                                    n_jobs=10)
        log_memory_usage()
        # Step3: Motif Scan
        mdata_fil = scm.tl.match_motif(mdata_fil, 
                                    motif_db=cfg.motif_db, 
                                    threshold=cfg.motif_cutoff)
        
        log_memory_usage()
        
        if cfg.tmp_save:
            mdata_fil.write(os.path.join(tmp_save_dir, f"scmagnify_{lin}.h5mu"))
            
        # Step4: Save the basal GRN
        basal_grn = _build_basal_grn(mdata_fil)
        basal_grn.to_csv(os.path.join(basal_grn_dir, f"basal_grn-SEACells_overall-rs_{cfg.span}-moods_{cfg.motif_db}_{cfg.motif_cutoff}-{lin}.csv"), index=False)
        
        # TODO : Add tensor decomposition and visualization
        logging.info(f"Finish Cell State {lin}")
        
    logging.info("Basal GRN construction finished!")

if __name__ == "__main__":
    main()