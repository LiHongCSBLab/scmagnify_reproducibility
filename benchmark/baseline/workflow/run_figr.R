#!/usr/bin/env Rscript
# FigR benchmark script

suppressPackageStartupMessages({
    library(dplyr)
    library(tidyverse)
    library(SummarizedExperiment)
    library(Matrix)
    library(FNN)
    library(FigR)
    library(chromVAR)
    library(doParallel)
    library(BuenColors)
    library(ggplot2)
    library(logging)
    library(argparse)
    library(Seurat)
    library(Signac)
    library(stats)
})

# Parse command-line arguments
parse_args <- function() {
    parser <- ArgumentParser()
    parser$add_argument(
        "--dataset", dest = "dataset", type = "character", required = TRUE,
        help = "Dataset key"
    )
    parser$add_argument(
        "--home", dest = "dirPjtHome", type = "character", required = TRUE,
        help = "Path to the project home directory"
    )
    parser$add_argument(
        "--cell", dest = "celllist", type = "character", required = TRUE,
        help = "Path to cell list file (.csv)"
    )
    parser$add_argument(
        "--gene", dest = "genelist", type = "character", required = TRUE,
        help = "Path to gene list file (.csv)"
    )
    parser$add_argument(
        "--version", dest = "version", type = "character", required = TRUE,
        help = "Benchmark version"
    )
    parser$add_argument(
        "--tmp-save", dest = "tmp_save", type = "logical", required = TRUE, default = FALSE,
        help = "Save temporary files"
    )
    parser$add_argument(
        "--seed", dest = "seed", type = "integer", required = TRUE, default = 0,
        help = "Random seed"
    )
    parser$add_argument(
        "--ref-genome", dest = "ref_genome", type = "character", required = TRUE,
        help = "Reference genome (hg38 or mm10)"
    )

    return(parser$parse_args())
}


# # Log memory usage
# log_memory_usage <- function() {
#     memory_usage <- pryr::mem_used()
#     loginfo(paste("Memory usage:", format(memory_usage, units = "MB")))
# }


# Main function
main <- function(args) {
    ## Configurations
    dirPjtHome <- args$dirPjtHome
    algoWorkDir <- file.path(dirPjtHome, "tmp", "figr_wd")
    tmpSaveDir <- file.path(dirPjtHome, "tmp", "figr_wd", args$version)
    if (args$tmp_save) {
        dir.create(tmpSaveDir, showWarnings = FALSE, recursive = TRUE)
    }

    benchmarkDir <- file.path(dirPjtHome, "benchmark", args$version)
    if (!dir.exists(benchmarkDir)) {
        dir.create(benchmarkDir, showWarnings = FALSE, recursive = TRUE)
        dir.create(file.path(benchmarkDir, "net"), showWarnings = FALSE, recursive = TRUE)
        dir.create(file.path(benchmarkDir, "log"), showWarnings = FALSE, recursive = TRUE)
    }

    if (args$ref_genome == "hg38") {
        library(BSgenome.Hsapiens.UCSC.hg38)
        main.chroms <- standardChromosomes(BSgenome.Hsapiens.UCSC.hg38)
    } else if (args$ref_genome == "mm10") {
        library(BSgenome.Mmusculus.UCSC.mm10)
    } else {
        stop("Invalid reference genome")
    }

    # Set up logging
    log_file <- file.path(benchmarkDir, "log", "FigR.log")
    basicConfig()
    addHandler(writeToFile, file = log_file, level = "INFO")
    
    set.seed(args$seed)
    loginfo(paste("Benchmark version:", args$version, "with seed:", args$seed))
    loginfo(paste("Packages Version: FigR", packageVersion("FigR")))
    # log_memory_usage()

    # Load gene and cell lists
    gene_selected <- read.csv(args$genelist, header = FALSE)
    loginfo(paste("Gene list:", args$genelist))
    loginfo(paste("Number of genes:", nrow(gene_selected)))
    # log_memory_usage()

    cell_selected <- read.csv(args$celllist, row.names = 1)
    loginfo(paste("Cell list:", args$celllist))
    loginfo(paste("Number of cells:", nrow(cell_selected)))
    # log_memory_usage()


    # Load input data
    loginfo(paste("[1/3] Loading the data from", args$dataset))
    obj <- readRDS(file.path(args$dirPjtHome, "benchmark", "data", paste0(args$dataset, ".rds")))
    # log_memory_usage()
    # Intersect the cell list with the obj colnames
    cell_selected <- cell_selected[rownames(cell_selected) %in% colnames(obj), ]

    # Preprocess data
    loginfo("[2/3] Preprocessing the data...")
    DefaultAssay(obj) <- "ATAC"
    keep.peaks <- as.logical(seqnames(granges(obj)) %in% standardChromosomes(BSgenome.Hsapiens.UCSC.hg38))
    obj[["ATAC"]] <- subset(obj[["ATAC"]], features = rownames(obj[["ATAC"]])[keep.peaks])
    # log_memory_usage()

    # Process each lineage
    lineages <- colnames(cell_selected)
    for (lin in lineages) {
        loginfo(paste("Start to run", lin))
        path <- file.path(algoWorkDir, lin)
        if (!dir.exists(path)) {
            dir.create(path, recursive = TRUE)
        }
        setwd(path)

        # Step 1: Convert Seurat object to SummarizedExperiment
        step1_start_time <- Sys.time()
        obj_lin <- obj[, unlist(cell_selected[lin]) == "True"]
        DefaultAssay(obj_lin) <- "ATAC"
        ATAC.se <- SummarizedExperiment(
            assay = list(counts = obj_lin$ATAC@counts),
            rowRanges = granges(obj_lin),
            colData = obj_lin@meta.data
        )

        RNAmat <- obj_lin$RNA@data
        DefaultAssay(obj_lin) <- "RNA"
        RNAmat <- RNAmat[unlist(gene_selected), ]
        RNAmat <- RNAmat[Matrix::rowSums(RNAmat) != 0, ]  # Remove genes with zero expression
        step1_end_time <- Sys.time()
        loginfo(paste("Step 1: Convert Seurat object to SummarizedExperiment for", lin, "takes", step1_end_time - step1_start_time))
        # log_memory_usage()

        # Step 2: Smooth RNA using cell KNNs
        step2_start_time <- Sys.time()
        lsi <- obj_lin@reductions$lsi@cell.embeddings
        cellkNN <- FNN::get.knn(lsi, k = 30)$nn.index
        rownames(cellkNN) <- colnames(RNAmat)
        colnames(RNAmat) <- colnames(ATAC.se)
        RNAmat.s <- smoothScoresNN(NNmat = cellkNN, mat = RNAmat, nCores = 10)

        if (args$tmp_save) {
            saveRDS(ATAC.se, file = file.path(path, paste0(lin, "_t-cell-depleted-bm_atac_SE.rds")))
            saveRDS(RNAmat, file = file.path(path, paste0(lin, "_t-cell-depleted-bm_RNAnorm.rds")))
            saveRDS(cellkNN, file = file.path(path, paste0(lin, "_t-cell-depleted-bm_cellkNN.rds")))
            saveRDS(RNAmat.s, file = file.path(path, paste0(lin, "_t-cell-depleted-bm_RNAnorm_s.rds")))
        }
        step2_end_time <- Sys.time()
        loginfo(paste("Step 2: Smooth RNA using cell KNNs for", lin, "takes", step2_end_time - step2_start_time))
        # log_memory_usage()

        # Step 3: Run gene-peak correlation
        step3_start_time <- Sys.time()
        cisCorr <- suppressWarnings(FigR::runGenePeakcorr(
            ATAC.se = ATAC.se,
            RNAmat = RNAmat,
            genome = args$ref_genome,
            nCores = 10,
            p.cut = NULL,
            n_bg = 100
        ))

        if (args$tmp_save) {
            saveRDS(cisCorr, file = file.path(path, paste0(lin, "_t-cell-depleted-bm_cisCorr.rds")))
        }
        step3_end_time <- Sys.time()
        loginfo(paste("Step 3: Run gene-peak correlation for", lin, "takes", step3_end_time - step3_start_time))
        # log_memory_usage()

        # Step 4: Define DORC genes
        p <- 0.05
        cutoff <- 0
        step4_start_time <- Sys.time()
        cisCorr.filt <- cisCorr %>% filter(pvalZ <= p)
        dorcGenes <- dorcJPlot(
            dorcTab = cisCorr.filt,
            cutoff = cutoff,
            labelTop = 20,
            returnGeneList = TRUE,
            force = 5
        )
        
        suffix <- paste0("_p", p, "_cutoff", cutoff, ".rds")

        if (args$tmp_save) {
            saveRDS(cisCorr.filt, file = file.path(path, paste0(lin, "_t-cell-depleted-bm_cisCorr_filt", suffix)))
        }

        # dorcGenes <- dorcGenes[dorcGenes != "GRB10"]  # Remove GRB10

        dorcMat <- getDORCScores(
            ATAC.se = ATAC.se,
            dorcTab = cisCorr.filt,
            geneList = dorcGenes,
            nCores = 10
        )

        if (args$tmp_save) {
            saveRDS(dorcMat, file = file.path(path, paste0(lin, "_t-cell-depleted-bm_dorcMat", suffix)))
        }

        dorcMat.s <- smoothScoresNN(NNmat = cellkNN, mat = dorcMat, nCores = 10)

        # which(rowSums(dorcMat.s) == 0) deletes the rows with all zeros
        dorcMat.s <- dorcMat.s[which(rowSums(dorcMat.s) != 0), ]


        if (args$tmp_save) {
            saveRDS(dorcMat.s, file = file.path(path, paste0(lin, "_t-cell-depleted-bm_dorcMat_s", suffix)))
        }


        step4_end_time <- Sys.time()
        loginfo(paste("Step 4: Define DORC genes for", lin, "takes", step4_end_time - step4_start_time))
        # log_memory_usage()

        # Step 5: Run FigR
        step5_start_time <- Sys.time()
        figR.d <- suppressWarnings(runFigRGRN(
            ATAC.se = ATAC.se,
            dorcTab = cisCorr.filt,
            genome = args$ref_genome,
            dorcMat = dorcMat.s,
            dorcK = 5,
            rnaMat = RNAmat.s,
            nCores = 10
        ))

        if (args$tmp_save) {
            saveRDS(figR.d, file = file.path(path, paste0(lin, "_t-cell-depleted-bm_figR", suffix)))
        }

        edge_df <- figR.d[, c('Motif', 'DORC', 'Score')]
        colnames(edge_df) <- c('TF', 'Target', 'Score')
        write.csv(edge_df, file = file.path(benchmarkDir, "net", paste0("FigR_", lin, ".csv")), row.names = FALSE, quote = FALSE)
        step5_end_time <- Sys.time()
        loginfo(paste("Step 5: Run FigR for", lin, "takes", step5_end_time - step5_start_time))
        # log_memory_usage()
    }

    loginfo("FigR benchmark finished!")
    # log_memory_usage()
}

# Execute main function
main(parse_args())