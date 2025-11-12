#!/usr/bin/env Rscript
# Pando benchmark script

suppressPackageStartupMessages({
    library(tidyverse)
    library(Pando)
    library(Signac)
    library(Seurat)
    library(doParallel)
    library(argparse)
    library(logging)
})

parse_args <- function() {
    parser <- ArgumentParser()
    parser$add_argument(
        "--home", dest = "dirPjtHome", type = "character", required = TRUE,
        help = "Path to the project home directory"
    )
    parser$add_argument(
        "--dataset", dest = "dataset", type = "character", required = TRUE,
        help = "Dataset key"
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

# log_memory_usage <- function() {
#     memory_usage <- pryr::mem_used()
#     loginfo(paste("Memory usage:", format(memory_usage, units = "MB")))
# }

main <- function(args) {
    ## Configurations
    dirPjtHome <- args$dirPjtHome
    algoWorkDir <- file.path(dirPjtHome, "tmp", "pando_wd")
    tmpSaveDir <- file.path(dirPjtHome, "tmp", "pando_wd", args$version)
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
        data('motifs')
        data('motif2tf')
        data('phastConsElements20Mammals.UCSC.hg38')
    } else if (args$ref_genome == "mm10") {
        library(BSgenome.Mmusculus.UCSC.mm10)
    } else {
        stop("Invalid reference genome")
    }

    # Setting the logger to INFO
    log_file <- file.path(benchmarkDir, "log", "Pando.log")
    basicConfig()
    addHandler(writeToFile, file=log_file, level='INFO')

    set.seed(args$seed)

    loginfo(paste("Benchmark version:", args$version, "with seed:", args$seed))
    loginfo(paste("Packages Version: Pando", packageVersion("Pando")))
    # log_memory_usage()

    gene_selected <- read.csv(args$genelist, header = FALSE)
    loginfo(paste("Gene list:", args$genelist))
    loginfo(paste("Number of genes:", nrow(gene_selected)))
    # log_memory_usage()

    cell_selected <- read.csv(args$celllist, row.names = 1)
    loginfo(paste("Cell list:", args$celllist))
    loginfo(paste("Number of cells:", nrow(cell_selected)))
    # log_memory_usage()

    ## Load data
    loginfo(paste("[1/3] Loading the data from", args$dataset))
    obj <- readRDS(file.path(args$dirPjtHome, "benchmark", "data", paste0(args$dataset, ".rds")))

    # Intersect the cell list with the obj colnames
    cell_selected <- cell_selected[rownames(cell_selected) %in% colnames(obj), ]
    loginfo(paste("Number of cells in the cell list:", nrow(cell_selected)))
    # log_memory_usage()

    loginfo("[2/3] Preprocessing the data...")
    DefaultAssay(obj) <- 'ATAC'
    keep.peaks <- as.logical(seqnames(granges(obj)) %in% main.chroms)
    obj[["ATAC"]] <- subset(obj[["ATAC"]], features = rownames(obj[["ATAC"]])[keep.peaks])
    # log_memory_usage()

    loginfo("[3/3] Running Pando and saving the results...")
    lineages <- colnames(cell_selected)

    for (lin in lineages) {
        i <- 1
        path <- file.path(algoWorkDir, lin)
        if (!dir.exists(path)) {
            dir.create(path, recursive = TRUE)
        }
        setwd(path)

        obj_lin <- obj[, unlist(cell_selected[lin]) == "True"]
        DefaultAssay(obj_lin) <- "RNA"

        # Step 1: Initiate the GRN object
        step1_start_time <- Sys.time()
        obj_lin <- initiate_grn(
            obj_lin,
            rna_assay = 'RNA',
            peak_assay = 'ATAC'
        )
        step1_end_time <- Sys.time()
        loginfo(paste("Step 1: Initiate GRN for", lin, "done in", step1_end_time - step1_start_time, "seconds"))
        # log_memory_usage()

        # Step 2: Motif scanning
        step2_start_time <- Sys.time()
        obj_lin <- suppressWarnings(find_motifs(
            obj_lin,
            pfm = motifs,
            motif_tfs = motif2tf,
            genome = BSgenome.Hsapiens.UCSC.hg38
        ))
        step2_end_time <- Sys.time()
        loginfo(paste("Step 2: Motif scanning for", lin, "done in", step2_end_time - step2_start_time, "seconds"))
        # log_memory_usage()

        DefaultAssay(obj_lin) <- 'RNA'
        # Step 3: Infer GRN
        step3_start_time <- Sys.time()
        registerDoParallel(10)
        obj_lin <- infer_grn(
            obj_lin,
            peak_to_gene_method = 'Signac',
            genes = unlist(gene_selected),
            parallel = TRUE
        )

        if (args$tmp_save) {
            saveRDS(obj_lin, file = file.path(tmpSaveDir, paste0(lin, "_pando_res.rds")))
            write.csv(coef(obj_lin), file = file.path(tmpSaveDir, "_coef.csv"), row.names = FALSE)
        }

        edge_df <- coef(obj_lin)[, c('tf', 'target', 'estimate')]
        colnames(edge_df) <- c("TF", "Target", "Score")
        write.csv(edge_df, file = file.path(benchmarkDir, "net", paste0("Pando_", lin, ".csv")), row.names = FALSE, quote = FALSE)
        step3_end_time <- Sys.time()
        loginfo(paste("Step 3: Infer GRN for", lin, "done in", step3_end_time - step3_start_time, "seconds"))
        # log_memory_usage()
    }

    loginfo("Pando benchmark finished!")
    # log_memory_usage()
}

main(parse_args())