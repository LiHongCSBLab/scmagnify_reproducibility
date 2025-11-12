#!/usr/bin/env Rscript
# FigR benchmark script

suppressPackageStartupMessages({
    library(tidyverse)
    library(SummarizedExperiment)
    library(Matrix)
    library(FNN)
    library(FigR)
    library(pryr)  
    library(logging)
    library(argparse)
})

# Parse command-line arguments
parse_args <- function() {
    parser <- ArgumentParser()
    parser$add_argument(
        "--input", dest = "input", type = "character", required = TRUE,
        help = "Path to input dataset (.rds)"
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

# Log memory usage
log_memory_usage <- function() {
    memory_usage <- pryr::mem_used()
    loginfo(paste("Memory usage:", format(memory_usage, units = "MB")))
}

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

    # Set up logging
    log_file <- file.path(benchmarkDir, "log", "figr.log")
    logging::basicConfig(
        file = log_file,
        level = logging::INFO,
        format = "%(asctime)s - %(levelname)s - %(message)s"
    )

    set.seed(args$seed)
    loginfo(paste("Benchmark version:", args$version, "with seed:", args$seed))
    loginfo(paste("Packages Version:", sessionInfo()))
    log_memory_usage()

    # Load gene and cell lists
    gene_selected <- read.csv(args$genelist, header = FALSE)
    loginfo(paste("Gene list:", args$genelist))
    loginfo(paste("Number of genes:", nrow(gene_selected)))
    log_memory_usage()

    cell_selected <- read.csv(args$celllist, row.names = 1)
    loginfo(paste("Cell list:", args$celllist))
    loginfo(paste("Number of cells:", nrow(cell_selected)))
    log_memory_usage()

    # Load input data
    loginfo(paste("[1/3] Loading the data from", args$input))
    obj <- readRDS(args$input)
    log_memory_usage()

}