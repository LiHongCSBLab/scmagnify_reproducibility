#!/usr/bin/env Rscript
# SINCERITIES benchmark script

suppressPackageStartupMessages({
    library(tidyverse)
    library(kSamples)
    library(glmnet)
    library(ppcor)
    library(Signac)
    library(Seurat)
    library(argparse)
    library(logging)
    library(reshape2)
})


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

# log_memory_usage <- function() {
#     memory_usage <- pryr::mem_used()
#     loginfo(paste("Memory usage:", format(memory_usage, units = "MB")))
# }

preprocess <- function(df){
  
  NAIdx <- is.na(apply(df,1,sum))
  df <- df[!NAIdx,]
  
  totDATA <- df[,1:dim(df)[2]-1]
  timeline <- df[,dim(df)[2]]
  DATA.time <- sort(unique(timeline))
  DATA.num_time_points <- length(DATA.time)
  DATA.totDATA <- matrix(ncol = dim(df)[2]-1)
  DATA.timeline <- vector()
  
  for (k in 1:DATA.num_time_points) {
    I <- which(timeline==DATA.time[k])
    DATA.totDATA <- rbind(DATA.totDATA,as.matrix(totDATA[I,]))
    DATA.timeline <- c(DATA.timeline,timeline[I])
  }
  DATA.totDATA <- DATA.totDATA[-1,]
  DATA.totDATA[is.na(DATA.totDATA)] <- 0
  DATA.numGENES <- dim(DATA.totDATA)[2]
  
  DATA.genes <- colnames(df)[1:dim(df)[2]-1]
  
  DATA.singleCELLdata <- by(DATA.totDATA,DATA.timeline,identity)
  DATA <- list(time=DATA.time,num_time_points=DATA.num_time_points,
               totDATA=DATA.totDATA,timeline=DATA.timeline,numGENES=DATA.numGENES,
               genes=DATA.genes,singleCELLdata=DATA.singleCELLdata)
  return(DATA)
}

SINCERITITES <- dget("/home/chenxufeng/WorkSpace/scMagnify/scMagnify-benchmark/baseline/workflow/SINCERITIES.R")

main <- function(args) {
    ## Configurations
    dirPjtHome <- args$dirPjtHome
    algoWorkDir <- file.path(dirPjtHome, "tmp", "sincerities_wd")
    tmpSaveDir <- file.path(dirPjtHome, "tmp", "sincerities_wd", args$version)
    if (args$tmp_save) {
        dir.create(tmpSaveDir, showWarnings = FALSE, recursive = TRUE)
    }

    benchmarkDir <- file.path(dirPjtHome, "benchmark", args$version)
    if (!dir.exists(benchmarkDir)) {
        dir.create(benchmarkDir, showWarnings = FALSE, recursive = TRUE)
        dir.create(file.path(benchmarkDir, "net"), showWarnings = FALSE, recursive = TRUE)
        dir.create(file.path(benchmarkDir, "log"), showWarnings = FALSE, recursive = TRUE)
    }

    # Setting the logger to INFO
    log_file <- file.path(benchmarkDir, "log", "SINCERITIES.log")
    basicConfig()
    addHandler(writeToFile, file = log_file, level = "INFO")

    set.seed(args$seed)

    loginfo(paste("Benchmark version:", args$version, "with seed:", args$seed))
    loginfo(paste("Packages Version:", packageVersion("Seurat")))
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
    loginfo(paste("[1/2] Loading the data from", args$input))
    obj <- readRDS(file.path(args$dirPjtHome, "benchmark", "data", paste0(args$dataset, ".rds")))
    # log_memory_usage()

    # Intersect the cell list with the obj colnames
    cell_selected <- cell_selected[rownames(cell_selected) %in% colnames(obj), ]

    loginfo("[2/2] Running SIN and saving the results...")
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

        ExprMatrix <- t(as.matrix(obj_lin@assays$RNA@counts))
        rownames(ExprMatrix) <- colnames(obj_lin@assays$RNA@counts)
        colnames(ExprMatrix) <- rownames(obj_lin@assays$RNA@counts)

        ExprMatrix <- ExprMatrix[,colnames(ExprMatrix) %in% gene_selected$V1]

        Pseudotime <- as.numeric(obj_lin$palantir_pseudotime)

        sds <- apply(ExprMatrix, 2, sd)
        ExprMatrix<- ExprMatrix[, which(sds > quantile(sds, prob = 0.2))]

        sorted_indices <- order(Pseudotime)
        bin_indices <- cut(seq_along(sorted_indices), breaks = 10, labels = FALSE) - 1

        Pseudotime$time <- NA
        Pseudotime$time[sorted_indices] <- bin_indices

        DATA <- preprocess(cbind(ExprMatrix, Pseudotime$time))

        SIGN <- 0
        result <- SINCERITITES(DATA,distance=1,method = 1,noDIAG = 0,SIGN = SIGN)

        adj_matrix <- result$adj_matrix
        dimnames(adj_matrix) <- list(DATA$genes,DATA$genes)

        edge_df <- melt(adj_matrix, varnames = c("TF", "Target"), value.name = "Score")

        write.csv(edge_df, file = file.path(benchmarkDir, "net", paste0("SINCERITIES_", lin, ".csv")), row.names = FALSE, quote = FALSE)

        loginfo(paste("SINCERITITES for", lin, "done"))
        # log_memory_usage()

        if (args$tmp_save) {
            saveRDS(result, file = paste0(tmpSaveDir, "/SINCERITITES_", lin, ".rds"))
        }
    
    }

}

# Execute main function
main(parse_args())


    