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

get_rna_feature_aliases <- function(obj) {
    features <- rownames(obj[["RNA"]])
    aliases <- features

    meta_features <- tryCatch(obj[["RNA"]]@meta.features, error = function(e) NULL)
    if (!is.null(meta_features) && nrow(meta_features) > 0) {
        candidate_cols <- c(
            "gene_name", "gene", "symbol", "gene_symbol",
            "SYMBOL", "GeneSymbol", "feature_name"
        )
        use_cols <- intersect(candidate_cols, colnames(meta_features))
        for (col in use_cols) {
            aliases <- c(aliases, as.character(meta_features[[col]]))
        }
    }

    aliases <- aliases[!is.na(aliases) & aliases != ""]
    unique(aliases)
}

build_mouse_motif_resources <- function(obj) {
    data("mouse_pwms_v2")

    motif_names <- names(mouse_pwms_v2@listData)
    motif_tf <- vapply(mouse_pwms_v2@listData, function(x) x@name, character(1))

    motif2tf <- data.frame(
        motif = motif_names,
        tf = motif_tf,
        origin = "CIS-BP",
        gene_id = gsub("_[[:alnum:][:punct:]]*", "", motif_names),
        family = NA_character_,
        name = NA_character_,
        symbol = NA_character_,
        motif_tf = NA_character_,
        stringsAsFactors = FALSE
    )
    motif2tf <- subset(motif2tf, gene_id != "XP" & gene_id != "NP")

    aliases <- get_rna_feature_aliases(obj)
    alias_map <- data.frame(
        alias = aliases,
        alias_lower = tolower(aliases),
        stringsAsFactors = FALSE
    )
    alias_map <- alias_map[!duplicated(alias_map$alias_lower), , drop = FALSE]

    match_tf <- match(tolower(motif2tf$tf), alias_map$alias_lower)
    match_gene_id <- match(tolower(motif2tf$gene_id), alias_map$alias_lower)
    use_gene_id <- is.na(match_tf) & !is.na(match_gene_id)

    motif2tf$tf[!is.na(match_tf)] <- alias_map$alias[match_tf[!is.na(match_tf)]]
    motif2tf$tf[use_gene_id] <- alias_map$alias[match_gene_id[use_gene_id]]

    keep <- !is.na(match_tf) | !is.na(match_gene_id)
    motif2tf <- motif2tf[keep, , drop = FALSE]

    if (nrow(motif2tf) == 0) {
        stop("No motif TFs could be mapped to RNA features for mm10. Check RNA feature naming (symbol vs Ensembl IDs).")
    }

    motifs <- subset(mouse_pwms_v2, names(mouse_pwms_v2@listData) %in% motif2tf$motif)
    list(motifs = motifs, motif2tf = motif2tf)
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
        genome_ref <- BSgenome.Hsapiens.UCSC.hg38
        data('motifs')
        data('motif2tf')
        data('phastConsElements20Mammals.UCSC.hg38')
    } else if (args$ref_genome == "mm10") {
        library(chromVARmotifs)
        library(BSgenome.Mmusculus.UCSC.mm10)
        main.chroms <- standardChromosomes(BSgenome.Mmusculus.UCSC.mm10)
        genome_ref <- BSgenome.Mmusculus.UCSC.mm10
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

    gene_selected <- read.csv(args$genelist, header = FALSE, stringsAsFactors = FALSE)
    loginfo(paste("Number of genes:", nrow(gene_selected)))
    loginfo(paste("Number of genes containing 'Rik':", sum(grepl("Rik", gene_selected[[1]]))))
    gene_selected <- gene_selected[!grepl("Rik", gene_selected[[1]]), , drop = FALSE]
    loginfo(paste("Number of genes after removing 'Rik':", nrow(gene_selected)))
    # log_memory_usage()

    cell_selected <- read.csv(args$celllist, row.names = 1)
    loginfo(paste("Cell list:", args$celllist))
    loginfo(paste("Number of cells:", nrow(cell_selected)))
    # log_memory_usage()

    ## Load data
    loginfo(paste("[1/3] Loading the data from", args$dataset))
    obj <- readRDS(file.path(args$dirPjtHome, "benchmark", "data", paste0(args$dataset, ".rds")))

    if (args$ref_genome == "mm10") {
        mouse_motifs <- build_mouse_motif_resources(obj)
        motifs <- mouse_motifs$motifs
        motif2tf <- mouse_motifs$motif2tf
        loginfo(paste(
            "Using mm10 mouse_pwms_v2 motif DB with",
            length(motifs@listData), "motifs and",
            nrow(motif2tf), "TF mappings matched to RNA features"
        ))
    }

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
            genome = genome_ref
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