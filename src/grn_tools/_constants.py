from pathlib import Path

ROOT = Path(__file__).parents[3].resolve()

DATA_DIR = "/mnt/TrueNas/project/chenxufeng/Data/"




# ============== RealData Benchmarking ==============

DATASETS = {"PMID36973557_NatBiotechnol2023_CD34":['240921',['Ery', 'Mono', 'CLP', 'Mega', 'cDC', 'pDC']],
            "PMID36973557_NatBiotechnol2023_T-cell-depleted": ['240704', ['Ery', 'Mono', 'NaiveB']],
            "bioRxiv_Klein2023_Pancreas": ['240914', ['Alpha', 'Beta', 'Delta', 'Epsilon']]}


DIRPJTHOMES = {
    "PMID36973557_NatBiotechnol2023_CD34": "/mnt/TrueNas/project/chenxufeng/Data/PMID36973557_NatBiotechnol2023_CD34/",
    "PMID36973557_NatBiotechnol2023_T-cell-depleted": "/mnt/TrueNas/project/chenxufeng/Data/PMID36973557_NatBiotechnol2023_T-cell-depleted/",
}


GT_DIR = "/mnt/TrueNas/project/chenxufeng/Database/CistromeDB/"

TISSUE_MAPPING = {
    "PMID36973557_NatBiotechnol2023_CD34": "BoneMarrowHemato",
    "PMID36973557_NatBiotechnol2023_T-cell-depleted": "BoneMarrowHemato",
    "bioRxiv_Klein2023_Pancreas": "PancreaticEndo"
}

GROUNDTRUTHS_LINEAGE = {
    ("PMID36973557_NatBiotechnol2023_CD34", "Ery"): "/mnt/TrueNas/project/chenxufeng/Database/CistromeDB/Human_Factor_hg38/BoneMarrowHemato/Cistrome_human_factor_Ery_top1000_network.csv",
    ("PMID36973557_NatBiotechnol2023_CD34", "Mono"): "/mnt/TrueNas/project/chenxufeng/Database/CistromeDB/Human_Factor_hg38/BoneMarrowHemato/Cistrome_human_factor_Mono_top1000_network.csv",
    ("PMID36973557_NatBiotechnol2023_CD34", "CLP"): "/mnt/TrueNas/project/chenxufeng/Database/CistromeDB/Human_Factor_hg38/BoneMarrowHemato/Cistrome_human_factor_CLP_top1000_network.csv",
    ("PMID36973557_NatBiotechnol2023_CD34", "Mega"): "/mnt/TrueNas/project/chenxufeng/Database/CistromeDB/Human_Factor_hg38/BoneMarrowHemato/Cistrome_human_factor_Mega_top1000_network.csv",
    ("PMID36973557_NatBiotechnol2023_CD34", "cDC"): "/mnt/TrueNas/project/chenxufeng/Database/CistromeDB/Human_Factor_hg38/BoneMarrowHemato/Cistrome_human_factor_cDC_top1000_network.csv",
    ("PMID36973557_NatBiotechnol2023_CD34", "pDC"): "/mnt/TrueNas/project/chenxufeng/Database/CistromeDB/Human_Factor_hg38/BoneMarrowHemato/Cistrome_human_factor_pDC_top1000_network.csv",
    ("PMID36973557_NatBiotechnol2023_T-cell-depleted", "Ery"): "/mnt/TrueNas/project/chenxufeng/Database/CistromeDB/Human_Factor_hg38/BoneMarrowHemato/Cistrome_human_factor_Ery_top1000_network.csv",
    ("PMID36973557_NatBiotechnol2023_T-cell-depleted", "Mono"): "/mnt/TrueNas/project/chenxufeng/Database/CistromeDB/Human_Factor_hg38/BoneMarrowHemato/Cistrome_human_factor_Mono_top1000_network.csv",
    ("PMID36973557_NatBiotechnol2023_T-cell-depleted", "NaiveB"): "/mnt/TrueNas/project/chenxufeng/Database/CistromeDB/Human_Factor_hg38/BoneMarrowHemato/Cistrome_human_factor_NaiveB_top1000_network.csv"
}

GROUNDTRUTHS_TISSUE = {
    ("PMID36973557_NatBiotechnol2023_CD34", "Ery"): "/mnt/TrueNas/project/chenxufeng/Database/CistromeDB/Human_Factor_hg38/BoneMarrowHemato/Cistrome_human_factor_Blood_top1000_network.csv",
    ("PMID36973557_NatBiotechnol2023_CD34", "Mono"): "/mnt/TrueNas/project/chenxufeng/Database/CistromeDB/Human_Factor_hg38/BoneMarrowHemato/Cistrome_human_factor_Blood_top1000_network.csv",
    ("PMID36973557_NatBiotechnol2023_CD34", "CLP"): "/mnt/TrueNas/project/chenxufeng/Database/CistromeDB/Human_Factor_hg38/BoneMarrowHemato/Cistrome_human_factor_Blood_top1000_network.csv",
    ("PMID36973557_NatBiotechnol2023_CD34", "Mega"): "/mnt/TrueNas/project/chenxufeng/Database/CistromeDB/Human_Factor_hg38/BoneMarrowHemato/Cistrome_human_factor_Blood_top1000_network.csv",
    ("PMID36973557_NatBiotechnol2023_CD34", "cDC"): "/mnt/TrueNas/project/chenxufeng/Database/CistromeDB/Human_Factor_hg38/BoneMarrowHemato/Cistrome_human_factor_Blood_top1000_network.csv",
    ("PMID36973557_NatBiotechnol2023_CD34", "pDC"): "/mnt/TrueNas/project/chenxufeng/Database/CistromeDB/Human_Factor_hg38/BoneMarrowHemato/Cistrome_human_factor_Blood_top1000_network.csv",
    ("PMID36973557_NatBiotechnol2023_T-cell-depleted", "Ery"): "/mnt/TrueNas/project/chenxufeng/Database/CistromeDB/Human_Factor_hg38/BoneMarrowHemato/Cistrome_human_factor_Blood_top1000_network.csv",
    ("PMID36973557_NatBiotechnol2023_T-cell-depleted", "Mono"): "/mnt/TrueNas/project/chenxufeng/Database/CistromeDB/Human_Factor_hg38/BoneMarrowHemato/Cistrome_human_factor_Blood_top1000_network.csv",
    ("PMID36973557_NatBiotechnol2023_T-cell-depleted", "NaiveB"): "/mnt/TrueNas/project/chenxufeng/Database/CistromeDB/Human_Factor_hg38/BoneMarrowHemato/Cistrome_human_factor_Blood_top1000_network.csv",
    ("bioRxiv_Klein2023_Pancreas", "Alpha"): "/mnt/TrueNas/project/chenxufeng/Database/CistromeDB/Mouse_Factor_mm10/PancreaticEndo/Cistrome_mouse_factor_Pancreas_top1000_network.csv",
    ("bioRxiv_Klein2023_Pancreas", "Beta"): "/mnt/TrueNas/project/chenxufeng/Database/CistromeDB/Mouse_Factor_mm10/PancreaticEndo/Cistrome_mouse_factor_Pancreas_top1000_network.csv",
    ("bioRxiv_Klein2023_Pancreas", "Delta"): "/mnt/TrueNas/project/chenxufeng/Database/CistromeDB/Mouse_Factor_mm10/PancreaticEndo/Cistrome_mouse_factor_Pancreas_top1000_network.csv",
    ("bioRxiv_Klein2023_Pancreas", "Epsilon"): "/mnt/TrueNas/project/chenxufeng/Database/CistromeDB/Mouse_Factor_mm10/PancreaticEndo/Cistrome_mouse_factor_Pancreas_top1000_network.csv"
}


# ============== scMultiSim Benchmarking ==============

NET_DIR_SCMULTISIM = "/mnt/TrueNas/project/chenxufeng/Data/scMultiSim/bench_grn/net/"

GROUNDTRUTHS_SCMULTISIM = {"grn100": "/mnt/TrueNas/project/chenxufeng/Data/scMultiSim/bench_grn/net/GRN_100.csv",
                           "grn1139": "/mnt/TrueNas/project/chenxufeng/Data/scMultiSim/bench_grn/net/GRN_1139.csv"}


# ============== scMAGNIFY Benchmarking ==============

def generate_groundtruths(dataset_name, cell_types, tissue_type):
    groundtruths = {}
    for cell_type in cell_types:
        if tissue_type == "BoneMarrowHemato":
            path = f"{GT_DIR}Human_Factor_hg38/{tissue_type}/Cistrome_human_factor_{cell_type}_top1000_network.csv"
        elif tissue_type == "PancreaticEndo":
            path = f"{GT_DIR}Mouse_Factor_mm10/{tissue_type}/Cistrome_mouse_factor_Pancreas_top1000_network.csv"
        groundtruths[(dataset_name, cell_type)] = path
    return groundtruths

# GROUNDTRUTHS_LINEAGE = {}
# for dataset, (_, cell_types) in DATASETS.items():
#     if dataset.startswith("PMID36973557"):
#         GROUNDTRUTHS_LINEAGE.update(generate_groundtruths(dataset, cell_types, "BoneMarrowHemato"))

# GROUNDTRUTHS_TISSUE = {}
# for dataset, (_, cell_types) in DATASETS.items():
#     if dataset.startswith("PMID36973557"):
#         GROUNDTRUTHS_TISSUE.update(generate_groundtruths(dataset, cell_types, "BoneMarrowHemato"))
#     elif dataset.startswith("bioRxiv_Klein2023"):
#         GROUNDTRUTHS_TISSUE.update(generate_groundtruths(dataset, cell_types, "PancreaticEndo"))

# ============== Plotting ==============

METHOD_PALETTE_SCMULTISIM = {
    "scMagnify": "#ff7f0e",
    "CellOracle": "#1f77b4",
}

METHOD_PALETTE_CISTROME = {
    "scMagnify": "#ff7f0e", 
    "scMagnify-nobasal": "#F6AE2D", 
    "BasalGRN": "#E8AF7F",
    "Dictys": "#e377c2", 
    "FigR": "#2ca02c", 
    "CellOracle": "#1f77b4", 
    "SCENIC": "#d62728", 
    "Velorama": "#8BBEB2",
    "SINCERITIES": "#17becf",
    "GRNBoost2": "#CC7061",
    "Pando": "#9471C9",
    "LINGER": "#8c564b"
}

METHOD_ORDER_CISTROME = [
    "scMagnify", 
    "scMagnify-nobasal", 
    "BasalGRN",
    "LINGER",
    "Dictys", 
    "CellOracle", 
    "FigR", 
    "Pando",
    "SCENIC", 
    "SINCERITIES",
    "GRNBoost2",
]