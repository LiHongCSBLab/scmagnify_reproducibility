#!/bin/bash

# 检查参数数量
if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <input_dir> <ref_genome>"
  echo "Supported reference genomes: hg38, mm10"
  exit 1
fi

# 设置输入目录和参考基因组
INPUT_DIR="$1"
REF_GENOME="$2"

# 设置数据库根目录
DB_ROOT="/home/chenxufeng/picb_cxf/Ref"

# 根据参考基因组设置数据库路径
if [ "${REF_GENOME}" == "hg38" ]; then
  DB_DIR="${DB_ROOT}/human/hg38/cisTarget_db/"
  ALL_TFS="${DB_DIR}/allTFs_hg38.txt"
  FEATHER_1="${DB_DIR}/hg38_500bp_up_100bp_down_full_tx_v10_clust.genes_vs_motifs.scores.feather"
  FEATHER_2="${DB_DIR}/hg38_10kbp_up_10kbp_down_full_tx_v10_clust.genes_vs_motifs.rankings.feather"
  FEATHER_3="${DB_DIR}/hg38_500bp_up_100bp_down_full_tx_v10_clust.genes_vs_motifs.rankings.feather"
  FEATHER_4="${DB_DIR}/hg38_500bp_up_100bp_down_full_tx_v10_clust.genes_vs_motifs.scores.feather"
  ANNOTATIONS="${DB_DIR}/motifs-v10-nr.hgnc-m0.00001-o0.0.tbl"
elif [ "${REF_GENOME}" == "mm10" ]; then
  DB_DIR="${DB_ROOT}/mouse/mm10/cisTarget_db/"
  ALL_TFS="${DB_DIR}/allTFs_mm10.txt"
  FEATHER_1="${DB_DIR}/mm10_500bp_up_100bp_down_full_tx_v10_clust.genes_vs_motifs.scores.feather"
  FEATHER_2="${DB_DIR}/mm10_10kbp_up_10kbp_down_full_tx_v10_clust.genes_vs_motifs.rankings.feather"
  FEATHER_3="${DB_DIR}/mm10_500bp_up_100bp_down_full_tx_v10_clust.genes_vs_motifs.rankings.feather"
  FEATHER_4="${DB_DIR}/mm10_500bp_up_100bp_down_full_tx_v10_clust.genes_vs_motifs.scores.feather"
  ANNOTATIONS="${DB_DIR}/motifs-v10-nr.mgi-m0.00001-o0.0.tbl"
else
  echo "Unsupported reference genome: ${REF_GENOME}"
  echo "Supported reference genomes: hg38, mm10"
  exit 1
fi

# 检查输入目录是否存在
if [ ! -d "${INPUT_DIR}" ]; then
  echo "Input directory does not exist: ${INPUT_DIR}"
  exit 1
fi

# 遍历输入目录下的所有 .loom 文件
for LOOM_FILE in "${INPUT_DIR}"/*.loom; do
  # 获取文件名
  FILENAME=$(basename "${LOOM_FILE}")
  
  # 提取输出目录名（最后一个下划线后的字符串，去掉 .loom 后缀）
  OUTPUT_DIR_NAME=$(echo "${FILENAME}" | rev | cut -d'_' -f1 | rev | sed 's/\.loom$//')
  OUTPUT_DIR="${INPUT_DIR}/${OUTPUT_DIR_NAME}"

  # 检查输出目录是否存在，如果不存在则创建
  if [ ! -d "${OUTPUT_DIR}" ]; then
    mkdir -p "${OUTPUT_DIR}"
    echo "Created output directory: ${OUTPUT_DIR}"
  else
    echo "Output directory already exists: ${OUTPUT_DIR}"
  fi

  # 运行 pyscenic grn 命令
  pyscenic grn \
    --num_workers 20 \
    -o "${OUTPUT_DIR}/expr_mat.adjacencies.tsv" \
    --method grnboost2 \
    "${LOOM_FILE}" \
    "${ALL_TFS}" 

  # 运行 pyscenic ctx 命令
  pyscenic ctx \
    "${OUTPUT_DIR}/expr_mat.adjacencies.tsv" \
    "${FEATHER_1}" \
    "${FEATHER_2}" \
    "${FEATHER_3}" \
    "${FEATHER_4}" \
    --annotations_fname "${ANNOTATIONS}" \
    --expression_mtx_fname "${LOOM_FILE}" \
    --mode "dask_multiprocessing" \
    --min_genes 10 \
    --output "${OUTPUT_DIR}/regulons.csv" \
    --num_workers 20 \
    --mask_dropouts

  echo "Processing completed for ${FILENAME}. Output saved to ${OUTPUT_DIR}"
done