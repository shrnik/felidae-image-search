#!/bin/bash
#SBATCH -o logs/uwisc_batch_%j.out
#SBATCH -e logs/uwisc_batch_%j.err
#SBATCH -N 1
#SBATCH -p superChip
#SBATCH --cpus-per-task=72
#SBATCH --mem=128G
#SBATCH --mail-user=s.borad@gwu.edu
#SBATCH --mail-type=ALL
#SBATCH -t 24:00:00

source ~/miniconda3/etc/profile.d/conda.sh
conda activate contrail_detection

REPO_DIR="$HOME/felidae-image-search"
DATA_DIR="$REPO_DIR/data"
IMAGE_DIR="$DATA_DIR/images"
METADATA_ZIP="$DATA_DIR/felidae_conservation_fund_2020_2025.zip"
METADATA_JSON="$REPO_DIR/felidae_conservation_fund_2020_2025.json"




python "$REPO_DIR/scripts/compute_embeddings_siglip2.py" \
    --mode local \
    --image-dir "$IMAGE_DIR" \
    --metadata "$METADATA_JSON" \