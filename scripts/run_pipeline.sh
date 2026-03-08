#!/bin/bash

#SBATCH -o download%j.out
#SBATCH -e download%j.err
#SBATCH -N 1
#SBATCH -p superChip
#SBATCH --mail-user=s.borad@gwu.edu
#SBATCH --mail-type=ALL
#SBATCH -t 10:00:00
#SBATCH --cpus-per-task=16

source ~/miniconda3/etc/profile.d/conda.sh
conda activate contrail_detection


# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_DIR="$HOME/felidae-image-search"
DATA_DIR="$REPO_DIR/data"
IMAGE_DIR="$DATA_DIR/images"
METADATA_ZIP="$DATA_DIR/felidae_conservation_fund_2020_2025.zip"
METADATA_JSON="$REPO_DIR/felidae_conservation_fund_2020_2025.json"

mkdir -p "$DATA_DIR" "$IMAGE_DIR"
cd "$REPO_DIR"

# ── Step 1: Download metadata JSON ────────────────────────────────────────────
if [ ! -f "$METADATA_JSON" ]; then
    echo "[$(date)] Downloading metadata..."
    wget -q -O "$METADATA_ZIP" \
        "https://lilawildlife.blob.core.windows.net/lila-wildlife/felidae-conservation-fund/felidae_conservation_fund_2020_2025.zip"
    unzip -q "$METADATA_ZIP" -d "$DATA_DIR"
    echo "[$(date)] Metadata ready: $METADATA_JSON"
else
    echo "[$(date)] Metadata already exists, skipping download."
fi

# ── Step 2: Download images (parallel, skips existing) ───────────────────────
echo "[$(date)] Downloading images with $SLURM_CPUS_PER_TASK workers..."
python scripts/download_images.py \
    --metadata "$METADATA_JSON" \
    --out-dir   "$IMAGE_DIR" \
    --workers   "$SLURM_CPUS_PER_TASK"

echo "[$(date)] Image download complete."

# ── Step 3: Compute CLIP embeddings ──────────────────────────────────────────
echo "[$(date)] Computing embeddings..."
python scripts/compute_embeddings.py \
    --metadata   "$METADATA_JSON" \
    --mode       local \
    --image-dir  "$IMAGE_DIR" \
    --io-workers "$SLURM_CPUS_PER_TASK" \
    --resume

echo "[$(date)] Embeddings written to $DATA_DIR/embeddings.bin"
echo "[$(date)] Pipeline complete."
