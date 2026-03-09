# Felidae Image Search

Semantic search over 357k+ camera trap images from the [Felidae Conservation Fund](https://www.felidaefund.org/) dataset, powered by [CLIP](https://openai.com/research/clip) and running entirely in the browser.

**[Live demo →](https://shrnik.github.io/felidae-image-search/)**

## How it works

1. CLIP image embeddings are precomputed for all 357k images on a GPU cluster and stored as a raw float32 binary (`embeddings.bin`) on HuggingFace Hub.
2. On first visit the browser downloads `embeddings.bin` (~686 MB) and `metadata.csv` and caches them via the Cache API — subsequent visits are instant.
3. On each search query, the CLIP text encoder runs in-browser via [Transformers.js](https://github.com/xenova/transformers.js) to encode the query, then a dot-product over all image embeddings returns the top-K results.
4. Images are served directly from GCS (no backend needed).

## Dataset

- **Source**: [Felidae Conservation Fund 2020–2025](https://lila.science/datasets/felidae-conservation-fund/) via LILA Science
- **Size**: ~357,934 camera trap images
- **Labels**: 66 wildlife species (bobcat, puma, mule deer, gray fox, …)
- **Format**: COCO Camera Traps JSON

## Repo structure

```
scripts/
  download_images.py      # Download images from Azure blob storage
  compute_embeddings.py   # Compute CLIP embeddings (GPU, resumable)
  upload_to_hf.py         # Upload embeddings.bin + metadata.csv to HuggingFace Hub
  run_pipeline.sh         # Slurm batch script (end-to-end pipeline)
web/
  index.html              # Browser search UI
  search.js               # CLIP search logic (Transformers.js)
```

## Reproduce the embeddings

### 1. Download metadata

```bash
wget https://lilawildlife.blob.core.windows.net/lila-wildlife/felidae-conservation-fund/felidae_conservation_fund_2020_2025.zip
unzip felidae_conservation_fund_2020_2025.zip
```

### 2. Download images

```bash
python scripts/download_images.py \
  --metadata felidae_conservation_fund_2020_2025.json \
  --out-dir data/images/ \
  --workers 16
```

### 3. Compute embeddings

```bash
python scripts/compute_embeddings.py \
  --metadata felidae_conservation_fund_2020_2025.json \
  --mode local \
  --image-dir data/images/ \
  --io-workers 16 \
  --resume
```

Add `--dry-run` to process only the first image as a sanity check.

### 4. Upload to HuggingFace Hub

```bash
huggingface-cli login
python scripts/upload_to_hf.py --repo your-username/felidae-image-search
```

### On a Slurm cluster

```bash
sbatch scripts/run_pipeline.sh
```

## Requirements

```bash
pip install -r requirements.txt
```

- Python 3.10+
- PyTorch 2.0+
- `transformers`, `Pillow`, `requests`, `numpy`, `huggingface_hub`, `tqdm`

## Model

[`openai/clip-vit-base-patch32`](https://huggingface.co/openai/clip-vit-base-patch32) — 512-dimensional embeddings, in-browser text encoding via `Xenova/clip-vit-base-patch32`.
