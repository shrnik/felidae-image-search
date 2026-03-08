"""
Compute CLIP embeddings for the Felidae Conservation Fund camera trap dataset.

Usage:
    # Stream images from Azure blob (slower, no local disk needed):
    python compute_embeddings.py --mode stream

    # Process from locally downloaded images (faster, recommended):
    python compute_embeddings.py --mode local --image-dir /path/to/images

    # Resume an interrupted run:
    python compute_embeddings.py --mode local --image-dir /path/to/images --resume

Download images locally first with azcopy (much faster than streaming):
    azcopy copy \
        "https://lilawildlife.blob.core.windows.net/lila-wildlife/felidae-conservation-fund/*" \
        ./data/images/ \
        --recursive
"""

import argparse
import csv
import json
import os
import time
from io import BytesIO
from pathlib import Path

import numpy as np
import requests
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

# ── Config ──────────────────────────────────────────────────────────────────
METADATA_JSON = "felidae_conservation_fund_2020_2025.json"
BASE_URL = "https://lilawildlife.blob.core.windows.net/lila-wildlife/felidae-conservation-fund/"

OUTPUT_DIR = Path("data")
EMBEDDINGS_FILE = OUTPUT_DIR / "embeddings.bin"
METADATA_CSV = OUTPUT_DIR / "metadata.csv"
CHECKPOINT_FILE = OUTPUT_DIR / "checkpoint.json"
FAILED_FILE = OUTPUT_DIR / "failed.json"

BATCH_SIZE = 64
SAVE_EVERY = 1000   # save checkpoint every N images
REQUEST_TIMEOUT = 15
MAX_RETRIES = 3
# ────────────────────────────────────────────────────────────────────────────


def load_coco_metadata(json_path: str) -> tuple[list, dict]:
    """Parse COCO Camera Traps JSON and return (images list, image_id→category map)."""
    print(f"Loading metadata from {json_path} ...")
    with open(json_path) as f:
        coco = json.load(f)

    categories = {c["id"]: c["name"] for c in coco["categories"]}
    img_to_cat: dict[int, str] = {}
    for ann in coco["annotations"]:
        img_to_cat[ann["image_id"]] = categories[ann["category_id"]]

    images = coco["images"]
    print(f"  {len(images):,} images, {len(categories)} categories")
    return images, img_to_cat


def load_checkpoint() -> int:
    """Return the index of the next image to process (0 if no checkpoint)."""
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE) as f:
            return json.load(f)["next_index"]
    return 0


def save_checkpoint(next_index: int) -> None:
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump({"next_index": next_index}, f)


def load_existing_metadata() -> list[dict]:
    if not METADATA_CSV.exists():
        return []
    with open(METADATA_CSV, newline="") as f:
        return list(csv.DictReader(f))


def append_metadata(rows: list[dict]) -> None:
    write_header = not METADATA_CSV.exists()
    with open(METADATA_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "file_name", "category"])
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def append_embeddings(emb: np.ndarray) -> None:
    with open(EMBEDDINGS_FILE, "ab") as f:
        f.write(emb.astype(np.float32).tobytes())


def fetch_image_remote(url: str) -> Image.Image:
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            return Image.open(BytesIO(resp.content)).convert("RGB")
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                raise
            time.sleep(2 ** attempt)


def fetch_image_local(image_dir: Path, file_name: str) -> Image.Image:
    path = image_dir / file_name
    return Image.open(path).convert("RGB")


def embed_batch(
    model: CLIPModel,
    processor: CLIPProcessor,
    images: list[Image.Image],
    device: str,
) -> np.ndarray:
    inputs = processor(images=images, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        emb = model.get_image_features(**inputs)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy()


def run(args: argparse.Namespace) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    images, img_to_cat = load_coco_metadata(args.metadata)

    # ── CLIP model ──────────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()

    # ── Resume support ──────────────────────────────────────────────────────
    start_index = load_checkpoint() if args.resume else 0
    if start_index > 0:
        print(f"Resuming from image index {start_index:,}")
    else:
        # Clear stale output files if starting fresh
        for f in [EMBEDDINGS_FILE, METADATA_CSV, CHECKPOINT_FILE, FAILED_FILE]:
            if f.exists():
                f.unlink()

    failed: list[tuple[str, str]] = []
    batch_images: list[Image.Image] = []
    batch_meta: list[dict] = []
    processed = start_index  # total images processed so far (including previous runs)

    image_dir = Path(args.image_dir) if args.image_dir else None

    print(f"Processing {len(images) - start_index:,} images ...")
    for idx in range(start_index, len(images)):
        img_info = images[idx]
        file_name = img_info["file_name"]
        category = img_to_cat.get(img_info["id"], "unknown")

        try:
            if args.mode == "local":
                image = fetch_image_local(image_dir, file_name)
            else:
                image = fetch_image_remote(BASE_URL + file_name)
        except Exception as e:
            failed.append((file_name, str(e)))
            continue

        batch_images.append(image)
        batch_meta.append({"id": processed, "file_name": file_name, "category": category})
        processed += 1

        if len(batch_images) == BATCH_SIZE:
            emb = embed_batch(model, processor, batch_images, device)
            append_embeddings(emb)
            append_metadata(batch_meta)
            batch_images, batch_meta = [], []

        if processed % SAVE_EVERY == 0:
            save_checkpoint(idx + 1)
            if failed:
                with open(FAILED_FILE, "w") as f:
                    json.dump(failed, f)
            print(f"  {processed:,} / {len(images):,} done")

    # ── Flush final partial batch ────────────────────────────────────────────
    if batch_images:
        emb = embed_batch(model, processor, batch_images, device)
        append_embeddings(emb)
        append_metadata(batch_meta)

    # ── Final save ───────────────────────────────────────────────────────────
    save_checkpoint(len(images))
    with open(FAILED_FILE, "w") as f:
        json.dump(failed, f)

    total_rows = len(load_existing_metadata())
    print(f"\nDone! {total_rows:,} embeddings saved, {len(failed):,} failed.")
    print(f"  Embeddings: {EMBEDDINGS_FILE}")
    print(f"  Metadata:   {METADATA_CSV}")
    if failed:
        print(f"  Failed:     {FAILED_FILE}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute CLIP embeddings for Felidae dataset")
    parser.add_argument(
        "--metadata",
        default=METADATA_JSON,
        help="Path to COCO Camera Traps JSON (default: %(default)s)",
    )
    parser.add_argument(
        "--mode",
        choices=["stream", "local"],
        default="stream",
        help="stream = fetch images from Azure; local = read from disk (default: stream)",
    )
    parser.add_argument(
        "--image-dir",
        default=None,
        help="Root directory of locally downloaded images (required when --mode=local)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint",
    )
    args = parser.parse_args()

    if args.mode == "local" and not args.image_dir:
        parser.error("--image-dir is required when --mode=local")

    run(args)


if __name__ == "__main__":
    main()
