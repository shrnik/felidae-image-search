"""
Compute SigLIP-2 embeddings for the Felidae Conservation Fund camera trap dataset.

Model: google/siglip2-base-patch16-224 loaded with bitsandbytes INT8 quantization,
matching the INT8 quantization used by the ONNX text model at search time in JS
(onnx-community/siglip2-base-patch16-224-ONNX).

Usage:
    # Process from locally downloaded images (faster, recommended):
    python compute_embeddings_siglip2.py --mode local --image-dir /path/to/images

    # Resume an interrupted run:
    python compute_embeddings_siglip2.py --mode local --image-dir /path/to/images --resume

    # Sanity check — embed the first image and print dimension/norm:
    python compute_embeddings_siglip2.py --mode local --image-dir /path/to/images --dry-run

    # Stream images from Azure (no local disk needed):
    python compute_embeddings_siglip2.py --mode stream
"""

import argparse
import csv
import json
import time
from io import BytesIO
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests
import torch
from PIL import Image
from transformers import AutoModel, BitsAndBytesConfig, SiglipProcessor


# ── Config ──────────────────────────────────────────────────────────────────
MODEL_ID = "google/siglip2-base-patch16-224"
EMBEDDING_DIM = 768

METADATA_JSON = "felidae_conservation_fund_2020_2025.json"
BASE_URL = "https://lilawildlife.blob.core.windows.net/lila-wildlife/felidae-conservation-fund/"

OUTPUT_DIR = Path("data/siglip2")
EMBEDDINGS_FILE = OUTPUT_DIR / "embeddings.bin"
METADATA_CSV = OUTPUT_DIR / "metadata.csv"
CHECKPOINT_FILE = OUTPUT_DIR / "checkpoint.json"
FAILED_FILE = OUTPUT_DIR / "failed.json"

BATCH_SIZE = 256
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
        except Exception:
            if attempt == MAX_RETRIES - 1:
                raise
            time.sleep(2 ** attempt)


def fetch_image_local(image_dir: Path, file_name: str) -> Image.Image:
    return Image.open(image_dir / file_name).convert("RGB")


def embed_batch(
    model: AutoModel,
    processor: SiglipProcessor,
    images: list[Image.Image],
) -> np.ndarray:
    inputs = processor(images=images, return_tensors="pt").to(model.device)
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
    image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
    return image_features.cpu().numpy()


def run(args: argparse.Namespace) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    images, img_to_cat = load_coco_metadata(args.metadata)

    print(f"Loading {MODEL_ID} with INT8 quantization ...")
    bnb_config = BitsAndBytesConfig(load_in_4bit=True)
    model = AutoModel.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation="sdpa"
    )
    processor = SiglipProcessor.from_pretrained(MODEL_ID)
    model.eval()

    # ── Resume support ───────────────────────────────────────────────────────
    start_index = load_checkpoint() if args.resume else 0
    if start_index > 0:
        print(f"Resuming from image index {start_index:,}")
    else:
        for f in [EMBEDDINGS_FILE, METADATA_CSV, CHECKPOINT_FILE, FAILED_FILE]:
            if f.exists():
                f.unlink()

    failed: list[tuple[str, str]] = []
    processed = start_index

    image_dir = Path(args.image_dir) if args.image_dir else None
    todo = images[start_index:start_index + 1] if args.dry_run else images[start_index:]

    def load_one(img_info: dict) -> tuple[dict, Image.Image | None, str]:
        file_name = img_info["file_name"]
        try:
            if args.mode == "local":
                return img_info, fetch_image_local(image_dir, file_name), ""
            else:
                return img_info, fetch_image_remote(BASE_URL + file_name), ""
        except Exception as e:
            return img_info, None, str(e)

    print(f"Processing {len(todo):,} images ...")
    with ThreadPoolExecutor(max_workers=args.io_workers) as pool:
        chunk = BATCH_SIZE * 2
        for batch_start in range(0, len(todo), chunk):
            batch_infos = todo[batch_start : batch_start + chunk]
            results = list(pool.map(load_one, batch_infos))

            batch_images: list[Image.Image] = []
            batch_meta: list[dict] = []
            last_idx = batch_start

            for offset, (img_info, image, err) in enumerate(results):
                file_name = img_info["file_name"]
                if image is None:
                    if len(failed) < 5:
                        print(f"  FAILED {file_name}: {err}")
                    failed.append((file_name, err))
                    continue

                batch_images.append(image)
                batch_meta.append({
                    "id": processed,
                    "file_name": file_name,
                    "category": img_to_cat.get(img_info["id"], "unknown"),
                })
                processed += 1
                last_idx = batch_start + offset

                if len(batch_images) == BATCH_SIZE:
                    emb = embed_batch(model, processor, batch_images)
                    append_embeddings(emb)
                    append_metadata(batch_meta)
                    batch_images, batch_meta = [], []

            if batch_images:
                emb = embed_batch(model, processor, batch_images)
                append_embeddings(emb)
                append_metadata(batch_meta)

            save_checkpoint(start_index + last_idx + 1)
            if failed:
                with open(FAILED_FILE, "w") as f:
                    json.dump(failed, f)
            print(f"  {processed:,} / {len(images):,} done")

    # ── Final save ───────────────────────────────────────────────────────────
    save_checkpoint(len(images))
    with open(FAILED_FILE, "w") as f:
        json.dump(failed, f)

    if args.dry_run:
        emb_check = np.frombuffer(open(EMBEDDINGS_FILE, "rb").read(), dtype=np.float32)
        norm = float((emb_check ** 2).sum() ** 0.5)
        print(f"\nDry-run complete.")
        print(f"  Dimension: {emb_check.shape[0]}")
        print(f"  L2 norm:   {norm:.6f} ({'ok' if abs(norm - 1.0) < 1e-3 else 'NOT normalized'})")
    else:
        total_rows = len(load_existing_metadata())
        print(f"\nDone! {total_rows:,} embeddings saved, {len(failed):,} failed.")
        print(f"  Embeddings: {EMBEDDINGS_FILE}")
        print(f"  Metadata:   {METADATA_CSV}")
        if failed:
            print(f"  Failed:     {FAILED_FILE}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=f"Compute {MODEL_ID} embeddings for the Felidae dataset"
    )
    parser.add_argument("--metadata", default=METADATA_JSON,
                        help="Path to COCO Camera Traps JSON (default: %(default)s)")
    parser.add_argument("--mode", choices=["stream", "local"], default="stream",
                        help="stream = fetch from Azure; local = read from disk (default: stream)")
    parser.add_argument("--image-dir", default=None,
                        help="Root directory of locally downloaded images (required for --mode=local)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last checkpoint")
    parser.add_argument("--io-workers", type=int, default=16,
                        help="Threads for parallel image loading (default: 16)")
    parser.add_argument("--quantize", choices=["4bit", "8bit", "none"], default="none",
                        help="bitsandbytes quantization: 4bit, 8bit, or none (default: none)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Embed only the first image and print dimension/norm, then exit")
    args = parser.parse_args()

    if args.mode == "local" and not args.image_dir and not args.dry_run:
        parser.error("--image-dir is required when --mode=local")

    run(args)


if __name__ == "__main__":
    main()
