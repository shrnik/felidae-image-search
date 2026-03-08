"""
Download Felidae Conservation Fund images from Azure blob storage.

Uses parallel threads with requests (no azcopy or Azure SDK needed —
blobs are publicly accessible over HTTPS).

Usage:
    python download_images.py --metadata felidae_conservation_fund_2020_2025.json \
                              --out-dir ./data/images \
                              --workers 16

Resume: already-downloaded files are skipped automatically.
"""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from tqdm import tqdm

BASE_URL = "https://lilawildlife.blob.core.windows.net/lila-wildlife/felidae-conservation-fund/"
MAX_RETRIES = 3
TIMEOUT = 20


def download_one(file_name: str, out_dir: Path) -> tuple[str, bool, str]:
    """Download a single image. Returns (file_name, success, error_msg)."""
    dest = out_dir / file_name
    if dest.exists():
        return file_name, True, "skipped"

    dest.parent.mkdir(parents=True, exist_ok=True)
    url = BASE_URL + file_name

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, timeout=TIMEOUT, stream=True)
            resp.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in resp.iter_content(chunk_size=65536):
                    f.write(chunk)
            return file_name, True, ""
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                return file_name, False, str(e)
            time.sleep(2 ** attempt)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Felidae images from Azure")
    parser.add_argument(
        "--metadata",
        default="felidae_conservation_fund_2020_2025.json",
        help="Path to COCO Camera Traps JSON",
    )
    parser.add_argument(
        "--out-dir",
        default="data/images",
        help="Root directory to save images (default: data/images)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Number of parallel download threads (default: 16)",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading metadata from {args.metadata} ...")
    with open(args.metadata) as f:
        coco = json.load(f)
    file_names = [img["file_name"] for img in coco["images"]]
    print(f"  {len(file_names):,} images total")

    # Skip already-downloaded files
    todo = [fn for fn in file_names if not (out_dir / fn).exists()]
    print(f"  {len(file_names) - len(todo):,} already downloaded, {len(todo):,} remaining")

    if not todo:
        print("Nothing to do.")
        return

    failed = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download_one, fn, out_dir): fn for fn in todo}
        with tqdm(total=len(todo), unit="img", dynamic_ncols=True) as pbar:
            for fut in as_completed(futures):
                fn, ok, err = fut.result()
                if not ok:
                    failed.append((fn, err))
                pbar.update(1)
                if failed and len(failed) % 100 == 0:
                    pbar.set_postfix(failed=len(failed))

    print(f"\nDone. {len(todo) - len(failed):,} downloaded, {len(failed):,} failed.")
    if failed:
        failed_path = out_dir / "failed_downloads.json"
        with open(failed_path, "w") as f:
            json.dump(failed, f, indent=2)
        print(f"Failed list saved to {failed_path}")


if __name__ == "__main__":
    main()
