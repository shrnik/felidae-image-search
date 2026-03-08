"""
Upload embeddings.bin and metadata.csv to a HuggingFace Hub dataset repo.

Usage:
    huggingface-cli login      # one-time auth
    python upload_to_hf.py --repo your-username/felidae-image-search

The browser app can then fetch the files directly from:
    https://huggingface.co/datasets/<repo>/resolve/main/embeddings.bin
    https://huggingface.co/datasets/<repo>/resolve/main/metadata.csv
"""

import argparse
from pathlib import Path

from huggingface_hub import HfApi, create_repo

DATA_DIR = Path("data")
FILES = ["embeddings.bin", "metadata.csv"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload Felidae embeddings to HuggingFace Hub")
    parser.add_argument("--repo", required=True, help="HuggingFace dataset repo id, e.g. username/felidae-image-search")
    parser.add_argument("--private", action="store_true", help="Create a private repo")
    args = parser.parse_args()

    api = HfApi()

    print(f"Creating/verifying repo: {args.repo}")
    create_repo(args.repo, repo_type="dataset", private=args.private, exist_ok=True)

    for fname in FILES:
        local = DATA_DIR / fname
        if not local.exists():
            print(f"  SKIP {fname} (not found)")
            continue
        size_mb = local.stat().st_size / 1e6
        print(f"  Uploading {fname} ({size_mb:.1f} MB) ...")
        api.upload_file(
            path_or_fileobj=str(local),
            path_in_repo=fname,
            repo_id=args.repo,
            repo_type="dataset",
        )
        print(f"  Done: {fname}")

    base = f"https://huggingface.co/datasets/{args.repo}/resolve/main"
    print(f"\nUpload complete. Files available at:")
    for fname in FILES:
        print(f"  {base}/{fname}")


if __name__ == "__main__":
    main()
