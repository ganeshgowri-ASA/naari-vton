#!/usr/bin/env python3
"""
Comprehensive Dataset Download Script for Jewelry Virtual Try-On

Downloads jewelry datasets from multiple sources:
- HuggingFace: raresense/jewelry-dwpose-dataset (11,147 samples with poses & masks)
- Kaggle: sapnilpatel/tanishq-jewellery-dataset (Indian jewelry)
- Roboflow: piezee/segmentation-for-jewelry-images-svv6j
- HuggingFace: Marqo/deepfashion-multimodal (has jewelry parsing labels)

Environment Variables:
    HUGGINGFACE_TOKEN: Optional. HuggingFace API token for private repos.
    KAGGLE_USERNAME: Optional. Kaggle username for API access.
    KAGGLE_KEY: Optional. Kaggle API key for API access.
    ROBOFLOW_API_KEY: Optional. Roboflow API key for dataset access.
"""

import os
import sys
import json
import shutil
import zipfile
import tarfile
import argparse
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse
import time

# Default download directory
DEFAULT_DOWNLOAD_DIR = "dataset/raw_downloads"
DEFAULT_OUTPUT_DIR = "dataset/jewelry_combined"

# Dataset configurations
DATASETS = {
    "jewelry_dwpose": {
        "name": "Jewelry DWPose Dataset",
        "source": "huggingface",
        "repo_id": "raresense/jewelry-dwpose-dataset",
        "description": "11,147 samples with poses and masks for jewelry try-on",
        "expected_samples": 11147,
        "image_subdirs": ["images", "train", "test", "val"],
    },
    "tanishq_jewelry": {
        "name": "Tanishq Jewellery Dataset",
        "source": "kaggle",
        "dataset_id": "sapnilpatel/tanishq-jewellery-dataset",
        "description": "Indian jewelry collection from Tanishq",
        "expected_samples": 500,
        "image_subdirs": ["images", "train", "jewelry"],
    },
    "roboflow_segmentation": {
        "name": "Jewelry Segmentation Dataset",
        "source": "roboflow",
        "project": "piezee/segmentation-for-jewelry-images-svv6j",
        "version": 1,
        "description": "Segmentation masks for jewelry images",
        "expected_samples": 1000,
        "image_subdirs": ["train/images", "valid/images", "test/images"],
    },
    "deepfashion_multimodal": {
        "name": "DeepFashion Multimodal",
        "source": "huggingface",
        "repo_id": "Marqo/deepfashion-multimodal",
        "description": "Fashion dataset with jewelry parsing labels",
        "expected_samples": 5000,
        "image_subdirs": ["images", "train", "data"],
        "filter_jewelry": True,  # Filter for jewelry-related items
    },
}


class DatasetDownloader:
    """Handles downloading datasets from various sources."""

    def __init__(self, download_dir: str, verbose: bool = True):
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose
        self.stats = {"downloaded": 0, "failed": 0, "skipped": 0}

    def log(self, message: str, level: str = "info"):
        """Log messages with level indicators."""
        if not self.verbose and level == "debug":
            return
        prefix = {"info": "[INFO]", "warn": "[WARN]", "error": "[ERROR]", "debug": "[DEBUG]"}
        print(f"{prefix.get(level, '[INFO]')} {message}")

    def download_file(self, url: str, dest_path: Path, chunk_size: int = 8192) -> bool:
        """Download a file with progress indication."""
        try:
            headers = {}
            hf_token = os.environ.get("HUGGINGFACE_TOKEN")
            if hf_token and "huggingface.co" in url:
                headers["Authorization"] = f"Bearer {hf_token}"

            response = requests.get(url, stream=True, headers=headers, timeout=60)
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))
            dest_path.parent.mkdir(parents=True, exist_ok=True)

            downloaded = 0
            with open(dest_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0 and self.verbose:
                            pct = (downloaded / total_size) * 100
                            print(f"\r  Progress: {pct:.1f}% ({downloaded}/{total_size})", end="")

            if self.verbose:
                print()  # New line after progress
            return True

        except requests.exceptions.RequestException as e:
            self.log(f"Failed to download {url}: {e}", "error")
            return False

    def download_huggingface(self, config: dict, dest_dir: Path) -> bool:
        """Download dataset from HuggingFace Hub."""
        repo_id = config["repo_id"]
        self.log(f"Downloading from HuggingFace: {repo_id}")

        try:
            from huggingface_hub import snapshot_download, hf_hub_download, list_repo_files

            token = os.environ.get("HUGGINGFACE_TOKEN")

            # Create destination directory
            dest_dir.mkdir(parents=True, exist_ok=True)

            # Try to list files first to check if repo exists
            try:
                files = list_repo_files(repo_id, repo_type="dataset", token=token)
                self.log(f"Found {len(files)} files in repository")
            except Exception as e:
                self.log(f"Could not list files (may require authentication): {e}", "warn")
                files = []

            # Download the entire dataset
            local_path = snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                local_dir=dest_dir,
                token=token,
                ignore_patterns=["*.md", "*.txt", "*.json", ".git*"],
            )

            self.log(f"Downloaded to: {local_path}")
            return True

        except ImportError:
            self.log("huggingface_hub not installed. Installing...", "warn")
            os.system(f"{sys.executable} -m pip install huggingface_hub")
            return self.download_huggingface(config, dest_dir)

        except Exception as e:
            self.log(f"Failed to download HuggingFace dataset {repo_id}: {e}", "error")
            return False

    def download_kaggle(self, config: dict, dest_dir: Path) -> bool:
        """Download dataset from Kaggle."""
        dataset_id = config["dataset_id"]
        self.log(f"Downloading from Kaggle: {dataset_id}")

        # Check for Kaggle credentials
        kaggle_user = os.environ.get("KAGGLE_USERNAME")
        kaggle_key = os.environ.get("KAGGLE_KEY")

        if not kaggle_user or not kaggle_key:
            # Check for kaggle.json
            kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
            if not kaggle_json.exists():
                self.log("Kaggle credentials not found. Set KAGGLE_USERNAME and KAGGLE_KEY or create ~/.kaggle/kaggle.json", "warn")
                self.log("Skipping Kaggle dataset", "warn")
                return False

        try:
            import kaggle

            dest_dir.mkdir(parents=True, exist_ok=True)

            # Download and unzip
            kaggle.api.dataset_download_files(
                dataset_id,
                path=str(dest_dir),
                unzip=True,
            )

            self.log(f"Downloaded Kaggle dataset to: {dest_dir}")
            return True

        except ImportError:
            self.log("kaggle package not installed. Installing...", "warn")
            os.system(f"{sys.executable} -m pip install kaggle")
            return self.download_kaggle(config, dest_dir)

        except Exception as e:
            self.log(f"Failed to download Kaggle dataset {dataset_id}: {e}", "error")
            return False

    def download_roboflow(self, config: dict, dest_dir: Path) -> bool:
        """Download dataset from Roboflow."""
        project = config["project"]
        version = config.get("version", 1)
        self.log(f"Downloading from Roboflow: {project} (v{version})")

        api_key = os.environ.get("ROBOFLOW_API_KEY")
        if not api_key:
            self.log("ROBOFLOW_API_KEY not set. Skipping Roboflow dataset", "warn")
            return False

        try:
            from roboflow import Roboflow

            rf = Roboflow(api_key=api_key)

            # Parse project name (workspace/project format)
            if "/" in project:
                workspace, proj_name = project.split("/", 1)
            else:
                self.log("Invalid project format. Use workspace/project", "error")
                return False

            project_obj = rf.workspace(workspace).project(proj_name)
            dataset = project_obj.version(version).download("yolov8", location=str(dest_dir))

            self.log(f"Downloaded Roboflow dataset to: {dest_dir}")
            return True

        except ImportError:
            self.log("roboflow package not installed. Installing...", "warn")
            os.system(f"{sys.executable} -m pip install roboflow")
            return self.download_roboflow(config, dest_dir)

        except Exception as e:
            self.log(f"Failed to download Roboflow dataset {project}: {e}", "error")
            return False

    def download_dataset(self, dataset_key: str, config: dict) -> bool:
        """Download a single dataset based on its source."""
        dest_dir = self.download_dir / dataset_key

        if dest_dir.exists() and any(dest_dir.iterdir()):
            self.log(f"Dataset {dataset_key} already exists at {dest_dir}", "info")
            self.stats["skipped"] += 1
            return True

        source = config["source"]
        success = False

        if source == "huggingface":
            success = self.download_huggingface(config, dest_dir)
        elif source == "kaggle":
            success = self.download_kaggle(config, dest_dir)
        elif source == "roboflow":
            success = self.download_roboflow(config, dest_dir)
        else:
            self.log(f"Unknown source: {source}", "error")

        if success:
            self.stats["downloaded"] += 1
        else:
            self.stats["failed"] += 1

        return success

    def download_all(self, datasets: dict = None) -> dict:
        """Download all configured datasets."""
        if datasets is None:
            datasets = DATASETS

        self.log(f"Starting download of {len(datasets)} datasets")
        self.log(f"Download directory: {self.download_dir}")
        print("-" * 60)

        results = {}
        for key, config in datasets.items():
            print()
            self.log(f"Processing: {config['name']}")
            self.log(f"Description: {config['description']}", "debug")
            results[key] = self.download_dataset(key, config)

        return results


def find_images(directory: Path, extensions: tuple = (".jpg", ".jpeg", ".png", ".webp")) -> list:
    """Recursively find all image files in a directory."""
    images = []
    for ext in extensions:
        images.extend(directory.rglob(f"*{ext}"))
        images.extend(directory.rglob(f"*{ext.upper()}"))
    return images


def get_dataset_stats(download_dir: Path) -> dict:
    """Get statistics about downloaded datasets."""
    stats = {}

    for dataset_dir in download_dir.iterdir():
        if dataset_dir.is_dir():
            images = find_images(dataset_dir)
            size = sum(f.stat().st_size for f in images if f.exists())
            stats[dataset_dir.name] = {
                "image_count": len(images),
                "size_mb": round(size / (1024 * 1024), 2),
                "path": str(dataset_dir),
            }

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Download jewelry datasets from multiple sources",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Environment Variables:
  HUGGINGFACE_TOKEN   - HuggingFace API token (for private repos)
  KAGGLE_USERNAME     - Kaggle username
  KAGGLE_KEY          - Kaggle API key
  ROBOFLOW_API_KEY    - Roboflow API key

Examples:
  # Download all datasets
  python download_datasets.py

  # Download specific datasets
  python download_datasets.py --datasets jewelry_dwpose tanishq_jewelry

  # Custom output directory
  python download_datasets.py --output-dir ./my_datasets

  # List available datasets
  python download_datasets.py --list
        """
    )

    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=DEFAULT_DOWNLOAD_DIR,
        help=f"Output directory for downloads (default: {DEFAULT_DOWNLOAD_DIR})"
    )
    parser.add_argument(
        "--datasets", "-d",
        nargs="+",
        choices=list(DATASETS.keys()),
        help="Specific datasets to download (default: all)"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available datasets and exit"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show statistics for downloaded datasets"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Minimal output"
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force re-download even if exists"
    )

    args = parser.parse_args()

    # List datasets
    if args.list:
        print("\nAvailable Datasets:")
        print("=" * 60)
        for key, config in DATASETS.items():
            print(f"\n{key}:")
            print(f"  Name: {config['name']}")
            print(f"  Source: {config['source']}")
            print(f"  Description: {config['description']}")
            if "expected_samples" in config:
                print(f"  Expected samples: ~{config['expected_samples']}")
        print()
        return

    # Show stats
    download_dir = Path(args.output_dir)
    if args.stats:
        if not download_dir.exists():
            print(f"Download directory does not exist: {download_dir}")
            return

        stats = get_dataset_stats(download_dir)
        print("\nDataset Statistics:")
        print("=" * 60)
        total_images = 0
        total_size = 0
        for name, info in stats.items():
            print(f"\n{name}:")
            print(f"  Images: {info['image_count']}")
            print(f"  Size: {info['size_mb']} MB")
            total_images += info["image_count"]
            total_size += info["size_mb"]
        print(f"\nTotal: {total_images} images, {total_size:.2f} MB")
        return

    # Force re-download
    if args.force and download_dir.exists():
        print(f"Removing existing downloads: {download_dir}")
        shutil.rmtree(download_dir)

    # Download datasets
    print("=" * 60)
    print("Jewelry Dataset Downloader")
    print("=" * 60)

    downloader = DatasetDownloader(
        download_dir=args.output_dir,
        verbose=not args.quiet
    )

    # Select datasets to download
    if args.datasets:
        datasets_to_download = {k: v for k, v in DATASETS.items() if k in args.datasets}
    else:
        datasets_to_download = DATASETS

    # Download
    results = downloader.download_all(datasets_to_download)

    # Summary
    print()
    print("=" * 60)
    print("Download Summary")
    print("=" * 60)
    print(f"  Downloaded: {downloader.stats['downloaded']}")
    print(f"  Skipped (existing): {downloader.stats['skipped']}")
    print(f"  Failed: {downloader.stats['failed']}")

    # Show final stats
    if download_dir.exists():
        stats = get_dataset_stats(download_dir)
        total_images = sum(s["image_count"] for s in stats.values())
        print(f"\nTotal images available: {total_images}")

    print(f"\nDownload location: {download_dir.absolute()}")
    print("\nNext step: Run prepare_training_data.py to organize images for training")


if __name__ == "__main__":
    main()
