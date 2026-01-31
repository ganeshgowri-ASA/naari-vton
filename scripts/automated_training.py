#!/usr/bin/env python3
"""
Automated Training Pipeline for Jewelry Virtual Try-On

This script automates the entire training workflow:
1. Uploads training dataset to HuggingFace Hub (provides public URL for Replicate)
2. Downloads additional datasets from Kaggle, Roboflow, and HuggingFace
3. Runs Replicate training with the uploaded dataset

Environment Variables:
    HUGGINGFACE_TOKEN: Required. HuggingFace API token with write access.
    REPLICATE_API_TOKEN: Required. Replicate API token for training.
    KAGGLE_USERNAME: Optional. Kaggle username for API access.
    KAGGLE_KEY: Optional. Kaggle API key for API access.
    ROBOFLOW_API_KEY: Optional. Roboflow API key for dataset access.

Usage:
    python scripts/automated_training.py
    python scripts/automated_training.py --skip-download
    python scripts/automated_training.py --upload-only
"""

import os
import sys
import time
import argparse
import subprocess
from pathlib import Path
from typing import Optional, Tuple


# Configuration
DEFAULT_DATASET_ZIP = "dataset/jewelry-training-dataset.zip"
DEFAULT_HF_REPO = "ganeshgowri-asa/jewelry-vton-dataset"
DEFAULT_HF_FILENAME = "jewelry-training-dataset.zip"
DEFAULT_REPLICATE_DESTINATION = "ganeshgowri-asa/naari-jewelry-vton"


def log(message: str, level: str = "info"):
    """Log messages with level indicators."""
    prefix = {
        "info": "[INFO]",
        "warn": "[WARN]",
        "error": "[ERROR]",
        "success": "[SUCCESS]",
        "step": "[STEP]"
    }
    print(f"{prefix.get(level, '[INFO]')} {message}")


def check_dependencies():
    """Check and install required dependencies."""
    log("Checking dependencies...", "step")

    required = ["huggingface_hub", "replicate", "requests"]
    optional = ["kaggle", "roboflow", "datasets"]

    for package in required:
        try:
            __import__(package)
            log(f"  {package}: OK")
        except ImportError:
            log(f"  {package}: Installing...", "warn")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package, "-q"])

    for package in optional:
        try:
            __import__(package)
            log(f"  {package}: OK (optional)")
        except ImportError:
            log(f"  {package}: Not installed (optional)", "warn")


def validate_environment(
    need_hf_token: bool = True,
    need_replicate_token: bool = True,
) -> Tuple[bool, list]:
    """Validate required environment variables."""
    log("Validating environment...", "step")

    errors = []
    warnings = []

    # Required tokens based on operation mode
    hf_token = os.environ.get("HUGGINGFACE_TOKEN")
    replicate_token = os.environ.get("REPLICATE_API_TOKEN")

    if need_hf_token:
        if not hf_token:
            errors.append("HUGGINGFACE_TOKEN not set (required for upload)")
        else:
            log("  HUGGINGFACE_TOKEN: Set")
    else:
        if hf_token:
            log("  HUGGINGFACE_TOKEN: Set (optional)")
        else:
            log("  HUGGINGFACE_TOKEN: Not set (not required for this operation)")

    if need_replicate_token:
        if not replicate_token:
            errors.append("REPLICATE_API_TOKEN not set (required for training)")
        else:
            log("  REPLICATE_API_TOKEN: Set")
    else:
        if replicate_token:
            log("  REPLICATE_API_TOKEN: Set (optional)")
        else:
            log("  REPLICATE_API_TOKEN: Not set (not required for this operation)")

    # Optional
    if not os.environ.get("KAGGLE_USERNAME") or not os.environ.get("KAGGLE_KEY"):
        warnings.append("KAGGLE_USERNAME/KAGGLE_KEY not set (Kaggle downloads will be skipped)")
    else:
        log("  KAGGLE credentials: Set")

    if not os.environ.get("ROBOFLOW_API_KEY"):
        warnings.append("ROBOFLOW_API_KEY not set (Roboflow downloads will be skipped)")
    else:
        log("  ROBOFLOW_API_KEY: Set")

    for warning in warnings:
        log(f"  Warning: {warning}", "warn")

    return len(errors) == 0, errors


def upload_to_huggingface(
    zip_path: str,
    repo_id: str,
    filename: str,
    token: str,
) -> Optional[str]:
    """
    Upload dataset ZIP to HuggingFace Hub and return the raw file URL.

    Args:
        zip_path: Path to the local ZIP file
        repo_id: HuggingFace repository ID (e.g., "username/dataset-name")
        filename: Name for the file in the repository
        token: HuggingFace API token with write access

    Returns:
        Raw file URL that can be used by Replicate, or None on failure
    """
    log(f"Uploading {zip_path} to HuggingFace Hub...", "step")

    try:
        from huggingface_hub import HfApi, create_repo

        api = HfApi(token=token)

        # Create repository if it doesn't exist
        try:
            log(f"  Creating/verifying repository: {repo_id}")
            create_repo(
                repo_id=repo_id,
                repo_type="dataset",
                token=token,
                exist_ok=True,
                private=False  # Must be public for Replicate to access
            )
            log(f"  Repository ready: {repo_id}")
        except Exception as e:
            log(f"  Repository exists or created: {e}", "warn")

        # Upload the file
        zip_path = Path(zip_path)
        if not zip_path.exists():
            log(f"  File not found: {zip_path}", "error")
            return None

        file_size_mb = zip_path.stat().st_size / (1024 * 1024)
        log(f"  Uploading {filename} ({file_size_mb:.2f} MB)...")

        api.upload_file(
            path_or_fileobj=str(zip_path),
            path_in_repo=filename,
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
        )

        # Construct the raw file URL
        # HuggingFace datasets raw URL format:
        # https://huggingface.co/datasets/{repo_id}/resolve/main/{filename}
        raw_url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{filename}"

        log(f"  Upload complete!", "success")
        log(f"  Raw URL: {raw_url}")

        return raw_url

    except Exception as e:
        log(f"Failed to upload to HuggingFace: {e}", "error")
        return None


def download_kaggle_dataset(dataset_id: str, output_dir: Path) -> bool:
    """Download dataset from Kaggle."""
    log(f"Downloading Kaggle dataset: {dataset_id}", "step")

    if not os.environ.get("KAGGLE_USERNAME") or not os.environ.get("KAGGLE_KEY"):
        log("  Kaggle credentials not set, skipping", "warn")
        return False

    try:
        import kaggle

        output_dir.mkdir(parents=True, exist_ok=True)

        kaggle.api.dataset_download_files(
            dataset_id,
            path=str(output_dir),
            unzip=True,
        )

        log(f"  Downloaded to: {output_dir}", "success")
        return True

    except ImportError:
        log("  kaggle package not installed", "warn")
        return False
    except Exception as e:
        log(f"  Failed: {e}", "error")
        return False


def download_roboflow_dataset(
    workspace: str,
    project: str,
    version: int,
    output_dir: Path,
) -> bool:
    """Download dataset from Roboflow."""
    log(f"Downloading Roboflow dataset: {workspace}/{project}", "step")

    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        log("  ROBOFLOW_API_KEY not set, skipping", "warn")
        return False

    try:
        from roboflow import Roboflow

        rf = Roboflow(api_key=api_key)
        project_obj = rf.workspace(workspace).project(project)
        dataset = project_obj.version(version).download(
            "yolov8",
            location=str(output_dir)
        )

        log(f"  Downloaded to: {output_dir}", "success")
        return True

    except ImportError:
        log("  roboflow package not installed", "warn")
        return False
    except Exception as e:
        log(f"  Failed: {e}", "error")
        return False


def download_huggingface_dataset(repo_id: str, output_dir: Path) -> bool:
    """Download dataset from HuggingFace using datasets library."""
    log(f"Downloading HuggingFace dataset: {repo_id}", "step")

    try:
        from datasets import load_dataset

        output_dir.mkdir(parents=True, exist_ok=True)

        # Load dataset - this will cache it
        dataset = load_dataset(repo_id, trust_remote_code=True)

        # Save to disk
        dataset.save_to_disk(str(output_dir))

        log(f"  Downloaded to: {output_dir}", "success")
        return True

    except ImportError:
        log("  datasets package not installed", "warn")
        # Try alternative: huggingface_hub snapshot
        try:
            from huggingface_hub import snapshot_download

            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                local_dir=str(output_dir),
                token=os.environ.get("HUGGINGFACE_TOKEN"),
            )
            log(f"  Downloaded via snapshot to: {output_dir}", "success")
            return True
        except Exception as e:
            log(f"  Snapshot download also failed: {e}", "error")
            return False
    except Exception as e:
        log(f"  Failed: {e}", "error")
        return False


def download_additional_datasets(base_dir: Path) -> dict:
    """Download additional datasets from various sources."""
    log("Downloading additional datasets...", "step")

    results = {
        "kaggle": False,
        "roboflow": False,
        "huggingface": False,
    }

    downloads_dir = base_dir / "dataset" / "additional_downloads"
    downloads_dir.mkdir(parents=True, exist_ok=True)

    # 1. Kaggle: Tanishq Jewelry Dataset
    kaggle_dir = downloads_dir / "kaggle_tanishq"
    if not kaggle_dir.exists() or not any(kaggle_dir.iterdir() if kaggle_dir.exists() else []):
        results["kaggle"] = download_kaggle_dataset(
            "sapnilpatel/tanishq-jewellery-dataset",
            kaggle_dir
        )
    else:
        log("  Kaggle dataset already downloaded", "info")
        results["kaggle"] = True

    # 2. Roboflow: Jewelry Segmentation
    roboflow_dir = downloads_dir / "roboflow_segmentation"
    if not roboflow_dir.exists() or not any(roboflow_dir.iterdir() if roboflow_dir.exists() else []):
        results["roboflow"] = download_roboflow_dataset(
            workspace="piezee",
            project="segmentation-for-jewelry-images-svv6j",
            version=1,
            output_dir=roboflow_dir
        )
    else:
        log("  Roboflow dataset already downloaded", "info")
        results["roboflow"] = True

    # 3. HuggingFace: Jewelry DWPose Dataset
    hf_dir = downloads_dir / "hf_jewelry_dwpose"
    if not hf_dir.exists() or not any(hf_dir.iterdir() if hf_dir.exists() else []):
        results["huggingface"] = download_huggingface_dataset(
            "raresense/jewelry-dwpose-dataset",
            hf_dir
        )
    else:
        log("  HuggingFace dataset already downloaded", "info")
        results["huggingface"] = True

    return results


def run_replicate_training(
    dataset_url: str,
    destination: str,
    api_token: str,
    preset: str = "standard",
    wait: bool = False,
) -> Optional[str]:
    """
    Run training on Replicate with the uploaded dataset.

    Args:
        dataset_url: Public URL to the training dataset ZIP
        destination: Replicate model destination (e.g., "username/model-name")
        api_token: Replicate API token
        preset: Training preset (quick, standard, high_quality, production)
        wait: Whether to wait for training to complete

    Returns:
        Training ID or None on failure
    """
    log("Starting Replicate training...", "step")

    try:
        import replicate

        # Training configuration
        configs = {
            "quick": {
                "max_train_steps": 500,
                "learning_rate": 2e-4,
                "lora_rank": 16,
            },
            "standard": {
                "max_train_steps": 2000,
                "learning_rate": 1e-4,
                "lora_rank": 32,
            },
            "high_quality": {
                "max_train_steps": 4000,
                "learning_rate": 5e-5,
                "lora_rank": 64,
            },
            "production": {
                "max_train_steps": 6000,
                "learning_rate": 5e-5,
                "lora_rank": 64,
            },
        }

        config = configs.get(preset, configs["standard"])

        log(f"  Preset: {preset}")
        log(f"  Dataset URL: {dataset_url}")
        log(f"  Destination: {destination}")
        log(f"  Max steps: {config['max_train_steps']}")
        log(f"  Learning rate: {config['learning_rate']}")
        log(f"  LoRA rank: {config['lora_rank']}")

        # Set API token
        os.environ["REPLICATE_API_TOKEN"] = api_token

        # Create training
        training = replicate.trainings.create(
            version="stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b",
            input={
                "input_images": dataset_url,
                "token_string": "JEWELRYVTON",
                "caption_prefix": "a photo of JEWELRYVTON",
                "max_train_steps": config["max_train_steps"],
                "learning_rate": config["learning_rate"],
                "use_lora": True,
                "lora_rank": config["lora_rank"],
                "resolution": 1024,
                "train_text_encoder": True,
                "use_8bit_adam": True,
                "gradient_checkpointing": True,
            },
            destination=destination,
        )

        log(f"  Training created!", "success")
        log(f"  Training ID: {training.id}")
        log(f"  Status: {training.status}")
        log(f"  View at: https://replicate.com/p/{training.id}")

        if wait:
            log("  Waiting for training to complete...")
            while training.status not in ["succeeded", "failed", "canceled"]:
                time.sleep(30)
                training.reload()
                log(f"  Status: {training.status}")

            if training.status == "succeeded":
                log(f"  Training completed successfully!", "success")
                if training.output:
                    log(f"  Output model: {training.output}")
            else:
                log(f"  Training {training.status}", "error")

        return training.id

    except Exception as e:
        log(f"Failed to start Replicate training: {e}", "error")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Automated training pipeline for Jewelry VTON",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Environment Variables:
  HUGGINGFACE_TOKEN    - HuggingFace API token (write access required)
  REPLICATE_API_TOKEN  - Replicate API token
  KAGGLE_USERNAME      - Kaggle username (optional)
  KAGGLE_KEY           - Kaggle API key (optional)
  ROBOFLOW_API_KEY     - Roboflow API key (optional)

Examples:
  # Full automated training
  python scripts/automated_training.py

  # Upload only (don't start training)
  python scripts/automated_training.py --upload-only

  # Skip additional downloads
  python scripts/automated_training.py --skip-download

  # Use different training preset
  python scripts/automated_training.py --preset high_quality
        """
    )

    parser.add_argument(
        "--dataset-zip", "-d",
        type=str,
        default=DEFAULT_DATASET_ZIP,
        help=f"Path to dataset ZIP file (default: {DEFAULT_DATASET_ZIP})"
    )
    parser.add_argument(
        "--hf-repo", "-r",
        type=str,
        default=DEFAULT_HF_REPO,
        help=f"HuggingFace repository ID (default: {DEFAULT_HF_REPO})"
    )
    parser.add_argument(
        "--hf-filename", "-f",
        type=str,
        default=DEFAULT_HF_FILENAME,
        help=f"Filename in HuggingFace repo (default: {DEFAULT_HF_FILENAME})"
    )
    parser.add_argument(
        "--replicate-dest",
        type=str,
        default=DEFAULT_REPLICATE_DESTINATION,
        help=f"Replicate model destination (default: {DEFAULT_REPLICATE_DESTINATION})"
    )
    parser.add_argument(
        "--preset", "-p",
        choices=["quick", "standard", "high_quality", "production"],
        default="standard",
        help="Training preset (default: standard)"
    )
    parser.add_argument(
        "--upload-only",
        action="store_true",
        help="Only upload to HuggingFace, don't start training"
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip downloading additional datasets"
    )
    parser.add_argument(
        "--skip-upload",
        action="store_true",
        help="Skip upload, use existing HuggingFace URL"
    )
    parser.add_argument(
        "--wait", "-w",
        action="store_true",
        help="Wait for training to complete"
    )
    parser.add_argument(
        "--dataset-url",
        type=str,
        help="Use existing dataset URL instead of uploading"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration without uploading or training"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Automated Training Pipeline for Jewelry Virtual Try-On")
    print("=" * 70)
    print()

    # Get project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    # Check dependencies
    check_dependencies()
    print()

    # Determine which tokens are needed based on operation mode
    need_hf_token = not args.skip_upload and not args.dataset_url and not args.dry_run
    need_replicate_token = not args.upload_only and not args.dry_run

    # Validate environment
    valid, errors = validate_environment(
        need_hf_token=need_hf_token,
        need_replicate_token=need_replicate_token,
    )
    if not valid:
        print()
        log("Environment validation failed:", "error")
        for error in errors:
            log(f"  - {error}", "error")
        print()
        log("Please set the required environment variables and try again.")
        log("Example:")
        log("  export HUGGINGFACE_TOKEN='your-hf-token'")
        log("  export REPLICATE_API_TOKEN='your-replicate-token'")
        sys.exit(1)
    print()

    # Get tokens
    hf_token = os.environ.get("HUGGINGFACE_TOKEN")
    replicate_token = os.environ.get("REPLICATE_API_TOKEN")

    # Dry run mode
    if args.dry_run:
        log("Dry run mode - validating configuration only")
        print()

        zip_path = project_root / args.dataset_zip
        if zip_path.exists():
            size_mb = zip_path.stat().st_size / (1024 * 1024)
            log(f"Dataset ZIP found: {zip_path} ({size_mb:.2f} MB)", "success")
        else:
            log(f"Dataset ZIP not found: {zip_path}", "error")

        log(f"HuggingFace repo: {args.hf_repo}")
        log(f"Replicate destination: {args.replicate_dest}")
        log(f"Training preset: {args.preset}")

        print()
        print("=" * 70)
        log("Dry run complete. To run for real, remove --dry-run flag", "success")
        print("=" * 70)
        return

    # Step 1: Download additional datasets (optional)
    if not args.skip_download and not args.upload_only:
        print("-" * 70)
        download_results = download_additional_datasets(project_root)
        print()
        log("Download results:")
        for source, success in download_results.items():
            status = "success" if success else "skipped/failed"
            log(f"  {source}: {status}")
        print()

    # Step 2: Upload to HuggingFace
    dataset_url = args.dataset_url

    if not dataset_url and not args.skip_upload:
        print("-" * 70)
        zip_path = project_root / args.dataset_zip

        if not zip_path.exists():
            log(f"Dataset ZIP not found: {zip_path}", "error")
            log("Please run prepare_training_data.py first to create the ZIP file.")
            sys.exit(1)

        dataset_url = upload_to_huggingface(
            zip_path=str(zip_path),
            repo_id=args.hf_repo,
            filename=args.hf_filename,
            token=hf_token,
        )

        if not dataset_url:
            log("Failed to upload dataset to HuggingFace", "error")
            sys.exit(1)
        print()
    elif args.skip_upload:
        # Construct URL from repo info
        dataset_url = f"https://huggingface.co/datasets/{args.hf_repo}/resolve/main/{args.hf_filename}"
        log(f"Using existing HuggingFace URL: {dataset_url}")
        print()

    if args.upload_only:
        print("=" * 70)
        log("Upload complete! Dataset URL:", "success")
        log(f"  {dataset_url}")
        print()
        log("To start training manually, run:")
        log(f"  export DATASET_URL='{dataset_url}'")
        log(f"  python scripts/train_replicate.py --preset {args.preset}")
        print("=" * 70)
        return

    # Step 3: Run Replicate training
    print("-" * 70)
    training_id = run_replicate_training(
        dataset_url=dataset_url,
        destination=args.replicate_dest,
        api_token=replicate_token,
        preset=args.preset,
        wait=args.wait,
    )

    if not training_id:
        log("Failed to start Replicate training", "error")
        sys.exit(1)

    # Summary
    print()
    print("=" * 70)
    log("Automated training pipeline completed!", "success")
    print("=" * 70)
    print()
    log("Summary:")
    log(f"  Dataset URL: {dataset_url}")
    log(f"  Training ID: {training_id}")
    log(f"  Training preset: {args.preset}")
    log(f"  Monitor at: https://replicate.com/p/{training_id}")
    print()

    if not args.wait:
        log("Training is running in the background.")
        log("To monitor progress, visit the URL above or run:")
        log(f"  python -c \"import replicate; t = replicate.trainings.get('{training_id}'); print(t.status)\"")


if __name__ == "__main__":
    main()
