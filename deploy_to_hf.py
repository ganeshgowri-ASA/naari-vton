#!/usr/bin/env python3
"""
Deploy Naari Studio Jewelry Try-On to HuggingFace Spaces

Usage:
    python deploy_to_hf.py --token YOUR_HF_TOKEN

Or set the HF_TOKEN environment variable:
    export HF_TOKEN=your_token_here
    python deploy_to_hf.py
"""

import os
import argparse
from huggingface_hub import HfApi, upload_file, upload_folder


SPACE_ID = "GaneshGowri/naari-avatar"
FILES_TO_UPLOAD = [
    "app.py",
    "jewelry_engine.py",
    "requirements.txt",
]
FOLDERS_TO_UPLOAD = [
    "assets",
]


def deploy(token: str):
    """Deploy files to HuggingFace Space."""
    api = HfApi(token=token)

    print(f"Deploying to: https://huggingface.co/spaces/{SPACE_ID}")
    print("-" * 50)

    # Upload individual files
    for filepath in FILES_TO_UPLOAD:
        if os.path.exists(filepath):
            print(f"Uploading: {filepath}")
            upload_file(
                path_or_fileobj=filepath,
                path_in_repo=filepath,
                repo_id=SPACE_ID,
                repo_type="space",
                token=token
            )
            print(f"  ✓ {filepath} uploaded")
        else:
            print(f"  ✗ {filepath} not found, skipping")

    # Upload folders
    for folder in FOLDERS_TO_UPLOAD:
        if os.path.exists(folder):
            print(f"Uploading folder: {folder}/")
            upload_folder(
                folder_path=folder,
                path_in_repo=folder,
                repo_id=SPACE_ID,
                repo_type="space",
                token=token
            )
            print(f"  ✓ {folder}/ uploaded")
        else:
            print(f"  ✗ {folder}/ not found, skipping")

    print("-" * 50)
    print(f"Deployment complete!")
    print(f"View your Space: https://huggingface.co/spaces/{SPACE_ID}")


def main():
    parser = argparse.ArgumentParser(description="Deploy to HuggingFace Spaces")
    parser.add_argument("--token", "-t", help="HuggingFace API token")
    args = parser.parse_args()

    token = args.token or os.environ.get("HF_TOKEN")

    if not token:
        print("Error: No HuggingFace token provided.")
        print()
        print("Get your token from: https://huggingface.co/settings/tokens")
        print()
        print("Then either:")
        print("  1. Run: python deploy_to_hf.py --token YOUR_TOKEN")
        print("  2. Or set: export HF_TOKEN=YOUR_TOKEN")
        return 1

    deploy(token)
    return 0


if __name__ == "__main__":
    exit(main())
