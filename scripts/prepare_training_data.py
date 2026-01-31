#!/usr/bin/env python3
"""
Data Preparation Script for Jewelry Virtual Try-On Training

This script:
1. Organizes downloaded jewelry images into Replicate SDXL training format
2. Generates proper captions with JEWELRYVTON token
3. Resizes images to optimal training resolution
4. Creates a ZIP file ready for upload to Replicate

The output format follows Replicate's SDXL fine-tuning requirements:
- Images named with their captions (e.g., "a photo of JEWELRYVTON necklace.jpg")
- Or images with corresponding .txt caption files
- Optimal resolution: 1024x1024 for SDXL

Environment Variables:
    CAPTION_PREFIX: Optional. Custom caption prefix (default: "a photo of JEWELRYVTON")
"""

import os
import sys
import json
import shutil
import zipfile
import hashlib
import argparse
import re
from pathlib import Path
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Tuple, List
import random

# Configuration
DEFAULT_INPUT_DIR = "dataset/raw_downloads"
DEFAULT_OUTPUT_DIR = "dataset/training_prepared"
DEFAULT_ZIP_NAME = "jewelry-training-dataset.zip"
DEFAULT_TOKEN = "JEWELRYVTON"
DEFAULT_RESOLUTION = 1024
MIN_IMAGE_SIZE = 256  # Minimum dimension to include image

# Jewelry type detection patterns
JEWELRY_PATTERNS = {
    "necklace": [
        r"necklace", r"pendant", r"choker", r"chain", r"collar",
        r"mangalsutra", r"haar", r"mala", r"locket"
    ],
    "earring": [
        r"earring", r"ear\s*ring", r"jhumka", r"jhumki", r"stud",
        r"hoop", r"chandbali", r"danglers?", r"drops?"
    ],
    "ring": [
        r"\bring\b", r"finger\s*ring", r"band", r"engagement",
        r"wedding\s*ring", r"cocktail\s*ring", r"signet"
    ],
    "bracelet": [
        r"bracelet", r"bangle", r"kangan", r"kada", r"cuff",
        r"charm\s*bracelet", r"tennis\s*bracelet", r"wrist"
    ],
    "maang_tikka": [
        r"maang\s*tikka", r"tikka", r"matha\s*patti", r"forehead",
        r"bridal\s*headpiece", r"hair\s*accessory"
    ],
    "nose_ring": [
        r"nose\s*ring", r"nath", r"nose\s*pin", r"nose\s*stud",
        r"septum", r"nostril"
    ],
    "anklet": [
        r"anklet", r"payal", r"ankle\s*bracelet", r"toe\s*ring"
    ],
    "brooch": [
        r"brooch", r"pin", r"badge", r"clip"
    ],
}

# Material patterns for enhanced captions
MATERIAL_PATTERNS = {
    "gold": [r"gold", r"golden", r"22k", r"24k", r"18k", r"sone"],
    "silver": [r"silver", r"sterling", r"chandi"],
    "diamond": [r"diamond", r"heera", r"brilliant"],
    "pearl": [r"pearl", r"moti"],
    "kundan": [r"kundan", r"polki", r"jadau"],
    "ruby": [r"ruby", r"manik"],
    "emerald": [r"emerald", r"panna"],
    "platinum": [r"platinum"],
}


class ImageProcessor:
    """Handles image processing for training data preparation."""

    def __init__(
        self,
        output_dir: Path,
        token: str = DEFAULT_TOKEN,
        target_resolution: int = DEFAULT_RESOLUTION,
        use_caption_files: bool = True,
    ):
        self.output_dir = Path(output_dir)
        self.token = token
        self.target_resolution = target_resolution
        self.use_caption_files = use_caption_files
        self.processed_hashes = set()  # Track duplicates
        self.stats = {
            "processed": 0,
            "skipped_small": 0,
            "skipped_duplicate": 0,
            "skipped_error": 0,
            "by_type": {},
        }

    def detect_jewelry_type(self, filename: str, parent_dirs: List[str] = None) -> str:
        """Detect jewelry type from filename or directory structure."""
        # Combine filename and parent directories for matching
        search_text = filename.lower()
        if parent_dirs:
            search_text = " ".join(parent_dirs).lower() + " " + search_text

        for jewelry_type, patterns in JEWELRY_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, search_text, re.IGNORECASE):
                    return jewelry_type

        return "jewelry"  # Default fallback

    def detect_materials(self, filename: str, parent_dirs: List[str] = None) -> List[str]:
        """Detect materials from filename or directory structure."""
        search_text = filename.lower()
        if parent_dirs:
            search_text = " ".join(parent_dirs).lower() + " " + search_text

        materials = []
        for material, patterns in MATERIAL_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, search_text, re.IGNORECASE):
                    materials.append(material)
                    break

        return materials

    def generate_caption(
        self,
        jewelry_type: str,
        materials: List[str] = None,
        style: str = None,
    ) -> str:
        """Generate training caption with JEWELRYVTON token."""
        # Base caption structure
        parts = [f"a photo of {self.token}"]

        # Add materials
        if materials:
            parts.append(" ".join(materials[:2]))  # Max 2 materials

        # Add jewelry type (replace underscores)
        type_display = jewelry_type.replace("_", " ")
        parts.append(type_display)

        # Add style if available
        if style:
            parts.append(f"in {style} style")

        caption = " ".join(parts)

        # Clean up caption
        caption = re.sub(r"\s+", " ", caption).strip()

        return caption

    def compute_hash(self, image_path: Path) -> str:
        """Compute hash of image for duplicate detection."""
        hasher = hashlib.md5()
        with open(image_path, "rb") as f:
            # Read in chunks for large files
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def resize_image(
        self,
        image: Image.Image,
        target_size: int,
        method: str = "contain",
    ) -> Image.Image:
        """Resize image to target size while preserving aspect ratio."""
        width, height = image.size

        if method == "contain":
            # Resize so largest dimension is target_size
            ratio = target_size / max(width, height)
            new_size = (int(width * ratio), int(height * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)

            # Create square canvas and paste
            canvas = Image.new("RGB", (target_size, target_size), (255, 255, 255))
            paste_x = (target_size - new_size[0]) // 2
            paste_y = (target_size - new_size[1]) // 2
            canvas.paste(image, (paste_x, paste_y))
            return canvas

        elif method == "cover":
            # Resize so smallest dimension is target_size, then crop
            ratio = target_size / min(width, height)
            new_size = (int(width * ratio), int(height * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)

            # Center crop
            left = (new_size[0] - target_size) // 2
            top = (new_size[1] - target_size) // 2
            return image.crop((left, top, left + target_size, top + target_size))

        else:
            # Simple resize (may distort)
            return image.resize((target_size, target_size), Image.Resampling.LANCZOS)

    def process_image(
        self,
        image_path: Path,
        parent_dirs: List[str] = None,
    ) -> Optional[Tuple[str, str]]:
        """Process a single image and return (output_path, caption) or None."""
        try:
            # Check for duplicates
            img_hash = self.compute_hash(image_path)
            if img_hash in self.processed_hashes:
                self.stats["skipped_duplicate"] += 1
                return None
            self.processed_hashes.add(img_hash)

            # Open and validate image
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode in ("RGBA", "P", "LA", "L"):
                    img = img.convert("RGB")

                # Check minimum size
                if min(img.size) < MIN_IMAGE_SIZE:
                    self.stats["skipped_small"] += 1
                    return None

                # Detect jewelry type and materials
                jewelry_type = self.detect_jewelry_type(
                    image_path.stem, parent_dirs
                )
                materials = self.detect_materials(image_path.stem, parent_dirs)

                # Generate caption
                caption = self.generate_caption(jewelry_type, materials)

                # Resize image
                processed_img = self.resize_image(img, self.target_resolution)

                # Generate unique filename
                unique_id = img_hash[:8]
                safe_type = jewelry_type.replace(" ", "_")

                if self.use_caption_files:
                    # Save image with simple name, caption in .txt file
                    output_name = f"{safe_type}_{unique_id}.jpg"
                    caption_name = f"{safe_type}_{unique_id}.txt"
                else:
                    # Embed caption in filename (Replicate style)
                    safe_caption = re.sub(r"[^\w\s-]", "", caption)
                    safe_caption = re.sub(r"\s+", "_", safe_caption)
                    output_name = f"{safe_caption}_{unique_id}.jpg"
                    caption_name = None

                output_path = self.output_dir / output_name
                processed_img.save(output_path, "JPEG", quality=95)

                # Save caption file if using that method
                if caption_name:
                    caption_path = self.output_dir / caption_name
                    with open(caption_path, "w") as f:
                        f.write(caption)

                # Update stats
                self.stats["processed"] += 1
                self.stats["by_type"][jewelry_type] = (
                    self.stats["by_type"].get(jewelry_type, 0) + 1
                )

                return (str(output_path), caption)

        except Exception as e:
            self.stats["skipped_error"] += 1
            print(f"  Error processing {image_path}: {e}")
            return None


def find_images(
    directory: Path,
    extensions: tuple = (".jpg", ".jpeg", ".png", ".webp", ".bmp"),
) -> List[Path]:
    """Recursively find all image files in a directory."""
    images = []
    for ext in extensions:
        images.extend(directory.rglob(f"*{ext}"))
        images.extend(directory.rglob(f"*{ext.upper()}"))
    return images


def create_zip(
    source_dir: Path,
    output_path: Path,
    include_patterns: tuple = (".jpg", ".jpeg", ".png", ".txt"),
) -> int:
    """Create a ZIP file from the processed images."""
    file_count = 0

    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in source_dir.iterdir():
            if file_path.suffix.lower() in include_patterns:
                zf.write(file_path, file_path.name)
                file_count += 1

    return file_count


def prepare_training_data(
    input_dir: Path,
    output_dir: Path,
    token: str = DEFAULT_TOKEN,
    resolution: int = DEFAULT_RESOLUTION,
    max_samples: int = None,
    use_caption_files: bool = True,
    create_validation_split: bool = True,
    validation_ratio: float = 0.1,
) -> dict:
    """Main function to prepare training data from downloaded datasets."""
    print(f"Preparing training data from: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Token: {token}")
    print(f"Target resolution: {resolution}x{resolution}")
    print("-" * 60)

    # Create output directory
    output_dir = Path(output_dir)
    if output_dir.exists():
        print(f"Cleaning existing output directory...")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    # Find all images
    input_dir = Path(input_dir)
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        return {"error": "Input directory not found"}

    all_images = find_images(input_dir)
    print(f"Found {len(all_images)} images in source directories")

    if not all_images:
        print("No images found!")
        return {"error": "No images found"}

    # Shuffle and limit if specified
    random.shuffle(all_images)
    if max_samples and len(all_images) > max_samples:
        all_images = all_images[:max_samples]
        print(f"Limited to {max_samples} samples")

    # Create processor
    processor = ImageProcessor(
        output_dir=output_dir,
        token=token,
        target_resolution=resolution,
        use_caption_files=use_caption_files,
    )

    # Process images with progress
    print(f"\nProcessing {len(all_images)} images...")
    processed_items = []

    for i, img_path in enumerate(all_images):
        # Get parent directories for context
        try:
            rel_path = img_path.relative_to(input_dir)
            parent_dirs = [p.name for p in rel_path.parents if p.name]
        except ValueError:
            parent_dirs = []

        result = processor.process_image(img_path, parent_dirs)
        if result:
            processed_items.append(result)

        # Progress update
        if (i + 1) % 100 == 0 or i == len(all_images) - 1:
            print(f"  Processed: {i + 1}/{len(all_images)} "
                  f"(kept: {processor.stats['processed']})")

    # Print stats
    print()
    print("=" * 60)
    print("Processing Summary")
    print("=" * 60)
    print(f"Total processed: {processor.stats['processed']}")
    print(f"Skipped (small): {processor.stats['skipped_small']}")
    print(f"Skipped (duplicate): {processor.stats['skipped_duplicate']}")
    print(f"Skipped (error): {processor.stats['skipped_error']}")

    print("\nBy jewelry type:")
    for jtype, count in sorted(processor.stats["by_type"].items()):
        print(f"  {jtype}: {count}")

    # Create manifest
    manifest = {
        "token": token,
        "resolution": resolution,
        "total_samples": processor.stats["processed"],
        "by_type": processor.stats["by_type"],
        "samples": [{"path": p, "caption": c} for p, c in processed_items],
    }

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return processor.stats


def main():
    parser = argparse.ArgumentParser(
        description="Prepare jewelry images for Replicate SDXL fine-tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Prepare all downloaded datasets
  python prepare_training_data.py

  # Custom resolution and output
  python prepare_training_data.py --resolution 768 --output ./my_training_data

  # Limit samples for testing
  python prepare_training_data.py --max-samples 500

  # Create ZIP file for upload
  python prepare_training_data.py --create-zip

  # Use filename-based captions (Replicate default)
  python prepare_training_data.py --caption-in-filename
        """
    )

    parser.add_argument(
        "--input-dir", "-i",
        type=str,
        default=DEFAULT_INPUT_DIR,
        help=f"Input directory with downloaded datasets (default: {DEFAULT_INPUT_DIR})"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for prepared data (default: {DEFAULT_OUTPUT_DIR})"
    )
    parser.add_argument(
        "--token", "-t",
        type=str,
        default=DEFAULT_TOKEN,
        help=f"Token string for captions (default: {DEFAULT_TOKEN})"
    )
    parser.add_argument(
        "--resolution", "-r",
        type=int,
        default=DEFAULT_RESOLUTION,
        help=f"Target image resolution (default: {DEFAULT_RESOLUTION})"
    )
    parser.add_argument(
        "--max-samples", "-m",
        type=int,
        default=None,
        help="Maximum number of samples to process (default: all)"
    )
    parser.add_argument(
        "--caption-in-filename",
        action="store_true",
        help="Embed captions in filenames instead of .txt files"
    )
    parser.add_argument(
        "--create-zip", "-z",
        action="store_true",
        help="Create ZIP file after processing"
    )
    parser.add_argument(
        "--zip-name",
        type=str,
        default=DEFAULT_ZIP_NAME,
        help=f"Name for the ZIP file (default: {DEFAULT_ZIP_NAME})"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)

    # Prepare data
    print("=" * 60)
    print("Jewelry Training Data Preparation")
    print("=" * 60)
    print()

    stats = prepare_training_data(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        token=args.token,
        resolution=args.resolution,
        max_samples=args.max_samples,
        use_caption_files=not args.caption_in_filename,
    )

    if "error" in stats:
        print(f"\nError: {stats['error']}")
        sys.exit(1)

    # Create ZIP if requested
    if args.create_zip:
        print()
        print("-" * 60)
        print("Creating ZIP file...")

        output_dir = Path(args.output_dir)
        zip_path = output_dir.parent / args.zip_name

        file_count = create_zip(output_dir, zip_path)

        zip_size = zip_path.stat().st_size / (1024 * 1024)
        print(f"ZIP created: {zip_path}")
        print(f"  Files: {file_count}")
        print(f"  Size: {zip_size:.2f} MB")

        print(f"\nTo train on Replicate:")
        print(f"  1. Upload {zip_path} to a public URL (GitHub releases, S3, etc.)")
        print(f"  2. Set DATASET_URL environment variable to the URL")
        print(f"  3. Run: python scripts/train_replicate.py")

    print("\nDone!")


if __name__ == "__main__":
    main()
