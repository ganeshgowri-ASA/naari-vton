"""
Jewelry Processor Module

This module handles jewelry image preprocessing including:
- Background removal using rembg
- Feature extraction (colors, material type)
- Natural language description generation
"""

import io
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple, List, Dict, Any
from collections import Counter

import numpy as np
from PIL import Image

try:
    from rembg import remove
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False
    logging.warning("rembg not available. Background removal will be disabled.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class JewelryType(Enum):
    """Enum for different jewelry types."""
    NECKLACE = "necklace"
    EARRING = "earring"
    BANGLE = "bangle"
    RING = "ring"
    BRACELET = "bracelet"
    PENDANT = "pendant"
    CHAIN = "chain"
    UNKNOWN = "unknown"


class MaterialType(Enum):
    """Enum for jewelry material types based on color analysis."""
    GOLD = "gold"
    SILVER = "silver"
    ROSE_GOLD = "rose_gold"
    PLATINUM = "platinum"
    COPPER = "copper"
    PEARL = "pearl"
    DIAMOND = "diamond"
    GEMSTONE = "gemstone"
    MIXED = "mixed"
    UNKNOWN = "unknown"


@dataclass
class JewelryFeatures:
    """Data class to hold extracted jewelry features."""
    dominant_colors: List[Tuple[int, int, int]]
    material_type: MaterialType
    brightness: float
    contrast: float
    has_transparency: bool
    estimated_size: Tuple[int, int]
    color_palette: List[str]
    sparkle_factor: float


@dataclass
class ProcessedJewelry:
    """Data class for processed jewelry output."""
    original_image: Image.Image
    processed_image: Image.Image
    mask: Optional[Image.Image]
    features: JewelryFeatures
    jewelry_type: JewelryType
    description: str


class JewelryProcessor:
    """
    Advanced jewelry image processor with background removal,
    feature extraction, and description generation capabilities.
    """

    # Color ranges for material detection (in HSV-like approximation)
    MATERIAL_COLOR_RANGES = {
        MaterialType.GOLD: {
            'hue_range': (20, 50),
            'saturation_min': 0.3,
            'value_min': 0.5
        },
        MaterialType.SILVER: {
            'hue_range': (0, 360),
            'saturation_max': 0.15,
            'value_min': 0.6
        },
        MaterialType.ROSE_GOLD: {
            'hue_range': (0, 30),
            'saturation_min': 0.2,
            'value_min': 0.5
        },
        MaterialType.COPPER: {
            'hue_range': (10, 35),
            'saturation_min': 0.4,
            'value_min': 0.3
        },
        MaterialType.PEARL: {
            'hue_range': (0, 360),
            'saturation_max': 0.1,
            'value_min': 0.85
        }
    }

    # Named colors for description
    COLOR_NAMES = {
        (255, 215, 0): "golden",
        (192, 192, 192): "silver",
        (183, 110, 121): "rose gold",
        (229, 228, 226): "platinum",
        (184, 115, 51): "copper",
        (255, 255, 255): "white",
        (0, 0, 0): "black",
        (255, 0, 0): "red",
        (0, 255, 0): "green",
        (0, 0, 255): "blue",
        (255, 192, 203): "pink",
        (128, 0, 128): "purple",
        (0, 255, 255): "turquoise"
    }

    def __init__(self, enable_gpu: bool = False):
        """
        Initialize the jewelry processor.

        Args:
            enable_gpu: Whether to enable GPU acceleration for rembg
        """
        self.enable_gpu = enable_gpu
        self._validate_dependencies()

    def _validate_dependencies(self) -> None:
        """Validate that required dependencies are available."""
        if not REMBG_AVAILABLE:
            logger.warning(
                "rembg is not installed. Background removal will use fallback method. "
                "Install with: pip install rembg"
            )

    def remove_background(
        self,
        image: Image.Image,
        alpha_matting: bool = True,
        alpha_matting_foreground_threshold: int = 240,
        alpha_matting_background_threshold: int = 10
    ) -> Tuple[Image.Image, Image.Image]:
        """
        Remove background from jewelry image using rembg.

        Args:
            image: Input PIL Image
            alpha_matting: Whether to use alpha matting for better edges
            alpha_matting_foreground_threshold: Foreground threshold for matting
            alpha_matting_background_threshold: Background threshold for matting

        Returns:
            Tuple of (processed_image, mask)
        """
        if not REMBG_AVAILABLE:
            return self._fallback_background_removal(image)

        try:
            # Convert to bytes for rembg
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)

            # Remove background with rembg
            output_bytes = remove(
                img_byte_arr.getvalue(),
                alpha_matting=alpha_matting,
                alpha_matting_foreground_threshold=alpha_matting_foreground_threshold,
                alpha_matting_background_threshold=alpha_matting_background_threshold
            )

            # Convert back to PIL Image
            processed_image = Image.open(io.BytesIO(output_bytes)).convert('RGBA')

            # Extract mask from alpha channel
            mask = processed_image.split()[3]

            logger.info("Background removed successfully using rembg")
            return processed_image, mask

        except Exception as e:
            logger.error(f"Error in background removal: {e}")
            return self._fallback_background_removal(image)

    def _fallback_background_removal(
        self,
        image: Image.Image
    ) -> Tuple[Image.Image, Image.Image]:
        """
        Fallback background removal using simple thresholding.

        Args:
            image: Input PIL Image

        Returns:
            Tuple of (processed_image, mask)
        """
        logger.info("Using fallback background removal method")

        # Convert to RGBA
        if image.mode != 'RGBA':
            image = image.convert('RGBA')

        # Convert to numpy array
        img_array = np.array(image)

        # Simple background detection based on edge colors
        # Assume background is the dominant color at edges
        edges = np.concatenate([
            img_array[0, :, :3],      # Top edge
            img_array[-1, :, :3],     # Bottom edge
            img_array[:, 0, :3],      # Left edge
            img_array[:, -1, :3]      # Right edge
        ])

        # Calculate median background color
        bg_color = np.median(edges, axis=0)

        # Create mask based on color distance from background
        rgb = img_array[:, :, :3].astype(np.float32)
        distance = np.sqrt(np.sum((rgb - bg_color) ** 2, axis=2))

        # Threshold to create mask
        threshold = 30
        mask_array = (distance > threshold).astype(np.uint8) * 255

        # Apply some morphological operations to clean up
        from scipy import ndimage
        mask_array = ndimage.binary_dilation(mask_array, iterations=2)
        mask_array = ndimage.binary_erosion(mask_array, iterations=1)
        mask_array = (mask_array * 255).astype(np.uint8)

        # Create mask image
        mask = Image.fromarray(mask_array, mode='L')

        # Apply mask to image
        img_array[:, :, 3] = mask_array
        processed_image = Image.fromarray(img_array, mode='RGBA')

        return processed_image, mask

    def extract_features(self, image: Image.Image) -> JewelryFeatures:
        """
        Extract visual features from jewelry image.

        Args:
            image: Input PIL Image (preferably with background removed)

        Returns:
            JewelryFeatures object containing extracted features
        """
        # Convert to RGBA if needed
        if image.mode != 'RGBA':
            image = image.convert('RGBA')

        img_array = np.array(image)
        rgb = img_array[:, :, :3]
        alpha = img_array[:, :, 3] if img_array.shape[2] == 4 else np.ones(rgb.shape[:2]) * 255

        # Get only foreground pixels
        foreground_mask = alpha > 128
        foreground_pixels = rgb[foreground_mask]

        if len(foreground_pixels) == 0:
            foreground_pixels = rgb.reshape(-1, 3)

        # Extract dominant colors using k-means-like clustering
        dominant_colors = self._extract_dominant_colors(foreground_pixels, n_colors=5)

        # Determine material type
        material_type = self._detect_material_type(dominant_colors, foreground_pixels)

        # Calculate brightness
        brightness = np.mean(foreground_pixels) / 255.0

        # Calculate contrast
        contrast = np.std(foreground_pixels) / 255.0

        # Check for transparency
        has_transparency = np.any(alpha < 255) and np.any(alpha > 0)

        # Estimate size from bounding box of non-transparent pixels
        if has_transparency:
            rows = np.any(foreground_mask, axis=1)
            cols = np.any(foreground_mask, axis=0)
            if np.any(rows) and np.any(cols):
                rmin, rmax = np.where(rows)[0][[0, -1]]
                cmin, cmax = np.where(cols)[0][[0, -1]]
                estimated_size = (cmax - cmin, rmax - rmin)
            else:
                estimated_size = image.size
        else:
            estimated_size = image.size

        # Generate color palette names
        color_palette = [self._get_color_name(color) for color in dominant_colors[:3]]

        # Calculate sparkle factor (based on brightness variance)
        sparkle_factor = self._calculate_sparkle_factor(foreground_pixels)

        return JewelryFeatures(
            dominant_colors=dominant_colors,
            material_type=material_type,
            brightness=brightness,
            contrast=contrast,
            has_transparency=has_transparency,
            estimated_size=estimated_size,
            color_palette=color_palette,
            sparkle_factor=sparkle_factor
        )

    def _extract_dominant_colors(
        self,
        pixels: np.ndarray,
        n_colors: int = 5
    ) -> List[Tuple[int, int, int]]:
        """
        Extract dominant colors from pixel array using simplified clustering.

        Args:
            pixels: Numpy array of RGB pixels
            n_colors: Number of dominant colors to extract

        Returns:
            List of RGB tuples representing dominant colors
        """
        if len(pixels) == 0:
            return [(128, 128, 128)]

        # Quantize colors to reduce complexity
        quantized = (pixels // 32) * 32

        # Count color occurrences
        color_counts = Counter(map(tuple, quantized))

        # Get most common colors
        most_common = color_counts.most_common(n_colors)
        dominant_colors = [color for color, count in most_common]

        return dominant_colors

    def _detect_material_type(
        self,
        dominant_colors: List[Tuple[int, int, int]],
        pixels: np.ndarray
    ) -> MaterialType:
        """
        Detect the material type based on color analysis.

        Args:
            dominant_colors: List of dominant RGB colors
            pixels: Array of foreground pixels

        Returns:
            Detected MaterialType
        """
        if len(dominant_colors) == 0:
            return MaterialType.UNKNOWN

        # Convert dominant colors to HSV-like values for analysis
        scores = {material: 0 for material in MaterialType}

        for color in dominant_colors[:3]:
            r, g, b = color

            # Simple HSV approximation
            max_c = max(r, g, b)
            min_c = min(r, g, b)
            diff = max_c - min_c

            # Value (brightness)
            value = max_c / 255.0

            # Saturation
            saturation = diff / max_c if max_c > 0 else 0

            # Hue approximation
            if diff == 0:
                hue = 0
            elif max_c == r:
                hue = 60 * (((g - b) / diff) % 6)
            elif max_c == g:
                hue = 60 * (((b - r) / diff) + 2)
            else:
                hue = 60 * (((r - g) / diff) + 4)

            # Score each material type
            # Gold detection
            if 20 <= hue <= 50 and saturation > 0.3 and value > 0.5:
                scores[MaterialType.GOLD] += 3

            # Silver detection (low saturation, high value)
            if saturation < 0.15 and value > 0.6:
                scores[MaterialType.SILVER] += 2

            # Rose gold detection
            if 0 <= hue <= 30 and 0.2 <= saturation <= 0.5 and value > 0.5:
                scores[MaterialType.ROSE_GOLD] += 2

            # Pearl detection (very low saturation, very high value)
            if saturation < 0.1 and value > 0.85:
                scores[MaterialType.PEARL] += 2

            # Copper detection
            if 10 <= hue <= 35 and saturation > 0.4:
                scores[MaterialType.COPPER] += 1

        # Check for high variance indicating gemstones or diamonds
        if np.std(pixels) > 80:
            scores[MaterialType.DIAMOND] += 1
            scores[MaterialType.GEMSTONE] += 1

        # Return the material with highest score
        best_material = max(scores, key=scores.get)

        if scores[best_material] == 0:
            return MaterialType.UNKNOWN

        return best_material

    def _get_color_name(self, rgb: Tuple[int, int, int]) -> str:
        """
        Get a human-readable color name for an RGB value.

        Args:
            rgb: RGB tuple

        Returns:
            Color name string
        """
        # Find closest named color
        min_distance = float('inf')
        closest_name = "unknown"

        for named_rgb, name in self.COLOR_NAMES.items():
            distance = sum((a - b) ** 2 for a, b in zip(rgb, named_rgb))
            if distance < min_distance:
                min_distance = distance
                closest_name = name

        # If distance is too large, generate a descriptive name
        if min_distance > 30000:
            r, g, b = rgb
            if r > g and r > b:
                if r > 200:
                    return "bright red-toned"
                return "red-toned"
            elif g > r and g > b:
                return "green-toned"
            elif b > r and b > g:
                return "blue-toned"
            elif r > 200 and g > 200 and b > 200:
                return "light"
            elif r < 50 and g < 50 and b < 50:
                return "dark"
            else:
                return "neutral"

        return closest_name

    def _calculate_sparkle_factor(self, pixels: np.ndarray) -> float:
        """
        Calculate a sparkle factor based on brightness variance.
        Higher variance suggests more reflective/sparkly jewelry.

        Args:
            pixels: Array of foreground pixels

        Returns:
            Sparkle factor between 0 and 1
        """
        if len(pixels) == 0:
            return 0.0

        # Calculate local brightness variance
        brightness = np.mean(pixels, axis=1)
        variance = np.var(brightness)

        # Normalize to 0-1 range
        sparkle = min(variance / 5000, 1.0)

        return sparkle

    def detect_jewelry_type(
        self,
        image: Image.Image,
        aspect_ratio_hint: Optional[float] = None
    ) -> JewelryType:
        """
        Detect the type of jewelry from the image.

        Args:
            image: Input PIL Image
            aspect_ratio_hint: Optional aspect ratio to help detection

        Returns:
            Detected JewelryType
        """
        width, height = image.size
        aspect_ratio = width / height if height > 0 else 1.0

        if aspect_ratio_hint is not None:
            aspect_ratio = aspect_ratio_hint

        # Simple heuristics based on aspect ratio
        if 0.8 <= aspect_ratio <= 1.2:
            # Square-ish images could be earrings, rings, or pendants
            if max(width, height) < 200:
                return JewelryType.EARRING
            return JewelryType.PENDANT
        elif aspect_ratio > 2.0:
            # Wide images suggest bangles or bracelets
            return JewelryType.BANGLE
        elif aspect_ratio < 0.5:
            # Tall images suggest necklaces or chains
            return JewelryType.NECKLACE
        elif 1.2 < aspect_ratio <= 2.0:
            # Slightly wide could be bracelet
            return JewelryType.BRACELET
        else:
            # Default to necklace for typical jewelry images
            return JewelryType.NECKLACE

    def generate_description(
        self,
        features: JewelryFeatures,
        jewelry_type: JewelryType
    ) -> str:
        """
        Generate a natural language description of the jewelry.

        Args:
            features: Extracted jewelry features
            jewelry_type: Type of jewelry

        Returns:
            Natural language description string
        """
        # Build description components
        material_desc = self._get_material_description(features.material_type)
        color_desc = self._get_color_description(features.color_palette)
        finish_desc = self._get_finish_description(features)
        type_desc = jewelry_type.value

        # Construct the description
        descriptions = []

        # Main description
        if material_desc:
            descriptions.append(f"A {material_desc} {type_desc}")
        else:
            descriptions.append(f"A {type_desc}")

        # Color details
        if color_desc:
            descriptions.append(f"featuring {color_desc} tones")

        # Finish and sparkle
        if finish_desc:
            descriptions.append(f"with a {finish_desc}")

        # Size hint
        if features.estimated_size[0] > 300 or features.estimated_size[1] > 300:
            descriptions.append("in a statement size")
        elif features.estimated_size[0] < 100 and features.estimated_size[1] < 100:
            descriptions.append("in a delicate size")

        return " ".join(descriptions) + "."

    def _get_material_description(self, material: MaterialType) -> str:
        """Get descriptive text for material type."""
        material_descriptions = {
            MaterialType.GOLD: "luxurious gold",
            MaterialType.SILVER: "elegant silver",
            MaterialType.ROSE_GOLD: "romantic rose gold",
            MaterialType.PLATINUM: "premium platinum",
            MaterialType.COPPER: "warm copper",
            MaterialType.PEARL: "classic pearl",
            MaterialType.DIAMOND: "brilliant diamond-studded",
            MaterialType.GEMSTONE: "colorful gemstone-adorned",
            MaterialType.MIXED: "mixed metal",
            MaterialType.UNKNOWN: ""
        }
        return material_descriptions.get(material, "")

    def _get_color_description(self, color_palette: List[str]) -> str:
        """Generate color description from palette."""
        if not color_palette:
            return ""

        unique_colors = list(dict.fromkeys(color_palette))  # Remove duplicates preserving order

        if len(unique_colors) == 1:
            return unique_colors[0]
        elif len(unique_colors) == 2:
            return f"{unique_colors[0]} and {unique_colors[1]}"
        else:
            return f"{', '.join(unique_colors[:-1])}, and {unique_colors[-1]}"

    def _get_finish_description(self, features: JewelryFeatures) -> str:
        """Generate finish description based on features."""
        descriptions = []

        if features.sparkle_factor > 0.7:
            descriptions.append("brilliant sparkle")
        elif features.sparkle_factor > 0.4:
            descriptions.append("subtle shimmer")

        if features.brightness > 0.7:
            descriptions.append("polished finish")
        elif features.brightness < 0.3:
            descriptions.append("matte finish")

        if descriptions:
            return " and ".join(descriptions)
        return ""

    def process(
        self,
        image: Image.Image,
        jewelry_type_hint: Optional[JewelryType] = None
    ) -> ProcessedJewelry:
        """
        Full processing pipeline for jewelry image.

        Args:
            image: Input PIL Image
            jewelry_type_hint: Optional hint for jewelry type

        Returns:
            ProcessedJewelry object with all processed data
        """
        logger.info("Starting jewelry processing pipeline")

        # Step 1: Remove background
        processed_image, mask = self.remove_background(image)
        logger.info("Background removal complete")

        # Step 2: Extract features
        features = self.extract_features(processed_image)
        logger.info(f"Features extracted: material={features.material_type.value}")

        # Step 3: Detect jewelry type
        if jewelry_type_hint:
            jewelry_type = jewelry_type_hint
        else:
            jewelry_type = self.detect_jewelry_type(processed_image)
        logger.info(f"Jewelry type detected: {jewelry_type.value}")

        # Step 4: Generate description
        description = self.generate_description(features, jewelry_type)
        logger.info(f"Description generated: {description}")

        return ProcessedJewelry(
            original_image=image,
            processed_image=processed_image,
            mask=mask,
            features=features,
            jewelry_type=jewelry_type,
            description=description
        )


def process_jewelry_image(
    image_path: str,
    jewelry_type: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function to process a jewelry image from file path.

    Args:
        image_path: Path to the jewelry image
        jewelry_type: Optional jewelry type hint

    Returns:
        Dictionary with processing results
    """
    processor = JewelryProcessor()
    image = Image.open(image_path)

    type_hint = None
    if jewelry_type:
        try:
            type_hint = JewelryType(jewelry_type.lower())
        except ValueError:
            logger.warning(f"Unknown jewelry type: {jewelry_type}")

    result = processor.process(image, type_hint)

    return {
        'processed_image': result.processed_image,
        'mask': result.mask,
        'jewelry_type': result.jewelry_type.value,
        'material': result.features.material_type.value,
        'colors': result.features.color_palette,
        'description': result.description,
        'brightness': result.features.brightness,
        'sparkle_factor': result.features.sparkle_factor
    }


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        result = process_jewelry_image(image_path)
        print(f"Jewelry Type: {result['jewelry_type']}")
        print(f"Material: {result['material']}")
        print(f"Colors: {result['colors']}")
        print(f"Description: {result['description']}")
    else:
        print("Usage: python jewelry_processor.py <image_path>")
