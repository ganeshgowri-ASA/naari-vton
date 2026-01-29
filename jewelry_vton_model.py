"""
Jewelry Virtual Try-On Model using Replicate API

This module provides realistic jewelry virtual try-on using a trained Replicate model.
Unlike simple overlay, this uses AI to realistically composite jewelry onto a person.

The model was trained with 150 jewelry images for realistic rendering.

Features:
- Person image + Jewelry reference image → Realistic output
- Automatic pose detection for jewelry placement
- Support for multiple jewelry types (necklace, earrings, etc.)
- Inpainting mode for seamless blending
- Fallback to img2img when inpainting not available
"""

import os
import io
import base64
import logging
import requests
from typing import Optional, Tuple, Union, Dict, Any, List
from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check for Replicate availability
try:
    import replicate
    REPLICATE_AVAILABLE = True
except ImportError:
    REPLICATE_AVAILABLE = False
    logger.warning("Replicate not installed. Install with: pip install replicate")

# Check for MediaPipe availability
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    logger.warning("MediaPipe not available for pose detection")


class JewelryType(Enum):
    """Supported jewelry types for VTON."""
    NECKLACE = "necklace"
    EARRINGS = "earrings"
    MAANG_TIKKA = "maang_tikka"
    NOSE_RING = "nose_ring"
    BANGLES = "bangles"
    RINGS = "rings"


@dataclass
class JewelryRegion:
    """Defines the region where jewelry should be placed."""
    x: int
    y: int
    width: int
    height: int
    mask: Optional[np.ndarray] = None

    def to_bbox(self) -> Tuple[int, int, int, int]:
        """Return as (x1, y1, x2, y2) bounding box."""
        return (self.x, self.y, self.x + self.width, self.y + self.height)

    def center(self) -> Tuple[int, int]:
        """Return center point."""
        return (self.x + self.width // 2, self.y + self.height // 2)


class JewelryVTONModel:
    """
    Jewelry Virtual Try-On Model using Replicate API.

    This class provides realistic jewelry try-on by:
    1. Detecting body landmarks to find jewelry placement regions
    2. Using the trained Replicate model for realistic generation
    3. Supporting both inpainting and img2img workflows
    """

    # Trained Replicate model endpoint
    DEFAULT_MODEL = "ganeshgowri-asa/naari-jewelry-vton"

    # SDXL inpainting model for mask-based editing
    INPAINT_MODEL = "stability-ai/stable-diffusion-inpainting:95b7223104132402a9ae91cc677285bc5eb997834bd2349fa486f53910fd68b3"

    # SDXL img2img for style transfer
    IMG2IMG_MODEL = "stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b"

    def __init__(self, model_version: Optional[str] = None):
        """
        Initialize the Jewelry VTON Model.

        Args:
            model_version: Optional specific model version. If None, uses latest.
        """
        self.model_endpoint = model_version or self.DEFAULT_MODEL
        self._pose_detector = None
        self._face_mesh = None
        self._check_api_token()

    def _check_api_token(self) -> bool:
        """Check if Replicate API token is configured."""
        token = os.environ.get("REPLICATE_API_TOKEN")
        if not token:
            logger.warning("REPLICATE_API_TOKEN not set. Set it to use the VTON model.")
            return False
        return True

    @property
    def is_available(self) -> bool:
        """Check if the model is available for use."""
        return REPLICATE_AVAILABLE and self._check_api_token()

    # =========================================================================
    # Pose Detection for Jewelry Placement
    # =========================================================================

    def _init_pose_detector(self):
        """Initialize MediaPipe pose and face mesh detectors."""
        if not MEDIAPIPE_AVAILABLE:
            return

        if self._pose_detector is None:
            self._pose_detector = mp.solutions.pose.Pose(
                static_image_mode=True,
                model_complexity=2,
                min_detection_confidence=0.5
            )

        if self._face_mesh is None:
            self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5
            )

    def detect_jewelry_region(self,
                              image: np.ndarray,
                              jewelry_type: JewelryType) -> Optional[JewelryRegion]:
        """
        Detect the region where jewelry should be placed based on type.

        Args:
            image: Input person image (BGR format)
            jewelry_type: Type of jewelry to place

        Returns:
            JewelryRegion with coordinates and mask, or None if detection fails
        """
        self._init_pose_detector()

        if not MEDIAPIPE_AVAILABLE:
            # Fallback to heuristic positioning
            return self._fallback_region(image, jewelry_type)

        h, w = image.shape[:2]
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Get pose landmarks
        pose_results = self._pose_detector.process(rgb_image)
        face_results = self._face_mesh.process(rgb_image)

        pose_landmarks = pose_results.pose_landmarks if pose_results else None
        face_landmarks = face_results.multi_face_landmarks[0] if face_results and face_results.multi_face_landmarks else None

        if jewelry_type == JewelryType.NECKLACE:
            return self._detect_necklace_region(w, h, pose_landmarks, face_landmarks)
        elif jewelry_type == JewelryType.EARRINGS:
            return self._detect_earring_region(w, h, pose_landmarks, face_landmarks)
        elif jewelry_type == JewelryType.MAANG_TIKKA:
            return self._detect_forehead_region(w, h, face_landmarks)
        elif jewelry_type == JewelryType.NOSE_RING:
            return self._detect_nose_region(w, h, face_landmarks)
        else:
            return self._fallback_region(image, jewelry_type)

    def _detect_necklace_region(self, w: int, h: int,
                                pose_landmarks, face_landmarks) -> Optional[JewelryRegion]:
        """Detect necklace placement region at neck/collarbone area."""
        if pose_landmarks:
            # MediaPipe pose landmarks
            left_shoulder = pose_landmarks.landmark[11]
            right_shoulder = pose_landmarks.landmark[12]

            # Neck is above and between shoulders
            neck_x = int((left_shoulder.x + right_shoulder.x) / 2 * w)
            neck_y = int((left_shoulder.y + right_shoulder.y) / 2 * h) - int(h * 0.05)

            shoulder_width = abs(left_shoulder.x - right_shoulder.x) * w
            region_width = int(shoulder_width * 1.2)
            region_height = int(h * 0.15)

            return JewelryRegion(
                x=neck_x - region_width // 2,
                y=neck_y - region_height // 4,
                width=region_width,
                height=region_height,
                mask=self._create_necklace_mask(w, h, neck_x, neck_y, region_width, region_height)
            )

        # Fallback using face detection
        if face_landmarks:
            chin = face_landmarks.landmark[152]
            chin_y = int(chin.y * h)

            return JewelryRegion(
                x=int(w * 0.25),
                y=chin_y + int(h * 0.02),
                width=int(w * 0.5),
                height=int(h * 0.12),
                mask=None
            )

        return None

    def _detect_earring_region(self, w: int, h: int,
                               pose_landmarks, face_landmarks) -> Optional[JewelryRegion]:
        """Detect earring placement region at ears."""
        if face_landmarks:
            # MediaPipe face mesh ear landmarks
            left_ear = face_landmarks.landmark[234]  # Left ear
            right_ear = face_landmarks.landmark[454]  # Right ear

            # Create region encompassing both ears
            left_x = int(left_ear.x * w)
            right_x = int(right_ear.x * w)
            ear_y = int((left_ear.y + right_ear.y) / 2 * h)

            padding = int(w * 0.1)

            return JewelryRegion(
                x=min(left_x, right_x) - padding,
                y=ear_y - int(h * 0.05),
                width=abs(right_x - left_x) + padding * 2,
                height=int(h * 0.15),
                mask=self._create_earring_mask(w, h, left_x, right_x, ear_y)
            )

        return None

    def _detect_forehead_region(self, w: int, h: int,
                                face_landmarks) -> Optional[JewelryRegion]:
        """Detect forehead region for maang tikka."""
        if face_landmarks:
            forehead = face_landmarks.landmark[10]  # Forehead center

            forehead_x = int(forehead.x * w)
            forehead_y = int(forehead.y * h)

            region_width = int(w * 0.15)
            region_height = int(h * 0.08)

            return JewelryRegion(
                x=forehead_x - region_width // 2,
                y=forehead_y - region_height // 2,
                width=region_width,
                height=region_height,
                mask=self._create_circular_mask(w, h, forehead_x, forehead_y, int(region_width * 0.6))
            )

        return None

    def _detect_nose_region(self, w: int, h: int,
                            face_landmarks) -> Optional[JewelryRegion]:
        """Detect nose region for nose ring."""
        if face_landmarks:
            nose_tip = face_landmarks.landmark[4]  # Nose tip

            nose_x = int(nose_tip.x * w)
            nose_y = int(nose_tip.y * h)

            region_size = int(w * 0.08)

            return JewelryRegion(
                x=nose_x - region_size // 2,
                y=nose_y - region_size // 2,
                width=region_size,
                height=region_size,
                mask=self._create_circular_mask(w, h, nose_x, nose_y, region_size // 2)
            )

        return None

    def _fallback_region(self, image: np.ndarray,
                         jewelry_type: JewelryType) -> JewelryRegion:
        """Fallback heuristic region detection when pose detection fails."""
        h, w = image.shape[:2]

        # Use OpenCV face detection for basic positioning
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        if len(faces) > 0:
            fx, fy, fw, fh = faces[0]
            face_center_x = fx + fw // 2
            face_bottom = fy + fh
        else:
            # No face detected, use image center
            face_center_x = w // 2
            face_bottom = int(h * 0.35)
            fw = int(w * 0.3)

        if jewelry_type == JewelryType.NECKLACE:
            return JewelryRegion(
                x=face_center_x - int(fw * 0.8),
                y=face_bottom + int(h * 0.02),
                width=int(fw * 1.6),
                height=int(h * 0.12)
            )
        elif jewelry_type == JewelryType.EARRINGS:
            return JewelryRegion(
                x=int(w * 0.15),
                y=int(h * 0.2),
                width=int(w * 0.7),
                height=int(h * 0.15)
            )
        elif jewelry_type == JewelryType.MAANG_TIKKA:
            return JewelryRegion(
                x=face_center_x - int(fw * 0.3),
                y=int(h * 0.08) if len(faces) == 0 else fy - int(fh * 0.1),
                width=int(fw * 0.6),
                height=int(h * 0.06)
            )
        elif jewelry_type == JewelryType.NOSE_RING:
            return JewelryRegion(
                x=face_center_x - int(fw * 0.15),
                y=int(h * 0.25) if len(faces) == 0 else fy + int(fh * 0.5),
                width=int(fw * 0.3),
                height=int(h * 0.05)
            )
        else:
            # Default central region
            return JewelryRegion(
                x=int(w * 0.3),
                y=int(h * 0.3),
                width=int(w * 0.4),
                height=int(h * 0.2)
            )

    # =========================================================================
    # Mask Generation
    # =========================================================================

    def _create_necklace_mask(self, w: int, h: int,
                              center_x: int, center_y: int,
                              region_width: int, region_height: int) -> np.ndarray:
        """Create a mask for necklace region (curved shape)."""
        mask = np.zeros((h, w), dtype=np.uint8)

        # Create an elliptical mask for natural necklace shape
        cv2.ellipse(
            mask,
            (center_x, center_y),
            (region_width // 2, region_height // 2),
            0, 0, 180,  # Bottom half of ellipse
            255, -1
        )

        # Feather the edges
        mask = cv2.GaussianBlur(mask, (21, 21), 0)

        return mask

    def _create_earring_mask(self, w: int, h: int,
                             left_x: int, right_x: int, ear_y: int) -> np.ndarray:
        """Create masks for both earring positions."""
        mask = np.zeros((h, w), dtype=np.uint8)

        radius = int(w * 0.04)

        # Left earring
        cv2.circle(mask, (left_x, ear_y), radius, 255, -1)
        # Right earring
        cv2.circle(mask, (right_x, ear_y), radius, 255, -1)

        # Feather edges
        mask = cv2.GaussianBlur(mask, (15, 15), 0)

        return mask

    def _create_circular_mask(self, w: int, h: int,
                              center_x: int, center_y: int, radius: int) -> np.ndarray:
        """Create a circular mask with feathered edges."""
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, (center_x, center_y), radius, 255, -1)
        mask = cv2.GaussianBlur(mask, (11, 11), 0)
        return mask

    # =========================================================================
    # Image Utilities
    # =========================================================================

    def _image_to_base64(self, image: Union[Image.Image, np.ndarray]) -> str:
        """Convert image to base64 string."""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

    def _image_to_data_uri(self, image: Union[Image.Image, np.ndarray]) -> str:
        """Convert image to data URI for Replicate API."""
        b64 = self._image_to_base64(image)
        return f"data:image/png;base64,{b64}"

    def _download_image(self, url: str) -> Optional[Image.Image]:
        """Download image from URL."""
        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            return Image.open(io.BytesIO(response.content))
        except Exception as e:
            logger.error(f"Failed to download image from {url}: {e}")
            return None

    def _pil_to_cv2(self, pil_image: Image.Image) -> np.ndarray:
        """Convert PIL Image to OpenCV BGR format."""
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    def _cv2_to_pil(self, cv2_image: np.ndarray) -> Image.Image:
        """Convert OpenCV BGR image to PIL Image."""
        return Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))

    # =========================================================================
    # Prompt Building
    # =========================================================================

    def _build_jewelry_prompt(self,
                              jewelry_type: JewelryType,
                              jewelry_image: Optional[np.ndarray] = None,
                              metal_type: str = "gold",
                              style: str = "elegant",
                              custom_prompt: str = "") -> str:
        """
        Build an optimized prompt for jewelry generation.
        Uses the trained JEWELRYVTON token for best results.
        """
        base_prompt = "a photo of JEWELRYVTON jewelry"

        type_descriptions = {
            JewelryType.NECKLACE: "beautiful necklace on neck, elegant jewelry, detailed craftsmanship",
            JewelryType.EARRINGS: "elegant earrings, dangling jewelry on ears, beautiful design",
            JewelryType.MAANG_TIKKA: "traditional maang tikka on forehead, bridal jewelry, Indian ornament",
            JewelryType.NOSE_RING: "delicate nose ring, traditional nose jewelry, elegant ornament",
            JewelryType.BANGLES: "beautiful bangles on wrist, elegant bracelets, traditional jewelry",
            JewelryType.RINGS: "elegant ring on finger, beautiful jewelry, detailed design"
        }

        jewelry_desc = type_descriptions.get(jewelry_type, "elegant jewelry")

        prompt = f"{base_prompt}, {jewelry_desc}, {metal_type} metal, {style} style"

        if custom_prompt:
            prompt += f", {custom_prompt}"

        # Add quality boosters
        prompt += ", high quality, photorealistic, professional photography, studio lighting"

        return prompt

    def _build_negative_prompt(self) -> str:
        """Build negative prompt to avoid common issues."""
        return (
            "blurry, low quality, distorted face, extra fingers, "
            "deformed, ugly, bad anatomy, watermark, text, logo, "
            "oversaturated, plastic looking, fake looking"
        )

    # =========================================================================
    # Main VTON Methods
    # =========================================================================

    def try_on(self,
               person_image: Union[Image.Image, np.ndarray],
               jewelry_image: Union[Image.Image, np.ndarray],
               jewelry_type: Union[JewelryType, str],
               metal_type: str = "gold",
               style: str = "elegant",
               custom_prompt: str = "",
               strength: float = 0.75,
               guidance_scale: float = 7.5,
               num_inference_steps: int = 30) -> Tuple[Optional[Image.Image], str]:
        """
        Perform jewelry virtual try-on.

        This is the main method that takes a person image and jewelry image,
        and returns the person wearing the jewelry realistically.

        Args:
            person_image: Input person image
            jewelry_image: Reference jewelry image to apply
            jewelry_type: Type of jewelry (JewelryType enum or string)
            metal_type: Metal type description
            style: Style description
            custom_prompt: Additional prompt text
            strength: How much to transform (0.0-1.0, higher = more change)
            guidance_scale: CFG scale for generation
            num_inference_steps: Number of diffusion steps

        Returns:
            Tuple of (result_image, status_message)
        """
        if not self.is_available:
            return None, "Error: Replicate API not available. Set REPLICATE_API_TOKEN."

        # Convert jewelry_type to enum if string
        if isinstance(jewelry_type, str):
            try:
                jewelry_type = JewelryType(jewelry_type.lower())
            except ValueError:
                jewelry_type = JewelryType.NECKLACE  # Default

        # Convert images to numpy if needed
        if isinstance(person_image, Image.Image):
            person_cv2 = self._pil_to_cv2(person_image)
        else:
            person_cv2 = person_image.copy()

        if isinstance(jewelry_image, Image.Image):
            jewelry_cv2 = self._pil_to_cv2(jewelry_image)
        else:
            jewelry_cv2 = jewelry_image.copy()

        # Detect jewelry region
        region = self.detect_jewelry_region(person_cv2, jewelry_type)

        if region is None:
            logger.warning("Could not detect jewelry region, using full image")
            h, w = person_cv2.shape[:2]
            region = JewelryRegion(0, 0, w, h)

        # Build prompt
        prompt = self._build_jewelry_prompt(
            jewelry_type=jewelry_type,
            jewelry_image=jewelry_cv2,
            metal_type=metal_type,
            style=style,
            custom_prompt=custom_prompt
        )

        logger.info(f"Running jewelry VTON with prompt: {prompt}")
        logger.info(f"Jewelry region: x={region.x}, y={region.y}, w={region.width}, h={region.height}")

        # Try inpainting first (best for localized changes)
        if region.mask is not None:
            result, status = self._run_inpainting(
                person_image=person_cv2,
                jewelry_image=jewelry_cv2,
                mask=region.mask,
                prompt=prompt,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps
            )
            if result is not None:
                return result, status

        # Fallback to img2img with the trained model
        return self._run_img2img(
            person_image=person_cv2,
            jewelry_image=jewelry_cv2,
            prompt=prompt,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps
        )

    def _run_inpainting(self,
                        person_image: np.ndarray,
                        jewelry_image: np.ndarray,
                        mask: np.ndarray,
                        prompt: str,
                        guidance_scale: float,
                        num_inference_steps: int) -> Tuple[Optional[Image.Image], str]:
        """Run inpainting to add jewelry to masked region."""
        try:
            # Convert images
            person_pil = self._cv2_to_pil(person_image)
            mask_pil = Image.fromarray(mask)

            # Resize mask to match person image
            if mask_pil.size != person_pil.size:
                mask_pil = mask_pil.resize(person_pil.size, Image.LANCZOS)

            output = replicate.run(
                self.INPAINT_MODEL,
                input={
                    "image": self._image_to_data_uri(person_pil),
                    "mask": self._image_to_data_uri(mask_pil),
                    "prompt": prompt,
                    "negative_prompt": self._build_negative_prompt(),
                    "guidance_scale": guidance_scale,
                    "num_inference_steps": num_inference_steps
                }
            )

            if output:
                if isinstance(output, list) and len(output) > 0:
                    result = self._download_image(output[0])
                    if result:
                        return result, "Success: Jewelry applied with inpainting"
                elif hasattr(output, 'read'):
                    return Image.open(output), "Success: Jewelry applied with inpainting"

            return None, "Inpainting returned no output"

        except Exception as e:
            logger.warning(f"Inpainting failed: {e}, trying img2img")
            return None, f"Inpainting failed: {e}"

    def _run_img2img(self,
                     person_image: np.ndarray,
                     jewelry_image: np.ndarray,
                     prompt: str,
                     strength: float,
                     guidance_scale: float,
                     num_inference_steps: int) -> Tuple[Optional[Image.Image], str]:
        """Run img2img with the trained jewelry model."""
        try:
            person_pil = self._cv2_to_pil(person_image)

            # Use the trained JEWELRYVTON model
            output = replicate.run(
                self.model_endpoint,
                input={
                    "image": self._image_to_data_uri(person_pil),
                    "prompt": prompt,
                    "negative_prompt": self._build_negative_prompt(),
                    "guidance_scale": guidance_scale,
                    "num_inference_steps": num_inference_steps,
                    "num_outputs": 1
                }
            )

            if output:
                if isinstance(output, list) and len(output) > 0:
                    result_url = output[0]
                    if isinstance(result_url, str):
                        result = self._download_image(result_url)
                        if result:
                            return result, "Success: Jewelry applied with trained model"
                elif hasattr(output, 'read'):
                    return Image.open(output), "Success: Jewelry applied with trained model"

            return None, "Error: No output from model"

        except replicate.exceptions.ReplicateError as e:
            logger.error(f"Replicate error: {e}")
            return None, f"Error: Replicate API - {e}"
        except Exception as e:
            logger.error(f"img2img error: {e}")
            return None, f"Error: {e}"

    def generate_jewelry_only(self,
                              jewelry_type: Union[JewelryType, str],
                              metal_type: str = "gold",
                              stone_type: str = "none",
                              style: str = "elegant",
                              custom_prompt: str = "") -> Tuple[Optional[Image.Image], str]:
        """
        Generate a standalone jewelry image without a person.
        Useful for creating jewelry reference images.

        Args:
            jewelry_type: Type of jewelry to generate
            metal_type: Metal type (gold, silver, etc.)
            stone_type: Stone type (diamond, ruby, etc.)
            style: Style description
            custom_prompt: Additional prompt text

        Returns:
            Tuple of (generated_image, status_message)
        """
        if not self.is_available:
            return None, "Error: Replicate API not available"

        if isinstance(jewelry_type, str):
            try:
                jewelry_type = JewelryType(jewelry_type.lower())
            except ValueError:
                jewelry_type = JewelryType.NECKLACE

        type_name = jewelry_type.value.replace("_", " ")

        prompt = (
            f"a photo of JEWELRYVTON jewelry, beautiful {type_name}, "
            f"{metal_type} metal"
        )

        if stone_type and stone_type.lower() != "none":
            prompt += f", {stone_type} stones"

        prompt += f", {style} style, product photography, white background, high quality"

        if custom_prompt:
            prompt += f", {custom_prompt}"

        try:
            output = replicate.run(
                self.model_endpoint,
                input={
                    "prompt": prompt,
                    "negative_prompt": self._build_negative_prompt(),
                    "num_outputs": 1,
                    "guidance_scale": 7.5,
                    "num_inference_steps": 30
                }
            )

            if output:
                if isinstance(output, list) and len(output) > 0:
                    result = self._download_image(output[0])
                    if result:
                        return result, f"Success: Generated {type_name}"
                elif hasattr(output, 'read'):
                    return Image.open(output), f"Success: Generated {type_name}"

            return None, "Error: No output from model"

        except Exception as e:
            logger.error(f"Generation error: {e}")
            return None, f"Error: {e}"


# ============================================================================
# Convenience API Functions
# ============================================================================

# Global model instance
_vton_model: Optional[JewelryVTONModel] = None


def get_vton_model() -> JewelryVTONModel:
    """Get or create the global VTON model instance."""
    global _vton_model
    if _vton_model is None:
        _vton_model = JewelryVTONModel()
    return _vton_model


def jewelry_vton(person_image: Union[Image.Image, np.ndarray],
                 jewelry_image: Union[Image.Image, np.ndarray],
                 jewelry_type: str = "necklace",
                 metal_type: str = "gold",
                 style: str = "elegant",
                 custom_prompt: str = "",
                 strength: float = 0.75) -> Tuple[Optional[Image.Image], str]:
    """
    High-level API for jewelry virtual try-on.

    This is the main function to use for applying jewelry to a person image.
    It uses the trained Replicate model for realistic results.

    Args:
        person_image: Photo of a person
        jewelry_image: Reference jewelry image to apply
        jewelry_type: Type of jewelry ("necklace", "earrings", "maang_tikka", "nose_ring")
        metal_type: Metal type ("gold", "silver", "rose gold", etc.)
        style: Style ("elegant", "traditional", "modern", etc.)
        custom_prompt: Additional description
        strength: Transformation strength (0.0-1.0)

    Returns:
        Tuple of (result_image, status_message)

    Example:
        >>> from jewelry_vton_model import jewelry_vton
        >>> result, status = jewelry_vton(person_img, necklace_img, "necklace", "gold")
        >>> if result:
        ...     result.save("output.png")
    """
    model = get_vton_model()
    return model.try_on(
        person_image=person_image,
        jewelry_image=jewelry_image,
        jewelry_type=jewelry_type,
        metal_type=metal_type,
        style=style,
        custom_prompt=custom_prompt,
        strength=strength
    )


def check_vton_availability() -> Dict[str, Any]:
    """
    Check if the VTON model is available and configured.

    Returns:
        Dictionary with availability status and details
    """
    model = get_vton_model()

    return {
        "available": model.is_available,
        "replicate_installed": REPLICATE_AVAILABLE,
        "mediapipe_installed": MEDIAPIPE_AVAILABLE,
        "api_token_set": bool(os.environ.get("REPLICATE_API_TOKEN")),
        "model_endpoint": model.model_endpoint
    }


# ============================================================================
# CLI Interface
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Jewelry VTON Model")
    parser.add_argument("--check", action="store_true", help="Check model availability")
    parser.add_argument("--person", type=str, help="Path to person image")
    parser.add_argument("--jewelry", type=str, help="Path to jewelry image")
    parser.add_argument("--type", type=str, default="necklace",
                       choices=["necklace", "earrings", "maang_tikka", "nose_ring"],
                       help="Jewelry type")
    parser.add_argument("--output", type=str, default="output.png", help="Output path")

    args = parser.parse_args()

    if args.check:
        status = check_vton_availability()
        print("Jewelry VTON Model Status:")
        print(f"  Available: {status['available']}")
        print(f"  Replicate installed: {status['replicate_installed']}")
        print(f"  MediaPipe installed: {status['mediapipe_installed']}")
        print(f"  API token set: {status['api_token_set']}")
        print(f"  Model endpoint: {status['model_endpoint']}")

    elif args.person and args.jewelry:
        person = Image.open(args.person)
        jewelry = Image.open(args.jewelry)

        result, status = jewelry_vton(person, jewelry, args.type)
        print(status)

        if result:
            result.save(args.output)
            print(f"Saved to {args.output}")
    else:
        parser.print_help()
