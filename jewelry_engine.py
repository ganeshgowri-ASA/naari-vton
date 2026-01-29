"""
Naari Studio - Jewelry Virtual Try-On Engine

Supports two modes:
1. OVERLAY MODE: Uses OpenCV Haar Cascades for jewelry overlay positioning
2. GENERATION MODE: Uses Replicate trained model to generate jewelry on person images

Features:
- AI-powered jewelry generation via Replicate model (ganeshgowri-asa/naari-jewelry-vton:f6b844b4)
- Pose detection using cvzone for accurate landmark detection (neck/ear/wrist)
- Traditional overlay mode for pre-designed jewelry images
- Support for necklace, earrings, bangles, rings, maang tikka, nose ring
"""

import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Optional, Union, Dict, Any
import logging
import math
import os
import io
import base64
import requests
from urllib.parse import urlparse

# Optional imports for enhanced features
try:
    import replicate
    REPLICATE_AVAILABLE = True
except ImportError:
    REPLICATE_AVAILABLE = False

try:
    from cvzone.PoseModule import PoseDetector
    CVZONE_AVAILABLE = True
except ImportError:
    CVZONE_AVAILABLE = False

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# OpenCV Face Detection - ZeroGPU compatible
FACE_CASCADE = None
EYE_CASCADE = None


def _init_cascades():
    """Initialize OpenCV Haar Cascades for face/eye detection."""
    global FACE_CASCADE, EYE_CASCADE

    if FACE_CASCADE is not None:
        return True

    try:
        cv2_data_path = cv2.data.haarcascades

        face_cascade_path = os.path.join(cv2_data_path, 'haarcascade_frontalface_default.xml')
        eye_cascade_path = os.path.join(cv2_data_path, 'haarcascade_eye.xml')

        if os.path.exists(face_cascade_path):
            FACE_CASCADE = cv2.CascadeClassifier(face_cascade_path)
            logger.info("OpenCV face cascade loaded successfully")
        else:
            logger.warning(f"Face cascade not found at {face_cascade_path}")
            return False

        if os.path.exists(eye_cascade_path):
            EYE_CASCADE = cv2.CascadeClassifier(eye_cascade_path)
            logger.info("OpenCV eye cascade loaded successfully")

        return True

    except Exception as e:
        logger.error(f"Failed to load cascades: {e}")
        return False


# Initialize cascades at module load
_init_cascades()

# ============================================================================
# REPLICATE MODEL CONFIGURATION
# ============================================================================

# Trained jewelry generation model endpoint
REPLICATE_MODEL = "ganeshgowri-asa/naari-jewelry-vton:f6b844b4"

# Jewelry type configurations
JEWELRY_TYPES = {
    "necklace": {
        "prompt_prefix": "a beautiful JEWELRYVTON necklace",
        "landmarks": ["neck", "shoulders"],
        "default_style": "elegant"
    },
    "earrings": {
        "prompt_prefix": "a pair of JEWELRYVTON earrings",
        "landmarks": ["ears"],
        "default_style": "drop"
    },
    "bangles": {
        "prompt_prefix": "a set of JEWELRYVTON bangles on wrist",
        "landmarks": ["wrist"],
        "default_style": "traditional"
    },
    "rings": {
        "prompt_prefix": "a JEWELRYVTON ring on finger",
        "landmarks": ["hand"],
        "default_style": "solitaire"
    },
    "maang_tikka": {
        "prompt_prefix": "a JEWELRYVTON maang tikka on forehead",
        "landmarks": ["forehead"],
        "default_style": "bridal"
    },
    "nose_ring": {
        "prompt_prefix": "a JEWELRYVTON nose ring",
        "landmarks": ["nose"],
        "default_style": "stud"
    }
}

# Metal type options
METAL_TYPES = ["gold", "silver", "rose gold", "platinum", "oxidized silver", "antique gold"]

# Stone options
STONE_TYPES = ["diamond", "ruby", "emerald", "sapphire", "pearl", "kundan", "polki", "none"]

# Style options per jewelry type
STYLE_OPTIONS = {
    "necklace": ["choker", "princess", "matinee", "opera", "statement", "layered", "pendant"],
    "earrings": ["studs", "drops", "hoops", "chandeliers", "jhumkas", "cuffs"],
    "bangles": ["traditional", "modern", "kada", "charm", "cuff", "tennis"],
    "rings": ["solitaire", "band", "cluster", "eternity", "cocktail", "stackable"],
    "maang_tikka": ["bridal", "simple", "elaborate", "kundan", "pearl"],
    "nose_ring": ["stud", "hoop", "nath", "septum"]
}


# ============================================================================
# POSE DETECTION ENGINE (cvzone/MediaPipe)
# ============================================================================

class PoseDetectionEngine:
    """
    Enhanced pose detection using cvzone PoseModule and MediaPipe.
    Provides accurate landmark detection for jewelry placement.
    """

    def __init__(self):
        """Initialize pose detection with available backends."""
        self.pose_detector = None
        self.face_mesh = None
        self.mp_pose = None

        # Try to initialize cvzone PoseDetector
        if CVZONE_AVAILABLE:
            try:
                self.pose_detector = PoseDetector(staticMode=False, modelComplexity=1)
                logger.info("cvzone PoseDetector initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize cvzone PoseDetector: {e}")

        # Try to initialize MediaPipe Face Mesh for face landmarks
        if MEDIAPIPE_AVAILABLE:
            try:
                self.mp_face_mesh = mp.solutions.face_mesh
                self.face_mesh = self.mp_face_mesh.FaceMesh(
                    static_image_mode=True,
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5
                )
                self.mp_pose = mp.solutions.pose.Pose(
                    static_image_mode=True,
                    model_complexity=1,
                    min_detection_confidence=0.5
                )
                logger.info("MediaPipe Face Mesh and Pose initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize MediaPipe: {e}")

    def detect_landmarks(self, cv2_image: np.ndarray) -> Dict[str, Any]:
        """
        Detect all relevant landmarks for jewelry placement.

        Args:
            cv2_image: OpenCV image (BGR format)

        Returns:
            Dictionary with detected landmarks and confidence scores
        """
        img_h, img_w = cv2_image.shape[:2]
        landmarks = {
            "detected": False,
            "image_size": (img_w, img_h),
            "neck": None,
            "shoulders": None,
            "left_ear": None,
            "right_ear": None,
            "left_wrist": None,
            "right_wrist": None,
            "forehead": None,
            "nose": None,
            "face_bounds": None,
            "pose_landmarks": None,
            "face_landmarks": None
        }

        # Try cvzone pose detection first
        if self.pose_detector is not None:
            try:
                img_rgb = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
                img_detected, lm_list = self.pose_detector.findPose(img_rgb, draw=False)

                if lm_list and len(lm_list) > 0:
                    landmarks["detected"] = True
                    landmarks["pose_landmarks"] = lm_list

                    # Extract key landmarks (cvzone uses 33 pose landmarks)
                    # 0: nose, 7: left ear, 8: right ear, 9: mouth left, 10: mouth right
                    # 11: left shoulder, 12: right shoulder
                    # 15: left wrist, 16: right wrist

                    if len(lm_list) > 12:
                        # Neck position (midpoint between shoulders)
                        left_shoulder = lm_list[11][:2] if len(lm_list) > 11 else None
                        right_shoulder = lm_list[12][:2] if len(lm_list) > 12 else None

                        if left_shoulder and right_shoulder:
                            neck_x = (left_shoulder[0] + right_shoulder[0]) // 2
                            neck_y = min(left_shoulder[1], right_shoulder[1]) - 20
                            landmarks["neck"] = (neck_x, neck_y)
                            landmarks["shoulders"] = {
                                "left": tuple(left_shoulder),
                                "right": tuple(right_shoulder),
                                "width": abs(right_shoulder[0] - left_shoulder[0])
                            }

                    # Ears
                    if len(lm_list) > 8:
                        landmarks["left_ear"] = tuple(lm_list[7][:2]) if lm_list[7] else None
                        landmarks["right_ear"] = tuple(lm_list[8][:2]) if lm_list[8] else None

                    # Wrists
                    if len(lm_list) > 16:
                        landmarks["left_wrist"] = tuple(lm_list[15][:2]) if lm_list[15] else None
                        landmarks["right_wrist"] = tuple(lm_list[16][:2]) if lm_list[16] else None

                    # Nose
                    if len(lm_list) > 0 and lm_list[0]:
                        landmarks["nose"] = tuple(lm_list[0][:2])

                    logger.info(f"cvzone detected landmarks: neck={landmarks['neck']}, ears=({landmarks['left_ear']}, {landmarks['right_ear']})")

            except Exception as e:
                logger.warning(f"cvzone pose detection failed: {e}")

        # Try MediaPipe for face landmarks if cvzone didn't find everything
        if self.face_mesh is not None:
            try:
                rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
                results = self.face_mesh.process(rgb_image)

                if results.multi_face_landmarks:
                    face_lm = results.multi_face_landmarks[0]
                    landmarks["detected"] = True
                    landmarks["face_landmarks"] = face_lm

                    # Get key face landmarks
                    # 10: forehead center, 234: left ear, 454: right ear
                    # 1: nose tip, 4: nose bridge

                    def get_pixel_coords(landmark_idx):
                        lm = face_lm.landmark[landmark_idx]
                        return (int(lm.x * img_w), int(lm.y * img_h))

                    # Forehead (top of face)
                    landmarks["forehead"] = get_pixel_coords(10)

                    # Ears (if not already detected)
                    if landmarks["left_ear"] is None:
                        landmarks["left_ear"] = get_pixel_coords(234)
                    if landmarks["right_ear"] is None:
                        landmarks["right_ear"] = get_pixel_coords(454)

                    # Nose
                    if landmarks["nose"] is None:
                        landmarks["nose"] = get_pixel_coords(1)

                    # Face bounding box from landmarks
                    all_x = [face_lm.landmark[i].x * img_w for i in range(468)]
                    all_y = [face_lm.landmark[i].y * img_h for i in range(468)]
                    landmarks["face_bounds"] = {
                        "x": int(min(all_x)),
                        "y": int(min(all_y)),
                        "width": int(max(all_x) - min(all_x)),
                        "height": int(max(all_y) - min(all_y))
                    }

                    logger.info(f"MediaPipe detected face landmarks: forehead={landmarks['forehead']}")

            except Exception as e:
                logger.warning(f"MediaPipe face detection failed: {e}")

        # Fallback to OpenCV Haar Cascade if nothing else worked
        if not landmarks["detected"] and FACE_CASCADE is not None:
            face_data = self._detect_face_opencv(cv2_image)
            if face_data:
                landmarks["detected"] = True
                landmarks.update(face_data)

        return landmarks

    def _detect_face_opencv(self, cv2_image: np.ndarray) -> Optional[Dict]:
        """Fallback face detection using OpenCV Haar Cascades."""
        if FACE_CASCADE is None:
            return None

        try:
            img_h, img_w = cv2_image.shape[:2]
            gray = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)

            faces = FACE_CASCADE.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=3,
                minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE
            )

            if len(faces) == 0:
                return None

            # Take largest face
            face = max(faces, key=lambda f: f[2] * f[3])
            x, y, w, h = face

            return {
                "face_bounds": {"x": x, "y": y, "width": w, "height": h},
                "forehead": (x + w // 2, y + int(h * 0.05)),
                "neck": (x + w // 2, y + h + int(h * 0.2)),
                "left_ear": (x - int(w * 0.08), y + int(h * 0.6)),
                "right_ear": (x + w + int(w * 0.08), y + int(h * 0.6)),
                "nose": (x + w // 2, y + int(h * 0.65)),
                "shoulders": {
                    "left": (x - int(w * 0.3), y + h + int(h * 0.3)),
                    "right": (x + w + int(w * 0.3), y + h + int(h * 0.3)),
                    "width": int(w * 1.6)
                }
            }
        except Exception as e:
            logger.error(f"OpenCV face detection failed: {e}")
            return None


# Global pose detection engine instance
_pose_engine: Optional[PoseDetectionEngine] = None


def get_pose_engine() -> PoseDetectionEngine:
    """Get or create the global pose detection engine instance."""
    global _pose_engine
    if _pose_engine is None:
        _pose_engine = PoseDetectionEngine()
    return _pose_engine


# ============================================================================
# JEWELRY GENERATION ENGINE (Replicate Model)
# ============================================================================

class JewelryGenerationEngine:
    """
    AI-powered jewelry generation using the trained Replicate model.
    Generates jewelry images based on text prompts and applies them to person images.
    """

    def __init__(self, model_endpoint: str = REPLICATE_MODEL):
        """
        Initialize the jewelry generation engine.

        Args:
            model_endpoint: Replicate model endpoint (default: ganeshgowri-asa/naari-jewelry-vton:f6b844b4)
        """
        self.model_endpoint = model_endpoint
        self.pose_engine = get_pose_engine()
        self._overlay_engine = None

        if not REPLICATE_AVAILABLE:
            logger.warning("Replicate package not installed. Install with: pip install replicate")

    @property
    def overlay_engine(self):
        """Lazy load the overlay engine."""
        if self._overlay_engine is None:
            self._overlay_engine = JewelryEngine()
        return self._overlay_engine

    def _build_prompt(self, jewelry_type: str, metal_type: str = "gold",
                      stones: str = "none", style: str = None,
                      custom_prompt: str = None) -> str:
        """
        Build a detailed prompt for jewelry generation.

        Args:
            jewelry_type: Type of jewelry (necklace, earrings, etc.)
            metal_type: Metal type (gold, silver, etc.)
            stones: Stone type (diamond, ruby, etc.)
            style: Style variant
            custom_prompt: Custom prompt to append

        Returns:
            Complete prompt string for the model
        """
        jewelry_config = JEWELRY_TYPES.get(jewelry_type.lower(), JEWELRY_TYPES["necklace"])
        base_prompt = jewelry_config["prompt_prefix"]

        if style is None:
            style = jewelry_config["default_style"]

        # Build detailed prompt
        prompt_parts = [base_prompt]

        # Add metal type
        prompt_parts.append(f"made of {metal_type}")

        # Add stones if specified
        if stones and stones.lower() != "none":
            prompt_parts.append(f"with {stones} stones")

        # Add style
        prompt_parts.append(f"{style} style")

        # Add quality descriptors
        prompt_parts.append("high quality, detailed, professional jewelry photography")

        # Add custom prompt if provided
        if custom_prompt:
            prompt_parts.append(custom_prompt)

        return ", ".join(prompt_parts)

    def _image_to_base64(self, image: Union[Image.Image, np.ndarray]) -> str:
        """Convert image to base64 string for API calls."""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _download_image(self, url: str) -> Optional[Image.Image]:
        """Download image from URL."""
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            return Image.open(io.BytesIO(response.content))
        except Exception as e:
            logger.error(f"Failed to download image from {url}: {e}")
            return None

    def generate_jewelry(self, person_image: Union[Image.Image, np.ndarray],
                        jewelry_type: str,
                        metal_type: str = "gold",
                        stones: str = "none",
                        style: str = None,
                        custom_prompt: str = None,
                        num_outputs: int = 1,
                        guidance_scale: float = 7.5,
                        num_inference_steps: int = 30) -> Tuple[Optional[Image.Image], str]:
        """
        Generate jewelry on a person image using the trained Replicate model.

        Args:
            person_image: Input person image (PIL Image or numpy array)
            jewelry_type: Type of jewelry to generate
            metal_type: Metal type for the jewelry
            stones: Stone type for embellishments
            style: Style variant
            custom_prompt: Additional prompt text
            num_outputs: Number of images to generate
            guidance_scale: Guidance scale for generation
            num_inference_steps: Number of inference steps

        Returns:
            Tuple of (generated image, status message)
        """
        if not REPLICATE_AVAILABLE:
            return None, "Error: Replicate package not installed. Install with: pip install replicate"

        # Check for API token
        api_token = os.environ.get("REPLICATE_API_TOKEN")
        if not api_token:
            return None, "Error: REPLICATE_API_TOKEN environment variable not set"

        try:
            # Convert image to PIL if needed
            if isinstance(person_image, np.ndarray):
                person_image = Image.fromarray(cv2.cvtColor(person_image, cv2.COLOR_BGR2RGB))

            # Build prompt
            prompt = self._build_prompt(
                jewelry_type=jewelry_type,
                metal_type=metal_type,
                stones=stones,
                style=style,
                custom_prompt=custom_prompt
            )

            logger.info(f"Generating jewelry with prompt: {prompt}")

            # Convert image to base64 for API
            image_base64 = self._image_to_base64(person_image)
            image_uri = f"data:image/png;base64,{image_base64}"

            # Run the model
            output = replicate.run(
                self.model_endpoint,
                input={
                    "image": image_uri,
                    "prompt": prompt,
                    "num_outputs": num_outputs,
                    "guidance_scale": guidance_scale,
                    "num_inference_steps": num_inference_steps
                }
            )

            # Process output
            if output:
                # Output is typically a list of URLs
                if isinstance(output, list) and len(output) > 0:
                    result_url = output[0]
                    if isinstance(result_url, str):
                        result_image = self._download_image(result_url)
                        if result_image:
                            return result_image, f"Success: Generated {jewelry_type} with {metal_type} and {stones}"
                elif hasattr(output, 'read'):
                    # FileOutput object
                    result_image = Image.open(output)
                    return result_image, f"Success: Generated {jewelry_type} with {metal_type} and {stones}"

            return None, "Error: No output received from model"

        except replicate.exceptions.ReplicateError as e:
            logger.error(f"Replicate API error: {e}")
            return None, f"Error: Replicate API error - {str(e)}"
        except Exception as e:
            logger.error(f"Jewelry generation error: {e}")
            import traceback
            traceback.print_exc()
            return None, f"Error: {str(e)}"

    def jewelry_tryon(self, person_image: Union[Image.Image, np.ndarray],
                      jewelry_prompt: str,
                      jewelry_type: str = "necklace",
                      metal_type: str = "gold",
                      stones: str = "none",
                      style: str = None,
                      opacity: float = 1.0) -> Tuple[Optional[Image.Image], str, Dict]:
        """
        Complete jewelry try-on pipeline: generate jewelry and overlay on person.

        This function:
        1. Detects pose landmarks using cvzone/MediaPipe
        2. Generates jewelry using the trained Replicate model
        3. Overlays the generated jewelry on the person with proper positioning

        Args:
            person_image: Input person image
            jewelry_prompt: Text prompt for jewelry generation
            jewelry_type: Type of jewelry
            metal_type: Metal type
            stones: Stone type
            style: Style variant
            opacity: Overlay opacity (0.0 to 1.0)

        Returns:
            Tuple of (result image, status message, landmark data)
        """
        try:
            # Convert to numpy for pose detection
            if isinstance(person_image, Image.Image):
                cv2_image = cv2.cvtColor(np.array(person_image), cv2.COLOR_RGB2BGR)
            else:
                cv2_image = person_image.copy()

            # Detect landmarks
            landmarks = self.pose_engine.detect_landmarks(cv2_image)

            if not landmarks["detected"]:
                logger.warning("No landmarks detected, using fallback positioning")

            # Generate jewelry
            result_image, gen_message = self.generate_jewelry(
                person_image=person_image,
                jewelry_type=jewelry_type,
                metal_type=metal_type,
                stones=stones,
                style=style,
                custom_prompt=jewelry_prompt
            )

            if result_image is None:
                # If generation fails, return original with error
                if isinstance(person_image, np.ndarray):
                    person_image = Image.fromarray(cv2.cvtColor(person_image, cv2.COLOR_BGR2RGB))
                return person_image, gen_message, landmarks

            # The Replicate model should return the person image with jewelry already applied
            # If the model returns just jewelry, we need to overlay it

            # Check if result is significantly different size (indicating just jewelry)
            person_size = person_image.size if isinstance(person_image, Image.Image) else (cv2_image.shape[1], cv2_image.shape[0])
            result_size = result_image.size

            size_ratio = (result_size[0] * result_size[1]) / (person_size[0] * person_size[1])

            if size_ratio < 0.3:  # Result is much smaller, likely just jewelry
                # Overlay the jewelry using detected landmarks
                logger.info("Result appears to be jewelry only, performing overlay")
                result_image, overlay_message = self._overlay_with_landmarks(
                    person_image, result_image, jewelry_type, landmarks, opacity
                )
                return result_image, f"{gen_message}. {overlay_message}", landmarks

            # Result is full image with jewelry applied
            return result_image, gen_message, landmarks

        except Exception as e:
            logger.error(f"Jewelry try-on error: {e}")
            import traceback
            traceback.print_exc()
            if isinstance(person_image, np.ndarray):
                person_image = Image.fromarray(cv2.cvtColor(person_image, cv2.COLOR_BGR2RGB))
            return person_image, f"Error: {str(e)}", {}

    def _overlay_with_landmarks(self, person_image: Union[Image.Image, np.ndarray],
                                jewelry_image: Image.Image,
                                jewelry_type: str,
                                landmarks: Dict,
                                opacity: float) -> Tuple[Image.Image, str]:
        """
        Overlay jewelry on person using detected landmarks.

        Args:
            person_image: Person image
            jewelry_image: Generated jewelry image
            jewelry_type: Type of jewelry for positioning
            landmarks: Detected landmark positions
            opacity: Overlay opacity

        Returns:
            Tuple of (result image, status message)
        """
        jewelry_type = jewelry_type.lower()

        # Use the overlay engine for actual positioning
        if jewelry_type == "necklace":
            return self.overlay_engine.apply_necklace(person_image, jewelry_image, opacity)
        elif jewelry_type in ["earrings", "earring"]:
            return self.overlay_engine.apply_earrings(person_image, jewelry_image, opacity)
        elif jewelry_type in ["maang_tikka", "maangtikka", "tikka"]:
            return self.overlay_engine.apply_maang_tikka(person_image, jewelry_image, opacity)
        elif jewelry_type in ["nose_ring", "nosering", "nath"]:
            return self.overlay_engine.apply_nose_ring(person_image, jewelry_image)
        elif jewelry_type in ["bangles", "bangle", "bracelet"]:
            return self._apply_bangles(person_image, jewelry_image, landmarks, opacity)
        elif jewelry_type in ["rings", "ring"]:
            return self._apply_ring(person_image, jewelry_image, landmarks, opacity)
        else:
            # Default to necklace positioning
            return self.overlay_engine.apply_necklace(person_image, jewelry_image, opacity)

    def _apply_bangles(self, person_image: Union[Image.Image, np.ndarray],
                       jewelry_image: Image.Image,
                       landmarks: Dict,
                       opacity: float) -> Tuple[Image.Image, str]:
        """Apply bangles at wrist position."""
        try:
            if isinstance(person_image, np.ndarray):
                cv2_person = person_image if len(person_image.shape) == 3 and person_image.shape[2] == 3 else cv2.cvtColor(person_image, cv2.COLOR_RGB2BGR)
            else:
                cv2_person = cv2.cvtColor(np.array(person_image), cv2.COLOR_RGB2BGR)

            cv2_jewelry = cv2.cvtColor(np.array(jewelry_image.convert('RGBA')), cv2.COLOR_RGBA2BGRA)

            img_h, img_w = cv2_person.shape[:2]

            # Get wrist position
            wrist_pos = landmarks.get("left_wrist") or landmarks.get("right_wrist")

            if wrist_pos is None:
                # Fallback: estimate wrist position
                wrist_pos = (int(img_w * 0.25), int(img_h * 0.7))
                logger.warning("No wrist detected, using fallback position")

            # Size bangles appropriately
            bangle_width = int(img_w * 0.15)
            bangle_height = int(bangle_width * 0.8)

            resized = self.overlay_engine._resize_jewelry(cv2_jewelry, bangle_width, bangle_height)
            result = self.overlay_engine._overlay_png(cv2_person, resized, wrist_pos, opacity)

            return self.overlay_engine._cv2_to_pil(result), f"Success: Bangles applied at wrist position {wrist_pos}"

        except Exception as e:
            logger.error(f"Bangles application error: {e}")
            if isinstance(person_image, np.ndarray):
                person_image = Image.fromarray(cv2.cvtColor(person_image, cv2.COLOR_BGR2RGB))
            return person_image, f"Error: {str(e)}"

    def _apply_ring(self, person_image: Union[Image.Image, np.ndarray],
                    jewelry_image: Image.Image,
                    landmarks: Dict,
                    opacity: float) -> Tuple[Image.Image, str]:
        """Apply ring at hand/finger position."""
        try:
            if isinstance(person_image, np.ndarray):
                cv2_person = person_image if len(person_image.shape) == 3 and person_image.shape[2] == 3 else cv2.cvtColor(person_image, cv2.COLOR_RGB2BGR)
            else:
                cv2_person = cv2.cvtColor(np.array(person_image), cv2.COLOR_RGB2BGR)

            cv2_jewelry = cv2.cvtColor(np.array(jewelry_image.convert('RGBA')), cv2.COLOR_RGBA2BGRA)

            img_h, img_w = cv2_person.shape[:2]

            # Get hand position (estimate from wrist)
            wrist_pos = landmarks.get("left_wrist") or landmarks.get("right_wrist")

            if wrist_pos:
                # Ring finger is slightly below and to the side of wrist
                ring_pos = (wrist_pos[0] + int(img_w * 0.05), wrist_pos[1] + int(img_h * 0.1))
            else:
                # Fallback position
                ring_pos = (int(img_w * 0.3), int(img_h * 0.75))
                logger.warning("No hand detected, using fallback position")

            # Size ring appropriately (small)
            ring_size = int(img_w * 0.05)

            resized = self.overlay_engine._resize_jewelry(cv2_jewelry, ring_size, ring_size)
            result = self.overlay_engine._overlay_png(cv2_person, resized, ring_pos, opacity)

            return self.overlay_engine._cv2_to_pil(result), f"Success: Ring applied at position {ring_pos}"

        except Exception as e:
            logger.error(f"Ring application error: {e}")
            if isinstance(person_image, np.ndarray):
                person_image = Image.fromarray(cv2.cvtColor(person_image, cv2.COLOR_BGR2RGB))
            return person_image, f"Error: {str(e)}"


# Global jewelry generation engine instance
_generation_engine: Optional[JewelryGenerationEngine] = None


def get_generation_engine() -> JewelryGenerationEngine:
    """Get or create the global jewelry generation engine instance."""
    global _generation_engine
    if _generation_engine is None:
        _generation_engine = JewelryGenerationEngine()
    return _generation_engine


class JewelryEngine:
    """
    Simplified Jewelry Engine using ONLY OpenCV face detection.

    NO MEDIAPIPE, NO CVZONE POSE - Works everywhere including ZeroGPU.
    """

    def __init__(self, scale_factor: float = 1.1, min_neighbors: int = 4):
        """
        Initialize the Jewelry Engine.

        Args:
            scale_factor: Scale factor for face cascade detection
            min_neighbors: Min neighbors for face cascade detection
        """
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        _init_cascades()

    def _pil_to_cv2(self, pil_image: Image.Image) -> np.ndarray:
        """Convert PIL Image to OpenCV format (BGR)."""
        rgb_array = np.array(pil_image.convert('RGB'))
        return cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)

    def _cv2_to_pil(self, cv2_image: np.ndarray) -> Image.Image:
        """Convert OpenCV image (BGR) to PIL Image."""
        if len(cv2_image.shape) == 3 and cv2_image.shape[2] == 4:
            rgba = cv2.cvtColor(cv2_image, cv2.COLOR_BGRA2RGBA)
            return Image.fromarray(rgba)
        else:
            rgb_array = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
            return Image.fromarray(rgb_array)

    def _pil_to_cv2_rgba(self, pil_image: Image.Image) -> np.ndarray:
        """Convert PIL RGBA to OpenCV BGRA."""
        rgba_array = np.array(pil_image.convert('RGBA'))
        return cv2.cvtColor(rgba_array, cv2.COLOR_RGBA2BGRA)

    def detect_face(self, cv2_image: np.ndarray) -> Optional[dict]:
        """
        Detect face and estimate facial landmarks geometrically.

        Args:
            cv2_image: OpenCV image (BGR)

        Returns:
            Dictionary with face bounds and estimated landmark positions, or None
        """
        if FACE_CASCADE is None:
            logger.error("Face cascade not loaded")
            return None

        img_h, img_w = cv2_image.shape[:2]
        logger.debug(f"detect_face: Image dimensions {img_w}x{img_h}")

        try:
            gray = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2GRAY)

            # Apply histogram equalization for better detection in varied lighting
            gray = cv2.equalizeHist(gray)

            # Calculate dynamic minSize based on image dimensions
            # Face should be at least 10% of image width/height
            min_face_size = max(30, min(img_w, img_h) // 10)
            logger.debug(f"detect_face: Using minSize={min_face_size}")

            # Try multiple detection strategies with increasingly loose parameters
            detection_configs = [
                # Config 1: Standard parameters
                {'scaleFactor': self.scale_factor, 'minNeighbors': self.min_neighbors, 'minSize': (min_face_size, min_face_size)},
                # Config 2: Looser - lower minNeighbors
                {'scaleFactor': 1.1, 'minNeighbors': 3, 'minSize': (min_face_size, min_face_size)},
                # Config 3: Very loose - even smaller minSize
                {'scaleFactor': 1.05, 'minNeighbors': 2, 'minSize': (max(20, min_face_size // 2), max(20, min_face_size // 2))},
                # Config 4: Most aggressive - smallest faces
                {'scaleFactor': 1.03, 'minNeighbors': 2, 'minSize': (20, 20)},
            ]

            faces = []
            for i, config in enumerate(detection_configs):
                faces = FACE_CASCADE.detectMultiScale(
                    gray,
                    scaleFactor=config['scaleFactor'],
                    minNeighbors=config['minNeighbors'],
                    minSize=config['minSize'],
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
                if len(faces) > 0:
                    logger.debug(f"detect_face: Found {len(faces)} face(s) with config {i+1}")
                    break

            if len(faces) == 0:
                logger.warning("detect_face: No faces detected after all attempts")
                return None

            # Take the largest face
            face = max(faces, key=lambda f: f[2] * f[3])
            x, y, w, h = face

            logger.info(f"detect_face: Face detected at ({x}, {y}) size {w}x{h}")

            # Calculate landmarks based on face geometry
            # These ratios are based on standard facial proportions
            landmarks = {
                'face_bounds': (x, y, w, h),
                'face_center': (x + w // 2, y + h // 2),
                'face_width': w,
                'face_height': h,
                # Chin is at bottom of face box
                'chin': (x + w // 2, y + h),
                # Forehead/hairline is at very top of face box (maang tikka position)
                'forehead': (x + w // 2, y + int(h * 0.05)),
                # Earlobes at sides, approximately 60% down face height (below ears)
                'left_earlobe': (x - int(w * 0.08), y + int(h * 0.60)),
                'right_earlobe': (x + w + int(w * 0.08), y + int(h * 0.60)),
                # Nose positions - adjusted for better accuracy
                'nose_tip': (x + w // 2, y + int(h * 0.65)),
                'left_nostril': (x + w // 2 - int(w * 0.08), y + int(h * 0.72)),
                'right_nostril': (x + w // 2 + int(w * 0.08), y + int(h * 0.72)),
                'septum': (x + w // 2, y + int(h * 0.75)),
                # Eye positions (for angle calculation) - at 35% down face
                'left_eye': (x + int(w * 0.30), y + int(h * 0.35)),
                'right_eye': (x + int(w * 0.70), y + int(h * 0.35)),
            }

            # Calculate face angle from eye positions
            delta_x = landmarks['right_eye'][0] - landmarks['left_eye'][0]
            delta_y = landmarks['right_eye'][1] - landmarks['left_eye'][1]
            landmarks['face_angle'] = math.degrees(math.atan2(delta_y, delta_x))

            logger.debug(f"detect_face: Landmarks calculated - chin={landmarks['chin']}, forehead={landmarks['forehead']}")

            return landmarks

        except Exception as e:
            logger.error(f"Face detection error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _resize_jewelry(self, jewelry: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
        """Resize jewelry image maintaining aspect ratio."""
        h, w = jewelry.shape[:2]
        if w == 0 or h == 0:
            return jewelry

        aspect_ratio = w / h

        if target_width / max(target_height, 1) > aspect_ratio:
            new_height = target_height
            new_width = int(new_height * aspect_ratio)
        else:
            new_width = target_width
            new_height = int(new_width / aspect_ratio)

        new_width = max(new_width, 10)
        new_height = max(new_height, 10)

        return cv2.resize(jewelry, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)

    def _rotate_jewelry(self, jewelry: np.ndarray, angle: float) -> np.ndarray:
        """Rotate jewelry image around center."""
        if abs(angle) < 1:  # Skip rotation for small angles
            return jewelry

        h, w = jewelry.shape[:2]
        center = (w // 2, h // 2)

        rotation_matrix = cv2.getRotationMatrix2D(center, -angle, 1.0)

        cos = abs(rotation_matrix[0, 0])
        sin = abs(rotation_matrix[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))

        rotation_matrix[0, 2] += (new_w / 2) - center[0]
        rotation_matrix[1, 2] += (new_h / 2) - center[1]

        return cv2.warpAffine(jewelry, rotation_matrix, (new_w, new_h),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=(0, 0, 0, 0))

    def _overlay_png(self, background: np.ndarray, overlay: np.ndarray,
                     position: Tuple[int, int], opacity: float = 1.0) -> np.ndarray:
        """
        Overlay PNG with alpha blending.

        Args:
            background: Background image (BGR)
            overlay: Overlay image with alpha channel (BGRA)
            position: (x, y) center position for overlay
            opacity: Opacity multiplier (0.0 to 1.0)
        """
        # Ensure background is BGR
        if len(background.shape) == 2:
            background = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)
        elif background.shape[2] == 4:
            background = cv2.cvtColor(background, cv2.COLOR_BGRA2BGR)

        # Ensure overlay has alpha channel
        if len(overlay.shape) == 2:
            overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGRA)
        elif overlay.shape[2] == 3:
            overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2BGRA)

        # Apply opacity
        if opacity < 1.0:
            overlay = overlay.copy()
            overlay[:, :, 3] = (overlay[:, :, 3] * opacity).astype(np.uint8)

        oh, ow = overlay.shape[:2]
        bh, bw = background.shape[:2]

        # Calculate top-left corner from center position
        x = position[0] - ow // 2
        y = position[1] - oh // 2

        # Calculate overlap region
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(bw, x + ow), min(bh, y + oh)

        ox1, oy1 = max(0, -x), max(0, -y)
        ox2, oy2 = ox1 + (x2 - x1), oy1 + (y2 - y1)

        if x2 <= x1 or y2 <= y1:
            return background

        # Extract regions
        roi = background[y1:y2, x1:x2].copy()
        overlay_roi = overlay[oy1:oy2, ox1:ox2]

        # Alpha blending
        alpha = overlay_roi[:, :, 3:4].astype(np.float32) / 255.0
        overlay_rgb = overlay_roi[:, :, :3].astype(np.float32)
        roi_float = roi.astype(np.float32)

        blended = overlay_rgb * alpha + roi_float * (1 - alpha)

        result = background.copy()
        result[y1:y2, x1:x2] = blended.astype(np.uint8)

        return result

    def apply_necklace(self, person_image: Image.Image,
                       necklace_image: Image.Image,
                       opacity: float = 1.0) -> Tuple[Image.Image, str]:
        """
        Apply necklace using face-based positioning.

        Places necklace below chin at neck level based on face detection.

        Args:
            person_image: PIL Image of the person
            necklace_image: PIL Image of the necklace
            opacity: Opacity of the necklace overlay

        Returns:
            Tuple of (result image, status message)
        """
        try:
            cv2_person = self._pil_to_cv2(person_image)
            cv2_necklace = self._pil_to_cv2_rgba(necklace_image)

            img_h, img_w = cv2_person.shape[:2]
            logger.info(f"apply_necklace: Image size {img_w}x{img_h}")

            # Try to detect face for positioning
            face = self.detect_face(cv2_person)

            if face is not None:
                # Face detected - calculate necklace position based on face geometry
                chin_x, chin_y = face['chin']
                face_width = face['face_width']
                face_height = face['face_height']
                face_angle = face.get('face_angle', 0)

                # Position necklace at neck level: 15-25% of face height below chin
                # This places it in the neck/collarbone area
                necklace_y = chin_y + int(face_height * 0.25)
                necklace_x = chin_x

                # Ensure necklace doesn't go below image bounds
                necklace_y = min(necklace_y, img_h - 20)

                # Necklace width = 1.8x face width (to span shoulders)
                necklace_width = int(face_width * 1.8)
                necklace_height = int(necklace_width * 0.5)

                logger.info(f"apply_necklace: Face at chin=({chin_x}, {chin_y}), "
                           f"placing necklace at ({necklace_x}, {necklace_y}), "
                           f"size {necklace_width}x{necklace_height}")
            else:
                # NO FACE DETECTED - Use intelligent fallback positioning
                logger.warning("apply_necklace: No face detected - using fallback positioning")

                # For typical portrait/selfie images, estimate neck position
                # Assume face is in upper third of image
                necklace_x = img_w // 2
                # Place necklace at approximately 35-40% down the image
                # This is typically where neck/collarbone would be in a portrait
                necklace_y = int(img_h * 0.38)

                # Size relative to image
                necklace_width = int(img_w * 0.45)
                necklace_height = int(necklace_width * 0.5)
                face_angle = 0

                logger.info(f"apply_necklace: Fallback position ({necklace_x}, {necklace_y}), "
                           f"size {necklace_width}x{necklace_height}")

            # Resize necklace maintaining aspect ratio
            resized = self._resize_jewelry(cv2_necklace, necklace_width, necklace_height)

            # Rotate if face is tilted
            rotated = self._rotate_jewelry(resized, face_angle)

            # Overlay necklace on person at calculated position
            result = self._overlay_png(cv2_person, rotated, (necklace_x, necklace_y), opacity)

            result_pil = self._cv2_to_pil(result)

            if face is not None:
                return result_pil, f"Success: Necklace applied at neck position ({necklace_x}, {necklace_y})"
            else:
                return result_pil, f"Success: Necklace applied with fallback positioning at ({necklace_x}, {necklace_y})"

        except Exception as e:
            logger.error(f"Necklace application error: {str(e)}")
            import traceback
            traceback.print_exc()
            return person_image, f"Error: {str(e)}"

    def apply_earrings(self, person_image: Image.Image,
                       earring_image: Image.Image,
                       opacity: float = 1.0) -> Tuple[Image.Image, str]:
        """
        Apply earrings using OpenCV face detection.

        Positions earrings at estimated earlobe locations based on face geometry.

        Args:
            person_image: PIL Image of the person
            earring_image: PIL Image of a single earring
            opacity: Opacity of the earring overlay

        Returns:
            Tuple of (result image, status message)
        """
        try:
            cv2_person = self._pil_to_cv2(person_image)
            cv2_earring = self._pil_to_cv2_rgba(earring_image)

            img_h, img_w = cv2_person.shape[:2]
            logger.info(f"apply_earrings: Image size {img_w}x{img_h}")

            face = self.detect_face(cv2_person)

            if face is None:
                # Fallback positioning for earrings
                logger.warning("apply_earrings: No face detected - using fallback positioning")
                # Estimate ear positions for typical portrait
                # Ears are typically at 25-30% from edges, 28-32% down from top
                left_pos = (int(img_w * 0.22), int(img_h * 0.30))
                right_pos = (int(img_w * 0.78), int(img_h * 0.30))
                earring_size = int(img_w * 0.08)
                face_angle = 0

                logger.info(f"apply_earrings: Fallback positions - left={left_pos}, right={right_pos}")
            else:
                face_width = face['face_width']
                face_height = face['face_height']
                face_angle = face.get('face_angle', 0)
                left_pos = face['left_earlobe']
                right_pos = face['right_earlobe']

                # Earring size proportional to face - about 20% of face width
                earring_size = int(face_width * 0.20)

                logger.info(f"apply_earrings: Face detected - left_ear={left_pos}, "
                           f"right_ear={right_pos}, earring_size={earring_size}")

            # Ensure minimum earring size
            earring_size = max(earring_size, 15)

            # Resize earring maintaining aspect ratio
            resized_earring = self._resize_jewelry(cv2_earring, earring_size, int(earring_size * 1.5))
            rotated_left = self._rotate_jewelry(resized_earring, face_angle)

            # Mirror earring for right ear
            flipped_earring = cv2.flip(resized_earring, 1)
            rotated_right = self._rotate_jewelry(flipped_earring, face_angle)

            # Apply both earrings
            result = self._overlay_png(cv2_person, rotated_left, left_pos, opacity)
            result = self._overlay_png(result, rotated_right, right_pos, opacity)

            result_pil = self._cv2_to_pil(result)

            if face is not None:
                return result_pil, f"Success: Earrings applied at positions L={left_pos}, R={right_pos}"
            else:
                return result_pil, f"Success: Earrings applied with fallback at L={left_pos}, R={right_pos}"

        except Exception as e:
            logger.error(f"Earring application error: {str(e)}")
            import traceback
            traceback.print_exc()
            return person_image, f"Error: {str(e)}"

    def apply_maang_tikka(self, person_image: Image.Image,
                          tikka_image: Image.Image,
                          opacity: float = 1.0) -> Tuple[Image.Image, str]:
        """
        Apply Maang Tikka (Indian forehead jewelry) using OpenCV face detection.

        Positions the tikka at the center of the forehead/hairline area.

        Args:
            person_image: PIL Image of the person
            tikka_image: PIL Image of the maang tikka
            opacity: Opacity of the overlay

        Returns:
            Tuple of (result image, status message)
        """
        try:
            cv2_person = self._pil_to_cv2(person_image)
            cv2_tikka = self._pil_to_cv2_rgba(tikka_image)

            img_h, img_w = cv2_person.shape[:2]
            logger.info(f"apply_maang_tikka: Image size {img_w}x{img_h}")

            face = self.detect_face(cv2_person)

            if face is None:
                # Fallback positioning for maang tikka
                logger.warning("apply_maang_tikka: No face detected - using fallback positioning")
                # Position at top-center of image where forehead typically is
                forehead_pos = (img_w // 2, int(img_h * 0.15))
                tikka_width = int(img_w * 0.06)
                tikka_height = int(tikka_width * 2.5)
                face_angle = 0

                logger.info(f"apply_maang_tikka: Fallback position at {forehead_pos}")
            else:
                face_width = face['face_width']
                face_angle = face.get('face_angle', 0)
                forehead_pos = face['forehead']

                # Tikka size proportional to face
                tikka_width = int(face_width * 0.12)
                tikka_height = int(tikka_width * 2.5)

                logger.info(f"apply_maang_tikka: Face detected - forehead at {forehead_pos}, "
                           f"tikka size {tikka_width}x{tikka_height}")

            # Ensure minimum size
            tikka_width = max(tikka_width, 12)
            tikka_height = max(tikka_height, 25)

            resized_tikka = self._resize_jewelry(cv2_tikka, tikka_width, tikka_height)
            rotated_tikka = self._rotate_jewelry(resized_tikka, face_angle)

            result = self._overlay_png(cv2_person, rotated_tikka, forehead_pos, opacity)

            result_pil = self._cv2_to_pil(result)

            if face is not None:
                return result_pil, f"Success: Maang Tikka applied at forehead position {forehead_pos}"
            else:
                return result_pil, f"Success: Maang Tikka applied with fallback at {forehead_pos}"

        except Exception as e:
            logger.error(f"Maang Tikka application error: {str(e)}")
            import traceback
            traceback.print_exc()
            return person_image, f"Error: {str(e)}"

    def apply_nose_ring(self, person_image: Image.Image,
                        nose_ring_image: Image.Image,
                        side: str = "left",
                        ring_style: str = "stud",
                        opacity: float = 1.0) -> Tuple[Image.Image, str]:
        """
        Apply nose ring using OpenCV face detection.

        Positions the nose ring at the appropriate nostril or septum location.

        Args:
            person_image: PIL Image of the person
            nose_ring_image: PIL Image of the nose ring
            side: Which side - "left", "right", or "septum"
            ring_style: Style - "stud", "hoop", or "nath"
            opacity: Opacity of the overlay

        Returns:
            Tuple of (result image, status message)
        """
        try:
            cv2_person = self._pil_to_cv2(person_image)
            cv2_ring = self._pil_to_cv2_rgba(nose_ring_image)

            img_h, img_w = cv2_person.shape[:2]
            logger.info(f"apply_nose_ring: Image size {img_w}x{img_h}, side={side}, style={ring_style}")

            face = self.detect_face(cv2_person)

            side_lower = side.lower()

            if face is None:
                # Fallback positioning for nose ring
                logger.warning("apply_nose_ring: No face detected - using fallback positioning")
                center_x = img_w // 2
                # Nose is typically around 28-32% down in a portrait
                center_y = int(img_h * 0.30)

                # Offset for left/right nostril
                nostril_offset = int(img_w * 0.025)
                if side_lower == "left":
                    nose_pos = (center_x - nostril_offset, center_y)
                elif side_lower == "right":
                    nose_pos = (center_x + nostril_offset, center_y)
                else:  # septum
                    nose_pos = (center_x, int(center_y * 1.05))

                ring_size = int(img_w * 0.04)
                face_angle = 0

                logger.info(f"apply_nose_ring: Fallback position at {nose_pos}")
            else:
                face_width = face['face_width']
                face_angle = face.get('face_angle', 0)

                if side_lower == "septum":
                    nose_pos = face['septum']
                elif side_lower == "right":
                    nose_pos = face['right_nostril']
                else:
                    nose_pos = face['left_nostril']

                # Determine ring size based on style
                if ring_style.lower() == "nath":
                    # Nath is larger, ornate nose ring
                    ring_size = int(face_width * 0.20)
                elif ring_style.lower() == "hoop":
                    ring_size = int(face_width * 0.10)
                else:  # stud
                    ring_size = int(face_width * 0.07)

                logger.info(f"apply_nose_ring: Face detected - nose position at {nose_pos}, "
                           f"ring_size={ring_size}")

            # Ensure minimum size
            ring_size = max(ring_size, 8)

            # For nath style, make it taller than wide
            if ring_style.lower() == "nath":
                resized_ring = self._resize_jewelry(cv2_ring, ring_size, int(ring_size * 1.5))
            else:
                resized_ring = self._resize_jewelry(cv2_ring, ring_size, ring_size)

            rotated_ring = self._rotate_jewelry(resized_ring, face_angle)

            # Mirror for right side
            if side_lower == "right":
                rotated_ring = cv2.flip(rotated_ring, 1)

            result = self._overlay_png(cv2_person, rotated_ring, nose_pos, opacity)

            result_pil = self._cv2_to_pil(result)

            if face is not None:
                return result_pil, f"Success: Nose ring ({ring_style}) applied at {nose_pos}"
            else:
                return result_pil, f"Success: Nose ring applied with fallback at {nose_pos}"

        except Exception as e:
            logger.error(f"Nose ring application error: {str(e)}")
            import traceback
            traceback.print_exc()
            return person_image, f"Error: {str(e)}"

    def close(self):
        """Release resources."""
        pass


# Global engine instance
_engine_instance: Optional[JewelryEngine] = None


def get_engine() -> JewelryEngine:
    """Get or create the global engine instance."""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = JewelryEngine()
    return _engine_instance


def remove_jewelry_background(jewelry_image: Union[Image.Image, np.ndarray]) -> Image.Image:
    """
    Remove background from jewelry image.

    NOTE: rembg is disabled for CPU performance. Returns RGBA without removal.

    Args:
        jewelry_image: PIL Image or numpy array of the jewelry

    Returns:
        PIL Image with RGBA mode
    """
    try:
        if isinstance(jewelry_image, np.ndarray):
            jewelry_image = Image.fromarray(cv2.cvtColor(jewelry_image, cv2.COLOR_BGR2RGB))
        return jewelry_image.convert('RGBA')
    except Exception as e:
        logger.error(f"Background conversion error: {str(e)}")
        if isinstance(jewelry_image, np.ndarray):
            jewelry_image = Image.fromarray(cv2.cvtColor(jewelry_image, cv2.COLOR_BGR2RGB))
        return jewelry_image.convert('RGBA')


def apply_jewelry(person_image: Union[Image.Image, np.ndarray, str],
                  jewelry_image: Union[Image.Image, np.ndarray, str],
                  jewelry_type: str,
                  opacity: float = 1.0,
                  **kwargs) -> Tuple[Image.Image, str]:
    """
    Main wrapper function for jewelry virtual try-on.

    This is the primary interface for the Gradio app.

    Args:
        person_image: Person photo (PIL Image, numpy array, or file path)
        jewelry_image: Jewelry image (PIL Image, numpy array, or file path)
        jewelry_type: Type - "necklace", "earrings", "maang_tikka", "nose_ring"
        opacity: Opacity of the jewelry overlay (0.0 to 1.0)
        **kwargs: Additional arguments (e.g., 'side' for nose ring)

    Returns:
        Tuple of (result PIL Image, status message)
    """
    try:
        # Convert person image to PIL
        if isinstance(person_image, str):
            person_image = Image.open(person_image)
        elif isinstance(person_image, np.ndarray):
            person_image = Image.fromarray(person_image)

        # Convert jewelry image to PIL
        if isinstance(jewelry_image, str):
            jewelry_image = Image.open(jewelry_image)
        elif isinstance(jewelry_image, np.ndarray):
            jewelry_image = Image.fromarray(jewelry_image)

        if person_image is None:
            return None, "Error: Please upload a person photo."
        if jewelry_image is None:
            return None, "Error: Please upload a jewelry image."

        person_image = person_image.convert('RGB')
        jewelry_image = jewelry_image.convert('RGBA')

    except Exception as e:
        return None, f"Error: Could not load images - {str(e)}"

    engine = get_engine()

    jewelry_type = jewelry_type.lower().replace(" ", "_").replace("-", "_")

    if jewelry_type == "necklace":
        return engine.apply_necklace(person_image, jewelry_image, opacity)

    elif jewelry_type in ["earrings", "earring"]:
        return engine.apply_earrings(person_image, jewelry_image, opacity)

    elif jewelry_type in ["maang_tikka", "maangtikka", "tikka", "maang tikka"]:
        return engine.apply_maang_tikka(person_image, jewelry_image, opacity)

    elif jewelry_type in ["nose_ring", "nosering", "nose ring", "nath"]:
        side = kwargs.get("side", "left")
        ring_style = kwargs.get("ring_style", "stud")
        return engine.apply_nose_ring(person_image, jewelry_image, side, ring_style, opacity)

    else:
        return person_image, f"Error: Unknown jewelry type '{jewelry_type}'. Supported: necklace, earrings, maang_tikka, nose_ring"


# ============================================================================
# API FUNCTION FOR JEWELRY TRY-ON (with AI generation)
# ============================================================================

def jewelry_tryon_api(person_image: Union[Image.Image, np.ndarray, str],
                      jewelry_prompt: str,
                      jewelry_type: str = "necklace",
                      metal_type: str = "gold",
                      stones: str = "none",
                      style: str = None,
                      opacity: float = 1.0) -> Dict[str, Any]:
    """
    API function for jewelry virtual try-on with AI generation.

    This is the main API endpoint function that:
    1. Accepts person_image and jewelry_prompt
    2. Generates jewelry using the trained Replicate model
    3. Uses pose detection (cvzone/MediaPipe) to detect neck/ear/wrist landmarks
    4. Overlays generated jewelry on person with proper scaling and positioning
    5. Returns the composite image with metadata

    Args:
        person_image: Person photo (PIL Image, numpy array, or file path)
        jewelry_prompt: Text prompt describing desired jewelry
        jewelry_type: Type of jewelry ("necklace", "earrings", "bangles", "rings", "maang_tikka", "nose_ring")
        metal_type: Metal type ("gold", "silver", "rose gold", "platinum", etc.)
        stones: Stone type ("diamond", "ruby", "emerald", "sapphire", "pearl", "kundan", "none")
        style: Style variant (depends on jewelry type)
        opacity: Overlay opacity (0.0 to 1.0)

    Returns:
        Dictionary with:
        - "success": bool - Whether the operation succeeded
        - "image": PIL.Image or None - Result image
        - "message": str - Status message
        - "landmarks": dict - Detected landmark positions
        - "prompt_used": str - Full prompt sent to model
    """
    result = {
        "success": False,
        "image": None,
        "message": "",
        "landmarks": {},
        "prompt_used": ""
    }

    try:
        # Convert person image to PIL
        if isinstance(person_image, str):
            if os.path.exists(person_image):
                person_image = Image.open(person_image)
            else:
                result["message"] = f"Error: File not found - {person_image}"
                return result
        elif isinstance(person_image, np.ndarray):
            person_image = Image.fromarray(person_image)

        if person_image is None:
            result["message"] = "Error: Please provide a person photo."
            return result

        # Ensure RGB mode
        person_image = person_image.convert('RGB')

        # Validate jewelry type
        jewelry_type = jewelry_type.lower().replace(" ", "_").replace("-", "_")
        if jewelry_type not in JEWELRY_TYPES:
            result["message"] = f"Error: Unknown jewelry type '{jewelry_type}'. Supported: {', '.join(JEWELRY_TYPES.keys())}"
            return result

        # Validate metal type
        if metal_type.lower() not in [m.lower() for m in METAL_TYPES]:
            logger.warning(f"Unknown metal type '{metal_type}', using as-is")

        # Validate stone type
        if stones.lower() not in [s.lower() for s in STONE_TYPES]:
            logger.warning(f"Unknown stone type '{stones}', using as-is")

        # Get style options for this jewelry type
        valid_styles = STYLE_OPTIONS.get(jewelry_type, [])
        if style and style.lower() not in [s.lower() for s in valid_styles]:
            logger.warning(f"Unknown style '{style}' for {jewelry_type}, using default")
            style = None

        # Get the generation engine
        gen_engine = get_generation_engine()

        # Build the full prompt
        full_prompt = gen_engine._build_prompt(
            jewelry_type=jewelry_type,
            metal_type=metal_type,
            stones=stones,
            style=style,
            custom_prompt=jewelry_prompt
        )
        result["prompt_used"] = full_prompt

        # Perform jewelry try-on
        result_image, message, landmarks = gen_engine.jewelry_tryon(
            person_image=person_image,
            jewelry_prompt=jewelry_prompt,
            jewelry_type=jewelry_type,
            metal_type=metal_type,
            stones=stones,
            style=style,
            opacity=opacity
        )

        result["image"] = result_image
        result["message"] = message
        result["landmarks"] = landmarks
        result["success"] = result_image is not None and "Error" not in message

        return result

    except Exception as e:
        logger.error(f"Jewelry try-on API error: {e}")
        import traceback
        traceback.print_exc()
        result["message"] = f"Error: {str(e)}"
        return result


def get_available_options() -> Dict[str, Any]:
    """
    Get all available options for the jewelry try-on API.

    Returns:
        Dictionary with available jewelry types, metal types, stones, and styles
    """
    return {
        "jewelry_types": list(JEWELRY_TYPES.keys()),
        "metal_types": METAL_TYPES,
        "stone_types": STONE_TYPES,
        "style_options": STYLE_OPTIONS,
        "replicate_available": REPLICATE_AVAILABLE,
        "cvzone_available": CVZONE_AVAILABLE,
        "mediapipe_available": MEDIAPIPE_AVAILABLE,
        "model_endpoint": REPLICATE_MODEL
    }


if __name__ == "__main__":
    print("=" * 60)
    print("Naari Studio - Jewelry Virtual Try-On Engine")
    print("AI-Powered Jewelry Generation + Overlay")
    print("=" * 60)

    # Show available backends
    print("\nAvailable Backends:")
    print(f"  - OpenCV Face Cascade: {FACE_CASCADE is not None}")
    print(f"  - cvzone PoseDetector: {CVZONE_AVAILABLE}")
    print(f"  - MediaPipe Face Mesh: {MEDIAPIPE_AVAILABLE}")
    print(f"  - Replicate API: {REPLICATE_AVAILABLE}")

    print(f"\nReplicate Model: {REPLICATE_MODEL}")

    print("\nSupported Jewelry Types:")
    for jtype, config in JEWELRY_TYPES.items():
        print(f"  - {jtype}: {config['prompt_prefix']}")

    print("\nMetal Types:", ", ".join(METAL_TYPES))
    print("Stone Types:", ", ".join(STONE_TYPES))

    print("\nStyle Options:")
    for jtype, styles in STYLE_OPTIONS.items():
        print(f"  - {jtype}: {', '.join(styles)}")

    print("=" * 60)

    # Quick self-test
    print("\nRunning self-test...")
    try:
        engine = JewelryEngine()

        # Create a test image (white background with a simple "face" circle)
        test_img = np.ones((400, 300, 3), dtype=np.uint8) * 255
        cv2.circle(test_img, (150, 150), 80, (200, 180, 160), -1)  # Face
        cv2.circle(test_img, (130, 130), 10, (50, 50, 50), -1)  # Left eye
        cv2.circle(test_img, (170, 130), 10, (50, 50, 50), -1)  # Right eye

        # Create a simple red jewelry overlay
        jewelry_img = np.zeros((50, 100, 4), dtype=np.uint8)
        jewelry_img[:, :, 2] = 255  # Red
        jewelry_img[:, :, 3] = 200  # Semi-transparent

        # Convert to PIL
        person_pil = Image.fromarray(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB))
        jewelry_pil = Image.fromarray(jewelry_img)

        # Test necklace application
        result, msg = engine.apply_necklace(person_pil, jewelry_pil)

        # Verify result is different from input
        result_arr = np.array(result)
        person_arr = np.array(person_pil)

        if np.array_equal(result_arr, person_arr):
            print("WARNING: Result is same as input - overlay may not be working!")
        else:
            print("SUCCESS: Overlay engine working - result differs from input")

        print(f"Overlay test result: {msg}")

        # Test pose detection
        print("\nTesting pose detection...")
        pose_engine = get_pose_engine()
        landmarks = pose_engine.detect_landmarks(cv2.cvtColor(test_img, cv2.COLOR_RGB2BGR))
        print(f"Landmarks detected: {landmarks['detected']}")
        if landmarks['detected']:
            print(f"  - Neck: {landmarks.get('neck')}")
            print(f"  - Ears: L={landmarks.get('left_ear')}, R={landmarks.get('right_ear')}")

        print("\nSelf-test complete!")

    except Exception as e:
        print(f"Self-test failed: {e}")
        import traceback
        traceback.print_exc()
