"""
Naari Studio - Jewelry Virtual Try-On Engine (SIMPLIFIED)

Uses ONLY OpenCV Haar Cascades for all jewelry positioning.
NO MEDIAPIPE, NO CVZONE POSE DETECTION - 100% ZeroGPU compatible.

Simplified approach:
- Necklace: Detect face, place necklace 20% below chin center
- Earrings: Face detection with geometric earlobe estimation
- Maang Tikka: Forehead center via face geometry
- Nose Ring: Nose landmark via face geometry
"""

import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Optional, Union
import logging
import math
import os

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

        try:
            gray = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2GRAY)

            # Try multiple detection parameters for robustness
            faces = FACE_CASCADE.detectMultiScale(
                gray,
                scaleFactor=self.scale_factor,
                minNeighbors=self.min_neighbors,
                minSize=(60, 60),
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            # If no faces found, try with looser parameters
            if len(faces) == 0:
                faces = FACE_CASCADE.detectMultiScale(
                    gray,
                    scaleFactor=1.05,
                    minNeighbors=3,
                    minSize=(40, 40),
                    flags=cv2.CASCADE_SCALE_IMAGE
                )

            if len(faces) == 0:
                return None

            # Take the largest face
            face = max(faces, key=lambda f: f[2] * f[3])
            x, y, w, h = face

            landmarks = {
                'face_bounds': (x, y, w, h),
                'face_center': (x + w // 2, y + h // 2),
                'face_width': w,
                'face_height': h,
                # Chin is at bottom of face box
                'chin': (x + w // 2, y + h),
                # Forehead is at top center
                'forehead': (x + w // 2, y + int(h * 0.12)),
                # Earlobes at sides, 50% down face
                'left_earlobe': (x - int(w * 0.05), y + int(h * 0.50)),
                'right_earlobe': (x + w + int(w * 0.05), y + int(h * 0.50)),
                # Nose positions
                'nose_tip': (x + w // 2, y + int(h * 0.60)),
                'left_nostril': (x + w // 2 - int(w * 0.10), y + int(h * 0.65)),
                'right_nostril': (x + w // 2 + int(w * 0.10), y + int(h * 0.65)),
                'septum': (x + w // 2, y + int(h * 0.70)),
                # Eye positions (for angle calculation)
                'left_eye': (x + int(w * 0.30), y + int(h * 0.35)),
                'right_eye': (x + int(w * 0.70), y + int(h * 0.35)),
            }

            # Calculate face angle from eye positions
            delta_x = landmarks['right_eye'][0] - landmarks['left_eye'][0]
            delta_y = landmarks['right_eye'][1] - landmarks['left_eye'][1]
            landmarks['face_angle'] = math.degrees(math.atan2(delta_y, delta_x))

            return landmarks

        except Exception as e:
            logger.error(f"Face detection error: {e}")
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
        Apply necklace using SIMPLE face-based positioning.

        Places necklace 20% below chin center - NO POSE DETECTION NEEDED.

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

            h, w = cv2_person.shape[:2]

            # Try to detect face for positioning
            face = self.detect_face(cv2_person)

            if face is not None:
                # Face detected - use chin position
                chin_x, chin_y = face['chin']
                face_width = face['face_width']
                face_angle = face.get('face_angle', 0)

                # Position necklace 20% of face height below chin
                necklace_y = chin_y + int(face['face_height'] * 0.20)
                necklace_x = chin_x

                # Necklace width = 2x face width (to cover shoulders)
                necklace_width = int(face_width * 2.0)
                necklace_height = int(necklace_width * 0.6)

                logger.info(f"Face detected. Placing necklace at ({necklace_x}, {necklace_y})")
            else:
                # NO FACE DETECTED - Use fallback center positioning
                # Still place the necklace, don't give up!
                logger.warning("No face detected - using center fallback positioning")

                necklace_x = w // 2
                necklace_y = int(h * 0.45)  # 45% down the image (roughly neck area)
                necklace_width = int(w * 0.5)  # 50% of image width
                necklace_height = int(necklace_width * 0.6)
                face_angle = 0

            # Resize necklace
            resized = self._resize_jewelry(cv2_necklace, necklace_width, necklace_height)

            # Rotate if face is tilted
            rotated = self._rotate_jewelry(resized, face_angle)

            # Overlay necklace on person
            result = self._overlay_png(cv2_person, rotated, (necklace_x, necklace_y), opacity)

            result_pil = self._cv2_to_pil(result)

            if face is not None:
                return result_pil, "Success: Necklace applied using face detection!"
            else:
                return result_pil, "Success: Necklace applied with fallback positioning (face not detected)"

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

            h, w = cv2_person.shape[:2]

            face = self.detect_face(cv2_person)

            if face is None:
                # Fallback positioning for earrings
                logger.warning("No face detected - using fallback earring positioning")
                left_pos = (int(w * 0.25), int(h * 0.35))
                right_pos = (int(w * 0.75), int(h * 0.35))
                earring_size = int(w * 0.08)
                face_angle = 0
            else:
                face_width = face['face_width']
                face_angle = face.get('face_angle', 0)
                left_pos = face['left_earlobe']
                right_pos = face['right_earlobe']
                earring_size = int(face_width * 0.22)

            earring_size = max(earring_size, 20)

            resized_earring = self._resize_jewelry(cv2_earring, earring_size, earring_size)
            rotated_left = self._rotate_jewelry(resized_earring, face_angle)

            # Mirror for right ear
            flipped_earring = cv2.flip(resized_earring, 1)
            rotated_right = self._rotate_jewelry(flipped_earring, face_angle)

            # Apply earrings
            result = self._overlay_png(cv2_person, rotated_left, left_pos, opacity)
            result = self._overlay_png(result, rotated_right, right_pos, opacity)

            result_pil = self._cv2_to_pil(result)

            if face is not None:
                return result_pil, "Success: Earrings applied successfully!"
            else:
                return result_pil, "Success: Earrings applied with fallback positioning"

        except Exception as e:
            logger.error(f"Earring application error: {str(e)}")
            return person_image, f"Error: {str(e)}"

    def apply_maang_tikka(self, person_image: Image.Image,
                          tikka_image: Image.Image,
                          opacity: float = 1.0) -> Tuple[Image.Image, str]:
        """
        Apply Maang Tikka (Indian forehead jewelry) using OpenCV face detection.

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

            h, w = cv2_person.shape[:2]

            face = self.detect_face(cv2_person)

            if face is None:
                # Fallback positioning
                logger.warning("No face detected - using fallback tikka positioning")
                forehead_pos = (w // 2, int(h * 0.20))
                tikka_width = int(w * 0.08)
                tikka_height = int(tikka_width * 2)
                face_angle = 0
            else:
                face_width = face['face_width']
                face_angle = face.get('face_angle', 0)
                forehead_pos = face['forehead']
                tikka_width = int(face_width * 0.14)
                tikka_height = int(face_width * 0.28)

            tikka_width = max(tikka_width, 15)
            tikka_height = max(tikka_height, 30)

            resized_tikka = self._resize_jewelry(cv2_tikka, tikka_width, tikka_height)
            rotated_tikka = self._rotate_jewelry(resized_tikka, face_angle)

            result = self._overlay_png(cv2_person, rotated_tikka, forehead_pos, opacity)

            result_pil = self._cv2_to_pil(result)

            if face is not None:
                return result_pil, "Success: Maang Tikka applied successfully!"
            else:
                return result_pil, "Success: Maang Tikka applied with fallback positioning"

        except Exception as e:
            logger.error(f"Maang Tikka application error: {str(e)}")
            return person_image, f"Error: {str(e)}"

    def apply_nose_ring(self, person_image: Image.Image,
                        nose_ring_image: Image.Image,
                        side: str = "left",
                        ring_style: str = "stud",
                        opacity: float = 1.0) -> Tuple[Image.Image, str]:
        """
        Apply nose ring using OpenCV face detection.

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

            h, w = cv2_person.shape[:2]

            face = self.detect_face(cv2_person)

            side_lower = side.lower()

            if face is None:
                # Fallback positioning
                logger.warning("No face detected - using fallback nose ring positioning")
                center_x = w // 2
                center_y = int(h * 0.40)
                if side_lower == "left":
                    nose_pos = (center_x - int(w * 0.03), center_y)
                elif side_lower == "right":
                    nose_pos = (center_x + int(w * 0.03), center_y)
                else:
                    nose_pos = (center_x, center_y)
                ring_size = int(w * 0.05)
                face_angle = 0
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
                    ring_size = int(face_width * 0.18)
                elif ring_style.lower() == "hoop":
                    ring_size = int(face_width * 0.12)
                else:
                    ring_size = int(face_width * 0.08)

            ring_size = max(ring_size, 10)

            resized_ring = self._resize_jewelry(cv2_ring, ring_size, ring_size)
            rotated_ring = self._rotate_jewelry(resized_ring, face_angle)

            if side_lower == "right":
                rotated_ring = cv2.flip(rotated_ring, 1)

            result = self._overlay_png(cv2_person, rotated_ring, nose_pos, opacity)

            result_pil = self._cv2_to_pil(result)

            if face is not None:
                return result_pil, f"Success: Nose ring applied on {side} side!"
            else:
                return result_pil, f"Success: Nose ring applied with fallback positioning"

        except Exception as e:
            logger.error(f"Nose ring application error: {str(e)}")
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


if __name__ == "__main__":
    print("=" * 60)
    print("Naari Studio - Jewelry Virtual Try-On Engine (SIMPLIFIED)")
    print("100% ZeroGPU Compatible - OpenCV Only")
    print("=" * 60)
    print(f"\nOpenCV face cascade loaded: {FACE_CASCADE is not None}")
    print(f"OpenCV eye cascade loaded: {EYE_CASCADE is not None}")
    print("\nDetection Method: OpenCV Haar Cascades ONLY")
    print("  - NO MediaPipe dependency")
    print("  - NO cvzone pose detection")
    print("  - Works on CPU, GPU, ZeroGPU")
    print("\nNecklace Positioning:")
    print("  - Uses face detection to find chin")
    print("  - Places necklace 20% below chin center")
    print("  - Fallback to center if no face detected")
    print("\nSupported jewelry types:")
    print("  - necklace: Face-based positioning")
    print("  - earrings: Face detection")
    print("  - maang_tikka: Face detection")
    print("  - nose_ring: Face detection")
    print("=" * 60)

    # Quick self-test
    print("\nRunning self-test...")
    try:
        engine = JewelryEngine()

        # Create a test image (white background with a simple "face" circle)
        test_img = np.ones((400, 300, 3), dtype=np.uint8) * 255
        # Draw a circle as a fake face
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
            print("SUCCESS: Overlay is working - result differs from input")

        print(f"Test result: {msg}")
        print("Self-test complete!")

    except Exception as e:
        print(f"Self-test failed: {e}")
        import traceback
        traceback.print_exc()
