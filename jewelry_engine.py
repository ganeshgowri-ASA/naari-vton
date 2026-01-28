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
