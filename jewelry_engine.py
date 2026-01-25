"""
Naari Studio - Jewelry Virtual Try-On Engine
Uses MediaPipe Pose and Face Mesh for accurate jewelry placement.

Supported jewelry types:
- Necklace: Placed on chest/neck area using shoulder landmarks
- Earrings: Placed at ear positions using face mesh
- Maang Tikka: Placed on forehead center using face mesh
- Nose Ring: Placed on nose using face mesh landmarks
"""

import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Optional, Union, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MediaPipe imports with error handling
try:
    import mediapipe as mp
    mp_pose = mp.solutions.pose
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    logger.warning("MediaPipe not available. Install with: pip install mediapipe")
    MEDIAPIPE_AVAILABLE = False


class JewelryEngine:
    """
    Core engine for jewelry virtual try-on using MediaPipe.

    Handles detection of body/face landmarks and jewelry overlay.
    """

    # MediaPipe Face Mesh landmark indices
    # Reference: https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png

    # Ear landmarks (approximate positions near ear)
    LEFT_EAR_INDICES = [234, 127, 162, 21]  # Left ear region
    RIGHT_EAR_INDICES = [454, 356, 389, 251]  # Right ear region

    # Forehead center for Maang Tikka
    FOREHEAD_CENTER_INDICES = [10, 151, 9]  # Top of forehead
    FOREHEAD_BOTTOM_INDICES = [8, 168]  # Bridge of nose for reference

    # Nose landmarks for nose ring
    NOSE_TIP_INDEX = 4
    NOSE_BOTTOM_INDEX = 94
    NOSE_LEFT_INDEX = 279  # Left nostril area
    NOSE_RIGHT_INDEX = 49  # Right nostril area

    # Face contour for scaling reference
    FACE_OVAL_INDICES = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                         397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                         172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

    def __init__(self,
                 pose_confidence: float = 0.5,
                 face_confidence: float = 0.5):
        """
        Initialize the Jewelry Engine.

        Args:
            pose_confidence: Minimum confidence for pose detection
            face_confidence: Minimum confidence for face mesh detection
        """
        if not MEDIAPIPE_AVAILABLE:
            raise ImportError("MediaPipe is required. Install with: pip install mediapipe")

        self.pose_confidence = pose_confidence
        self.face_confidence = face_confidence

        # Initialize MediaPipe models (lazy loading)
        self._pose = None
        self._face_mesh = None

    @property
    def pose(self):
        """Lazy initialization of pose model."""
        if self._pose is None:
            self._pose = mp_pose.Pose(
                static_image_mode=True,
                model_complexity=2,
                enable_segmentation=False,
                min_detection_confidence=self.pose_confidence
            )
        return self._pose

    @property
    def face_mesh(self):
        """Lazy initialization of face mesh model."""
        if self._face_mesh is None:
            self._face_mesh = mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=self.face_confidence
            )
        return self._face_mesh

    def _pil_to_cv2(self, pil_image: Image.Image) -> np.ndarray:
        """Convert PIL Image to OpenCV format (BGR)."""
        rgb_array = np.array(pil_image.convert('RGB'))
        return cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)

    def _cv2_to_pil(self, cv2_image: np.ndarray) -> Image.Image:
        """Convert OpenCV image (BGR) to PIL Image."""
        rgb_array = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb_array)

    def _ensure_rgba(self, image: Image.Image) -> Image.Image:
        """Ensure image has alpha channel."""
        if image.mode != 'RGBA':
            return image.convert('RGBA')
        return image

    def _get_landmark_point(self, landmarks, index: int,
                           img_width: int, img_height: int) -> Tuple[int, int]:
        """Get pixel coordinates from normalized landmark."""
        landmark = landmarks.landmark[index]
        return (int(landmark.x * img_width), int(landmark.y * img_height))

    def _get_average_point(self, landmarks, indices: list,
                          img_width: int, img_height: int) -> Tuple[int, int]:
        """Get average point from multiple landmarks."""
        x_sum, y_sum = 0, 0
        for idx in indices:
            point = self._get_landmark_point(landmarks, idx, img_width, img_height)
            x_sum += point[0]
            y_sum += point[1]
        return (x_sum // len(indices), y_sum // len(indices))

    def _calculate_face_width(self, landmarks, img_width: int, img_height: int) -> float:
        """Calculate face width for scaling reference."""
        left_point = self._get_landmark_point(landmarks, 234, img_width, img_height)
        right_point = self._get_landmark_point(landmarks, 454, img_width, img_height)
        return np.sqrt((right_point[0] - left_point[0])**2 +
                      (right_point[1] - left_point[1])**2)

    def _calculate_face_angle(self, landmarks, img_width: int, img_height: int) -> float:
        """Calculate face rotation angle for jewelry alignment."""
        left_eye = self._get_landmark_point(landmarks, 33, img_width, img_height)
        right_eye = self._get_landmark_point(landmarks, 263, img_width, img_height)

        delta_x = right_eye[0] - left_eye[0]
        delta_y = right_eye[1] - left_eye[1]

        angle = np.degrees(np.arctan2(delta_y, delta_x))
        return angle

    def _resize_jewelry(self, jewelry: Image.Image,
                        target_width: int, target_height: int) -> Image.Image:
        """Resize jewelry maintaining aspect ratio."""
        jewelry = self._ensure_rgba(jewelry)

        orig_width, orig_height = jewelry.size
        aspect_ratio = orig_width / orig_height

        if target_width / target_height > aspect_ratio:
            new_height = target_height
            new_width = int(new_height * aspect_ratio)
        else:
            new_width = target_width
            new_height = int(new_width / aspect_ratio)

        return jewelry.resize((new_width, new_height), Image.Resampling.LANCZOS)

    def _rotate_jewelry(self, jewelry: Image.Image, angle: float) -> Image.Image:
        """Rotate jewelry image around center."""
        return jewelry.rotate(-angle, expand=True, resample=Image.Resampling.BICUBIC)

    def _overlay_with_alpha(self, background: Image.Image,
                           overlay: Image.Image,
                           position: Tuple[int, int],
                           opacity: float = 1.0) -> Image.Image:
        """
        Overlay image with alpha blending.

        Args:
            background: Base image
            overlay: Image to overlay (must have alpha channel)
            position: (x, y) top-left position for overlay
            opacity: Opacity of overlay (0.0 to 1.0)
        """
        background = self._ensure_rgba(background)
        overlay = self._ensure_rgba(overlay)

        # Apply opacity to overlay
        if opacity < 1.0:
            overlay_array = np.array(overlay)
            overlay_array[:, :, 3] = (overlay_array[:, :, 3] * opacity).astype(np.uint8)
            overlay = Image.fromarray(overlay_array)

        # Create a copy to avoid modifying original
        result = background.copy()

        # Calculate paste position (center the overlay at position)
        x = position[0] - overlay.width // 2
        y = position[1] - overlay.height // 2

        # Create a temporary image for compositing
        temp = Image.new('RGBA', result.size, (0, 0, 0, 0))
        temp.paste(overlay, (x, y))

        # Composite the images
        result = Image.alpha_composite(result, temp)

        return result

    def detect_pose_landmarks(self, person_image: Image.Image) -> Optional[Any]:
        """
        Detect pose landmarks in the person image.

        Args:
            person_image: PIL Image of the person

        Returns:
            MediaPipe pose landmarks or None if detection failed
        """
        cv2_image = self._pil_to_cv2(person_image)
        rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)

        results = self.pose.process(rgb_image)

        if results.pose_landmarks:
            return results.pose_landmarks
        return None

    def detect_face_landmarks(self, person_image: Image.Image) -> Optional[Any]:
        """
        Detect face mesh landmarks in the person image.

        Args:
            person_image: PIL Image of the person

        Returns:
            MediaPipe face landmarks or None if detection failed
        """
        cv2_image = self._pil_to_cv2(person_image)
        rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)

        results = self.face_mesh.process(rgb_image)

        if results.multi_face_landmarks and len(results.multi_face_landmarks) > 0:
            return results.multi_face_landmarks[0]
        return None

    def apply_necklace(self, person_image: Image.Image,
                       necklace_image: Image.Image,
                       opacity: float = 1.0) -> Tuple[Image.Image, str]:
        """
        Apply necklace to person image.

        Uses pose detection to find shoulder positions and places
        the necklace on the chest/neck area.

        Args:
            person_image: PIL Image of the person
            necklace_image: PIL Image of the necklace
            opacity: Opacity of the necklace overlay (0.0 to 1.0)

        Returns:
            Tuple of (result image, status message)
        """
        try:
            pose_landmarks = self.detect_pose_landmarks(person_image)

            if pose_landmarks is None:
                return person_image, "Error: Could not detect body pose. Please use a clear front-facing photo."

            img_width, img_height = person_image.size

            # Get shoulder landmarks
            left_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]

            # Calculate shoulder positions
            left_pos = (int(left_shoulder.x * img_width), int(left_shoulder.y * img_height))
            right_pos = (int(right_shoulder.x * img_width), int(right_shoulder.y * img_height))

            # Calculate necklace placement
            center_x = (left_pos[0] + right_pos[0]) // 2
            center_y = (left_pos[1] + right_pos[1]) // 2

            # Shoulder width for scaling
            shoulder_width = abs(right_pos[0] - left_pos[0])

            # Calculate angle for rotation (shoulder tilt)
            angle = np.degrees(np.arctan2(right_pos[1] - left_pos[1],
                                          right_pos[0] - left_pos[0]))

            # Scale necklace to shoulder width (with padding)
            necklace_width = int(shoulder_width * 1.2)
            necklace_height = int(necklace_width * 0.8)  # Typical necklace aspect

            # Resize and rotate necklace
            resized_necklace = self._resize_jewelry(necklace_image,
                                                    necklace_width,
                                                    necklace_height)
            rotated_necklace = self._rotate_jewelry(resized_necklace, angle)

            # Adjust vertical position (slightly below shoulders for necklace)
            center_y += int(shoulder_width * 0.15)

            # Apply overlay
            result = self._overlay_with_alpha(person_image, rotated_necklace,
                                             (center_x, center_y), opacity)

            return result, "Success: Necklace applied successfully!"

        except Exception as e:
            logger.error(f"Necklace application error: {str(e)}")
            return person_image, f"Error: {str(e)}"

    def apply_earrings(self, person_image: Image.Image,
                       earring_image: Image.Image,
                       opacity: float = 1.0) -> Tuple[Image.Image, str]:
        """
        Apply earrings to person image.

        Uses face mesh to detect ear positions and applies
        earrings at the correct locations.

        Args:
            person_image: PIL Image of the person
            earring_image: PIL Image of a single earring
            opacity: Opacity of the earring overlay (0.0 to 1.0)

        Returns:
            Tuple of (result image, status message)
        """
        try:
            face_landmarks = self.detect_face_landmarks(person_image)

            if face_landmarks is None:
                return person_image, "Error: Could not detect face. Please use a clear front-facing photo."

            img_width, img_height = person_image.size

            # Get ear positions
            left_ear_pos = self._get_average_point(face_landmarks,
                                                   self.LEFT_EAR_INDICES,
                                                   img_width, img_height)
            right_ear_pos = self._get_average_point(face_landmarks,
                                                    self.RIGHT_EAR_INDICES,
                                                    img_width, img_height)

            # Calculate face width for scaling
            face_width = self._calculate_face_width(face_landmarks, img_width, img_height)

            # Calculate face angle for rotation
            face_angle = self._calculate_face_angle(face_landmarks, img_width, img_height)

            # Scale earrings relative to face size
            earring_size = int(face_width * 0.2)  # Earring size relative to face

            # Prepare earring images
            left_earring = self._resize_jewelry(earring_image, earring_size, earring_size)
            right_earring = left_earring.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

            # Rotate earrings to match face angle
            left_earring = self._rotate_jewelry(left_earring, face_angle)
            right_earring = self._rotate_jewelry(right_earring, face_angle)

            # Adjust ear positions (slightly down and outward for earlobes)
            left_ear_pos = (left_ear_pos[0] - int(face_width * 0.05),
                          left_ear_pos[1] + int(face_width * 0.1))
            right_ear_pos = (right_ear_pos[0] + int(face_width * 0.05),
                           right_ear_pos[1] + int(face_width * 0.1))

            # Apply left earring
            result = self._overlay_with_alpha(person_image, left_earring,
                                             left_ear_pos, opacity)

            # Apply right earring
            result = self._overlay_with_alpha(result, right_earring,
                                             right_ear_pos, opacity)

            return result, "Success: Earrings applied successfully!"

        except Exception as e:
            logger.error(f"Earring application error: {str(e)}")
            return person_image, f"Error: {str(e)}"

    def apply_maang_tikka(self, person_image: Image.Image,
                          tikka_image: Image.Image,
                          opacity: float = 1.0) -> Tuple[Image.Image, str]:
        """
        Apply Maang Tikka to person image.

        Maang Tikka is a traditional Indian jewelry piece worn on the forehead,
        typically at the center hairline.

        Args:
            person_image: PIL Image of the person
            tikka_image: PIL Image of the maang tikka
            opacity: Opacity of the overlay (0.0 to 1.0)

        Returns:
            Tuple of (result image, status message)
        """
        try:
            face_landmarks = self.detect_face_landmarks(person_image)

            if face_landmarks is None:
                return person_image, "Error: Could not detect face. Please use a clear front-facing photo."

            img_width, img_height = person_image.size

            # Get forehead center position
            forehead_pos = self._get_average_point(face_landmarks,
                                                   self.FOREHEAD_CENTER_INDICES,
                                                   img_width, img_height)

            # Calculate face measurements
            face_width = self._calculate_face_width(face_landmarks, img_width, img_height)
            face_angle = self._calculate_face_angle(face_landmarks, img_width, img_height)

            # Scale tikka relative to face size
            tikka_width = int(face_width * 0.15)
            tikka_height = int(face_width * 0.25)  # Tikkas are typically longer than wide

            # Resize and rotate tikka
            resized_tikka = self._resize_jewelry(tikka_image, tikka_width, tikka_height)
            rotated_tikka = self._rotate_jewelry(resized_tikka, face_angle)

            # Adjust position (slightly up from detected forehead center)
            adjusted_pos = (forehead_pos[0], forehead_pos[1] - int(face_width * 0.02))

            # Apply overlay
            result = self._overlay_with_alpha(person_image, rotated_tikka,
                                             adjusted_pos, opacity)

            return result, "Success: Maang Tikka applied successfully!"

        except Exception as e:
            logger.error(f"Maang Tikka application error: {str(e)}")
            return person_image, f"Error: {str(e)}"

    def apply_nose_ring(self, person_image: Image.Image,
                        nose_ring_image: Image.Image,
                        side: str = "left",
                        opacity: float = 1.0) -> Tuple[Image.Image, str]:
        """
        Apply nose ring to person image.

        Args:
            person_image: PIL Image of the person
            nose_ring_image: PIL Image of the nose ring
            side: Which side to apply - "left" or "right"
            opacity: Opacity of the overlay (0.0 to 1.0)

        Returns:
            Tuple of (result image, status message)
        """
        try:
            face_landmarks = self.detect_face_landmarks(person_image)

            if face_landmarks is None:
                return person_image, "Error: Could not detect face. Please use a clear front-facing photo."

            img_width, img_height = person_image.size

            # Get nose positions
            if side.lower() == "left":
                nose_pos = self._get_landmark_point(face_landmarks,
                                                    self.NOSE_LEFT_INDEX,
                                                    img_width, img_height)
            else:
                nose_pos = self._get_landmark_point(face_landmarks,
                                                    self.NOSE_RIGHT_INDEX,
                                                    img_width, img_height)

            # Calculate face measurements
            face_width = self._calculate_face_width(face_landmarks, img_width, img_height)
            face_angle = self._calculate_face_angle(face_landmarks, img_width, img_height)

            # Scale nose ring relative to face size (small)
            ring_size = int(face_width * 0.08)

            # Resize and rotate nose ring
            resized_ring = self._resize_jewelry(nose_ring_image, ring_size, ring_size)
            rotated_ring = self._rotate_jewelry(resized_ring, face_angle)

            # Flip if right side
            if side.lower() == "right":
                rotated_ring = rotated_ring.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

            # Apply overlay
            result = self._overlay_with_alpha(person_image, rotated_ring,
                                             nose_pos, opacity)

            return result, f"Success: Nose ring applied on {side} side!"

        except Exception as e:
            logger.error(f"Nose ring application error: {str(e)}")
            return person_image, f"Error: {str(e)}"

    def close(self):
        """Release MediaPipe resources."""
        if self._pose is not None:
            self._pose.close()
            self._pose = None
        if self._face_mesh is not None:
            self._face_mesh.close()
            self._face_mesh = None


# Global engine instance for the wrapper function
_engine_instance: Optional[JewelryEngine] = None


def get_engine() -> JewelryEngine:
    """Get or create the global engine instance."""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = JewelryEngine()
    return _engine_instance


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
        jewelry_type: Type of jewelry - "necklace", "earrings", "maang_tikka", "nose_ring"
        opacity: Opacity of the jewelry overlay (0.0 to 1.0)
        **kwargs: Additional arguments (e.g., 'side' for nose ring)

    Returns:
        Tuple of (result PIL Image, status message)

    Example:
        result, message = apply_jewelry(person_img, necklace_img, "necklace", 0.9)
    """
    # Input validation and conversion
    try:
        # Convert person image to PIL
        if isinstance(person_image, str):
            person_image = Image.open(person_image)
        elif isinstance(person_image, np.ndarray):
            person_image = Image.fromarray(cv2.cvtColor(person_image, cv2.COLOR_BGR2RGB))

        # Convert jewelry image to PIL
        if isinstance(jewelry_image, str):
            jewelry_image = Image.open(jewelry_image)
        elif isinstance(jewelry_image, np.ndarray):
            jewelry_image = Image.fromarray(cv2.cvtColor(jewelry_image, cv2.COLOR_BGR2RGB))

        # Validate inputs
        if person_image is None:
            return None, "Error: Please upload a person photo."
        if jewelry_image is None:
            return None, "Error: Please upload a jewelry image."

        # Ensure images are in correct format
        person_image = person_image.convert('RGB')
        jewelry_image = jewelry_image.convert('RGBA')

    except Exception as e:
        return None, f"Error: Could not load images - {str(e)}"

    # Get engine and apply jewelry
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
        return engine.apply_nose_ring(person_image, jewelry_image, side, opacity)

    else:
        return person_image, f"Error: Unknown jewelry type '{jewelry_type}'. Supported: necklace, earrings, maang_tikka, nose_ring"


def remove_jewelry_background(jewelry_image: Image.Image) -> Image.Image:
    """
    Remove background from jewelry image using rembg.

    Args:
        jewelry_image: PIL Image of the jewelry

    Returns:
        PIL Image with transparent background
    """
    try:
        from rembg import remove
        return remove(jewelry_image)
    except ImportError:
        logger.warning("rembg not installed. Install with: pip install rembg")
        return jewelry_image.convert('RGBA')
    except Exception as e:
        logger.error(f"Background removal error: {str(e)}")
        return jewelry_image.convert('RGBA')


if __name__ == "__main__":
    # Test the engine with a simple check
    print("Jewelry Engine Module")
    print("=" * 50)
    print(f"MediaPipe available: {MEDIAPIPE_AVAILABLE}")
    print("\nSupported jewelry types:")
    print("  - necklace")
    print("  - earrings")
    print("  - maang_tikka")
    print("  - nose_ring")
    print("\nUsage:")
    print("  from jewelry_engine import apply_jewelry")
    print("  result, message = apply_jewelry(person_img, jewelry_img, 'necklace', 0.9)")
