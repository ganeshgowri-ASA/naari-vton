"""
Naari Studio - Jewelry Virtual Try-On Engine
Uses cvzone PoseModule and OpenCV for ZeroGPU compatibility.

NO MEDIAPIPE DEPENDENCY - Works on HuggingFace ZeroGPU Spaces.

Supported jewelry types:
- Necklace: Uses cvzone pose landmarks 11, 12 for shoulder positioning
- Earrings: Uses OpenCV face detection with geometric estimation
- Maang Tikka: Forehead center positioning via face geometry
- Nose Ring: Nose landmark via face geometry
"""

import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Optional, Union, List
import logging
import math
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# cvzone imports for pose detection and overlay
CVZONE_AVAILABLE = False
try:
    from cvzone.PoseModule import PoseDetector
    import cvzone
    CVZONE_AVAILABLE = True
    logger.info("cvzone PoseModule loaded successfully")
except ImportError as e:
    logger.warning(f"cvzone not available: {e}. Install with: pip install cvzone")

# OpenCV Face Detection - ZeroGPU compatible (no MediaPipe needed)
FACE_CASCADE = None
EYE_CASCADE = None

def _init_cascades():
    """Initialize OpenCV Haar Cascades for face/eye detection."""
    global FACE_CASCADE, EYE_CASCADE

    if FACE_CASCADE is not None:
        return True

    try:
        # Try to find cascade files in OpenCV data directory
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


class FaceDetector:
    """
    OpenCV-based face detector for facial jewelry positioning.
    Uses Haar Cascades - fully compatible with ZeroGPU.
    No MediaPipe dependency.
    """

    def __init__(self, scale_factor: float = 1.1, min_neighbors: int = 5):
        """
        Initialize the face detector.

        Args:
            scale_factor: Scale factor for cascade detection
            min_neighbors: Minimum neighbors for cascade detection
        """
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        _init_cascades()

    def detect_face(self, cv2_image: np.ndarray) -> Optional[dict]:
        """
        Detect face and estimate facial landmarks geometrically.

        Args:
            cv2_image: OpenCV image (BGR)

        Returns:
            Dictionary with face bounds and estimated landmark positions
        """
        if FACE_CASCADE is None:
            logger.error("Face cascade not loaded")
            return None

        try:
            # Convert to grayscale for detection
            gray = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2GRAY)

            # Detect faces
            faces = FACE_CASCADE.detectMultiScale(
                gray,
                scaleFactor=self.scale_factor,
                minNeighbors=self.min_neighbors,
                minSize=(80, 80),
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            if len(faces) == 0:
                return None

            # Take the largest face
            face = max(faces, key=lambda f: f[2] * f[3])
            x, y, w, h = face

            # Calculate facial landmark estimates based on face geometry
            # These are proportional estimates based on average face proportions

            landmarks = {
                'face_bounds': (x, y, w, h),
                'face_center': (x + w // 2, y + h // 2),
                'face_width': w,
                'face_height': h,
            }

            # Ear positions - at sides of face, about 40% down from top
            # Ears are typically slightly outside the face bounding box
            ear_y = y + int(h * 0.40)
            landmarks['left_ear'] = (x - int(w * 0.05), ear_y)
            landmarks['right_ear'] = (x + w + int(w * 0.05), ear_y)

            # Earlobe positions - lower, where earrings hang
            earlobe_y = y + int(h * 0.50)
            landmarks['left_earlobe'] = (x - int(w * 0.02), earlobe_y)
            landmarks['right_earlobe'] = (x + w + int(w * 0.02), earlobe_y)

            # Forehead center - top center of face, slightly inside the box
            landmarks['forehead'] = (x + w // 2, y + int(h * 0.12))

            # Hairline - very top center
            landmarks['hairline'] = (x + w // 2, y + int(h * 0.05))

            # Nose tip - center of face, about 60% down
            landmarks['nose_tip'] = (x + w // 2, y + int(h * 0.60))

            # Nose bottom - slightly lower
            landmarks['nose_bottom'] = (x + w // 2, y + int(h * 0.68))

            # Left and right nostrils
            nostril_offset = int(w * 0.10)
            nostril_y = y + int(h * 0.65)
            landmarks['left_nostril'] = (x + w // 2 - nostril_offset, nostril_y)
            landmarks['right_nostril'] = (x + w // 2 + nostril_offset, nostril_y)

            # Septum - center bottom of nose
            landmarks['septum'] = (x + w // 2, y + int(h * 0.70))

            # Eye positions for angle calculation
            eye_y = y + int(h * 0.35)
            eye_offset = int(w * 0.20)
            landmarks['left_eye'] = (x + w // 2 - eye_offset, eye_y)
            landmarks['right_eye'] = (x + w // 2 + eye_offset, eye_y)

            # Detect eyes for better angle calculation
            if EYE_CASCADE is not None:
                face_roi = gray[y:y+h, x:x+w]
                eyes = EYE_CASCADE.detectMultiScale(
                    face_roi,
                    scaleFactor=1.1,
                    minNeighbors=10,
                    minSize=(20, 20)
                )

                if len(eyes) >= 2:
                    # Sort eyes by x position (left to right)
                    eyes = sorted(eyes, key=lambda e: e[0])

                    # Take first two eyes
                    left_eye = eyes[0]
                    right_eye = eyes[1]

                    # Calculate eye centers (relative to full image)
                    landmarks['left_eye'] = (
                        x + left_eye[0] + left_eye[2] // 2,
                        y + left_eye[1] + left_eye[3] // 2
                    )
                    landmarks['right_eye'] = (
                        x + right_eye[0] + right_eye[2] // 2,
                        y + right_eye[1] + right_eye[3] // 2
                    )

                    # Recalculate ear positions based on eye positions
                    eye_mid_y = (landmarks['left_eye'][1] + landmarks['right_eye'][1]) // 2
                    landmarks['left_ear'] = (x - int(w * 0.05), eye_mid_y + int(h * 0.10))
                    landmarks['right_ear'] = (x + w + int(w * 0.05), eye_mid_y + int(h * 0.10))
                    landmarks['left_earlobe'] = (x - int(w * 0.02), eye_mid_y + int(h * 0.18))
                    landmarks['right_earlobe'] = (x + w + int(w * 0.02), eye_mid_y + int(h * 0.18))

            # Calculate face angle from eye positions
            left_eye = landmarks['left_eye']
            right_eye = landmarks['right_eye']
            delta_x = right_eye[0] - left_eye[0]
            delta_y = right_eye[1] - left_eye[1]
            landmarks['face_angle'] = math.degrees(math.atan2(delta_y, delta_x))

            return landmarks

        except Exception as e:
            logger.error(f"Face detection error: {e}")
            return None


class JewelryEngine:
    """
    Core engine for jewelry virtual try-on.

    Uses cvzone PoseModule for body pose detection (necklace positioning)
    and OpenCV Haar Cascades for facial jewelry (earrings, tikka, nose ring).

    NO MEDIAPIPE DEPENDENCY - ZeroGPU compatible.
    """

    # cvzone PoseDetector landmark indices
    POSE_LEFT_MOUTH = 9
    POSE_RIGHT_MOUTH = 10
    POSE_LEFT_SHOULDER = 11
    POSE_RIGHT_SHOULDER = 12
    POSE_LEFT_EAR = 7
    POSE_RIGHT_EAR = 8
    POSE_NOSE = 0

    def __init__(self,
                 pose_detection_confidence: float = 0.7,
                 pose_tracking_confidence: float = 0.7,
                 face_scale_factor: float = 1.1,
                 face_min_neighbors: int = 5):
        """
        Initialize the Jewelry Engine.

        Args:
            pose_detection_confidence: Minimum confidence for pose detection
            pose_tracking_confidence: Minimum confidence for pose tracking
            face_scale_factor: Scale factor for face cascade detection
            face_min_neighbors: Min neighbors for face cascade detection
        """
        self.pose_detection_confidence = pose_detection_confidence
        self.pose_tracking_confidence = pose_tracking_confidence

        # Initialize models (lazy loading for pose, immediate for face)
        self._pose_detector = None
        self._face_detector = FaceDetector(face_scale_factor, face_min_neighbors)

    @property
    def pose_detector(self):
        """Lazy initialization of cvzone PoseDetector."""
        if self._pose_detector is None:
            if not CVZONE_AVAILABLE:
                raise ImportError("cvzone is required for pose detection. Install with: pip install cvzone")
            self._pose_detector = PoseDetector(
                staticMode=True,
                modelComplexity=1,
                smoothLandmarks=True,
                enableSegmentation=False,
                smoothSegmentation=True,
                detectionCon=self.pose_detection_confidence,
                trackCon=self.pose_tracking_confidence
            )
            logger.info("PoseDetector initialized")
        return self._pose_detector

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

    def _calculate_shoulder_angle(self, left_shoulder: Tuple[int, int],
                                   right_shoulder: Tuple[int, int]) -> float:
        """Calculate shoulder tilt angle."""
        delta_x = right_shoulder[0] - left_shoulder[0]
        delta_y = right_shoulder[1] - left_shoulder[1]
        return math.degrees(math.atan2(delta_y, delta_x))

    def _resize_jewelry_cv2(self, jewelry: np.ndarray,
                            target_width: int, target_height: int) -> np.ndarray:
        """Resize jewelry image maintaining aspect ratio."""
        h, w = jewelry.shape[:2]
        aspect_ratio = w / h

        if target_width / target_height > aspect_ratio:
            new_height = target_height
            new_width = int(new_height * aspect_ratio)
        else:
            new_width = target_width
            new_height = int(new_width / aspect_ratio)

        new_width = max(new_width, 10)
        new_height = max(new_height, 10)

        return cv2.resize(jewelry, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)

    def _rotate_jewelry_cv2(self, jewelry: np.ndarray, angle: float) -> np.ndarray:
        """Rotate jewelry image around center."""
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
            background: Background image (BGR or BGRA)
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

        x = position[0] - ow // 2
        y = position[1] - oh // 2

        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(bw, x + ow), min(bh, y + oh)

        ox1, oy1 = max(0, -x), max(0, -y)
        ox2, oy2 = ox1 + (x2 - x1), oy1 + (y2 - y1)

        if x2 <= x1 or y2 <= y1:
            return background

        roi = background[y1:y2, x1:x2].copy()
        overlay_roi = overlay[oy1:oy2, ox1:ox2]

        alpha = overlay_roi[:, :, 3:4].astype(np.float32) / 255.0
        overlay_rgb = overlay_roi[:, :, :3].astype(np.float32)
        roi_float = roi.astype(np.float32)

        blended = overlay_rgb * alpha + roi_float * (1 - alpha)

        result = background.copy()
        result[y1:y2, x1:x2] = blended.astype(np.uint8)

        return result

    def _cvzone_overlay_png(self, background: np.ndarray, overlay: np.ndarray,
                            position: Tuple[int, int], opacity: float = 1.0) -> np.ndarray:
        """Use cvzone.overlayPNG or fallback to custom implementation."""
        try:
            if CVZONE_AVAILABLE and hasattr(cvzone, 'overlayPNG'):
                if overlay.shape[2] == 3:
                    overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2BGRA)

                if opacity < 1.0:
                    overlay = overlay.copy()
                    overlay[:, :, 3] = (overlay[:, :, 3] * opacity).astype(np.uint8)

                oh, ow = overlay.shape[:2]
                x = position[0] - ow // 2
                y = position[1] - oh // 2

                result = cvzone.overlayPNG(background, overlay, [x, y])
                return result
        except Exception as e:
            logger.warning(f"cvzone.overlayPNG failed, using fallback: {e}")

        return self._overlay_png(background, overlay, position, opacity)

    def detect_pose_landmarks(self, cv2_image: np.ndarray) -> Optional[dict]:
        """
        Detect pose landmarks using cvzone PoseDetector.

        Args:
            cv2_image: OpenCV image (BGR)

        Returns:
            Dictionary with landmark positions or None if detection failed
        """
        try:
            img, lmList = self.pose_detector.findPose(cv2_image.copy(), draw=False)

            if not lmList or len(lmList) < 13:
                return None

            lm_dict = {lm[0]: (lm[1], lm[2]) for lm in lmList if len(lm) >= 3}

            landmarks = {}

            if self.POSE_LEFT_SHOULDER in lm_dict and self.POSE_RIGHT_SHOULDER in lm_dict:
                landmarks['left_shoulder'] = lm_dict[self.POSE_LEFT_SHOULDER]
                landmarks['right_shoulder'] = lm_dict[self.POSE_RIGHT_SHOULDER]
            else:
                return None

            if self.POSE_LEFT_MOUTH in lm_dict:
                landmarks['left_mouth'] = lm_dict[self.POSE_LEFT_MOUTH]
            if self.POSE_RIGHT_MOUTH in lm_dict:
                landmarks['right_mouth'] = lm_dict[self.POSE_RIGHT_MOUTH]
            if self.POSE_NOSE in lm_dict:
                landmarks['nose'] = lm_dict[self.POSE_NOSE]
            if self.POSE_LEFT_EAR in lm_dict:
                landmarks['left_ear'] = lm_dict[self.POSE_LEFT_EAR]
            if self.POSE_RIGHT_EAR in lm_dict:
                landmarks['right_ear'] = lm_dict[self.POSE_RIGHT_EAR]

            return landmarks

        except Exception as e:
            logger.error(f"Pose detection error: {e}")
            return None

    def detect_face_landmarks(self, cv2_image: np.ndarray) -> Optional[dict]:
        """
        Detect face landmarks using OpenCV Haar Cascades.

        Args:
            cv2_image: OpenCV image (BGR)

        Returns:
            Dictionary with facial landmark estimates or None
        """
        return self._face_detector.detect_face(cv2_image)

    def apply_necklace(self, person_image: Image.Image,
                       necklace_image: Image.Image,
                       opacity: float = 1.0) -> Tuple[Image.Image, str]:
        """
        Apply necklace to person image using pose landmarks 11, 12.

        Uses cvzone PoseDetector for accurate shoulder positioning.

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

            pose_landmarks = self.detect_pose_landmarks(cv2_person)

            if pose_landmarks is None:
                return person_image, "Error: Could not detect body pose. Please use a clear front-facing photo with visible shoulders."

            left_shoulder = pose_landmarks['left_shoulder']
            right_shoulder = pose_landmarks['right_shoulder']

            shoulder_width = abs(right_shoulder[0] - left_shoulder[0])

            if shoulder_width < 50:
                return person_image, "Error: Shoulders not clearly visible. Please use a photo with visible shoulders."

            center_x = (left_shoulder[0] + right_shoulder[0]) // 2
            center_y = (left_shoulder[1] + right_shoulder[1]) // 2

            if 'left_mouth' in pose_landmarks and 'right_mouth' in pose_landmarks:
                mouth_center_y = (pose_landmarks['left_mouth'][1] + pose_landmarks['right_mouth'][1]) // 2
                neck_y = mouth_center_y + int((center_y - mouth_center_y) * 0.4)
                center_y = neck_y
            else:
                center_y -= int(shoulder_width * 0.05)

            angle = self._calculate_shoulder_angle(left_shoulder, right_shoulder)

            necklace_width = int(shoulder_width * 1.2)
            necklace_height = int(necklace_width * 0.7)

            resized_necklace = self._resize_jewelry_cv2(cv2_necklace, necklace_width, necklace_height)

            if abs(angle) > 2:
                rotated_necklace = self._rotate_jewelry_cv2(resized_necklace, angle)
            else:
                rotated_necklace = resized_necklace

            final_y = center_y + int(shoulder_width * 0.15)

            result = self._cvzone_overlay_png(cv2_person, rotated_necklace,
                                               (center_x, final_y), opacity)

            result_pil = self._cv2_to_pil(result)

            return result_pil, "Success: Necklace applied successfully!"

        except Exception as e:
            logger.error(f"Necklace application error: {str(e)}")
            return person_image, f"Error: {str(e)}"

    def apply_earrings(self, person_image: Image.Image,
                       earring_image: Image.Image,
                       opacity: float = 1.0) -> Tuple[Image.Image, str]:
        """
        Apply earrings using OpenCV face detection.

        Single earring image is mirrored for left/right ears.

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

            face_landmarks = self.detect_face_landmarks(cv2_person)

            if face_landmarks is None:
                return person_image, "Error: Could not detect face. Please use a clear front-facing photo."

            face_width = face_landmarks['face_width']
            face_angle = face_landmarks.get('face_angle', 0)

            left_ear_pos = face_landmarks['left_earlobe']
            right_ear_pos = face_landmarks['right_earlobe']

            # Scale earrings relative to face size (20% of face width)
            earring_size = int(face_width * 0.22)
            earring_size = max(earring_size, 20)

            resized_earring = self._resize_jewelry_cv2(cv2_earring, earring_size, earring_size)

            rotated_left = self._rotate_jewelry_cv2(resized_earring, face_angle)

            # Mirror for right ear
            flipped_earring = cv2.flip(resized_earring, 1)
            rotated_right = self._rotate_jewelry_cv2(flipped_earring, face_angle)

            # Adjust positions slightly
            left_pos = (left_ear_pos[0] - int(face_width * 0.02),
                       left_ear_pos[1] + int(face_width * 0.05))
            right_pos = (right_ear_pos[0] + int(face_width * 0.02),
                        right_ear_pos[1] + int(face_width * 0.05))

            result = self._cvzone_overlay_png(cv2_person, rotated_left, left_pos, opacity)
            result = self._cvzone_overlay_png(result, rotated_right, right_pos, opacity)

            result_pil = self._cv2_to_pil(result)

            return result_pil, "Success: Earrings applied successfully!"

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

            face_landmarks = self.detect_face_landmarks(cv2_person)

            if face_landmarks is None:
                return person_image, "Error: Could not detect face. Please use a clear front-facing photo."

            face_width = face_landmarks['face_width']
            face_angle = face_landmarks.get('face_angle', 0)

            forehead_pos = face_landmarks['forehead']

            # Scale tikka relative to face size
            tikka_width = int(face_width * 0.14)
            tikka_height = int(face_width * 0.28)
            tikka_width = max(tikka_width, 15)
            tikka_height = max(tikka_height, 30)

            resized_tikka = self._resize_jewelry_cv2(cv2_tikka, tikka_width, tikka_height)
            rotated_tikka = self._rotate_jewelry_cv2(resized_tikka, face_angle)

            result = self._cvzone_overlay_png(cv2_person, rotated_tikka, forehead_pos, opacity)

            result_pil = self._cv2_to_pil(result)

            return result_pil, "Success: Maang Tikka applied successfully!"

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

            face_landmarks = self.detect_face_landmarks(cv2_person)

            if face_landmarks is None:
                return person_image, "Error: Could not detect face. Please use a clear front-facing photo."

            face_width = face_landmarks['face_width']
            face_angle = face_landmarks.get('face_angle', 0)

            side_lower = side.lower()
            if side_lower == "septum":
                nose_pos = face_landmarks['septum']
            elif side_lower == "right":
                nose_pos = face_landmarks['right_nostril']
            else:
                nose_pos = face_landmarks['left_nostril']

            # Determine ring size based on style
            if ring_style.lower() == "nath":
                ring_size = int(face_width * 0.18)
            elif ring_style.lower() == "hoop":
                ring_size = int(face_width * 0.12)
            else:
                ring_size = int(face_width * 0.08)

            ring_size = max(ring_size, 10)

            resized_ring = self._resize_jewelry_cv2(cv2_ring, ring_size, ring_size)
            rotated_ring = self._rotate_jewelry_cv2(resized_ring, face_angle)

            if side_lower == "right":
                rotated_ring = cv2.flip(rotated_ring, 1)

            result = self._cvzone_overlay_png(cv2_person, rotated_ring, nose_pos, opacity)

            result_pil = self._cv2_to_pil(result)

            return result_pil, f"Success: Nose ring applied on {side} side!"

        except Exception as e:
            logger.error(f"Nose ring application error: {str(e)}")
            return person_image, f"Error: {str(e)}"

    def close(self):
        """Release resources."""
        if self._pose_detector is not None:
            self._pose_detector = None


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
    Convert jewelry image to RGBA format.

    NOTE: rembg background removal is disabled for performance reasons.
    On CPU basic hardware, rembg takes 16+ minutes which is unacceptable.
    This function now simply converts the input to RGBA without background removal.
    Users should upload jewelry images that already have transparent backgrounds.

    Args:
        jewelry_image: PIL Image or numpy array of the jewelry

    Returns:
        PIL Image in RGBA format (no background removal performed)
    """
    # Skip rembg entirely - too slow on CPU (16+ minutes)
    # from rembg import remove  # DISABLED for performance

    if isinstance(jewelry_image, np.ndarray):
        jewelry_image = Image.fromarray(cv2.cvtColor(jewelry_image, cv2.COLOR_BGR2RGB))

    # Just convert to RGBA and return (no background removal)
    logger.info("Background removal skipped for performance - returning image as RGBA")
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
            if len(person_image.shape) == 3 and person_image.shape[2] in [3, 4]:
                person_image = Image.fromarray(person_image)
            else:
                person_image = Image.fromarray(person_image)

        # Convert jewelry image to PIL
        if isinstance(jewelry_image, str):
            jewelry_image = Image.open(jewelry_image)
        elif isinstance(jewelry_image, np.ndarray):
            if len(jewelry_image.shape) == 3 and jewelry_image.shape[2] in [3, 4]:
                jewelry_image = Image.fromarray(jewelry_image)
            else:
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
    print("Naari Studio - Jewelry Virtual Try-On Engine")
    print("ZeroGPU Compatible (NO MEDIAPIPE)")
    print("=" * 60)
    print(f"\ncvzone available: {CVZONE_AVAILABLE}")
    print(f"OpenCV face cascade loaded: {FACE_CASCADE is not None}")
    print(f"OpenCV eye cascade loaded: {EYE_CASCADE is not None}")
    print("\nPose Detection (cvzone PoseDetector):")
    print("  - 11: left_shoulder")
    print("  - 12: right_shoulder")
    print("  - Used for necklace positioning")
    print("\nFace Detection (OpenCV Haar Cascades):")
    print("  - Face bounding box detection")
    print("  - Geometric landmark estimation")
    print("  - Used for earrings, tikka, nose ring")
    print("\nSupported jewelry types:")
    print("  - necklace: Uses pose landmarks")
    print("  - earrings: Uses face detection")
    print("  - maang_tikka: Uses face detection")
    print("  - nose_ring: Uses face detection")
    print("=" * 60)
