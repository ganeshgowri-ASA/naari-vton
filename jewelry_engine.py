"""
Naari Studio - Jewelry Virtual Try-On Engine
Comprehensive jewelry placement using cvzone PoseModule and MediaPipe Face Mesh.

Supported jewelry types:
- Necklace: Uses pose landmarks 9, 10, 11, 12 for shoulder/neck positioning
- Earrings: Uses face mesh ear landmarks, mirrored for left/right
- Maang Tikka: Forehead center positioning
- Nose Ring: Nose landmark positioning

Based on GemFit NecklaceTryOn implementation pattern.
"""

import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Optional, Union, Any, List
import logging
import math

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

# MediaPipe imports with HuggingFace Spaces compatibility
# Try standard import first, then fallback to mediapipe.python.solutions
MEDIAPIPE_AVAILABLE = False
mp_face_mesh = None
mp_drawing = None

try:
    # Standard import (works on most systems)
    import mediapipe as mp
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    MEDIAPIPE_AVAILABLE = True
    logger.info("MediaPipe loaded with standard import")
except Exception:
    # Fallback for HuggingFace Spaces or other environments
    try:
        from mediapipe.python.solutions import face_mesh as mp_face_mesh_module
        from mediapipe.python.solutions import drawing_utils as mp_drawing_module
        mp_face_mesh = mp_face_mesh_module
        mp_drawing = mp_drawing_module
        MEDIAPIPE_AVAILABLE = True
        logger.info("MediaPipe loaded with HuggingFace compatible import (mediapipe.python.solutions)")
    except Exception as e:
        logger.warning(f"MediaPipe not available: {e}. Install with: pip install mediapipe")
        mp_face_mesh = None
        mp_drawing = None


class JewelryEngine:
    """
    Core engine for jewelry virtual try-on.

    Uses cvzone PoseModule for body pose detection (necklace positioning)
    and MediaPipe Face Mesh for facial jewelry (earrings, tikka, nose ring).
    """

    # cvzone PoseDetector landmark indices
    # Reference: https://google.github.io/mediapipe/solutions/pose.html
    # 9: mouth_left, 10: mouth_right (used for neck reference)
    # 11: left_shoulder, 12: right_shoulder
    POSE_LEFT_MOUTH = 9
    POSE_RIGHT_MOUTH = 10
    POSE_LEFT_SHOULDER = 11
    POSE_RIGHT_SHOULDER = 12
    POSE_LEFT_EAR = 7
    POSE_RIGHT_EAR = 8
    POSE_NOSE = 0

    # MediaPipe Face Mesh landmark indices
    # Reference: https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png

    # Ear landmarks (tragus/earlobe region)
    LEFT_EAR_INDICES = [234, 127, 162, 21, 54]  # Left ear region
    RIGHT_EAR_INDICES = [454, 356, 389, 251, 284]  # Right ear region
    LEFT_EARLOBE = 177  # Left earlobe attachment point
    RIGHT_EARLOBE = 401  # Right earlobe attachment point

    # Forehead center for Maang Tikka
    FOREHEAD_CENTER_INDICES = [10, 151, 9, 8]  # Top of forehead
    FOREHEAD_HAIRLINE = 10  # Hairline center

    # Nose landmarks for nose ring
    NOSE_TIP_INDEX = 4
    NOSE_BOTTOM_INDEX = 94
    NOSE_LEFT_NOSTRIL = 129  # Left nostril outer
    NOSE_RIGHT_NOSTRIL = 358  # Right nostril outer
    NOSE_LEFT_ALA = 279  # Left ala (wing)
    NOSE_RIGHT_ALA = 49  # Right ala (wing)
    NOSE_SEPTUM = 2  # Septum for bull ring style

    # Face contour for scaling reference
    FACE_LEFT_EDGE = 234  # Left temple
    FACE_RIGHT_EDGE = 454  # Right temple

    # Eye landmarks for face angle calculation
    LEFT_EYE_OUTER = 33
    RIGHT_EYE_OUTER = 263

    def __init__(self,
                 pose_detection_confidence: float = 0.7,
                 pose_tracking_confidence: float = 0.7,
                 face_detection_confidence: float = 0.5,
                 face_tracking_confidence: float = 0.5):
        """
        Initialize the Jewelry Engine.

        Args:
            pose_detection_confidence: Minimum confidence for pose detection
            pose_tracking_confidence: Minimum confidence for pose tracking
            face_detection_confidence: Minimum confidence for face mesh detection
            face_tracking_confidence: Minimum confidence for face mesh tracking
        """
        self.pose_detection_confidence = pose_detection_confidence
        self.pose_tracking_confidence = pose_tracking_confidence
        self.face_detection_confidence = face_detection_confidence
        self.face_tracking_confidence = face_tracking_confidence

        # Initialize models (lazy loading)
        self._pose_detector = None
        self._face_mesh = None

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

    @property
    def face_mesh(self):
        """Lazy initialization of MediaPipe Face Mesh."""
        if self._face_mesh is None:
            if not MEDIAPIPE_AVAILABLE:
                raise ImportError("MediaPipe is required for face mesh. Install with: pip install mediapipe")
            self._face_mesh = mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=self.face_detection_confidence,
                min_tracking_confidence=self.face_tracking_confidence
            )
            logger.info("FaceMesh initialized")
        return self._face_mesh

    def _pil_to_cv2(self, pil_image: Image.Image) -> np.ndarray:
        """Convert PIL Image to OpenCV format (BGR)."""
        rgb_array = np.array(pil_image.convert('RGB'))
        return cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)

    def _cv2_to_pil(self, cv2_image: np.ndarray) -> Image.Image:
        """Convert OpenCV image (BGR) to PIL Image."""
        if len(cv2_image.shape) == 3 and cv2_image.shape[2] == 4:
            # BGRA to RGBA
            rgba = cv2.cvtColor(cv2_image, cv2.COLOR_BGRA2RGBA)
            return Image.fromarray(rgba)
        else:
            # BGR to RGB
            rgb_array = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
            return Image.fromarray(rgb_array)

    def _ensure_rgba(self, image: Image.Image) -> Image.Image:
        """Ensure image has alpha channel."""
        if image.mode != 'RGBA':
            return image.convert('RGBA')
        return image

    def _ensure_bgra(self, cv2_image: np.ndarray) -> np.ndarray:
        """Ensure OpenCV image has alpha channel (BGRA)."""
        if len(cv2_image.shape) == 2:
            # Grayscale to BGRA
            return cv2.cvtColor(cv2_image, cv2.COLOR_GRAY2BGRA)
        elif cv2_image.shape[2] == 3:
            # BGR to BGRA
            return cv2.cvtColor(cv2_image, cv2.COLOR_BGR2BGRA)
        elif cv2_image.shape[2] == 4:
            return cv2_image
        return cv2_image

    def _pil_to_cv2_rgba(self, pil_image: Image.Image) -> np.ndarray:
        """Convert PIL RGBA to OpenCV BGRA."""
        rgba_array = np.array(pil_image.convert('RGBA'))
        return cv2.cvtColor(rgba_array, cv2.COLOR_RGBA2BGRA)

    def _get_face_landmark_point(self, landmarks, index: int,
                                  img_width: int, img_height: int) -> Tuple[int, int]:
        """Get pixel coordinates from normalized face mesh landmark."""
        landmark = landmarks.landmark[index]
        return (int(landmark.x * img_width), int(landmark.y * img_height))

    def _get_face_average_point(self, landmarks, indices: List[int],
                                 img_width: int, img_height: int) -> Tuple[int, int]:
        """Get average point from multiple face mesh landmarks."""
        x_sum, y_sum = 0, 0
        for idx in indices:
            point = self._get_face_landmark_point(landmarks, idx, img_width, img_height)
            x_sum += point[0]
            y_sum += point[1]
        return (x_sum // len(indices), y_sum // len(indices))

    def _calculate_face_width(self, landmarks, img_width: int, img_height: int) -> float:
        """Calculate face width for scaling reference."""
        left_point = self._get_face_landmark_point(landmarks, self.FACE_LEFT_EDGE, img_width, img_height)
        right_point = self._get_face_landmark_point(landmarks, self.FACE_RIGHT_EDGE, img_width, img_height)
        return math.sqrt((right_point[0] - left_point[0])**2 +
                        (right_point[1] - left_point[1])**2)

    def _calculate_face_angle(self, landmarks, img_width: int, img_height: int) -> float:
        """Calculate face rotation angle for jewelry alignment."""
        left_eye = self._get_face_landmark_point(landmarks, self.LEFT_EYE_OUTER, img_width, img_height)
        right_eye = self._get_face_landmark_point(landmarks, self.RIGHT_EYE_OUTER, img_width, img_height)

        delta_x = right_eye[0] - left_eye[0]
        delta_y = right_eye[1] - left_eye[1]

        angle = math.degrees(math.atan2(delta_y, delta_x))
        return angle

    def _calculate_shoulder_angle(self, left_shoulder: Tuple[int, int],
                                   right_shoulder: Tuple[int, int]) -> float:
        """Calculate shoulder tilt angle."""
        delta_x = right_shoulder[0] - left_shoulder[0]
        delta_y = right_shoulder[1] - left_shoulder[1]
        return math.degrees(math.atan2(delta_y, delta_x))

    def _resize_jewelry_cv2(self, jewelry: np.ndarray,
                            target_width: int, target_height: int) -> np.ndarray:
        """Resize jewelry image maintaining aspect ratio (OpenCV)."""
        h, w = jewelry.shape[:2]
        aspect_ratio = w / h

        if target_width / target_height > aspect_ratio:
            new_height = target_height
            new_width = int(new_height * aspect_ratio)
        else:
            new_width = target_width
            new_height = int(new_width / aspect_ratio)

        # Ensure minimum size
        new_width = max(new_width, 10)
        new_height = max(new_height, 10)

        return cv2.resize(jewelry, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)

    def _rotate_jewelry_cv2(self, jewelry: np.ndarray, angle: float) -> np.ndarray:
        """Rotate jewelry image around center (OpenCV)."""
        h, w = jewelry.shape[:2]
        center = (w // 2, h // 2)

        # Get rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, -angle, 1.0)

        # Calculate new bounding box size
        cos = abs(rotation_matrix[0, 0])
        sin = abs(rotation_matrix[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))

        # Adjust the rotation matrix
        rotation_matrix[0, 2] += (new_w / 2) - center[0]
        rotation_matrix[1, 2] += (new_h / 2) - center[1]

        # Perform rotation with transparent background
        return cv2.warpAffine(jewelry, rotation_matrix, (new_w, new_h),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=(0, 0, 0, 0))

    def _overlay_png(self, background: np.ndarray, overlay: np.ndarray,
                     position: Tuple[int, int], opacity: float = 1.0) -> np.ndarray:
        """
        Overlay PNG with alpha blending using cvzone.overlayPNG pattern.

        Args:
            background: Background image (BGR or BGRA)
            overlay: Overlay image with alpha channel (BGRA)
            position: (x, y) center position for overlay
            opacity: Opacity multiplier (0.0 to 1.0)
        """
        # Ensure background is BGR (3 channels)
        if len(background.shape) == 2:
            background = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)
        elif background.shape[2] == 4:
            background = cv2.cvtColor(background, cv2.COLOR_BGRA2BGR)

        # Ensure overlay has alpha channel
        if len(overlay.shape) == 2:
            overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGRA)
        elif overlay.shape[2] == 3:
            overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2BGRA)

        # Apply opacity to overlay alpha channel
        if opacity < 1.0:
            overlay = overlay.copy()
            overlay[:, :, 3] = (overlay[:, :, 3] * opacity).astype(np.uint8)

        oh, ow = overlay.shape[:2]
        bh, bw = background.shape[:2]

        # Calculate top-left corner (center the overlay at position)
        x = position[0] - ow // 2
        y = position[1] - oh // 2

        # Calculate the valid region to overlay
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(bw, x + ow), min(bh, y + oh)

        # Calculate corresponding region in overlay
        ox1, oy1 = max(0, -x), max(0, -y)
        ox2, oy2 = ox1 + (x2 - x1), oy1 + (y2 - y1)

        # Ensure valid dimensions
        if x2 <= x1 or y2 <= y1:
            return background

        # Extract ROI
        roi = background[y1:y2, x1:x2].copy()
        overlay_roi = overlay[oy1:oy2, ox1:ox2]

        # Alpha blending
        alpha = overlay_roi[:, :, 3:4].astype(np.float32) / 255.0
        overlay_rgb = overlay_roi[:, :, :3].astype(np.float32)
        roi_float = roi.astype(np.float32)

        # Blend: output = overlay * alpha + background * (1 - alpha)
        blended = overlay_rgb * alpha + roi_float * (1 - alpha)

        # Put back
        result = background.copy()
        result[y1:y2, x1:x2] = blended.astype(np.uint8)

        return result

    def _cvzone_overlay_png(self, background: np.ndarray, overlay: np.ndarray,
                            position: Tuple[int, int], opacity: float = 1.0) -> np.ndarray:
        """
        Use cvzone.overlayPNG for proper transparent overlay.
        Falls back to custom implementation if cvzone overlay fails.
        """
        try:
            if CVZONE_AVAILABLE and hasattr(cvzone, 'overlayPNG'):
                # Ensure overlay has alpha channel
                if overlay.shape[2] == 3:
                    overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2BGRA)

                # Apply opacity
                if opacity < 1.0:
                    overlay = overlay.copy()
                    overlay[:, :, 3] = (overlay[:, :, 3] * opacity).astype(np.uint8)

                oh, ow = overlay.shape[:2]
                x = position[0] - ow // 2
                y = position[1] - oh // 2

                # cvzone.overlayPNG expects top-left position
                result = cvzone.overlayPNG(background, overlay, [x, y])
                return result
        except Exception as e:
            logger.warning(f"cvzone.overlayPNG failed, using fallback: {e}")

        # Fallback to custom implementation
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
            # cvzone PoseDetector expects BGR image
            img, lmList = self.pose_detector.findPose(cv2_image.copy(), draw=False)

            if not lmList or len(lmList) < 13:
                return None

            # lmList format: [[id, x, y, z], ...]
            # Extract key landmarks for necklace positioning
            landmarks = {}

            # Convert list to dict for easier access
            lm_dict = {lm[0]: (lm[1], lm[2]) for lm in lmList if len(lm) >= 3}

            if self.POSE_LEFT_SHOULDER in lm_dict and self.POSE_RIGHT_SHOULDER in lm_dict:
                landmarks['left_shoulder'] = lm_dict[self.POSE_LEFT_SHOULDER]
                landmarks['right_shoulder'] = lm_dict[self.POSE_RIGHT_SHOULDER]
            else:
                return None

            # Optional landmarks
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

    def detect_face_landmarks(self, cv2_image: np.ndarray) -> Optional[Any]:
        """
        Detect face mesh landmarks using MediaPipe.

        Args:
            cv2_image: OpenCV image (BGR)

        Returns:
            MediaPipe face landmarks or None if detection failed
        """
        try:
            rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_image)

            if results.multi_face_landmarks and len(results.multi_face_landmarks) > 0:
                return results.multi_face_landmarks[0]
            return None

        except Exception as e:
            logger.error(f"Face detection error: {e}")
            return None

    def apply_necklace(self, person_image: Image.Image,
                       necklace_image: Image.Image,
                       opacity: float = 1.0) -> Tuple[Image.Image, str]:
        """
        Apply necklace to person image using pose landmarks 9, 10, 11, 12.

        Uses cvzone PoseDetector for accurate shoulder/neck positioning.
        Scales necklace based on shoulder width and rotates based on shoulder tilt.

        Args:
            person_image: PIL Image of the person
            necklace_image: PIL Image of the necklace (with transparent background)
            opacity: Opacity of the necklace overlay (0.0 to 1.0)

        Returns:
            Tuple of (result image, status message)
        """
        try:
            # Convert to OpenCV format
            cv2_person = self._pil_to_cv2(person_image)
            cv2_necklace = self._pil_to_cv2_rgba(necklace_image)

            h, w = cv2_person.shape[:2]

            # Detect pose landmarks
            pose_landmarks = self.detect_pose_landmarks(cv2_person)

            if pose_landmarks is None:
                return person_image, "Error: Could not detect body pose. Please use a clear front-facing photo with visible shoulders."

            # Get shoulder positions (landmarks 11, 12)
            left_shoulder = pose_landmarks['left_shoulder']
            right_shoulder = pose_landmarks['right_shoulder']

            # Calculate shoulder width
            shoulder_width = abs(right_shoulder[0] - left_shoulder[0])

            if shoulder_width < 50:
                return person_image, "Error: Shoulders not clearly visible. Please use a photo with visible shoulders."

            # Calculate neck/chest center position
            # Use average of shoulders, with optional mouth landmarks (9, 10) for better neck reference
            center_x = (left_shoulder[0] + right_shoulder[0]) // 2
            center_y = (left_shoulder[1] + right_shoulder[1]) // 2

            # If mouth landmarks available, use them for better vertical positioning
            if 'left_mouth' in pose_landmarks and 'right_mouth' in pose_landmarks:
                mouth_center_y = (pose_landmarks['left_mouth'][1] + pose_landmarks['right_mouth'][1]) // 2
                # Position necklace between mouth and shoulders
                neck_y = mouth_center_y + int((center_y - mouth_center_y) * 0.4)
                center_y = neck_y
            else:
                # Adjust upward from shoulders for chest placement
                center_y -= int(shoulder_width * 0.05)

            # Calculate shoulder tilt angle for rotation
            angle = self._calculate_shoulder_angle(left_shoulder, right_shoulder)

            # Scale necklace based on shoulder width
            necklace_width = int(shoulder_width * 1.2)  # 120% of shoulder width
            necklace_height = int(necklace_width * 0.7)  # Typical necklace aspect ratio

            # Resize necklace
            resized_necklace = self._resize_jewelry_cv2(cv2_necklace, necklace_width, necklace_height)

            # Rotate necklace to match shoulder angle
            if abs(angle) > 2:  # Only rotate if significant tilt
                rotated_necklace = self._rotate_jewelry_cv2(resized_necklace, angle)
            else:
                rotated_necklace = resized_necklace

            # Move position down slightly for necklace to sit on chest
            final_y = center_y + int(shoulder_width * 0.15)

            # Apply overlay using cvzone.overlayPNG
            result = self._cvzone_overlay_png(cv2_person, rotated_necklace,
                                               (center_x, final_y), opacity)

            # Convert back to PIL
            result_pil = self._cv2_to_pil(result)

            return result_pil, "Success: Necklace applied successfully!"

        except Exception as e:
            logger.error(f"Necklace application error: {str(e)}")
            return person_image, f"Error: {str(e)}"

    def apply_earrings(self, person_image: Image.Image,
                       earring_image: Image.Image,
                       opacity: float = 1.0) -> Tuple[Image.Image, str]:
        """
        Apply earrings to person image using face mesh ear landmarks.

        Single earring image is mirrored for left/right ears.
        Scales based on face width.

        Args:
            person_image: PIL Image of the person
            earring_image: PIL Image of a single earring
            opacity: Opacity of the earring overlay (0.0 to 1.0)

        Returns:
            Tuple of (result image, status message)
        """
        try:
            # Convert to OpenCV format
            cv2_person = self._pil_to_cv2(person_image)
            cv2_earring = self._pil_to_cv2_rgba(earring_image)

            h, w = cv2_person.shape[:2]

            # Detect face landmarks
            face_landmarks = self.detect_face_landmarks(cv2_person)

            if face_landmarks is None:
                return person_image, "Error: Could not detect face. Please use a clear front-facing photo."

            # Calculate face measurements
            face_width = self._calculate_face_width(face_landmarks, w, h)
            face_angle = self._calculate_face_angle(face_landmarks, w, h)

            # Get ear/earlobe positions
            # Use earlobe landmarks for more accurate earring placement
            left_ear_pos = self._get_face_landmark_point(face_landmarks, self.LEFT_EARLOBE, w, h)
            right_ear_pos = self._get_face_landmark_point(face_landmarks, self.RIGHT_EARLOBE, w, h)

            # Scale earrings relative to face size (20% of face width)
            earring_size = int(face_width * 0.20)
            earring_size = max(earring_size, 20)  # Minimum size

            # Resize earrings
            resized_earring = self._resize_jewelry_cv2(cv2_earring, earring_size, earring_size)

            # Rotate to match face angle
            rotated_left = self._rotate_jewelry_cv2(resized_earring, face_angle)

            # Mirror for right ear
            flipped_earring = cv2.flip(resized_earring, 1)  # Horizontal flip
            rotated_right = self._rotate_jewelry_cv2(flipped_earring, face_angle)

            # Adjust ear positions (slightly down and outward for earlobes)
            left_pos = (left_ear_pos[0] - int(face_width * 0.03),
                       left_ear_pos[1] + int(face_width * 0.08))
            right_pos = (right_ear_pos[0] + int(face_width * 0.03),
                        right_ear_pos[1] + int(face_width * 0.08))

            # Apply left earring
            result = self._cvzone_overlay_png(cv2_person, rotated_left, left_pos, opacity)

            # Apply right earring
            result = self._cvzone_overlay_png(result, rotated_right, right_pos, opacity)

            # Convert back to PIL
            result_pil = self._cv2_to_pil(result)

            return result_pil, "Success: Earrings applied successfully!"

        except Exception as e:
            logger.error(f"Earring application error: {str(e)}")
            return person_image, f"Error: {str(e)}"

    def apply_maang_tikka(self, person_image: Image.Image,
                          tikka_image: Image.Image,
                          opacity: float = 1.0) -> Tuple[Image.Image, str]:
        """
        Apply Maang Tikka (Indian forehead jewelry) to person image.

        Positioned at forehead center (hairline), scaled to face proportions.

        Args:
            person_image: PIL Image of the person
            tikka_image: PIL Image of the maang tikka
            opacity: Opacity of the overlay (0.0 to 1.0)

        Returns:
            Tuple of (result image, status message)
        """
        try:
            # Convert to OpenCV format
            cv2_person = self._pil_to_cv2(person_image)
            cv2_tikka = self._pil_to_cv2_rgba(tikka_image)

            h, w = cv2_person.shape[:2]

            # Detect face landmarks
            face_landmarks = self.detect_face_landmarks(cv2_person)

            if face_landmarks is None:
                return person_image, "Error: Could not detect face. Please use a clear front-facing photo."

            # Calculate face measurements
            face_width = self._calculate_face_width(face_landmarks, w, h)
            face_angle = self._calculate_face_angle(face_landmarks, w, h)

            # Get forehead center position (use hairline landmark)
            forehead_pos = self._get_face_average_point(face_landmarks,
                                                         self.FOREHEAD_CENTER_INDICES,
                                                         w, h)

            # Scale tikka relative to face size
            # Tikka is typically narrower but longer (chain extends into hair)
            tikka_width = int(face_width * 0.12)
            tikka_height = int(face_width * 0.25)
            tikka_width = max(tikka_width, 15)
            tikka_height = max(tikka_height, 30)

            # Resize tikka
            resized_tikka = self._resize_jewelry_cv2(cv2_tikka, tikka_width, tikka_height)

            # Rotate to match face angle
            rotated_tikka = self._rotate_jewelry_cv2(resized_tikka, face_angle)

            # Position slightly up from detected forehead center
            adjusted_pos = (forehead_pos[0], forehead_pos[1] - int(face_width * 0.01))

            # Apply overlay
            result = self._cvzone_overlay_png(cv2_person, rotated_tikka, adjusted_pos, opacity)

            # Convert back to PIL
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
        Apply nose ring to person image using nose landmarks.

        Args:
            person_image: PIL Image of the person
            nose_ring_image: PIL Image of the nose ring
            side: Which side to apply - "left", "right", or "septum"
            ring_style: Style of ring - "stud", "hoop", or "nath" (larger traditional)
            opacity: Opacity of the overlay (0.0 to 1.0)

        Returns:
            Tuple of (result image, status message)
        """
        try:
            # Convert to OpenCV format
            cv2_person = self._pil_to_cv2(person_image)
            cv2_ring = self._pil_to_cv2_rgba(nose_ring_image)

            h, w = cv2_person.shape[:2]

            # Detect face landmarks
            face_landmarks = self.detect_face_landmarks(cv2_person)

            if face_landmarks is None:
                return person_image, "Error: Could not detect face. Please use a clear front-facing photo."

            # Calculate face measurements
            face_width = self._calculate_face_width(face_landmarks, w, h)
            face_angle = self._calculate_face_angle(face_landmarks, w, h)

            # Get nose position based on side
            side_lower = side.lower()
            if side_lower == "septum":
                nose_pos = self._get_face_landmark_point(face_landmarks,
                                                          self.NOSE_SEPTUM, w, h)
            elif side_lower == "right":
                nose_pos = self._get_face_landmark_point(face_landmarks,
                                                          self.NOSE_RIGHT_NOSTRIL, w, h)
            else:  # Default to left
                nose_pos = self._get_face_landmark_point(face_landmarks,
                                                          self.NOSE_LEFT_NOSTRIL, w, h)

            # Determine ring size based on style
            if ring_style.lower() == "nath":
                # Nath is a larger traditional nose ring
                ring_size = int(face_width * 0.15)
            elif ring_style.lower() == "hoop":
                ring_size = int(face_width * 0.10)
            else:  # stud
                ring_size = int(face_width * 0.06)

            ring_size = max(ring_size, 10)  # Minimum size

            # Resize ring
            resized_ring = self._resize_jewelry_cv2(cv2_ring, ring_size, ring_size)

            # Rotate to match face angle
            rotated_ring = self._rotate_jewelry_cv2(resized_ring, face_angle)

            # Flip if right side (assuming jewelry image is for left side)
            if side_lower == "right":
                rotated_ring = cv2.flip(rotated_ring, 1)

            # Apply overlay
            result = self._cvzone_overlay_png(cv2_person, rotated_ring, nose_pos, opacity)

            # Convert back to PIL
            result_pil = self._cv2_to_pil(result)

            return result_pil, f"Success: Nose ring applied on {side} side!"

        except Exception as e:
            logger.error(f"Nose ring application error: {str(e)}")
            return person_image, f"Error: {str(e)}"

    def close(self):
        """Release resources."""
        if self._pose_detector is not None:
            self._pose_detector = None
        if self._face_mesh is not None:
            self._face_mesh.close()
            self._face_mesh = None


# Global engine instance for singleton pattern
_engine_instance: Optional[JewelryEngine] = None


def get_engine() -> JewelryEngine:
    """Get or create the global engine instance."""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = JewelryEngine()
    return _engine_instance


def remove_jewelry_background(jewelry_image: Union[Image.Image, np.ndarray]) -> Image.Image:
    """
    Remove background from jewelry image using rembg.

    Args:
        jewelry_image: PIL Image or numpy array of the jewelry

    Returns:
        PIL Image with transparent background (RGBA)
    """
    try:
        from rembg import remove

        # Convert to PIL if numpy
        if isinstance(jewelry_image, np.ndarray):
            jewelry_image = Image.fromarray(cv2.cvtColor(jewelry_image, cv2.COLOR_BGR2RGB))

        # Remove background
        result = remove(jewelry_image)
        return result

    except ImportError:
        logger.warning("rembg not installed. Install with: pip install rembg")
        if isinstance(jewelry_image, np.ndarray):
            jewelry_image = Image.fromarray(cv2.cvtColor(jewelry_image, cv2.COLOR_BGR2RGB))
        return jewelry_image.convert('RGBA')

    except Exception as e:
        logger.error(f"Background removal error: {str(e)}")
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
        jewelry_type: Type of jewelry - "necklace", "earrings", "maang_tikka", "nose_ring"
        opacity: Opacity of the jewelry overlay (0.0 to 1.0)
        **kwargs: Additional arguments (e.g., 'side' for nose ring, 'ring_style' for nose ring)

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
            # Handle both RGB and BGR numpy arrays
            if len(person_image.shape) == 3 and person_image.shape[2] == 3:
                # Assume RGB from Gradio
                person_image = Image.fromarray(person_image)
            elif len(person_image.shape) == 3 and person_image.shape[2] == 4:
                # RGBA
                person_image = Image.fromarray(person_image)
            else:
                person_image = Image.fromarray(person_image)

        # Convert jewelry image to PIL
        if isinstance(jewelry_image, str):
            jewelry_image = Image.open(jewelry_image)
        elif isinstance(jewelry_image, np.ndarray):
            if len(jewelry_image.shape) == 3 and jewelry_image.shape[2] == 3:
                jewelry_image = Image.fromarray(jewelry_image)
            elif len(jewelry_image.shape) == 3 and jewelry_image.shape[2] == 4:
                jewelry_image = Image.fromarray(jewelry_image)
            else:
                jewelry_image = Image.fromarray(jewelry_image)

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
        ring_style = kwargs.get("ring_style", "stud")
        return engine.apply_nose_ring(person_image, jewelry_image, side, ring_style, opacity)

    else:
        return person_image, f"Error: Unknown jewelry type '{jewelry_type}'. Supported: necklace, earrings, maang_tikka, nose_ring"


if __name__ == "__main__":
    # Test the engine with a simple check
    print("=" * 60)
    print("Naari Studio - Jewelry Virtual Try-On Engine")
    print("=" * 60)
    print(f"\ncvzone available: {CVZONE_AVAILABLE}")
    print(f"MediaPipe available: {MEDIAPIPE_AVAILABLE}")
    print("\nPose Landmarks Used (cvzone PoseDetector):")
    print("  - 9: left_mouth (neck reference)")
    print("  - 10: right_mouth (neck reference)")
    print("  - 11: left_shoulder (necklace positioning)")
    print("  - 12: right_shoulder (necklace positioning)")
    print("\nSupported jewelry types:")
    print("  - necklace: Uses pose landmarks for shoulder/chest placement")
    print("  - earrings: Uses face mesh for ear positioning")
    print("  - maang_tikka: Uses face mesh for forehead center")
    print("  - nose_ring: Uses face mesh for nostril positioning")
    print("\nUsage:")
    print("  from jewelry_engine import apply_jewelry, remove_jewelry_background")
    print("  result, message = apply_jewelry(person_img, jewelry_img, 'necklace', 0.9)")
    print("=" * 60)
