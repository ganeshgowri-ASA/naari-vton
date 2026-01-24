"""
Naari Studio - MediaPipe-based Jewelry Try-On Engine

Professional virtual jewelry try-on using MediaPipe Pose Landmarker and Face Mesh
for accurate anatomical positioning of necklaces, earrings, and other jewelry.

Author: Naari Studio (AnahataSri)
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any
from enum import Enum
import logging

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    logging.warning("MediaPipe not installed. Install with: pip install mediapipe")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class JewelryType(Enum):
    """Supported jewelry types."""
    NECKLACE = "necklace"
    EARRING_LEFT = "earring_left"
    EARRING_RIGHT = "earring_right"
    EARRING_PAIR = "earring_pair"
    NOSE_RING = "nose_ring"
    MAANG_TIKKA = "maang_tikka"
    BINDI = "bindi"


@dataclass
class LandmarkPoint:
    """Represents a 2D/3D landmark point with visibility."""
    x: float
    y: float
    z: float = 0.0
    visibility: float = 1.0

    def to_pixel(self, width: int, height: int) -> Tuple[int, int]:
        """Convert normalized coordinates to pixel coordinates."""
        return (int(self.x * width), int(self.y * height))


@dataclass
class JewelryPlacement:
    """Computed jewelry placement parameters."""
    center: Tuple[int, int]
    scale: float
    rotation_angle: float
    perspective_matrix: Optional[np.ndarray] = None
    anchor_points: Optional[Dict[str, Tuple[int, int]]] = None


class MediaPipeLandmarkDetector:
    """
    Handles MediaPipe Pose and Face Mesh landmark detection.

    Pose Landmarks used:
    - 11: Left shoulder
    - 12: Right shoulder
    - 0: Nose (for reference)

    Face Mesh Landmarks used:
    - 234: Right ear tragion
    - 454: Left ear tragion
    - 10: Forehead center (for maang tikka)
    - 6: Nose tip
    - 4: Nose bridge
    - 168: Forehead (bindi position)
    """

    # MediaPipe Pose landmark indices
    POSE_LEFT_SHOULDER = 11
    POSE_RIGHT_SHOULDER = 12
    POSE_LEFT_EAR = 7
    POSE_RIGHT_EAR = 8
    POSE_NOSE = 0

    # MediaPipe Face Mesh landmark indices
    FACE_RIGHT_EAR = 234
    FACE_LEFT_EAR = 454
    FACE_FOREHEAD_TOP = 10
    FACE_NOSE_TIP = 6
    FACE_NOSE_BRIDGE = 4
    FACE_BINDI = 168
    FACE_CHIN = 152
    FACE_LEFT_CHEEK = 234
    FACE_RIGHT_CHEEK = 454

    # Ear lobe landmarks (approximate)
    FACE_LEFT_EAR_LOBE = 323
    FACE_RIGHT_EAR_LOBE = 93

    def __init__(self,
                 pose_model_path: Optional[str] = None,
                 face_model_path: Optional[str] = None,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        """
        Initialize the landmark detector.

        Args:
            pose_model_path: Path to pose landmarker model (uses default if None)
            face_model_path: Path to face mesh model (uses default if None)
            min_detection_confidence: Minimum confidence for detection
            min_tracking_confidence: Minimum confidence for tracking
        """
        if not MEDIAPIPE_AVAILABLE:
            raise RuntimeError("MediaPipe is not installed. Install with: pip install mediapipe")

        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        # Initialize pose detector using legacy API (more widely available)
        self.mp_pose = mp.solutions.pose
        self.pose_detector = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

        # Initialize face mesh detector
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh_detector = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

        logger.info("MediaPipe detectors initialized successfully")

    def detect_pose(self, image: np.ndarray) -> Optional[List[LandmarkPoint]]:
        """
        Detect pose landmarks in an image.

        Args:
            image: BGR image array

        Returns:
            List of LandmarkPoint objects or None if detection fails
        """
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose_detector.process(image_rgb)

        if not results.pose_landmarks:
            logger.warning("No pose landmarks detected")
            return None

        landmarks = []
        for lm in results.pose_landmarks.landmark:
            landmarks.append(LandmarkPoint(
                x=lm.x,
                y=lm.y,
                z=lm.z,
                visibility=lm.visibility
            ))

        return landmarks

    def detect_face_mesh(self, image: np.ndarray) -> Optional[List[LandmarkPoint]]:
        """
        Detect face mesh landmarks in an image.

        Args:
            image: BGR image array

        Returns:
            List of LandmarkPoint objects or None if detection fails
        """
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh_detector.process(image_rgb)

        if not results.multi_face_landmarks:
            logger.warning("No face mesh landmarks detected")
            return None

        # Use first detected face
        face_landmarks = results.multi_face_landmarks[0]
        landmarks = []
        for lm in face_landmarks.landmark:
            landmarks.append(LandmarkPoint(
                x=lm.x,
                y=lm.y,
                z=lm.z
            ))

        return landmarks

    def get_shoulder_points(self, pose_landmarks: List[LandmarkPoint],
                           image_width: int, image_height: int) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        Extract shoulder pixel coordinates from pose landmarks.

        Args:
            pose_landmarks: List of pose landmarks
            image_width: Image width in pixels
            image_height: Image height in pixels

        Returns:
            Tuple of (left_shoulder, right_shoulder) pixel coordinates
        """
        if len(pose_landmarks) < max(self.POSE_LEFT_SHOULDER, self.POSE_RIGHT_SHOULDER) + 1:
            return None

        left = pose_landmarks[self.POSE_LEFT_SHOULDER]
        right = pose_landmarks[self.POSE_RIGHT_SHOULDER]

        # Check visibility
        if left.visibility < 0.5 or right.visibility < 0.5:
            logger.warning(f"Shoulder visibility too low: L={left.visibility:.2f}, R={right.visibility:.2f}")
            return None

        return (
            left.to_pixel(image_width, image_height),
            right.to_pixel(image_width, image_height)
        )

    def get_ear_points(self, face_landmarks: List[LandmarkPoint],
                      image_width: int, image_height: int) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        Extract ear lobe pixel coordinates from face mesh landmarks.

        Args:
            face_landmarks: List of face mesh landmarks
            image_width: Image width in pixels
            image_height: Image height in pixels

        Returns:
            Tuple of (left_ear, right_ear) pixel coordinates
        """
        if len(face_landmarks) < max(self.FACE_LEFT_EAR_LOBE, self.FACE_RIGHT_EAR_LOBE) + 1:
            return None

        left = face_landmarks[self.FACE_LEFT_EAR_LOBE]
        right = face_landmarks[self.FACE_RIGHT_EAR_LOBE]

        return (
            left.to_pixel(image_width, image_height),
            right.to_pixel(image_width, image_height)
        )

    def close(self):
        """Release detector resources."""
        self.pose_detector.close()
        self.face_mesh_detector.close()


class JewelryEngine:
    """
    Main jewelry try-on engine using MediaPipe for accurate placement.

    Supports:
    - Necklaces: Positioned at neck/chest area using shoulder landmarks
    - Earrings: Positioned at ear lobes using face mesh
    - Nose rings: Positioned at nose using face mesh
    - Maang tikka/Bindi: Positioned on forehead using face mesh
    """

    def __init__(self,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        """
        Initialize the jewelry engine.

        Args:
            min_detection_confidence: Minimum confidence for landmark detection
            min_tracking_confidence: Minimum confidence for landmark tracking
        """
        self.detector = MediaPipeLandmarkDetector(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self._cached_pose_landmarks = None
        self._cached_face_landmarks = None
        self._cached_image_shape = None

    def _detect_landmarks(self, image: np.ndarray, force_refresh: bool = False):
        """
        Detect and cache landmarks for an image.

        Args:
            image: BGR image array
            force_refresh: Force re-detection even if cached
        """
        current_shape = image.shape

        if (force_refresh or
            self._cached_image_shape != current_shape or
            self._cached_pose_landmarks is None):

            self._cached_pose_landmarks = self.detector.detect_pose(image)
            self._cached_face_landmarks = self.detector.detect_face_mesh(image)
            self._cached_image_shape = current_shape

    def calculate_necklace_placement(self,
                                     image: np.ndarray,
                                     necklace_drop_ratio: float = 0.15,
                                     size_ratio: float = 0.7) -> Optional[JewelryPlacement]:
        """
        Calculate optimal necklace placement using shoulder landmarks.

        The necklace center is positioned between the shoulders, slightly below
        the shoulder line to sit at the natural necklace position on the chest.

        Args:
            image: BGR image array
            necklace_drop_ratio: How far below shoulders to place necklace (0.0-1.0)
                                 as a ratio of shoulder width
            size_ratio: Necklace width as ratio of shoulder width (0.5-1.5)

        Returns:
            JewelryPlacement object with computed parameters
        """
        self._detect_landmarks(image)

        if self._cached_pose_landmarks is None:
            logger.error("Cannot calculate necklace placement: no pose landmarks")
            return None

        h, w = image.shape[:2]

        # Get shoulder points
        shoulder_points = self.detector.get_shoulder_points(
            self._cached_pose_landmarks, w, h
        )

        if shoulder_points is None:
            logger.error("Cannot calculate necklace placement: shoulders not visible")
            return None

        left_shoulder, right_shoulder = shoulder_points

        # Calculate shoulder midpoint and width
        shoulder_mid_x = (left_shoulder[0] + right_shoulder[0]) // 2
        shoulder_mid_y = (left_shoulder[1] + right_shoulder[1]) // 2
        shoulder_width = abs(right_shoulder[0] - left_shoulder[0])

        # Calculate necklace center (below shoulder midpoint)
        necklace_drop = int(shoulder_width * necklace_drop_ratio)
        necklace_center = (shoulder_mid_x, shoulder_mid_y + necklace_drop)

        # Calculate rotation angle from shoulder tilt
        dy = right_shoulder[1] - left_shoulder[1]
        dx = right_shoulder[0] - left_shoulder[0]
        rotation_angle = np.degrees(np.arctan2(dy, dx))

        # Calculate scale based on shoulder width
        scale = (shoulder_width * size_ratio) / 100.0  # Assuming 100px reference

        # Calculate perspective transformation if face landmarks available
        perspective_matrix = None
        if self._cached_face_landmarks is not None:
            perspective_matrix = self._calculate_perspective_transform(
                self._cached_face_landmarks, w, h
            )

        return JewelryPlacement(
            center=necklace_center,
            scale=scale,
            rotation_angle=rotation_angle,
            perspective_matrix=perspective_matrix,
            anchor_points={
                'left_shoulder': left_shoulder,
                'right_shoulder': right_shoulder,
                'shoulder_midpoint': (shoulder_mid_x, shoulder_mid_y)
            }
        )

    def calculate_earring_placement(self,
                                    image: np.ndarray,
                                    earring_type: JewelryType = JewelryType.EARRING_PAIR,
                                    drop_offset: int = 5,
                                    size_ratio: float = 0.08) -> Optional[Dict[str, JewelryPlacement]]:
        """
        Calculate optimal earring placement using face mesh ear landmarks.

        Args:
            image: BGR image array
            earring_type: Which earring(s) to place
            drop_offset: Pixels below ear lobe to place earring
            size_ratio: Earring size as ratio of face width

        Returns:
            Dictionary with 'left' and/or 'right' JewelryPlacement objects
        """
        self._detect_landmarks(image)

        if self._cached_face_landmarks is None:
            logger.error("Cannot calculate earring placement: no face landmarks")
            return None

        h, w = image.shape[:2]

        # Get ear points
        ear_points = self.detector.get_ear_points(
            self._cached_face_landmarks, w, h
        )

        if ear_points is None:
            logger.error("Cannot calculate earring placement: ears not visible")
            return None

        left_ear, right_ear = ear_points

        # Calculate face width for scaling
        face_width = abs(left_ear[0] - right_ear[0])
        scale = (face_width * size_ratio) / 50.0  # Assuming 50px reference

        # Calculate head tilt from face landmarks
        rotation_angle = self._calculate_head_tilt(
            self._cached_face_landmarks, w, h
        )

        result = {}

        if earring_type in [JewelryType.EARRING_LEFT, JewelryType.EARRING_PAIR]:
            result['left'] = JewelryPlacement(
                center=(left_ear[0], left_ear[1] + drop_offset),
                scale=scale,
                rotation_angle=rotation_angle,
                anchor_points={'ear_lobe': left_ear}
            )

        if earring_type in [JewelryType.EARRING_RIGHT, JewelryType.EARRING_PAIR]:
            result['right'] = JewelryPlacement(
                center=(right_ear[0], right_ear[1] + drop_offset),
                scale=scale,
                rotation_angle=rotation_angle,
                anchor_points={'ear_lobe': right_ear}
            )

        return result if result else None

    def calculate_forehead_placement(self,
                                     image: np.ndarray,
                                     jewelry_type: JewelryType = JewelryType.BINDI,
                                     size_ratio: float = 0.03) -> Optional[JewelryPlacement]:
        """
        Calculate optimal forehead jewelry placement (bindi/maang tikka).

        Args:
            image: BGR image array
            jewelry_type: BINDI or MAANG_TIKKA
            size_ratio: Jewelry size as ratio of face width

        Returns:
            JewelryPlacement object with computed parameters
        """
        self._detect_landmarks(image)

        if self._cached_face_landmarks is None:
            logger.error("Cannot calculate forehead placement: no face landmarks")
            return None

        h, w = image.shape[:2]

        # Get forehead point
        if jewelry_type == JewelryType.MAANG_TIKKA:
            # Maang tikka at hairline
            landmark_idx = MediaPipeLandmarkDetector.FACE_FOREHEAD_TOP
        else:
            # Bindi between eyebrows
            landmark_idx = MediaPipeLandmarkDetector.FACE_BINDI

        forehead_point = self._cached_face_landmarks[landmark_idx].to_pixel(w, h)

        # Calculate face width for scaling
        ear_points = self.detector.get_ear_points(
            self._cached_face_landmarks, w, h
        )
        face_width = abs(ear_points[0][0] - ear_points[1][0]) if ear_points else w // 3

        scale = (face_width * size_ratio) / 20.0  # Assuming 20px reference

        rotation_angle = self._calculate_head_tilt(
            self._cached_face_landmarks, w, h
        )

        return JewelryPlacement(
            center=forehead_point,
            scale=scale,
            rotation_angle=rotation_angle,
            anchor_points={'forehead': forehead_point}
        )

    def calculate_nose_ring_placement(self,
                                      image: np.ndarray,
                                      side: str = 'left',
                                      size_ratio: float = 0.025) -> Optional[JewelryPlacement]:
        """
        Calculate optimal nose ring placement.

        Args:
            image: BGR image array
            side: 'left' or 'right' nostril
            size_ratio: Ring size as ratio of face width

        Returns:
            JewelryPlacement object with computed parameters
        """
        self._detect_landmarks(image)

        if self._cached_face_landmarks is None:
            logger.error("Cannot calculate nose ring placement: no face landmarks")
            return None

        h, w = image.shape[:2]

        # Nose tip and nostril landmarks
        nose_tip = self._cached_face_landmarks[MediaPipeLandmarkDetector.FACE_NOSE_TIP].to_pixel(w, h)

        # MediaPipe face mesh nostril landmarks
        LEFT_NOSTRIL = 129
        RIGHT_NOSTRIL = 358

        nostril_idx = LEFT_NOSTRIL if side == 'left' else RIGHT_NOSTRIL
        nostril_point = self._cached_face_landmarks[nostril_idx].to_pixel(w, h)

        # Calculate face width for scaling
        ear_points = self.detector.get_ear_points(
            self._cached_face_landmarks, w, h
        )
        face_width = abs(ear_points[0][0] - ear_points[1][0]) if ear_points else w // 3

        scale = (face_width * size_ratio) / 15.0

        rotation_angle = self._calculate_head_tilt(
            self._cached_face_landmarks, w, h
        )

        return JewelryPlacement(
            center=nostril_point,
            scale=scale,
            rotation_angle=rotation_angle,
            anchor_points={'nostril': nostril_point, 'nose_tip': nose_tip}
        )

    def _calculate_head_tilt(self, face_landmarks: List[LandmarkPoint],
                             image_width: int, image_height: int) -> float:
        """Calculate head tilt angle from face landmarks."""
        # Use eye corners for tilt calculation
        LEFT_EYE_OUTER = 263
        RIGHT_EYE_OUTER = 33

        left_eye = face_landmarks[LEFT_EYE_OUTER].to_pixel(image_width, image_height)
        right_eye = face_landmarks[RIGHT_EYE_OUTER].to_pixel(image_width, image_height)

        dy = left_eye[1] - right_eye[1]
        dx = left_eye[0] - right_eye[0]

        return np.degrees(np.arctan2(dy, dx))

    def _calculate_perspective_transform(self, face_landmarks: List[LandmarkPoint],
                                         image_width: int, image_height: int) -> Optional[np.ndarray]:
        """
        Calculate perspective transformation matrix based on face orientation.

        Uses face landmarks to estimate 3D head pose and create appropriate
        perspective warp for jewelry overlay.
        """
        # Get key face points for perspective estimation
        nose_tip = face_landmarks[MediaPipeLandmarkDetector.FACE_NOSE_TIP]
        chin = face_landmarks[MediaPipeLandmarkDetector.FACE_CHIN]
        forehead = face_landmarks[MediaPipeLandmarkDetector.FACE_FOREHEAD_TOP]

        # Calculate face depth from z coordinates (relative)
        face_depth = abs(nose_tip.z)

        # If face is nearly frontal, no perspective transform needed
        if face_depth < 0.05:
            return None

        # Create subtle perspective based on face orientation
        # This is a simplified perspective - for production use,
        # consider full 3D pose estimation

        left_ear = face_landmarks[MediaPipeLandmarkDetector.FACE_LEFT_EAR]
        right_ear = face_landmarks[MediaPipeLandmarkDetector.FACE_RIGHT_EAR]

        # Determine if face is turned left or right
        z_diff = left_ear.z - right_ear.z

        # Create perspective points for transformation
        # This creates a subtle keystone effect matching head turn
        src_points = np.float32([
            [0, 0], [100, 0], [100, 100], [0, 100]
        ])

        offset = int(z_diff * 20)  # Scale factor for perspective
        dst_points = np.float32([
            [offset, offset//2], [100-offset, -offset//2],
            [100-offset, 100+offset//2], [offset, 100-offset//2]
        ])

        return cv2.getPerspectiveTransform(src_points, dst_points)

    def apply_jewelry(self,
                      person_image: np.ndarray,
                      jewelry_image: np.ndarray,
                      placement: JewelryPlacement,
                      blend_mode: str = 'alpha') -> np.ndarray:
        """
        Apply jewelry image to person image at calculated placement.

        Args:
            person_image: BGR image of person
            jewelry_image: BGRA image of jewelry (with alpha channel)
            placement: Calculated JewelryPlacement parameters
            blend_mode: 'alpha' for standard alpha blend, 'soft_light' for subtle blend

        Returns:
            Composite image with jewelry applied
        """
        result = person_image.copy()

        # Ensure jewelry image has alpha channel
        if jewelry_image.shape[2] == 3:
            jewelry_image = cv2.cvtColor(jewelry_image, cv2.COLOR_BGR2BGRA)

        # Scale jewelry
        jewelry_h, jewelry_w = jewelry_image.shape[:2]
        new_w = int(jewelry_w * placement.scale)
        new_h = int(jewelry_h * placement.scale)

        if new_w <= 0 or new_h <= 0:
            logger.warning("Jewelry scale too small, skipping")
            return result

        scaled_jewelry = cv2.resize(jewelry_image, (new_w, new_h),
                                    interpolation=cv2.INTER_LANCZOS4)

        # Apply rotation
        if abs(placement.rotation_angle) > 0.5:
            center = (new_w // 2, new_h // 2)
            rotation_matrix = cv2.getRotationMatrix2D(
                center, -placement.rotation_angle, 1.0
            )

            # Calculate new bounding box size
            cos = abs(rotation_matrix[0, 0])
            sin = abs(rotation_matrix[0, 1])
            new_w_rot = int(new_h * sin + new_w * cos)
            new_h_rot = int(new_h * cos + new_w * sin)

            # Adjust rotation matrix for new size
            rotation_matrix[0, 2] += (new_w_rot - new_w) / 2
            rotation_matrix[1, 2] += (new_h_rot - new_h) / 2

            scaled_jewelry = cv2.warpAffine(
                scaled_jewelry, rotation_matrix, (new_w_rot, new_h_rot),
                flags=cv2.INTER_LANCZOS4,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0, 0)
            )
            new_w, new_h = new_w_rot, new_h_rot

        # Apply perspective transform if available
        if placement.perspective_matrix is not None:
            scaled_jewelry = cv2.warpPerspective(
                scaled_jewelry,
                placement.perspective_matrix,
                (new_w, new_h),
                flags=cv2.INTER_LANCZOS4,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0, 0)
            )

        # Calculate overlay position (centered on placement point)
        x = placement.center[0] - new_w // 2
        y = placement.center[1] - new_h // 2

        # Blend jewelry onto person image
        result = self._alpha_blend(result, scaled_jewelry, x, y, blend_mode)

        return result

    def _alpha_blend(self,
                     background: np.ndarray,
                     overlay: np.ndarray,
                     x: int, y: int,
                     blend_mode: str = 'alpha') -> np.ndarray:
        """
        Alpha blend overlay onto background at position (x, y).

        Handles edge cases where overlay extends beyond image boundaries.
        """
        result = background.copy()
        bg_h, bg_w = background.shape[:2]
        ov_h, ov_w = overlay.shape[:2]

        # Calculate valid overlay region
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(bg_w, x + ov_w), min(bg_h, y + ov_h)

        # Calculate corresponding overlay region
        ox1 = x1 - x
        oy1 = y1 - y
        ox2 = ox1 + (x2 - x1)
        oy2 = oy1 + (y2 - y1)

        if x2 <= x1 or y2 <= y1:
            return result  # No overlap

        # Extract regions
        bg_region = result[y1:y2, x1:x2]
        ov_region = overlay[oy1:oy2, ox1:ox2]

        # Extract alpha channel and normalize
        alpha = ov_region[:, :, 3:4].astype(float) / 255.0

        if blend_mode == 'soft_light':
            # Soft light blending for subtle jewelry effects
            alpha = alpha * 0.85  # Reduce opacity slightly

        # Perform alpha blending
        ov_rgb = ov_region[:, :, :3].astype(float)
        bg_rgb = bg_region.astype(float)

        blended = (ov_rgb * alpha + bg_rgb * (1 - alpha)).astype(np.uint8)
        result[y1:y2, x1:x2] = blended

        return result

    def try_on_necklace(self,
                        person_image: np.ndarray,
                        necklace_image: np.ndarray,
                        drop_ratio: float = 0.15,
                        size_ratio: float = 0.7) -> Tuple[Optional[np.ndarray], Optional[JewelryPlacement]]:
        """
        Complete necklace try-on pipeline.

        Args:
            person_image: BGR image of person
            necklace_image: BGRA image of necklace
            drop_ratio: How far below shoulders to place necklace
            size_ratio: Necklace width as ratio of shoulder width

        Returns:
            Tuple of (result_image, placement) or (None, None) if failed
        """
        placement = self.calculate_necklace_placement(
            person_image,
            necklace_drop_ratio=drop_ratio,
            size_ratio=size_ratio
        )

        if placement is None:
            return None, None

        result = self.apply_jewelry(person_image, necklace_image, placement)
        return result, placement

    def try_on_earrings(self,
                        person_image: np.ndarray,
                        earring_image: np.ndarray,
                        earring_type: JewelryType = JewelryType.EARRING_PAIR) -> Tuple[Optional[np.ndarray], Optional[Dict[str, JewelryPlacement]]]:
        """
        Complete earring try-on pipeline.

        Args:
            person_image: BGR image of person
            earring_image: BGRA image of earring
            earring_type: Which earring(s) to apply

        Returns:
            Tuple of (result_image, placements_dict) or (None, None) if failed
        """
        placements = self.calculate_earring_placement(
            person_image,
            earring_type=earring_type
        )

        if placements is None:
            return None, None

        result = person_image.copy()

        for side, placement in placements.items():
            # Mirror earring for left side
            if side == 'left':
                earring_to_use = cv2.flip(earring_image, 1)
            else:
                earring_to_use = earring_image

            result = self.apply_jewelry(result, earring_to_use, placement)

        return result, placements

    def try_on_complete_set(self,
                            person_image: np.ndarray,
                            jewelry_set: Dict[JewelryType, np.ndarray]) -> Tuple[np.ndarray, Dict[JewelryType, JewelryPlacement]]:
        """
        Apply a complete jewelry set (necklace, earrings, etc.)

        Args:
            person_image: BGR image of person
            jewelry_set: Dictionary mapping JewelryType to jewelry images

        Returns:
            Tuple of (result_image, placements_dict)
        """
        result = person_image.copy()
        all_placements = {}

        # Apply in order: necklace first, then earrings, then other
        order = [
            JewelryType.NECKLACE,
            JewelryType.EARRING_PAIR,
            JewelryType.EARRING_LEFT,
            JewelryType.EARRING_RIGHT,
            JewelryType.MAANG_TIKKA,
            JewelryType.BINDI,
            JewelryType.NOSE_RING
        ]

        for jewelry_type in order:
            if jewelry_type not in jewelry_set:
                continue

            jewelry_image = jewelry_set[jewelry_type]

            if jewelry_type == JewelryType.NECKLACE:
                result, placement = self.try_on_necklace(result, jewelry_image)
                if placement:
                    all_placements[jewelry_type] = placement

            elif jewelry_type in [JewelryType.EARRING_PAIR, JewelryType.EARRING_LEFT, JewelryType.EARRING_RIGHT]:
                result, placements = self.try_on_earrings(result, jewelry_image, jewelry_type)
                if placements:
                    all_placements[jewelry_type] = placements

            elif jewelry_type == JewelryType.BINDI:
                placement = self.calculate_forehead_placement(result, JewelryType.BINDI)
                if placement:
                    result = self.apply_jewelry(result, jewelry_image, placement)
                    all_placements[jewelry_type] = placement

            elif jewelry_type == JewelryType.MAANG_TIKKA:
                placement = self.calculate_forehead_placement(result, JewelryType.MAANG_TIKKA)
                if placement:
                    result = self.apply_jewelry(result, jewelry_image, placement)
                    all_placements[jewelry_type] = placement

            elif jewelry_type == JewelryType.NOSE_RING:
                placement = self.calculate_nose_ring_placement(result)
                if placement:
                    result = self.apply_jewelry(result, jewelry_image, placement)
                    all_placements[jewelry_type] = placement

        return result, all_placements

    def visualize_landmarks(self,
                           image: np.ndarray,
                           show_pose: bool = True,
                           show_face: bool = True,
                           show_jewelry_points: bool = True) -> np.ndarray:
        """
        Visualize detected landmarks on image (useful for debugging).

        Args:
            image: BGR image
            show_pose: Draw pose landmarks
            show_face: Draw face mesh
            show_jewelry_points: Draw jewelry anchor points

        Returns:
            Annotated image
        """
        self._detect_landmarks(image)
        result = image.copy()
        h, w = image.shape[:2]

        if show_pose and self._cached_pose_landmarks:
            # Draw shoulder points
            shoulders = self.detector.get_shoulder_points(
                self._cached_pose_landmarks, w, h
            )
            if shoulders:
                left, right = shoulders
                cv2.circle(result, left, 8, (0, 255, 0), -1)  # Green
                cv2.circle(result, right, 8, (0, 255, 0), -1)
                cv2.line(result, left, right, (0, 255, 0), 2)

                # Draw shoulder midpoint
                mid = ((left[0] + right[0]) // 2, (left[1] + right[1]) // 2)
                cv2.circle(result, mid, 6, (255, 0, 0), -1)  # Blue

        if show_face and self._cached_face_landmarks:
            # Draw ear points
            ears = self.detector.get_ear_points(
                self._cached_face_landmarks, w, h
            )
            if ears:
                left, right = ears
                cv2.circle(result, left, 6, (0, 0, 255), -1)  # Red
                cv2.circle(result, right, 6, (0, 0, 255), -1)

            # Draw forehead point
            forehead = self._cached_face_landmarks[
                MediaPipeLandmarkDetector.FACE_BINDI
            ].to_pixel(w, h)
            cv2.circle(result, forehead, 5, (255, 255, 0), -1)  # Cyan

        if show_jewelry_points:
            # Draw necklace placement point
            placement = self.calculate_necklace_placement(image)
            if placement:
                cv2.circle(result, placement.center, 10, (0, 255, 255), 3)  # Yellow
                cv2.putText(result, "Necklace",
                           (placement.center[0] - 40, placement.center[1] + 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        return result

    def close(self):
        """Release resources."""
        self.detector.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def create_jewelry_engine(**kwargs) -> JewelryEngine:
    """Factory function to create a JewelryEngine instance."""
    return JewelryEngine(**kwargs)


# Example usage and testing
if __name__ == "__main__":
    import sys

    # Simple test
    print("Naari Studio Jewelry Engine")
    print("=" * 40)
    print(f"MediaPipe available: {MEDIAPIPE_AVAILABLE}")

    if len(sys.argv) < 2:
        print("\nUsage: python jewelry_engine.py <person_image> [necklace_image]")
        print("\nThis module provides:")
        print("  - JewelryEngine: Main try-on engine")
        print("  - JewelryType: Enum of supported jewelry types")
        print("  - JewelryPlacement: Dataclass for placement parameters")
        sys.exit(0)

    # Load and process test image
    person_path = sys.argv[1]
    person_img = cv2.imread(person_path)

    if person_img is None:
        print(f"Error: Could not load image: {person_path}")
        sys.exit(1)

    print(f"\nProcessing: {person_path}")
    print(f"Image size: {person_img.shape[1]}x{person_img.shape[0]}")

    with JewelryEngine() as engine:
        # Visualize landmarks
        viz = engine.visualize_landmarks(person_img)
        cv2.imwrite("landmarks_debug.jpg", viz)
        print("Saved landmark visualization to: landmarks_debug.jpg")

        # Calculate necklace placement
        placement = engine.calculate_necklace_placement(person_img)
        if placement:
            print(f"\nNecklace placement calculated:")
            print(f"  Center: {placement.center}")
            print(f"  Scale: {placement.scale:.3f}")
            print(f"  Rotation: {placement.rotation_angle:.2f} degrees")
            if placement.anchor_points:
                print(f"  Anchors: {placement.anchor_points}")
        else:
            print("\nCould not calculate necklace placement (landmarks not detected)")

        # If necklace image provided, do full try-on
        if len(sys.argv) >= 3:
            necklace_path = sys.argv[2]
            necklace_img = cv2.imread(necklace_path, cv2.IMREAD_UNCHANGED)

            if necklace_img is not None:
                result, _ = engine.try_on_necklace(person_img, necklace_img)
                if result is not None:
                    cv2.imwrite("tryon_result.jpg", result)
                    print(f"\nSaved try-on result to: tryon_result.jpg")
            else:
                print(f"Error: Could not load necklace image: {necklace_path}")
