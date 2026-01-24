"""
Hybrid Jewelry Engine

This module combines MediaPipe detection, jewelry processing, and Imagen rendering
to create a complete photorealistic jewelry try-on system.

Features:
- Body landmark detection using MediaPipe
- Jewelry processing and feature extraction
- Region mask creation for inpainting
- Photorealistic rendering with fallback support
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple, List, Dict, Any, Union

import numpy as np
from PIL import Image, ImageDraw

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    logging.warning("MediaPipe not available. Pose detection will be disabled.")

from jewelry_processor import (
    JewelryProcessor,
    JewelryType,
    ProcessedJewelry,
    JewelryFeatures
)
from imagen_renderer import (
    ImagenRenderer,
    RenderConfig,
    RenderMode,
    RenderResult,
    JewelryPlacement
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BodyLandmark(Enum):
    """Key body landmarks for jewelry placement."""
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22


@dataclass
class DetectedLandmarks:
    """Container for detected body landmarks."""
    landmarks: Dict[BodyLandmark, Tuple[float, float, float]]
    confidence: float
    image_size: Tuple[int, int]

    def get_pixel_coords(self, landmark: BodyLandmark) -> Optional[Tuple[int, int]]:
        """Convert normalized landmark to pixel coordinates."""
        if landmark not in self.landmarks:
            return None
        x, y, _ = self.landmarks[landmark]
        return (int(x * self.image_size[0]), int(y * self.image_size[1]))


@dataclass
class PlacementConfig:
    """Configuration for jewelry placement."""
    jewelry_type: JewelryType
    primary_landmark: BodyLandmark
    secondary_landmarks: List[BodyLandmark] = field(default_factory=list)
    offset: Tuple[int, int] = (0, 0)
    scale_factor: float = 1.0
    rotation: float = 0.0
    region_padding: int = 20


@dataclass
class TryOnResult:
    """Result of the jewelry try-on operation."""
    success: bool
    result_image: Optional[Image.Image]
    jewelry_type: JewelryType
    placement_info: Dict[str, Any]
    render_mode: RenderMode
    error_message: Optional[str] = None


class HybridJewelryEngine:
    """
    Complete jewelry try-on engine combining detection, processing, and rendering.
    """

    # Default placement configurations for each jewelry type
    DEFAULT_PLACEMENTS = {
        JewelryType.NECKLACE: PlacementConfig(
            jewelry_type=JewelryType.NECKLACE,
            primary_landmark=BodyLandmark.LEFT_SHOULDER,
            secondary_landmarks=[BodyLandmark.RIGHT_SHOULDER],
            offset=(0, -30),
            scale_factor=1.2,
            region_padding=40
        ),
        JewelryType.EARRING: PlacementConfig(
            jewelry_type=JewelryType.EARRING,
            primary_landmark=BodyLandmark.LEFT_EAR,
            secondary_landmarks=[BodyLandmark.RIGHT_EAR],
            offset=(0, 10),
            scale_factor=0.8,
            region_padding=25
        ),
        JewelryType.BANGLE: PlacementConfig(
            jewelry_type=JewelryType.BANGLE,
            primary_landmark=BodyLandmark.LEFT_WRIST,
            secondary_landmarks=[],
            offset=(0, 0),
            scale_factor=1.0,
            region_padding=30
        ),
        JewelryType.BRACELET: PlacementConfig(
            jewelry_type=JewelryType.BRACELET,
            primary_landmark=BodyLandmark.RIGHT_WRIST,
            secondary_landmarks=[],
            offset=(0, 0),
            scale_factor=1.0,
            region_padding=30
        ),
        JewelryType.RING: PlacementConfig(
            jewelry_type=JewelryType.RING,
            primary_landmark=BodyLandmark.LEFT_INDEX,
            secondary_landmarks=[],
            offset=(0, 0),
            scale_factor=0.5,
            region_padding=15
        ),
        JewelryType.PENDANT: PlacementConfig(
            jewelry_type=JewelryType.PENDANT,
            primary_landmark=BodyLandmark.LEFT_SHOULDER,
            secondary_landmarks=[BodyLandmark.RIGHT_SHOULDER],
            offset=(0, 50),
            scale_factor=0.8,
            region_padding=35
        )
    }

    def __init__(
        self,
        google_api_key: Optional[str] = None,
        render_mode: str = "hybrid",
        enable_gpu: bool = False
    ):
        """
        Initialize the hybrid jewelry engine.

        Args:
            google_api_key: API key for Google Imagen
            render_mode: Rendering mode ('imagen_inpaint', 'overlay', 'hybrid')
            enable_gpu: Enable GPU acceleration where available
        """
        self.enable_gpu = enable_gpu

        # Initialize components
        self.jewelry_processor = JewelryProcessor(enable_gpu=enable_gpu)

        render_mode_enum = {
            'imagen_inpaint': RenderMode.IMAGEN_INPAINT,
            'overlay': RenderMode.OVERLAY,
            'hybrid': RenderMode.HYBRID
        }.get(render_mode.lower(), RenderMode.HYBRID)

        self.renderer = ImagenRenderer(
            api_key=google_api_key,
            config=RenderConfig(mode=render_mode_enum)
        )

        # Initialize MediaPipe if available
        self.pose_detector = None
        self.face_mesh = None
        if MEDIAPIPE_AVAILABLE:
            self._init_mediapipe()

        logger.info(f"HybridJewelryEngine initialized with mode: {render_mode}")

    def _init_mediapipe(self) -> None:
        """Initialize MediaPipe components."""
        try:
            # Initialize pose detector for body landmarks
            self.mp_pose = mp.solutions.pose
            self.pose_detector = self.mp_pose.Pose(
                static_image_mode=True,
                model_complexity=2,
                enable_segmentation=False,
                min_detection_confidence=0.5
            )

            # Initialize face mesh for detailed face landmarks (ears, etc.)
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5
            )

            logger.info("MediaPipe components initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize MediaPipe: {e}")
            self.pose_detector = None
            self.face_mesh = None

    def detect_landmarks(self, person_image: Image.Image) -> Optional[DetectedLandmarks]:
        """
        Detect body landmarks in a person image.

        Args:
            person_image: PIL Image of a person

        Returns:
            DetectedLandmarks object or None if detection fails
        """
        if not MEDIAPIPE_AVAILABLE or self.pose_detector is None:
            logger.warning("MediaPipe not available, using fallback detection")
            return self._fallback_landmark_detection(person_image)

        try:
            # Convert to RGB numpy array
            image_rgb = np.array(person_image.convert('RGB'))

            # Run pose detection
            results = self.pose_detector.process(image_rgb)

            if results.pose_landmarks is None:
                logger.warning("No pose landmarks detected")
                return self._fallback_landmark_detection(person_image)

            # Extract relevant landmarks
            landmarks = {}
            pose_landmarks = results.pose_landmarks.landmark

            # Map MediaPipe landmarks to our BodyLandmark enum
            landmark_mapping = {
                0: BodyLandmark.NOSE,
                2: BodyLandmark.LEFT_EYE,
                5: BodyLandmark.RIGHT_EYE,
                7: BodyLandmark.LEFT_EAR,
                8: BodyLandmark.RIGHT_EAR,
                11: BodyLandmark.LEFT_SHOULDER,
                12: BodyLandmark.RIGHT_SHOULDER,
                13: BodyLandmark.LEFT_ELBOW,
                14: BodyLandmark.RIGHT_ELBOW,
                15: BodyLandmark.LEFT_WRIST,
                16: BodyLandmark.RIGHT_WRIST,
                17: BodyLandmark.LEFT_PINKY,
                18: BodyLandmark.RIGHT_PINKY,
                19: BodyLandmark.LEFT_INDEX,
                20: BodyLandmark.RIGHT_INDEX,
                21: BodyLandmark.LEFT_THUMB,
                22: BodyLandmark.RIGHT_THUMB
            }

            for mp_idx, body_landmark in landmark_mapping.items():
                if mp_idx < len(pose_landmarks):
                    lm = pose_landmarks[mp_idx]
                    landmarks[body_landmark] = (lm.x, lm.y, lm.visibility)

            # Calculate average confidence
            confidences = [lm[2] for lm in landmarks.values()]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0

            return DetectedLandmarks(
                landmarks=landmarks,
                confidence=avg_confidence,
                image_size=person_image.size
            )

        except Exception as e:
            logger.error(f"Landmark detection error: {e}")
            return self._fallback_landmark_detection(person_image)

    def _fallback_landmark_detection(
        self,
        person_image: Image.Image
    ) -> DetectedLandmarks:
        """
        Fallback landmark detection using image proportions.

        Args:
            person_image: PIL Image of a person

        Returns:
            Estimated DetectedLandmarks
        """
        width, height = person_image.size

        # Estimate landmarks based on typical body proportions
        # These are rough estimates for a front-facing portrait
        landmarks = {
            BodyLandmark.NOSE: (0.5, 0.15, 0.5),
            BodyLandmark.LEFT_EYE: (0.45, 0.12, 0.5),
            BodyLandmark.RIGHT_EYE: (0.55, 0.12, 0.5),
            BodyLandmark.LEFT_EAR: (0.38, 0.14, 0.5),
            BodyLandmark.RIGHT_EAR: (0.62, 0.14, 0.5),
            BodyLandmark.LEFT_SHOULDER: (0.35, 0.28, 0.5),
            BodyLandmark.RIGHT_SHOULDER: (0.65, 0.28, 0.5),
            BodyLandmark.LEFT_ELBOW: (0.25, 0.45, 0.5),
            BodyLandmark.RIGHT_ELBOW: (0.75, 0.45, 0.5),
            BodyLandmark.LEFT_WRIST: (0.20, 0.60, 0.5),
            BodyLandmark.RIGHT_WRIST: (0.80, 0.60, 0.5),
            BodyLandmark.LEFT_INDEX: (0.18, 0.65, 0.5),
            BodyLandmark.RIGHT_INDEX: (0.82, 0.65, 0.5)
        }

        return DetectedLandmarks(
            landmarks=landmarks,
            confidence=0.3,  # Low confidence for fallback
            image_size=person_image.size
        )

    def calculate_placement(
        self,
        landmarks: DetectedLandmarks,
        jewelry_type: JewelryType,
        jewelry_size: Tuple[int, int],
        custom_config: Optional[PlacementConfig] = None
    ) -> Dict[str, Any]:
        """
        Calculate optimal placement for jewelry based on landmarks.

        Args:
            landmarks: Detected body landmarks
            jewelry_type: Type of jewelry to place
            jewelry_size: Size of jewelry image (width, height)
            custom_config: Optional custom placement configuration

        Returns:
            Dictionary with placement information
        """
        config = custom_config or self.DEFAULT_PLACEMENTS.get(
            jewelry_type,
            PlacementConfig(
                jewelry_type=jewelry_type,
                primary_landmark=BodyLandmark.LEFT_SHOULDER,
                secondary_landmarks=[BodyLandmark.RIGHT_SHOULDER],
                offset=(0, 0),
                scale_factor=1.0,
                region_padding=30
            )
        )

        # Get primary landmark position
        primary_pos = landmarks.get_pixel_coords(config.primary_landmark)

        if primary_pos is None:
            # Use center of image as fallback
            primary_pos = (landmarks.image_size[0] // 2, landmarks.image_size[1] // 3)

        # Calculate position based on jewelry type
        if jewelry_type == JewelryType.NECKLACE:
            placement = self._calculate_necklace_placement(
                landmarks, config, jewelry_size
            )
        elif jewelry_type == JewelryType.EARRING:
            placement = self._calculate_earring_placement(
                landmarks, config, jewelry_size
            )
        elif jewelry_type in [JewelryType.BANGLE, JewelryType.BRACELET]:
            placement = self._calculate_wrist_placement(
                landmarks, config, jewelry_size
            )
        elif jewelry_type == JewelryType.RING:
            placement = self._calculate_ring_placement(
                landmarks, config, jewelry_size
            )
        else:
            # Default placement at primary landmark
            placement = {
                'position': (
                    primary_pos[0] + config.offset[0] - jewelry_size[0] // 2,
                    primary_pos[1] + config.offset[1] - jewelry_size[1] // 2
                ),
                'scale': config.scale_factor,
                'rotation': config.rotation
            }

        # Add region mask information
        pos = placement['position']
        scaled_size = (
            int(jewelry_size[0] * placement['scale']),
            int(jewelry_size[1] * placement['scale'])
        )

        placement['region'] = {
            'type': 'rectangle',
            'coords': [
                pos[0] - config.region_padding,
                pos[1] - config.region_padding,
                pos[0] + scaled_size[0] + config.region_padding,
                pos[1] + scaled_size[1] + config.region_padding
            ]
        }

        return placement

    def _calculate_necklace_placement(
        self,
        landmarks: DetectedLandmarks,
        config: PlacementConfig,
        jewelry_size: Tuple[int, int]
    ) -> Dict[str, Any]:
        """Calculate necklace placement between shoulders."""
        left_shoulder = landmarks.get_pixel_coords(BodyLandmark.LEFT_SHOULDER)
        right_shoulder = landmarks.get_pixel_coords(BodyLandmark.RIGHT_SHOULDER)

        if left_shoulder and right_shoulder:
            # Center between shoulders
            center_x = (left_shoulder[0] + right_shoulder[0]) // 2
            center_y = (left_shoulder[1] + right_shoulder[1]) // 2

            # Calculate scale based on shoulder width
            shoulder_width = abs(right_shoulder[0] - left_shoulder[0])
            scale = (shoulder_width * 1.2) / jewelry_size[0]
            scale = max(0.5, min(scale, 2.0))  # Clamp scale

            return {
                'position': (
                    center_x - int(jewelry_size[0] * scale / 2) + config.offset[0],
                    center_y + config.offset[1]
                ),
                'scale': scale * config.scale_factor,
                'rotation': 0
            }

        # Fallback to default position
        return {
            'position': (
                landmarks.image_size[0] // 2 - jewelry_size[0] // 2,
                landmarks.image_size[1] // 4
            ),
            'scale': config.scale_factor,
            'rotation': 0
        }

    def _calculate_earring_placement(
        self,
        landmarks: DetectedLandmarks,
        config: PlacementConfig,
        jewelry_size: Tuple[int, int]
    ) -> Dict[str, Any]:
        """Calculate earring placement at ears."""
        left_ear = landmarks.get_pixel_coords(BodyLandmark.LEFT_EAR)
        right_ear = landmarks.get_pixel_coords(BodyLandmark.RIGHT_EAR)

        placements = []

        if left_ear:
            placements.append({
                'position': (
                    left_ear[0] - jewelry_size[0] // 2 + config.offset[0],
                    left_ear[1] + config.offset[1]
                ),
                'ear': 'left'
            })

        if right_ear:
            placements.append({
                'position': (
                    right_ear[0] - jewelry_size[0] // 2 + config.offset[0],
                    right_ear[1] + config.offset[1]
                ),
                'ear': 'right'
            })

        # Return combined placement info
        return {
            'position': placements[0]['position'] if placements else (100, 100),
            'scale': config.scale_factor,
            'rotation': 0,
            'multiple_placements': placements
        }

    def _calculate_wrist_placement(
        self,
        landmarks: DetectedLandmarks,
        config: PlacementConfig,
        jewelry_size: Tuple[int, int]
    ) -> Dict[str, Any]:
        """Calculate bangle/bracelet placement at wrist."""
        wrist = landmarks.get_pixel_coords(config.primary_landmark)

        if wrist:
            # Calculate rotation based on arm angle
            elbow = landmarks.get_pixel_coords(
                BodyLandmark.LEFT_ELBOW if config.primary_landmark == BodyLandmark.LEFT_WRIST
                else BodyLandmark.RIGHT_ELBOW
            )

            rotation = 0
            if elbow:
                dx = wrist[0] - elbow[0]
                dy = wrist[1] - elbow[1]
                rotation = np.degrees(np.arctan2(dy, dx))

            return {
                'position': (
                    wrist[0] - jewelry_size[0] // 2 + config.offset[0],
                    wrist[1] - jewelry_size[1] // 2 + config.offset[1]
                ),
                'scale': config.scale_factor,
                'rotation': rotation
            }

        return {
            'position': (landmarks.image_size[0] // 4, landmarks.image_size[1] // 2),
            'scale': config.scale_factor,
            'rotation': 0
        }

    def _calculate_ring_placement(
        self,
        landmarks: DetectedLandmarks,
        config: PlacementConfig,
        jewelry_size: Tuple[int, int]
    ) -> Dict[str, Any]:
        """Calculate ring placement on finger."""
        finger = landmarks.get_pixel_coords(config.primary_landmark)

        if finger:
            return {
                'position': (
                    finger[0] - jewelry_size[0] // 2 + config.offset[0],
                    finger[1] - jewelry_size[1] // 2 + config.offset[1]
                ),
                'scale': config.scale_factor,
                'rotation': 0
            }

        return {
            'position': (landmarks.image_size[0] // 5, landmarks.image_size[1] * 2 // 3),
            'scale': config.scale_factor,
            'rotation': 0
        }

    def create_region_mask(
        self,
        image_size: Tuple[int, int],
        placement: Dict[str, Any],
        jewelry_type: JewelryType
    ) -> Image.Image:
        """
        Create a region mask for inpainting.

        Args:
            image_size: Size of the target image
            placement: Placement information dictionary
            jewelry_type: Type of jewelry

        Returns:
            Mask image for inpainting
        """
        mask = Image.new('L', image_size, 0)
        draw = ImageDraw.Draw(mask)

        region = placement.get('region', {})
        region_type = region.get('type', 'rectangle')
        coords = region.get('coords', [])

        if len(coords) >= 4:
            if region_type == 'rectangle':
                draw.rectangle(coords, fill=255)
            elif region_type == 'ellipse':
                draw.ellipse(coords, fill=255)

        # Handle earrings with multiple placements
        if jewelry_type == JewelryType.EARRING:
            multiple = placement.get('multiple_placements', [])
            for p in multiple:
                pos = p.get('position', (0, 0))
                # Draw a region around each earring position
                padding = 30
                draw.ellipse([
                    pos[0] - padding,
                    pos[1] - padding,
                    pos[0] + padding * 2,
                    pos[1] + padding * 3
                ], fill=255)

        # Apply blur for softer edges
        from PIL import ImageFilter
        mask = mask.filter(ImageFilter.GaussianBlur(radius=10))

        return mask

    def process_jewelry(
        self,
        jewelry_image: Image.Image,
        jewelry_type_hint: Optional[str] = None
    ) -> ProcessedJewelry:
        """
        Process a jewelry image for try-on.

        Args:
            jewelry_image: PIL Image of the jewelry
            jewelry_type_hint: Optional hint for jewelry type

        Returns:
            ProcessedJewelry object
        """
        type_hint = None
        if jewelry_type_hint:
            try:
                type_hint = JewelryType(jewelry_type_hint.lower())
            except ValueError:
                logger.warning(f"Unknown jewelry type hint: {jewelry_type_hint}")

        return self.jewelry_processor.process(jewelry_image, type_hint)

    def try_on(
        self,
        person_image: Image.Image,
        jewelry_image: Image.Image,
        jewelry_type: Optional[str] = None,
        custom_placement: Optional[Dict[str, Any]] = None
    ) -> TryOnResult:
        """
        Complete jewelry try-on pipeline.

        Args:
            person_image: PIL Image of the person
            jewelry_image: PIL Image of the jewelry
            jewelry_type: Optional jewelry type hint
            custom_placement: Optional custom placement configuration

        Returns:
            TryOnResult with the rendered image
        """
        logger.info("Starting jewelry try-on pipeline")

        try:
            # Step 1: Process jewelry image
            logger.info("Processing jewelry image...")
            processed_jewelry = self.process_jewelry(jewelry_image, jewelry_type)
            logger.info(f"Jewelry processed: {processed_jewelry.jewelry_type.value}")

            # Step 2: Detect landmarks in person image
            logger.info("Detecting body landmarks...")
            landmarks = self.detect_landmarks(person_image)

            if landmarks is None:
                return TryOnResult(
                    success=False,
                    result_image=None,
                    jewelry_type=processed_jewelry.jewelry_type,
                    placement_info={},
                    render_mode=RenderMode.OVERLAY,
                    error_message="Failed to detect body landmarks"
                )

            logger.info(f"Landmarks detected with confidence: {landmarks.confidence:.2f}")

            # Step 3: Calculate placement
            logger.info("Calculating jewelry placement...")
            placement = self.calculate_placement(
                landmarks,
                processed_jewelry.jewelry_type,
                processed_jewelry.processed_image.size,
                custom_placement
            )
            logger.info(f"Placement calculated: position={placement['position']}")

            # Step 4: Create region mask for inpainting
            mask = self.create_region_mask(
                person_image.size,
                placement,
                processed_jewelry.jewelry_type
            )

            # Step 5: Render
            logger.info("Rendering jewelry on person...")
            render_result = self.renderer.render(
                person_image=person_image,
                jewelry_image=processed_jewelry.processed_image,
                jewelry_type=processed_jewelry.jewelry_type.value,
                jewelry_description=processed_jewelry.description,
                placement_region=placement['region'],
                position=placement['position'],
                scale=placement['scale'],
                rotation=placement['rotation']
            )

            if render_result.success:
                logger.info(f"Rendering successful using {render_result.mode_used.value}")
                return TryOnResult(
                    success=True,
                    result_image=render_result.image,
                    jewelry_type=processed_jewelry.jewelry_type,
                    placement_info=placement,
                    render_mode=render_result.mode_used
                )
            else:
                logger.error(f"Rendering failed: {render_result.error_message}")
                return TryOnResult(
                    success=False,
                    result_image=None,
                    jewelry_type=processed_jewelry.jewelry_type,
                    placement_info=placement,
                    render_mode=render_result.mode_used,
                    error_message=render_result.error_message
                )

        except Exception as e:
            logger.error(f"Try-on pipeline error: {e}")
            return TryOnResult(
                success=False,
                result_image=None,
                jewelry_type=JewelryType.UNKNOWN,
                placement_info={},
                render_mode=RenderMode.OVERLAY,
                error_message=str(e)
            )

    def try_on_multiple(
        self,
        person_image: Image.Image,
        jewelry_items: List[Tuple[Image.Image, Optional[str]]]
    ) -> TryOnResult:
        """
        Try on multiple jewelry items on the same person.

        Args:
            person_image: PIL Image of the person
            jewelry_items: List of (jewelry_image, jewelry_type) tuples

        Returns:
            TryOnResult with all jewelry rendered
        """
        current_image = person_image.copy()
        all_placements = []

        for jewelry_image, jewelry_type in jewelry_items:
            result = self.try_on(current_image, jewelry_image, jewelry_type)

            if result.success and result.result_image:
                current_image = result.result_image
                all_placements.append({
                    'type': result.jewelry_type.value,
                    'placement': result.placement_info
                })
            else:
                logger.warning(f"Failed to apply {jewelry_type}: {result.error_message}")

        return TryOnResult(
            success=True,
            result_image=current_image,
            jewelry_type=JewelryType.MIXED if len(jewelry_items) > 1 else JewelryType.UNKNOWN,
            placement_info={'items': all_placements},
            render_mode=RenderMode.HYBRID
        )

    def get_supported_jewelry_types(self) -> List[str]:
        """Return list of supported jewelry types."""
        return [jt.value for jt in JewelryType if jt != JewelryType.UNKNOWN]

    def cleanup(self) -> None:
        """Release resources."""
        if self.pose_detector:
            self.pose_detector.close()
        if self.face_mesh:
            self.face_mesh.close()
        logger.info("Resources cleaned up")


def create_engine(
    api_key: Optional[str] = None,
    mode: str = "hybrid"
) -> HybridJewelryEngine:
    """
    Factory function to create a HybridJewelryEngine.

    Args:
        api_key: Google API key for Imagen
        mode: Render mode ('imagen_inpaint', 'overlay', 'hybrid')

    Returns:
        Configured HybridJewelryEngine instance
    """
    return HybridJewelryEngine(
        google_api_key=api_key,
        render_mode=mode
    )


def try_on_jewelry(
    person_image_path: str,
    jewelry_image_path: str,
    jewelry_type: Optional[str] = None,
    output_path: Optional[str] = None,
    api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function for single jewelry try-on.

    Args:
        person_image_path: Path to person image
        jewelry_image_path: Path to jewelry image
        jewelry_type: Optional jewelry type hint
        output_path: Optional path to save result
        api_key: Optional Google API key

    Returns:
        Dictionary with results
    """
    engine = create_engine(api_key=api_key)

    person_image = Image.open(person_image_path)
    jewelry_image = Image.open(jewelry_image_path)

    result = engine.try_on(person_image, jewelry_image, jewelry_type)

    if result.success and result.result_image:
        if output_path:
            result.result_image.save(output_path)
            logger.info(f"Result saved to {output_path}")

        return {
            'success': True,
            'image': result.result_image,
            'jewelry_type': result.jewelry_type.value,
            'render_mode': result.render_mode.value,
            'placement': result.placement_info
        }

    return {
        'success': False,
        'error': result.error_message
    }


if __name__ == "__main__":
    import sys

    print("Hybrid Jewelry Try-On Engine")
    print("=" * 50)
    print(f"MediaPipe available: {MEDIAPIPE_AVAILABLE}")

    engine = create_engine(mode="hybrid")
    print(f"Engine created successfully")
    print(f"Supported jewelry types: {engine.get_supported_jewelry_types()}")

    if len(sys.argv) >= 3:
        person_path = sys.argv[1]
        jewelry_path = sys.argv[2]
        jewelry_type = sys.argv[3] if len(sys.argv) > 3 else None
        output_path = sys.argv[4] if len(sys.argv) > 4 else "result.png"

        result = try_on_jewelry(
            person_path,
            jewelry_path,
            jewelry_type,
            output_path
        )

        if result['success']:
            print(f"\nSuccess! Result saved to {output_path}")
            print(f"Jewelry type: {result['jewelry_type']}")
            print(f"Render mode: {result['render_mode']}")
        else:
            print(f"\nFailed: {result['error']}")
    else:
        print("\nUsage: python hybrid_jewelry_engine.py <person_image> <jewelry_image> [jewelry_type] [output_path]")

    engine.cleanup()
