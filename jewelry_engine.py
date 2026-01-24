"""
Naari Studio - Jewelry Virtual Try-On Engine
Uses OpenCV Haar Cascade for face detection and proportional landmark estimation
Supports: Necklace, Earrings, Maang Tikka, Bangles
"""

import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Optional, Dict, Any
import os


class JewelryTryOnEngine:
    """
    Jewelry virtual try-on engine using OpenCV Haar Cascade for face detection.
    Positions jewelry based on proportional calculations from face bounding box.
    """

    # Jewelry type constants
    NECKLACE = "necklace"
    EARRINGS = "earrings"
    MAANG_TIKKA = "maang_tikka"
    BANGLES = "bangles"

    def __init__(self):
        """Initialize the engine with OpenCV Haar Cascade classifier."""
        # Load Haar Cascade for face detection
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

        if self.face_cascade.empty():
            raise RuntimeError("Failed to load Haar Cascade classifier")

    def detect_face(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect the largest face in the image using Haar Cascade.

        Args:
            image: BGR image as numpy array

        Returns:
            Tuple of (x, y, width, height) for the largest face, or None if no face detected
        """
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces with multiple scale factors for better detection
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        if len(faces) == 0:
            # Try with more lenient parameters
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.05,
                minNeighbors=3,
                minSize=(20, 20)
            )

        if len(faces) == 0:
            return None

        # Return the largest face (most likely the main subject)
        largest_face = max(faces, key=lambda f: f[2] * f[3])
        return tuple(largest_face)

    def calculate_landmarks(self, face_rect: Tuple[int, int, int, int],
                           image_shape: Tuple[int, int]) -> Dict[str, Any]:
        """
        Calculate proportional landmark positions from face bounding box.

        Args:
            face_rect: (x, y, width, height) of detected face
            image_shape: (height, width) of the image

        Returns:
            Dictionary with estimated landmark positions for jewelry placement
        """
        x, y, w, h = face_rect
        img_h, img_w = image_shape[:2]

        # Face center
        face_center_x = x + w // 2
        face_center_y = y + h // 2

        # Proportional calculations based on typical face anatomy
        landmarks = {
            # Forehead center (for Maang Tikka) - top of face
            "forehead_center": (face_center_x, y + int(h * 0.1)),

            # Ears (for Earrings) - sides of face at eye level
            "left_ear": (x - int(w * 0.1), y + int(h * 0.35)),
            "right_ear": (x + w + int(w * 0.1), y + int(h * 0.35)),

            # Chin bottom (for Necklace positioning)
            "chin": (face_center_x, y + h),

            # Neck center (below chin for necklace)
            "neck_center": (face_center_x, y + h + int(h * 0.3)),

            # Shoulders (estimated from face width)
            "left_shoulder": (face_center_x - int(w * 1.2), y + h + int(h * 0.8)),
            "right_shoulder": (face_center_x + int(w * 1.2), y + h + int(h * 0.8)),

            # Wrists (estimated from shoulders - for bangles)
            # Estimate body proportions: wrists are roughly 2x face height below shoulders
            "left_wrist": (max(50, face_center_x - int(w * 1.8)),
                          min(img_h - 50, y + h + int(h * 2.5))),
            "right_wrist": (min(img_w - 50, face_center_x + int(w * 1.8)),
                           min(img_h - 50, y + h + int(h * 2.5))),

            # Face dimensions for scaling jewelry
            "face_width": w,
            "face_height": h,
            "face_center": (face_center_x, face_center_y)
        }

        return landmarks

    def overlay_image_alpha(self, background: np.ndarray, overlay: np.ndarray,
                           position: Tuple[int, int], opacity: float = 1.0) -> np.ndarray:
        """
        Overlay an image with alpha channel onto a background.

        Args:
            background: BGR background image
            overlay: BGRA overlay image with alpha channel
            position: (x, y) position for overlay center
            opacity: Overall opacity (0.0 to 1.0)

        Returns:
            Blended image as BGR numpy array
        """
        result = background.copy()

        # Get overlay dimensions
        oh, ow = overlay.shape[:2]

        # Calculate top-left corner from center position
        x = position[0] - ow // 2
        y = position[1] - oh // 2

        # Calculate the valid region to overlay
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(background.shape[1], x + ow), min(background.shape[0], y + oh)

        # Calculate corresponding region in overlay
        ox1 = x1 - x
        oy1 = y1 - y
        ox2 = ox1 + (x2 - x1)
        oy2 = oy1 + (y2 - y1)

        if x2 <= x1 or y2 <= y1:
            return result

        # Extract the regions
        bg_region = result[y1:y2, x1:x2]
        overlay_region = overlay[oy1:oy2, ox1:ox2]

        # Handle alpha channel
        if overlay_region.shape[2] == 4:
            # Extract alpha channel and apply opacity
            alpha = (overlay_region[:, :, 3] / 255.0) * opacity
            alpha = np.expand_dims(alpha, axis=2)

            # Blend colors
            overlay_rgb = overlay_region[:, :, :3]
            blended = (alpha * overlay_rgb + (1 - alpha) * bg_region).astype(np.uint8)
            result[y1:y2, x1:x2] = blended
        else:
            # No alpha channel, just blend with opacity
            blended = cv2.addWeighted(overlay_region, opacity, bg_region, 1 - opacity, 0)
            result[y1:y2, x1:x2] = blended

        return result

    def resize_jewelry(self, jewelry_img: np.ndarray, target_width: int,
                       maintain_aspect: bool = True) -> np.ndarray:
        """
        Resize jewelry image to target width while maintaining aspect ratio.

        Args:
            jewelry_img: Jewelry image (BGRA)
            target_width: Target width in pixels
            maintain_aspect: Whether to maintain aspect ratio

        Returns:
            Resized jewelry image
        """
        h, w = jewelry_img.shape[:2]

        if maintain_aspect:
            aspect = h / w
            target_height = int(target_width * aspect)
        else:
            target_height = h

        resized = cv2.resize(jewelry_img, (target_width, target_height),
                            interpolation=cv2.INTER_AREA if target_width < w else cv2.INTER_CUBIC)
        return resized

    def apply_necklace(self, image: np.ndarray, jewelry_img: np.ndarray,
                       landmarks: Dict, opacity: float = 1.0) -> np.ndarray:
        """Apply necklace at neck area below chin."""
        face_width = landmarks["face_width"]
        neck_center = landmarks["neck_center"]

        # Scale necklace to approximately 2x face width
        target_width = int(face_width * 2.2)
        jewelry_resized = self.resize_jewelry(jewelry_img, target_width)

        # Position at neck center
        return self.overlay_image_alpha(image, jewelry_resized, neck_center, opacity)

    def apply_earrings(self, image: np.ndarray, jewelry_img: np.ndarray,
                       landmarks: Dict, opacity: float = 1.0) -> np.ndarray:
        """Apply earrings at both ear positions."""
        face_width = landmarks["face_width"]
        left_ear = landmarks["left_ear"]
        right_ear = landmarks["right_ear"]

        # Scale earrings to approximately 0.4x face width
        target_width = int(face_width * 0.4)
        jewelry_resized = self.resize_jewelry(jewelry_img, target_width)

        # Apply to left ear
        result = self.overlay_image_alpha(image, jewelry_resized, left_ear, opacity)

        # Flip horizontally for right ear and apply
        jewelry_flipped = cv2.flip(jewelry_resized, 1)
        result = self.overlay_image_alpha(result, jewelry_flipped, right_ear, opacity)

        return result

    def apply_maang_tikka(self, image: np.ndarray, jewelry_img: np.ndarray,
                          landmarks: Dict, opacity: float = 1.0) -> np.ndarray:
        """Apply maang tikka at forehead center."""
        face_width = landmarks["face_width"]
        forehead_center = landmarks["forehead_center"]

        # Scale maang tikka to approximately 0.5x face width
        target_width = int(face_width * 0.5)
        jewelry_resized = self.resize_jewelry(jewelry_img, target_width)

        return self.overlay_image_alpha(image, jewelry_resized, forehead_center, opacity)

    def apply_bangles(self, image: np.ndarray, jewelry_img: np.ndarray,
                      landmarks: Dict, opacity: float = 1.0,
                      hand: str = "both") -> np.ndarray:
        """
        Apply bangles at wrist positions.

        Args:
            hand: "left", "right", or "both"
        """
        face_width = landmarks["face_width"]

        # Scale bangles to approximately 0.8x face width
        target_width = int(face_width * 0.8)
        jewelry_resized = self.resize_jewelry(jewelry_img, target_width)

        result = image.copy()

        if hand in ["left", "both"]:
            left_wrist = landmarks["left_wrist"]
            result = self.overlay_image_alpha(result, jewelry_resized, left_wrist, opacity)

        if hand in ["right", "both"]:
            right_wrist = landmarks["right_wrist"]
            jewelry_flipped = cv2.flip(jewelry_resized, 1)
            result = self.overlay_image_alpha(result, jewelry_flipped, right_wrist, opacity)

        return result

    def load_jewelry_image(self, jewelry_path: str) -> Optional[np.ndarray]:
        """
        Load a jewelry image with alpha channel support.

        Args:
            jewelry_path: Path to jewelry PNG image

        Returns:
            BGRA image as numpy array, or None if loading failed
        """
        if not os.path.exists(jewelry_path):
            return None

        # Load with alpha channel
        img = cv2.imread(jewelry_path, cv2.IMREAD_UNCHANGED)

        if img is None:
            return None

        # Ensure 4 channels (BGRA)
        if len(img.shape) == 2:
            # Grayscale, convert to BGRA
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
        elif img.shape[2] == 3:
            # BGR, add alpha channel
            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

        return img

    def apply_jewelry(self, person_image: np.ndarray,
                      jewelry_image: np.ndarray,
                      jewelry_type: str,
                      opacity: float = 1.0,
                      **kwargs) -> Tuple[np.ndarray, Dict]:
        """
        Main function to apply jewelry to a person image.

        Args:
            person_image: BGR image of the person
            jewelry_image: BGRA image of the jewelry
            jewelry_type: One of "necklace", "earrings", "maang_tikka", "bangles"
            opacity: Opacity level (0.0 to 1.0)
            **kwargs: Additional arguments (e.g., hand="left" for bangles)

        Returns:
            Tuple of (result_image, info_dict)
        """
        info = {
            "success": False,
            "message": "",
            "face_detected": False,
            "jewelry_type": jewelry_type
        }

        # Validate jewelry type
        valid_types = [self.NECKLACE, self.EARRINGS, self.MAANG_TIKKA, self.BANGLES]
        if jewelry_type not in valid_types:
            info["message"] = f"Invalid jewelry type. Must be one of: {valid_types}"
            return person_image, info

        # Detect face
        face_rect = self.detect_face(person_image)

        if face_rect is None:
            info["message"] = "No face detected in the image. Please use a clear front-facing photo."
            return person_image, info

        info["face_detected"] = True
        info["face_rect"] = face_rect

        # Calculate landmarks
        landmarks = self.calculate_landmarks(face_rect, person_image.shape)
        info["landmarks"] = {k: v for k, v in landmarks.items()
                           if not isinstance(v, (np.ndarray,))}

        # Apply jewelry based on type
        try:
            if jewelry_type == self.NECKLACE:
                result = self.apply_necklace(person_image, jewelry_image, landmarks, opacity)
            elif jewelry_type == self.EARRINGS:
                result = self.apply_earrings(person_image, jewelry_image, landmarks, opacity)
            elif jewelry_type == self.MAANG_TIKKA:
                result = self.apply_maang_tikka(person_image, jewelry_image, landmarks, opacity)
            elif jewelry_type == self.BANGLES:
                hand = kwargs.get("hand", "both")
                result = self.apply_bangles(person_image, jewelry_image, landmarks, opacity, hand)

            info["success"] = True
            info["message"] = f"Successfully applied {jewelry_type}"
            return result, info

        except Exception as e:
            info["message"] = f"Error applying jewelry: {str(e)}"
            return person_image, info


def apply_jewelry(person_image_path: str,
                  jewelry_image_path: str,
                  jewelry_type: str,
                  opacity: float = 1.0,
                  **kwargs) -> Tuple[Image.Image, Dict]:
    """
    Convenience function to apply jewelry from file paths.

    Args:
        person_image_path: Path to person image
        jewelry_image_path: Path to jewelry PNG image
        jewelry_type: One of "necklace", "earrings", "maang_tikka", "bangles"
        opacity: Opacity level (0.0 to 1.0)

    Returns:
        Tuple of (PIL Image result, info dict)
    """
    engine = JewelryTryOnEngine()

    # Load images
    person_img = cv2.imread(person_image_path)
    if person_img is None:
        return None, {"success": False, "message": "Failed to load person image"}

    jewelry_img = engine.load_jewelry_image(jewelry_image_path)
    if jewelry_img is None:
        return None, {"success": False, "message": "Failed to load jewelry image"}

    # Apply jewelry
    result, info = engine.apply_jewelry(person_img, jewelry_img, jewelry_type, opacity, **kwargs)

    # Convert to PIL Image
    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    result_pil = Image.fromarray(result_rgb)

    return result_pil, info


def apply_jewelry_pil(person_image: Image.Image,
                      jewelry_image: Image.Image,
                      jewelry_type: str,
                      opacity: float = 1.0,
                      **kwargs) -> Tuple[Image.Image, Dict]:
    """
    Apply jewelry using PIL Images directly.

    Args:
        person_image: PIL Image of the person
        jewelry_image: PIL Image of the jewelry (should have transparency)
        jewelry_type: One of "necklace", "earrings", "maang_tikka", "bangles"
        opacity: Opacity level (0.0 to 1.0)

    Returns:
        Tuple of (PIL Image result, info dict)
    """
    engine = JewelryTryOnEngine()

    # Convert PIL to OpenCV format
    person_cv = cv2.cvtColor(np.array(person_image), cv2.COLOR_RGB2BGR)

    # Convert jewelry to BGRA
    jewelry_rgba = jewelry_image.convert("RGBA")
    jewelry_np = np.array(jewelry_rgba)
    # Convert RGBA to BGRA
    jewelry_cv = cv2.cvtColor(jewelry_np, cv2.COLOR_RGBA2BGRA)

    # Apply jewelry
    result, info = engine.apply_jewelry(person_cv, jewelry_cv, jewelry_type, opacity, **kwargs)

    # Convert back to PIL
    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    result_pil = Image.fromarray(result_rgb)

    return result_pil, info


# Debug visualization function
def visualize_landmarks(image_path: str, output_path: str = None):
    """
    Visualize detected face and landmarks for debugging.

    Args:
        image_path: Path to input image
        output_path: Path to save visualization (optional)
    """
    engine = JewelryTryOnEngine()

    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load image: {image_path}")
        return None

    face_rect = engine.detect_face(img)

    if face_rect is None:
        print("No face detected")
        return None

    # Draw face rectangle
    x, y, w, h = face_rect
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Calculate and draw landmarks
    landmarks = engine.calculate_landmarks(face_rect, img.shape)

    landmark_colors = {
        "forehead_center": (255, 0, 0),    # Blue
        "left_ear": (0, 255, 0),            # Green
        "right_ear": (0, 255, 0),           # Green
        "chin": (0, 0, 255),                # Red
        "neck_center": (255, 255, 0),       # Cyan
        "left_shoulder": (255, 0, 255),     # Magenta
        "right_shoulder": (255, 0, 255),    # Magenta
        "left_wrist": (0, 255, 255),        # Yellow
        "right_wrist": (0, 255, 255)        # Yellow
    }

    for name, pos in landmarks.items():
        if isinstance(pos, tuple) and len(pos) == 2:
            color = landmark_colors.get(name, (128, 128, 128))
            cv2.circle(img, pos, 8, color, -1)
            cv2.putText(img, name, (pos[0] + 10, pos[1]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    if output_path:
        cv2.imwrite(output_path, img)
        print(f"Saved visualization to: {output_path}")

    # Convert to PIL for display
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)


if __name__ == "__main__":
    print("Naari Studio - Jewelry Try-On Engine")
    print("=" * 40)
    print("Supported jewelry types:")
    print("  - necklace: Positioned at neck area")
    print("  - earrings: Positioned at ear level")
    print("  - maang_tikka: Positioned at forehead center")
    print("  - bangles: Positioned at wrist area")
    print()
    print("Usage:")
    print("  from jewelry_engine import apply_jewelry_pil")
    print("  result, info = apply_jewelry_pil(person_img, jewelry_img, 'necklace')")
