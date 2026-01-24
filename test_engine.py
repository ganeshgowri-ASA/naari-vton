"""
Test script for the Jewelry Try-On Engine
"""

import cv2
import numpy as np
from PIL import Image
import os

from jewelry_engine import JewelryTryOnEngine, apply_jewelry_pil


def create_test_person_image(output_path: str = "test_person.png"):
    """
    Create a simple test image with a face-like oval shape.
    This is for testing when no real photo is available.
    """
    width, height = 400, 600
    img = np.ones((height, width, 3), dtype=np.uint8) * 200  # Gray background

    # Draw a simple face oval (skin color)
    center_x, center_y = width // 2, 180
    face_w, face_h = 80, 100
    cv2.ellipse(img, (center_x, center_y), (face_w, face_h), 0, 0, 360, (200, 180, 160), -1)

    # Draw eyes
    eye_y = center_y - 20
    cv2.ellipse(img, (center_x - 30, eye_y), (15, 10), 0, 0, 360, (50, 50, 50), -1)
    cv2.ellipse(img, (center_x + 30, eye_y), (15, 10), 0, 0, 360, (50, 50, 50), -1)

    # Draw nose
    cv2.line(img, (center_x, center_y - 10), (center_x, center_y + 20), (150, 130, 120), 2)

    # Draw mouth
    cv2.ellipse(img, (center_x, center_y + 50), (25, 10), 0, 0, 180, (180, 100, 100), 2)

    # Draw body/shoulders
    body_top = center_y + face_h + 30
    pts = np.array([
        [center_x - 120, height],
        [center_x - 80, body_top],
        [center_x + 80, body_top],
        [center_x + 120, height]
    ], np.int32)
    cv2.fillPoly(img, [pts], (100, 80, 150))

    # Draw neck
    cv2.rectangle(img, (center_x - 30, center_y + face_h - 20),
                  (center_x + 30, body_top + 10), (190, 170, 150), -1)

    cv2.imwrite(output_path, img)
    print(f"Created test person image: {output_path}")
    return output_path


def test_face_detection():
    """Test face detection functionality."""
    print("\n=== Testing Face Detection ===")

    engine = JewelryTryOnEngine()
    print("Engine initialized successfully!")

    # Create test image
    test_path = "test_person.png"
    create_test_person_image(test_path)

    # Load and test
    img = cv2.imread(test_path)
    face = engine.detect_face(img)

    if face:
        print(f"Face detected at: x={face[0]}, y={face[1]}, w={face[2]}, h={face[3]}")
        print("Face detection: PASSED")
    else:
        print("No face detected (this might happen with synthetic images)")
        print("Note: Haar Cascade works best with real photos")

    # Clean up
    if os.path.exists(test_path):
        os.remove(test_path)

    return face is not None


def test_jewelry_loading():
    """Test jewelry image loading."""
    print("\n=== Testing Jewelry Loading ===")

    engine = JewelryTryOnEngine()

    assets_dir = "assets/jewelry"
    jewelry_files = ["necklace.png", "earring.png", "maang_tikka.png", "bangle.png"]

    all_loaded = True
    for filename in jewelry_files:
        filepath = os.path.join(assets_dir, filename)
        img = engine.load_jewelry_image(filepath)

        if img is not None:
            print(f"Loaded {filename}: shape={img.shape}, dtype={img.dtype}")
            if img.shape[2] == 4:
                print(f"  - Has alpha channel: YES")
            else:
                print(f"  - Has alpha channel: NO (will be added)")
        else:
            print(f"FAILED to load: {filename}")
            all_loaded = False

    print(f"\nJewelry loading: {'PASSED' if all_loaded else 'FAILED'}")
    return all_loaded


def test_landmark_calculation():
    """Test landmark calculation from face rect."""
    print("\n=== Testing Landmark Calculation ===")

    engine = JewelryTryOnEngine()

    # Simulated face detection result
    face_rect = (100, 80, 150, 180)  # x, y, width, height
    image_shape = (600, 400, 3)  # height, width, channels

    landmarks = engine.calculate_landmarks(face_rect, image_shape)

    print("Calculated landmarks:")
    for name, value in landmarks.items():
        print(f"  {name}: {value}")

    # Verify all expected landmarks exist
    expected_keys = [
        "forehead_center", "left_ear", "right_ear", "chin",
        "neck_center", "left_shoulder", "right_shoulder",
        "left_wrist", "right_wrist", "face_width", "face_height", "face_center"
    ]

    all_present = all(key in landmarks for key in expected_keys)
    print(f"\nLandmark calculation: {'PASSED' if all_present else 'FAILED'}")
    return all_present


def test_overlay():
    """Test image overlay functionality."""
    print("\n=== Testing Image Overlay ===")

    engine = JewelryTryOnEngine()

    # Create a simple background
    background = np.ones((200, 200, 3), dtype=np.uint8) * 128

    # Create an overlay with alpha
    overlay = np.zeros((50, 50, 4), dtype=np.uint8)
    overlay[:, :, 0] = 255  # Blue
    overlay[:, :, 3] = 200  # Semi-transparent alpha

    result = engine.overlay_image_alpha(background, overlay, (100, 100), opacity=1.0)

    # Check that overlay was applied
    changed = not np.array_equal(result, background)
    print(f"Overlay applied changes: {changed}")
    print(f"Image overlay: {'PASSED' if changed else 'FAILED'}")
    return changed


def test_pil_interface():
    """Test PIL interface functions."""
    print("\n=== Testing PIL Interface ===")

    # Create test images
    person_img = Image.new('RGB', (400, 600), color=(200, 180, 160))
    jewelry_img = Image.new('RGBA', (100, 50), color=(218, 165, 32, 255))

    try:
        result, info = apply_jewelry_pil(person_img, jewelry_img, "necklace", 1.0)

        print(f"Result type: {type(result)}")
        print(f"Info success: {info.get('success', 'N/A')}")
        print(f"Info message: {info.get('message', 'N/A')}")

        # Note: May fail face detection with synthetic image
        print(f"\nPIL interface: PASSED (function executed without errors)")
        return True
    except Exception as e:
        print(f"Error: {e}")
        print("PIL interface: FAILED")
        return False


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("NAARI STUDIO - Jewelry Engine Test Suite")
    print("=" * 60)

    results = {
        "Face Detection": test_face_detection(),
        "Jewelry Loading": test_jewelry_loading(),
        "Landmark Calculation": test_landmark_calculation(),
        "Image Overlay": test_overlay(),
        "PIL Interface": test_pil_interface()
    }

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    all_passed = True
    for test_name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"  {test_name}: {status}")
        if not passed:
            all_passed = False

    print("\n" + "=" * 60)
    print(f"Overall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    run_all_tests()
