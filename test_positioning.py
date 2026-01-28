#!/usr/bin/env python3
"""
Diagnostic script to test jewelry positioning and face detection.
Creates test images and visualizes where jewelry would be placed.

Run this script to verify jewelry positioning is working correctly.
"""

import cv2
import numpy as np
from PIL import Image
import os
import sys

# Import the jewelry engine
from jewelry_engine import JewelryEngine, FACE_CASCADE, _init_cascades

def create_realistic_test_face(width=400, height=600):
    """
    Create a more realistic test image with a face-like structure
    that Haar cascades can detect.
    """
    # Create skin-toned background (like a person)
    img = np.ones((height, width, 3), dtype=np.uint8) * 200  # Light gray background

    # Draw a face shape (oval) - skin color
    face_center = (width // 2, height // 3)  # Face in upper third
    face_width = width // 3
    face_height = int(face_width * 1.3)

    # Face oval (skin tone)
    cv2.ellipse(img, face_center, (face_width // 2, face_height // 2), 0, 0, 360, (180, 160, 140), -1)

    # Eyes (dark circles for contrast - Haar cascades need this)
    eye_y = face_center[1] - face_height // 6
    eye_offset = face_width // 4
    eye_size = face_width // 10

    # Left eye
    cv2.circle(img, (face_center[0] - eye_offset, eye_y), eye_size, (255, 255, 255), -1)  # White
    cv2.circle(img, (face_center[0] - eye_offset, eye_y), eye_size // 2, (50, 50, 50), -1)  # Pupil

    # Right eye
    cv2.circle(img, (face_center[0] + eye_offset, eye_y), eye_size, (255, 255, 255), -1)  # White
    cv2.circle(img, (face_center[0] + eye_offset, eye_y), eye_size // 2, (50, 50, 50), -1)  # Pupil

    # Eyebrows (dark lines above eyes)
    brow_y = eye_y - eye_size - 5
    cv2.line(img, (face_center[0] - eye_offset - 15, brow_y),
             (face_center[0] - eye_offset + 15, brow_y), (50, 50, 50), 3)
    cv2.line(img, (face_center[0] + eye_offset - 15, brow_y),
             (face_center[0] + eye_offset + 15, brow_y), (50, 50, 50), 3)

    # Nose (subtle shading)
    nose_top = eye_y + 20
    nose_bottom = face_center[1] + face_height // 6
    cv2.line(img, (face_center[0], nose_top), (face_center[0], nose_bottom), (160, 140, 120), 2)

    # Mouth
    mouth_y = face_center[1] + face_height // 3
    cv2.ellipse(img, (face_center[0], mouth_y), (20, 8), 0, 0, 180, (100, 80, 80), 2)

    # Neck (below face)
    neck_top = face_center[1] + face_height // 2
    neck_width = face_width // 2
    cv2.rectangle(img,
                  (face_center[0] - neck_width // 2, neck_top),
                  (face_center[0] + neck_width // 2, neck_top + 100),
                  (180, 160, 140), -1)

    # Shoulders
    shoulder_y = neck_top + 80
    cv2.rectangle(img, (0, shoulder_y), (width, height), (100, 100, 150), -1)  # Blue shirt

    return img


def create_jewelry_overlay(width, height, color=(0, 0, 255)):
    """Create a semi-transparent colored rectangle as jewelry placeholder."""
    jewelry = np.zeros((height, width, 4), dtype=np.uint8)
    jewelry[:, :, :3] = color  # BGR color
    jewelry[:, :, 3] = 180  # Alpha

    # Add a border to make it more visible
    cv2.rectangle(jewelry, (0, 0), (width-1, height-1), (0, 255, 0, 255), 2)

    return jewelry


def draw_debug_markers(img, landmarks, label=""):
    """Draw debug markers on the image showing detected landmarks."""
    debug_img = img.copy()

    if landmarks is None:
        cv2.putText(debug_img, "NO FACE DETECTED", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return debug_img

    # Draw face bounding box
    x, y, w, h = landmarks['face_bounds']
    cv2.rectangle(debug_img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    cv2.putText(debug_img, f"Face: ({x},{y}) {w}x{h}", (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

    # Draw each landmark
    markers = [
        ('chin', (0, 255, 0), 'C'),
        ('forehead', (255, 0, 255), 'F'),
        ('left_earlobe', (0, 255, 255), 'LE'),
        ('right_earlobe', (0, 255, 255), 'RE'),
        ('nose_tip', (255, 255, 0), 'NT'),
        ('left_nostril', (255, 128, 0), 'LN'),
        ('right_nostril', (255, 128, 0), 'RN'),
        ('septum', (128, 0, 255), 'S'),
        ('left_eye', (0, 128, 255), 'LE'),
        ('right_eye', (0, 128, 255), 'RE'),
    ]

    for key, color, label_text in markers:
        if key in landmarks:
            pos = landmarks[key]
            cv2.circle(debug_img, pos, 5, color, -1)
            cv2.putText(debug_img, f"{label_text}:{pos}", (pos[0]+5, pos[1]-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

    # Draw necklace position (chin + 20% face height)
    chin = landmarks['chin']
    face_height = landmarks['face_height']
    necklace_pos = (chin[0], chin[1] + int(face_height * 0.20))
    cv2.circle(debug_img, necklace_pos, 8, (0, 0, 255), 2)
    cv2.putText(debug_img, f"NECKLACE:{necklace_pos}", (necklace_pos[0]+10, necklace_pos[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    # Add image dimensions
    h_img, w_img = debug_img.shape[:2]
    cv2.putText(debug_img, f"Image: {w_img}x{h_img}", (10, h_img - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    return debug_img


def test_with_real_face_detection():
    """Test face detection with a synthetic image."""
    print("=" * 60)
    print("JEWELRY POSITIONING DIAGNOSTIC TEST")
    print("=" * 60)

    # Check cascade status
    _init_cascades()
    print(f"\nFace Cascade Loaded: {FACE_CASCADE is not None}")

    # Create test image
    print("\n1. Creating test image with face-like features...")
    test_img = create_realistic_test_face(400, 600)

    # Initialize engine
    engine = JewelryEngine()

    # Run face detection
    print("\n2. Running face detection...")
    landmarks = engine.detect_face(test_img)

    if landmarks:
        print("\n   FACE DETECTED!")
        print(f"   Face bounds: {landmarks['face_bounds']}")
        print(f"   Face center: {landmarks['face_center']}")
        print(f"   Chin position: {landmarks['chin']}")
        print(f"   Forehead position: {landmarks['forehead']}")
        print(f"   Left earlobe: {landmarks['left_earlobe']}")
        print(f"   Right earlobe: {landmarks['right_earlobe']}")
        print(f"   Nose tip: {landmarks['nose_tip']}")
        print(f"   Face angle: {landmarks['face_angle']:.2f} degrees")

        # Calculate where necklace would go
        chin_x, chin_y = landmarks['chin']
        face_height = landmarks['face_height']
        necklace_y = chin_y + int(face_height * 0.20)
        print(f"\n   NECKLACE would be placed at: ({chin_x}, {necklace_y})")
        print(f"   Image height: {test_img.shape[0]}")
        print(f"   Necklace Y as % of image: {necklace_y / test_img.shape[0] * 100:.1f}%")
    else:
        print("\n   NO FACE DETECTED - Using fallback positioning")
        h, w = test_img.shape[:2]
        necklace_x = w // 2
        necklace_y = int(h * 0.45)
        print(f"   FALLBACK necklace position: ({necklace_x}, {necklace_y})")
        print(f"   This is at {necklace_y / h * 100:.1f}% down the image")

    # Create debug visualization
    debug_img = draw_debug_markers(test_img, landmarks)

    # Save debug image
    debug_path = "/home/user/naari-vton/debug_face_detection.png"
    cv2.imwrite(debug_path, debug_img)
    print(f"\n3. Debug image saved to: {debug_path}")

    # Test jewelry application
    print("\n4. Testing necklace application...")
    test_pil = Image.fromarray(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB))

    # Create necklace image
    necklace_cv = create_jewelry_overlay(100, 60)
    necklace_pil = Image.fromarray(necklace_cv)

    result, msg = engine.apply_necklace(test_pil, necklace_pil)
    print(f"   Result: {msg}")

    # Save result
    result_path = "/home/user/naari-vton/debug_necklace_result.png"
    result.save(result_path)
    print(f"   Result image saved to: {result_path}")

    # Compare input and output
    result_arr = np.array(result)
    input_arr = np.array(test_pil)

    if np.array_equal(result_arr, input_arr):
        print("\n   WARNING: Result is IDENTICAL to input - jewelry not applied!")
    else:
        # Find where the difference is
        diff = cv2.absdiff(result_arr, input_arr)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
        non_zero = cv2.findNonZero(diff_gray)

        if non_zero is not None:
            y_coords = non_zero[:, 0, 1]
            x_coords = non_zero[:, 0, 0]
            print(f"\n   Changes detected in image!")
            print(f"   Y range of changes: {y_coords.min()} to {y_coords.max()}")
            print(f"   X range of changes: {x_coords.min()} to {x_coords.max()}")
            print(f"   Center of changes: ({(x_coords.min() + x_coords.max())//2}, {(y_coords.min() + y_coords.max())//2})")

            # Check if changes are in expected location
            if landmarks:
                expected_y = landmarks['chin'][1] + int(landmarks['face_height'] * 0.20)
                actual_y_center = (y_coords.min() + y_coords.max()) // 2
                print(f"\n   Expected necklace Y: {expected_y}")
                print(f"   Actual change center Y: {actual_y_center}")
                print(f"   Difference: {abs(expected_y - actual_y_center)} pixels")

    print("\n" + "=" * 60)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 60)


def test_with_sample_image(image_path):
    """Test with a real image file."""
    print(f"\nTesting with image: {image_path}")

    if not os.path.exists(image_path):
        print(f"   ERROR: Image not found at {image_path}")
        return

    img = cv2.imread(image_path)
    if img is None:
        print(f"   ERROR: Could not read image")
        return

    engine = JewelryEngine()
    landmarks = engine.detect_face(img)

    if landmarks:
        print(f"   Face detected: {landmarks['face_bounds']}")
        print(f"   Chin: {landmarks['chin']}")
    else:
        print("   NO FACE DETECTED")

    debug_img = draw_debug_markers(img, landmarks)
    debug_path = image_path.replace('.', '_debug.')
    cv2.imwrite(debug_path, debug_img)
    print(f"   Debug saved to: {debug_path}")


if __name__ == "__main__":
    test_with_real_face_detection()
