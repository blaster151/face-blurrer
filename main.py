# pylint: disable=no-member

import os
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


class FaceBlurrer:
    def __init__(self):
        self.mp_face_detection = mp.solutions.face_detection
        # Three detectors with different settings
        self.face_detection_1 = self.mp_face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.2  # longer range
        )
        self.face_detection_2 = self.mp_face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.15  # close range
        )
        self.face_detection_3 = self.mp_face_detection.FaceDetection(
            model_selection=1,  # longer range with very low confidence
            min_detection_confidence=0.1,  # Even lower threshold
        )

    def detect_faces(self, image_rgb):
        all_detections = []

        # List of images to try detection on
        test_images = [
            ("original", image_rgb),
            # Brighten
            ("brightened", cv2.convertScaleAbs(image_rgb, alpha=1.5, beta=30)),
            # Increase contrast
            ("contrast", cv2.convertScaleAbs(image_rgb, alpha=1.3, beta=0)),
            # Histogram equalization
            ("equalized", self.equalize_histogram(image_rgb.copy())),
        ]

        print("\nTrying multiple detection approaches...")

        for name, test_image in test_images:
            for detector in [self.face_detection_1, self.face_detection_2, self.face_detection_3]:
                results = detector.process(test_image)
                if results.detections:
                    print(f"Found faces in {name} image")
                    for detection in results.detections:
                        bbox = detection.location_data.relative_bounding_box
                        is_new = True
                        # Check if this detection overlaps with previous ones
                        for existing in all_detections:
                            existing_bbox = existing.location_data.relative_bounding_box
                            if self.boxes_overlap(bbox, existing_bbox):
                                is_new = False
                                break
                        if is_new:
                            all_detections.append(detection)

        print(f"Total unique faces detected: {len(all_detections)}")
        return all_detections

    def boxes_overlap(self, box1, box2):  # Removed unused threshold parameter
        # Calculate intersection of boxes
        x1 = max(box1.xmin, box2.xmin)
        y1 = max(box1.ymin, box2.ymin)
        x2 = min(box1.xmin + box1.width, box2.xmin + box2.width)
        y2 = min(box1.ymin + box1.height, box2.ymin + box2.height)

        if x2 < x1 or y2 < y1:
            return False

        return True

    def equalize_histogram(self, img):
        # Convert to LAB color space
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)

        # Equalize the L channel
        l_eq = cv2.equalizeHist(l)

        # Merge channels
        lab_eq = cv2.merge([l_eq, a, b])

        # Convert back to RGB
        return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)

    def create_adaptive_hair_mask(self, height, width):
        # Create base mask
        mask = np.zeros((height, width), dtype=np.uint8)
        center = (width // 2, height // 2)

        # Base face oval (slightly wider)
        face_axes = (int(width // 2.1), int(height // 1.8))
        cv2.ellipse(mask, center, face_axes, 0, 0, 360, 255, -1)

        # More conservative hair region
        hair_center = (center[0], int(center[1] - height * 0.25))  # Reduced upward shift
        hair_axes = (int(width // 2.3), int(height * 0.7))  # Reduced height
        cv2.ellipse(mask, hair_center, hair_axes, 0, 0, 180, 255, -1)

        # Reduced top coverage
        top_center = (center[0], int(height * 0.3))  # Lower top point
        top_axes = (int(width // 2.5), int(height * 0.3))  # Reduced height
        cv2.ellipse(mask, top_center, top_axes, 0, 0, 180, 255, -1)

        # Smooth the mask edges
        mask = cv2.GaussianBlur(mask, (9, 9), 3)
        return mask

    def should_extend_blur(self, face_region):
        # Convert to grayscale
        gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)

        # Check top quarter of the image for significant edges/content
        top_quarter = gray[: gray.shape[0] // 4, :]
        
        # Split into left and right halves to check for uneven lighting
        left_half = top_quarter[:, :top_quarter.shape[1]//2]
        right_half = top_quarter[:, top_quarter.shape[1]//2:]
        
        # Calculate stats for both halves
        left_std = np.std(left_half)
        right_std = np.std(right_half)
        print(f"DEBUG: Left std: {left_std:.3f}, Right std: {right_std:.3f}")

        # If one side is much darker than the other, it might be background
        if abs(left_std - right_std) > 20:
            print("DEBUG: Uneven lighting detected - might be background interference")
            return False

        # Regular edge detection
        edges = cv2.Canny(top_quarter, 75, 150)
        edge_density = np.count_nonzero(edges) / edges.size

        # Overall contrast
        std_dev = np.std(top_quarter)

        # Look for dark pixels in the center top area only
        center_slice = top_quarter[:, top_quarter.shape[1]//4:3*top_quarter.shape[1]//4]
        dark_threshold = 60
        dark_pixels = np.sum(center_slice < dark_threshold) / center_slice.size

        print(f"DEBUG: Edge density: {edge_density:.3f}, Std dev: {std_dev:.3f}, Center dark pixels: {dark_pixels:.3f}")
        print(f"DEBUG: Thresholds - Edge density > 0.2, Std dev > 55, Dark pixels > 0.4")
        
        # Require higher std_dev AND significant dark pixels in center for hair detection
        should_extend = (edge_density > 0.2 or std_dev > 55) and dark_pixels > 0.4
        print(f"DEBUG: Should extend blur: {should_extend}")
        return should_extend

    def calculate_blur_height(self, face_region, base_height):
        # Convert to grayscale
        gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)

        # Analyze strips above the face
        extension_factor = 0.0
        strips = 5  # Check 5 strips above the face
        strip_height = base_height // 3

        for i in range(strips):
            start_y = max(0, -((i + 1) * strip_height))
            end_y = max(0, -(i * strip_height))
            if start_y == end_y:
                break

            strip = gray[start_y:end_y, :]
            if strip.size == 0:
                break

            # Calculate edge density for this strip
            edges = cv2.Canny(strip, 30, 150)
            edge_density = np.count_nonzero(edges) / edges.size

            # Calculate contrast for this strip
            std_dev = np.std(strip)

            # Add to extension factor based on content
            if edge_density > 0.1 or std_dev > 25:
                extension_factor += 0.3  # Each active strip adds 30%

        return min(1.5, max(0.8, extension_factor))  # Limit between 80% and 150%

    def detect_hair_region(self, image, face_bbox):
        x, y, width, height = face_bbox
        print(f"DEBUG: Original face bbox - x:{x}, y:{y}, w:{width}, h:{height}")

        # Look above the face
        search_height = int(height * 2)
        top_y = max(0, y - search_height)
        print(f"DEBUG: Searching for hair up to {search_height}px above face (y={top_y})")

        # Extract region above face
        hair_region = image[top_y : y + height // 2, x : x + width]
        if hair_region.size == 0:
            print("DEBUG: No space above face to check for hair")
            return None

        # Convert to HSV for better hair detection
        hsv = cv2.cvtColor(hair_region, cv2.COLOR_BGR2HSV)
        lower_bounds = np.array([0, 0, 0])
        upper_bounds = np.array([180, 255, 180])
        hair_mask = cv2.inRange(hsv, lower_bounds, upper_bounds)

        # Find the highest point with significant hair content
        rows = np.sum(hair_mask, axis=1)
        significant_rows = np.where(rows > width * 0.3)[0]
        
        if len(significant_rows) > 0:
            print(f"DEBUG: Found significant hair content in {len(significant_rows)} rows")
            result = (x, top_y, width, y - top_y + height // 2)
            print(f"DEBUG: New region with hair - x:{result[0]}, y:{result[1]}, w:{result[2]}, h:{result[3]}")
            return result
        
        print("DEBUG: No significant hair content found")
        return None

    def blur_faces(self, input_path, output_path):
        print(f"\nProcessing {os.path.basename(input_path)}")
        # Read image
        image = cv2.imread(input_path)
        if image is None:
            # Try alternate method for webp
            if input_path.lower().endswith(".webp"):
                print("Attempting alternate WebP reading method...")
                with open(input_path, "rb") as f:
                    image_data = f.read()
                image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)

        if image is None:
            print(f"Could not read image: {input_path}")
            return False

        print(f"Image shape: {image.shape}")  # Debug info

        # Convert to RGB for MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Use multi-pass detection
        detections = self.detect_faces(image_rgb)

        if detections:
            print(f"Found {len(detections)} faces")
            for i, detection in enumerate(detections):
                print(f"\nProcessing face #{i+1}")
                bbox = detection.location_data.relative_bounding_box
                h, w, _ = image.shape

                # Face region
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                print(f"DEBUG: Initial face dimensions - x:{x}, y:{y}, w:{width}, h:{height}")

                # Only detect hair region if needed
                face_region = image[y : y + height, x : x + width]
                print("\nChecking if blur should be extended:")
                should_extend = self.should_extend_blur(face_region)
                
                if should_extend:
                    print("DEBUG: Attempting to detect hair region")
                    hair_bbox = self.detect_hair_region(image, (x, y, width, height))
                    if hair_bbox is not None:
                        old_height = height
                        x, y = hair_bbox[0], hair_bbox[1]
                        height = (hair_bbox[1] + hair_bbox[3]) - y
                        print(f"DEBUG: Height increased from {old_height} to {height}")
                    else:
                        print("DEBUG: No hair region detected despite high edge content")
                else:
                    print("DEBUG: Using standard face region (no extension needed)")

                # Get the combined region
                face_region = image[y : y + height, x : x + width]
                print(f"DEBUG: Final region dimensions - x:{x}, y:{y}, w:{width}, h:{height}")

                # Create mask
                print("\nCreating blur mask:")
                if should_extend:  # Use the same decision we made earlier
                    print("DEBUG: Using extended adaptive hair mask")
                    mask = self.create_adaptive_hair_mask(height, width)
                else:
                    print("DEBUG: Using standard face mask")
                    mask = np.zeros((height, width), dtype=np.uint8)
                    center = (width // 2, height // 2)
                    # Make the standard mask slightly taller but still compact
                    face_axes = (int(width // 1.9), int(height // 1.4))
                    cv2.ellipse(mask, center, face_axes, 0, 0, 360, 255, -1)
                    # Add a small upper extension for forehead
                    top_center = (center[0], int(height * 0.35))
                    top_axes = (int(width // 2.2), int(height * 0.2))
                    cv2.ellipse(mask, top_center, top_axes, 0, 0, 180, 255, -1)

                print("DEBUG: Applying final blur")
                # Smooth the mask
                mask = cv2.GaussianBlur(mask, (31, 31), 10)
                mask_3d = np.stack([mask] * 3, axis=2) / 255.0

                # Apply strong blur
                color_preserved = cv2.GaussianBlur(face_region, (71, 71), 30)
                strong_blur = cv2.GaussianBlur(face_region, (201, 201), 100)
                blurred_face = cv2.addWeighted(color_preserved, 0.1, strong_blur, 0.9, 0)
                blurred_face = cv2.GaussianBlur(blurred_face, (151, 151), 70)

                # Apply the mask
                face_region[:] = (blurred_face * mask_3d + face_region * (1 - mask_3d)).astype(
                    np.uint8
                )

        cv2.imwrite(output_path, image)
        return True

    def process_directory(self, input_dir, output_dir):
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        for filename in os.listdir(input_dir):
            if filename.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                input_path = os.path.join(input_dir, filename)
                output_path = os.path.join(output_dir, f"blurred_{filename}")
                if self.blur_faces(input_path, output_path):
                    print(f"Successfully processed {filename}")
                else:
                    print(f"Failed to process {filename}")


# Example usage
if __name__ == "__main__":
    # Define input/output directories relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(script_dir, "input")
    output_dir = os.path.join(script_dir, "output")

    # Create directories if they don't exist
    Path(input_dir).mkdir(parents=True, exist_ok=True)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    blurrer = FaceBlurrer()
    blurrer.process_directory(input_dir, output_dir)
