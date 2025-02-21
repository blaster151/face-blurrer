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

        # Much taller hair region
        hair_center = (center[0], int(center[1] - height * 0.4))  # Move up more
        hair_axes = (int(width // 2.3), int(height * 1.0))  # Full height
        cv2.ellipse(mask, hair_center, hair_axes, 0, 0, 180, 255, -1)

        # Additional top coverage
        top_center = (center[0], int(height * 0.2))  # Very high up
        top_axes = (int(width // 2.5), int(height * 0.5))
        cv2.ellipse(mask, top_center, top_axes, 0, 0, 180, 255, -1)

        # Smooth the mask edges
        mask = cv2.GaussianBlur(mask, (9, 9), 3)
        return mask

    def should_extend_blur(self, face_region):
        # Convert to grayscale
        gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)

        # Check top third of the image for significant edges/content
        top_third = gray[: gray.shape[0] // 3, :]
        edges = cv2.Canny(top_third, 30, 150)
        edge_density = np.count_nonzero(edges) / edges.size

        # Check for significant contrast in top portion
        std_dev = np.std(top_third)

        # Return True if there's significant hair/feature content above face
        return edge_density > 0.1 or std_dev > 25

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

        # Look above the face
        search_height = int(height * 2)  # Look up to 2x face height
        top_y = max(0, y - search_height)

        # Extract region above face
        hair_region = image[top_y : y + height // 2, x : x + width]
        if hair_region.size == 0:
            return None

        # Convert to HSV for better hair detection
        hsv = cv2.cvtColor(hair_region, cv2.COLOR_BGR2HSV)

        # Create mask for likely hair pixels
        lower_bounds = np.array([0, 0, 0])
        upper_bounds = np.array([180, 255, 180])
        hair_mask = cv2.inRange(hsv, lower_bounds, upper_bounds)

        # Find the highest point with significant hair content
        rows = np.sum(hair_mask, axis=1)
        significant_rows = np.where(rows > width * 0.3)[0]

        if len(significant_rows) > 0:
            return (x, top_y, width, y - top_y + height // 2)
        return None

    def blur_faces(self, input_path, output_path):
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
            for detection in detections:
                bbox = detection.location_data.relative_bounding_box
                h, w, _ = image.shape

                # Face region
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)

                # Detect hair region
                hair_bbox = self.detect_hair_region(image, (x, y, width, height))

                if hair_bbox is not None:
                    # Combine face and hair regions
                    x, y = hair_bbox[0], hair_bbox[1]
                    height = (hair_bbox[1] + hair_bbox[3]) - y

                # Get the combined region
                face_region = image[y : y + height, x : x + width]

                # Create mask
                mask = np.zeros((height, width), dtype=np.uint8)

                # Fill the entire region
                cv2.rectangle(mask, (0, 0), (width, height), 255, -1)

                # Add oval bottom for better transition
                bottom_center = (width // 2, height - height // 4)
                bottom_axes = (int(width // 1.8), int(height // 2))
                cv2.ellipse(mask, bottom_center, bottom_axes, 0, 0, 180, 255, -1)

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
