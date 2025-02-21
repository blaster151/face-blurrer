import os
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import mediapipe as mp
import shutil  # Add shutil import

class ImageSquarifier:
    def __init__(self):
        # Initialize face detection
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1,  # Use full-range model
            min_detection_confidence=0.5
        )
        
        # Parameters
        self.target_size = 1000  # Output size (1000x1000)
        self.edge_threshold = 100  # For edge detection
        self.content_weight = 1.0  # Weight for content preservation
        self.structure_weight = 0.8  # Weight for structural elements
        self.face_weight = 2.0  # Weight for face regions
        self.text_weight = 1.5  # Weight for text regions
        self.vertical_text_threshold = 0.3  # Threshold for detecting vertical text layouts
        self.white_threshold = 250  # Threshold for detecting white background
        self.white_coverage = 0.95  # Required coverage for white border detection
        
    def detect_faces(self, image):
        """Detect faces in the image and return their bounding boxes."""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_image)
        faces = []
        
        if results.detections:
            height, width = image.shape[:2]
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * width)
                y = int(bbox.ymin * height)
                w = int(bbox.width * width)
                h = int(bbox.height * height)
                faces.append((x, y, w, h))
        
        return faces

    def detect_vertical_text(self, image):
        """Detect if image has significant vertical text layout."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, self.edge_threshold, self.edge_threshold * 2)
        
        # Check right side of image for strong vertical elements
        height, width = image.shape[:2]
        right_region = edges[:, int(width * 0.8):]
        
        # Calculate vertical vs horizontal edge strength
        vertical_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        horizontal_kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        
        vertical_strength = cv2.filter2D(right_region.astype(float), -1, vertical_kernel)
        horizontal_strength = cv2.filter2D(right_region.astype(float), -1, horizontal_kernel)
        
        v_score = np.sum(np.abs(vertical_strength))
        h_score = np.sum(np.abs(horizontal_strength))
        
        return v_score > h_score * self.vertical_text_threshold

    def detect_white_borders(self, image):
        """Detect if image has white borders that can be extended."""
        height, width = image.shape[:2]
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Check left and right borders
        left_border = gray[:, :10]
        right_border = gray[:, -10:]
        
        # Calculate percentage of white pixels
        left_white = np.mean(left_border > self.white_threshold)
        right_white = np.mean(right_border > self.white_threshold)
        
        # Check if both borders are predominantly white
        has_white_borders = (left_white > self.white_coverage and 
                           right_white > self.white_coverage)
        
        return has_white_borders

    def extend_white_background(self, image):
        """Extend white background to make image square with slight zoom on central content."""
        height, width = image.shape[:2]
        
        if width == height:
            return image
        
        # Create square white canvas
        target_size = max(width, height)
        square = np.full((target_size, target_size, 3), 255, dtype=np.uint8)
        
        # Calculate zoom factor (12% larger)
        zoom_factor = 1.12
        
        # Calculate new dimensions after zoom
        if width > height:
            new_height = int(min(height * zoom_factor, width))  # Don't exceed width
            new_width = width
        else:
            new_width = int(min(width * zoom_factor, height))  # Don't exceed height
            new_height = height
        
        # Resize image with zoom
        zoomed = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
        
        # Calculate padding to center the zoomed image
        if width > height:
            y_offset = (width - new_height) // 2
            square[y_offset:y_offset+new_height, :] = zoomed
        else:
            x_offset = (height - new_width) // 2
            square[:, x_offset:x_offset+new_width] = zoomed
        
        return square

    def compute_saliency_map(self, image, faces):
        """Create a saliency map combining edges, faces, text, and center bias."""
        height, width = image.shape[:2]
        
        # Edge detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, self.edge_threshold, self.edge_threshold * 2)
        edge_map = edges.astype(float) / 255.0
        
        # Center bias (gaussian)
        y, x = np.ogrid[:height, :width]
        center_y, center_x = height / 2, width / 2
        center_map = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * (min(width, height) / 4)**2))
        
        # Face regions
        face_map = np.zeros((height, width), dtype=float)
        for (x, y, w, h) in faces:
            # Add padding around faces
            pad_x = int(w * 0.2)
            pad_y = int(h * 0.2)
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(width, x + w + pad_x)
            y2 = min(height, y + h + pad_y)
            face_map[y1:y2, x1:x2] = 1.0
        
        # Text detection using EAST text detector or simple edge-based approach
        text_map = np.zeros((height, width), dtype=float)
        
        # Simple text detection based on horizontal edges and contrast
        dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
        dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
        
        # Get gradient magnitude and direction
        mag = np.sqrt(dx*dx + dy*dy)
        angle = np.arctan2(dy, dx) * 180 / np.pi
        
        # Look for horizontal-ish lines (common in text)
        horizontal_mask = np.abs(angle) < 30
        text_edges = (mag > 50) & horizontal_mask
        
        # Dilate to connect nearby text regions
        kernel = np.ones((3,15), np.uint8)  # Horizontal kernel
        text_regions = cv2.dilate(text_edges.astype(np.uint8), kernel)
        
        # Add extra weight to bottom area where titles usually appear
        title_zone = np.zeros((height, width), dtype=float)
        title_height = int(height * 0.2)  # Bottom 20%
        title_zone[-title_height:, :] = 1.0
        
        # Fade the title zone weight from bottom to top
        for i in range(title_height):
            title_zone[-title_height+i, :] = i / title_height
        
        # Combine text detection with title zone
        text_map = text_regions.astype(float) * (1.0 + title_zone)
        
        # Check for vertical text layout
        has_vertical_text = self.detect_vertical_text(image)
        if has_vertical_text:
            # Add weight to right side of image
            right_text_map = np.zeros((height, width), dtype=float)
            right_text_map[:, int(width * 0.8):] = 1.0
            text_map = np.maximum(text_map, right_text_map)
        
        # Combine maps with adjusted weights
        saliency_map = (edge_map * self.structure_weight + 
                       center_map * self.content_weight +
                       face_map * self.face_weight +
                       text_map * 2.0)  # Increased weight for text
        
        return cv2.normalize(saliency_map, None, 0, 1, cv2.NORM_MINMAX)

    def find_best_square_crop(self, image, saliency_map):
        """Find the best square crop based on saliency and layout."""
        height, width = image.shape[:2]
        size = min(width, height)
        
        if width == height:
            return image
            
        # Check for vertical text layout
        has_vertical_text = self.detect_vertical_text(image)
        
        # Detect text regions
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        text_regions = self._detect_text_regions(gray)
        
        best_score = -1
        best_x = 0
        best_y = 0
        
        # For wide images, slide the crop window horizontally
        if width > height:
            step = max(1, (width - size) // 20)  # Reduce computation by using steps
            for x in range(0, width - size + 1, step):
                # Check if this crop would cut through text
                crop_region = text_regions[:, x:x+size]
                text_cut_score = 1.0
                if np.any(crop_region):
                    # Penalize crops that cut through text
                    text_cut_score = 0.5
                
                # Increase weight of right side if vertical text detected
                if has_vertical_text:
                    right_bias = 1.0 if x + size > width * 0.7 else 0.5
                else:
                    right_bias = 1.0
                    
                score = np.sum(saliency_map[:, x:x+size]) * right_bias * text_cut_score
                if score > best_score:
                    best_score = score
                    best_x = x
            
            # Fine-tune around best position
            x_start = max(0, best_x - step)
            x_end = min(width - size, best_x + step)
            for x in range(x_start, x_end + 1):
                crop_region = text_regions[:, x:x+size]
                text_cut_score = 1.0 if not np.any(crop_region) else 0.5
                
                if has_vertical_text:
                    right_bias = 1.0 if x + size > width * 0.7 else 0.5
                else:
                    right_bias = 1.0
                
                score = np.sum(saliency_map[:, x:x+size]) * right_bias * text_cut_score
                if score > best_score:
                    best_score = score
                    best_x = x
            
            return image[:, best_x:best_x+size]
        
        # For tall images, use intelligent vertical cropping
        else:
            step = max(1, (height - size) // 20)
            
            for y in range(0, height - size + 1, step):
                # Check if this crop would cut through text
                crop_region = text_regions[y:y+size, :]
                text_cut_score = 1.0 if not np.any(crop_region) else 0.5
                
                score = np.sum(saliency_map[y:y+size, :]) * text_cut_score
                if score > best_score:
                    best_score = score
                    best_y = y
            
            # Fine-tune around best position
            y_start = max(0, best_y - step)
            y_end = min(height - size, best_y + step)
            for y in range(y_start, y_end + 1):
                crop_region = text_regions[y:y+size, :]
                text_cut_score = 1.0 if not np.any(crop_region) else 0.5
                
                score = np.sum(saliency_map[y:y+size, :]) * text_cut_score
                if score > best_score:
                    best_score = score
                    best_y = y
            
            return image[best_y:best_y+size, :]

    def _detect_text_regions(self, gray_image):
        """Detect regions likely to contain text."""
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 11, 2)
        
        # Create kernels for morphological operations
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
        
        # Detect horizontal and vertical text
        horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
        vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)
        
        # Combine detections
        text_regions = cv2.bitwise_or(horizontal, vertical)
        
        # Dilate to connect nearby text
        kernel = np.ones((3,3), np.uint8)
        text_regions = cv2.dilate(text_regions, kernel, iterations=2)
        
        return text_regions

    def extend_background(self, image, target_size):
        """Extend the background to make the image square."""
        height, width = image.shape[:2]
        
        if width == height:
            return image
        
        # Create square canvas
        size = max(width, height)
        square = np.zeros((size, size, 3), dtype=np.uint8)
        
        # Calculate padding
        if width > height:
            y_offset = (width - height) // 2
            square[y_offset:y_offset+height, :] = image
        else:
            x_offset = (height - width) // 2
            square[:, x_offset:x_offset+width] = image
        
        # Fill extended areas using inpainting
        mask = np.zeros((size, size), dtype=np.uint8)
        if width > height:
            mask[:y_offset] = 255
            mask[y_offset+height:] = 255
        else:
            mask[:, :x_offset] = 255
            mask[:, x_offset+width:] = 255
        
        result = cv2.inpaint(square, mask, 3, cv2.INPAINT_TELEA)
        return result

    def detect_uniform_borders(self, image):
        """Detect if image has uniform-colored borders that can be extended."""
        height, width = image.shape[:2]
        
        # Check borders (10 pixels from each edge)
        left_border = image[:, :10]
        right_border = image[:, -10:]
        top_border = image[:10, :]
        bottom_border = image[-10:, :]
        
        # Calculate color variance in each border
        def color_variance(region):
            return np.mean([np.var(region[:,:,c]) for c in range(3)])
        
        # Check if borders are uniform (low variance) and similar to each other
        variance_threshold = 50.0
        left_var = color_variance(left_border)
        right_var = color_variance(right_border)
        top_var = color_variance(top_border)
        bottom_var = color_variance(bottom_border)
        
        # Get average colors
        left_color = np.mean(left_border, axis=(0,1))
        right_color = np.mean(right_border, axis=(0,1))
        top_color = np.mean(top_border, axis=(0,1))
        bottom_color = np.mean(bottom_border, axis=(0,1))
        
        # Check color difference between borders
        def color_difference(c1, c2):
            return np.sqrt(np.sum((c1 - c2) ** 2))
        
        color_threshold = 30.0
        colors_match = all([
            color_difference(left_color, right_color) < color_threshold,
            color_difference(top_color, bottom_color) < color_threshold,
            color_difference(left_color, top_color) < color_threshold
        ])
        
        # All borders should be uniform and match each other
        borders_uniform = all([
            left_var < variance_threshold,
            right_var < variance_threshold,
            top_var < variance_threshold,
            bottom_var < variance_threshold
        ])
        
        return borders_uniform and colors_match, left_color

    def extend_uniform_background(self, image):
        """Extend uniform background to make image square with enhanced zoom."""
        height, width = image.shape[:2]
        
        if width == height:
            return image
        
        # Create square canvas with detected border color
        target_size = max(width, height)
        _, border_color = self.detect_uniform_borders(image)
        square = np.full((target_size, target_size, 3), border_color, dtype=np.uint8)
        
        # Calculate zoom factor (25% larger)
        zoom_factor = 1.25
        
        # Calculate new dimensions while preserving aspect ratio
        if height > width:
            # For tall images, scale width up but maintain aspect ratio
            new_width = int(min(width * zoom_factor, height))
            scale = new_width / width
            new_height = min(int(height * scale), target_size)
        else:
            # For wide images, scale height up but maintain aspect ratio
            new_height = int(min(height * zoom_factor, width))
            scale = new_height / height
            new_width = min(int(width * scale), target_size)
        
        # Ensure we don't exceed the target size
        if new_width > target_size or new_height > target_size:
            # Scale down to fit if necessary
            scale = min(target_size / new_width, target_size / new_height)
            new_width = int(new_width * scale)
            new_height = int(new_height * scale)
        
        # Resize image with zoom while preserving aspect ratio
        zoomed = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
        
        # Calculate padding to center the zoomed image
        x_offset = (target_size - new_width) // 2
        y_offset = (target_size - new_height) // 2
        
        # Place the zoomed image in the center of the square canvas
        square[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = zoomed
        
        return square

    def copy_to_temp(self, source_path, filename):
        """Copy a file to the temp directory while preserving its name."""
        temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp")
        Path(temp_dir).mkdir(parents=True, exist_ok=True)
        temp_path = os.path.join(temp_dir, f"3_square_{filename}")
        shutil.copy2(source_path, temp_path)
        print(f"Copied to temp: {temp_path}")

    def process_image(self, input_path, output_path):
        """Process a single image to make it square while preserving content."""
        print(f"\nProcessing {os.path.basename(input_path)}")
        
        # Read image
        image = cv2.imread(input_path)
        if image is None:
            print(f"Could not read image: {input_path}")
            return False
        
        height, width = image.shape[:2]
        aspect_ratio = width / height
        print(f"Original dimensions: {width}x{height}, aspect ratio: {aspect_ratio:.2f}")
        
        # Check for uniform borders that can be extended
        has_uniform_borders, _ = self.detect_uniform_borders(image)
        
        # Detect faces
        faces = self.detect_faces(image)
        print(f"Found {len(faces)} faces")
        
        # Choose strategy based on image characteristics
        if has_uniform_borders and aspect_ratio != 1.0:
            print("Using uniform border extension strategy")
            result = self.extend_uniform_background(image)
        elif 0.8 <= aspect_ratio <= 1.2:
            # Near square - use simple cropping
            print("Using simple crop strategy (near square)")
            saliency_map = self.compute_saliency_map(image, faces)
            result = self.find_best_square_crop(image, saliency_map)
        elif aspect_ratio < 0.5 or aspect_ratio > 2.0:
            # Extreme aspect ratio - use background extension
            print("Using background extension strategy (extreme aspect ratio)")
            result = self.extend_background(image, self.target_size)
        else:
            # Standard poster ratio - use intelligent cropping
            print("Using intelligent crop strategy (standard ratio)")
            saliency_map = self.compute_saliency_map(image, faces)
            result = self.find_best_square_crop(image, saliency_map)
        
        # Resize to target size
        result = cv2.resize(result, (self.target_size, self.target_size))
        
        # Save result
        cv2.imwrite(output_path, result)
        print(f"Saved square image to {output_path}")
        return True

    def process_directory(self, input_dir, output_dir):
        """Process all images in the input directory."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        for filename in os.listdir(input_dir):
            if filename.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                input_path = os.path.join(input_dir, filename)
                output_path = os.path.join(output_dir, f"square_{filename}")
                if self.process_image(input_path, output_path):
                    print(f"Successfully processed {filename}")
                    self.copy_to_temp(output_path, filename)  # Copy to temp
                else:
                    print(f"Failed to process {filename}")


if __name__ == "__main__":
    # Define input/output directories relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(script_dir, "final")  # Use output from name concealment
    output_dir = os.path.join(script_dir, "square")  # Final square images

    # Create directories if they don't exist
    Path(input_dir).mkdir(parents=True, exist_ok=True)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    squarifier = ImageSquarifier()
    squarifier.process_directory(input_dir, output_dir) 