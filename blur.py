# pylint: disable=no-member

import os
from pathlib import Path
import shutil  # Add shutil import
import json
import time
from datetime import datetime

import cv2
import numpy as np


class FaceBlurrer:
    def __init__(self, debug=True):
        # Load the pre-trained Haar Cascade classifier for face detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Detection parameters
        self.scale_factors = [1.0]  # Single scale for more reliable detection
        self.min_neighbors = 5  # More strict neighbor requirement
        self.min_face_size = (50, 50)  # Slightly larger minimum size
        self.max_face_size = None  # No maximum size limit
        
        # Blur parameters
        self.max_blur_kernel = 99  # Maximum blur kernel size
        self.min_blur_kernel = 51  # Minimum blur kernel size
        self.margin_factor = 0.2   # Margin around face as percentage of face size
        self.blur_scale = 0.5      # Scale factor for blur kernel relative to face size
        
        # Debug flag
        self.debug = debug
        
        # Initialize metrics
        self.metrics = {}
        self._load_version()

    def detect_faces(self, img):
        """Detect faces in an image using Haar Cascade classifier."""
        if self.debug:
            print("\nStarting face detection...")
        
        height, width = img.shape[:2]
        all_faces = []
        
        # Convert to grayscale for detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Try different preprocessing enhancements
        preprocessed_images = [
            ("Original", gray),
            ("Brightened", cv2.convertScaleAbs(gray, alpha=1.5, beta=30)),
            ("Contrast", cv2.convertScaleAbs(gray, alpha=1.8, beta=0)),
            ("Equalized", cv2.equalizeHist(gray))
        ]
        
        # Try detection with each preprocessing method
        for preprocess_name, processed_img in preprocessed_images:
            if self.debug:
                print(f"\nTrying {preprocess_name} pass.")
            
            # Detect faces with OpenCV
            faces = self.face_cascade.detectMultiScale(
                processed_img,
                scaleFactor=1.1,
                minNeighbors=self.min_neighbors,
                minSize=self.min_face_size,
                maxSize=self.max_face_size,
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            # Add detected faces to the list
            for (x, y, w, h) in faces:
                if self.debug:
                    print(f"Found face at x={x}, y={y}, w={w}, h={h}")
                all_faces.append((x, y, w, h))
        
        # Remove overlapping detections
        unique_faces = self._remove_overlaps(all_faces, overlap_thresh=0.5)
        
        if self.debug:
            print("\nFace regions by area:")
            # Sort faces by area for debugging
            face_areas = [(x, y, w, h, w*h) for (x, y, w, h) in all_faces]
            face_areas.sort(key=lambda x: x[4], reverse=True)
            for i, (x, y, w, h, area) in enumerate(face_areas, 1):
                print(f"  {i}. Area: {area}px² at ({x}, {y}) size {w}x{w}")
            
            print(f"\nKept {len(unique_faces)} faces after overlap removal")
            for i, (x, y, w, h) in enumerate(unique_faces, 1):
                print(f"  Face {i}: ({x}, {y}) size {w}x{h}")
        
        return unique_faces

    def _remove_overlaps(self, faces, overlap_thresh=0.5):
        """Remove overlapping face detections."""
        if not faces:
            return []
        
        # Convert to numpy array for easier processing
        boxes = np.array(faces)
        pick = []
        
        # Get coordinates
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 0] + boxes[:, 2]
        y2 = boxes[:, 1] + boxes[:, 3]
        
        # Compute areas
        area = (x2 - x1) * (y2 - y1)
        idxs = np.argsort(y2)
        
        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)
            
            # Find overlapping boxes
            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])
            
            # Compute overlap
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            overlap = (w * h) / area[idxs[:last]]
            
            # Delete overlapping boxes
            idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlap_thresh)[0])))
        
        return boxes[pick].tolist()

    def create_blur_mask(self, height, width, extended=False):
        """Create an elliptical mask for face blurring."""
        mask = np.zeros((height, width), dtype=np.uint8)
        center = (width // 2, height // 2)
        
        if extended:
            # Extended mask for hair/head coverage
            face_axes = (int(width // 1.8), int(height // 1.6))
            cv2.ellipse(mask, center, face_axes, 0, 0, 360, 255, -1)
            
            # Add top extension for hair
            hair_center = (center[0], int(center[1] - height * 0.2))
            hair_axes = (int(width // 2.0), int(height * 0.6))
            cv2.ellipse(mask, hair_center, hair_axes, 0, 0, 180, 255, -1)
        else:
            # Standard face mask
            face_axes = (int(width // 1.9), int(height // 1.7))
            cv2.ellipse(mask, center, face_axes, 0, 0, 360, 255, -1)
        
        # Smooth the mask edges
        mask = cv2.GaussianBlur(mask, (31, 31), 10)
        return mask

    def should_extend_blur(self, face_region):
        """Determine if blur region should be extended for hair/head coverage."""
        gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        
        # Check top quarter of the image
        top_quarter = gray[:gray.shape[0] // 4, :]
        
        # Calculate edge density
        edges = cv2.Canny(top_quarter, 50, 150)
        edge_density = np.count_nonzero(edges) / edges.size
        
        # Calculate contrast
        std_dev = np.std(top_quarter)
        
        # Check for dark pixels in center top area
        center_slice = top_quarter[:, top_quarter.shape[1]//4:3*top_quarter.shape[1]//4]
        dark_pixels = np.sum(center_slice < 60) / center_slice.size
        
        return (edge_density > 0.15 or std_dev > 45) and dark_pixels > 0.3

    def _load_version(self):
        """Load or initialize version information."""
        self.version = {
            "detector": "haarcascade_frontalface_default",
            "version": cv2.__version__,
            "last_updated": datetime.now().isoformat()
        }

    def _update_metrics(self, image_name, faces, method="haarcascade", quality="default"):
        """Update metrics for face detection."""
        if image_name not in self.metrics:
            self.metrics[image_name] = []
        
        # Add new detection result
        self.metrics[image_name].append({
            "timestamp": datetime.now().isoformat(),
            "faces_detected": len(faces),
            "face_dimensions": [(w, h) for _, _, w, h in faces],
            "method": method,
            "quality": quality,
            "version": self.version
        })

    def blur_faces(self, input_path, output_path):
        """Detect and blur faces in an image."""
        try:
            start_time = time.time()
            
            # Read the image
            img = cv2.imread(input_path)
            if img is None:
                print(f"Failed to read image: {input_path}")
                return False
                
            # Get image dimensions
            height, width = img.shape[:2]
            
            # Create a copy for blurring
            blurred = img.copy()
            
            # Process each face
            faces = self.detect_faces(img)
            if len(faces) == 0:
                print(f"No faces found in {input_path}")
                return False
                
            print(f"Found {len(faces)} faces")
            
            for i, (x, y, w, h) in enumerate(faces):
                print(f"\nProcessing face #{i+1}")
                
                # Calculate margin based on face size
                margin = int(min(w, h) * self.margin_factor)  # 20% margin
                
                # Calculate region bounds with margin
                x1 = max(0, x - margin)
                y1 = max(0, y - margin)
                x2 = min(width, x + w + margin)
                y2 = min(height, y + h + margin)
                
                # Extract face region
                face_region = img[y1:y2, x1:x2]
                region_height, region_width = face_region.shape[:2]
                
                # Calculate blur kernel size proportional to face size
                base_kernel_size = int(min(w, h) * self.blur_scale)  # 50% of face size
                kernel_size = min(max(base_kernel_size, self.min_blur_kernel), self.max_blur_kernel)
                kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size
                
                # Create elliptical mask centered on the face
                mask = np.zeros((region_height, region_width), dtype=np.uint8)
                center = ((x2-x1)//2, (y2-y1)//2)
                
                # Make axes slightly smaller than the region for tighter masking
                axes = (int((x2-x1)//2 * 0.9), int((y2-y1)//2 * 0.9))
                cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
                
                try:
                    # Apply Gaussian blur
                    blurred_region = cv2.GaussianBlur(face_region, (kernel_size, kernel_size), kernel_size//6)
                    
                    # Smooth the mask edges
                    mask = cv2.GaussianBlur(mask, (9, 9), 3)
                    mask = mask.astype(float) / 255
                    mask = mask[..., np.newaxis]
                    
                    # Combine original and blurred using the mask
                    result_region = (blurred_region * mask + face_region * (1 - mask)).astype(np.uint8)
                    blurred[y1:y2, x1:x2] = result_region
                    
                except cv2.error as e:
                    print(f"Error applying blur to face {i+1}: {e}")
                    continue
            
            # Update metrics
            processing_time = round(time.time() - start_time, 2)
            self._update_metrics(
                os.path.basename(input_path),
                faces,
                method="haarcascade",
                quality="default"
            )
            
            # Save the blurred image
            cv2.imwrite(output_path, blurred)
            print(f"Successfully processed {input_path}")
            
            # Copy to temp directory for reference
            temp_path = os.path.join("temp", f"{i+1}_blur_blurred_{os.path.basename(input_path)}")
            cv2.imwrite(temp_path, blurred)
            print(f"Copied to temp: {os.path.abspath(temp_path)}")
            
            return True
            
        except Exception as e:
            print(f"Error processing {input_path}: {e}")
            return False

    def copy_to_temp(self, source_path, filename):
        """Copy a file to the temp directory while preserving its name."""
        temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp")
        Path(temp_dir).mkdir(parents=True, exist_ok=True)
        temp_path = os.path.join(temp_dir, f"1_blur_{filename}")
        shutil.copy2(source_path, temp_path)
        print(f"Copied to temp: {temp_path}")

    def process_directory(self, input_dir, output_dir):
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        for filename in os.listdir(input_dir):
            if filename.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                input_path = os.path.join(input_dir, filename)
                output_path = os.path.join(output_dir, f"blurred_{filename}")
                if self.blur_faces(input_path, output_path):
                    print(f"Successfully processed {filename}")
                    self.copy_to_temp(output_path, f"blurred_{filename}")  # Copy to temp
                else:
                    print(f"Failed to process {filename}")


if __name__ == "__main__":
    # Process all images in the input directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(script_dir, "input")
    output_dir = os.path.join(script_dir, "output")

    # Create directories if they don't exist
    Path(input_dir).mkdir(parents=True, exist_ok=True)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    blurrer = FaceBlurrer(debug=True)
    blurrer.process_directory(input_dir, output_dir)
