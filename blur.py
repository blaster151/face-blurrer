# pylint: disable=no-member

import os
from pathlib import Path
import shutil  # Add shutil import

import cv2
import mediapipe as mp
import numpy as np


class FaceBlurrer:
    def __init__(self, debug=True):
        # Initialize face detection with both front and side profile models
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1,  # Use full-range model
            min_detection_confidence=0.2  # Slightly increased confidence threshold
        )
        
        self.debug = debug  # Control output verbosity
        self.min_confidence = 0.2
        self.min_face_size = 20
        self.detection_passes = [
            self._period_enhanced_pass,  # Try period-enhanced first
            self._romcom_enhanced_pass,  # Add new pass for romantic comedies
            self._original_pass,
            self._brightened_pass,
            self._contrast_pass,
            self._equalized_pass,
            self._dark_enhanced_pass,     # Enhanced for backlit scenes
            self._detect_silhouettes      # New pass specifically for silhouettes
        ]
        self.MIN_FACE_SIZE = 20

    def _original_pass(self, image):
        """Original image pass."""
        return self._detect_faces(image)

    def _brightened_pass(self, image):
        """Brightened image pass."""
        brightened = cv2.convertScaleAbs(image, alpha=1.5, beta=30)
        return self._detect_faces(brightened)

    def _contrast_pass(self, image):
        """Contrast enhanced pass."""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        enhanced = cv2.merge((cl,a,b))
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        return self._detect_faces(enhanced)

    def _equalized_pass(self, image):
        """Histogram equalized pass."""
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        channels = cv2.split(ycrcb)
        cv2.equalizeHist(channels[0], channels[0])
        cv2.merge(channels, ycrcb)
        equalized = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        return self._detect_faces(equalized)

    def _sharpened_pass(self, image):
        """Sharpen the image to enhance facial features."""
        kernel = np.array([[-1,-1,-1],
                         [-1, 9,-1],
                         [-1,-1,-1]])
        sharpened = cv2.filter2D(image, -1, kernel)
        return self._detect_faces(sharpened)

    def _edge_enhanced_pass(self, image):
        """Enhance edges to improve profile detection."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        enhanced = cv2.addWeighted(image, 1.0, cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), 0.5, 0)
        return self._detect_faces(enhanced)

    def _flipped_pass(self, image):
        """Try detecting faces in horizontally flipped image."""
        flipped = cv2.flip(image, 1)  # Horizontal flip
        faces = self._detect_faces(flipped)
        if faces:
            # Adjust coordinates for the flip
            width = image.shape[1]
            return [(width - (x + w), y, w, h) for x, y, w, h in faces]
        return []

    def _scaled_pass(self, image):
        """Try detecting faces at different scales."""
        faces = []
        height, width = image.shape[:2]
        
        scales = [0.5, 0.75, 1.5, 2.0]  # Try different scales
        for scale in scales:
            # Calculate new dimensions
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            # Resize image
            scaled = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            detected = self._detect_faces(scaled)
            
            # Adjust coordinates back to original scale
            if detected:
                for (x, y, w, h) in detected:
                    orig_x = int(x / scale)
                    orig_y = int(y / scale)
                    orig_w = int(w / scale)
                    orig_h = int(h / scale)
                    faces.append((orig_x, orig_y, orig_w, orig_h))
        
        return faces

    def _rotated_pass(self, image):
        """Try detecting faces at different angles."""
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        faces = []
        
        # Try more angles for profile detection
        for angle in [-45, -30, -15, 15, 30, 45]:  # Extended angle range
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(image, matrix, (width, height))
            detected = self._detect_faces(rotated)
            
            # Transform detected coordinates back
            if detected:
                inv_matrix = cv2.getRotationMatrix2D(center, -angle, 1.0)
                for (x, y, w, h) in detected:
                    # Transform bounding box corners
                    corners = np.array([
                        [x, y],
                        [x + w, y],
                        [x, y + h],
                        [x + w, y + h]
                    ], dtype=np.float32)
                    
                    # Reshape for transformation
                    corners = corners.reshape(-1, 1, 2)
                    transformed = cv2.transform(corners, inv_matrix)
                    
                    # Get new bounding box
                    transformed = transformed.reshape(-1, 2)
                    min_x = max(0, int(np.min(transformed[:, 0])))
                    min_y = max(0, int(np.min(transformed[:, 1])))
                    max_x = min(width, int(np.max(transformed[:, 0])))
                    max_y = min(height, int(np.max(transformed[:, 1])))
                    
                    faces.append((min_x, min_y, max_x - min_x, max_y - min_y))
        
        return faces

    def _period_enhanced_pass(self, image):
        """Enhanced detection specifically for period costumes and formal poses."""
        if self.debug:
            print("\nRunning period-enhanced detection pass")
        faces = []
        height, width = image.shape[:2]
        
        # First try different scales since the image might be too large
        scales = [0.25, 0.5, 1.0] if max(width, height) > 1000 else [1.0]
        
        if self.debug:
            print(f"Image dimensions: {width}x{height}")
            print(f"Testing {len(scales)} scales: {scales}")
        
        for scale in scales:
            if self.debug:
                print(f"\nTrying scale: {scale}")
            if scale != 1.0:
                scaled = cv2.resize(image, (int(width * scale), int(height * scale)))
                if self.debug:
                    print(f"Scaled dimensions: {scaled.shape[1]}x{scaled.shape[0]}")
            else:
                scaled = image
            
            # Try multiple color spaces and enhancements
            enhancements = []
            
            # 1. LAB color space with CLAHE
            lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            l_enhanced = clahe.apply(l)
            lab_enhanced = cv2.merge([l_enhanced, a, b])
            enhancements.append(cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR))
            
            # 2. YCrCb color space with histogram equalization
            ycrcb = cv2.cvtColor(scaled, cv2.COLOR_BGR2YCrCb)
            y, cr, cb = cv2.split(ycrcb)
            y_eq = cv2.equalizeHist(y)
            ycrcb_enhanced = cv2.merge([y_eq, cr, cb])
            enhancements.append(cv2.cvtColor(ycrcb_enhanced, cv2.COLOR_YCrCb2BGR))
            
            # 3. High contrast version
            high_contrast = cv2.convertScaleAbs(scaled, alpha=1.5, beta=0)
            enhancements.append(high_contrast)
            
            # 4. Edge-preserved smoothing
            smooth = cv2.edgePreservingFilter(scaled, flags=1, sigma_s=60, sigma_r=0.4)
            enhancements.append(smooth)
            
            # Try detection on each enhancement
            for i, enhanced in enumerate(enhancements):
                if self.debug:
                    print(f"  Trying enhancement #{i+1}")
                
                # Try multiple detection passes with different parameters
                detection_params = [
                    (1, 0.15),  # Model 1 with lower confidence
                    (1, 0.2),   # Model 1 with medium confidence
                    (0, 0.2),   # Model 0 (faster) with medium confidence
                ]
                
                for model_selection, min_detection_confidence in detection_params:
                    with self.mp_face_detection.FaceDetection(
                        model_selection=model_selection,
                        min_detection_confidence=min_detection_confidence
                    ) as face_detection:
                        results = face_detection.process(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
                        
                        if results.detections:
                            h, w = enhanced.shape[:2]
                            for detection in results.detections:
                                bbox = detection.location_data.relative_bounding_box
                                x = int(bbox.xmin * w)
                                y = int(bbox.ymin * h)
                                w = int(bbox.width * w)
                                h = int(bbox.height * h)
                                
                                # Skip very small detections
                                if w < 20 or h < 20:
                                    continue
                                    
                                # Skip detections with extreme aspect ratios
                                aspect_ratio = w / h
                                if aspect_ratio < 0.5 or aspect_ratio > 2.0:
                                    continue
                                
                                # Scale coordinates back if needed
                                if scale != 1.0:
                                    x = int(x / scale)
                                    y = int(y / scale)
                                    w = int(w / scale)
                                    h = int(h / scale)
                                
                                faces.append((x, y, w, h))
                                if self.debug:
                                    print(f"    Found face at x={x}, y={y}, w={w}, h={h}")
        
        return faces

    def _romcom_enhanced_pass(self, image):
        """Enhanced detection specifically for romantic comedy posters."""
        if self.debug:
            print("\nRunning romantic comedy enhanced detection pass")
        faces = []
        height, width = image.shape[:2]
        
        # Convert to HSV to enhance warm tones and soft lighting
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # Common face locations in romantic comedy posters
        regions_of_interest = [
            # Center upper third (common for close-ups)
            (int(width*0.25), 0, int(width*0.5), int(height*0.4)),
            # Upper half (common for standing poses)
            (0, 0, width, int(height*0.6)),
            # Left side (common for profile shots)
            (0, 0, int(width*0.5), height),
            # Right side (common for profile shots)
            (int(width*0.5), 0, int(width*0.5), height),
            # Center region (common for medium shots)
            (int(width*0.2), int(height*0.2), int(width*0.6), int(height*0.6)),
            # Full image
            (0, 0, width, height)
        ]
        
        # Create multiple enhanced versions optimized for romantic comedies
        enhancements = []
        
        # 1. Soft light enhancement
        v_soft = cv2.GaussianBlur(v, (0, 0), 3)
        v_soft = cv2.addWeighted(v, 0.7, v_soft, 0.3, 0)
        soft_hsv = cv2.merge([h, s, v_soft])
        enhancements.append(cv2.cvtColor(soft_hsv, cv2.COLOR_HSV2BGR))
        
        # 2. Warm tone enhancement
        h_mask = cv2.inRange(h, 0, 30)  # Warm colors (reds, oranges, yellows)
        s_warm = cv2.add(s, 30, mask=h_mask)
        warm_hsv = cv2.merge([h, s_warm, v])
        enhancements.append(cv2.cvtColor(warm_hsv, cv2.COLOR_HSV2BGR))
        
        # 3. High key lighting (common in romantic comedies)
        v_high = cv2.normalize(v, None, 150, 255, cv2.NORM_MINMAX)
        high_hsv = cv2.merge([h, s, v_high])
        enhancements.append(cv2.cvtColor(high_hsv, cv2.COLOR_HSV2BGR))
        
        # 4. Local contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        v_enhanced = clahe.apply(v)
        contrast_hsv = cv2.merge([h, s, v_enhanced])
        enhancements.append(cv2.cvtColor(contrast_hsv, cv2.COLOR_HSV2BGR))
        
        # Detection parameters optimized for romantic comedies
        detection_params = [
            (1, 0.15),  # Model 1 with moderate confidence
            (1, 0.1),   # Model 1 with low confidence
            (0, 0.15),  # Model 0 with moderate confidence
        ]
        
        for roi_x, roi_y, roi_w, roi_h in regions_of_interest:
            # Ensure ROI is within image bounds
            roi_x = max(0, roi_x)
            roi_y = max(0, roi_y)
            roi_w = min(width - roi_x, roi_w)
            roi_h = min(height - roi_y, roi_h)
            
            if roi_w <= 0 or roi_h <= 0:
                continue
                
            roi = image[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
            
            # Try each enhancement
            for i, enhanced in enumerate(enhancements):
                if self.debug:
                    print(f"  Trying enhancement #{i+1} on ROI at ({roi_x}, {roi_y})")
                
                roi_enhanced = enhanced[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
                
                # Try detection with different parameters
                for model_selection, min_detection_confidence in detection_params:
                    with self.mp_face_detection.FaceDetection(
                        model_selection=model_selection,
                        min_detection_confidence=min_detection_confidence
                    ) as face_detection:
                        results = face_detection.process(cv2.cvtColor(roi_enhanced, cv2.COLOR_BGR2RGB))
                        
                        if results.detections:
                            h, w = roi_enhanced.shape[:2]
                            for detection in results.detections:
                                bbox = detection.location_data.relative_bounding_box
                                x = int(bbox.xmin * w) + roi_x
                                y = int(bbox.ymin * h) + roi_y
                                w = int(bbox.width * w)
                                h = int(bbox.height * h)
                                
                                # Skip very small detections
                                if w < self.MIN_FACE_SIZE or h < self.MIN_FACE_SIZE:
                                    continue
                                    
                                # Skip detections with extreme aspect ratios
                                aspect_ratio = w / h
                                if aspect_ratio < 0.5 or aspect_ratio > 2.0:
                                    continue
                                
                                # Ensure coordinates are within image bounds
                                x = max(0, min(width - w, x))
                                y = max(0, min(height - h, y))
                                
                                faces.append((x, y, w, h))
                                if self.debug:
                                    print(f"    Found face at x={x}, y={y}, w={w}, h={h}")
        
        return faces

    def _dark_enhanced_pass(self, image):
        """Enhanced detection pass specifically for dark or shadowed faces."""
        faces = []
        
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, _, _ = cv2.split(lab)
        
        # Apply CLAHE with strong parameters
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)
        
        # Create multiple brightness enhanced versions
        brightness_levels = [1.5, 2.0, 2.5, 3.0]  # Added higher brightness for backlit scenes
        for brightness in brightness_levels:
            # Adjust brightness
            l_bright = cv2.convertScaleAbs(l_enhanced, alpha=brightness, beta=50)  # Increased base brightness
            
            # Apply additional contrast enhancement
            l_contrast = cv2.normalize(l_bright, None, 0, 255, cv2.NORM_MINMAX)
            
            # Create color normalized image
            lab_enhanced = cv2.merge([l_contrast, np.full_like(l, 127), np.full_like(l, 127)])
            bgr_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
            
            # Try face detection
            detected = self._detect_faces(bgr_enhanced, min_confidence=0.15)
            if detected:
                faces.extend(detected)
        
        # Add silhouette detection for backlit scenes
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Create binary image to detect silhouettes
        _, binary = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)
        
        # Find contours of potential silhouettes
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        height, width = image.shape[:2]
        min_area = (width * height) * 0.01  # Minimum area for a potential face region
        max_area = (width * height) * 0.4   # Maximum area for a potential face region
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h
                
                # Check if the contour could be a person (aspect ratio check)
                if 0.2 < aspect_ratio < 0.8:  # Typical aspect ratio for standing person
                    # Extract region and enhance for face detection
                    roi = image[y:y+h, x:x+w]
                    if roi.size > 0:
                        # Try multiple contrast enhancements
                        for gamma in [0.5, 0.7, 1.0]:
                            # Apply gamma correction
                            gamma_corrected = np.power(roi.astype(float)/255, gamma) * 255
                            gamma_corrected = gamma_corrected.astype(np.uint8)
                            
                            # Enhance local contrast
                            lab_roi = cv2.cvtColor(gamma_corrected, cv2.COLOR_BGR2LAB)
                            l_roi = cv2.split(lab_roi)[0]
                            clahe_roi = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(4, 4))
                            l_roi = clahe_roi.apply(l_roi)
                            
                            # Merge back and detect faces
                            enhanced_roi = cv2.merge([l_roi, l_roi, l_roi])  # Use luminance for all channels
                            detected = self._detect_faces(enhanced_roi, min_confidence=0.1)  # Lower confidence for silhouettes
                            
                            if detected:
                                # Adjust coordinates back to full image
                                faces.extend([(x + dx, y + dy, w, h) for dx, dy, w, h in detected])
        
        return self._remove_overlapping(faces)

    def _detect_silhouettes(self, image):
        """New pass specifically for detecting silhouetted figures."""
        faces = []
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Create gradient magnitude image
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(sobelx**2 + sobely**2)
        
        # Normalize and threshold
        gradient_mag = cv2.normalize(gradient_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, binary = cv2.threshold(gradient_mag, 50, 255, cv2.THRESH_BINARY)
        
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)
        
        height, width = image.shape[:2]
        for i in range(1, num_labels):  # Skip background label 0
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]
            
            # Filter potential human silhouettes
            if (0.01 * width * height < area < 0.4 * width * height and  # Size constraints
                0.2 < w/h < 0.8):  # Aspect ratio for standing person
                
                # Extract and enhance region
                roi = image[y:y+h, x:x+w]
                if roi.size > 0:
                    # Create multiple enhanced versions
                    enhancements = []
                    
                    # Version 1: High contrast
                    high_contrast = cv2.convertScaleAbs(roi, alpha=2.0, beta=50)
                    enhancements.append(high_contrast)
                    
                    # Version 2: Local contrast enhancement
                    lab_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
                    l, a, b = cv2.split(lab_roi)
                    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(4, 4))
                    l_enhanced = clahe.apply(l)
                    enhanced = cv2.merge([l_enhanced, a, b])
                    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
                    enhancements.append(enhanced)
                    
                    # Version 3: Edge-preserved smoothing with contrast
                    smooth = cv2.edgePreservingFilter(roi, flags=1, sigma_s=60, sigma_r=0.4)
                    smooth_contrast = cv2.convertScaleAbs(smooth, alpha=1.5, beta=30)
                    enhancements.append(smooth_contrast)
                    
                    # Try face detection on each enhancement
                    for enhanced in enhancements:
                        detected = self._detect_faces(enhanced, min_confidence=0.1)
                        if detected:
                            # Adjust coordinates back to full image
                            faces.extend([(x + dx, y + dy, w, h) for dx, dy, w, h in detected])
        
        return self._remove_overlapping(faces)

    def _detect_faces(self, image, min_confidence=None):
        """Detect faces in an image using MediaPipe Face Detection.
        
        Args:
            image: The image to detect faces in
            min_confidence: Optional minimum confidence threshold. If not provided,
                           uses the default threshold set during initialization.
        
        Returns:
            List of detected face regions as (x, y, w, h) tuples
        """
        if min_confidence is None:
            min_confidence = self.min_confidence
        
        # Convert BGR to RGB for MediaPipe
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_image)
        
        faces = []
        if results.detections:
            height, width = image.shape[:2]
            for detection in results.detections:
                if detection.score[0] < min_confidence:
                    continue
                
                # Get bounding box coordinates
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * width)
                y = int(bbox.ymin * height)
                w = int(bbox.width * width)
                h = int(bbox.height * height)
                
                # Increased padding for better coverage
                pad_x = int(w * 0.2)  # Further increased from 0.15
                pad_y = int(h * 0.2)  # Further increased from 0.15
                x = max(0, x - pad_x)
                y = max(0, y - pad_y)
                w = min(width - x, w + 2 * pad_x)
                h = min(height - y, h + 2 * pad_y)
                
                if w >= self.min_face_size and h >= self.min_face_size:
                    faces.append((x, y, w, h))
        
        return faces

    def detect_faces(self, image):
        """Detect faces using multiple passes with enhanced logging."""
        faces = []
        print("\nStarting face detection passes...")
        
        # Add period-enhanced pass to the beginning of detection passes
        detection_passes = [self._period_enhanced_pass] + self.detection_passes
        
        for detection_pass in detection_passes:
            pass_name = detection_pass.__doc__.strip() if detection_pass.__doc__ else "Unnamed pass"
            print(f"\nTrying {pass_name}")
            detected = detection_pass(image)
            if detected:
                print(f"[+] Found {len(detected)} faces")
                for j, (x, y, w, h) in enumerate(detected):
                    print(f"  Face {j+1}: x={x}, y={y}, width={w}, height={h}")
                faces.extend(detected)
            else:
                print("[-] No faces found")
        
        # Remove overlapping before final output
        unique_faces = self._remove_overlapping(faces)
        print(f"\nFinal unique faces after removing overlaps: {len(unique_faces)}")
        return unique_faces

    def _remove_overlapping(self, faces, iou_threshold=0.5):
        """Remove overlapping face detections using IoU."""
        if not faces:
            return []
            
        # First, filter out faces that are outside image bounds or have invalid dimensions
        valid_faces = []
        for face in faces:
            x, y, w, h = face
            # Skip faces with negative coordinates or zero/negative dimensions
            if x < 0 or y < 0 or w <= 0 or h <= 0:
                if self.debug:
                    print(f"Skipping face with invalid coordinates: x={x}, y={y}, w={w}, h={h}")
                continue
            valid_faces.append(face)
            
        if not valid_faces:
            return []
            
        # Sort faces by area in descending order
        faces_with_area = [(x, y, w, h, w*h) for x, y, w, h in valid_faces]
        faces_with_area.sort(key=lambda x: x[4], reverse=True)
        
        if self.debug:
            print("\nFace regions by area:")
            for i, (x, y, w, h, area) in enumerate(faces_with_area, 1):
                print(f"  {i}. Area: {area}px² at ({x}, {y}) size {w}x{h}")
        
        # Convert to format expected by NMS
        boxes = np.array([[x, y, x + w, y + h] for x, y, w, h in valid_faces])
        scores = np.ones(len(boxes))
        
        # Perform NMS
        indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), 0.0, iou_threshold)
        if len(indices) == 0:
            return []
        
        # Get final faces
        final_faces = [(int(boxes[i][0]), int(boxes[i][1]),
                       int(boxes[i][2] - boxes[i][0]), int(boxes[i][3] - boxes[i][1]))
                      for i in indices.flatten()]
        
        if self.debug:
            print(f"\nKept {len(final_faces)} faces after overlap removal")
            for i, (x, y, w, h) in enumerate(final_faces, 1):
                print(f"  Face {i}: ({x}, {y}) size {w}x{h}")
        
        return final_faces

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
        """Detect and blur faces in an image."""
        image = cv2.imread(input_path)
        if image is None:
            print(f"Failed to load image: {input_path}")
            return False
            
        height, width = image.shape[:2]
        faces = self.detect_faces(image)
        
        if not faces:
            print("No faces found")
            return False
            
        print(f"Found {len(faces)} faces")
        for i, (x, y, w, h) in enumerate(faces, 1):
            # Skip faces that would be partially outside the image
            if x < 0 or y < 0 or x + w > width or y + h > height:
                if self.debug:
                    print(f"\nSkipping face #{i} - outside image bounds: x={x}, y={y}, w={w}, h={h}")
                continue
                
            print(f"\nProcessing face #{i}")
            if self.debug:
                print(f"DEBUG: Initial face dimensions - x:{x}, y:{y}, w:{w}, h:{h}\n")
            
            # Skip faces that are too small (likely false positives)
            if w < self.MIN_FACE_SIZE or h < self.MIN_FACE_SIZE:
                print(f"DEBUG: Skipping face #{i} - too small (w:{w}, h:{h} < minimum:{self.MIN_FACE_SIZE})")
                continue

            # Only detect hair region if needed
            face_region = image[y : y + h, x : x + w]
            print("\nChecking if blur should be extended:")
            should_extend = self.should_extend_blur(face_region)
            
            if should_extend:
                print("DEBUG: Attempting to detect hair region")
                hair_bbox = self.detect_hair_region(image, (x, y, w, h))
                if hair_bbox is not None:
                    old_height = h
                    x, y = hair_bbox[0], hair_bbox[1]
                    h = (hair_bbox[1] + hair_bbox[3]) - y
                    print(f"DEBUG: Height increased from {old_height} to {h}")
                else:
                    print("DEBUG: No hair region detected despite high edge content")
            else:
                print("DEBUG: Using standard face region (no extension needed)")

            # Get the combined region
            face_region = image[y : y + h, x : x + w]
            print(f"DEBUG: Final region dimensions - x:{x}, y:{y}, w:{w}, h:{h}")

            # Create mask
            print("\nCreating blur mask:")
            if should_extend:  # Use the same decision we made earlier
                print("DEBUG: Using extended adaptive hair mask")
                mask = self.create_adaptive_hair_mask(h, w)
            else:
                print("DEBUG: Using standard face mask")
                mask = np.zeros((h, w), dtype=np.uint8)
                center = (w // 2, h // 2)
                # Make the standard mask slightly taller but still compact
                face_axes = (int(w // 1.9), int(h // 1.4))
                cv2.ellipse(mask, center, face_axes, 0, 0, 360, 255, -1)
                # Add a small upper extension for forehead
                top_center = (center[0], int(h * 0.35))
                top_axes = (int(w // 2.2), int(h * 0.2))
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


def test_period_detection():
    """Test function for period image detection."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    test_image = os.path.join(script_dir, "input", "TheFavourite.jpg")  # Changed path
    output_image = os.path.join(script_dir, "output", "TheFavourite_test.jpg")
    
    print(f"\nTesting period detection on {test_image}")
    if not os.path.exists(test_image):
        print(f"Error: Test image not found at {test_image}")
        return
        
    blurrer = FaceBlurrer(debug=True)  # Enable debug output
    blurrer.blur_faces(test_image, output_image)

if __name__ == "__main__":
    # Test period detection first
    test_period_detection()
    
    # Then process all images
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(script_dir, "input")
    output_dir = os.path.join(script_dir, "output")

    # Create directories if they don't exist
    Path(input_dir).mkdir(parents=True, exist_ok=True)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    blurrer = FaceBlurrer(debug=True)  # Enable debug output
    blurrer.process_directory(input_dir, output_dir)
