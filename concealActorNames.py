import os
from pathlib import Path

import cv2
import numpy as np
import pytesseract
from PIL import Image


class ActorNameConcealer:
    def __init__(self):
        # Configure Tesseract path
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        
        # Parameters for text detection
        self.min_confidence = 40  # Increased confidence threshold
        self.padding_x = 10  # Reduced padding
        self.padding_y = 15
        self.height_multiplier = 1.5  # Reduced height multiplier
        self.name_distance_threshold = 25  # More strict distance threshold
        self.blur_kernel = (99, 99)
        self.blur_sigma = 30
        
        # Parameters for pattern preservation
        self.texture_threshold = 40
        self.pattern_preserve_ratio = 0.4
        self.direction_kernel_size = 31

    def combine_text_regions(self, data):
        """Combine consecutive text regions that might form names."""
        combined_regions = []
        current_region = None
        
        # Sort by vertical position and then horizontal
        n_boxes = len(data['text'])
        regions = []
        for i in range(n_boxes):
            if float(data['conf'][i]) < 0 or not data['text'][i].strip():
                continue
            regions.append({
                'text': data['text'][i].strip(),
                'x': data['left'][i],
                'y': data['top'][i],
                'w': data['width'][i],
                'h': data['height'][i],
                'conf': float(data['conf'][i])
            })
        
        # Sort by y position first, then x position
        regions.sort(key=lambda r: (r['y'], r['x']))
        
        for region in regions:
            if current_region is None:
                current_region = region.copy()
                continue
            
            # Check if this region is close to the previous one
            y_dist = abs(region['y'] - current_region['y'])
            x_dist = region['x'] - (current_region['x'] + current_region['w'])
            
            if y_dist < self.name_distance_threshold and x_dist < 50:
                # Combine regions
                current_region['text'] += ' ' + region['text']
                current_region['w'] = (region['x'] + region['w']) - current_region['x']
                current_region['h'] = max(current_region['h'], region['h'])
            else:
                if len(current_region['text'].split()) >= 2:  # Only keep multi-word combinations
                    combined_regions.append(current_region)
                current_region = region.copy()
        
        if current_region and len(current_region['text'].split()) >= 2:
            combined_regions.append(current_region)
        
        return combined_regions

    def extract_potential_names(self, text):
        """Extract potential names from a longer text string."""
        # Remove quotes and other common punctuation
        text = text.replace('"', ' ').replace(';', ' ').replace('_', ' ')
        words = text.split()
        
        # Extended list of words to skip
        skip_words = {
            "FROM", "THE", "AND", "WITH", "OF", "BY", "FOR", "IN", "TO", "A",
            "AUTHOR", "DIRECTOR", "STARRING", "PRESENTS", "PRODUCTION", "PRODUCTIONS",
            "WINNER", "AWARD", "ACADEMY", "PRIZE", "MESSAGE", "JOURNEY", "WINNING",
            "HEART", "UNIVERSE", "DEEP", "SPACE", "PULITZER", "COMPANY", "ENTERTAINMENT",
            "BROS", "PICTURES", "FILMS", "WARNER", "PRESENTS", "FILM", "STUDIO",
            "FIRST", "GO", "WHO", "WILL", "BE"
        }
        
        # Look for consecutive capitalized words that could be names
        potential_names = []
        current_name = []
        
        for word in words:
            # Skip common words and short words
            if (word.upper() in skip_words or len(word) < 2 or 
                not all(c.isalpha() or c in "-." for c in word)):
                if current_name:
                    potential_names.append(" ".join(current_name))
                    current_name = []
                continue
                
            # If word starts with a capital letter and contains valid characters
            if word and word[0].isupper():
                current_name.append(word)
            else:
                if current_name:
                    potential_names.append(" ".join(current_name))
                    current_name = []
        
        if current_name:
            potential_names.append(" ".join(current_name))
        
        return [name for name in potential_names if len(name.split()) >= 2]

    def is_actor_name(self, text, y_position, image_height):
        """Determine if detected text is likely an actor name."""
        # Extract potential names from the text
        potential_names = self.extract_potential_names(text)
        
        if not potential_names:
            print(f"No potential names found in: '{text}'")
            return False
            
        # Check each potential name
        for name in potential_names:
            print(f"Analyzing potential name: '{name}' at y={y_position} (image height={image_height})")
            
            # More precise position check for actor names
            is_in_title_area = 0.05 < (y_position / image_height) < 0.3  # Main title area
            is_valid_chars = all(c.isalpha() or c.isspace() or c in "-." for c in name)
            words = name.split()
            word_count = len(words)
            valid_length = 4 <= len(name) <= 30  # Stricter length requirement
            valid_words = all(len(word) >= 2 for word in words)
            
            # Debug info
            print(f"  In title area: {is_in_title_area}")
            print(f"  Valid chars: {is_valid_chars}")
            print(f"  Word count: {word_count}")
            print(f"  Valid length: {valid_length}")
            print(f"  Valid words: {valid_words}")
            
            if (is_in_title_area and is_valid_chars and word_count >= 2 
                    and valid_length and valid_words):
                return True
        
        return False

    def get_dominant_direction(self, gray_image):
        """Calculate the dominant gradient direction in the image."""
        # Calculate gradients
        grad_x = cv2.Sobel(gray_image, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_image, cv2.CV_32F, 0, 1, ksize=3)
        
        # Calculate gradient direction
        angle = np.arctan2(grad_y, grad_x)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Get dominant direction (weighted by magnitude)
        dominant_angle = np.average(angle, weights=magnitude)
        return dominant_angle

    def process_image(self, input_path, output_path):
        """Process a single image to conceal actor names."""
        print(f"\nProcessing {os.path.basename(input_path)}")
        
        # Read image
        image = cv2.imread(input_path)
        if image is None:
            print(f"Could not read image: {input_path}")
            return False

        # Convert to RGB for better text detection
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get image dimensions
        height, width = image.shape[:2]

        # Create a copy of the image for processing
        result = image.copy()

        # Enhance image for better text detection
        enhanced = cv2.convertScaleAbs(rgb, alpha=1.5, beta=0)
        
        # Perform OCR with custom configuration
        print("Performing OCR...")
        custom_config = r'--oem 3 --psm 6'
        data = pytesseract.image_to_data(enhanced, output_type=pytesseract.Output.DICT, config=custom_config)
        
        print(f"Found {len(data['text'])} text regions")
        
        # Combine text regions that might form names
        combined_regions = self.combine_text_regions(data)
        print(f"Combined into {len(combined_regions)} potential name regions")
        
        # Process each combined region
        for region in combined_regions:
            text = region['text']
            x = region['x']
            y = region['y']
            w = region['w']
            h = int(region['h'] * self.height_multiplier)
            
            # Check if this is likely an actor name
            if self.is_actor_name(text, y, height):
                print(f"Found potential actor name: {text}")
                
                # Shift y coordinates up by 10 pixels
                y = max(0, y - 10)  # Ensure we don't go above image bounds
                
                # Calculate the main region with extra padding for pattern analysis
                x1 = max(0, x - self.padding_x * 2)
                y1 = max(0, y - self.padding_y * 2)
                x2 = min(width, x + w + self.padding_x * 2)
                y2 = min(height, y + h + self.padding_y * 2)
                
                # Get the region of interest
                roi = result[y1:y2, x1:x2]
                if roi.size > 0:
                    # Convert ROI to grayscale for analysis
                    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    
                    # Get dominant direction of the pattern
                    angle = self.get_dominant_direction(gray_roi)
                    
                    # Create directional kernel for blurring
                    direction_kernel = (self.direction_kernel_size, self.direction_kernel_size)
                    
                    # First pass: Strong blur to fully conceal text
                    text_region = result[y:y+h, x:x+w]
                    blurred_text = cv2.GaussianBlur(text_region, (99, 99), 40)
                    result[y:y+h, x:x+w] = blurred_text
                    
                    # Second pass: Directional blur for pattern preservation
                    roi = result[y1:y2, x1:x2]
                    pattern_blur = cv2.GaussianBlur(roi, direction_kernel, 20)
                    
                    # Create gradient mask from center
                    center_y, center_x = roi.shape[0] // 2, roi.shape[1] // 2
                    Y, X = np.ogrid[:roi.shape[0], :roi.shape[1]]
                    
                    # Adjust coordinates based on dominant direction
                    rotated_X = X * np.cos(angle) - Y * np.sin(angle)
                    rotated_Y = X * np.sin(angle) + Y * np.cos(angle)
                    
                    # Create elliptical mask aligned with pattern direction
                    mask = ((rotated_X - center_x)**2 / (center_x**2) + 
                           (rotated_Y - center_y)**2 / (center_y**2)) <= 1
                    mask = mask.astype(np.float32)
                    
                    # Smooth the mask
                    mask = cv2.GaussianBlur(mask, (5, 5), 0)
                    
                    # Combine blurred versions
                    mask = mask[:, :, np.newaxis]
                    result_roi = (pattern_blur * mask + 
                                roi * (1 - mask) * self.pattern_preserve_ratio)
                    
                    # Final smoothing
                    result_roi = cv2.GaussianBlur(result_roi.astype(np.uint8), (3, 3), 0)
                    result[y1:y2, x1:x2] = result_roi

        # Save the result
        cv2.imwrite(output_path, result)
        print(f"Saved processed image to {output_path}")
        return True

    def process_directory(self, input_dir, output_dir):
        """Process all images in the input directory."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        for filename in os.listdir(input_dir):
            if filename.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                input_path = os.path.join(input_dir, filename)
                output_path = os.path.join(output_dir, f"nonames_{filename}")
                if self.process_image(input_path, output_path):
                    print(f"Successfully processed {filename}")
                else:
                    print(f"Failed to process {filename}")


if __name__ == "__main__":
    # Define input/output directories relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(script_dir, "output")  # Use blur.py output as input
    output_dir = os.path.join(script_dir, "final")  # Final output with names concealed

    # Create directories if they don't exist
    Path(input_dir).mkdir(parents=True, exist_ok=True)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    concealer = ActorNameConcealer()
    concealer.process_directory(input_dir, output_dir) 