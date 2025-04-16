import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image, ImageDraw, ImageFilter, ImageOps, ImageEnhance, ImageChops
import random
import os
import json
import math
import time
from scipy import ndimage
from typing import List, Tuple, Dict, Any, Union, Optional

class ComprehensivePuzzleGenerator:
    """
    Generate comprehensive multimodal puzzle tasks for testing vision models.
    Focused on three main puzzle types:
    1. Jigsaw: Pieces with tabs and blanks along edges
    2. Multi-piece: Grid-based puzzles with multiple pieces removed
    3. Fractal: Complex organic shapes with irregular boundaries
    """
    
    def __init__(self, seed=None):
        """Initialize the generator with optional random seed for reproducibility."""
        if seed:
            random.seed(seed)
            np.random.seed(seed)
        
        # Task-specific parameters for customization
        self.jigsaw_params = {
            'easy': {
                'tab_count_range': (0, 1),  # Fewer tabs/blanks
                'tab_size_factor': 0.15,    # Smaller tabs
                'shape_complexity': 'simple' # Simpler shapes for tabs
            },
            'medium': {
                'tab_count_range': (1, 2),  # Medium number of tabs
                'tab_size_factor': 0.2,     # Medium sized tabs
                'shape_complexity': 'medium' # More complex tab shapes
            },
            'hard': {
                'tab_count_range': (2, 3),  # More tabs/blanks
                'tab_size_factor': 0.25,    # Larger tabs
                'shape_complexity': 'complex' # Complex tab shapes
            }
        }
        
        self.multi_piece_params = {
            'easy': {
                'grid_size': 3,           # 3x3 grid
                'num_pieces_removed': 1,  # Only one piece removed
                'distractor_difficulty': 'low' # Easy to distinguish wrong pieces
            },
            'medium': {
                'grid_size': 4,           # 4x4 grid
                'num_pieces_removed': 2,  # Two pieces removed
                'distractor_difficulty': 'medium' # Harder to distinguish
            },
            'hard': {
                'grid_size': 5,           # 5x5 grid
                'num_pieces_removed': 3,  # Three pieces removed
                'distractor_difficulty': 'high' # Very similar distractors
            }
        }
        
        self.fractal_params = {
            'easy': {
                'complexity': 2,          # Fewer iterations/subdivisions
                'irregularity': 0.1,      # Less irregularity in shapes
                'smoothing': 1.0          # More smoothing
            },
            'medium': {
                'complexity': 3,
                'irregularity': 0.3,
                'smoothing': 0.5
            },
            'hard': {
                'complexity': 4,          # More iterations/subdivisions
                'irregularity': 0.5,      # More irregularity
                'smoothing': 0.1          # Less smoothing
            }
        }
        
        # Distractor generation strategies
        self.distractor_strategies = {
            'jigsaw': [
                'flip_tabs',     # Invert tabs and blanks
                'rotate',        # Rotate the piece
                'shift',         # Take a piece from elsewhere
                'modify_tabs',   # Change tab shapes
                'color_shift'    # Modify colors
            ],
            'multi_piece': [
                'other_cell',    # Take from another grid cell
                'transform',     # Apply transformations
                'shifted_grid',  # Shift the grid slightly
                'color_modify',  # Modify colors
                'pattern_modify' # Add/remove patterns
            ],
            'fractal': [
                'different_seed',  # Generate with different parameters
                'transform',       # Apply transformations
                'boundary_modify', # Modify the boundary
                'shift',           # Take from elsewhere
                'hybrid'           # Combine strategies
            ]
        }
    
    def load_image(self, image_path: str) -> Tuple[Image.Image, np.ndarray]:
        """Load an image from path using both PIL and OpenCV."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at: {image_path}")
        
        pil_img = Image.open(image_path).convert("RGB")
        cv_img = cv2.imread(image_path)
        if cv_img is None:
            raise ValueError(f"Failed to load image with OpenCV: {image_path}")
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        
        return pil_img, cv_img
    
    def add_noise_to_mask(self, mask: Image.Image, noise_level: float = 0.2) -> Image.Image:
        """Add noise to a mask to create more organic, natural-looking edges."""
        if noise_level <= 0:
            return mask
            
        # Convert mask to numpy array
        mask_np = np.array(mask)
        
        # Create perlin-like noise using gaussian filtering of random noise
        noise = np.random.rand(*mask_np.shape) * 255
        noise = ndimage.gaussian_filter(noise, sigma=3)
        
        # Scale noise effect based on noise_level
        noise = noise * noise_level
        
        # Apply noise only near the edges
        edge_kernel = np.ones((5, 5), np.uint8)
        edges = cv2.dilate(mask_np, edge_kernel) - cv2.erode(mask_np, edge_kernel)
        
        # Apply noise only to edge regions
        mask_np = mask_np.astype(float)
        mask_np[edges > 0] += noise[edges > 0] - 128 * noise_level
        
        # Clip values
        mask_np = np.clip(mask_np, 0, 255).astype(np.uint8)
        
        return Image.fromarray(mask_np)
    
    def smooth_mask(self, mask: Image.Image, smoothing_factor: float = 1.0) -> Image.Image:
        """Apply smoothing to a mask to create more natural edges."""
        if smoothing_factor <= 0:
            return mask
            
        # Convert to PIL and apply Gaussian blur
        smooth_radius = max(1, int(smoothing_factor * 2))
        smoothed_mask = mask.filter(ImageFilter.GaussianBlur(radius=smooth_radius))
        
        # Threshold to ensure binary mask
        smoothed_mask = smoothed_mask.point(lambda p: 255 if p > 127 else 0)
        
        return smoothed_mask
    
    ##############################
    # 1. JIGSAW PUZZLE PIECE TASKS
    ##############################
    
    def generate_jigsaw_piece(self, 
                             image_path: str, 
                             output_dir: str, 
                             num_options: int = 4, 
                             difficulty: str = 'medium',
                             custom_params: Optional[Dict] = None,
                             distractor_types: Optional[List[str]] = None) -> Dict:
        """
        Generate a task with a jigsaw-like piece (with tabs and blanks) cut out of an image.
        
        Args:
            image_path: Path to the source image
            output_dir: Directory to save the task files
            num_options: Number of piece options to generate (default: 4)
            difficulty: Difficulty level ('easy', 'medium', 'hard')
            custom_params: Optional dict to override default parameters
            distractor_types: List of distractor strategies to use (randomly chosen if None)
            
        Returns:
            Dictionary with task information
        """
        # Create output directory
        task_dir = os.path.join(output_dir, f"jigsaw_task_{difficulty}")
        os.makedirs(task_dir, exist_ok=True)
        
        # Get parameters based on difficulty level
        params = self.jigsaw_params.get(difficulty, self.jigsaw_params['medium']).copy()
        
        # Override with custom parameters if provided
        if custom_params:
            params.update(custom_params)
        
        # Load the image
        pil_img, cv_img = self.load_image(image_path)
        width, height = pil_img.size
        
        # Define the base rectangular cutout parameters
        cutout_width = int(width * random.uniform(0.15, 0.3))  # 15-30% of image width
        cutout_height = int(height * random.uniform(0.15, 0.3))  # 15-30% of image height
        
        # Randomly position the cutout (not too close to the edges)
        margin = 50  # Larger margin to allow for jigsaw tabs
        left = random.randint(margin, width - cutout_width - margin)
        top = random.randint(margin, height - cutout_height - margin)
        
        # Create a mask for the jigsaw piece
        mask = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask)
        
        # Draw the base rectangle
        draw.rectangle((left, top, left + cutout_width, top + cutout_height), fill=255)
        
        # Add jigsaw tabs (bulges) and blanks (indentations)
        sides = ["top", "right", "bottom", "left"]
        tab_size_factor = params['tab_size_factor']
        tab_width = int(cutout_width * tab_size_factor)  # Width of tab/blank
        tab_height = int(cutout_height * tab_size_factor)  # Height of tab/blank
        
        # Shape complexity affects the type of shapes used for tabs
        shape_complexity = params['shape_complexity']
        
        # Record information about tabs and blanks for metadata
        jigsaw_features = {}
        
        for side in sides:
            # Decide how many features (tabs/blanks) for this side based on difficulty
            tab_count_range = params['tab_count_range']
            num_features = random.randint(*tab_count_range)
            
            if num_features == 0:
                continue
                
            jigsaw_features[side] = []
            
            for i in range(num_features):
                # Decide if it's a tab (outward) or blank (inward)
                is_tab = random.choice([True, False])
                
                if side == "top" or side == "bottom":
                    # Feature along horizontal edge
                    center_x = left + int((i + 1) * cutout_width / (num_features + 1))
                    feature_left = center_x - tab_width // 2
                    feature_right = center_x + tab_width // 2
                    
                    if side == "top":
                        if is_tab:
                            # Outward tab on top
                            self._draw_tab_shape(draw, shape_complexity, 
                                               feature_left, top - tab_height, 
                                               feature_right, top, is_horizontal=True)
                        else:
                            # Inward blank on top
                            self._draw_tab_shape(draw, shape_complexity, 
                                               feature_left, top, 
                                               feature_right, top + tab_height, 
                                               is_horizontal=True, is_blank=True)
                    else:  # bottom
                        if is_tab:
                            # Outward tab on bottom
                            self._draw_tab_shape(draw, shape_complexity, 
                                               feature_left, top + cutout_height, 
                                               feature_right, top + cutout_height + tab_height, 
                                               is_horizontal=True)
                        else:
                            # Inward blank on bottom
                            self._draw_tab_shape(draw, shape_complexity, 
                                               feature_left, top + cutout_height - tab_height, 
                                               feature_right, top + cutout_height, 
                                               is_horizontal=True, is_blank=True)
                else:
                    # Feature along vertical edge
                    center_y = top + int((i + 1) * cutout_height / (num_features + 1))
                    feature_top = center_y - tab_height // 2
                    feature_bottom = center_y + tab_height // 2
                    
                    if side == "left":
                        if is_tab:
                            # Outward tab on left
                            self._draw_tab_shape(draw, shape_complexity, 
                                               left - tab_width, feature_top, 
                                               left, feature_bottom, 
                                               is_horizontal=False)
                        else:
                            # Inward blank on left
                            self._draw_tab_shape(draw, shape_complexity, 
                                               left, feature_top, 
                                               left + tab_width, feature_bottom, 
                                               is_horizontal=False, is_blank=True)
                    else:  # right
                        if is_tab:
                            # Outward tab on right
                            self._draw_tab_shape(draw, shape_complexity, 
                                               left + cutout_width, feature_top, 
                                               left + cutout_width + tab_width, feature_bottom, 
                                               is_horizontal=False)
                        else:
                            # Inward blank on right
                            self._draw_tab_shape(draw, shape_complexity, 
                                               left + cutout_width - tab_width, feature_top, 
                                               left + cutout_width, feature_bottom, 
                                               is_horizontal=False, is_blank=True)
                
                # Record feature for metadata
                feature_info = {
                    "position": i,
                    "is_tab": is_tab,
                    "center": center_x if side in ["top", "bottom"] else center_y,
                    "shape": shape_complexity
                }
                
                if side in ["top", "bottom"]:
                    feature_info["left"] = feature_left
                    feature_info["right"] = feature_right
                else:
                    feature_info["top"] = feature_top
                    feature_info["bottom"] = feature_bottom
                
                jigsaw_features[side].append(feature_info)
        
        # Apply noise and smoothing based on complexity
        if shape_complexity == 'medium':
            mask = self.add_noise_to_mask(mask, noise_level=0.1)
            mask = self.smooth_mask(mask, smoothing_factor=0.5)
        elif shape_complexity == 'complex':
            mask = self.add_noise_to_mask(mask, noise_level=0.2)
            mask = self.smooth_mask(mask, smoothing_factor=0.3)
        
        # Convert to numpy array for OpenCV operations
        mask_np = np.array(mask)
        
        # Get the bounding box of the jigsaw piece
        contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        x, y, w, h = cv2.boundingRect(contours[0])
        
        # Create the masked image (with gray filling the jigsaw piece)
        masked_pil = pil_img.copy()
        masked_np = np.array(masked_pil)
        masked_np[mask_np > 0] = [128, 128, 128]
        masked_img = Image.fromarray(masked_np)
        
        # Extract the correct cutout piece with transparency
        img_rgba = Image.new("RGBA", pil_img.size)
        img_rgba.paste(pil_img)
        
        # Create alpha channel from mask
        img_rgba.putalpha(mask)
        
        # Crop to the bounding box
        correct_piece = img_rgba.crop((x, y, x + w, y + h))
        
        # Save the masked image
        masked_img_path = os.path.join(task_dir, "masked_image.jpg")
        masked_img.save(masked_img_path)
        
        # Generate the options
        correct_idx = random.randint(0, num_options - 1)
        options = []
        option_paths = []
        
        # Select distractor strategies if not specified
        if not distractor_types:
            num_strategies = min(num_options - 1, len(self.distractor_strategies['jigsaw']))
            distractor_types = random.sample(self.distractor_strategies['jigsaw'], num_strategies)
        
        for i in range(num_options):
            if i == correct_idx:
                # This is the correct piece
                piece = correct_piece
            else:
                # Generate an incorrect piece using different strategies
                strategy_idx = (i - 1) % len(distractor_types)
                strategy = distractor_types[strategy_idx]
                
                if strategy == 'flip_tabs' and jigsaw_features:
                    # Strategy 1: Flip tabs and blanks
                    piece = self._jigsaw_strategy_flip_tabs(
                        pil_img, jigsaw_features, left, top, 
                        cutout_width, cutout_height, tab_width, tab_height,
                        shape_complexity, width, height
                    )
                
                elif strategy == 'rotate':
                    # Strategy 2: Rotate the piece
                    rotation_angle = random.choice([90, 180, 270])
                    piece = correct_piece.rotate(rotation_angle, expand=True)
                
                elif strategy == 'shift':
                    # Strategy 3: Take a piece from elsewhere
                    piece = self._jigsaw_strategy_shift(
                        pil_img, mask, mask_np, x, y, w, h, width, height, margin
                    )
                
                elif strategy == 'modify_tabs':
                    # Strategy 4: Modify tab shapes
                    piece = self._jigsaw_strategy_modify_tabs(
                        pil_img, jigsaw_features, left, top, 
                        cutout_width, cutout_height, tab_width, tab_height,
                        shape_complexity, width, height
                    )
                
                elif strategy == 'color_shift':
                    # Strategy 5: Color modification
                    piece = correct_piece.copy()
                    
                    # Apply color transformations
                    transforms = [
                        lambda img: ImageEnhance.Color(img).enhance(random.uniform(0.7, 1.3)),
                        lambda img: ImageEnhance.Brightness(img).enhance(random.uniform(0.8, 1.2)),
                        lambda img: ImageEnhance.Contrast(img).enhance(random.uniform(0.8, 1.2)),
                    ]
                    
                    # Apply 1-2 random transformations
                    for _ in range(random.randint(1, 2)):
                        transform = random.choice(transforms)
                        piece = transform(piece)
                
                else:
                    # Default fallback: Take a random crop
                    fake_x = random.randint(margin, width - w - margin)
                    fake_y = random.randint(margin, height - h - margin)
                    
                    # Make sure it's far enough from the original
                    while abs(fake_x - x) < w/2 and abs(fake_y - y) < h/2:
                        fake_x = random.randint(margin, width - w - margin)
                        fake_y = random.randint(margin, height - h - margin)
                    
                    # Create a shape similar to the original but at a different position
                    shifted_mask = Image.new("L", (width, height), 0)
                    shifted_mask.paste(mask.crop((x, y, x+w, y+h)), (fake_x, fake_y))
                    
                    # Apply to the image
                    fake_rgba = Image.new("RGBA", pil_img.size)
                    fake_rgba.paste(pil_img)
                    fake_rgba.putalpha(shifted_mask)
                    
                    piece = fake_rgba.crop((fake_x, fake_y, fake_x + w, fake_y + h))
            
            # Save the piece
            piece_path = os.path.join(task_dir, f"option_{chr(65+i)}.png")
            piece.save(piece_path)
            
            options.append(piece)
            option_paths.append(piece_path)
        
        # Save the original image for reference
        original_path = os.path.join(task_dir, "original.jpg")
        pil_img.save(original_path)
        
        # Create a visualization of the task
        vis_path = self._create_visualization(masked_img, options, correct_idx, task_dir, show_border=True)
        
        # Save task metadata
        metadata = {
            "task_type": "jigsaw_piece",
            "difficulty": difficulty,
            "original_image": os.path.basename(original_path),
            "masked_image": os.path.basename(masked_img_path),
            "options": [os.path.basename(path) for path in option_paths],
            "correct_idx": correct_idx,
            "correct_option": chr(65 + correct_idx),
            "cutout_bounds": {
                "x": x,
                "y": y,
                "width": w,
                "height": h
            },
            "base_rectangle": {
                "left": left,
                "top": top,
                "width": cutout_width,
                "height": cutout_height
            },
            "jigsaw_features": jigsaw_features,
            "parameters": params,
            "distractor_strategies": [distractor_types[i-1] if i != correct_idx else "correct" 
                                     for i in range(num_options)]
        }
        
        metadata_path = os.path.join(task_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        return {
            "task_dir": task_dir,
            "masked_image": masked_img_path,
            "options": option_paths,
            "correct_idx": correct_idx,
            "metadata": metadata
        }
    
    def _draw_tab_shape(self, draw, complexity, x1, y1, x2, y2, is_horizontal=True, is_blank=False):
        """Helper to draw different tab shapes based on complexity."""
        if complexity == 'simple':
            # Simple elliptical tab/blank
            draw.ellipse((x1, y1, x2, y2), fill=0 if is_blank else 255)
        
        elif complexity == 'medium':
            # More complex rounded tab/blank
            if is_horizontal:
                # For horizontal tabs (top/bottom)
                width, height = x2 - x1, y2 - y1
                mid_x, mid_y = (x1 + x2) // 2, (y1 + y2) // 2
                
                # Draw a more complex shape
                points = []
                num_points = 12
                
                for i in range(num_points):
                    angle = 2 * math.pi * i / num_points
                    # Make the shape more elliptical
                    r_x = width / 2
                    r_y = height / 2
                    x = mid_x + r_x * math.cos(angle)
                    y = mid_y + r_y * math.sin(angle)
                    points.append((x, y))
                
                draw.polygon(points, fill=0 if is_blank else 255)
                
            else:
                # For vertical tabs (left/right)
                width, height = x2 - x1, y2 - y1
                mid_x, mid_y = (x1 + x2) // 2, (y1 + y2) // 2
                
                # Draw a more complex shape
                points = []
                num_points = 12
                
                for i in range(num_points):
                    angle = 2 * math.pi * i / num_points
                    # Make the shape more elliptical
                    r_x = width / 2
                    r_y = height / 2
                    x = mid_x + r_x * math.cos(angle)
                    y = mid_y + r_y * math.sin(angle)
                    points.append((x, y))
                
                draw.polygon(points, fill=0 if is_blank else 255)
        
        elif complexity == 'complex':
            # Complex tab shape with irregular edges
            if is_horizontal:
                # For horizontal tabs
                width, height = x2 - x1, y2 - y1
                mid_x, mid_y = (x1 + x2) // 2, (y1 + y2) // 2
                
                # Create a complex tab shape with intentional asymmetry
                points = []
                num_points = 16  # More points for higher complexity
                
                for i in range(num_points):
                    angle = 2 * math.pi * i / num_points
                    # Add irregularity
                    r_x = width / 2 * (1 + random.uniform(-0.1, 0.1))
                    r_y = height / 2 * (1 + random.uniform(-0.1, 0.1))
                    x = mid_x + r_x * math.cos(angle)
                    y = mid_y + r_y * math.sin(angle)
                    points.append((x, y))
                
                draw.polygon(points, fill=0 if is_blank else 255)
                
            else:
                # For vertical tabs
                width, height = x2 - x1, y2 - y1
                mid_x, mid_y = (x1 + x2) // 2, (y1 + y2) // 2
                
                # Create a complex tab shape
                points = []
                num_points = 16
                
                for i in range(num_points):
                    angle = 2 * math.pi * i / num_points
                    # Add irregularity
                    r_x = width / 2 * (1 + random.uniform(-0.1, 0.1))
                    r_y = height / 2 * (1 + random.uniform(-0.1, 0.1))
                    x = mid_x + r_x * math.cos(angle)
                    y = mid_y + r_y * math.sin(angle)
                    points.append((x, y))
                
                draw.polygon(points, fill=0 if is_blank else 255)
        
        else:
            # Fallback to simple ellipse
            draw.ellipse((x1, y1, x2, y2), fill=0 if is_blank else 255)
    
    def _jigsaw_strategy_flip_tabs(self, pil_img, jigsaw_features, left, top, 
                                   cutout_width, cutout_height, tab_width, tab_height,
                                   shape_complexity, width, height):
        """Helper for jigsaw distractor strategy: flipping tabs and blanks."""
        # Create a new mask with flipped tabs/blanks
        fake_mask = Image.new("L", (width, height), 0)
        fake_draw = ImageDraw.Draw(fake_mask)
        
        # Draw the base rectangle
        fake_draw.rectangle((left, top, left + cutout_width, top + cutout_height), fill=255)
        
        # Invert some of the tabs/blanks
        for side, features in jigsaw_features.items():
            for feature in features:
                # Flip tabs to blanks and vice versa
                is_tab = not feature["is_tab"]
                
                if side == "top":
                    center_x = feature["center"]
                    feature_left = feature["left"]
                    feature_right = feature["right"]
                    
                    if is_tab:
                        # Outward tab on top
                        self._draw_tab_shape(fake_draw, shape_complexity, 
                                           feature_left, top - tab_height, 
                                           feature_right, top, is_horizontal=True)
                    else:
                        # Inward blank on top
                        self._draw_tab_shape(fake_draw, shape_complexity, 
                                           feature_left, top, 
                                           feature_right, top + tab_height, 
                                           is_horizontal=True, is_blank=True)
                
                elif side == "bottom":
                    center_x = feature["center"]
                    feature_left = feature["left"]
                    feature_right = feature["right"]
                    
                    if is_tab:
                        # Outward tab on bottom
                        self._draw_tab_shape(fake_draw, shape_complexity, 
                                           feature_left, top + cutout_height, 
                                           feature_right, top + cutout_height + tab_height, 
                                           is_horizontal=True)
                    else:
                        # Inward blank on bottom
                        self._draw_tab_shape(fake_draw, shape_complexity, 
                                           feature_left, top + cutout_height - tab_height, 
                                           feature_right, top + cutout_height, 
                                           is_horizontal=True, is_blank=True)
                
                elif side == "left":
                    center_y = feature["center"]
                    feature_top = feature["top"]
                    feature_bottom = feature["bottom"]
                    
                    if is_tab:
                        # Outward tab on left
                        self._draw_tab_shape(fake_draw, shape_complexity, 
                                           left - tab_width, feature_top, 
                                           left, feature_bottom, is_horizontal=False)
                    else:
                        # Inward blank on left
                        self._draw_tab_shape(fake_draw, shape_complexity, 
                                           left, feature_top, 
                                           left + tab_width, feature_bottom, 
                                           is_horizontal=False, is_blank=True)
                
                elif side == "right":
                    center_y = feature["center"]
                    feature_top = feature["top"]
                    feature_bottom = feature["bottom"]
                    
                    if is_tab:
                        # Outward tab on right
                        self._draw_tab_shape(fake_draw, shape_complexity, 
                                           left + cutout_width, feature_top, 
                                           left + cutout_width + tab_width, feature_bottom, 
                                           is_horizontal=False)
                    else:
                        # Inward blank on right
                        self._draw_tab_shape(fake_draw, shape_complexity, 
                                           left + cutout_width - tab_width, feature_top, 
                                           left + cutout_width, feature_bottom, 
                                           is_horizontal=False, is_blank=True)
        
        # Apply noise or smoothing if needed
        if shape_complexity == 'medium':
            fake_mask = self.add_noise_to_mask(fake_mask, noise_level=0.1)
            fake_mask = self.smooth_mask(fake_mask, smoothing_factor=0.5)
        elif shape_complexity == 'complex':
            fake_mask = self.add_noise_to_mask(fake_mask, noise_level=0.2)
            fake_mask = self.smooth_mask(fake_mask, smoothing_factor=0.3)
        
        # Get bounding box
        fake_mask_np = np.array(fake_mask)
        fake_contours, _ = cv2.findContours(fake_mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        fake_x, fake_y, fake_w, fake_h = cv2.boundingRect(fake_contours[0])
        
        # Apply mask to create the piece
        fake_rgba = Image.new("RGBA", pil_img.size)
        fake_rgba.paste(pil_img)
        fake_rgba.putalpha(fake_mask)
        
        return fake_rgba.crop((fake_x, fake_y, fake_x + fake_w, fake_y + fake_h))
    
    def _jigsaw_strategy_shift(self, pil_img, mask, mask_np, x, y, w, h, width, height, margin):
        """Helper for jigsaw distractor strategy: shifting the piece location."""
        # Take from a different area
        # But keep the same shape
        fake_x = random.randint(margin, width - w - margin)
        fake_y = random.randint(margin, height - h - margin)
        
        # Make sure it's far enough from the original
        while abs(fake_x - x) < w/2 and abs(fake_y - y) < h/2:
            fake_x = random.randint(margin, width - w - margin)
            fake_y = random.randint(margin, height - h - margin)
        
        # Shift the mask to the new position
        shifted_mask = Image.new("L", (width, height), 0)
        shifted_mask.paste(mask.crop((x, y, x+w, y+h)), (fake_x, fake_y))
        
        # Apply to original image
        fake_rgba = Image.new("RGBA", pil_img.size)
        fake_rgba.paste(pil_img)
        
        fake_rgba.putalpha(shifted_mask)
        return fake_rgba.crop((fake_x, fake_y, fake_x + w, fake_y + h))
    
    def _jigsaw_strategy_modify_tabs(self, pil_img, jigsaw_features, left, top, 
                                     cutout_width, cutout_height, tab_width, tab_height,
                                     shape_complexity, width, height):
        """Helper for jigsaw distractor strategy: modifying tab shapes."""
        # Create a new mask with modified tab shapes
        fake_mask = Image.new("L", (width, height), 0)
        fake_draw = ImageDraw.Draw(fake_mask)
        
        # Draw the base rectangle
        fake_draw.rectangle((left, top, left + cutout_width, top + cutout_height), fill=255)
        
        # Modify tab shapes
        new_complexity = shape_complexity
        if shape_complexity == 'simple':
            new_complexity = random.choice(['medium', 'complex'])
        elif shape_complexity == 'medium':
            new_complexity = random.choice(['simple', 'complex'])
        else:  # complex
            new_complexity = random.choice(['simple', 'medium'])
        
        # Redraw all tabs/blanks with the new complexity
        for side, features in jigsaw_features.items():
            for feature in features:
                is_tab = feature["is_tab"]
                
                if side == "top":
                    center_x = feature["center"]
                    feature_left = feature["left"]
                    feature_right = feature["right"]
                    
                    if is_tab:
                        # Outward tab on top
                        self._draw_tab_shape(fake_draw, new_complexity, 
                                           feature_left, top - tab_height, 
                                           feature_right, top, is_horizontal=True)
                    else:
                        # Inward blank on top
                        self._draw_tab_shape(fake_draw, new_complexity, 
                                           feature_left, top, 
                                           feature_right, top + tab_height, 
                                           is_horizontal=True, is_blank=True)
                
                elif side == "bottom":
                    center_x = feature["center"]
                    feature_left = feature["left"]
                    feature_right = feature["right"]
                    
                    if is_tab:
                        # Outward tab on bottom
                        self._draw_tab_shape(fake_draw, new_complexity, 
                                           feature_left, top + cutout_height, 
                                           feature_right, top + cutout_height + tab_height, 
                                           is_horizontal=True)
                    else:
                        # Inward blank on bottom
                        self._draw_tab_shape(fake_draw, new_complexity, 
                                           feature_left, top + cutout_height - tab_height, 
                                           feature_right, top + cutout_height, 
                                           is_horizontal=True, is_blank=True)
                
                elif side == "left":
                    center_y = feature["center"]
                    feature_top = feature["top"]
                    feature_bottom = feature["bottom"]
                    
                    if is_tab:
                        # Outward tab on left
                        self._draw_tab_shape(fake_draw, new_complexity, 
                                           left - tab_width, feature_top, 
                                           left, feature_bottom, is_horizontal=False)
                    else:
                        # Inward blank on left
                        self._draw_tab_shape(fake_draw, new_complexity, 
                                           left, feature_top, 
                                           left + tab_width, feature_bottom, 
                                           is_horizontal=False, is_blank=True)
                
                elif side == "right":
                    center_y = feature["center"]
                    feature_top = feature["top"]
                    feature_bottom = feature["bottom"]
                    
                    if is_tab:
                        # Outward tab on right
                        self._draw_tab_shape(fake_draw, new_complexity, 
                                           left + cutout_width, feature_top, 
                                           left + cutout_width + tab_width, feature_bottom, 
                                           is_horizontal=False)
                    else:
                        # Inward blank on right
                        self._draw_tab_shape(fake_draw, new_complexity, 
                                           left + cutout_width - tab_width, feature_top, 
                                           left + cutout_width, feature_bottom, 
                                           is_horizontal=False, is_blank=True)
        
        # Apply noise and smoothing
        if new_complexity == 'medium':
            fake_mask = self.add_noise_to_mask(fake_mask, noise_level=0.15)
            fake_mask = self.smooth_mask(fake_mask, smoothing_factor=0.4)
        elif new_complexity == 'complex':
            fake_mask = self.add_noise_to_mask(fake_mask, noise_level=0.25)
            fake_mask = self.smooth_mask(fake_mask, smoothing_factor=0.2)
        
        # Get bounding box
        fake_mask_np = np.array(fake_mask)
        fake_contours, _ = cv2.findContours(fake_mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        fake_x, fake_y, fake_w, fake_h = cv2.boundingRect(fake_contours[0])
        
        # Apply mask to create the piece
        fake_rgba = Image.new("RGBA", pil_img.size)
        fake_rgba.paste(pil_img)
        fake_rgba.putalpha(fake_mask)
        
        return fake_rgba.crop((fake_x, fake_y, fake_x + fake_w, fake_y + fake_h))
    
    ##############################
    # 2. MULTI-PIECE PUZZLE TASKS
    ##############################
    
    def generate_multi_piece_puzzle(self, 
                                   image_path: str, 
                                   output_dir: str, 
                                   num_options: int = 4, 
                                   difficulty: str = 'medium',
                                   custom_params: Optional[Dict] = None,
                                   distractor_types: Optional[List[str]] = None) -> Dict:
        """
        Generate a task with multiple pieces of a grid removed, where the model
        must identify which piece belongs in a specific location.
        
        Args:
            image_path: Path to the source image
            output_dir: Directory to save the task files
            num_options: Number of piece options to generate (default: 4)
            difficulty: Difficulty level ('easy', 'medium', 'hard')
            custom_params: Optional dict to override default parameters
            distractor_types: List of distractor strategies to use (randomly chosen if None)
            
        Returns:
            Dictionary with task information
        """
        # Create output directory
        task_dir = os.path.join(output_dir, f"multi_piece_task_{difficulty}")
        os.makedirs(task_dir, exist_ok=True)
        
        # Get parameters based on difficulty level
        params = self.multi_piece_params.get(difficulty, self.multi_piece_params['medium']).copy()
        
        # Override with custom parameters if provided
        if custom_params:
            params.update(custom_params)
        
        # Load the image
        pil_img, cv_img = self.load_image(image_path)
        width, height = pil_img.size
        
        # Create a grid
        grid_size = params['grid_size']
        cell_width = width // grid_size
        cell_height = height // grid_size
        
        # Create a list of all grid cells
        cells = []
        for row in range(grid_size):
            for col in range(grid_size):
                cells.append((col, row))  # (x, y) cell coordinates
        
        # Select cells to remove
        num_removed = params['num_pieces_removed']
        removed_cells = random.sample(cells, num_removed)
        
        # Select one cell that will be the "test" piece (the one we ask to identify)
        test_cell_idx = random.randint(0, num_removed - 1)
        test_cell = removed_cells[test_cell_idx]
        
        # Create the masked image with gray rectangles for removed cells
        masked_img = pil_img.copy()
        draw = ImageDraw.Draw(masked_img)
        
        for cell in removed_cells:
            col, row = cell
            left = col * cell_width
            top = row * cell_height
            right = left + cell_width
            bottom = top + cell_height
            
            draw.rectangle((left, top, right, bottom), fill=(128, 128, 128))
        
        # Draw grid lines for better visualization
        for i in range(grid_size + 1):
            # Vertical lines
            x = i * cell_width
            draw.line([(x, 0), (x, height)], fill=(200, 200, 200), width=1)
            
            # Horizontal lines
            y = i * cell_height
            draw.line([(0, y), (width, y)], fill=(200, 200, 200), width=1)
        
        # Extract the correct piece (test cell)
        test_col, test_row = test_cell
        left = test_col * cell_width
        top = test_row * cell_height
        right = left + cell_width
        bottom = top + cell_height
        
        correct_piece = pil_img.crop((left, top, right, bottom))
        
        # Save the masked image with highlighted target cell
        highlight_img = masked_img.copy()
        hdraw = ImageDraw.Draw(highlight_img)
        
        # Draw highlighted rectangle around the target cell
        hdraw.rectangle((left, top, right, bottom), outline=(255, 0, 0), width=3)
        
        # Save both versions of the masked image
        masked_img_path = os.path.join(task_dir, "masked_image.jpg")
        masked_img.save(masked_img_path)
        
        highlighted_img_path = os.path.join(task_dir, "highlighted_target.jpg")
        highlight_img.save(highlighted_img_path)
        
        # Generate the options
        correct_idx = random.randint(0, num_options - 1)
        options = []
        option_paths = []
        
        # Select distractor strategies if not specified
        if not distractor_types:
            num_strategies = min(num_options - 1, len(self.distractor_strategies['multi_piece']))
            distractor_types = random.sample(self.distractor_strategies['multi_piece'], num_strategies)
        
        for i in range(num_options):
            if i == correct_idx:
                # This is the correct piece
                piece = correct_piece
            else:
                # Generate an incorrect piece using different strategies
                strategy_idx = (i - 1) % len(distractor_types)
                strategy = distractor_types[strategy_idx]
                
                if strategy == 'other_cell':
                    # Strategy 1: Take a piece from another cell
                    piece = self._multi_piece_strategy_other_cell(
                        pil_img, cells, removed_cells, test_cell,
                        cell_width, cell_height, params['distractor_difficulty']
                    )
                
                elif strategy == 'transform':
                    # Strategy 2: Apply transformations to the correct piece
                    piece = self._multi_piece_strategy_transform(
                        correct_piece, params['distractor_difficulty']
                    )
                
                elif strategy == 'shifted_grid':
                    # Strategy 3: Shift the grid slightly and take a piece
                    piece = self._multi_piece_strategy_shifted_grid(
                        pil_img, test_cell, grid_size, cell_width, cell_height,
                        params['distractor_difficulty']
                    )
                
                elif strategy == 'color_modify':
                    # Strategy 4: Modify the colors of the correct piece
                    piece = self._multi_piece_strategy_color_modify(
                        correct_piece, params['distractor_difficulty']
                    )
                
                elif strategy == 'pattern_modify':
                    # Strategy 5: Add or remove patterns from the correct piece
                    piece = self._multi_piece_strategy_pattern_modify(
                        correct_piece, params['distractor_difficulty']
                    )
                
                else:
                    # Default fallback: Take from a random cell
                    available_cells = [cell for cell in cells if cell != test_cell]
                    random_cell = random.choice(available_cells)
                    
                    r_col, r_row = random_cell
                    r_left = r_col * cell_width
                    r_top = r_row * cell_height
                    r_right = r_left + cell_width
                    r_bottom = r_top + cell_height
                    
                    piece = pil_img.crop((r_left, r_top, r_right, r_bottom))
            
            # Save the piece
            piece_path = os.path.join(task_dir, f"option_{chr(65+i)}.jpg")
            piece.save(piece_path)
            
            options.append(piece)
            option_paths.append(piece_path)
        
        # Save the original image for reference
        original_path = os.path.join(task_dir, "original.jpg")
        pil_img.save(original_path)
        
        # Create a visualization of the task
        vis_path = self._create_visualization(
            highlight_img, options, correct_idx, task_dir, show_grid=True, 
            grid_size=grid_size, target_cell=test_cell
        )
        
        # Save task metadata
        metadata = {
            "task_type": "multi_piece_puzzle",
            "difficulty": difficulty,
            "original_image": os.path.basename(original_path),
            "masked_image": os.path.basename(masked_img_path),
            "highlighted_target": os.path.basename(highlighted_img_path),
            "options": [os.path.basename(path) for path in option_paths],
            "correct_idx": correct_idx,
            "correct_option": chr(65 + correct_idx),
            "grid_size": grid_size,
            "cell_width": cell_width,
            "cell_height": cell_height,
            "removed_cells": removed_cells,
            "test_cell": test_cell,
            "parameters": params,
            "distractor_strategies": [distractor_types[i-1] if i != correct_idx else "correct" 
                                     for i in range(num_options)]
        }
        
        metadata_path = os.path.join(task_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        return {
            "task_dir": task_dir,
            "masked_image": masked_img_path,
            "highlighted_target": highlighted_img_path,
            "options": option_paths,
            "correct_idx": correct_idx,
            "metadata": metadata
        }
    
    def _multi_piece_strategy_other_cell(self, pil_img, cells, removed_cells, test_cell,
                                         cell_width, cell_height, distractor_difficulty):
        """Helper for multi-piece distractor strategy: taking from another cell."""
        other_cells = [cell for cell in cells if cell != test_cell]
        
        # For harder distractors, prefer cells that are close to the test cell
        test_col, test_row = test_cell
        
        if distractor_difficulty == 'high':
            # Sort cells by distance to test cell, take one of the closest
            other_cells.sort(key=lambda cell: abs(cell[0] - test_col) + abs(cell[1] - test_row))
            cell_candidates = other_cells[:max(3, len(other_cells) // 3)]
            fake_cell = random.choice(cell_candidates)
            
        elif distractor_difficulty == 'medium':
            # Take a cell that's moderately distant
            cell_candidates = [cell for cell in other_cells 
                              if 1 <= abs(cell[0] - test_col) + abs(cell[1] - test_row) <= 3]
            if not cell_candidates:
                cell_candidates = other_cells
            fake_cell = random.choice(cell_candidates)
            
        else:  # 'low' difficulty
            # Take any random cell
            fake_cell = random.choice(other_cells)
        
        # Crop the image at the cell position
        fake_col, fake_row = fake_cell
        left = fake_col * cell_width
        top = fake_row * cell_height
        right = left + cell_width
        bottom = top + cell_height
        
        return pil_img.crop((left, top, right, bottom))
    
    def _multi_piece_strategy_transform(self, correct_piece, distractor_difficulty):
        """Helper for multi-piece distractor strategy: transforming the correct piece."""
        piece = correct_piece.copy()
        
        if distractor_difficulty == 'high':
            # Apply multiple subtle transformations
            transforms = []
            
            # Potentially rotate by small angle
            if random.choice([True, False]):
                angle = random.uniform(-15, 15)
                transforms.append(lambda img: img.rotate(angle, expand=False))
            
            # Potentially flip
            if random.choice([True, False]):
                transforms.append(lambda img: ImageOps.mirror(img))
            
            # Potentially slight color shift
            if random.choice([True, False]):
                factor = random.uniform(0.9, 1.1)
                transforms.append(lambda img: ImageEnhance.Color(img).enhance(factor))
            
            # Apply the transformations
            for transform in transforms:
                piece = transform(piece)
                
        elif distractor_difficulty == 'medium':
            # Apply a moderate transformation
            transform_type = random.choice(['rotate', 'flip', 'color'])
            
            if transform_type == 'rotate':
                angle = random.uniform(-30, 30)
                piece = piece.rotate(angle, expand=False)
            elif transform_type == 'flip':
                piece = ImageOps.mirror(piece)
            else:  # 'color'
                factor = random.uniform(0.8, 1.2)
                piece = ImageEnhance.Color(piece).enhance(factor)
                
        else:  # 'low' difficulty
            # Apply a significant transformation
            transform_type = random.choice(['rotate', 'flip', 'color'])
            
            if transform_type == 'rotate':
                angle = random.choice([90, 180, 270])
                piece = piece.rotate(angle, expand=False)
            elif transform_type == 'flip':
                if random.choice([True, False]):
                    piece = ImageOps.mirror(piece)
                else:
                    piece = ImageOps.flip(piece)
            else:  # 'color'
                factor = random.uniform(0.5, 1.5)
                piece = ImageEnhance.Color(piece).enhance(factor)
        
        return piece
    
    def _multi_piece_strategy_shifted_grid(self, pil_img, test_cell, grid_size, 
                                          cell_width, cell_height, distractor_difficulty):
        """Helper for multi-piece distractor strategy: shifted grid cells."""
        test_col, test_row = test_cell
        width, height = pil_img.size
        
        # Calculate shift amount based on difficulty
        if distractor_difficulty == 'high':
            # Small shift
            shift_x = random.randint(-cell_width // 4, cell_width // 4)
            shift_y = random.randint(-cell_height // 4, cell_height // 4)
            
        elif distractor_difficulty == 'medium':
            # Medium shift
            shift_x = random.randint(-cell_width // 3, cell_width // 3)
            shift_y = random.randint(-cell_height // 3, cell_height // 3)
            
        else:  # 'low' difficulty
            # Large shift
            shift_x = random.randint(-cell_width // 2, cell_width // 2)
            shift_y = random.randint(-cell_height // 2, cell_height // 2)
        
        # Calculate the shifted coordinates
        left = (test_col * cell_width) + shift_x
        top = (test_row * cell_height) + shift_y
        
        # Make sure it stays within bounds
        left = max(0, min(width - cell_width, left))
        top = max(0, min(height - cell_height, top))
        
        right = left + cell_width
        bottom = top + cell_height
        
        return pil_img.crop((left, top, right, bottom))
    
    def _multi_piece_strategy_color_modify(self, correct_piece, distractor_difficulty):
        """Helper for multi-piece distractor strategy: color modifications."""
        piece = correct_piece.copy()
        
        if distractor_difficulty == 'high':
            # Subtle color changes
            transforms = []
            
            # Choose 1-2 color transformations
            for _ in range(random.randint(1, 2)):
                transform_type = random.choice(['color', 'brightness', 'contrast'])
                
                if transform_type == 'color':
                    factor = random.uniform(0.9, 1.1)
                    transforms.append(lambda img: ImageEnhance.Color(img).enhance(factor))
                elif transform_type == 'brightness':
                    factor = random.uniform(0.9, 1.1)
                    transforms.append(lambda img: ImageEnhance.Brightness(img).enhance(factor))
                else:  # 'contrast'
                    factor = random.uniform(0.9, 1.1)
                    transforms.append(lambda img: ImageEnhance.Contrast(img).enhance(factor))
            
            # Apply the transformations
            for transform in transforms:
                piece = transform(piece)
                
        elif distractor_difficulty == 'medium':
            # Moderate color changes
            transform_type = random.choice(['color', 'brightness', 'contrast'])
            
            if transform_type == 'color':
                factor = random.uniform(0.7, 1.3)
                piece = ImageEnhance.Color(piece).enhance(factor)
            elif transform_type == 'brightness':
                factor = random.uniform(0.7, 1.3)
                piece = ImageEnhance.Brightness(piece).enhance(factor)
            else:  # 'contrast'
                factor = random.uniform(0.7, 1.3)
                piece = ImageEnhance.Contrast(piece).enhance(factor)
                
        else:  # 'low' difficulty
            # Significant color changes
            transforms = []
            
            # Apply multiple significant transformations
            for _ in range(random.randint(1, 3)):
                transform_type = random.choice(['color', 'brightness', 'contrast', 'invert'])
                
                if transform_type == 'color':
                    factor = random.uniform(0.5, 1.5)
                    transforms.append(lambda img: ImageEnhance.Color(img).enhance(factor))
                elif transform_type == 'brightness':
                    factor = random.uniform(0.5, 1.5)
                    transforms.append(lambda img: ImageEnhance.Brightness(img).enhance(factor))
                elif transform_type == 'contrast':
                    factor = random.uniform(0.5, 1.5)
                    transforms.append(lambda img: ImageEnhance.Contrast(img).enhance(factor))
                else:  # 'invert'
                    transforms.append(lambda img: ImageOps.invert(img))
            
            # Apply the transformations
            for transform in transforms:
                piece = transform(piece)
        
        return piece
    
    def _multi_piece_strategy_pattern_modify(self, correct_piece, distractor_difficulty):
        """Helper for multi-piece distractor strategy: adding/removing patterns."""
        piece = correct_piece.copy()
        width, height = piece.size
        draw = ImageDraw.Draw(piece)
        
        if distractor_difficulty == 'high':
            # Subtle pattern modifications
            if random.choice([True, False]):
                # Add subtle noise
                for _ in range(int(width * height * 0.01)):  # Cover 1% of pixels
                    x = random.randint(0, width - 1)
                    y = random.randint(0, height - 1)
                    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                    draw.point((x, y), fill=color)
            else:
                # Add a very subtle line or shape
                line_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                x1 = random.randint(0, width)
                y1 = random.randint(0, height)
                x2 = random.randint(0, width)
                y2 = random.randint(0, height)
                draw.line([(x1, y1), (x2, y2)], fill=line_color, width=1)
                
        elif distractor_difficulty == 'medium':
            # Moderate pattern modifications
            pattern_type = random.choice(['noise', 'lines', 'shapes', 'text'])
            
            if pattern_type == 'noise':
                # Add moderate noise
                for _ in range(int(width * height * 0.05)):  # Cover 5% of pixels
                    x = random.randint(0, width - 1)
                    y = random.randint(0, height - 1)
                    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                    draw.point((x, y), fill=color)
                    
            elif pattern_type == 'lines':
                # Add a few lines
                for _ in range(random.randint(1, 3)):
                    line_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                    x1 = random.randint(0, width)
                    y1 = random.randint(0, height)
                    x2 = random.randint(0, width)
                    y2 = random.randint(0, height)
                    draw.line([(x1, y1), (x2, y2)], fill=line_color, width=random.randint(1, 2))
                    
            elif pattern_type == 'shapes':
                # Add a simple shape
                shape_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                shape_type = random.choice(['rectangle', 'circle'])
                
                if shape_type == 'rectangle':
                    x1 = random.randint(0, width // 2)
                    y1 = random.randint(0, height // 2)
                    x2 = random.randint(x1 + width // 4, width)
                    y2 = random.randint(y1 + height // 4, height)
                    draw.rectangle([x1, y1, x2, y2], outline=shape_color, width=1)
                else:  # circle
                    x1 = random.randint(width // 4, 3 * width // 4)
                    y1 = random.randint(height // 4, 3 * height // 4)
                    r = min(width, height) // 6
                    draw.ellipse((x1 - r, y1 - r, x1 + r, y1 + r), outline=shape_color, width=1)
                    
            else:  # text
                # Add small text
                text_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                try:
                    draw.text((width // 4, height // 4), "X", fill=text_color)
                except:
                    # Fallback if text drawing fails
                    x1 = random.randint(0, width // 2)
                    y1 = random.randint(0, height // 2)
                    x2 = random.randint(x1 + 10, x1 + 20)
                    y2 = random.randint(y1 + 10, y1 + 20)
                    draw.rectangle([x1, y1, x2, y2], fill=text_color)
                
        else:  # 'low' difficulty
            # Significant pattern modifications
            pattern_type = random.choice(['heavy_noise', 'grid', 'large_shapes', 'overlay'])
            
            if pattern_type == 'heavy_noise':
                # Add significant noise
                for _ in range(int(width * height * 0.2)):  # Cover 20% of pixels
                    x = random.randint(0, width - 1)
                    y = random.randint(0, height - 1)
                    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                    draw.point((x, y), fill=color)
                    
            elif pattern_type == 'grid':
                # Add a grid pattern
                grid_size = random.randint(4, 8)
                cell_w = width // grid_size
                cell_h = height // grid_size
                
                grid_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                
                for i in range(grid_size + 1):
                    # Vertical lines
                    x = i * cell_w
                    draw.line([(x, 0), (x, height)], fill=grid_color, width=1)
                    
                    # Horizontal lines
                    y = i * cell_h
                    draw.line([(0, y), (width, y)], fill=grid_color, width=1)
                    
            elif pattern_type == 'large_shapes':
                # Add a large shape
                shape_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                shape_type = random.choice(['rectangle', 'circle', 'triangle'])
                
                if shape_type == 'rectangle':
                    x1 = random.randint(0, width // 3)
                    y1 = random.randint(0, height // 3)
                    x2 = random.randint(2 * width // 3, width)
                    y2 = random.randint(2 * height // 3, height)
                    draw.rectangle([x1, y1, x2, y2], outline=shape_color, width=2)
                    
                elif shape_type == 'circle':
                    x1 = width // 2
                    y1 = height // 2
                    r = min(width, height) // 3
                    draw.ellipse((x1 - r, y1 - r, x1 + r, y1 + r), outline=shape_color, width=2)
                    
                else:  # triangle
                    draw.polygon([
                        (width // 2, height // 4),
                        (width // 4, 3 * height // 4),
                        (3 * width // 4, 3 * height // 4)
                    ], outline=shape_color, width=2)
                    
            else:  # overlay
                # Create a semi-transparent overlay
                overlay = Image.new('RGBA', (width, height), 
                                   (random.randint(0, 255), 
                                    random.randint(0, 255), 
                                    random.randint(0, 255), 
                                    128))  # Alpha = 128 (semi-transparent)
                
                # Convert piece to RGBA for blending
                if piece.mode != 'RGBA':
                    piece = piece.convert('RGBA')
                
                # Blend the images
                piece = Image.alpha_composite(piece, overlay)
        
        return piece
    
    ##############################
    # 3. FRACTAL CUTOUT TASKS
    ##############################
    
    def generate_fractal_cutout(self, 
                               image_path: str, 
                               output_dir: str, 
                               num_options: int = 4, 
                               difficulty: str = 'medium',
                               custom_params: Optional[Dict] = None,
                               distractor_types: Optional[List[str]] = None) -> Dict:
        """
        Generate a task with a complex fractal-like edge cutout from an image.
        
        Args:
            image_path: Path to the source image
            output_dir: Directory to save the task files
            num_options: Number of piece options to generate (default: 4)
            difficulty: Difficulty level ('easy', 'medium', 'hard')
            custom_params: Optional dict to override default parameters
            distractor_types: List of distractor strategies to use (randomly chosen if None)
            
        Returns:
            Dictionary with task information
        """
        # Create output directory
        task_dir = os.path.join(output_dir, f"fractal_cutout_task_{difficulty}")
        os.makedirs(task_dir, exist_ok=True)
        
        # Get parameters based on difficulty level
        params = self.fractal_params.get(difficulty, self.fractal_params['medium']).copy()
        
        # Override with custom parameters if provided
        if custom_params:
            params.update(custom_params)
        
        # Load the image
        pil_img, cv_img = self.load_image(image_path)
        width, height = pil_img.size
        
        # Create a base shape for the cutout
        base_x = random.randint(width // 4, 3 * width // 4)
        base_y = random.randint(height // 4, 3 * height // 4)
        base_size = min(width, height) // 4
        
        # Create a mask for the fractal shape
        mask = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask)
        
        # Set parameters for the fractal generation
        max_depth = params['complexity']
        irregularity = params['irregularity']
        
        # Create a fractal-like edge by recursive subdivision
        def draw_fractal_edge(x, y, size, depth=0):
            if depth >= max_depth or size < 5:
                # Draw a simple shape at the leaf nodes
                draw.ellipse((x - size//2, y - size//2, x + size//2, y + size//2), fill=255)
                return
            
            # Draw the main shape
            draw.ellipse((x - size//2, y - size//2, x + size//2, y + size//2), fill=255)
            
            # Number of children varies by depth
            if depth == 0:
                num_children = random.randint(4, 6)
            else:
                num_children = random.randint(3, 5)
            
            # Create child nodes around the perimeter
            for i in range(num_children):
                angle = 2 * math.pi * i / num_children
                # Add randomness to angle based on irregularity parameter
                angle += random.uniform(-irregularity, irregularity)
                
                # Calculate position of child (near the edge of parent)
                distance = size//2 * random.uniform(0.9, 1.1)
                child_x = int(x + distance * math.cos(angle))
                child_y = int(y + distance * math.sin(angle))
                
                # Child size is a fraction of parent
                child_size = int(size * random.uniform(0.3, 0.5))
                
                # Recursive call for child
                draw_fractal_edge(child_x, child_y, child_size, depth + 1)
        
        # Start the fractal drawing
        draw_fractal_edge(base_x, base_y, base_size)
        
        # Apply smoothing based on the parameter
        smoothing_factor = params['smoothing']
        mask = self.smooth_mask(mask, smoothing_factor)
        
        # Convert mask to numpy array for OpenCV operations
        mask_np = np.array(mask)
        
        # Find the bounding box of the shape
        contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            # Fallback if no contours were found
            print("No fractal contours found, using rectangular cutout instead")
            return self.generate_jigsaw_piece(image_path, output_dir, num_options, difficulty)
            
        x, y, w, h = cv2.boundingRect(contours[0])
        
        # Create the masked image (with gray filling the cutout)
        masked_pil = pil_img.copy()
        masked_np = np.array(masked_pil)
        masked_np[mask_np > 0] = [128, 128, 128]
        masked_img = Image.fromarray(masked_np)
        
        # Extract the correct cutout piece with transparency
        img_rgba = Image.new("RGBA", pil_img.size)
        img_rgba.paste(pil_img)
        
        # Create alpha channel from mask
        img_rgba.putalpha(mask)
        
        # Crop to bounding box
        correct_piece = img_rgba.crop((x, y, x+w, y+h))
        
        # Save the masked image
        masked_img_path = os.path.join(task_dir, "masked_image.jpg")
        masked_img.save(masked_img_path)
        
        # Generate the options
        correct_idx = random.randint(0, num_options - 1)
        options = []
        option_paths = []
        
        # Select distractor strategies if not specified
        if not distractor_types:
            num_strategies = min(num_options - 1, len(self.distractor_strategies['fractal']))
            distractor_types = random.sample(self.distractor_strategies['fractal'], num_strategies)
        
        for i in range(num_options):
            if i == correct_idx:
                # This is the correct piece
                piece = correct_piece
            else:
                # Generate an incorrect piece using different strategies
                strategy_idx = (i - 1) % len(distractor_types)
                strategy = distractor_types[strategy_idx]
                
                if strategy == 'different_seed':
                    # Strategy 1: Generate a different fractal pattern
                    piece = self._fractal_strategy_different_seed(
                        pil_img, width, height, base_x, base_y, base_size,
                        max_depth, irregularity, smoothing_factor
                    )
                
                elif strategy == 'transform':
                    # Strategy 2: Apply transformations to the correct piece
                    piece = self._fractal_strategy_transform(
                        correct_piece, difficulty
                    )
                
                elif strategy == 'boundary_modify':
                    # Strategy 3: Modify the boundary of the fractal
                    piece = self._fractal_strategy_boundary_modify(
                        pil_img, mask, mask_np, x, y, w, h, difficulty
                    )
                
                elif strategy == 'shift':
                    # Strategy 4: Take from a different area with the same mask
                    piece = self._fractal_strategy_shift(
                        pil_img, mask, mask_np, x, y, w, h, width, height
                    )
                
                elif strategy == 'hybrid':
                    # Strategy 5: Combine multiple strategies
                    piece = self._fractal_strategy_hybrid(
                        pil_img, correct_piece, mask, mask_np, x, y, w, h, 
                        width, height, base_x, base_y, base_size, 
                        max_depth, irregularity, smoothing_factor, difficulty
                    )
                
                else:
                    # Default fallback: Take a random crop with similar size
                    fake_x = random.randint(0, width - w - 1)
                    fake_y = random.randint(0, height - h - 1)
                    
                    # Make sure it's far enough from the original
                    while abs(fake_x - x) < w/2 and abs(fake_y - y) < h/2:
                        fake_x = random.randint(0, width - w - 1)
                        fake_y = random.randint(0, height - h - 1)
                    
                    # Create a random shape
                    fake_mask = Image.new("L", (width, height), 0)
                    fake_draw = ImageDraw.Draw(fake_mask)
                    
                    # Draw random ellipse or polygon
                    if random.choice([True, False]):
                        # Ellipse
                        fake_draw.ellipse((fake_x, fake_y, fake_x + w, fake_y + h), fill=255)
                    else:
                        # Polygon
                        points = []
                        num_points = random.randint(5, 8)
                        center_x = fake_x + w // 2
                        center_y = fake_y + h // 2
                        
                        for j in range(num_points):
                            angle = 2 * math.pi * j / num_points
                            r = min(w, h) // 2 * random.uniform(0.8, 1.2)
                            point_x = center_x + int(r * math.cos(angle))
                            point_y = center_y + int(r * math.sin(angle))
                            points.append((point_x, point_y))
                        
                        fake_draw.polygon(points, fill=255)
                    
                    # Apply to image
                    fake_rgba = Image.new("RGBA", pil_img.size)
                    fake_rgba.paste(pil_img)
                    fake_rgba.putalpha(fake_mask)
                    
                    # Find new bounding box
                    fake_mask_np = np.array(fake_mask)
                    fake_contours, _ = cv2.findContours(fake_mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    if fake_contours:
                        fx, fy, fw, fh = cv2.boundingRect(fake_contours[0])
                        piece = fake_rgba.crop((fx, fy, fx + fw, fy + fh))
                    else:
                        # Fallback
                        piece = fake_rgba.crop((fake_x, fake_y, fake_x + w, fake_y + h))
            
            # Save the piece
            piece_path = os.path.join(task_dir, f"option_{chr(65+i)}.png")
            piece.save(piece_path)
            
            options.append(piece)
            option_paths.append(piece_path)
        
        # Save the original image for reference
        original_path = os.path.join(task_dir, "original.jpg")
        pil_img.save(original_path)
        
        # Create a visualization of the task
        vis_path = self._create_visualization(masked_img, options, correct_idx, task_dir, show_border=True)
        
        # Save task metadata
        metadata = {
            "task_type": "fractal_cutout",
            "difficulty": difficulty,
            "original_image": os.path.basename(original_path),
            "masked_image": os.path.basename(masked_img_path),
            "options": [os.path.basename(path) for path in option_paths],
            "correct_idx": correct_idx,
            "correct_option": chr(65 + correct_idx),
            "cutout_bounds": {
                "x": x,
                "y": y,
                "width": w,
                "height": h
            },
            "base_params": {
                "x": base_x,
                "y": base_y,
                "size": base_size
            },
            "parameters": params,
            "distractor_strategies": [distractor_types[i-1] if i != correct_idx else "correct" 
                                     for i in range(num_options)]
        }
        
        metadata_path = os.path.join(task_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        return {
            "task_dir": task_dir,
            "masked_image": masked_img_path,
            "options": option_paths,
            "correct_idx": correct_idx,
            "metadata": metadata
        }
    
    def _fractal_strategy_different_seed(self, pil_img, width, height, base_x, base_y, 
                                        base_size, max_depth, irregularity, smoothing_factor):
        """Helper for fractal distractor strategy: different fractal seed."""
        # Create a new fractal with different parameters
        fake_mask = Image.new("L", (width, height), 0)
        fake_draw = ImageDraw.Draw(fake_mask)
        
        # Slightly vary the parameters
        fake_x = base_x + random.randint(-base_size//3, base_size//3)
        fake_y = base_y + random.randint(-base_size//3, base_size//3)
        fake_size = base_size * random.uniform(0.8, 1.2)
        fake_max_depth = max_depth
        fake_irregularity = irregularity * random.uniform(0.8, 1.2)
        
        # Function to draw a different fractal pattern
        def draw_different_fractal(x, y, size, depth=0):
            if depth >= fake_max_depth or size < 5:
                # At leaf nodes, draw a different shape (rectangle instead of ellipse)
                half_size = size // 2
                fake_draw.rectangle((x - half_size, y - half_size, 
                                    x + half_size, y + half_size), fill=255)
                return
            
            # Main shape - use an ellipse or rectangle randomly
            if random.choice([True, False]):
                fake_draw.ellipse((x - size//2, y - size//2, x + size//2, y + size//2), fill=255)
            else:
                fake_draw.rectangle((x - size//2, y - size//2, x + size//2, y + size//2), fill=255)
            
            # Different number of children
            num_children = random.randint(3, 6)
            
            for i in range(num_children):
                angle = 2 * math.pi * i / num_children
                # Different randomness in angle
                angle += random.uniform(-fake_irregularity, fake_irregularity)
                
                # Position of child
                distance = size//2 * random.uniform(0.8, 1.2)
                child_x = int(x + distance * math.cos(angle))
                child_y = int(y + distance * math.sin(angle))
                
                # Different size ratio
                child_size = int(size * random.uniform(0.25, 0.55))
                
                draw_different_fractal(child_x, child_y, child_size, depth + 1)
        
        # Draw the fractal
        draw_different_fractal(fake_x, fake_y, fake_size)
        
        # Apply smoothing
        fake_mask = self.smooth_mask(fake_mask, smoothing_factor)
        
        # Get the bounding box
        fake_mask_np = np.array(fake_mask)
        fake_contours, _ = cv2.findContours(fake_mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not fake_contours:
            # Fallback if no contours found
            fake_x, fake_y = width // 2, height // 2
            fake_w, fake_h = width // 4, height // 4
            
            # Create a simple shape
            fake_mask = Image.new("L", (width, height), 0)
            fake_draw = ImageDraw.Draw(fake_mask)
            fake_draw.ellipse((fake_x - fake_w//2, fake_y - fake_h//2, 
                              fake_x + fake_w//2, fake_y + fake_h//2), fill=255)
            
            fake_mask_np = np.array(fake_mask)
            fake_contours, _ = cv2.findContours(fake_mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not fake_contours:
                # If still no contours, create a simple rectangle
                fx, fy = width // 4, height // 4
                fw, fh = width // 2, height // 2
            else:
                fx, fy, fw, fh = cv2.boundingRect(fake_contours[0])
        else:
            fx, fy, fw, fh = cv2.boundingRect(fake_contours[0])
        
        # Apply to image
        fake_rgba = Image.new("RGBA", pil_img.size)
        fake_rgba.paste(pil_img)
        fake_rgba.putalpha(fake_mask)
        
        return fake_rgba.crop((fx, fy, fx + fw, fy + fh))
    
    def _fractal_strategy_transform(self, correct_piece, difficulty):
        """Helper for fractal distractor strategy: transforming the piece."""
        piece = correct_piece.copy()
        
        if difficulty == 'hard':
            # Subtle transformations
            transform_type = random.choice(['rotate_small', 'flip', 'slight_resize'])
            
            if transform_type == 'rotate_small':
                angle = random.uniform(-20, 20)  # Small rotation angle
                piece = piece.rotate(angle, expand=True, resample=Image.BICUBIC)
            elif transform_type == 'flip':
                if random.choice([True, False]):
                    piece = ImageOps.mirror(piece)
                else:
                    piece = ImageOps.flip(piece)
            else:  # slight_resize
                factor = random.uniform(0.9, 1.1)
                width, height = piece.size
                new_width, new_height = int(width * factor), int(height * factor)
                piece = piece.resize((new_width, new_height), Image.LANCZOS)
        
        elif difficulty == 'medium':
            # More noticeable transformations
            transform_type = random.choice(['rotate', 'flip', 'resize', 'color'])
            
            if transform_type == 'rotate':
                angle = random.uniform(-45, 45)  # Larger rotation angle
                piece = piece.rotate(angle, expand=True, resample=Image.BICUBIC)
            elif transform_type == 'flip':
                if random.choice([True, False]):
                    piece = ImageOps.mirror(piece)
                else:
                    piece = ImageOps.flip(piece)
            elif transform_type == 'resize':
                factor = random.uniform(0.8, 1.2)
                width, height = piece.size
                new_width, new_height = int(width * factor), int(height * factor)
                piece = piece.resize((new_width, new_height), Image.LANCZOS)
            else:  # color
                enhancer = ImageEnhance.Color(piece)
                factor = random.uniform(0.7, 1.3)
                piece = enhancer.enhance(factor)
        
        else:  # easy
            # Significant transformations
            transform_type = random.choice(['rotate_large', 'flip_both', 'large_resize', 'multiple'])
            
            if transform_type == 'rotate_large':
                angle = random.choice([90, 180, 270])  # Major rotation
                piece = piece.rotate(angle, expand=True)
            elif transform_type == 'flip_both':
                # Both horizontal and vertical flip
                piece = ImageOps.mirror(piece)
                piece = ImageOps.flip(piece)
            elif transform_type == 'large_resize':
                factor = random.uniform(0.5, 1.5)
                width, height = piece.size
                new_width, new_height = int(width * factor), int(height * factor)
                piece = piece.resize((new_width, new_height), Image.LANCZOS)
            else:  # multiple
                # Apply multiple transformations
                # 1. Rotate
                angle = random.uniform(-30, 30)
                piece = piece.rotate(angle, expand=True)
                
                # 2. Flip
                if random.choice([True, False]):
                    piece = ImageOps.mirror(piece)
                
                # 3. Color adjustment
                enhancer = ImageEnhance.Color(piece)
                factor = random.uniform(0.6, 1.4)
                piece = enhancer.enhance(factor)
        
        return piece
    
    def _fractal_strategy_boundary_modify(self, pil_img, mask, mask_np, x, y, w, h, difficulty):
        """Helper for fractal distractor strategy: modifying the boundary."""
        # Create a new mask based on the original but with modified boundaries
        modified_mask = mask.copy()
        
        # Convert to numpy for easier manipulation
        modified_mask_np = np.array(modified_mask)
        
        if difficulty == 'hard':
            # Subtle boundary modifications
            # Apply morphological operations with small kernels
            kernel_size = 3
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            
            if random.choice([True, False]):
                # Dilate (expand)
                modified_mask_np = cv2.dilate(modified_mask_np, kernel, iterations=1)
            else:
                # Erode (shrink)
                modified_mask_np = cv2.erode(modified_mask_np, kernel, iterations=1)
            
        elif difficulty == 'medium':
            # More noticeable boundary modifications
            kernel_size = random.choice([5, 7])
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            
            op_type = random.choice(['dilate', 'erode', 'open', 'close'])
            
            if op_type == 'dilate':
                modified_mask_np = cv2.dilate(modified_mask_np, kernel, iterations=1)
            elif op_type == 'erode':
                modified_mask_np = cv2.erode(modified_mask_np, kernel, iterations=1)
            elif op_type == 'open':
                modified_mask_np = cv2.morphologyEx(modified_mask_np, cv2.MORPH_OPEN, kernel)
            else:  # 'close'
                modified_mask_np = cv2.morphologyEx(modified_mask_np, cv2.MORPH_CLOSE, kernel)
            
        else:  # 'easy'
            # Significant boundary modifications
            kernel_size = random.choice([7, 9, 11])
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            
            # Apply multiple operations
            ops = random.sample(['dilate', 'erode', 'open', 'close'], k=2)
            
            for op in ops:
                if op == 'dilate':
                    modified_mask_np = cv2.dilate(modified_mask_np, kernel, iterations=1)
                elif op == 'erode':
                    modified_mask_np = cv2.erode(modified_mask_np, kernel, iterations=1)
                elif op == 'open':
                    modified_mask_np = cv2.morphologyEx(modified_mask_np, cv2.MORPH_OPEN, kernel)
                else:  # 'close'
                    modified_mask_np = cv2.morphologyEx(modified_mask_np, cv2.MORPH_CLOSE, kernel)
            
            # Add noise to the boundary
            # Create a noise mask
            noise = np.random.rand(*modified_mask_np.shape) * 255
            noise = cv2.GaussianBlur(noise, (15, 15), 0)
            
            # Find the boundary
            boundary = cv2.dilate(modified_mask_np, kernel) - cv2.erode(modified_mask_np, kernel)
            
            # Apply noise only to boundary areas
            modified_mask_np[boundary > 0] = (modified_mask_np[boundary > 0] * 0.7 + 
                                             noise[boundary > 0] * 0.3 > 128).astype(np.uint8) * 255
        
        # Convert back to PIL
        modified_mask = Image.fromarray(modified_mask_np)
        
        # Find new bounding box
        modified_mask_np = np.array(modified_mask)
        modified_contours, _ = cv2.findContours(modified_mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not modified_contours:
            # Fallback to original
            mod_x, mod_y, mod_w, mod_h = x, y, w, h
        else:
            mod_x, mod_y, mod_w, mod_h = cv2.boundingRect(modified_contours[0])
        
        # Apply to image
        modified_rgba = Image.new("RGBA", pil_img.size)
        modified_rgba.paste(pil_img)
        modified_rgba.putalpha(modified_mask)
        
        return modified_rgba.crop((mod_x, mod_y, mod_x + mod_w, mod_y + mod_h))
    
    def _fractal_strategy_shift(self, pil_img, mask, mask_np, x, y, w, h, width, height):
        """Helper for fractal distractor strategy: shifting the cutout location."""
        # Take from a different area with same shape
        # Find a new position that doesn't overlap too much with original
        attempts = 0
        max_attempts = 10
        
        while attempts < max_attempts:
            # Generate random shift
            shift_x = random.randint(-width//3, width//3)
            shift_y = random.randint(-height//3, height//3)
            
            # Skip if the shift is too small
            if abs(shift_x) < w//4 and abs(shift_y) < h//4:
                attempts += 1
                continue
            
            # Calculate new position
            new_x = x + shift_x
            new_y = y + shift_y
            
            # Check if it's within bounds
            if (new_x >= 0 and new_y >= 0 and 
                new_x + w < width and new_y + h < height):
                # Position is valid
                break
                
            attempts += 1
        
        if attempts >= max_attempts:
            # If no x position found, use a position that's at least within bounds
            new_x = max(0, min(width - w, x + random.randint(-w, w)))
            new_y = max(0, min(height - h, y + random.randint(-h, h)))
        
        # Shift the mask to the new position
        shifted_mask = Image.new("L", (width, height), 0)
        shifted_mask.paste(mask.crop((x, y, x+w, y+h)), (new_x, new_y))
        
        # Apply to the image
        fake_rgba = Image.new("RGBA", pil_img.size)
        fake_rgba.paste(pil_img)
        fake_rgba.putalpha(shifted_mask)
        
        return fake_rgba.crop((new_x, new_y, new_x + w, new_y + h))
    
    def _fractal_strategy_hybrid(self, pil_img, correct_piece, mask, mask_np, x, y, w, h, 
                                width, height, base_x, base_y, base_size, 
                                max_depth, irregularity, smoothing_factor, difficulty):
        """Helper for fractal distractor strategy: combining multiple strategies."""
        # Choose 2-3 base strategies to combine
        base_strategies = ['different_seed', 'transform', 'boundary_modify', 'shift']
        num_strategies = 2 if difficulty == 'hard' else 3  # More strategies for easier difficulty
        
        strategies = random.sample(base_strategies, num_strategies)
        piece = correct_piece.copy()
        
        # Apply each strategy in sequence
        if 'different_seed' in strategies:
            piece = self._fractal_strategy_different_seed(
                pil_img, width, height, base_x, base_y, base_size,
                max_depth, irregularity, smoothing_factor
            )
        
        if 'transform' in strategies:
            piece = self._fractal_strategy_transform(piece, difficulty)
        
        if 'boundary_modify' in strategies:
            # Need to regenerate mask from the current piece
            if piece.mode != 'RGBA':
                # If not already RGBA, convert and add alpha channel
                piece = piece.convert('RGBA')
            
            # Extract alpha channel as mask
            piece_mask = piece.split()[3]
            
            # Apply boundary modifications
            piece = self._fractal_strategy_boundary_modify(
                pil_img, piece_mask, np.array(piece_mask), 0, 0, *piece.size, difficulty
            )
        
        if 'shift' in strategies:
            # This is trickier because we need mask information
            # Instead of actual shift, we'll apply a "similar" effect
            
            # Create a new image with the piece
            canvas = Image.new('RGBA', (width, height), (0, 0, 0, 0))
            canvas.paste(piece, (width//4, height//4), piece)
            
            # Find a new crop area
            new_x = random.randint(0, width//2)
            new_y = random.randint(0, height//2)
            crop_size = min(piece.size)
            
            # Crop a different area
            piece = canvas.crop((new_x, new_y, new_x + crop_size, new_y + crop_size))
        
        return piece
    
    ##############################
    # HELPER FUNCTIONS
    ##############################
    
    def _create_visualization(self, masked_img, options, correct_idx, output_dir, 
                             show_border=False, show_grid=False, grid_size=None, target_cell=None):
        """
        Create a visualization of the task for human viewing.
        
        Args:
            masked_img: Image with the cutout filled with gray
            options: List of option images
            correct_idx: Index of the correct option
            output_dir: Directory to save the visualization
            show_border: Whether to show border around pieces
            show_grid: Whether to show a grid overlay
            grid_size: Size of the grid (if show_grid is True)
            target_cell: Target cell coordinates (if show_grid is True)
            
        Returns:
            Path to the saved visualization
        """
        num_options = len(options)
        
        # Create figure with masked image and options
        plt.figure(figsize=(12, 8))
        
        # Display masked image on top
        plt.subplot(2, 1, 1)
        plt.imshow(masked_img)
        title = "Image with Cutout"
        if show_grid and grid_size and target_cell:
            col, row = target_cell
            title += f" (Target cell: {col},{row})"
        plt.title(title, fontsize=14)
        plt.axis('off')
        
        # Add grid overlay if requested
        if show_grid and grid_size:
            ax = plt.gca()
            width, height = masked_img.size
            cell_width = width / grid_size
            cell_height = height / grid_size
            
            # Draw grid lines
            for i in range(grid_size + 1):
                # Vertical lines
                x = i * cell_width
                ax.axvline(x=x, color='gray', linestyle='-', linewidth=1)
                
                # Horizontal lines
                y = i * cell_height
                ax.axhline(y=y, color='gray', linestyle='-', linewidth=1)
        
        # Display options on bottom
        for i, option in enumerate(options):
            plt.subplot(2, num_options, num_options + i + 1)
            plt.imshow(option)
            
            if i == correct_idx:
                plt.title(f"Option {chr(65+i)} (CORRECT)", color='green', fontsize=12)
            else:
                plt.title(f"Option {chr(65+i)}", fontsize=12)
            
            plt.axis('off')
            
            # Add border around the piece if requested
            if show_border:
                ax = plt.gca()
                ax.spines['top'].set_visible(True)
                ax.spines['right'].set_visible(True)
                ax.spines['bottom'].set_visible(True)
                ax.spines['left'].set_visible(True)
                ax.spines['top'].set_color('gray')
                ax.spines['right'].set_color('gray')
                ax.spines['bottom'].set_color('gray')
                ax.spines['left'].set_color('gray')
        
        plt.tight_layout()
        
        # Save the visualization
        vis_path = os.path.join(output_dir, "visualization.png")
        plt.savefig(vis_path, bbox_inches='tight')
        plt.close()
        
        return vis_path
    
    def generate_all_task_types(self, image_path, output_dir, num_options=4, difficulty='medium'):
        """
        Generate all three main types of cutout tasks for a single image.
        
        Args:
            image_path: Path to the source image
            output_dir: Directory to save all task files
            num_options: Number of piece options to generate
            difficulty: Difficulty level ('easy', 'medium', 'hard')
            
        Returns:
            Dictionary mapping task types to task information
        """
        os.makedirs(output_dir, exist_ok=True)
        
        tasks = {
            'jigsaw': self.generate_jigsaw_piece(image_path, output_dir, num_options, difficulty),
            'multi_piece': self.generate_multi_piece_puzzle(image_path, output_dir, num_options, difficulty),
            'fractal': self.generate_fractal_cutout(image_path, output_dir, num_options, difficulty)
        }
        
        return tasks
    
    def generate_dataset(self, image_dir, output_dir, num_tasks_per_image=1, 
                         task_types=None, difficulty_levels=None, num_options=4,
                         include_visualizations=True):
        """
        Generate a dataset of tasks from multiple images.
        
        Args:
            image_dir: Directory containing source images
            output_dir: Directory to save all task files
            num_tasks_per_image: Number of tasks to generate per image
            task_types: List of task types to generate (default: all)
            difficulty_levels: List of difficulty levels (default: all)
            num_options: Number of piece options per task
            include_visualizations: Whether to include visualizations
            
        Returns:
            Dictionary with dataset metadata
        """
        if task_types is None:
            task_types = ['jigsaw', 'multi_piece', 'fractal']
        
        if difficulty_levels is None:
            difficulty_levels = ['easy', 'medium', 'hard']
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Get list of images
        image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        dataset = {
            'num_images': len(image_files),
            'num_tasks': len(image_files) * num_tasks_per_image,
            'task_types': task_types,
            'difficulty_levels': difficulty_levels,
            'tasks': []
        }
        
        # Generate tasks for each image
        for img_idx, img_file in enumerate(image_files):
            img_path = os.path.join(image_dir, img_file)
            img_name = os.path.splitext(img_file)[0]
            
            print(f"Processing image {img_idx+1}/{len(image_files)}: {img_file}")
            
            for i in range(num_tasks_per_image):
                # Select a random task type and difficulty
                task_type = random.choice(task_types)
                difficulty = random.choice(difficulty_levels)
                
                try:
                    # Create a subdirectory for this task
                    task_subdir = os.path.join(output_dir, f"{img_name}_{task_type}_{difficulty}_{i}")
                    os.makedirs(task_subdir, exist_ok=True)
                    
                    # Generate the task based on type
                    if task_type == 'jigsaw':
                        task = self.generate_jigsaw_piece(img_path, task_subdir, num_options, difficulty)
                    elif task_type == 'multi_piece':
                        task = self.generate_multi_piece_puzzle(img_path, task_subdir, num_options, difficulty)
                    elif task_type == 'fractal':
                        task = self.generate_fractal_cutout(img_path, task_subdir, num_options, difficulty)
                    else:
                        continue
                    
                    # Add to dataset
                    task_entry = {
                        'image_file': img_file,
                        'task_type': task_type,
                        'difficulty': difficulty,
                        'task_dir': os.path.relpath(task['task_dir'], output_dir),
                        'masked_image': os.path.relpath(task['masked_image'], output_dir),
                        'options': [os.path.relpath(opt, output_dir) for opt in task['options']],
                        'correct_idx': task['correct_idx'],
                        'correct_option': chr(65 + task['correct_idx'])
                    }
                    
                    # If multi-piece task, add highlighted target
                    if task_type == 'multi_piece' and 'highlighted_target' in task:
                        task_entry['highlighted_target'] = os.path.relpath(task['highlighted_target'], output_dir)
                    
                    dataset['tasks'].append(task_entry)
                    
                    # Remove visualization files if not requested
                    if not include_visualizations:
                        vis_path = os.path.join(task['task_dir'], "visualization.png")
                        if os.path.exists(vis_path):
                            os.remove(vis_path)
                    
                    print(f"  Created {task_type} task (difficulty: {difficulty})")
                
                except Exception as e:
                    print(f"  Error generating {task_type} task: {str(e)}")
                    continue
        
        # Save dataset metadata
        dataset_path = os.path.join(output_dir, "dataset.json")
        with open(dataset_path, 'w') as f:
            json.dump(dataset, f, indent=2)
        
        print(f"Dataset generated with {len(dataset['tasks'])} tasks")
        print(f"Metadata saved to {dataset_path}")
        
        return dataset, dataset_path


# Example usage
if __name__ == "__main__":
    # Create the comprehensive puzzle generator
    generator = ComprehensivePuzzleGenerator(seed=42)
    
    # Set paths
    image_path = "dogs.jpg"  # Replace with your image path
    output_dir = "output_hard_tasks"
        
    # Generate a jigsaw puzzle piece task
    jigsaw_task = generator.generate_jigsaw_piece(
        image_path=image_path,
        output_dir="output_hard_tasks",
        num_options=4,
        difficulty="hard",
        custom_params={"tab_count_range": (2, 3), "tab_size_factor": 0.2}
    )

    # Generate a dataset from a directory of images
    dataset, dataset_path = generator.generate_dataset(
        image_dir=image_path, 
        output_dir="dataset_output",
        num_tasks_per_image=2,
        task_types=['jigsaw', 'multi_piece', 'fractal'],
        difficulty_levels=['easy', 'medium', 'hard']
    )
    