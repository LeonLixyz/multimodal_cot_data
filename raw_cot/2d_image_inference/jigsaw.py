"""
Jigsaw puzzle generator.

This module provides functionality to generate jigsaw puzzle pieces
with tabs and blanks along the edges.
"""

import os
import json
import random
import math
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageEnhance, ImageOps
from typing import Dict, List, Any, Optional, Tuple, Union

from utils import load_image, add_noise_to_mask, smooth_mask, create_visualization


class JigsawPuzzleGenerator:
    """
    Generator for jigsaw puzzle pieces with tabs and blanks.
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the jigsaw puzzle generator.
        
        Args:
            seed: Optional random seed for reproducibility
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Default parameters for different difficulty levels
        self.params = {
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
        
        # Available distractor strategies
        self.distractor_strategies = [
            'flip_tabs',     # Invert tabs and blanks
            'rotate',        # Rotate the piece
            'shift',         # Take a piece from elsewhere
            'modify_tabs',   # Change tab shapes
            'color_shift'    # Modify colors
        ]
    
    def generate(self, 
                image_path: str, 
                output_dir: str, 
                num_options: int = 4, 
                difficulty: str = 'medium',
                num_pieces: int = 1,
                custom_params: Optional[Dict] = None,
                distractor_types: Optional[List[str]] = None) -> Dict:
        """
        Generate a jigsaw puzzle piece task.
        
        Args:
            image_path: Path to the source image
            output_dir: Directory to save the task files
            num_options: Number of piece options to generate (default: 4)
            difficulty: Difficulty level ('easy', 'medium', 'hard')
            num_pieces: Number of jigsaw pieces to cut out (default: 1)
            custom_params: Optional dict to override default parameters
            distractor_types: List of distractor strategies to use (randomly chosen if None)
            
        Returns:
            Dictionary with task information
        """
        # Create output directory
        task_dir = os.path.join(output_dir, f"jigsaw_piece_task_{difficulty}")
        os.makedirs(task_dir, exist_ok=True)
        
        # Get parameters based on difficulty level
        params = self.params.get(difficulty, self.params['medium']).copy()
        
        # Override with custom parameters if provided
        if custom_params:
            params.update(custom_params)
        
        # Load the image
        pil_img, cv_img = load_image(image_path)
        width, height = pil_img.size
        
        # Create the masked image (we'll fill in the cutouts later)
        masked_pil = pil_img.copy()
        
        # Lists to track information about pieces
        all_correct_pieces = []
        all_cutout_info = []
        all_jigsaw_features = []
        
        # Generate multiple jigsaw pieces
        for piece_idx in range(num_pieces):
            # Define the base rectangular cutout parameters
            # For multiple pieces, make them smaller to fit in the image
            size_factor = 0.3 if num_pieces == 1 else 0.2
            cutout_width = int(width * random.uniform(0.15, size_factor))
            cutout_height = int(height * random.uniform(0.15, size_factor))
            
            # Randomly position the cutout (not too close to the edges)
            margin = 50  # Larger margin to allow for jigsaw tabs
            
            # Make sure pieces don't overlap
            attempt = 0
            max_attempts = 50
            while attempt < max_attempts:
                left = random.randint(margin, width - cutout_width - margin)
                top = random.randint(margin, height - cutout_height - margin)
                
                # Check for overlap with existing pieces
                overlap = False
                for info in all_cutout_info:
                    prev_left, prev_top, prev_width, prev_height = info["base_rectangle"].values()
                    # Add some extra margin to prevent pieces from being too close
                    if (left < prev_left + prev_width + margin and 
                        left + cutout_width + margin > prev_left and
                        top < prev_top + prev_height + margin and
                        top + cutout_height + margin > prev_top):
                        overlap = True
                        break
                
                if not overlap:
                    break
                    
                attempt += 1
                
            # If we couldn't find a non-overlapping position, try with a smaller piece
            if attempt >= max_attempts:
                cutout_width = int(cutout_width * 0.8)
                cutout_height = int(cutout_height * 0.8)
                left = random.randint(margin, width - cutout_width - margin)
                top = random.randint(margin, height - cutout_height - margin)
            
            # Create a mask for this jigsaw piece
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
                mask = add_noise_to_mask(mask, noise_level=0.1)
                mask = smooth_mask(mask, smoothing_factor=0.5)
            elif shape_complexity == 'complex':
                mask = add_noise_to_mask(mask, noise_level=0.2)
                mask = smooth_mask(mask, smoothing_factor=0.3)
            
            # Convert to numpy array for OpenCV operations
            mask_np = np.array(mask)
            
            # Get the bounding box of the jigsaw piece
            contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            x, y, w, h = cv2.boundingRect(contours[0])
            
            # Fill in the cutout in the masked image
            masked_np = np.array(masked_pil)
            masked_np[mask_np > 0] = [128, 128, 128]
            masked_pil = Image.fromarray(masked_np)
            
            # Extract the correct cutout piece with transparency
            img_rgba = Image.new("RGBA", pil_img.size)
            img_rgba.paste(pil_img)
            
            # Create alpha channel from mask
            img_rgba.putalpha(mask)
            
            # Crop to the bounding box
            correct_piece = img_rgba.crop((x, y, x + w, y + h))
            
            # Add to our lists
            all_correct_pieces.append(correct_piece)
            all_jigsaw_features.append(jigsaw_features)
            all_cutout_info.append({
                "base_rectangle": {
                    "left": left,
                    "top": top,
                    "width": cutout_width,
                    "height": cutout_height
                },
                "cutout_bounds": {
                    "x": x,
                    "y": y,
                    "width": w,
                    "height": h
                }
            })
        
        # Save the masked image
        masked_img_path = os.path.join(task_dir, "masked_image.jpg")
        masked_pil.save(masked_img_path)
        
        # Generate the options
        # For each correct piece, decide where it will appear in the options
        correct_indices = []
        options = []
        
        # First, distribute all correct pieces
        for correct_piece in all_correct_pieces:
            # If num_options is less than the number of correct pieces * 2, 
            # increase it to ensure enough distractor options
            if num_options < num_pieces * 2:
                num_options = num_pieces * 2
                
            valid_indices = [i for i in range(num_options) if i not in correct_indices]
            if valid_indices:  # Check if there are still available indices
                correct_idx = random.choice(valid_indices)
                correct_indices.append(correct_idx)
        
        # Fill options list with Nones to be filled later
        options = [None] * num_options
        
        # Place correct pieces in their spots
        for i, correct_idx in enumerate(correct_indices):
            options[correct_idx] = all_correct_pieces[i]
        
        # Generate distractors for the remaining spots
        # Select distractor strategies if not specified
        if not distractor_types:
            num_strategies = min(num_options - num_pieces, len(self.distractor_strategies))
            distractor_types = random.sample(self.distractor_strategies, num_strategies)
        
        # Generate distractors
        distractor_count = 0
        for i in range(num_options):
            if i not in correct_indices:
                # This is a distractor slot
                # Pick a correct piece to base the distractor on
                correct_piece_idx = distractor_count % num_pieces
                correct_piece = all_correct_pieces[correct_piece_idx]
                cutout_info = all_cutout_info[correct_piece_idx]
                jigsaw_features = all_jigsaw_features[correct_piece_idx]
                
                # Pick a distractor strategy - MODIFIED: ensure each distractor uses a different strategy
                strategy_idx = distractor_count % len(distractor_types)
                strategy = distractor_types[strategy_idx]
                
                # Generate the distractor
                if strategy == 'flip_tabs' and jigsaw_features:
                    # Strategy 1: Flip tabs and blanks
                    piece = self._jigsaw_strategy_flip_tabs(
                        pil_img, jigsaw_features, 
                        cutout_info["base_rectangle"]["left"], 
                        cutout_info["base_rectangle"]["top"], 
                        cutout_info["base_rectangle"]["width"], 
                        cutout_info["base_rectangle"]["height"], 
                        int(cutout_info["base_rectangle"]["width"] * params['tab_size_factor']),
                        int(cutout_info["base_rectangle"]["height"] * params['tab_size_factor']),
                        shape_complexity, width, height
                    )
                    # ADDED: Apply an additional transform to ensure uniqueness
                    if random.choice([True, False]):
                        piece = piece.transpose(Image.FLIP_LEFT_RIGHT)
                
                elif strategy == 'rotate':
                    # Strategy 2: Rotate the piece
                    rotation_angle = random.choice([90, 180, 270])
                    piece = correct_piece.rotate(rotation_angle, expand=True)
                
                elif strategy == 'shift':
                    # Strategy 3: Take a piece from elsewhere
                    piece = self._jigsaw_strategy_shift(
                        pil_img, 
                        cutout_info["cutout_bounds"]["x"], 
                        cutout_info["cutout_bounds"]["y"], 
                        cutout_info["cutout_bounds"]["width"], 
                        cutout_info["cutout_bounds"]["height"], 
                        width, height, margin
                    )
                
                elif strategy == 'modify_tabs':
                    # Strategy 4: Modify tab shapes
                    piece = self._jigsaw_strategy_modify_tabs(
                        pil_img, jigsaw_features, 
                        cutout_info["base_rectangle"]["left"], 
                        cutout_info["base_rectangle"]["top"], 
                        cutout_info["base_rectangle"]["width"], 
                        cutout_info["base_rectangle"]["height"], 
                        int(cutout_info["base_rectangle"]["width"] * params['tab_size_factor']),
                        int(cutout_info["base_rectangle"]["height"] * params['tab_size_factor']),
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
                    x = cutout_info["cutout_bounds"]["x"]
                    y = cutout_info["cutout_bounds"]["y"]
                    w = cutout_info["cutout_bounds"]["width"]
                    h = cutout_info["cutout_bounds"]["height"]
                    
                    fake_x = random.randint(margin, width - w - margin)
                    fake_y = random.randint(margin, height - h - margin)
                    
                    # Make sure it's far enough from the original
                    while abs(fake_x - x) < w/2 and abs(fake_y - y) < h/2:
                        fake_x = random.randint(margin, width - w - margin)
                        fake_y = random.randint(margin, height - h - margin)
                    
                    # Create a shape similar to the original but at a different position
                    mask_copy = Image.new("L", (width, height), 0)
                    mask_copy.paste(Image.fromarray(mask_np).crop((x, y, x+w, y+h)), (fake_x, fake_y))
                    
                    # Apply to the image
                    fake_rgba = Image.new("RGBA", pil_img.size)
                    fake_rgba.paste(pil_img)
                    fake_rgba.putalpha(mask_copy)
                    
                    piece = fake_rgba.crop((fake_x, fake_y, fake_x + w, fake_y + h))
                
                options[i] = piece
                distractor_count += 1
        
        # Save all option pieces
        option_paths = []
        for i, option in enumerate(options):
            piece_path = os.path.join(task_dir, f"option_{chr(65+i)}.png")
            option.save(piece_path)
            option_paths.append(piece_path)
        
        # Save the original image for reference
        original_path = os.path.join(task_dir, "original.jpg")
        pil_img.save(original_path)
        
        # Create a visualization of the task
        vis_path = create_visualization(
            masked_pil, options, correct_indices, task_dir,
            title=f"Jigsaw Puzzle ({num_pieces} piece{'s' if num_pieces > 1 else ''})", 
            show_border=True
        )
        
        # Save task metadata
        metadata = {
            "task_type": "jigsaw_piece",
            "difficulty": difficulty,
            "num_pieces": num_pieces,
            "original_image": os.path.basename(original_path),
            "masked_image": os.path.basename(masked_img_path),
            "options": [os.path.basename(path) for path in option_paths],
            "correct_indices": correct_indices,
            "correct_options": [chr(65 + idx) for idx in correct_indices],
            "cutout_info": all_cutout_info,
            "jigsaw_features": all_jigsaw_features,
            "parameters": params
        }
        
        if distractor_types:
            # Create a mapping of which strategy was used for each distractor
            strategy_mapping = []
            for i in range(num_options):
                if i in correct_indices:
                    strategy_mapping.append("correct")
                else:
                    # Find the index of this distractor
                    distractor_idx = [j for j in range(num_options) if j not in correct_indices].index(i)
                    strategy_idx = distractor_idx % len(distractor_types)
                    strategy_mapping.append(distractor_types[strategy_idx])
            
            metadata["distractor_strategies"] = strategy_mapping
        
        metadata_path = os.path.join(task_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        return {
            "task_dir": task_dir,
            "masked_image": masked_img_path,
            "options": option_paths,
            "correct_indices": correct_indices,
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
            fake_mask = add_noise_to_mask(fake_mask, noise_level=0.1)
            fake_mask = smooth_mask(fake_mask, smoothing_factor=0.5)
        elif shape_complexity == 'complex':
            fake_mask = add_noise_to_mask(fake_mask, noise_level=0.2)
            fake_mask = smooth_mask(fake_mask, smoothing_factor=0.3)
        
        # Get bounding box
        fake_mask_np = np.array(fake_mask)
        fake_contours, _ = cv2.findContours(fake_mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        fake_x, fake_y, fake_w, fake_h = cv2.boundingRect(fake_contours[0])
        
        # Apply mask to create the piece
        fake_rgba = Image.new("RGBA", pil_img.size)
        fake_rgba.paste(pil_img)
        fake_rgba.putalpha(fake_mask)
        
        return fake_rgba.crop((fake_x, fake_y, fake_x + fake_w, fake_y + fake_h))
    
    def _jigsaw_strategy_shift(self, pil_img, x, y, w, h, width, height, margin):
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
        # For this to work properly, we need the mask around x, y with size w, h
        # But we don't have direct access to it here, so we create a new mask based on the shape
        
        # Create a rectangular mask at the new position
        shifted_mask = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(shifted_mask)
        draw.rectangle((fake_x, fake_y, fake_x + w, fake_y + h), fill=255)
        
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
            fake_mask = add_noise_to_mask(fake_mask, noise_level=0.15)
            fake_mask = smooth_mask(fake_mask, smoothing_factor=0.4)
        elif new_complexity == 'complex':
            fake_mask = add_noise_to_mask(fake_mask, noise_level=0.25)
            fake_mask = smooth_mask(fake_mask, smoothing_factor=0.2)
        
        # Get bounding box
        fake_mask_np = np.array(fake_mask)
        fake_contours, _ = cv2.findContours(fake_mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        fake_x, fake_y, fake_w, fake_h = cv2.boundingRect(fake_contours[0])
        
        # Apply mask to create the piece
        fake_rgba = Image.new("RGBA", pil_img.size)
        fake_rgba.paste(pil_img)
        fake_rgba.putalpha(fake_mask)
        
        return fake_rgba.crop((fake_x, fake_y, fake_x + fake_w, fake_y + fake_h))