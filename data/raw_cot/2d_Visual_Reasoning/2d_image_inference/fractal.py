"""
Fractal cutout generator.

This module provides functionality to generate fractal-like
cutouts with irregular boundaries.
"""

import os
import json
import random
import math
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageEnhance, ImageOps
from typing import Dict, List, Any, Optional, Tuple, Union

from utils import load_image, smooth_mask, create_visualization


class FractalCutoutGenerator:
    """
    Generator for fractal-like cutouts with irregular boundaries.
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the fractal cutout generator.
        
        Args:
            seed: Optional random seed for reproducibility
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Default parameters for different difficulty levels
        self.params = {
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
        
        # Available distractor strategies
        self.distractor_strategies = [
            'different_seed',  # Generate with different parameters
            'transform',       # Apply transformations
            'boundary_modify', # Modify the boundary
            'shift',           # Take from elsewhere
            'hybrid'           # Combine strategies
        ]
        
        # Available shape types
        self.shape_types = [
            'fractal',      # Classic fractal with recursive subdivision
            'corner_cut',   # Rounded corner cut
            'wave',         # Wavy edge pattern
            'zigzag',       # Zigzag pattern
            'star'          # Star-like shape
        ]
    
    def generate(self, 
                image_path: str, 
                output_dir: str, 
                num_options: int = 4, 
                difficulty: str = 'medium',
                num_pieces: int = 1,
                shape_types: Optional[List[str]] = None,
                custom_params: Optional[Dict] = None,
                distractor_types: Optional[List[str]] = None) -> Dict:
        """
        Generate a fractal cutout task.
        
        Args:
            image_path: Path to the source image
            output_dir: Directory to save the task files
            num_options: Number of piece options to generate (default: 4)
            difficulty: Difficulty level ('easy', 'medium', 'hard')
            num_pieces: Number of pieces to cut out (default: 1)
            shape_types: List of shape types to use (randomly chosen if None)
            custom_params: Optional dict to override default parameters
            distractor_types: List of distractor strategies to use (randomly chosen if None)
            
        Returns:
            Dictionary with task information
        """
        # Create output directory
        task_dir = os.path.join(output_dir, f"fractal_cutout_task_{difficulty}")
        os.makedirs(task_dir, exist_ok=True)
        
        # Get parameters based on difficulty level
        params = self.params.get(difficulty, self.params['medium']).copy()
        
        # Override with custom parameters if provided
        if custom_params:
            params.update(custom_params)
        
        # MODIFIED: If no shape types specified and only one piece, use 'fractal'
        # For demo purposes, if there are multiple pieces requested, use different shape types
        if not shape_types:
            if num_pieces == 1:
                shape_types = ['fractal']
            else:
                # Use as many different shape types as pieces, up to 5
                shape_types = self.shape_types[:num_pieces]
        
        # ADDED: Save individual demos of each shape type if requested
        if num_pieces == 1 and 'all_types' in (shape_types or []):
            # Generate a demo of each shape type
            all_results = {}
            for shape_type in self.shape_types:
                shape_dir = os.path.join(output_dir, f"fractal_{shape_type}")
                os.makedirs(shape_dir, exist_ok=True)
                
                print(f"Generating fractal cutout with shape type '{shape_type}'...")
                result = self.generate(
                    image_path=image_path,
                    output_dir=shape_dir,
                    num_options=num_options,
                    difficulty=difficulty,
                    num_pieces=1,
                    shape_types=[shape_type],
                    custom_params=custom_params,
                    distractor_types=distractor_types
                )
                all_results[shape_type] = result
            
            # Now return the original fractal shape type
            shape_types = ['fractal']
        
        # If fewer shape types than pieces, repeat some types
        while len(shape_types) < num_pieces:
            shape_types.append(random.choice(shape_types))
        
        # Load the image
        pil_img, cv_img = load_image(image_path)
        width, height = pil_img.size
        
        # Create a masked image (we'll fill in the cutouts as we go)
        masked_pil = pil_img.copy()
        
        # Lists to track information about pieces
        all_correct_pieces = []
        all_masks = []
        all_cutout_info = []
        all_shape_types = []
        
        # Generate multiple fractal-like cutouts
        for piece_idx in range(num_pieces):
            # Select a shape type for this piece
            shape_type = shape_types[piece_idx % len(shape_types)]
            all_shape_types.append(shape_type)
            
            # Create a base shape for the cutout
            # For multiple pieces, make them smaller to fit in the image
            size_factor = 0.25 if num_pieces == 1 else 0.15
            base_size = min(width, height) * size_factor
            
            # Try to find a position that doesn't overlap with existing pieces
            attempt = 0
            max_attempts = 50
            while attempt < max_attempts:
                base_x = random.randint(width // 4, 3 * width // 4)
                base_y = random.randint(height // 4, 3 * height // 4)
                
                # Check for overlap with existing pieces
                overlap = False
                for info in all_cutout_info:
                    prev_x, prev_y = info["base_params"]["x"], info["base_params"]["y"]
                    prev_size = info["base_params"]["size"]
                    
                    # Check if centers are too close
                    distance = math.sqrt((base_x - prev_x)**2 + (base_y - prev_y)**2)
                    if distance < (base_size + prev_size) * 0.8:  # 0.8 factor to avoid too close pieces
                        overlap = True
                        break
                
                if not overlap:
                    break
                    
                attempt += 1
                
            # If we couldn't find a non-overlapping position, use a smaller size
            base_size = int(min(width, height) * size_factor)

            # And later in the attempt loop, ensure integer casts:
            if attempt >= max_attempts:
                base_size = int(base_size * 0.8)
            
            # Create a mask for the shape
            mask = Image.new("L", (width, height), 0)
            draw = ImageDraw.Draw(mask)
            
            # Generate the shape based on its type
            if shape_type == 'fractal':
                # Set parameters for fractal generation
                max_depth = params['complexity']
                irregularity = params['irregularity']
                
                # Draw fractal shape using recursive subdivision
                self._draw_fractal_shape(draw, base_x, base_y, base_size, max_depth, irregularity)
                
            elif shape_type == 'corner_cut':
                # Draw a shape with rounded corner cuts
                self._draw_corner_cut_shape(draw, base_x, base_y, base_size, params)
                
            elif shape_type == 'wave':
                # Draw a shape with wavy edges
                self._draw_wave_shape(draw, base_x, base_y, base_size, params)
                
            elif shape_type == 'zigzag':
                # Draw a zigzag shape
                self._draw_zigzag_shape(draw, base_x, base_y, base_size, params)
                
            elif shape_type == 'star':
                # Draw a star-like shape
                self._draw_star_shape(draw, base_x, base_y, base_size, params)
                
            else:
                # Default to simple ellipse
                draw.ellipse((base_x - base_size//2, base_y - base_size//2, 
                              base_x + base_size//2, base_y + base_size//2), fill=255)
            
            # Apply smoothing based on the parameter
            smoothing_factor = params['smoothing']
            mask = smooth_mask(mask, smoothing_factor)
            
            # Convert mask to numpy array for OpenCV operations
            mask_np = np.array(mask)
            
            # Find the bounding box of the shape
            contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                # Fallback if no contours were found
                print(f"No contours found for {shape_type} shape, using simple ellipse instead")
                # Create a simple elliptical shape
                draw.ellipse((base_x - base_size, base_y - base_size, 
                              base_x + base_size, base_y + base_size), fill=255)
                mask_np = np.array(mask)
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
            
            # Crop to bounding box
            correct_piece = img_rgba.crop((x, y, x+w, y+h))
            
            # Add to our lists
            all_correct_pieces.append(correct_piece)
            all_masks.append(mask)
            all_cutout_info.append({
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
        
        # Select distractor strategies if not specified
        if not distractor_types:
            num_strategies = min(num_options - num_pieces, len(self.distractor_strategies))
            distractor_types = random.sample(self.distractor_strategies, num_strategies)
        
        # Generate distractors for the remaining spots
        distractor_count = 0
        for i in range(num_options):
            if i not in correct_indices:
                # This is a distractor slot
                # Pick a correct piece to base the distractor on
                correct_piece_idx = distractor_count % num_pieces
                correct_piece = all_correct_pieces[correct_piece_idx]
                cutout_info = all_cutout_info[correct_piece_idx]
                mask = all_masks[correct_piece_idx]
                shape_type = all_shape_types[correct_piece_idx]
                
                # Pick a distractor strategy
                strategy_idx = distractor_count % len(distractor_types)
                strategy = distractor_types[strategy_idx]
                
                # Extract parameters from cutout_info
                base_x = cutout_info["base_params"]["x"]
                base_y = cutout_info["base_params"]["y"]
                base_size = cutout_info["base_params"]["size"]
                x = cutout_info["cutout_bounds"]["x"]
                y = cutout_info["cutout_bounds"]["y"]
                w = cutout_info["cutout_bounds"]["width"]
                h = cutout_info["cutout_bounds"]["height"]
                
                # Generate the distractor
                if strategy == 'different_seed':
                    # Strategy 1: Generate a different shape/pattern
                    piece = self._fractal_strategy_different_seed(
                        pil_img, width, height, base_x, base_y, base_size,
                        params['complexity'], params['irregularity'], params['smoothing'],
                        shape_type
                    )
                
                elif strategy == 'transform':
                    # Strategy 2: Apply transformations to the correct piece
                    piece = self._fractal_strategy_transform(
                        correct_piece, difficulty
                    )
                
                elif strategy == 'boundary_modify':
                    # Strategy 3: Modify the boundary of the shape
                    piece = self._fractal_strategy_boundary_modify(
                        pil_img, mask, np.array(mask), x, y, w, h, difficulty
                    )
                
                elif strategy == 'shift':
                    # Strategy 4: Take from a different area with the same mask
                    piece = self._fractal_strategy_shift(
                        pil_img, mask, np.array(mask), x, y, w, h, width, height
                    )
                
                elif strategy == 'hybrid':
                    # Strategy 5: Combine multiple strategies
                    piece = self._fractal_strategy_hybrid(
                        pil_img, correct_piece, mask, np.array(mask), x, y, w, h, 
                        width, height, base_x, base_y, base_size, 
                        params['complexity'], params['irregularity'], params['smoothing'], 
                        difficulty, shape_type
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
            title=f"Fractal Cutout ({num_pieces} piece{'s' if num_pieces > 1 else ''})", 
            show_border=True
        )
        
        # Save task metadata
        metadata = {
            "task_type": "fractal_cutout",
            "difficulty": difficulty,
            "num_pieces": num_pieces,
            "shape_types": all_shape_types,
            "original_image": os.path.basename(original_path),
            "masked_image": os.path.basename(masked_img_path),
            "options": [os.path.basename(path) for path in option_paths],
            "correct_indices": correct_indices,
            "correct_options": [chr(65 + idx) for idx in correct_indices],
            "cutout_info": all_cutout_info,
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
    
    def _draw_fractal_shape(self, draw, x, y, size, max_depth, irregularity, depth=0):
        """Draw a fractal-like shape using recursive subdivision."""
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
            self._draw_fractal_shape(draw, child_x, child_y, child_size, max_depth, irregularity, depth + 1)
    
    def _draw_corner_cut_shape(self, draw, x, y, size, params):
        """Draw a shape with rounded corner cuts."""
        # Draw a base square
        half_size = int(size // 2)
        left, top = x - half_size, y - half_size
        right, bottom = x + half_size, y + half_size
        # Create points for polygon (with potential irregularity)
        irregularity = params['irregularity']
        
        # Corner points (clockwise from top-left)
        corners = [
            (left, top),
            (right, top),
            (right, bottom),
            (left, bottom)
        ]
        
        # Cut size determines how big the corner cuts are
        cut_size = int(size // 3)
        
        # Create points for the polygon with corner cuts
        points = []
        
        # For each corner, add points for the rounded cut
        for i in range(4):
            corner = corners[i]
            next_corner = corners[(i + 1) % 4]
            
            # Determine the cut center
            cut_x = corner[0]
            cut_y = corner[1]
            
            # Add points for the rounded cut
            num_cut_points = 5  # Number of points to use for each corner cut
            
            # Determine direction for the cut (depends on which corner)
            if i == 0:  # Top-left
                cut_start_angle = math.pi
                cut_end_angle = 3 * math.pi / 2
            elif i == 1:  # Top-right
                cut_start_angle = 3 * math.pi / 2
                cut_end_angle = 2 * math.pi
            elif i == 2:  # Bottom-right
                cut_start_angle = 0
                cut_end_angle = math.pi / 2
            else:  # Bottom-left
                cut_start_angle = math.pi / 2
                cut_end_angle = math.pi
            
            # Add points along the cut
            for j in range(num_cut_points):
                # Calculate angle
                angle = cut_start_angle + (cut_end_angle - cut_start_angle) * j / (num_cut_points - 1)
                
                # Add irregularity to the angle
                angle += random.uniform(-irregularity, irregularity) * 0.1
                
                # Calculate point on the cut
                cut_point_x = int(cut_x + cut_size * math.cos(angle))
                cut_point_y = int(cut_y + cut_size * math.sin(angle))
                
                points.append((cut_point_x, cut_point_y))
            
            # Add a point part way to the next corner (to create the straight edges)
            edge_x = int(next_corner[0] - (next_corner[0] - corner[0]) * (cut_size / half_size) / 2)
            edge_y = int(next_corner[1] - (next_corner[1] - corner[1]) * (cut_size / half_size) / 2)
            
            # Add some irregularity
            edge_x += int(random.uniform(-irregularity, irregularity) * size * 0.05)
            edge_y += int(random.uniform(-irregularity, irregularity) * size * 0.05)
            
            points.append((edge_x, edge_y))
        
        # Draw the polygon
        draw.polygon(points, fill=255)
    
    def _draw_wave_shape(self, draw, x, y, size, params):
        """Draw a shape with wavy edges."""
        # Base parameters
        half_size = int(size // 2)
        irregularity = params['irregularity']
        
        # Wave parameters
        num_waves = random.randint(6, 12)  # Number of wave peaks
        wave_amplitude = size * 0.15  # Height of wave peaks
        
        # Create points for the wavy circle
        points = []
        for i in range(num_waves * 3):  # Use more points for smoother waves
            # Base angle
            angle = 2 * math.pi * i / (num_waves * 3)
            
            # Wave pattern - sine wave with frequency = num_waves
            wave = math.sin(angle * num_waves) * wave_amplitude
            
            # Add irregularity
            wave += random.uniform(-irregularity, irregularity) * wave_amplitude * 0.5
            
            # Calculate radius with wave
            radius = half_size + wave
            
            # Calculate coordinates
            px = int(x + radius * math.cos(angle))
            py = int(y + radius * math.sin(angle))
            
            points.append((px, py))
        
        # Draw the polygon
        draw.polygon(points, fill=255)
    
    def _draw_zigzag_shape(self, draw, x, y, size, params):
        """Draw a shape with zigzag edges."""
        # Base parameters
        half_size = int(size // 2)
        irregularity = params['irregularity']
        
        # Zigzag parameters
        num_zigs = random.randint(8, 16)  # Number of zigzag points
        zig_amplitude = size * 0.2  # Depth of zigzags
        
        # Create points for the zigzag circle
        points = []
        for i in range(num_zigs):
            # Two points per zigzag segment
            for j in range(2):
                # Base angle
                angle = 2 * math.pi * (i + j * 0.5) / num_zigs
                
                # Zigzag pattern
                zig = (j % 2) * 2 - 1  # Alternates between -1 and 1
                zig *= zig_amplitude
                
                # Add irregularity
                zig += random.uniform(-irregularity, irregularity) * zig_amplitude * 0.3
                
                # Calculate radius with zigzag
                radius = half_size + zig
                
                # Calculate coordinates
                px = int(x + radius * math.cos(angle))
                py = int(y + radius * math.sin(angle))
                
                points.append((px, py))
        
        # Draw the polygon
        draw.polygon(points, fill=255)
    
    def _draw_star_shape(self, draw, x, y, size, params):
        """Draw a star-like shape."""
        # Base parameters
        half_size = int(size // 2)
        irregularity = params['irregularity']
        
        # Star parameters
        num_points = random.randint(5, 8)  # Number of star points
        inner_radius = half_size * 0.4  # Inner radius for star points
        
        # Create points for the star
        points = []
        for i in range(num_points * 2):
            # Alternate between outer and inner points
            is_outer = i % 2 == 0
            
            # Base angle
            angle = 2 * math.pi * i / (num_points * 2)
            
            # Add irregularity to angle
            angle += random.uniform(-irregularity, irregularity) * 0.1
            
            # Use outer or inner radius
            radius = half_size if is_outer else inner_radius
            
            # Add irregularity to radius
            radius_variation = random.uniform(-irregularity, irregularity) * half_size * 0.1
            radius += radius_variation
            
            # Calculate coordinates
            px = int(x + radius * math.cos(angle))
            py = int(y + radius * math.sin(angle))
            
            points.append((px, py))
        
        # Draw the polygon
        draw.polygon(points, fill=255)
    
    def _fractal_strategy_different_seed(self, pil_img, width, height, base_x, base_y, 
                                      base_size, max_depth, irregularity, smoothing_factor,
                                      shape_type):
        """Helper for fractal distractor strategy: different shape/pattern seed."""
        # Create a new shape with different parameters
        fake_mask = Image.new("L", (width, height), 0)
        fake_draw = ImageDraw.Draw(fake_mask)
        
        # Slightly vary the parameters
        # Ensure integers are used for randint parameters
        base_size_int = int(base_size)
        fake_x = base_x + random.randint(-base_size_int//3, base_size_int//3)
        fake_y = base_y + random.randint(-base_size_int//3, base_size_int//3)
        fake_size = base_size * random.uniform(0.8, 1.2)
        fake_max_depth = max_depth
        fake_irregularity = irregularity * random.uniform(0.8, 1.2)
        
        # Use a different shape type if possible
        other_types = [t for t in self.shape_types if t != shape_type]
        if other_types:
            fake_shape_type = random.choice(other_types)
        else:
            fake_shape_type = shape_type
        
        # Generate the shape based on its type
        if fake_shape_type == 'fractal':
            # Function to draw a different fractal pattern
            def draw_different_fractal(x, y, size, depth=0):
                # Ensure size is an integer for comparison
                size_int = int(size)
                if depth >= fake_max_depth or size_int < 5:
                    # At leaf nodes, draw a different shape (rectangle instead of ellipse)
                    half_size = int(size // 2)
                    fake_draw.rectangle((int(x - half_size), int(y - half_size), 
                                    int(x + half_size), int(y + half_size)), fill=255)
                    return
                
                # Main shape - use an ellipse or rectangle randomly
                if random.choice([True, False]):
                    fake_draw.ellipse((int(x - size//2), int(y - size//2), 
                                    int(x + size//2), int(y + size//2)), fill=255)
                else:
                    fake_draw.rectangle((int(x - size//2), int(y - size//2), 
                                        int(x + size//2), int(y + size//2)), fill=255)
                
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
                    child_size = size * random.uniform(0.25, 0.55)
                    
                    draw_different_fractal(child_x, child_y, child_size, depth + 1)
            
            # Draw the fractal
            draw_different_fractal(fake_x, fake_y, fake_size)
            
        elif fake_shape_type == 'corner_cut':
            # Draw a shape with different corner cuts
            self._draw_corner_cut_shape(fake_draw, fake_x, fake_y, int(fake_size), 
                                    {'irregularity': fake_irregularity})
            
        elif fake_shape_type == 'wave':
            # Draw a shape with different wavy edges
            self._draw_wave_shape(fake_draw, fake_x, fake_y, int(fake_size), 
                                {'irregularity': fake_irregularity})
            
        elif fake_shape_type == 'zigzag':
            # Draw a different zigzag shape
            self._draw_zigzag_shape(fake_draw, fake_x, fake_y, int(fake_size), 
                                {'irregularity': fake_irregularity})
            
        elif fake_shape_type == 'star':
            # Draw a different star-like shape
            self._draw_star_shape(fake_draw, fake_x, fake_y, int(fake_size), 
                                {'irregularity': fake_irregularity})
            
        else:
            # Default to simple ellipse
            fake_draw.ellipse((int(fake_x - fake_size//2), int(fake_y - fake_size//2), 
                            int(fake_x + fake_size//2), int(fake_y + fake_size//2)), fill=255)
        
        # Apply smoothing
        fake_mask = smooth_mask(fake_mask, smoothing_factor)
        
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
            fake_draw.ellipse((int(fake_x - fake_w//2), int(fake_y - fake_h//2), 
                            int(fake_x + fake_w//2), int(fake_y + fake_h//2)), fill=255)
            
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
        
        # Ensure the modified mask matches the size of the image before using putalpha
        if modified_mask.size != pil_img.size:
            # Resize the mask to match the image size
            resized_mask = Image.new("L", pil_img.size, 0)
            resized_mask.paste(modified_mask, (0, 0))
            modified_rgba.putalpha(resized_mask)
        else:
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
            # If no valid position found, use a position that's at least within bounds
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
                                max_depth, irregularity, smoothing_factor, difficulty, shape_type):
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
                max_depth, irregularity, smoothing_factor, shape_type
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