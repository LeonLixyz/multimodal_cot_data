"""
Multi-piece grid puzzle generator.

This module provides functionality to generate grid-based puzzles
where multiple pieces are removed, and the task is to identify
which options belong in which target cells.
"""

import os
import json
import random
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps
from typing import Dict, List, Any, Optional, Tuple, Union

from utils import load_image, create_visualization


class MultiPiecePuzzleGenerator:
    """
    Generator for multi-piece grid puzzles with support for multiple correct answers.
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the multi-piece puzzle generator.
        
        Args:
            seed: Optional random seed for reproducibility
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Default parameters for different difficulty levels
        self.params = {
            'easy': {
                'grid_size': 3,             # 3x3 grid
                'num_pieces_removed': 3,    # Remove 3 pieces
                'distractor_difficulty': 'low'  # Easy to distinguish distractors
            },
            'medium': {
                'grid_size': 4,             # 4x4 grid
                'num_pieces_removed': 4,    # Remove 4 pieces
                'distractor_difficulty': 'medium'  # Harder to distinguish
            },
            'hard': {
                'grid_size': 5,             # 5x5 grid
                'num_pieces_removed': 5,    # Remove 5 pieces
                'distractor_difficulty': 'high'  # Very similar distractors
            }
        }
        
        # Available distractor strategies
        self.distractor_strategies = [
            'other_cell',      # Take from another grid cell
            'transform',       # Apply transformations
            'shifted_grid',    # Shift the grid slightly
            'color_modify',    # Modify colors
            'pattern_modify'   # Add/remove patterns
        ]
    
    def generate(self, 
                image_path: str, 
                output_dir: str, 
                num_options: int = 6, 
                difficulty: str = 'medium',
                num_correct: int = 2,
                custom_params: Optional[Dict] = None,
                distractor_types: Optional[List[str]] = None) -> Dict:
        """
        Generate a multi-piece puzzle task with multiple correct answers.
        
        Args:
            image_path: Path to the source image
            output_dir: Directory to save the task files
            num_options: Number of piece options to generate (default: 6)
            difficulty: Difficulty level ('easy', 'medium', 'hard')
            num_correct: Number of correct answers to include (default: 2)
            custom_params: Optional dict to override default parameters
            distractor_types: List of distractor strategies to use (randomly chosen if None)
            
        Returns:
            Dictionary with task information
        """
        # Create output directory
        task_dir = os.path.join(output_dir, f"multi_piece_task_{difficulty}")
        os.makedirs(task_dir, exist_ok=True)
        
        # Get parameters based on difficulty level
        params = self.params.get(difficulty, self.params['medium']).copy()
        
        # Override with custom parameters if provided
        if custom_params:
            params.update(custom_params)
        
        # Load the image
        pil_img, cv_img = load_image(image_path)
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
        
        # FIXED: For multi-piece, remove exactly num_correct cells (2 in our case)
        # and all removed cells are the ones we're asking about
        num_removed = num_correct  # This ensures exactly 2 pieces are removed
        removed_cells = random.sample(cells, num_removed)  # Sample 2 cells
        test_cells = removed_cells.copy()  # All removed cells are test cells
        
        # Create the masked image with gray rectangles for removed cells
        masked_img = pil_img.copy()
        draw = ImageDraw.Draw(masked_img)
        
        # Only the test cells get gray rectangles
        for cell in test_cells:
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
        
        # Extract the correct pieces (test cells)
        correct_pieces = []
        for test_cell in test_cells:
            col, row = test_cell
            left = col * cell_width
            top = row * cell_height
            right = left + cell_width
            bottom = top + cell_height
            
            correct_piece = pil_img.crop((left, top, right, bottom))
            correct_pieces.append(correct_piece)
        
        # Save the masked image with highlighted target cells
        highlight_img = masked_img.copy()
        hdraw = ImageDraw.Draw(highlight_img)
        
        # Draw highlighted rectangle around each target cell
        for test_cell in test_cells:
            col, row = test_cell
            left = col * cell_width
            top = row * cell_height
            right = left + cell_width
            bottom = top + cell_height
            
            hdraw.rectangle((left, top, right, bottom), outline=(255, 0, 0), width=3)
        
        # Save both versions of the masked image
        masked_img_path = os.path.join(task_dir, "masked_image.jpg")
        masked_img.save(masked_img_path)
        
        highlighted_img_path = os.path.join(task_dir, "highlighted_targets.jpg")
        highlight_img.save(highlighted_img_path)
        
        # Generate the options
        options = []
        option_paths = []
        
        # First, add all correct pieces to the options
        for correct_piece in correct_pieces:
            options.append(correct_piece)
        
        # Then add distractors to fill up to num_options
        num_distractors = num_options - len(correct_pieces)
        
        # Select distractor strategies if not specified
        if not distractor_types:
            num_strategies = min(num_distractors, len(self.distractor_strategies))
            distractor_types = random.sample(self.distractor_strategies, num_strategies)
        
        # Generate the distractors
        for i in range(num_distractors):
            strategy_idx = i % len(distractor_types)
            strategy = distractor_types[strategy_idx]
            
            # Pick a random correct piece to use as a basis for the distractor
            correct_piece = random.choice(correct_pieces)
            test_cell = test_cells[correct_pieces.index(correct_piece)]
            
            if strategy == 'other_cell':
                # Strategy 1: Take a piece from another cell
                distractor = self._strategy_other_cell(
                    pil_img, cells, removed_cells, test_cells,
                    cell_width, cell_height, params['distractor_difficulty']
                )
            
            elif strategy == 'transform':
                # Strategy 2: Apply transformations to a correct piece
                distractor = self._strategy_transform(
                    correct_piece, params['distractor_difficulty']
                )
            
            elif strategy == 'shifted_grid':
                # Strategy 3: Shift the grid slightly and take a piece
                distractor = self._strategy_shifted_grid(
                    pil_img, test_cell, grid_size, cell_width, cell_height,
                    params['distractor_difficulty']
                )
            
            elif strategy == 'color_modify':
                # Strategy 4: Modify the colors of a correct piece
                distractor = self._strategy_color_modify(
                    correct_piece, params['distractor_difficulty']
                )
            
            elif strategy == 'pattern_modify':
                # Strategy 5: Add or remove patterns from a correct piece
                distractor = self._strategy_pattern_modify(
                    correct_piece, params['distractor_difficulty']
                )
            
            else:
                # Default fallback: Take from a random cell
                other_cells = [cell for cell in cells if cell not in test_cells]
                if other_cells:
                    random_cell = random.choice(other_cells)
                    
                    r_col, r_row = random_cell
                    r_left = r_col * cell_width
                    r_top = r_row * cell_height
                    r_right = r_left + cell_width
                    r_bottom = r_top + cell_height
                    
                    distractor = pil_img.crop((r_left, r_top, r_right, r_bottom))
                else:
                    # If no other cells, apply a transformation
                    distractor = correct_piece.transpose(Image.FLIP_LEFT_RIGHT)
            
            options.append(distractor)
        
        # Shuffle the options and track the correct indices
        indices = list(range(len(options)))
        random.shuffle(indices)
        
        shuffled_options = [options[i] for i in indices]
        correct_indices = [indices.index(i) for i in range(len(correct_pieces))]
        
        # Save the options
        for i, option in enumerate(shuffled_options):
            option_path = os.path.join(task_dir, f"option_{chr(65+i)}.jpg")
            option.save(option_path)
            option_paths.append(option_path)
        
        # Save the original image for reference
        original_path = os.path.join(task_dir, "original.jpg")
        pil_img.save(original_path)
        
        # Create a visualization of the task
        vis_path = create_visualization(
            highlight_img, shuffled_options, correct_indices, task_dir,
            title=f"Multi-Piece Puzzle ({num_correct} correct piece{'s' if num_correct > 1 else ''})", 
            show_grid=True, 
            grid_size=grid_size, 
            target_cells=test_cells
        )
        
        # Save task metadata
        metadata = {
            "task_type": "multi_piece_puzzle",
            "difficulty": difficulty,
            "num_correct": num_correct,
            "original_image": os.path.basename(original_path),
            "masked_image": os.path.basename(masked_img_path),
            "highlighted_targets": os.path.basename(highlighted_img_path),
            "options": [os.path.basename(path) for path in option_paths],
            "correct_indices": correct_indices,
            "correct_options": [chr(65 + idx) for idx in correct_indices],
            "grid_size": grid_size,
            "cell_width": cell_width,
            "cell_height": cell_height,
            "removed_cells": removed_cells,
            "test_cells": test_cells,
            "parameters": params
        }
        
        if distractor_types:
            # Record which strategies were used for each distractor
            strategy_mapping = []
            for i in range(len(shuffled_options)):
                if i in correct_indices:
                    strategy_mapping.append("correct")
                else:
                    # Calculate which distractor this corresponds to
                    distractor_idx = [j for j in range(len(shuffled_options)) if j not in correct_indices].index(i)
                    strategy_idx = distractor_idx % len(distractor_types)
                    strategy_mapping.append(distractor_types[strategy_idx])
            
            metadata["distractor_strategies"] = strategy_mapping
        
        metadata_path = os.path.join(task_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        return {
            "task_dir": task_dir,
            "masked_image": masked_img_path,
            "highlighted_target": highlighted_img_path,
            "options": option_paths,
            "correct_indices": correct_indices,
            "metadata": metadata
        }
    
    def _strategy_other_cell(self, pil_img, cells, removed_cells, test_cells,
                           cell_width, cell_height, distractor_difficulty):
        """Helper for distractor strategy: taking from another cell."""
        other_cells = [cell for cell in cells if cell not in test_cells]
        
        if not other_cells:
            # If no other cells available, use a cell from removed_cells
            # that's not a test cell
            other_cells = [cell for cell in removed_cells if cell not in test_cells]
            
            if not other_cells:
                # If still no cells available, create a blank cell
                distractor = Image.new('RGB', (cell_width, cell_height), (200, 200, 200))
                return distractor
        
        # Choose a cell based on difficulty
        if distractor_difficulty == 'high':
            # Try to find a cell adjacent to one of the test cells
            test_cell = random.choice(test_cells)
            test_col, test_row = test_cell
            
            adjacent_cells = [
                (test_col + 1, test_row),
                (test_col - 1, test_row),
                (test_col, test_row + 1),
                (test_col, test_row - 1)
            ]
            
            adjacent_cells = [cell for cell in adjacent_cells if cell in other_cells]
            
            if adjacent_cells:
                fake_cell = random.choice(adjacent_cells)
            else:
                # Sort cells by distance to the test cell
                other_cells.sort(key=lambda cell: abs(cell[0] - test_col) + abs(cell[1] - test_row))
                fake_cell = other_cells[0]
                
        elif distractor_difficulty == 'medium':
            # Take a cell that's moderately distant
            test_cell = random.choice(test_cells)
            test_col, test_row = test_cell
            
            # Sort cells by distance to the test cell
            other_cells.sort(key=lambda cell: abs(cell[0] - test_col) + abs(cell[1] - test_row))
            
            # Take a cell from the middle of the sorted list
            middle_idx = len(other_cells) // 2
            fake_cell = other_cells[middle_idx]
            
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
    
    def _strategy_transform(self, correct_piece, distractor_difficulty):
        """Helper for distractor strategy: transforming the correct piece."""
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
    
    def _strategy_shifted_grid(self, pil_img, test_cell, grid_size, 
                             cell_width, cell_height, distractor_difficulty):
        """Helper for distractor strategy: shifted grid cells."""
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
    
    def _strategy_color_modify(self, correct_piece, distractor_difficulty):
        """Helper for distractor strategy: color modifications."""
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
    
    def _strategy_pattern_modify(self, correct_piece, distractor_difficulty):
        """Helper for distractor strategy: adding/removing patterns."""
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