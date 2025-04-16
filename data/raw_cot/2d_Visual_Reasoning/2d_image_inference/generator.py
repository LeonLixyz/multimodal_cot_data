"""
Main generator class for puzzle tasks.
"""

import os
import json
import random
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union

from jigsaw import JigsawPuzzleGenerator
from multi_piece import MultiPiecePuzzleGenerator
from fractal import FractalCutoutGenerator


class PuzzleGenerator:
    """
    Main class for generating visual puzzle tasks for multimodal understanding.
    
    This class provides a unified interface to generate different types of
    puzzle tasks, including jigsaw pieces, multi-piece grid puzzles, and
    fractal cutouts.
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the puzzle generator.
        
        Args:
            seed: Optional random seed for reproducibility
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Initialize sub-generators
        self.jigsaw_generator = JigsawPuzzleGenerator(seed)
        self.multi_piece_generator = MultiPiecePuzzleGenerator(seed)
        self.fractal_generator = FractalCutoutGenerator(seed)
    
    def generate_jigsaw_piece(self, 
                             image_path: str, 
                             output_dir: str, 
                             num_options: int = 4, 
                             difficulty: str = 'medium',
                             num_pieces: Optional[int] = None,
                             custom_params: Optional[Dict] = None,
                             distractor_types: Optional[List[str]] = None) -> Dict:
        """
        Generate a task with jigsaw-like pieces (with tabs and blanks) cut out of an image.
        
        Args:
            image_path: Path to the source image
            output_dir: Directory to save the task files
            num_options: Number of piece options to generate (default: 4)
            difficulty: Difficulty level ('easy', 'medium', 'hard')
            num_pieces: Number of pieces to cut out (default: random 1-2)
            custom_params: Optional dict to override default parameters
            distractor_types: List of distractor strategies to use (randomly chosen if None)
            
        Returns:
            Dictionary with task information
        """
        # If num_pieces is not specified, randomly choose between 1 and 2
        if num_pieces is None:
            num_pieces = random.randint(1, 2)
            
        return self.jigsaw_generator.generate(
            image_path=image_path,
            output_dir=output_dir,
            num_options=num_options,
            difficulty=difficulty,
            num_pieces=num_pieces,
            custom_params=custom_params,
            distractor_types=distractor_types
        )
    
    def generate_multi_piece_puzzle(self, 
                                   image_path: str, 
                                   output_dir: str, 
                                   num_options: int = 6, 
                                   difficulty: str = 'medium',
                                   num_correct: Optional[int] = None,
                                   custom_params: Optional[Dict] = None,
                                   distractor_types: Optional[List[str]] = None) -> Dict:
        """
        Generate a task with multiple pieces of a grid removed, where the model
        must identify which pieces belong in specific locations.
        
        Args:
            image_path: Path to the source image
            output_dir: Directory to save the task files
            num_options: Number of piece options to generate (default: 6)
            difficulty: Difficulty level ('easy', 'medium', 'hard')
            num_correct: Number of correct answers (default: random 1-3)
            custom_params: Optional dict to override default parameters
            distractor_types: List of distractor strategies to use (randomly chosen if None)
            
        Returns:
            Dictionary with task information
        """
        # If num_correct is not specified, randomly choose between 1, 2, or 3
        if num_correct is None:
            num_correct = random.randint(1, 3)
            
        return self.multi_piece_generator.generate(
            image_path=image_path,
            output_dir=output_dir,
            num_options=num_options,
            difficulty=difficulty,
            num_correct=num_correct,
            custom_params=custom_params,
            distractor_types=distractor_types
        )
    
    def generate_fractal_cutout(self, 
                               image_path: str, 
                               output_dir: str, 
                               num_options: int = 4, 
                               difficulty: str = 'medium',
                               num_pieces: Optional[int] = None,
                               shape_types: Optional[List[str]] = None,
                               custom_params: Optional[Dict] = None,
                               distractor_types: Optional[List[str]] = None) -> Dict:
        """
        Generate a task with complex fractal-like edge cutouts from an image.
        
        Args:
            image_path: Path to the source image
            output_dir: Directory to save the task files
            num_options: Number of piece options to generate (default: 4)
            difficulty: Difficulty level ('easy', 'medium', 'hard')
            num_pieces: Number of pieces to cut out (default: random 1-3)
            shape_types: List of shape types to use (e.g., 'fractal', 'corner_cut', 'wave')
            custom_params: Optional dict to override default parameters
            distractor_types: List of distractor strategies to use (randomly chosen if None)
            
        Returns:
            Dictionary with task information
        """
        # If num_pieces is not specified, randomly choose between 1, 2, or 3
        if num_pieces is None:
            num_pieces = random.randint(1, 3)
            
        return self.fractal_generator.generate(
            image_path=image_path,
            output_dir=output_dir,
            num_options=num_options,
            difficulty=difficulty,
            num_pieces=num_pieces,
            shape_types=shape_types,
            custom_params=custom_params,
            distractor_types=distractor_types
        )
    
    def generate_all_task_types(self, 
                               image_path: str, 
                               output_dir: str, 
                               num_options: int = 6, 
                               difficulty: str = 'medium') -> Dict:
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
        
        # Randomly generate number of pieces for each task type
        jigsaw_num_pieces = random.randint(1, 2)
        multi_piece_num_correct = random.randint(1, 3)
        fractal_num_pieces = random.randint(1, 3)
        
        tasks = {
            'jigsaw': self.generate_jigsaw_piece(
                image_path, os.path.join(output_dir, "jigsaw"), 
                num_options, difficulty, num_pieces=jigsaw_num_pieces
            ),
            'multi_piece': self.generate_multi_piece_puzzle(
                image_path, os.path.join(output_dir, "multi_piece"), 
                num_options, difficulty, num_correct=multi_piece_num_correct
            ),
            'fractal': self.generate_fractal_cutout(
                image_path, os.path.join(output_dir, "fractal"), 
                num_options, difficulty, num_pieces=fractal_num_pieces
            )
        }
        
        return tasks
    
    def generate_dataset(self, 
                         image_dir: str, 
                         output_dir: str, 
                         num_tasks_per_image: int = 1, 
                         task_types: Optional[List[str]] = None, 
                         difficulty_levels: Optional[List[str]] = None, 
                         num_options: int = 6,
                         include_visualizations: bool = True) -> Tuple[Dict, str]:
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
            Tuple of (dataset metadata dictionary, path to dataset metadata file)
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
                    
                    # Generate the task based on type with random number of pieces/correct answers
                    if task_type == 'jigsaw':
                        num_pieces = random.randint(1, 2)
                        task = self.generate_jigsaw_piece(
                            img_path, task_subdir, num_options, difficulty, num_pieces=num_pieces
                        )
                        
                        # Add to dataset
                        task_entry = {
                            'image_file': img_file,
                            'task_type': task_type,
                            'difficulty': difficulty,
                            'task_dir': os.path.relpath(task['task_dir'], output_dir),
                            'masked_image': os.path.relpath(task['masked_image'], output_dir),
                            'options': [os.path.relpath(opt, output_dir) for opt in task['options']],
                            'correct_indices': task['correct_indices'],
                            'correct_options': [chr(65 + idx) for idx in task['correct_indices']],
                            'num_pieces': num_pieces
                        }
                    
                    elif task_type == 'multi_piece':
                        num_correct = random.randint(1, 3)
                        task = self.generate_multi_piece_puzzle(
                            img_path, task_subdir, num_options, difficulty, num_correct=num_correct
                        )
                        
                        # Add to dataset
                        task_entry = {
                            'image_file': img_file,
                            'task_type': task_type,
                            'difficulty': difficulty,
                            'task_dir': os.path.relpath(task['task_dir'], output_dir),
                            'masked_image': os.path.relpath(task['masked_image'], output_dir),
                            'options': [os.path.relpath(opt, output_dir) for opt in task['options']],
                            'correct_indices': task['correct_indices'],
                            'correct_options': [chr(65 + idx) for idx in task['correct_indices']],
                            'num_correct': num_correct
                        }
                        
                        # Add highlighted target if available
                        if 'highlighted_target' in task:
                            task_entry['highlighted_target'] = os.path.relpath(task['highlighted_target'], output_dir)
                    
                    elif task_type == 'fractal':
                        num_pieces = random.randint(1, 3)
                        task = self.generate_fractal_cutout(
                            img_path, task_subdir, num_options, difficulty, num_pieces=num_pieces
                        )
                        
                        # Add to dataset
                        task_entry = {
                            'image_file': img_file,
                            'task_type': task_type,
                            'difficulty': difficulty,
                            'task_dir': os.path.relpath(task['task_dir'], output_dir),
                            'masked_image': os.path.relpath(task['masked_image'], output_dir),
                            'options': [os.path.relpath(opt, output_dir) for opt in task['options']],
                            'correct_indices': task['correct_indices'],
                            'correct_options': [chr(65 + idx) for idx in task['correct_indices']],
                            'num_pieces': num_pieces
                        }
                    
                    else:
                        continue
                    
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