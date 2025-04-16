#!/usr/bin/env python3
import os
import argparse
import json
from generator import PuzzleGenerator
import time

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate specific puzzle types')
    parser.add_argument('--image', type=str, default='dogs.jpg', 
                        help='Path to the input image')
    parser.add_argument('--output_dir', type=str, default='outputs/fixed_puzzles', 
                        help='Directory to save results')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create a metadata structure to track all generated tasks
    dataset = {
        'image': args.image,
        'seed': args.seed,
        'generation_time': time.strftime("%Y-%m-%d %H:%M:%S"),
        'tasks': []
    }
    
    # Initialize the generator
    generator = PuzzleGenerator(seed=args.seed)
    
    # 1. Generate jigsaw puzzle (1 piece, 4 options, medium difficulty)
    jigsaw_dir = os.path.join(args.output_dir, "jigsaw_puzzle")
    os.makedirs(jigsaw_dir, exist_ok=True)
    
    print("Generating jigsaw puzzle (1 piece, 4 options)...")
    jigsaw_result = generator.generate_jigsaw_piece(
        image_path=args.image,
        output_dir=jigsaw_dir,
        num_options=4,
        difficulty='medium',
        num_pieces=1  # Exactly 1 piece
    )
    
    # Add to dataset
    dataset['tasks'].append({
        'type': 'jigsaw',
        'difficulty': 'medium',
        'output_dir': os.path.relpath(jigsaw_result['task_dir'], args.output_dir),
        'correct_indices': jigsaw_result['correct_indices'],
        'correct_options': [chr(65 + idx) for idx in jigsaw_result['correct_indices']],
        'num_pieces': 1
    })
    
    # 2. Generate multi-piece puzzle (2 correct pieces, 4 options, medium difficulty)
    multi_dir = os.path.join(args.output_dir, "multi_piece_puzzle")
    os.makedirs(multi_dir, exist_ok=True)
    
    print("Generating multi-piece puzzle (2 pieces, 4 options)...")
    multi_result = generator.generate_multi_piece_puzzle(
        image_path=args.image,
        output_dir=multi_dir,
        num_options=4,
        difficulty='medium',
        num_correct=2  # Exactly 2 correct pieces
    )
    
    # Add to dataset
    dataset['tasks'].append({
        'type': 'multi_piece',
        'difficulty': 'medium',
        'output_dir': os.path.relpath(multi_result['task_dir'], args.output_dir),
        'correct_indices': multi_result['correct_indices'],
        'correct_options': [chr(65 + idx) for idx in multi_result['correct_indices']],
        'num_correct': 2
    })
    
    # 3. Generate all types of fractal cutouts (1 piece each, 4 options, medium difficulty)
    print("Generating fractal cutouts for all shape types...")
    
    fractal_types = ['fractal', 'corner_cut', 'wave', 'zigzag', 'star']
    for shape_type in fractal_types:
        fractal_dir = os.path.join(args.output_dir, f"fractal_{shape_type}")
        os.makedirs(fractal_dir, exist_ok=True)
        
        print(f"  Generating '{shape_type}' fractal cutout...")
        fractal_result = generator.generate_fractal_cutout(
            image_path=args.image,
            output_dir=fractal_dir,
            num_options=4,
            difficulty='medium',
            num_pieces=1,  # Exactly 1 piece
            shape_types=[shape_type]  # Specific shape type
        )
        
        # Add to dataset
        dataset['tasks'].append({
            'type': 'fractal',
            'shape_type': shape_type,
            'difficulty': 'medium',
            'output_dir': os.path.relpath(fractal_result['task_dir'], args.output_dir),
            'correct_indices': fractal_result['correct_indices'],
            'correct_options': [chr(65 + idx) for idx in fractal_result['correct_indices']],
            'num_pieces': 1
        })
    
    # Save the dataset metadata
    metadata_path = os.path.join(args.output_dir, "fixed_puzzles_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print("\n=== Generation Summary ===")
    print(f"Total tasks generated: {len(dataset['tasks'])}")
    print(f"Output directory: {args.output_dir}")
    print(f"Dataset metadata: {metadata_path}")

if __name__ == "__main__":
    main()