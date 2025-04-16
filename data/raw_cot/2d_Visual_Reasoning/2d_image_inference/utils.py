"""
Utility functions for the puzzle generator.
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any, Union, Optional
import matplotlib.gridspec as gridspec


def load_image(image_path: str) -> Tuple[Image.Image, np.ndarray]:
    """
    Load an image from path using both PIL and OpenCV.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Tuple containing (PIL Image, OpenCV image)
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at: {image_path}")
    
    pil_img = Image.open(image_path).convert("RGB")
    cv_img = cv2.imread(image_path)
    if cv_img is None:
        raise ValueError(f"Failed to load image with OpenCV: {image_path}")
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    
    return pil_img, cv_img


def add_noise_to_mask(mask: Image.Image, noise_level: float = 0.2) -> Image.Image:
    """
    Add noise to a mask to create more organic, natural-looking edges.
    
    Args:
        mask: Binary mask as PIL Image
        noise_level: Amount of noise to add (0.0 to 1.0)
        
    Returns:
        Modified mask with noise added
    """
    if noise_level <= 0:
        return mask
        
    # Convert mask to numpy array
    mask_np = np.array(mask)
    
    # Create perlin-like noise using gaussian filtering of random noise
    noise = np.random.rand(*mask_np.shape) * 255
    noise = cv2.GaussianBlur(noise, (0, 0), 3)
    
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


def smooth_mask(mask: Image.Image, smoothing_factor: float = 1.0) -> Image.Image:
    """
    Apply smoothing to a mask to create more natural edges.
    
    Args:
        mask: Binary mask as PIL Image
        smoothing_factor: Amount of smoothing to apply (higher = smoother)
        
    Returns:
        Smoothed mask
    """
    if smoothing_factor <= 0:
        return mask
        
    # Apply Gaussian blur
    smooth_radius = max(1, int(smoothing_factor * 2))
    smoothed_mask = mask.filter(ImageFilter.GaussianBlur(radius=smooth_radius))
    
    # Threshold to ensure binary mask
    smoothed_mask = smoothed_mask.point(lambda p: 255 if p > 127 else 0)
    
    return smoothed_mask


def create_visualization(masked_img: Image.Image, 
                         options: List[Image.Image], 
                         correct_indices: Union[int, List[int]], 
                         output_dir: str,
                         title: str = "Image with Cutout",
                         show_border: bool = False, 
                         show_grid: bool = False, 
                         grid_size: Optional[int] = None, 
                         target_cells: Optional[List[Tuple[int, int]]] = None) -> str:
    """
    Enhanced visualization function that clearly shows both the problem and options.
    """
    num_options = len(options)
    
    # Convert single index to list for consistent processing
    if isinstance(correct_indices, int):
        correct_indices = [correct_indices]
    
    # Create a larger figure to accommodate both problem and options
    fig = plt.figure(figsize=(15, 10))
    
    # Create a proper GridSpec for the entire figure
    gs = gridspec.GridSpec(3, 1, figure=fig)
    
    # PROBLEM SECTION - Take up more space for the puzzle question
    ax_problem = fig.add_subplot(gs[0:2, 0])
    ax_problem.imshow(masked_img)
    
    # Make the title more informative
    if show_grid and grid_size and target_cells:
        cell_info = ", ".join([f"({col},{row})" for col, row in target_cells])
        problem_title = f"{title}\nWhich pieces fit in the gray areas? (Locations: {cell_info})"
    else:
        problem_title = f"{title}\nWhich piece fits in the gray area?"
    
    ax_problem.set_title(problem_title, fontsize=16, fontweight='bold')
    ax_problem.axis('off')
    
    # Add grid overlay if requested
    if show_grid and grid_size:
        width, height = masked_img.size
        cell_width = width / grid_size
        cell_height = height / grid_size
        
        # Draw grid lines
        for i in range(grid_size + 1):
            # Vertical lines
            x = i * cell_width
            ax_problem.axvline(x=x, color='red', linestyle='-', linewidth=1)
            
            # Horizontal lines
            y = i * cell_height
            ax_problem.axhline(y=y, color='red', linestyle='-', linewidth=1)
        
        # Highlight target cells with more prominent borders
        if target_cells:
            for col, row in target_cells:
                left = col * cell_width
                top = row * cell_height
                rect = plt.Rectangle((left, top), cell_width, cell_height, 
                                    fill=False, edgecolor='red', linewidth=3)
                ax_problem.add_patch(rect)
    
    # OPTIONS SECTION - Show clearly labeled options below
    ax_options_container = fig.add_subplot(gs[2, 0])
    ax_options_container.set_frame_on(False)
    ax_options_container.set_xticks([])
    ax_options_container.set_yticks([])
    ax_options_container.set_title("Options (Select all that apply)", fontsize=14, fontweight='bold')
    
    # Create a grid for the options - always in a single row for consistency
    ncols = min(num_options, 4)  # Maximum 4 columns
    option_gs = gridspec.GridSpecFromSubplotSpec(1, ncols, subplot_spec=gs[2, 0])
    
    # Add options to the grid
    for i, option in enumerate(options):
        # Create subplot for this option
        ax_opt = fig.add_subplot(option_gs[0, i % ncols])
        ax_opt.imshow(option)
        
        # Label each option clearly
        if i in correct_indices:
            ax_opt.set_title(f"Option {chr(65+i)} (CORRECT)", 
                             color='green', fontsize=12, fontweight='bold')
        else:
            ax_opt.set_title(f"Option {chr(65+i)}", fontsize=12)
        
        ax_opt.axis('off')
        
        # Add border around each option
        for spine in ax_opt.spines.values():
            spine.set_visible(True)
            spine.set_color('green' if i in correct_indices else 'gray')
            spine.set_linewidth(3 if i in correct_indices else 1)
    
    plt.tight_layout()
    
    # Save the visualization with high quality
    vis_path = os.path.join(output_dir, "visualization.png")
    plt.savefig(vis_path, bbox_inches='tight', dpi=200)
    plt.close()
    
    return vis_path


def generate_distractors(correct_piece: Image.Image, 
                        num_distractors: int, 
                        difficulty: str = 'medium') -> List[Image.Image]:
    """
    Generate generic distractor pieces based on the correct piece.
    
    Args:
        correct_piece: The correct puzzle piece
        num_distractors: Number of distractor pieces to generate
        difficulty: Difficulty level ('easy', 'medium', 'hard')
        
    Returns:
        List of distractor pieces
    """
    distractors = []
    
    # Define transformation intensity based on difficulty
    if difficulty == 'easy':
        # Easy distractors - very different from correct piece
        transform_options = [
            lambda img: img.rotate(180),
            lambda img: img.transpose(Image.FLIP_LEFT_RIGHT),
            lambda img: img.transpose(Image.FLIP_TOP_BOTTOM),
            lambda img: img.resize((int(img.width * 0.7), int(img.height * 0.7))),
        ]
    elif difficulty == 'medium':
        # Medium distractors - moderately different
        transform_options = [
            lambda img: img.rotate(45),
            lambda img: img.transpose(Image.FLIP_LEFT_RIGHT),
            lambda img: img.filter(ImageFilter.GaussianBlur(2)),
            lambda img: img.resize((int(img.width * 0.9), int(img.height * 0.9))),
        ]
    else:  # hard
        # Hard distractors - subtle differences
        transform_options = [
            lambda img: img.rotate(5),
            lambda img: img.filter(ImageFilter.GaussianBlur(1)),
            lambda img: img.resize((int(img.width * 0.95), int(img.height * 0.95))),
            lambda img: img.transpose(Image.FLIP_LEFT_RIGHT),
        ]
    
    # Generate the requested number of distractors
    for i in range(num_distractors):
        transform_func = transform_options[i % len(transform_options)]
        distractor = transform_func(correct_piece.copy())
        distractors.append(distractor)
    
    return distractors