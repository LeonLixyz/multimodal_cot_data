import re
import os
import sys
import shutil
from pathlib import Path

def create_directory_if_not_exists(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def process_images(content, folder_path):
    """
    Extract image paths from content and copy them to the destination folder.
    Returns updated content with new image paths and fixed LaTeX syntax.
    """
    # Create images directory in the folder
    images_dir = os.path.join(folder_path, 'images')
    create_directory_if_not_exists(images_dir)
    
    # Fix invalid LaTeX options in includegraphics commands
    # Replace 'max width=' with 'width='
    content = re.sub(r'\\includegraphics\[max width=([^]]*?)\]', r'\\includegraphics[width=\1]', content)
    
    # Remove 'center' from options and add \centering before
    content = re.sub(r'\\includegraphics\[(.*?),\s*center\s*(.*?)\]', r'\\centering\n\\includegraphics[\1\2]', content)
    content = re.sub(r'\\includegraphics\[center,\s*(.*?)\]', r'\\centering\n\\includegraphics[\1]', content)
    content = re.sub(r'\\includegraphics\[center\]', r'\\centering\n\\includegraphics', content)
    
    # Pattern to find includegraphics commands
    img_pattern = r'\\includegraphics(?:\[.*?\])?\{(.*?)\}'
    
    # Find all image paths
    image_paths = re.findall(img_pattern, content)
    
    # Process each image
    for img_path in image_paths:
        # Extract filename from path
        img_filename = os.path.basename(img_path)
        
        # Remove the prefix from the filename
        new_filename = re.sub(r'2025_04_17_97bc1f7e44d93c271a88g-', '', img_filename)

        if not img_filename.lower().endswith('.jpg'):
            img_filename += '.jpg'
        
        # Ensure new filename also has .jpg extension
        if not new_filename.lower().endswith('.jpg'):
            new_filename += '.jpg'
        
        # Source image path (assuming images are in a folder called 'images')
        source_path = os.path.join('/Users/leon66/Desktop/VLM Reasoning/VLM Reasoning Repo/data/raw_cot/Science/geometry/aux_lines/latex/images', img_filename)

        print(f"Looking for image at: {source_path}")
        print(f"File exists: {os.path.exists(source_path)}")
        
        # Destination path
        dest_path = os.path.join(images_dir, new_filename)
        
        # Copy the image if it exists
        if os.path.exists(source_path):
            shutil.copy(source_path, dest_path)
            print(f"    Copied image: {source_path} -> {dest_path}")
        else:
            print(f"    Warning: Image not found: {source_path}")
        
        # Replace the path in the content
        content = content.replace(f'{{{img_path}}}', f'{{images/{new_filename}}}')
    
    return content

def create_tex_file(folder_path, content, title):
    """Create a TeX file with the given content."""
    # Add basic LaTeX structure
    tex_content = f"""\\documentclass{{article}}
\\usepackage{{amsmath}}
\\usepackage{{amssymb}}
\\usepackage{{graphicx}}
\\usepackage{{hyperref}}
\\usepackage[version=4]{{mhchem}}

\\title{{{title}}}
\\date{{}}

\\begin{{document}}
\\maketitle

{content}
\\end{{document}}
"""
    
    # Create the tex file
    tex_file = os.path.join(folder_path, "main.tex")
    with open(tex_file, 'w', encoding='utf-8') as f:
        f.write(tex_content)
    print(f"Created file: {tex_file}")

def extract_and_process_sections(file_paths):
    """Extract sections marked with triple hash marks and process them."""
    # Dictionary to store all content from all files
    all_content = ""
    
    # Read all files and concatenate their content
    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                all_content += file.read() + "\n\n"
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
    
    # Find all section markers and extract content
    # The pattern captures: chapter number, section type, and set number
    pattern = r'###\s*CHAPTER\s+(\d+)\s+(EXAMPLE|PROBLEMS|SOLUTIONS)\s+1-(\d+)\s*###(.*?)(?=###|$)'
    matches = re.findall(pattern, all_content, re.DOTALL)
    
    # Print all matches found to help diagnose issues
    print(f"Found {len(matches)} section matches in total.")
    
    # Dictionary to store organized content
    organized_content = {}
    
    # Process each match
    for chapter_num, section_type, set_num, section_content in matches:
        # Create keys for organization
        chapter_key = f"chapter_{chapter_num}"
        set_key = f"set_{set_num}"
        
        # Initialize nested dictionaries if needed
        if chapter_key not in organized_content:
            organized_content[chapter_key] = {}
        
        if set_key not in organized_content[chapter_key]:
            organized_content[chapter_key][set_key] = {}
        
        # Store the content
        organized_content[chapter_key][set_key][section_type] = section_content.strip()
    
    # Process the organized content
    for chapter_key, chapter_data in organized_content.items():
        chapter_num = chapter_key.split('_')[1]
        print(f"\nProcessing Chapter {chapter_num}:")
        
        for set_key, set_data in chapter_data.items():
            set_num = set_key.split('_')[1]
            print(f"  Processing Set {set_num}:")
            
            # Process EXAMPLES
            if 'EXAMPLE' in set_data:
                example_content = set_data['EXAMPLE']
                
                # Extract individual examples
                example_pattern = re.compile(r'Example\s+(\d+)\.\s+(.*?)(?=Example\s+\d+\.|$)', re.DOTALL)
                examples = example_pattern.findall(example_content)
                
                if not examples:
                    # If no individual examples found, treat the whole content as one example
                    examples = [('1', example_content)]
                
                for example_num, example_text in examples:
                    # Create folder for this example
                    example_folder = f"chapter_{chapter_num}_example_{set_num}_{example_num}"
                    create_directory_if_not_exists(example_folder)
                    
                    # Process images in the example
                    processed_text = process_images(example_text, example_folder)
                    
                    # Create the TeX file
                    create_tex_file(
                        example_folder, 
                        processed_text, 
                        f"Example {example_num}"
                    )
                    print(f"    Created Example {example_num}")
            
            # Check if we have both PROBLEMS and SOLUTIONS
            if 'PROBLEMS' in set_data and 'SOLUTIONS' in set_data:
                problems_content = set_data['PROBLEMS']
                solutions_content = set_data['SOLUTIONS']
                
                # Extract individual problems
                problem_pattern = re.compile(r'Problem\s+(\d+)\.\s+(.*?)(?=Problem\s+\d+\.|$)', re.DOTALL)
                problems = problem_pattern.findall(problems_content)
                
                # Extract individual solutions
                solution_pattern = re.compile(r'Problem\s+(\d+)\.\s+Solution:(.*?)(?=Problem\s+\d+\.|$)', re.DOTALL)
                solutions = {num: solution.strip() for num, solution in solution_pattern.findall(solutions_content)}
                
                if not solutions:  # Try alternative pattern if no solutions found
                    solution_pattern = re.compile(r'Problem\s+(\d+)\.\s+Solution\.(.*?)(?=Problem\s+\d+\.|$)', re.DOTALL)
                    solutions = {num: solution.strip() for num, solution in solution_pattern.findall(solutions_content)}
                
                # Match problems with solutions
                for problem_num, problem_text in problems:
                    problem_solution = solutions.get(problem_num, "Solution not available.")
                    
                    # Skip problems with no solutions
                    if problem_solution == "Solution not available.":
                        print(f"    Skipping Problem {problem_num}: Solution not available")
                        continue
                    
                    # Create folder for this problem
                    problem_folder = f"chapter_{chapter_num}_problem_{set_num}_{problem_num}"
                    create_directory_if_not_exists(problem_folder)
                    
                    # Process images and fix LaTeX formatting
                    processed_problem = process_images(problem_text, problem_folder)
                    processed_solution = process_images(problem_solution, problem_folder)
                    
                    # Combined content
                    combined_content = f"""\\section*{{Problem}}
{processed_problem.strip()}

\\section*{{Solution}}
{processed_solution}
"""
                    
                    # Create the TeX file
                    create_tex_file(
                        problem_folder, 
                        combined_content, 
                        f"Problem {problem_num}"
                    )
                    
                    # Generate PDF
                    generate_pdf(problem_folder)
                    
                    # Clean up intermediate files
                    cleanup_latex_files(problem_folder)
                    
                    print(f"    Created Problem {problem_num} with Solution and PDF")
    
    return True

def generate_pdf(folder_path):
    """Generate PDF from LaTeX file using pdflatex."""
    import subprocess
    
    # Get the full path to the folder
    full_path = os.path.abspath(folder_path)
    
    # Run pdflatex twice to resolve references
    subprocess.run(
        ['pdflatex', '-interaction=nonstopmode', 'main.tex'],
        cwd=full_path,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    
    subprocess.run(
        ['pdflatex', '-interaction=nonstopmode', 'main.tex'],
        cwd=full_path,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

def cleanup_latex_files(folder_path):
    """Remove intermediate LaTeX files."""
    extensions = ['.aux', '.log', '.out', '.fls', '.synctex.gz']
    
    for ext in extensions:
        file_path = os.path.join(folder_path, f'main{ext}')
        if os.path.exists(file_path):
            os.remove(file_path)

def main(input_files):
    """Main function to process the input files."""
    if extract_and_process_sections(input_files):
        print("\nProcessing complete!")
    else:
        print("\nProcessing failed.")

# This is a Jupyter notebook, so we'll execute the main function directly
input_files = ["latex/latex_modified.tex"]
main(input_files)