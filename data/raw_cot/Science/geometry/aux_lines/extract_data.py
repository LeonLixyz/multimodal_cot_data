import re
import os
import sys
import shutil
import subprocess
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
    return tex_file

def compile_tex_to_pdf(tex_file_path):
    """Compile the TeX file to PDF using pdflatex, saving PDF even with errors."""
    folder_path = os.path.dirname(tex_file_path)
    filename = os.path.basename(tex_file_path)
    pdf_generated = False
    
    # Save current directory
    original_dir = os.getcwd()
    
    try:
        # Change to the directory containing the tex file
        os.chdir(folder_path)
        
        # Run pdflatex twice to ensure references are resolved
        for i in range(2):
            process = subprocess.run(
                ['pdflatex', '-interaction=nonstopmode', filename],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Check if PDF was created after each run
            pdf_filename = filename.replace('.tex', '.pdf')
            if os.path.exists(pdf_filename):
                pdf_generated = True
                
            # Log errors but continue
            if process.returncode != 0:
                print(f"    Warning: Compilation attempt {i+1} had errors for {filename}, but continuing")
        
        # Final check for PDF
        pdf_filename = filename.replace('.tex', '.pdf')
        if pdf_generated:
            print(f"    PDF generated: {os.path.join(folder_path, pdf_filename)}")
            return True
        else:
            print(f"    No PDF was generated for {filename} despite attempts")
            return False
            
    except Exception as e:
        print(f"    Error during PDF compilation: {e}")
        # Even after exception, check if PDF was created
        pdf_filename = filename.replace('.tex', '.pdf')
        if os.path.exists(pdf_filename):
            print(f"    PDF was still generated despite errors: {os.path.join(folder_path, pdf_filename)}")
            return True
        return False
    finally:
        # Return to original directory
        os.chdir(original_dir)

def extract_and_process_sections(file_paths, output_dir='compiled_pdfs', clean=True):
    """Extract sections marked with triple hash marks and process them."""
    # Clean output directory if it exists and clean flag is True
    if clean and os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    
    # Create output directory for all PDFs
    create_directory_if_not_exists(output_dir)
    
    # Clean any existing problem/example directories if clean flag is True
    if clean:
        for dir_name in os.listdir('.'):
            if (dir_name.startswith('chapter_') and 
                ('_example_' in dir_name or '_problem_' in dir_name) and 
                os.path.isdir(dir_name)):
                shutil.rmtree(dir_name)
    
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
    pattern = r'###\s*CHAPTER\s+(\d+)\s+(EXAMPLE|PROBLEMS|SOLUTIONS)\s+SET\s+(\d+)\s*###(.*?)(?=###|$)'
    matches = re.findall(pattern, all_content, re.DOTALL)
    
    # Print all matches found to help diagnose issues
    print(f"Found {len(matches)} section matches in total.")
    
    # Create sets to keep track of processed examples and problems
    processed_examples = set()
    processed_problems = set()
    
    # Process EXAMPLES first (they have both questions and solutions)
    example_sections = [m for m in matches if m[1] == 'EXAMPLE']
    for chapter_num, _, set_num, section_content in example_sections:
        print(f"\nProcessing Chapter {chapter_num} Example Set {set_num}:")
        
        # Extract individual examples
        example_pattern = re.compile(r'Example\s+(\d+)\.\s+(.*?)(?=Example\s+\d+\.|$)', re.DOTALL)
        examples = example_pattern.findall(section_content)
        
        if not examples:
            # If no individual examples found, treat the whole content as one example
            examples = [('1', section_content)]
        
        for example_num, example_text in examples:
            # Create a unique ID for this example
            example_id = f"chapter_{chapter_num}_example_{set_num}_{example_num}"
            
            # Skip if already processed
            if example_id in processed_examples:
                print(f"    Skipping already processed Example {example_num}")
                continue
            
            processed_examples.add(example_id)
            
            # Create folder for this example
            example_folder = example_id
            
            # Clean the folder if it exists
            if os.path.exists(example_folder):
                shutil.rmtree(example_folder)
            
            create_directory_if_not_exists(example_folder)
            
            # Process images in the example
            processed_text = process_images(example_text, example_folder)
            
            # Create the TeX file and compile to PDF
            tex_file = create_tex_file(
                example_folder, 
                processed_text, 
                f"Example {example_num}"
            )
            print(f"    Created Example {example_num}")
            
            # Compile to PDF
            compile_success = compile_tex_to_pdf(tex_file)
            
            # Copy PDF to output directory if compilation was successful
            if compile_success:
                pdf_path = os.path.join(example_folder, "main.pdf")
                if os.path.exists(pdf_path):
                    output_pdf_name = f"{example_id}.pdf"
                    output_pdf_path = os.path.join(output_dir, output_pdf_name)
                    shutil.copy(pdf_path, output_pdf_path)
                    print(f"    Copied PDF to {output_pdf_path}")
    
    # Now process PROBLEMS and SOLUTIONS
    # First, extract all problems and solutions by chapter and set
    problems_by_chapter_set = {}
    solutions_by_chapter_set = {}
    
    for chapter_num, section_type, set_num, section_content in matches:
        if section_type == 'PROBLEMS':
            if (chapter_num, set_num) not in problems_by_chapter_set:
                problems_by_chapter_set[(chapter_num, set_num)] = section_content
        elif section_type == 'SOLUTIONS':
            if (chapter_num, set_num) not in solutions_by_chapter_set:
                solutions_by_chapter_set[(chapter_num, set_num)] = section_content
    
    # Now process each chapter/set problem section with its matching solution section
    for (chapter_num, set_num), problems_content in problems_by_chapter_set.items():
        print(f"\nProcessing Chapter {chapter_num} Problems Set {set_num}:")
        
        # Find corresponding solutions for this chapter and set
        solutions_content = solutions_by_chapter_set.get((chapter_num, set_num), "")
        
        # Extract individual problems using regex
        problem_pattern = re.compile(r'Problem\s+(\d+)\.\s+(.*?)(?=Problem\s+\d+\.|$)', re.DOTALL)
        problems = problem_pattern.findall(problems_content)
        
        if not problems:
            print(f"    No problems found in Chapter {chapter_num} Set {set_num}")
            continue
        
        # Extract solutions for this chapter and set
        solutions = {}
        if solutions_content:
            # Try multiple patterns to extract solutions
            solution_patterns = [
                # Pattern 1: "Problem X. Solution: ..."
                re.compile(r'Problem\s+(\d+)\.\s+Solution:\s*(.*?)(?=Problem\s+\d+\.|$)', re.DOTALL),
                # Pattern 2: "Problem X. Solution. ..."
                re.compile(r'Problem\s+(\d+)\.\s+Solution\.\s*(.*?)(?=Problem\s+\d+\.|$)', re.DOTALL),
                # Pattern 3: "Problem X. (any text) Solution: ..."
                re.compile(r'Problem\s+(\d+)\..*?Solution:\s*(.*?)(?=Problem\s+\d+\.|$)', re.DOTALL)
            ]
            
            # Try each pattern until we find matches
            for pattern in solution_patterns:
                sol_matches = pattern.findall(solutions_content)
                if sol_matches:
                    for num, sol in sol_matches:
                        solutions[num] = sol.strip()
                    break
        
        # Process each problem with its matching solution
        for problem_num, problem_text in problems:
            # Create a unique ID for this problem
            problem_id = f"chapter_{chapter_num}_problem_{set_num}_{problem_num}"
            
            # Skip if already processed
            if problem_id in processed_problems:
                print(f"    Skipping already processed Problem {problem_num}")
                continue
            
            processed_problems.add(problem_id)
            
            solution_text = solutions.get(problem_num, "Solution not available.")
            
            # Create folder for this problem
            problem_folder = problem_id
            
            # Clean the folder if it exists
            if os.path.exists(problem_folder):
                shutil.rmtree(problem_folder)
            
            create_directory_if_not_exists(problem_folder)
            
            # Process images
            processed_problem = process_images(problem_text, problem_folder)
            processed_solution = process_images(solution_text, problem_folder)
            
            # Combined content
            combined_content = f"""\\section*{{Problem}}
{processed_problem.strip()}

\\section*{{Solution}}
{processed_solution}
"""
            
            # Create the TeX file
            tex_file = create_tex_file(
                problem_folder, 
                combined_content, 
                f"Problem {problem_num}"
            )
            print(f"    Created Problem {problem_num}" + 
                  (" with Solution" if problem_num in solutions else " (solution not found)"))
            
            # Compile to PDF and copy to output directory even if there are errors
            compile_success = compile_tex_to_pdf(tex_file)
            pdf_path = os.path.join(problem_folder, "main.pdf")

            # Copy PDF to output directory if it exists, regardless of compilation success
            if os.path.exists(pdf_path):
                output_pdf_name = f"{problem_id}.pdf"
                output_pdf_path = os.path.join(output_dir, output_pdf_name)
                shutil.copy(pdf_path, output_pdf_path)
                print(f"    Copied PDF to {output_pdf_path}")
            else:
                print(f"    No PDF was available to copy for {problem_id}")
    
    # Clean up temporary files in all the processed directories
    for folder in processed_examples.union(processed_problems):
        if os.path.exists(folder):
            for file in os.listdir(folder):
                if file.endswith(('.aux', '.log', '.out')):
                    os.remove(os.path.join(folder, file))
    
    return True

def main(input_files, output_dir='compiled_pdfs', clean=True):
    """Main function to process the input files."""
    if extract_and_process_sections(input_files, output_dir, clean):
        print(f"\nProcessing complete! PDFs are available in '{output_dir}' directory.")
    else:
        print("\nProcessing failed.")

# Input files and output directory
input_files = ["latex/latex_modified.tex"]
output_dir = "compiled_pdfs"
clean = True  # Set to False if you don't want to clean existing directories
main(input_files, output_dir, clean)