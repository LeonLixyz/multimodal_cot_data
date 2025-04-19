#!/usr/bin/env python3
import os
import json
import base64
import argparse
from pathlib import Path

def format_multimodal_trace(input_dir, prompt_file, output_file):
    """
    Format multimodal trace data for OpenAI API.
    
    Args:
        input_dir: Directory containing images folder and text file
        prompt_file: File containing the system prompt
        output_file: Path to save formatted messages
    """
    # Load prompt
    with open(prompt_file, 'r', encoding='utf-8') as f:
        system_prompt = f.read()
    
    # Load text content
    text_path = os.path.join(input_dir, "text.tex")
    with open(text_path, 'r', encoding='utf-8') as f:
        text_content = f.read()
    
    # Find images
    images_dir = os.path.join(input_dir, "images")
    images = {}
    
    for filename in os.listdir(images_dir):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            # Extract name without extension (e.g., "problem_image_1")
            name = os.path.splitext(filename)[0]
            images[name] = os.path.join(images_dir, filename)
    
    # Create messages
    messages = [
        {
            "role": "developer",
            "content": system_prompt
        }
    ]
    
    # Create user content
    user_content = [
        {
            "type": "input_text",
            "text": "Here is the problem and the raw reasoning trace with image placeholders:"
        },
        {
            "type": "input_text",
            "text": "<problem_statement_and_raw_trace_start>\n\n"
        },
        {
            "type": "input_text",
            "text": text_content
        },
        {
            "type": "input_text",
            "text": "<problem_statement_and_raw_trace_end>\n\n"
        }
    ]

    user_content.append({
        "type": "input_text",
        "text": "Below is the image data corresponding to the placeholders:\n\n"
    })
    user_content.append({
        "type": "input_text",
        "text": "<image_data_start>\n\n"
    })  

    # Add images
    for name, path in images.items():
        # Read image file
        with open(path, "rb") as image_file:
            image_data = image_file.read()
            base64_data = base64.b64encode(image_data).decode('utf-8')
        
        # Add image description
        user_content.append({
            "type": "input_text",
            "text": f"\nThe actual image for placeholder [{name}]:"
        })
        
        # Add image
        user_content.append({
            "type": "input_image",
            "image_url": f"data:image/jpeg;base64,{base64_data}"
        })

    user_content.append({
        "type": "input_text",
        "text": "<image_data_end>\n\n"
    })
    
    user_content.append({
        "type": "input_text",
        "text": "Following the instructions provided, generate the clean and logical coherent multimodal reasoning trace in the specified format using the raw trace and image data above."
    })
    
    # Add user message
    messages.append({
        "role": "user",
        "content": user_content
    })
    
    # Save to output file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    output_data = {
        "input_dir": input_dir,
        "messages": messages
    }
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Formatted messages saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Format multimodal traces for OpenAI API")
    parser.add_argument("--input_dir", required=True, help="Path to the input directory")
    parser.add_argument("--prompt_file", required=True, help="Path to the prompt file")
    parser.add_argument("--output_file", required=True, help="Path to save the formatted messages")
    
    args = parser.parse_args()
    
    format_multimodal_trace(args.input_dir, args.prompt_file, args.output_file)