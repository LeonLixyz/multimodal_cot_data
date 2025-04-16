import os
import json
import shutil
import re
from PIL import Image
import base64
from io import BytesIO

def process_conversation(input_file):
    """
    Process a conversation trace JSON file:
    1. Extract conversation text to a txt file
    2. Replace image URLs with placeholders in JSON
    3. Save intermediate images as separate files
    4. Save the input image as input_image.png
    
    Args:
        input_file: Path to the JSON file containing the conversation trace
    """
    # Get the original directory
    output_dir = os.path.dirname(input_file)
    print(f"Processing conversation to {output_dir}")
    
    # Read the original JSON file
    with open(input_file, 'r', encoding='utf-8') as f:
        conversation = json.load(f)
    
    # Extract conversation text and images
    text_content = []
    images = []
    modified_conversation = []
    
    for message in conversation:
        role = message.get('role', '')
        modified_message = message.copy()
        modified_content = []
        
        # Start each message with the role
        text_content.append(f"{role.upper()}:\n")
        
        for content_item in message.get('content', []):
            if content_item.get('type') == 'text':
                # Add text to conversation
                text = content_item.get('text', '')
                if text:
                    text_content.append(f"{text}\n\n")
                modified_content.append(content_item)
                
            elif content_item.get('type') == 'image_url':
                # Extract image data - directly use the URL as base64 data
                base64_str = content_item.get('image_url', {}).get('url', '')
                if base64_str:
                    images.append(base64_str)
                    # Replace with placeholder
                    image_index = len(images)
                    placeholder = f"intermediate image {image_index}"
                    
                    # Add placeholder to text content
                    text_content.append(f"[{placeholder}]\n\n")
                    
                    # Update JSON
                    modified_item = content_item.copy()
                    modified_item['image_url'] = {'url': placeholder}
                    modified_content.append(modified_item)
                else:
                    modified_content.append(content_item)
        
        modified_message['content'] = modified_content
        modified_conversation.append(modified_message)
        
        # Add separator between messages
        text_content.append("\n-------------------\n\n")
    
    # Save conversation text
    with open(os.path.join(output_dir, 'conversation.txt'), 'w', encoding='utf-8') as f:
        f.writelines(text_content)
    
    # Save modified JSON
    with open(os.path.join(output_dir, 'modified_conversation.json'), 'w', encoding='utf-8') as f:
        json.dump(modified_conversation, f, indent=2)
    
    # Save intermediate images - directly decode the base64 string
    for i, base64_str in enumerate(images):
        try:
            image_bytes = base64.b64decode(base64_str)
            with open(os.path.join(output_dir, f'intermediate_image_{i+1}.png'), 'wb') as f:
                f.write(image_bytes)
            print(f"Saved intermediate_image_{i+1}.png in {output_dir}")
        except Exception as e:
            print(f"Error saving image {i+1}: {e}")
    
    # Copy/rename the original input image
    try:
        input_image_path = os.path.join(output_dir, 'image.png')
        if os.path.exists(input_image_path):
            shutil.copy(input_image_path, os.path.join(output_dir, 'input_image.png'))
            print(f"Original image copied as input_image.png in {output_dir}")
        else:
            print(f"No image.png found in {output_dir}")
    except Exception as e:
        print(f"Error copying input image: {e}")

if __name__ == "__main__":
    # Process the conversation in outputs/geometry/1540/output.json
    process_conversation('outputs/geometry/1540/output.json')
    print("Conversation trace processed and saved in 'input_files' directory")