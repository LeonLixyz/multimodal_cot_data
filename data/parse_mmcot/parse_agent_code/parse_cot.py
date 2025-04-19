import json
import base64
import os
import re
import shutil
from openai import OpenAI
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from io import BytesIO

client = OpenAI(api_key="your_api_key")

def image_to_base64(image_path):
    """Convert an image file to base64 encoded string."""
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            return encoded_string
        
    except FileNotFoundError: 
        print(f"Warning: Image file not found at {image_path}")
        return None
    except Exception as e:
        print(f"Error encoding image {image_path}: {e}")
        return None

def process_cleaned_trace(input_geometry_dir, cleaned_trace_path, output_base_dir="../parsed_cot"):
    """
    Process a cleaned trace JSON file by replacing image placeholders with actual base64 images.
    
    Args:
        input_geometry_dir: Path to the geometry directory (e.g., 'raw_cot/VisualSketchpad/outputs/geometry/1540')
        cleaned_trace_path: Path to the cleaned_trace.json file
        output_base_dir: Base directory for output (default: '../parsed_cot' - one level above current)
    """
    # Parse directory path components
    parts = input_geometry_dir.split('/')
    geometry_id = parts[-1]  # e.g., "1540"
    
    # Setup output directory with same structure
    output_dir = os.path.join(output_base_dir, "VisualSketchpad/outputs/geometry", geometry_id)
    os.makedirs(output_dir, exist_ok=True)
    
    # Input image path
    input_image_path = os.path.join(input_geometry_dir, "input_image.png")
    
    # Get the base64 encoded input image
    input_image_base64 = image_to_base64(input_image_path)
    if not input_image_base64:
        print(f"ERROR: Could not encode input image at {input_image_path}")
        return
    
    # Load intermediate images
    intermediate_images_base64 = []
    i = 1
    while True:
        img_path = os.path.join(input_geometry_dir, f"intermediate_image_{i}.png")
        if not os.path.exists(img_path):
            break  # Stop when no more intermediate images are found
        
        img_base64 = image_to_base64(img_path)
        if img_base64:
            intermediate_images_base64.append(img_base64)
        i += 1
    
    # Load the cleaned trace JSON
    try:
        with open(cleaned_trace_path, 'r', encoding='utf-8') as f:
            cleaned_trace = json.load(f)
    except Exception as e:
        print(f"Error loading cleaned trace: {e}")
        return
    
    # Replace placeholders with actual base64 images
    for entry in cleaned_trace:
        if entry.get("role") == "user":
            for content_item in entry.get("content", []):
                if content_item.get("type") == "image_url":
                    url = content_item.get("image_url", {}).get("url", "")
                    if url == "[input_image]":
                        content_item["image_url"]["url"] = input_image_base64
        
        elif entry.get("role") == "assistant":
            for content_item in entry.get("content", []):
                if content_item.get("type") == "image_url":
                    url = content_item.get("image_url", {}).get("url", "")
                    if url.startswith("[intermediate_image_"):
                        try:
                            # Extract the image index from the placeholder
                            match = re.search(r'\[intermediate_image_(\d+)\]', url)
                            if match:
                                idx = int(match.group(1)) - 1
                                if idx < len(intermediate_images_base64):
                                    content_item["image_url"]["url"] = intermediate_images_base64[idx]
                        except Exception as e:
                            print(f"Error processing intermediate image: {e}")
    
    # Save the processed trace
    processed_trace_path = os.path.join(output_dir, "processed_trace.json")
    with open(processed_trace_path, 'w', encoding='utf-8') as f:
        json.dump(cleaned_trace, f, indent=2)
    
    print(f"✅ Processed trace saved to: {processed_trace_path}")
    
    # Create a flattened version for PDF generation
    flattened_data = []
    for entry in cleaned_trace:
        role = entry.get("role", "unknown")
        
        # Process each content item as a separate entry in the flattened data
        for content_item in entry.get("content", []):
            item_type = content_item.get("type")
            
            if item_type == "text":
                flattened_data.append({
                    "role": role,
                    "type": "text",
                    "text": content_item.get("text", "")
                })
            
            elif item_type == "image_url":
                flattened_data.append({
                    "role": role,
                    "type": "image_url",
                    "image": content_item.get("image_url", {}).get("url", "")
                })
    
    # Generate PDF
    pdf_path = os.path.join(output_dir, "cot_output.pdf")
    export_cot_to_pdf(flattened_data, pdf_path)
    
    # Copy original images to the output directory for reference
    shutil.copy(input_image_path, os.path.join(output_dir, "input_image.png"))
    for i in range(1, len(intermediate_images_base64) + 1):
        src_path = os.path.join(input_geometry_dir, f"intermediate_image_{i}.png")
        dst_path = os.path.join(output_dir, f"intermediate_image_{i}.png")
        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)
    
    return processed_trace_path, pdf_path

def export_cot_to_pdf(data, pdf_path="cot_output.pdf"):
    """Generate a PDF from the CoT data."""
    c = canvas.Canvas(pdf_path, pagesize=A4)
    width, height = A4
    y = height - 50  # start near the top

    def draw_text(text, indent=50):
        nonlocal y
        lines = text.split("\n")
        for line in lines:
            for wrapped in split_text(line, max_chars=100):
                if y < 80:
                    c.showPage()
                    y = height - 50
                c.drawString(indent, y, wrapped)
                y -= 15
        y -= 10  # space after block

    def split_text(text, max_chars=100):
        # simple word wrap
        words = text.split()
        lines = []
        current = ""
        for word in words:
            if len(current + " " + word) <= max_chars:
                current += " " + word if current else word
            else:
                lines.append(current)
                current = word
        if current:
            lines.append(current)
        return lines

    def draw_image_from_base64(b64):
        nonlocal y
        try:
            img_data = base64.b64decode(b64)
            img = ImageReader(BytesIO(img_data))
            img_width, img_height = img.getSize()
            ratio = min(400 / img_width, 200 / img_height)
            img_width *= ratio
            img_height *= ratio
            if y < img_height + 50:
                c.showPage()
                y = height - 50
            c.drawImage(img, 100, y - img_height, width=img_width, height=img_height)
            y -= img_height + 20
        except Exception as e:
            draw_text(f"[Error decoding base64 image: {e}]")

    for entry in data:
        role = entry.get("role", "unknown").upper()
        c.setFont("Helvetica-Bold", 10)
        draw_text(f"{role}:")

        c.setFont("Helvetica", 10)
        if entry["type"] == "text":
            draw_text(entry.get("text", ""))
        elif entry["type"] == "image_url":
            draw_image_from_base64(entry.get("image", ""))

        c.line(40, y, width - 40, y)
        y -= 20

    c.save()
    print(f"✅ PDF saved to: {pdf_path}")

if __name__ == "__main__":
    # Process the example geometry folder
    input_dir = "../../raw_cot/VisualSketchpad/outputs/geometry/1540"
    
    # The cleaned trace is in the current directory, not in parse_mmcot/visual_sketchpad
    cleaned_trace_path = "cleaned_trace.json"
    
    # Output directory will be saved one level above current directory
    process_cleaned_trace(input_dir, cleaned_trace_path, "../parsed_cot")
