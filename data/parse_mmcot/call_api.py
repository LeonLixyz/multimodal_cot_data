#!/usr/bin/env python3
import os
import json
import argparse
from openai import OpenAI

def call_api_with_folder(input_folder, temperature=0.7, model="gpt-4.1"):
    """
    Call OpenAI API using api_input_message.json in the input folder
    
    Args:
        input_folder: Directory containing api_input_message.json
        temperature: Temperature setting for the API call
        model: Model to use for the API call
    
    Returns:
        Path to the output JSON file
    """
    # Find the input JSON file
    input_json_path = os.path.join(input_folder, "api_input_message.json")
    if not os.path.exists(input_json_path):
        raise ValueError(f"Could not find api_input_message.json in {input_folder}")
    
    # Load the JSON input
    with open(input_json_path, 'r', encoding='utf-8') as f:
        messages = json.load(f)
    
    # Initialize the OpenAI client
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
    )
    
    # Call the API
    response = client.responses.create(
        model=model,
        input=messages,
        temperature=temperature,
    )
    
    # Create output path
    output_path = os.path.join(input_folder, "api_output.json")
    
    # Create the output JSON
    output_data = {
        "output_text": response.output_text,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "total_tokens": response.usage.total_tokens,
        "temperature": response.temperature
    }
    
    # Save to a JSON file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"API call completed. Output saved to {output_path}")
    print(f"Input tokens: {response.usage.input_tokens}")
    print(f"Output tokens: {response.usage.output_tokens}")
    print(f"Total tokens: {response.usage.total_tokens}")
    
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Call OpenAI API with input from a folder")
    parser.add_argument("--input_folder", required=True, help="Path to the input folder containing api_input_message.json")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for API call")
    parser.add_argument("--model", default="gpt-4.1", help="Model to use for API call")
    
    args = parser.parse_args()
    
    # Call the API
    call_api_with_folder(args.input_folder, args.temperature, args.model)

