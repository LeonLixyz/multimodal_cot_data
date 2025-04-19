import json
from openai import OpenAI
import os

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

with open('output.json', 'r') as f:
    messages = json.load(f)

response = client.responses.create(
    model="gpt-4.1",
    input=messages,
    temperature=0.7,
)