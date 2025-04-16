import os

# set up the agent
MAX_REPLY = 10

# set up the LLM for the agent
os.environ['OPENAI_API_KEY'] = 'your_api_key'
os.environ["AUTOGEN_USE_DOCKER"] = "False"
llm_config={"cache_seed": None, "config_list": [{"model": "gpt-4o", "temperature": 0.0, "api_key": os.environ.get("OPENAI_API_KEY")}]}

# Mock authentication for local services - typically would be admin/admin for development
os.environ["LOCAL_API_USERNAME"] = "admin"
os.environ["LOCAL_API_PASSWORD"] = "admin"
os.environ["LOCAL_API_KEY"] = "sk_local_development_token_12345"

# use this after building your own server. You can also set up the server in other machines and paste them here.
SOM_ADDRESS = "http://localhost:8080/"
GROUNDING_DINO_ADDRESS = "http://localhost:8081/"
DEPTH_ANYTHING_ADDRESS = "http://localhost:8082/"
