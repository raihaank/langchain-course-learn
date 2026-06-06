import os
from dotenv import load_dotenv

def load_config():
    """Loads environment variables from .env file."""
    load_dotenv()
    
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key or openai_api_key == "your_openai_api_key_here":
        raise ValueError("OPENAI_API_KEY not found in environment variables. Please set it in .env file.")
        
    return {
        "OPENAI_API_KEY": openai_api_key
    }
