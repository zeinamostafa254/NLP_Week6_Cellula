import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load environment variables from .env 
load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1"
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME") or "mistralai/mixtral-8x7b-instruct"

# Ensure API key is present
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY is missing. Please add it to your .env file.")

# Initialize llm
llm = ChatOpenAI(
    model=LLM_MODEL_NAME,         
    base_url=OPENROUTER_BASE_URL,  
    api_key=OPENROUTER_API_KEY,    
    temperature=0.0
)