import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load environment variables from .env 
load_dotenv()

# ============================================================
# LLM Configuration (Member 2)
# ============================================================
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

# ============================================================
# Ingestion Configuration (Member 1)
# ============================================================

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_PERSIST_DIR = os.path.join(BASE_DIR, "chroma_db")
DATA_DIR = os.path.join(BASE_DIR, "data")

# Chunking
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

# Embedding model name (HuggingFace)
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Feedback loop
MAX_FEEDBACK_ITERATIONS = 4

# Retrieval
TOP_K_RESULTS = 5