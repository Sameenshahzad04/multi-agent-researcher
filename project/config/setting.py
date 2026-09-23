
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

class Config:
    # Database & Core
    DATABASE_URL = os.getenv("DATABASE_URL")
    
    # OpenRouter API Settings
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL")
    SYSTEM_MESSAGE = os.getenv("SYSTEM_MESSAGE")

    # Search API Settings
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

    # =============================================================================
    # Tool Configuration (FREE TIER)
    # =============================================================================
    CALCULATOR_MAX_LENGTH = 100  # Max expression length for security safety
    DOCUMENT_SEARCH_TOP_K = 3    # Used later in Task 2 RAG


 # Fz


    # Task 2: RAG Pipeline Configuration
    # =============================================================================
    # FIX: Use absolute paths anchored to this file's location.
    # Relative Path("corpus") resolves relative to CWD — this breaks when you run
    # the agent from a different directory than where these folders live.
    from pathlib import Path
    _BASE_DIR = Path(__file__).parent.parent  # → Multi_agent_researcher/
    CORPUS_DIR   = _BASE_DIR / "corpus"         # Always: Multi_agent_researcher/corpus/
    CHROMA_DB_DIR = _BASE_DIR / "chroma_db"     # Always: Multi_agent_researcher/chroma_db/
    MODELS_DIR   = _BASE_DIR / "local_models"   # Always: Multi_agent_researcher/local_models/
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2" # Local embedding model
    CHUNK_SIZE = 400                        # Character size per chunk
    CHUNK_OVERLAP = 50                      # Overlap between chunks
    
    # Advanced Extraction & Chunking Configurations
    TABLE_MAX_CHARS = 1000
    TABLE_CHUNK_OVERLAP = 2
    CHUNK_TYPE_TEXT = "text"
    CHUNK_TYPE_TABLE = "table"

    # Create directories (now uses absolute paths — always creates in the right place)
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # =============================================================================
    # Agent Parameters
    # =============================================================================
    MAX_AGENT_ITERATIONS = 5 # Prevents infinite tool-calling loops
    AGENT_TIMEOUT = 60  

# Instantiate the configuration object globally
config = Config()
def get_llm():
    """Initializes and returns the ChatOpenAI model routed through OpenRouter."""
    print(f"DEBUG -> API Key loaded: {bool(config.OPENROUTER_API_KEY)} (Length: {len(config.OPENROUTER_API_KEY) if config.OPENROUTER_API_KEY else 0})")
    print(f"DEBUG -> Model loaded: {config.OPENROUTER_MODEL}")

    if not config.OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY is missing from environment variables.")

    return ChatOpenAI(
        base_url=config.OPENROUTER_BASE_URL,
        api_key=config.OPENROUTER_API_KEY,
        model=config.OPENROUTER_MODEL,
        temperature=0.2,
        default_headers={
            "Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
            "HTTP-Referer": "http://localhost:8000", # Optional, but recommended by OpenRouter
            "X-Title": "MultiAgentResearcher"         # Optional, but recommended by OpenRouter
        }
    )