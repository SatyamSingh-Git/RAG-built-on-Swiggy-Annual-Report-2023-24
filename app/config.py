"""
Configuration management for RAG application.
Loads settings from environment variables with sensible defaults.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
VECTORSTORE_DIR = DATA_DIR / "vectorstore"
EXPORTS_DIR = BASE_DIR / "exports"

# Create directories if they don't exist
VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
EXPORTS_DIR.mkdir(parents=True, exist_ok=True)

# API Configuration
try:
    import streamlit as st
    
    # helper to find key
    def get_secret_key():
        # 1. Top level
        if "GOOGLE_API_KEY" in st.secrets:
            return st.secrets["GOOGLE_API_KEY"]
        
        # 2. Nested under [gemini]
        if "gemini" in st.secrets:
            if "api_key" in st.secrets["gemini"]:
                return st.secrets["gemini"]["api_key"]
            if "GOOGLE_API_KEY" in st.secrets["gemini"]:
                return st.secrets["gemini"]["GOOGLE_API_KEY"]
                
        # 3. Nested under [general] or [secrets]
        for section in ["general", "secrets"]:
            if section in st.secrets and "GOOGLE_API_KEY" in st.secrets[section]:
                return st.secrets[section]["GOOGLE_API_KEY"]
                
        return None

    # Try getting from streamlit secrets
    secret_key = get_secret_key()
    
    if secret_key:
        print("DEBUG: Found GOOGLE_API_KEY in st.secrets")
        GOOGLE_API_KEY = secret_key
    else:
        print("DEBUG: GOOGLE_API_KEY NOT in st.secrets, checking os.getenv")
        # Debug available keys to help user
        try:
            print(f"DEBUG: Available top-level keys: {list(st.secrets.keys())}")
        except:
             print("DEBUG: Could not list secrets keys")
        GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

except ImportError:
    print("DEBUG: streamlit import failed, falling back to os.getenv")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
except Exception as e:
    print(f"DEBUG: Error reading secrets: {e}")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# Strip whitespace just in case
if GOOGLE_API_KEY:
    GOOGLE_API_KEY = GOOGLE_API_KEY.strip()
    print(f"DEBUG: Final API Key length: {len(GOOGLE_API_KEY)}")
else:
    print("DEBUG: Final API Key is EMPTY")

# Model Configuration
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")

# Chunking Configuration
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1024"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "124"))

# Retrieval Configuration
TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", "10"))
TOP_K_RERANK = int(os.getenv("TOP_K_RERANK", "5"))

# PDF Path
PDF_PATH = BASE_DIR / os.getenv("PDF_PATH", "Annual-Report-FY-2023-24.pdf")

# Validate configuration
def validate_config():
    """Validate that required configuration is present."""
    errors = []
    
    if not GOOGLE_API_KEY:
        errors.append("GOOGLE_API_KEY is not set. Please add it to your .env file.")
    
    if not PDF_PATH.exists():
        errors.append(f"PDF file not found at: {PDF_PATH}")
    
    return errors


# System prompts
SYSTEM_PROMPT = """You are a financial analyst assistant specializing in the Swiggy Annual Report FY 2023-24.

CRITICAL RULES:
1. Answer ONLY using information from the provided context
2. If the information is not in the context, respond: "I cannot find this information in the Swiggy Annual Report."
3. ALWAYS cite page numbers for your claims (e.g., "According to page 42...")
4. For numerical data, quote the exact figures from the report
5. If asked about topics not in the report (stock prices, future predictions, etc.), clearly state this is outside the report's scope

FORMAT:
- Be concise but thorough
- Use bullet points for lists
- Include specific page references
- Highlight key financial figures when relevant"""

PROXY_ANSWER_PROMPT = """Given this potentially unclear or short query, generate a hypothetical detailed answer that would help retrieve relevant information from a financial annual report.

Query: {query}

Generate a detailed proxy answer (2-3 sentences) that expands on what the user might be looking for. Focus on financial metrics, business segments, or specific report sections that would be relevant."""

QUERY_CLASSIFIER_PROMPT = """Classify the following query as either:
- CLEAR: A well-formed question that can be directly used for retrieval
- UNCLEAR: A short, vague, or ambiguous query that needs expansion

Query: {query}

Respond with only CLEAR or UNCLEAR."""
