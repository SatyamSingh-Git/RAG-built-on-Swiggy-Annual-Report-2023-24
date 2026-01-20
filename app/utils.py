import google.generativeai as genai
from typing import Optional, List
import os
from app.config import GOOGLE_API_KEY, GEMINI_MODEL

def get_fallback_chain(primary_model: str) -> List[str]:
    """Get a prioritized list of fallback models."""
    chain = [
        primary_model,
        "gemini-2.0-flash",
        "gemini-1.5-flash",
        "gemini-1.5-flash-lite",
        "gemini-1.5-pro",
        "gemini-1.0-pro"
    ]
    # Deduplicate while preserving order
    seen = set()
    return [x for x in chain if not (x in seen or seen.add(x))]

def init_gemini_model(model_name: str, system_instruction: Optional[str] = None):
    """Initialize a Gemini model with optional system instructions and fallback."""
    supports_system_instruction = True
    if "gemma" in model_name.lower() or "1.0" in model_name:
        supports_system_instruction = False
        
    try:
        if supports_system_instruction and system_instruction:
            model = genai.GenerativeModel(
                model_name=model_name,
                system_instruction=system_instruction
            )
        else:
            model = genai.GenerativeModel(model_name=model_name)
            supports_system_instruction = False
    except Exception:
        model = genai.GenerativeModel(model_name=model_name)
        supports_system_instruction = False
        
    return model, supports_system_instruction
