"""
Embeddings Module.
Handles text embedding using Google's Gemini Embedding API.
"""

from typing import List, Optional
import numpy as np
import google.generativeai as genai

from app.config import GOOGLE_API_KEY, EMBEDDING_MODEL


class GeminiEmbeddings:
    """
    Wrapper for Google's Gemini Embedding API.
    
    Uses gemini-embedding-001 model which is free tier eligible.
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = EMBEDDING_MODEL):
        """
        Initialize Gemini embeddings.
        
        Args:
            api_key: Google API key (uses env var if not provided)
            model: Embedding model name
        """
        self.api_key = api_key or GOOGLE_API_KEY
        self.model = model
        
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY is required. Set it in .env file.")
        
        genai.configure(api_key=self.api_key)
    
    def embed_text(self, text: str, task_type: str = "RETRIEVAL_DOCUMENT") -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            task_type: Type of task (RETRIEVAL_DOCUMENT, RETRIEVAL_QUERY, etc.)
            
        Returns:
            Embedding vector as list of floats
        """
        result = genai.embed_content(
            model=self.model,
            content=text,
            task_type=task_type
        )
        return result['embedding']
    
    def embed_documents(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        """
        Generate embeddings for multiple documents.
        
        Args:
            texts: List of texts to embed
            show_progress: Whether to show progress bar
            
        Returns:
            NumPy array of embeddings (n_texts, embedding_dim)
        """
        embeddings = []
        
        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(texts, desc="Generating embeddings")
        else:
            iterator = texts
        
        for text in iterator:
            # Truncate if too long (Gemini has 2048 token limit)
            truncated = text[:8000]  # Rough character limit
            embedding = self.embed_text(truncated, task_type="RETRIEVAL_DOCUMENT")
            embeddings.append(embedding)
        
        return np.array(embeddings, dtype=np.float32)
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a search query.
        
        Uses RETRIEVAL_QUERY task type for optimal query embedding.
        
        Args:
            query: Search query text
            
        Returns:
            Embedding vector as NumPy array
        """
        embedding = self.embed_text(query, task_type="RETRIEVAL_QUERY")
        return np.array(embedding, dtype=np.float32)
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension (3072 for gemini-embedding-001)."""
        return 3072


# CLI for testing
if __name__ == "__main__":
    embeddings = GeminiEmbeddings()
    
    # Test single embedding
    test_text = "Swiggy is India's leading on-demand delivery platform."
    embedding = embeddings.embed_text(test_text)
    print(f"Embedding dimension: {len(embedding)}")
    print(f"First 5 values: {embedding[:5]}")
    
    # Test query embedding
    query = "What is Swiggy's revenue?"
    query_emb = embeddings.embed_query(query)
    print(f"Query embedding shape: {query_emb.shape}")
