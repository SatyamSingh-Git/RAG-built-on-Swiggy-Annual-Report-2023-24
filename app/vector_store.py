"""
Vector Store Module.
Handles FAISS vector storage and retrieval operations.
"""

import pickle
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import faiss

from app.config import VECTORSTORE_DIR
from app.document_processor import Chunk


class FAISSVectorStore:
    """
    FAISS-based vector store for semantic search.
    
    Features:
    - Efficient similarity search using IndexFlatIP (inner product)
    - Persistence to disk
    - Metadata storage alongside vectors
    """
    
    def __init__(self, dimension: int = 3072):
        """
        Initialize vector store.
        
        Args:
            dimension: Embedding dimension (768 for Gemini embeddings)
        """
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        self.chunks: List[Chunk] = []
        self.id_to_idx: dict = {}
    
    def add_chunks(self, chunks: List[Chunk], embeddings: np.ndarray) -> None:
        """
        Add chunks with their embeddings to the store.
        
        Args:
            chunks: List of Chunk objects
            embeddings: NumPy array of embeddings (n_chunks, dimension)
        """
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings")
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Store chunks
        start_idx = len(self.chunks)
        for i, chunk in enumerate(chunks):
            self.id_to_idx[chunk.id] = start_idx + i
            self.chunks.append(chunk)
        
        # Add to FAISS index
        self.index.add(embeddings)
        
        print(f"Added {len(chunks)} chunks to vector store. Total: {len(self.chunks)}")
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10
    ) -> List[Tuple[Chunk, float]]:
        """
        Search for similar chunks.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            
        Returns:
            List of (Chunk, score) tuples sorted by relevance
        """
        # Normalize query embedding
        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding, min(top_k, len(self.chunks)))
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self.chunks):
                results.append((self.chunks[idx], float(score)))
        
        return results
    
    def save(self, path: Optional[Path] = None) -> None:
        """
        Save vector store to disk.
        
        Args:
            path: Directory to save to (uses default if not provided)
        """
        save_dir = path or VECTORSTORE_DIR
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        index_path = save_dir / "index.faiss"
        faiss.write_index(self.index, str(index_path))
        
        # Save chunks and metadata
        meta_path = save_dir / "metadata.pkl"
        with open(meta_path, 'wb') as f:
            pickle.dump({
                'chunks': [c.to_dict() for c in self.chunks],
                'id_to_idx': self.id_to_idx,
                'dimension': self.dimension
            }, f)
        
        print(f"Saved vector store to {save_dir}")
    
    @classmethod
    def load(cls, path: Optional[Path] = None) -> 'FAISSVectorStore':
        """
        Load vector store from disk.
        
        Args:
            path: Directory to load from
            
        Returns:
            Loaded FAISSVectorStore instance
        """
        load_dir = path or VECTORSTORE_DIR
        load_dir = Path(load_dir)
        
        index_path = load_dir / "index.faiss"
        meta_path = load_dir / "metadata.pkl"
        
        if not index_path.exists() or not meta_path.exists():
            raise FileNotFoundError(f"Vector store not found at {load_dir}")
        
        # Load metadata
        with open(meta_path, 'rb') as f:
            metadata = pickle.load(f)
        
        # Create instance
        store = cls(dimension=metadata['dimension'])
        store.index = faiss.read_index(str(index_path))
        store.id_to_idx = metadata['id_to_idx']
        store.chunks = [
            Chunk(
                id=c['id'],
                content=c['content'],
                page_number=c['page_number'],
                section=c.get('section', ''),
                chunk_type=c.get('chunk_type', 'text'),
                metadata=c.get('metadata', {})
            )
            for c in metadata['chunks']
        ]
        
        print(f"Loaded vector store with {len(store.chunks)} chunks")
        return store
    
    def exists(self, path: Optional[Path] = None) -> bool:
        """Check if vector store exists at path."""
        check_dir = path or VECTORSTORE_DIR
        check_dir = Path(check_dir)
        return (check_dir / "index.faiss").exists()
    
    def __len__(self) -> int:
        return len(self.chunks)


# CLI for testing
if __name__ == "__main__":
    # Test basic operations
    store = FAISSVectorStore(dimension=768)
    
    # Create dummy chunks and embeddings
    from app.document_processor import Chunk
    
    chunks = [
        Chunk(id="test_0", content="Swiggy revenue was 10,000 crores", page_number=1),
        Chunk(id="test_1", content="Food delivery is the main business", page_number=2),
    ]
    embeddings = np.random.randn(2, 768).astype(np.float32)
    
    store.add_chunks(chunks, embeddings)
    
    # Test search
    query_emb = np.random.randn(768).astype(np.float32)
    results = store.search(query_emb, top_k=2)
    
    for chunk, score in results:
        print(f"Score: {score:.4f} | {chunk.content[:50]}...")
