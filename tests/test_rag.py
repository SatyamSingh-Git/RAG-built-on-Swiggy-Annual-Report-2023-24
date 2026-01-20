"""
Tests for RAG Application.
"""

import sys
from pathlib import Path
import pytest
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestDocumentProcessor:
    """Tests for document processor module."""
    
    def test_chunk_dataclass(self):
        """Test Chunk dataclass creation."""
        from app.document_processor import Chunk
        
        chunk = Chunk(
            id="test_0",
            content="Test content",
            page_number=1,
            section="Test Section"
        )
        
        assert chunk.id == "test_0"
        assert chunk.content == "Test content"
        assert chunk.page_number == 1
        assert chunk.section == "Test Section"
        assert chunk.chunk_type == "text"
    
    def test_chunk_to_dict(self):
        """Test Chunk serialization."""
        from app.document_processor import Chunk
        
        chunk = Chunk(
            id="test_0",
            content="Test content",
            page_number=1
        )
        
        d = chunk.to_dict()
        assert d["id"] == "test_0"
        assert d["content"] == "Test content"
        assert d["page_number"] == 1
    
    def test_processor_initialization(self):
        """Test DocumentProcessor initialization."""
        from app.document_processor import DocumentProcessor
        from app.config import CHUNK_SIZE, CHUNK_OVERLAP
        
        processor = DocumentProcessor()
        assert processor.chunk_size == CHUNK_SIZE
        assert processor.chunk_overlap == CHUNK_OVERLAP


class TestVectorStore:
    """Tests for vector store module."""
    
    def test_vector_store_creation(self):
        """Test FAISSVectorStore creation."""
        from app.vector_store import FAISSVectorStore
        
        store = FAISSVectorStore(dimension=3072)
        assert len(store) == 0
        assert store.dimension == 3072
    
    def test_add_and_search(self):
        """Test adding chunks and searching."""
        from app.vector_store import FAISSVectorStore
        from app.document_processor import Chunk
        
        store = FAISSVectorStore(dimension=3072)
        
        # Create test chunks
        chunks = [
            Chunk(id="test_0", content="Revenue was 10000 crores", page_number=1),
            Chunk(id="test_1", content="Food delivery is the main segment", page_number=2),
        ]
        
        # Create random embeddings
        embeddings = np.random.randn(2, 3072).astype(np.float32)
        
        store.add_chunks(chunks, embeddings)
        
        assert len(store) == 2
        
        # Test search
        query_emb = np.random.randn(3072).astype(np.float32)
        results = store.search(query_emb, top_k=2)
        
        assert len(results) == 2
        assert results[0][0].id in ["test_0", "test_1"]


class TestRetriever:
    """Tests for retriever module."""
    
    def test_retrieval_result_dataclass(self):
        """Test RetrievalResult dataclass."""
        from app.retriever import RetrievalResult
        from app.document_processor import Chunk
        
        chunk = Chunk(id="test", content="test", page_number=1)
        result = RetrievalResult(
            chunk=chunk,
            dense_score=0.8,
            sparse_score=0.6,
            rerank_score=0.9,
            combined_score=0.75
        )
        
        assert result.dense_score == 0.8
        assert result.rerank_score == 0.9


class TestGenerator:
    """Tests for generator module."""
    
    def test_generation_result_dataclass(self):
        """Test GenerationResult dataclass."""
        from app.generator import GenerationResult
        
        result = GenerationResult(
            answer="Test answer",
            citations=[{"page": 1, "section": "Test"}],
            confidence=0.85,
            sources_used=1,
            raw_response="Test answer"
        )
        
        assert result.answer == "Test answer"
        assert result.confidence == 0.85
        assert len(result.citations) == 1


class TestConfig:
    """Tests for configuration module."""
    
    def test_config_values(self):
        """Test configuration values are set."""
        from app.config import CHUNK_SIZE, CHUNK_OVERLAP, TOP_K_RETRIEVAL
        
        assert CHUNK_SIZE == 1024
        assert CHUNK_OVERLAP == 124
        assert TOP_K_RETRIEVAL > 0
    
    def test_validate_config(self):
        """Test configuration validation."""
        from app.config import validate_config
        
        errors = validate_config()
        # May have errors if API key not set, but should not crash
        assert isinstance(errors, list)


# Sample questions for integration testing
SAMPLE_QUESTIONS = [
    "What was Swiggy's revenue in FY 2023-24?",
    "What are Swiggy's main business segments?",
    "Who is the CEO of Swiggy?",
    "What risks does Swiggy face?",
    "What is Swiggy's growth strategy?",
]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
