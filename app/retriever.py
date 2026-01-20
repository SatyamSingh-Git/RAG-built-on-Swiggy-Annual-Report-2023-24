"""
Retriever Module.
Implements hybrid retrieval (dense + BM25) with cross-encoder reranking.
"""

from typing import List, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from app.document_processor import Chunk
from app.embeddings import GeminiEmbeddings
from app.vector_store import FAISSVectorStore
from app.config import TOP_K_RETRIEVAL, TOP_K_RERANK


@dataclass
class RetrievalResult:
    """Result of retrieval with scoring details."""
    chunk: Chunk
    dense_score: float
    sparse_score: float
    rerank_score: float
    combined_score: float


class HybridRetriever:
    """
    Hybrid retriever combining dense (semantic) and sparse (BM25) search.
    
    Features:
    - Dense retrieval via FAISS with Gemini embeddings
    - Sparse retrieval via BM25 for keyword matching
    - Cross-encoder reranking to handle "lost in the middle" problem
    - Configurable fusion weights
    """
    
    def __init__(
        self,
        vector_store: FAISSVectorStore,
        embeddings: GeminiEmbeddings,
        dense_weight: float = 0.6,
        sparse_weight: float = 0.4,
        use_reranker: bool = True
    ):
        """
        Initialize hybrid retriever.
        
        Args:
            vector_store: FAISS vector store with chunks
            embeddings: Gemini embeddings instance
            dense_weight: Weight for dense retrieval scores
            sparse_weight: Weight for sparse (BM25) retrieval scores
            use_reranker: Whether to use cross-encoder reranking
        """
        self.vector_store = vector_store
        self.embeddings = embeddings
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.use_reranker = use_reranker
        
        # Build BM25 index
        self._build_bm25_index()
        
        # Load reranker if enabled
        if use_reranker:
            print("Loading cross-encoder reranker...")
            self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        else:
            self.reranker = None
    
    def _build_bm25_index(self) -> None:
        """Build BM25 index from chunks."""
        # Tokenize documents
        tokenized_docs = [
            chunk.content.lower().split()
            for chunk in self.vector_store.chunks
        ]
        self.bm25 = BM25Okapi(tokenized_docs)
        print(f"Built BM25 index with {len(tokenized_docs)} documents")
    
    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K_RETRIEVAL,
        rerank_top_k: int = TOP_K_RERANK
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant chunks using hybrid search.
        
        Args:
            query: User query
            top_k: Number of candidates to retrieve
            rerank_top_k: Number of final results after reranking
            
        Returns:
            List of RetrievalResult sorted by relevance
        """
        # Dense retrieval
        query_embedding = self.embeddings.embed_query(query)
        dense_results = self.vector_store.search(query_embedding, top_k=top_k * 2)
        
        # Sparse retrieval
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        # Create score lookup
        dense_scores = {chunk.id: score for chunk, score in dense_results}
        
        # Combine scores for all chunks
        candidates = {}
        
        for chunk, dense_score in dense_results:
            idx = self.vector_store.id_to_idx.get(chunk.id, 0)
            sparse_score = bm25_scores[idx] if idx < len(bm25_scores) else 0
            
            # Normalize sparse score
            max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1
            sparse_score_norm = sparse_score / max_bm25
            
            combined = (
                self.dense_weight * dense_score +
                self.sparse_weight * sparse_score_norm
            )
            
            candidates[chunk.id] = RetrievalResult(
                chunk=chunk,
                dense_score=dense_score,
                sparse_score=sparse_score_norm,
                rerank_score=0.0,
                combined_score=combined
            )
        
        # Sort by combined score
        sorted_candidates = sorted(
            candidates.values(),
            key=lambda x: x.combined_score,
            reverse=True
        )[:top_k]
        
        # Rerank if enabled
        if self.use_reranker and self.reranker and len(sorted_candidates) > 0:
            sorted_candidates = self._rerank(query, sorted_candidates, rerank_top_k)
        else:
            sorted_candidates = sorted_candidates[:rerank_top_k]
        
        return sorted_candidates
    
    def _rerank(
        self,
        query: str,
        candidates: List[RetrievalResult],
        top_k: int
    ) -> List[RetrievalResult]:
        """
        Rerank candidates using cross-encoder.
        
        This addresses the "lost in the middle" problem by ensuring
        the most relevant chunks are properly ranked.
        """
        # Prepare pairs for reranking
        pairs = [(query, c.chunk.content) for c in candidates]
        
        # Get reranker scores
        rerank_scores = self.reranker.predict(pairs)
        
        # Update results with rerank scores
        for i, result in enumerate(candidates):
            result.rerank_score = float(rerank_scores[i])
        
        # Sort by rerank score
        reranked = sorted(candidates, key=lambda x: x.rerank_score, reverse=True)
        
        # Return top_k with optimal ordering (best at start and end)
        final = reranked[:top_k]
        
        # Reorder to address "lost in the middle": best at start and end
        if len(final) > 2:
            final = self._optimal_chunk_ordering(final)
        
        return final
    
    def _optimal_chunk_ordering(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """
        Reorder chunks to place most relevant at start and end.
        
        LLMs tend to pay more attention to the beginning and end of context,
        so we interleave the best chunks at these positions.
        """
        if len(results) <= 2:
            return results
        
        sorted_by_score = sorted(results, key=lambda x: x.rerank_score, reverse=True)
        
        # Interleave: best at start, second best at end, etc.
        reordered = []
        left = []
        right = []
        
        for i, result in enumerate(sorted_by_score):
            if i % 2 == 0:
                left.append(result)
            else:
                right.append(result)
        
        reordered = left + list(reversed(right))
        return reordered
    
    def get_context_string(
        self,
        results: List[RetrievalResult],
        max_tokens: int = 4000
    ) -> str:
        """
        Build context string from retrieval results.
        
        Args:
            results: List of retrieval results
            max_tokens: Maximum context length (approximate)
            
        Returns:
            Formatted context string with citations
        """
        context_parts = []
        current_length = 0
        
        for result in results:
            chunk = result.chunk
            
            # Create formatted chunk with citation
            chunk_text = f"[Page {chunk.page_number}] [{chunk.section}]\n{chunk.content}\n"
            
            # Check length (rough token estimate: 4 chars per token)
            chunk_tokens = len(chunk_text) // 4
            if current_length + chunk_tokens > max_tokens:
                break
            
            context_parts.append(chunk_text)
            current_length += chunk_tokens
        
        return "\n---\n".join(context_parts)


# CLI for testing
if __name__ == "__main__":
    print("Hybrid Retriever module loaded successfully")
    print("Use with initialized vector store and embeddings")
