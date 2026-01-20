"""
Generator Module.
Handles answer generation using Gemini with grounding and citations.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import re
import google.generativeai as genai

import app.config as config
from app.retriever import RetrievalResult
from app.utils import get_fallback_chain, init_gemini_model


@dataclass
class GenerationResult:
    """Result of answer generation."""
    answer: str
    citations: List[Dict[str, Any]]
    confidence: float
    sources_used: int
    raw_response: str


class GeminiGenerator:
    """
    Answer generator using Google's Gemini LLM.
    
    Features:
    - Grounded generation with strict context adherence
    - Automatic citation extraction
    - Confidence scoring based on context relevance
    - Hallucination prevention prompts
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = config.GEMINI_MODEL
    ):
        """
        Initialize Gemini generator with fallback support.
        """
        self.api_key = api_key or config.GOOGLE_API_KEY
        self.primary_model_name = model
        self.fallback_chain = get_fallback_chain(self.primary_model_name)
        
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY is required")
        
        genai.configure(api_key=self.api_key)
        self.current_model_idx = 0
        self._init_current_model()

    def _init_current_model(self):
        """Initialize the model based on current index in fallback chain."""
        self.model_name = self.fallback_chain[self.current_model_idx]
        self.model, self.supports_system_instruction = init_gemini_model(
            self.model_name, 
            config.SYSTEM_PROMPT
        )

    def generate(
        self,
        query: str,
        context: str,
        retrieval_results: List[RetrievalResult],
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> GenerationResult:
        """
        Generate answer with automatic model fallback.
        """
        while self.current_model_idx < len(self.fallback_chain):
            prompt = self._build_prompt(query, context, conversation_history)
            
            try:
                print(f"DEBUG: Attempting generation with {self.model_name} (System Instruct: {self.supports_system_instruction})...")
                response = self.model.generate_content(prompt)
                raw_answer = response.text
                
                # Success!
                citations = self._extract_citations(raw_answer, retrieval_results)
                confidence = self._calculate_confidence(retrieval_results, raw_answer)
                clean_answer = self._format_answer(raw_answer)
                
                return GenerationResult(
                    answer=clean_answer,
                    citations=citations,
                    confidence=confidence,
                    sources_used=len(citations),
                    raw_response=raw_answer
                )
                
            except Exception as e:
                error_str = str(e)
                print(f"DEBUG: Model {self.model_name} failed: {error_str}")
                
                # Check for specific instruction error
                if ("Developer instruction" in error_str or "400" in error_str) and self.supports_system_instruction:
                    print(f"DEBUG: Retrying {self.model_name} without system instructions...")
                    self.supports_system_instruction = False
                    self.model, _ = init_gemini_model(self.model_name, None)
                    continue # Retry the same model immediately without system instruction
                
                # Try next model in chain
                self.current_model_idx += 1
                if self.current_model_idx < len(self.fallback_chain):
                    print(f"DEBUG: Switching to fallback model: {self.fallback_chain[self.current_model_idx]}...")
                    self._init_current_model()
                else:
                    raise e
        
        raise RuntimeError("All models in the fallback chain failed.")

    def _build_prompt(
        self,
        query: str,
        context: str,
        history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """Build the generation prompt."""
        prompt_parts = []
        
        # Inject system instructions into prompt if model doesn't support the parameter
        if not getattr(self, 'supports_system_instruction', False):
            prompt_parts.append(f"INSTRUCTIONS:\n{config.SYSTEM_PROMPT}\n")
            prompt_parts.append("-" * 30 + "\n")
        
        # Add conversation history if present
        if history:
            prompt_parts.append("Previous conversation:")
            for msg in history[-5:]: # Keep last 5 turns
                role = "User" if msg["role"] == "user" else "Assistant"
                prompt_parts.append(f"{role}: {msg['content']}")
            prompt_parts.append("-" * 20)
        
        # Add context
        prompt_parts.append(f"CONTEXT FROM SWIGGY ANNUAL REPORT:\n{context}\n")
        prompt_parts.append("-" * 30 + "\n")
        
        # Add user query
        prompt_parts.append(f"QUESTION: {query}\n\nFINAL ANSWER:")
        
        return "\n".join(prompt_parts)
        if history:
            prompt_parts.append("Previous conversation:")
            for turn in history[-5:]:  # Last 5 turns
                role = turn.get("role", "user")
                content = turn.get("content", "")
                prompt_parts.append(f"{role.upper()}: {content}")
            prompt_parts.append("")
        
        # Add context
        prompt_parts.append("CONTEXT FROM SWIGGY ANNUAL REPORT:")
        prompt_parts.append("=" * 50)
        prompt_parts.append(context)
        prompt_parts.append("=" * 50)
        prompt_parts.append("")
        
        # Add query
        prompt_parts.append(f"USER QUESTION: {query}")
        prompt_parts.append("")
        prompt_parts.append("Please provide a comprehensive answer based ONLY on the context above.")
        prompt_parts.append("Include specific page references for your claims.")
        
        return "\n".join(prompt_parts)
    
    def _extract_citations(
        self,
        answer: str,
        results: List[RetrievalResult]
    ) -> List[Dict[str, Any]]:
        """Extract page citations from answer and match to chunks."""
        citations = []
        
        # Find page references in answer
        page_pattern = r'(?:page|pg\.?)\s*(\d+)'
        mentioned_pages = set(int(m) for m in re.findall(page_pattern, answer.lower()))
        
        # Match to retrieval results
        for result in results:
            chunk = result.chunk
            if chunk.page_number in mentioned_pages:
                citations.append({
                    "page": chunk.page_number,
                    "section": chunk.section,
                    "relevance_score": result.rerank_score or result.combined_score,
                    "excerpt": chunk.content  # Return full content
                })
        
        # Also include top results even if not explicitly cited
        for result in results[:3]:
            chunk = result.chunk
            if chunk.page_number not in mentioned_pages:
                citations.append({
                    "page": chunk.page_number,
                    "section": chunk.section,
                    "relevance_score": result.rerank_score or result.combined_score,
                    "excerpt": chunk.content,  # Return full content
                    "implicit": True
                })
        
        # Deduplicate by page
        seen_pages = set()
        unique_citations = []
        for c in citations:
            if c["page"] not in seen_pages:
                unique_citations.append(c)
                seen_pages.add(c["page"])
        
        return unique_citations
    
    def _calculate_confidence(
        self,
        results: List[RetrievalResult],
        answer: str
    ) -> float:
        """
        Calculate confidence score based on retrieval quality and answer characteristics.
        
        Factors:
        - Average retrieval score
        - Whether answer admits uncertainty
        - Number of specific citations
        """
        if not results:
            return 0.0
        
        # Base confidence from retrieval scores
        avg_score = sum(r.rerank_score or r.combined_score for r in results[:3]) / 3
        base_confidence = min(avg_score, 1.0)
        
        # Penalty for uncertainty markers
        uncertainty_markers = [
            "cannot find", "not mentioned", "unclear",
            "not in the report", "no information", "i don't"
        ]
        answer_lower = answer.lower()
        if any(marker in answer_lower for marker in uncertainty_markers):
            base_confidence *= 0.5
        
        # Bonus for specific citations
        citation_count = len(re.findall(r'page\s*\d+', answer_lower))
        citation_bonus = min(citation_count * 0.05, 0.2)
        
        final_confidence = min(base_confidence + citation_bonus, 1.0)
        return round(final_confidence, 2)
    
    def _format_answer(self, answer: str) -> str:
        """Format and clean the answer."""
        # Remove excessive whitespace
        answer = re.sub(r'\n{3,}', '\n\n', answer)
        
        # Ensure proper formatting
        answer = answer.strip()
        
        return answer
    
    def compare_values(
        self,
        query: str,
        context: str,
        entities: List[str]
    ) -> str:
        """
        Generate a comparative analysis for specific entities.
        
        Used for queries like "Compare revenue across segments".
        """
        comparison_prompt = f"""
Based on the following context, create a comparison table or analysis for: {', '.join(entities)}

CONTEXT:
{context}

QUERY: {query}

Provide a structured comparison with:
1. Key metrics for each entity
2. Notable differences
3. Relevant page references

If data is not available for comparison, clearly state what information is missing.
"""
        
        response = self.model.generate_content(comparison_prompt)
        return response.text


# CLI for testing
if __name__ == "__main__":
    generator = GeminiGenerator()
    print(f"Initialized generator with model: {generator.model_name}")
