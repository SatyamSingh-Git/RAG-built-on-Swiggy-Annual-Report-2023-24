"""
LangGraph Agent Module.
Implements agentic RAG with query classification and proxy answer generation.
"""

from typing import TypedDict, Literal, List, Dict, Any, Optional
from dataclasses import dataclass
import google.generativeai as genai
from langgraph.graph import StateGraph, END

import app.config as config
from app.embeddings import GeminiEmbeddings
from app.vector_store import FAISSVectorStore
from app.retriever import HybridRetriever, RetrievalResult
from app.generator import GeminiGenerator, GenerationResult


class AgentState(TypedDict):
    """State passed through the LangGraph agent."""
    original_query: str
    processed_query: str
    query_type: str  # "CLEAR" or "UNCLEAR"
    proxy_answer: str
    retrieval_results: List[RetrievalResult]
    context: str
    generation_result: Optional[GenerationResult]
    conversation_history: List[Dict[str, str]]
    iteration: int


class RAGAgent:
    """
    LangGraph-based RAG agent with intelligent query handling.
    
    Flow:
    1. Classify query (clear vs unclear)
    2. For unclear queries, generate proxy answer
    3. Retrieve using enhanced query
    4. Evaluate retrieval quality
    5. Optionally iterate if quality is low
    6. Generate final answer
    """
    
    def __init__(
        self,
        vector_store: FAISSVectorStore,
        embeddings: Optional[GeminiEmbeddings] = None,
        max_iterations: int = 2
    ):
        """
        Initialize RAG agent.
        
        Args:
            vector_store: Initialized FAISS vector store
            embeddings: Gemini embeddings instance
            max_iterations: Maximum retrieval iterations
        """
        self.vector_store = vector_store
        self.embeddings = embeddings or GeminiEmbeddings()
        self.max_iterations = max_iterations
        
        # Initialize components
        self.retriever = HybridRetriever(vector_store, self.embeddings)
        self.generator = GeminiGenerator()
        
        # Configure Gemini for classification/proxy
        genai.configure(api_key=config.GOOGLE_API_KEY)
        self.llm = genai.GenerativeModel(config.GEMINI_MODEL)
        
        # Build the graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("classify_query", self._classify_query)
        workflow.add_node("generate_proxy", self._generate_proxy_answer)
        workflow.add_node("retrieve", self._retrieve_context)
        workflow.add_node("evaluate_retrieval", self._evaluate_retrieval)
        workflow.add_node("generate_answer", self._generate_answer)
        
        # Set entry point
        workflow.set_entry_point("classify_query")
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "classify_query",
            self._route_after_classify,
            {
                "generate_proxy": "generate_proxy",
                "retrieve": "retrieve"
            }
        )
        
        workflow.add_edge("generate_proxy", "retrieve")
        workflow.add_edge("retrieve", "evaluate_retrieval")
        
        workflow.add_conditional_edges(
            "evaluate_retrieval",
            self._route_after_evaluate,
            {
                "generate_answer": "generate_answer",
                "retrieve": "retrieve"
            }
        )
        
        workflow.add_edge("generate_answer", END)
        
        return workflow.compile()
    
    def _classify_query(self, state: AgentState) -> AgentState:
        """Classify query as clear or unclear."""
        query = state["original_query"]
        
        prompt = config.QUERY_CLASSIFIER_PROMPT.format(query=query)
        response = self.llm.generate_content(prompt)
        
        classification = response.text.strip().upper()
        query_type = "CLEAR" if "CLEAR" in classification else "UNCLEAR"
        
        return {
            **state,
            "query_type": query_type,
            "processed_query": query
        }
    
    def _generate_proxy_answer(self, state: AgentState) -> AgentState:
        """Generate a proxy answer for unclear queries."""
        query = state["original_query"]
        
        prompt = config.PROXY_ANSWER_PROMPT.format(query=query)
        response = self.llm.generate_content(prompt)
        
        proxy_answer = response.text.strip()
        
        # Combine original query with proxy for better retrieval
        enhanced_query = f"{query} {proxy_answer}"
        
        return {
            **state,
            "proxy_answer": proxy_answer,
            "processed_query": enhanced_query
        }
    
    def _retrieve_context(self, state: AgentState) -> AgentState:
        """Retrieve relevant context."""
        query = state["processed_query"]
        
        results = self.retriever.retrieve(query)
        context = self.retriever.get_context_string(results)
        
        return {
            **state,
            "retrieval_results": results,
            "context": context,
            "iteration": state.get("iteration", 0) + 1
        }
    
    def _evaluate_retrieval(self, state: AgentState) -> AgentState:
        """Evaluate retrieval quality (passthrough for now)."""
        # Could add quality checks here
        return state
    
    def _generate_answer(self, state: AgentState) -> AgentState:
        """Generate the final answer."""
        result = self.generator.generate(
            query=state["original_query"],
            context=state["context"],
            retrieval_results=state["retrieval_results"],
            conversation_history=state.get("conversation_history", [])
        )
        
        return {
            **state,
            "generation_result": result
        }
    
    def _route_after_classify(self, state: AgentState) -> Literal["generate_proxy", "retrieve"]:
        """Route based on query classification."""
        if state["query_type"] == "UNCLEAR":
            return "generate_proxy"
        return "retrieve"
    
    def _route_after_evaluate(self, state: AgentState) -> Literal["generate_answer", "retrieve"]:
        """Route based on retrieval quality evaluation."""
        # Check if we need to iterate
        iteration = state.get("iteration", 1)
        results = state.get("retrieval_results", [])
        
        # If poor results and haven't hit max iterations, retry
        if results and iteration < self.max_iterations:
            avg_score = sum(r.rerank_score or r.combined_score for r in results[:3]) / 3
            if avg_score < 0.3:  # Low quality threshold
                return "retrieve"
        
        return "generate_answer"
    
    def run(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Run the RAG agent.
        
        Args:
            query: User query
            conversation_history: Previous conversation turns
            
        Returns:
            Dictionary with answer, citations, confidence, etc.
        """
        initial_state: AgentState = {
            "original_query": query,
            "processed_query": "",
            "query_type": "",
            "proxy_answer": "",
            "retrieval_results": [],
            "context": "",
            "generation_result": None,
            "conversation_history": conversation_history or [],
            "iteration": 0
        }
        
        # Run the graph
        final_state = self.graph.invoke(initial_state)
        
        # Extract results
        gen_result = final_state.get("generation_result")
        
        if gen_result:
            return {
                "answer": gen_result.answer,
                "citations": gen_result.citations,
                "confidence": gen_result.confidence,
                "sources_used": gen_result.sources_used,
                "query_type": final_state["query_type"],
                "proxy_answer": final_state.get("proxy_answer", ""),
                "iterations": final_state.get("iteration", 1)
            }
        
        return {
            "answer": "Unable to generate an answer. Please try rephrasing your question.",
            "citations": [],
            "confidence": 0.0,
            "sources_used": 0,
            "query_type": final_state["query_type"],
            "proxy_answer": final_state.get("proxy_answer", ""),
            "iterations": final_state.get("iteration", 1)
        }


# CLI for testing
if __name__ == "__main__":
    print("LangGraph RAG Agent module loaded")
    print("Initialize with a vector store to use")
