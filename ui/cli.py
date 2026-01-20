"""
CLI Interface for RAG Application.
Simple command-line interface for testing.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import PDF_PATH, validate_config
from app.document_processor import DocumentProcessor
from app.embeddings import GeminiEmbeddings
from app.vector_store import FAISSVectorStore
from app.langgraph_agent import RAGAgent


def initialize_system():
    """Initialize or load the RAG system."""
    print("🔧 Initializing RAG system...")
    
    # Check for existing index
    vector_store = FAISSVectorStore(3072)
    
    if vector_store.exists():
        print("📂 Loading existing vector store...")
        vector_store = FAISSVectorStore.load()
    else:
        print("📄 Processing PDF document...")
        processor = DocumentProcessor()
        chunks = processor.process_document(PDF_PATH)
        
        print("🔢 Generating embeddings...")
        embeddings = GeminiEmbeddings()
        chunk_texts = [c.content for c in chunks]
        embeddings_array = embeddings.embed_documents(chunk_texts)
        
        print("💾 Saving vector store...")
        vector_store.add_chunks(chunks, embeddings_array)
        vector_store.save()
    
    # Initialize agent
    print("🤖 Initializing RAG agent...")
    embeddings = GeminiEmbeddings()
    agent = RAGAgent(vector_store, embeddings)
    
    print("✅ System ready!\n")
    return agent


def print_result(result: dict):
    """Pretty print a query result."""
    print("\n" + "=" * 60)
    print("📝 ANSWER:")
    print("-" * 60)
    print(result["answer"])
    print("-" * 60)
    
    # Confidence
    confidence = result.get("confidence", 0)
    bar_length = int(confidence * 20)
    bar = "█" * bar_length + "░" * (20 - bar_length)
    print(f"\n📊 Confidence: [{bar}] {int(confidence * 100)}%")
    
    # Query enhancement
    if result.get("query_type") == "UNCLEAR":
        print(f"\n🔍 Query was enhanced: {result.get('proxy_answer', '')[:100]}...")
    
    # Citations
    citations = result.get("citations", [])
    if citations:
        print(f"\n📚 Sources ({len(citations)}):")
        for c in citations[:5]:
            print(f"  • Page {c['page']}: {c['section']}")
    
    print("=" * 60)


def main():
    """Main CLI entry point."""
    # Validate configuration
    errors = validate_config()
    if errors:
        print("⚠️ Configuration Error:")
        for error in errors:
            print(f"  - {error}")
        print("\nPlease fix these issues and try again.")
        return
    
    # Initialize
    agent = initialize_system()
    
    # Conversation history
    history = []
    
    print("💬 Swiggy Annual Report Q&A")
    print("Type 'quit' to exit, 'clear' to reset history\n")
    
    while True:
        try:
            query = input("You: ").strip()
            
            if not query:
                continue
            
            if query.lower() == 'quit':
                print("👋 Goodbye!")
                break
            
            if query.lower() == 'clear':
                history = []
                print("🗑️ Conversation history cleared.\n")
                continue
            
            # Run query
            result = agent.run(query, history)
            
            # Print result
            print_result(result)
            
            # Update history
            history.append({"role": "user", "content": query})
            history.append({"role": "assistant", "content": result["answer"]})
            
            print()
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


if __name__ == "__main__":
    main()
