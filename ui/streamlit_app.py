"""
Streamlit Web Application.
Premium UI for RAG-based Q&A on Swiggy Annual Report.
"""

import streamlit as st
from pathlib import Path
import sys
import json
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import PDF_PATH, VECTORSTORE_DIR, validate_config
from app.document_processor import DocumentProcessor
from app.embeddings import GeminiEmbeddings
from app.vector_store import FAISSVectorStore
from app.langgraph_agent import RAGAgent


# Page configuration
st.set_page_config(
    page_title="Swiggy Annual Report Q&A",
    page_icon="🍔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium look
st.markdown("""
<style>
    /* Main theme */
    :root {
        --primary: #FC8019;
        --secondary: #1F1F2E;
        --accent: #FF6B35;
        --background: #0E0E16;
        --surface: #1A1A2E;
        --text: #FFFFFF;
        --text-muted: #A0A0B0;
    }
    
    /* Global styles */
    .stApp {
        background: linear-gradient(135deg, #0E0E16 0%, #1A1A2E 100%);
    }
    
    /* Header */
    .main-header {
        background: linear-gradient(90deg, #FC8019 0%, #FF6B35 100%);
        padding: 1.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(252, 128, 25, 0.3);
    }
    
    .main-header h1 {
        color: white;
        margin: 0;
        font-weight: 700;
    }
    
    .main-header p {
        color: rgba(255,255,255,0.9);
        margin: 0.5rem 0 0 0;
    }
    
    /* Chat messages */
    .user-message {
        background: linear-gradient(135deg, #2D2D44 0%, #1F1F2E 100%);
        padding: 1rem 1.5rem;
        border-radius: 16px 16px 4px 16px;
        margin: 1rem 0;
        border-left: 4px solid #FC8019;
    }
    
    .assistant-message {
        background: linear-gradient(135deg, #1A1A2E 0%, #16162A 100%);
        padding: 1.5rem;
        border-radius: 16px 16px 16px 4px;
        margin: 1rem 0;
        border: 1px solid rgba(252, 128, 25, 0.2);
    }
    
    /* Confidence meter */
    .confidence-meter {
        background: #1F1F2E;
        border-radius: 20px;
        height: 8px;
        overflow: hidden;
        margin: 0.5rem 0;
    }
    
    .confidence-fill {
        height: 100%;
        border-radius: 20px;
        transition: width 0.5s ease;
    }
    
    .confidence-high { background: linear-gradient(90deg, #10B981, #34D399); }
    .confidence-medium { background: linear-gradient(90deg, #F59E0B, #FBBF24); }
    .confidence-low { background: linear-gradient(90deg, #EF4444, #F87171); }
    
    /* Citations */
    .citation-card {
        background: #1F1F2E;
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        border: 1px solid rgba(255,255,255,0.1);
        transition: all 0.3s ease;
    }
    
    .citation-card:hover {
        border-color: #FC8019;
        transform: translateX(4px);
    }
    
    .page-badge {
        background: #FC8019;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    
    /* Sidebar */
    .sidebar-stat {
        background: linear-gradient(135deg, #2D2D44 0%, #1F1F2E 100%);
        padding: 1rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        text-align: center;
    }
    
    .stat-value {
        font-size: 2rem;
        font-weight: 700;
        color: #FC8019;
    }
    
    .stat-label {
        color: #A0A0B0;
        font-size: 0.85rem;
    }
    
    /* Input styling */
    .stTextInput > div > div > input {
        background: #1F1F2E !important;
        border: 2px solid rgba(252, 128, 25, 0.3) !important;
        border-radius: 12px !important;
        color: white !important;
        padding: 1rem !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #FC8019 !important;
        box-shadow: 0 0 0 2px rgba(252, 128, 25, 0.2) !important;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #FC8019 0%, #FF6B35 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 20px rgba(252, 128, 25, 0.4) !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: #1F1F2E !important;
        border-radius: 12px !important;
    }
    
    /* Suggested questions */
    .suggestion-chip {
        background: rgba(252, 128, 25, 0.1);
        border: 1px solid rgba(252, 128, 25, 0.3);
        color: #FC8019;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        margin: 0.25rem;
        cursor: pointer;
        display: inline-block;
        transition: all 0.3s ease;
    }
    
    .suggestion-chip:hover {
        background: rgba(252, 128, 25, 0.2);
    }
    
    /* Citation Card */
    .citation-card {
        background: #1F1F2E;
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        border: 1px solid rgba(255,255,255,0.1);
        transition: all 0.3s ease;
    }
    
    .citation-card:hover {
        border-color: #FC8019;
        transform: translateX(4px);
    }
    
    .page-badge {
        background: #FC8019;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []  # Full history for RAG context
    if "current_qa" not in st.session_state:
        st.session_state.current_qa = None  # Only the current Q&A to display
    if "agent" not in st.session_state:
        st.session_state.agent = None
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "initialized" not in st.session_state:
        st.session_state.initialized = False
    if "total_queries" not in st.session_state:
        st.session_state.total_queries = 0


def load_or_create_index():
    """Load existing vector store or create new one."""
    with st.spinner("🔧 Initializing RAG system..."):
        # Check for existing index
        if FAISSVectorStore(3072).exists():
            st.session_state.vector_store = FAISSVectorStore.load()
            st.success("✅ Loaded existing vector store")
        else:
            # Process document and create index
            st.info("📄 Processing PDF document...")
            processor = DocumentProcessor()
            chunks = processor.process_document(PDF_PATH)
            
            st.info("🔢 Generating embeddings...")
            embeddings = GeminiEmbeddings()
            chunk_texts = [c.content for c in chunks]
            embeddings_array = embeddings.embed_documents(chunk_texts)
            
            st.info("💾 Saving vector store...")
            st.session_state.vector_store = FAISSVectorStore()
            st.session_state.vector_store.add_chunks(chunks, embeddings_array)
            st.session_state.vector_store.save()
            
            st.success("✅ Created and saved vector store")
        
        # Initialize agent
        embeddings = GeminiEmbeddings()
        st.session_state.agent = RAGAgent(
            st.session_state.vector_store,
            embeddings
        )
        st.session_state.initialized = True


def render_header():
    """Render the main header."""
    st.markdown("""
    <div class="main-header">
        <h1>🍔 Swiggy Annual Report Q&A</h1>
        <p>Ask questions about Swiggy's FY 2023-24 Annual Report powered by AI</p>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar():
    """Render the sidebar with stats and options."""
    with st.sidebar:
        st.markdown("### 📊 Session Stats")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="sidebar-stat">
                <div class="stat-value">{st.session_state.total_queries}</div>
                <div class="stat-label">Queries</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            chunks = len(st.session_state.vector_store.chunks) if st.session_state.vector_store else 0
            st.markdown(f"""
            <div class="sidebar-stat">
                <div class="stat-value">{chunks}</div>
                <div class="stat-label">Chunks</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("### 💡 Suggested Questions")
        suggestions = [
            "What was Swiggy's total revenue in FY 2023-24?",
            "What are Swiggy's main business segments?",
            "What risks does the company face?",
            "Who are the key management personnel?",
            "What is Swiggy's growth strategy?",
        ]
        
        for suggestion in suggestions:
            if st.button(suggestion, key=f"sug_{suggestion[:20]}", use_container_width=True):
                return suggestion
        
        st.markdown("---")
        
        st.markdown("### ⚙️ Settings")
        
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.session_state.current_qa = None
            st.rerun()
        
        if st.button("🔄 Rebuild Index", use_container_width=True):
            # Delete existing index
            import shutil
            if VECTORSTORE_DIR.exists():
                shutil.rmtree(VECTORSTORE_DIR)
            st.session_state.initialized = False
            st.session_state.vector_store = None
            st.rerun()
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown("""
        This RAG application uses:
        - **Gemini 2.0 Flash** for generation
        - **gemini-embedding-001** for embeddings
        - **FAISS** for vector search
        - **LangGraph** for orchestration
        """)
    
    return None


def render_confidence_meter(confidence: float):
    """Render a visual confidence meter."""
    percentage = int(confidence * 100)
    
    if confidence >= 0.7:
        color_class = "confidence-high"
        label = "High Confidence"
    elif confidence >= 0.4:
        color_class = "confidence-medium"
        label = "Medium Confidence"
    else:
        color_class = "confidence-low"
        label = "Low Confidence"
    
    st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 1rem; margin: 1rem 0;">
        <span style="color: #A0A0B0; font-size: 0.85rem;">{label}</span>
        <div class="confidence-meter" style="flex: 1;">
            <div class="confidence-fill {color_class}" style="width: {percentage}%;"></div>
        </div>
        <span style="color: white; font-weight: 600;">{percentage}%</span>
    </div>
    """, unsafe_allow_html=True)


def render_citations(citations: list):
    """Render citation cards with native expander functionality."""
    import html
    if not citations:
        return
    
    st.markdown("### 📚 Source Citations")
    
    for idx, citation in enumerate(citations):
        implicit = citation.get("implicit", False)
        # Clean and escape the excerpt
        excerpt = citation.get('excerpt', '')
        excerpt_clean = html.escape(str(excerpt))
        excerpt_preview = excerpt_clean[:200] + "..." if len(excerpt_clean) > 200 else excerpt_clean
        
        # We render the card using columns to control layout precisely
        st.markdown(f"""
        <div class="citation-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                <span class="page-badge">Page {citation['page']}</span>
                <span style="color: #A0A0B0; font-size: 0.8rem;">{html.escape(str(citation['section']))}</span>
            </div>
            <p style="color: #E0E0E0; font-size: 0.9rem; margin: 0; margin-bottom: 0.5rem;">
                {excerpt_preview}
            </p>
            <div style="margin-bottom: 0.5rem;">
                <span style="color: #FC8019; font-size: 0.75rem;">
                    Relevance: {citation['relevance_score']:.2f}
                    {' (Related context)' if implicit else ''}
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Native expander immediately following the card, styled to look integrated
        with st.expander("Show Full Chunk"):
            st.markdown(f"""
            <div style="background: #1A1A2E; padding: 1rem; border-radius: 8px; border: 1px solid rgba(255,255,255,0.1);">
                <p style="color: #D0D0D0; font-size: 0.9rem; line-height: 1.6; margin: 0;">
                    {excerpt_clean}
                </p>
            </div>
            """, unsafe_allow_html=True)


def process_query(query: str):
    """Process a user query and generate response."""
    st.session_state.total_queries += 1
    
    # Get conversation history for context (before adding new message)
    history = [
        {"role": msg["role"], "content": msg["content"]}
        for msg in st.session_state.messages
    ]
    
    # Run agent
    with st.spinner("🤔 Thinking..."):
        result = st.session_state.agent.run(query, history)
    
    # Store in history for future context
    st.session_state.messages.append({
        "role": "user",
        "content": query,
        "timestamp": datetime.now().isoformat()
    })
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "timestamp": datetime.now().isoformat()
    })
    
    # Store current Q&A for display (only this will be shown)
    st.session_state.current_qa = {
        "question": query,
        "answer": result["answer"],
        "citations": result.get("citations", []),
        "confidence": result.get("confidence", 0.0),
        "query_type": result.get("query_type", "CLEAR"),
        "proxy_answer": result.get("proxy_answer", ""),
        "timestamp": datetime.now().isoformat()
    }


def render_chat():
    """Render the current Q&A only (not full history)."""
    qa = st.session_state.current_qa
    
    if qa is None:
        # Show welcome message if no query yet
        st.markdown("""
        <div class="assistant-message">
            <strong>🤖 Assistant:</strong>
            <p>Welcome! Ask me any question about the Swiggy Annual Report FY 2023-24. 
            I'll provide accurate answers with source citations.</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Show the current question
    st.markdown(f"""
    <div class="user-message">
        <strong>You:</strong> {qa['question']}
    </div>
    """, unsafe_allow_html=True)
    
    # Show the current answer
    st.markdown(f"""
    <div class="assistant-message">
        <strong>🤖 Assistant:</strong>
        <p>{qa['answer']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Show confidence meter
    render_confidence_meter(qa["confidence"])
    
    # Show query enhancement info
    if qa.get("query_type") == "UNCLEAR" and qa.get("proxy_answer"):
        with st.expander("🔍 Query Enhancement"):
            st.info(f"Your query was enhanced for better retrieval:\n\n_{qa['proxy_answer']}_")
    
    # Show citations
    if qa.get("citations"):
        render_citations(qa["citations"])


def main():
    """Main application entry point."""
    init_session_state()
    
    # Render header
    render_header()
    
    # Config validation
    config_errors = validate_config()
    if config_errors:
        st.error("⚠️ Configuration Error")
        for error in config_errors:
            st.error(error)
            
        # --- DEBUG SECTION ---
        with st.expander("🕵️ Debug Information (Click me if you are stuck)", expanded=True):
            st.warning("Debugging Secrets & Keys")
            
            # Check Streamlit Secrets
            try:
                secrets_keys = list(st.secrets.keys())
                st.write(f"**Available Secret Keys:** `{secrets_keys}`")
                
                if "gemini" in st.secrets:
                    st.write(f"**Found [gemini] section keys:** `{list(st.secrets['gemini'].keys())}`")
                    
            except Exception as e:
                st.write(f"❌ Error reading secrets: {e}")
            
            # Check Env Vars (safely)
            import os
            has_env_key = "GOOGLE_API_KEY" in os.environ
            st.write(f"**Has GOOGLE_API_KEY in os.environ:** `{has_env_key}`")
            
            from app import config
            st.write(f"**App Config sees Key:** `{bool(config.GOOGLE_API_KEY)}`")
            if config.GOOGLE_API_KEY:
                st.write(f"**Key Length:** `{len(config.GOOGLE_API_KEY)}`")
            else:
                st.write("**Key is Empty/None**")
                
        st.info("Please check your `.env` file and ensure the PDF is in the correct location.")
        st.stop()
    
    # Initialize if needed
    if not st.session_state.initialized:
        load_or_create_index()
    
    # Render sidebar and get any selected suggestion
    selected_suggestion = render_sidebar()
    
    # Main chat area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Query input at the TOP
        query = st.text_input(
            "Ask a question about the Swiggy Annual Report",
            placeholder="e.g., What was Swiggy's revenue growth?",
            key="query_input",
            label_visibility="collapsed"
        )
        
        # Handle suggestion selection
        if selected_suggestion:
            query = selected_suggestion
        
        col_a, col_b, col_c = st.columns([1, 1, 4])
        with col_a:
            send_clicked = st.button("🚀 Send", use_container_width=True)
        with col_b:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.messages = []
                st.session_state.current_qa = None
                st.rerun()
        
        if (send_clicked or selected_suggestion) and query:
            process_query(query)
            st.rerun()
        
        # Add some spacing
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Chat container (response area below input)
        chat_container = st.container()
        
        with chat_container:
            render_chat()
    
    with col2:
        # Export functionality
        st.markdown("### 📥 Export")
        
        if st.session_state.messages:
            # Prepare export data
            export_data = {
                "title": "Swiggy Annual Report Q&A Session",
                "date": datetime.now().isoformat(),
                "messages": st.session_state.messages
            }
            
            # JSON export
            json_str = json.dumps(export_data, indent=2)
            st.download_button(
                "📄 Download JSON",
                json_str,
                file_name=f"swiggy_qa_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
            
            # Text export
            text_lines = ["Swiggy Annual Report Q&A Session\n", "=" * 40 + "\n\n"]
            for msg in st.session_state.messages:
                role = "You" if msg["role"] == "user" else "Assistant"
                text_lines.append(f"{role}: {msg['content']}\n\n")
                if msg["role"] == "assistant" and "citations" in msg:
                    text_lines.append("Sources:\n")
                    for c in msg["citations"]:
                        text_lines.append(f"  - Page {c['page']}: {c['section']}\n")
                    text_lines.append("\n")
            
            st.download_button(
                "📝 Download TXT",
                "".join(text_lines),
                file_name=f"swiggy_qa_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )
        else:
            st.info("Start a conversation to enable export options.")


if __name__ == "__main__":
    main()
