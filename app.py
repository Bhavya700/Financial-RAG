import streamlit as st
import logging

from sec_extractor import fetch_sec_documents
from ingest import ingest_documents
from graph_rag import compile_rag_graph

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Financial SEC Assistant",
    page_icon="🤖",
    layout="wide",
)

# --- SESSION STATE INITIALIZATION ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "graph" not in st.session_state:
    st.session_state.graph = compile_rag_graph()


# --- SIDEBAR: DATA EXTRACTION & INGESTION ---
with st.sidebar:
    st.header("1. Load SEC Documents")
    st.markdown("Fetch filings from SEC EDGAR and ingest them into the vector database.")
    
    # User Inputs
    tickers_input = st.text_input("Tickers (comma separated)", value="AAPL, MSFT")
    form_types = st.multiselect("Form Types", ["10-K", "10-Q", "8-K"], default=["10-K", "10-Q"])
    years_input = st.text_input("Years (comma separated)", value="2023")
    
    # (Optional) We could parse parts of document here; our extractor extracts broadly.
    # quarters can be supported but depends on form types. 
    # For simplicity of UI, we use Q unknown / default quarters logic handled in fetch_sec_documents.
    
    if st.button("Fetch & Process Documents", type="primary"):
        tickers = [t.strip().upper() for t in tickers_input.split(",")]
        years = [y.strip() for y in years_input.split(",")]
        
        try:
            with st.spinner(f"Downloading {form_types} for {tickers}..."):
                # 1. Fetch
                docs = fetch_sec_documents(tickers, form_types, years)
                
            if docs:
                st.success(f"Successfully downloaded {len(docs)} documents.")
                with st.spinner("Ingesting into Vector Database..."):
                    # 2. Ingest
                    ingest_documents()
                st.success("✅ Ingestion complete! The AI is ready to answer questions.")
            else:
                st.warning("No documents found for the given criteria.")
                
        except Exception as e:
            st.error(f"Error processing documents: {e}")
            logger.error(f"UI Extraction Error: {e}")

    st.markdown("---")
    st.markdown("**Note:** Loading large documents may take a few minutes depending on Gemini API rate limits and ChromaDB processing.")


# --- MAIN PANEL: CHAT INTERFACE ---
st.title("2. Chat with Financial Documents 📊")
st.markdown("Ask natural language questions about the ingested SEC filings (e.g., *What are Apple's main risk factors in 2023?*)")

# Render chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # If assistant message has jargon, display it as an expander
        if "jargon" in message and message["jargon"]:
            with st.expander("📚 Learned Jargon"):
                for term, definition in message["jargon"].items():
                    st.markdown(f"**{term}**: {definition}")

# User Input
if prompt := st.chat_input("Ask a question about the filings..."):
    # Render user message
    st.session_state.messages.append({"role": "user", "content": prompt, "jargon": {}})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Agent Response with Status Tracker
    with st.chat_message("assistant"):
        # UI Elements for Audit Trail
        status_container = st.status("AI Agent Thinking...", expanded=True)
        
        # State Tracking Variables
        initial_state = {
            "question": prompt,
            "loop_count": 0
        }
        
        final_answer = ""
        learned_jargon = {}
        
        try:
            # Stream events from LangGraph
            for output in st.session_state.graph.stream(initial_state):
                for node_name, node_state in output.items():
                    # Update status dynamically based on current node
                    if node_name == "retrieve":
                        num_docs = len(node_state.get("documents", []))
                        status_container.write(f"🔍 Retrieving documents... (Found {num_docs} chunks)")
                    
                    elif node_name == "grade_documents":
                        num_docs = len(node_state.get("documents", []))
                        status_container.write(f"⚖️ Grading relevance... ({num_docs} chunks passed the relevance filter.)")
                    
                    elif node_name == "rewrite_query":
                        new_q = node_state.get("question", "")
                        status_container.write(f"🔄 Documents lacked relevance. Rewriting query to: *'{new_q}'*")
                        
                    elif node_name == "explain_jargon":
                        jargon = node_state.get("jargon_dict", {})
                        if jargon:
                            learned_jargon.update(jargon)
                            status_container.write(f"🧠 Translating {len(jargon)} financial terms into plain English...")
                            
                    elif node_name == "generate_answer":
                        status_container.write("✍️ Synthesizing plain-English response...")
                        final_answer = node_state.get("generation", "")
                        
            # Close the status container
            status_container.update(label="✅ Answer Generated!", state="complete", expanded=False)
            
            # Show final answer
            st.markdown(final_answer)
            
            # Show jargon expander if applicable
            if learned_jargon:
                with st.expander("📚 Learned Financial Jargon", expanded=True):
                    st.markdown("I translated these concepts to help explain the answer:")
                    for term, definition in learned_jargon.items():
                        st.markdown(f"- **{term}**: {definition}")
            
            # Save to history
            st.session_state.messages.append({
                "role": "assistant", 
                "content": final_answer,
                "jargon": learned_jargon
            })

        except Exception as e:
            status_container.update(label="❌ Error generating response", state="error")
            st.error("There was an error answering your question. Make sure your GOOGLE_API_KEY is active and documents are ingested.")
            logger.error(f"LangGraph Stream Error: {e}")

# Instructions below chat
if not st.session_state.messages:
    st.caption("👈 Use the sidebar first to fetch and ingest SEC data. Then, type your question above to begin our RAG LangGraph chain.")
