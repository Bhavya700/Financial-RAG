# Financial RAG Pipeline

A sophisticated Retrieval-Augmented Generation (RAG) system engineered to ingest, process, and demystify complex Securities and Exchange Commission (SEC) filings. Built with LangGraph, ChromaDB, and Google Gemini, this application dynamically fetches 10-K and 10-Q documents, vectorizes their contents, and utilizes a multi-step reasoning AI agent to translate complex financial jargon into easily understandable plain English.

## System Architecture

The project is structured into three primary pipelines:

1. **Extraction Layer (`sec_extractor.py`)**
   - Automatically queries the SEC EDGAR database to download requested company filings (10-K, 10-Q) based on Ticker and Year.
   - Cleans and parses the raw HTML submissions into structured plain text, removing bloated UI elements while preserving financial context.
   - Organizes extracted documents into a hierarchal local `data/` directory.

2. **Ingestion & Vectorization Layer (`ingest.py`)**
   - Processes extracted text using the `RecursiveCharacterTextSplitter` (1000 character chunks with 200 character overlaps) to maintain semantic boundaries.
   - Preserves core metadata (Ticker, Year, Quarter, Form Type).
   - Generates high-dimensional vector embeddings using Google's `gemini-embedding-001` model.
   - Persists embeddings locally within a ChromaDB vector database (`chroma_db/`) for rapid similarity search.

3. **LangGraph Reasoning Engine (`graph_rag.py`)**
   - Implements a stateful, cyclic workflow (LangGraph) replacing traditional linear RAG chains.
   - **Retrieval**: Queries ChromaDB for the most contextually relevant document chunks.
   - **Grading**: Utilizes an LLM with structured output (Pydantic) to evaluate document relevance. Irrelevant documents are filtered out to prevent hallucinations.
   - **Self-Correction (Query Rewriting)**: If retrieved documents are deemed insufficient, the agent autonomously rewrites the search query and loops back to the retrieval phase (up to 3 times).
   - **Jargon Translation**: Scans the filtered context for highly technical financial verbs and nouns, generating a simplified dictionary of analogies for the end user.
   - **Synthesis**: Combines the relevant context and translated jargon to generate a final, beginner-friendly response.

4. **Streamlit Frontend (`app.py`)**
   - Provides an interactive chat interface.
   - Features a sidebar for triggering the SEC EDGAR extraction and ChromaDB ingestion pipelines.
   - Displays real-time operational status of the LangGraph agent (retrieving, grading, rewriting) inside the UI for complete transparency.

## Installation and Setup

### Prerequisites
- Python 3.9+
- A Google Gemini API Key

### Configuration
1. Clone the repository:
   ```bash
   git clone https://github.com/Bhavya700/Financial-RAG.git
   cd Financial-RAG/New
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure Environment Variables:
   Create a `.env` file in the `New/` directory and add your Google API Key:
   ```env
   GOOGLE_API_KEY=your_gemini_api_key_here
   SEC_DOWNLOADER_COMPANY=YourCompanyName
   SEC_DOWNLOADER_EMAIL=your.email@example.com
   ```
   *(Note: The SEC requires a company name and email address to utilize their EDGAR downloading service).*

### Usage

Launch the interactive Streamlit application:
```bash
streamlit run app.py
```

1. Use the left sidebar to specify Tickers (e.g., AAPL, MSFT), Form Types, and Years.
2. Click **Fetch & Process Documents**. The application will hit the SEC servers, download the filings, chunk them, embed them via Gemini, and store them in ChromaDB.
3. Once ingestion is complete, use the main chat interface to ask complex financial questions.

## Technologies Used
- **Language**: Python
- **Orchestration**: LangChain, LangGraph
- **Vector Database**: ChromaDB
- **LLM & Embeddings**: Google Gemini API (`gemini-1.5-flash`, `gemini-embedding-001`)
- **Frontend**: Streamlit
- **Data Engineering**: SEC EDGAR Downloader, BeautifulSoup4, Pydantic
