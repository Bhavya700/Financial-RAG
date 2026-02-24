import os
import glob
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import logging

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CHROMA_DB_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")

def parse_metadata_from_filename(filepath: str) -> dict:
    """
    Extracts metadata from our standard filename format:
    {ticker}_{year}_{quarter}_{form_type}.txt
    """
    filename = os.path.basename(filepath)
    name_without_ext = os.path.splitext(filename)[0]
    parts = name_without_ext.split("_")
    
    metadata = {"source": filepath}
    
    if len(parts) >= 4:
        metadata["Ticker"] = parts[0]
        metadata["Year"] = parts[1]
        metadata["Quarter"] = parts[2]
        metadata["Form_Type"] = parts[3]
    else:
        logger.warning(f"Could not fully parse metadata from {filename}")
        
    return metadata

def ingest_documents():
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    if not os.path.exists(data_dir):
        logger.error(f"Data directory not found: {data_dir}. Run sec_extractor.py first.")
        return

    # Find all txt files recursively in the data directory
    txt_files = glob.glob(os.path.join(data_dir, "**", "*.txt"), recursive=True)
    
    if not txt_files:
        logger.warning("No .txt files found in data directory.")
        return
        
    logger.info(f"Found {len(txt_files)} document(s) to ingest.")

    documents = []
    for filepath in txt_files:
        loader = TextLoader(filepath, encoding="utf-8")
        docs = loader.load()
        
        # Inject our parsed metadata
        meta = parse_metadata_from_filename(filepath)
        for d in docs:
            d.metadata.update(meta)
            
        documents.extend(docs)

    logger.info("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    
    chunks = text_splitter.split_documents(documents)
    logger.info(f"Created {len(chunks)} chunks.")

    logger.info("Initializing OpenAI Embeddings and persisting to ChromaDB...")
    
    # Ensure GOOGLE_API_KEY is set in .env
    if not os.getenv("GOOGLE_API_KEY"):
        logger.error("GOOGLE_API_KEY not found in environment. Cannot create embeddings.")
        return
        
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_DIR
    )
    
    # Force persistence to disk
    vectorstore.persist()
    logger.info(f"Successfully ingested and saved vectors to {CHROMA_DB_DIR}")

def get_retriever():
    """Utility function for graph_rag.py to fetch the retriever"""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)
    # Return a retriever that fetches top 4 chunks
    return vectorstore.as_retriever(search_kwargs={"k": 4})

if __name__ == "__main__":
    logger.info("Starting ingestion process...")
    ingest_documents()
