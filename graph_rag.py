import os
import json
import logging
from typing import List, Dict, Any, TypedDict

from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from langgraph.graph import START, END, StateGraph
from ingest import get_retriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# 1. State Definition
# -----------------------------------------------------------------------------
class GraphState(TypedDict):
    """
    Represents the state of our LangGraph execution.
    """
    question: str
    documents: List[Document]
    generation: str
    jargon_dict: Dict[str, str]
    loop_count: int

# -----------------------------------------------------------------------------
# 2. Schema Definitions (Pydantic / Structured Output)
# -----------------------------------------------------------------------------
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    is_relevant: bool = Field(description="Documents contain info relevant to the user's question. Output 'True' if relevant, 'False' otherwise.")

# -----------------------------------------------------------------------------
# 3. Graph Nodes
# -----------------------------------------------------------------------------
def retrieve_node(state: GraphState):
    """
    Retrieve documents from ChromaDB based on the current question.
    """
    logger.info("---RETRIEVE DOCUMENTS---")
    question = state["question"]
    loop_count = state.get("loop_count", 0)
    
    # Initialize retriever
    retriever = get_retriever()
    documents = retriever.invoke(question)
    
    return {"documents": documents, "question": question, "loop_count": loop_count}

def grade_documents_node(state: GraphState):
    """
    Determines whether the retrieved documents are relevant to the question.
    Filters out any irrelevant documents.
    """
    logger.info("---GRADE DOCUMENTS---")
    question = state["question"]
    documents = state["documents"]
    loop_count = state["loop_count"]
    
    # Setup LLM with structured output
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    structured_llm_grader = llm.with_structured_output(GradeDocuments)
    
    system_prompt = """You are a grader assessing the relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
    It does not need to be a stringent test. The goal is to filter out entirely unrelated retrievals."""
    
    grade_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ])
    
    retrieval_grader = grade_prompt | structured_llm_grader
    
    filtered_docs = []
    for doc in documents:
        score = retrieval_grader.invoke({"question": question, "document": doc.page_content})
        grade = score.is_relevant
        if grade:
            logger.info("-> Document graded as relevant.")
            filtered_docs.append(doc)
        else:
            logger.info("-> Document graded as NOT relevant. Skipping.")
            
    return {"documents": filtered_docs, "question": question, "loop_count": loop_count}

def rewrite_query_node(state: GraphState):
    """
    Transform the query to produce a better question.
    """
    logger.info("---REWRITE QUERY---")
    question = state["question"]
    documents = state["documents"]
    loop_count = state["loop_count"]
    
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    
    system_prompt = """You are an AI generating an improved question optimized for database retrieval. \n
    Look at the input question and try to reason about the underlying semantic intent / keywords."""
    
    rewrite_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", f"Initial question: {question} \n\n Formulate an improved, highly-searchable question."),
    ])
    
    question_rewriter = rewrite_prompt | llm
    rewritten_question = question_rewriter.invoke({}).content
    
    # Increment the search loop counter
    loop_count += 1
    
    return {"question": rewritten_question, "documents": documents, "loop_count": loop_count}

def explain_jargon_node(state: GraphState):
    """
    Analyzes documents to extract accounting and financial jargon, returning simple 1-sentence analogies.
    """
    logger.info("---EXPLAIN JARGON FOR BEGINNERS---")
    documents = state["documents"]
    
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1)
    
    # Provide the page content to the LLM
    docs_text = "\n\n".join([d.page_content for d in documents])
    
    system_prompt = """You are an expert financial advisor talking to a complete beginner.
    Identify any highly technical financial, accounting, or SEC-specific terms in the text below (e.g., Amortization, EBITDA, Derivatives).
    Generate a very simple, 1-sentence plain-English analogy or definition for each technical term.
    Produce the output strictly as a JSON dictionary mapping the term to its simplified definition.
    Example: {{"Amortization": "A way of spreading out a big expense over time, like paying off a car loan in monthly chunks."}}"""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", f"Documents:\n\n{docs_text}")
    ])
    
    chain = prompt | llm
    response = chain.invoke({})
    
    # Try parsing JSON safely
    jargon_dict = {}
    try:
        # Strip markdown block formatting if present
        raw_output = response.content.replace("```json", "").replace("```", "").strip()
        jargon_dict = json.loads(raw_output)
        logger.info(f" -> Found {len(jargon_dict)} jargon terms to simplify.")
    except Exception as e:
        logger.warning(f"Failed to parse jargon dictionaries: {e}")

    return {"jargon_dict": jargon_dict}

def generate_answer_node(state: GraphState):
    """
    Generate the final synthesized answer.
    """
    logger.info("---GENERATE FINAL ANSWER---")
    question = state["question"]
    documents = state["documents"]
    loop_count = state["loop_count"]
    jargon_dict = state.get("jargon_dict", {})
    
    if not documents and loop_count >= 3:
        logger.warning("Max retrieval loops reached without finding relevant documents.")
        return {"generation": "I'm sorry, I couldn't find enough relevant information in the SEC filings to answer your question accurately."}
        
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)
    
    docs_text = "\n\n---\n\n".join([d.page_content for d in documents])
    # Format jargon for the prompt
    jargon_text = ""
    if jargon_dict:
        jargon_text = "Here is a list of complex terms found in the context with their simplified meanings:\n"
        for k, v in jargon_dict.items():
            jargon_text += f"- {k}: {v}\n"
            
    system_prompt = f"""You are a helpful, beginner-friendly financial assistant analyzing SEC documents.
    Answer the user's question using ONLY the provided document context.
    If you don't know the answer based strictly on the context, say you don't know.
    
    {jargon_text}
    
    Instruction: Use simple, plain English. If you must use a technical term, you MUST include its simplified meaning inline for the user to understand. Keep your tone encouraging and accessible."""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", f"Context: {docs_text}\n\nQuestion: {question}")
    ])
    
    chain = prompt | llm
    response = chain.invoke({})
    
    return {"generation": response.content}

# -----------------------------------------------------------------------------
# 4. Conditional Edges
# -----------------------------------------------------------------------------
def grade_decision_edge(state: GraphState):
    """
    Determines the next path based on document relevance grades.
    """
    logger.info("---CHECKING DOCUMENT RELEVANCE DECISION---")
    filtered_documents = state["documents"]
    loop_count = state["loop_count"]
    
    if not filtered_documents:
        # All documents were graded as irrelevant
        if loop_count >= 3:
            logger.info("-> Decision: Max retries (3) reached. Forcing generation fallback.")
            return "generate"
        logger.info("-> Decision: No relevant docs found. Rewriting Query.")
        return "rewrite"
    else:
        # We have relevant documents! Move to simplify step.
        logger.info("-> Decision: Relevant documents found. Proceeding to Explain Jargon.")
        return "explain_jargon"

# -----------------------------------------------------------------------------
# 5. Graph Compilation
# -----------------------------------------------------------------------------
def compile_rag_graph():
    """Build and compile the LangGraph workflow."""
    workflow = StateGraph(GraphState)

    # Add Nodes
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade_documents", grade_documents_node)
    workflow.add_node("rewrite_query", rewrite_query_node)
    workflow.add_node("explain_jargon", explain_jargon_node)
    workflow.add_node("generate_answer", generate_answer_node)

    # Compile edges
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    
    workflow.add_conditional_edges(
        "grade_documents",
        grade_decision_edge,
        {
            "explain_jargon": "explain_jargon",
            "rewrite": "rewrite_query",
            "generate": "generate_answer"
        }
    )
    
    workflow.add_edge("rewrite_query", "retrieve")
    workflow.add_edge("explain_jargon", "generate_answer")
    workflow.add_edge("generate_answer", END)

    app = workflow.compile()
    return app

# -----------------------------------------------------------------------------
# Test Execution
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    app = compile_rag_graph()
    
    # Dummy Test Question
    initial_state = {
        "question": "What are the primary risk factors mentioned regarding artificial intelligence or data privacy for Microsoft in the 10-Q?",
        "loop_count": 0
    }
    
    logger.info("Running test query through compiled LangGraph...")
    
    try:
        # We iterate over the steeam yielding the dictionary representing the node's output
        for output in app.stream(initial_state):
            for key, value in output.items():
                logger.info(f"Finished Node: {key}")
                
        # To get the final state cleanly
        final_state = app.invoke(initial_state)
        print("\n==============================================")
        print("FINAL SYNTHESIZED ANSWER:")
        print("==============================================")
        print(final_state["generation"])
        print("\n==============================================")
        print("EXTRACTED JARGON DICTIONARY:")
        print(final_state.get("jargon_dict"))
        
    except Exception as e:
        logger.error(f"Graph execution failed: {e}")
        logger.warning("Please ensure GOOGLE_API_KEY is set in .env and you have ingested documents into ChromaDB first.")
