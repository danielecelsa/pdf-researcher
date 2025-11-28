# app.py
# main application file for the Streamlit app
# Implements a chatbot that can answer questions based on uploaded PDF/TXT documents using LangGraph, LangChain, ChromaDB, and Google Gemini.

print("=== PRINT FROM PROCESS START ===")

# ------------------------------
# Imports
# ------------------------------
import os, io, time
import asyncio
import tempfile
from datetime import datetime, timezone
from pathlib import Path
import uuid
import json
import base64
import fitz  # pymupdf
from time import perf_counter

import streamlit as st
from dotenv import load_dotenv

# LangGraph / LangChain Core
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import StructuredTool
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate, PromptTemplate
from langchain_community.document_loaders import PyPDFLoader 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.callbacks import Callbacks


# Vector DB: Chroma integration
import chromadb
from langchain_chroma import Chroma
from chromadb.utils.embedding_functions.chroma_langchain_embedding_function import create_langchain_embedding

from logging_config import (
    get_logger,
)

# Helpers
from helpers import (
    compute_cost,
    get_user_info,
    process_agent_events,
    TokenUsageCallbackHandler
)

from async_bg import collect_events_from_agent

from prompts import RAG_AGENT_SYSTEM_PROMPT, RAG_RETRIEVAL_PROMPT

# Load environment variables from .env file if not in a rendering environment
if os.getenv("RENDER") != "true":
    load_dotenv()


# ------------------------------
# Configuration
# ------------------------------
MODEL = os.environ.get("GENAI_MODEL", "gemini-2.5-flash")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

LOG_DIR = Path(os.environ.get("CHAT_LOG_DIR", "./logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)

COST_PER_1K_INPUT = float(os.getenv("COST_PER_1K_TOKENS_USD_INPUT", "0.002"))
COST_PER_1K_OUTPUT = float(os.getenv("COST_PER_1K_TOKENS_USD_OUTPUT", "0.002"))

SAMPLE_PDF_PATH = "example_docs/llm_introduction.pdf"

# ------------------------------
# LOGGING SETUP
# ------------------------------

logger_local = get_logger("local")
logger_betterstack = get_logger("betterstack")
logger_redis = get_logger("redis")
logger_all = get_logger("all") 


# Initialize Streamllit session state variables
if "conversation_thread_id" not in st.session_state:
    st.session_state.conversation_thread_id = str(uuid.uuid4())
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [AIMessage(content="Hello, I am your personal chatbot. I can answer general knowledge questions, or document-related if you upload files. How can I help you?")]
if "uploaders" not in st.session_state:
    st.session_state.uploaders = []
if "example_loaded" not in st.session_state: # to avoid reloading example multiple times
    st.session_state["example_loaded"] = False
if "example_file" not in st.session_state:
    st.session_state["example_file"] = None
if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())
    st.session_state['collection_name'] = f"pdf_researcher_{st.session_state['session_id']}"
if "latency" not in st.session_state:
    st.session_state.latency = 0.0
if "trace" not in st.session_state:
    st.session_state.trace = []

# --- Token tracking ---
# Overall session tokens
if "total_tokens" not in st.session_state:
    st.session_state["total_tokens"] = 0
if "total_input_tokens" not in st.session_state: # to track total input tokens cost
    st.session_state["total_input_tokens"] = 0
if "total_output_tokens" not in st.session_state: # to track total output tokens cost
    st.session_state["total_output_tokens"] = 0
# Last interaction tokens
if "input_tokens_last" not in st.session_state:
    st.session_state.input_tokens_last = 0
if "output_tokens_last" not in st.session_state:
    st.session_state.output_tokens_last = 0
if "total_tokens_last" not in st.session_state:
    st.session_state.total_tokens_last = 0

# RAG-specific token tracking
if "RAG_total_tokens" not in st.session_state:
    st.session_state.RAG_total_tokens = 0
if "RAG_input_tokens" not in st.session_state:
    st.session_state.RAG_input_tokens = 0
if "RAG_output_tokens" not in st.session_state:
    st.session_state.RAG_output_tokens = 0
if "RAG_total_tokens_last" not in st.session_state:
    st.session_state.RAG_total_tokens_last = 0
if "RAG_input_tokens_last" not in st.session_state:
    st.session_state.RAG_input_tokens_last = 0
if "RAG_output_tokens_last" not in st.session_state:
    st.session_state.RAG_output_tokens_last = 0

# Cost tracking
if "usd" not in st.session_state:
    st.session_state.usd = 0.0
if "usd_last" not in st.session_state:
    st.session_state.usd_last = 0.0

logger_local.info("Session ID:  %s", st.session_state["session_id"])
logger_local.info("Conversation Thread ID: %s", st.session_state.conversation_thread_id)

collection_name = st.session_state['collection_name']

def update_token_usage(usage_last: dict, token_callback: TokenUsageCallbackHandler):
    """Update the token usage in the session state vars."""
    # Total tokens (last) from all sources - callbacks 
    usage_summary = token_callback.get_usage_dict()
    total_tokens_all_sources = usage_summary.get("total_tokens", 0)

    # Last interaction tokens
    st.session_state.total_tokens_last = total_tokens_all_sources
    st.session_state.input_tokens_last = usage_summary.get("input_tokens", 0)
    st.session_state.output_tokens_last = usage_summary.get("output_tokens", 0)

    # RAG tokens calculation
    agent_tokens_dict = usage_last # from process_agent_events - only agents no RAG
    st.session_state.RAG_total_tokens_last = st.session_state.total_tokens_last - agent_tokens_dict.get("total_tokens", 0)
    st.session_state.RAG_input_tokens_last = st.session_state.input_tokens_last - agent_tokens_dict.get("input_tokens", 0)
    st.session_state.RAG_output_tokens_last = st.session_state.output_tokens_last - agent_tokens_dict.get("output_tokens", 0)

    # Session totals
    st.session_state.total_input_tokens += st.session_state.input_tokens_last
    st.session_state.total_output_tokens += st.session_state.output_tokens_last
    st.session_state.total_tokens += st.session_state.total_tokens_last
    st.session_state.RAG_total_tokens += st.session_state.RAG_total_tokens_last
    st.session_state.RAG_input_tokens += st.session_state.RAG_input_tokens_last
    st.session_state.RAG_output_tokens += st.session_state.RAG_output_tokens_last   

# ------------------------------
# Chroma client helper
# ------------------------------
@st.cache_resource
def get_chroma_client(): # default: in-process, no persistence
    """Get or create a Chroma client instance."""
    client = chromadb.Client()
    logger_local.info("Collections_lru: %s", client.list_collections())
    return client

def clear_db():
    """Clear the Chroma collection for this session."""
    if st.button(":blue[Clear DataBase]"):
        st.session_state["example_loaded"] = False
        st.session_state.uploaders = []
        client = get_chroma_client()
        try:
            client.delete_collection(collection_name)
            st.warning("Cleared uploaded files.")
        except Exception as e:
            st.warning("No files to be cleared.")
        logger_all.info("Collections Deleted: %s", client.list_collections())
        time.sleep(1.5)
        st.rerun()

# ------------------------------
# Example PDF helper
# ------------------------------
def example_pdf():
    # Load a sample PDF from disk and add to session_state.uploaders
    if not os.path.exists(SAMPLE_PDF_PATH):
        st.error(f"Sample file not found at: {SAMPLE_PDF_PATH}")
    else:
        with open(SAMPLE_PDF_PATH, "rb") as f:
            data = f.read()
        bio = io.BytesIO(data)
        bio.name = os.path.basename(SAMPLE_PDF_PATH)  # important: use .name as metadata
        bio.seek(0)
        st.session_state["uploaders"] = [bio]
        update_vector_db()
    return bio

def show_pdf_iframe_base64(file_like, height=800):
    """Try to render PDF inside an iframe using a data: URI.
       Return True if we injected html (still may be blocked client-side)."""
    try:
        file_like.seek(0)
        pdf_bytes = file_like.read()
        base64_pdf = base64.b64encode(pdf_bytes).decode("utf-8")
        iframe_html = f'''
        <iframe
          src="data:application/pdf;base64,{base64_pdf}"
          width="100%" height="{height}"
          style="border: none;"
        >
        </iframe>
        '''
        st.components.v1.html(iframe_html, height=height)
        return True
    except Exception as e:
        st.warning(f"Could not render PDF iframe: {e}")
        return False

def pdf_first_page_to_png_bytes(file_like, zoom: float = 2.0):
    """Return PNG bytes of the first page of PDF using pymupdf (fitz)."""
    file_like.seek(0)
    pdf_bytes = file_like.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    if doc.page_count < 1:
        raise ValueError("PDF has no pages")
    page = doc.load_page(0)  # first page
    mat = fitz.Matrix(zoom, zoom)  # zoom for better resolution
    pix = page.get_pixmap(matrix=mat, alpha=False)
    png_bytes = pix.tobytes("png")
    return png_bytes

def show_pdf_preview_with_fallback(file_like):
    """Try iframe; if Chrome blocks, show a PNG of first page and provide download link."""
    # First attempt iframe
    ok = show_pdf_iframe_base64(file_like, height=800)
    # Even if ok==True, browser might still block. So show a recommended fallback control:
    st.info("If the embedded preview is blocked by your browser, use the image preview below or click 'Open in new tab' to view the full PDF.")
    # Show first-page image always as a robust fallback (optional: show only if user requests)
    try:
        png = pdf_first_page_to_png_bytes(file_like, zoom=2.0)
        st.image(png, caption="First page preview (fallback)", width='stretch')
    except Exception as e:
        st.warning(f"Could not render fallback image: {e}")

    # Provide a download/open link:
    try:
        file_like.seek(0)
        pdf_bytes = file_like.read()
        b64 = base64.b64encode(pdf_bytes).decode("utf-8")
        dl_html = f'''
        <a href="data:application/pdf;base64,{b64}" target="_blank" download="example_file.pdf">Open in new tab / Download PDF </a>
        '''
        st.markdown(dl_html, unsafe_allow_html=True)
    except Exception:
        st.warning("Could not create download link for PDF.")

# ------------------------------
# TOOL: research (uses Chroma)
# ------------------------------
# Useful to have many sessions with isolated collections. We are not using persistence here, but you could.
def research_factory(collection_name: str):
    """Factory to create a research tool bound to a specific Chroma collection/session of the user."""
    def research(query: str, callbacks: Callbacks = None) -> str:
        """
        Use Chroma vector store to retrieve and summarize information from the documents (PDFs or TXTs) uploaded by the user, and answer the user's question.
        
        Use this tool whenever the user's question involves the uploaded documents,
        even if the question is only partially related to their content.
        Do not use this tool for general knowledge questions unrelated to the uploaded documents.
        
        This tool will:
        - create the embeddings object (safely)
        - create a vectorstore from a Chroma collection, which contains the uploaded documents
        - use retriever to get top-k, run RAG chain and return a concise answer

        Args:
            query (str): The user question to be answered using only the uploaded documents.

        Returns:
            str: A concise, evidence-based answer derived exclusively from the uploaded documents.
        """
        logger_all.info("TOOL CALLED with: %s", query)

        # ensure embeddings client event loop exists (Google GenAI uses async clients)
        try:
            # ensure_event_loop if you have such helper; otherwise minimally:
            asyncio.get_event_loop()
        except Exception:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

        # Create the Chroma vectorstore from the existing collection
        client = get_chroma_client()  # in-process ephemeral
        logger_all.info("Collections_tool: %s ", collection_name)
        logger_local.info("Collections_tool_list: %s", client.list_collections())
        vect = Chroma(collection_name=collection_name, embedding_function=embeddings, client=client)

        # fai una similarity search di debug per vedere se la collection esiste e quanti risultati
        try:
            # debug: prova similarity_search_with_score
            top = vect.similarity_search_with_score(query, k=8)
            logger_all.info("Retrieved %d hits for query", len(top))
            for i, (doc, score) in enumerate(top):
                src = doc.metadata.get("source", doc.metadata.get("filename", "unknown"))
                excerpt = doc.page_content[:200].replace("\n", " ")
                logger_local.info("RANK %d score=%.4f source=%s excerpt=%s", i, score, src, excerpt)
        except Exception as e:
            logger_all.exception("Similarity search failed: %s", e)
            return "I could not search the vector DB."

        # Build retriever and RAG chain
        # choose k somewhat larger to increase recall
        retriever = vect.as_retriever(search_kwargs={"k": 4})

        rag_prompt = PromptTemplate.from_template(RAG_RETRIEVAL_PROMPT)

        llm = ChatGoogleGenerativeAI(model=MODEL)

        doc_chain = create_stuff_documents_chain(llm, rag_prompt)
        rag_chain = create_retrieval_chain(retriever, doc_chain)
        
        try:
            resp = rag_chain.invoke(
                {"input": query}, 
                config={"callbacks": callbacks}
            )
            logger_local.info("RAG CHAIN RESPONSE: %s", repr(resp))
        
        except Exception as e:
            logger_all.exception("RAG chain invocation failed: %s", e)
            return "I could not run the retrieval chain due to an internal error."

        # Normalize response
        try:
            if isinstance(resp, dict):
                result = resp.get("answer") or resp.get("text") or str(resp)
                logger_local.info("Tool's answer: %s", result)
                return result
            result = str(resp)
            logger_local.info("Tool's answer (fallback): %s", result)
            return result
        
        except Exception as e:
            logger_all.exception("Error extracting result from RAG response: %s", e)
            return "I could not parse the result from the retrieval chain."
    return research


# ------------------------------
# Build agent
# ------------------------------
@st.cache_resource
def get_prompt():
    system = SystemMessagePromptTemplate.from_template(RAG_AGENT_SYSTEM_PROMPT)

    hist = MessagesPlaceholder(variable_name="messages")
    prompt = ChatPromptTemplate.from_messages([system, hist])
    return prompt

@st.cache_resource
def get_llm():
    try:
        llm = ChatGoogleGenerativeAI(
            model=MODEL,
            google_api_key=GOOGLE_API_KEY,
            temperature=0.2,
            #convert_system_message_to_human=True,
            safety_settings=None,
        )
    except Exception as e:
        logger_all.exception("Could not initialize LLM: %s", e)
        llm = None
    return llm    

@st.cache_resource
def get_checkpointer():
    try:
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver as SqliteCheckpointer
        checkpointer = SqliteCheckpointer.from_conn_string(os.environ.get("CHECKPOINT_DB", "./langgraph_state.sqlite"))
    except Exception:
        try:
            from langgraph.checkpoint.memory import InMemorySaver as InMemoryCheckpointer
            checkpointer = InMemoryCheckpointer()
        except Exception:
            checkpointer = None
    return checkpointer

def build_agent(collection_name):

    research_tool_function = research_factory(collection_name)

    tools = [
        StructuredTool.from_function(
            research_tool_function,
            name="research",
            description=(
                "Use this tool to answer questions that require information from the uploaded PDF/text documents. "
                "Always call this tool when the user's question refers to facts, dates, quotes, or content contained in the uploaded files."
                "The tool accepts a single string question and returns a concise, evidence-based answer including sources when possible."
            ),
        )
    ]
    
    agent = create_react_agent(
        model=get_llm(),
        tools=tools,
        prompt=get_prompt(),
        checkpointer=get_checkpointer(),
    )

    return agent

if 'agent_for_session' not in st.session_state:
    st.session_state['agent_for_session'] = build_agent(st.session_state['collection_name'])


# ------------------------------
# Update vector DB using Chroma
# ------------------------------
def update_vector_db():
    """Update the vector database with newly uploaded documents (Chroma)."""
    if not st.session_state.uploaders:
        return "No files uploaded."

    # ensure event loop for embeddings init
    try:
        asyncio.get_event_loop()
    except Exception:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

    raw_docs = []
    for f in st.session_state.uploaders:
        # write temp file and load via PyPDFLoader
        if f.name.lower().endswith(".pdf"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(f.read())
                tmp_path = tmp.name
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
            # attach source metadata
            for d in docs:
                d.metadata = d.metadata or {}
                d.metadata["source"] = f.name
            raw_docs.extend(docs)
        elif f.name.lower().endswith(".txt"):
            content = f.read().decode("utf-8") if hasattr(f, "read") else f
            raw_docs.append(Document(page_content=content, metadata={"source": f.name}))

    if not raw_docs:
        return "No docs extracted from uploads."

    # split
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    final_docs = text_splitter.split_documents(raw_docs)

    # Create or update Chroma collection
    chroma_client = get_chroma_client()
    emb_chroma = create_langchain_embedding(embeddings)
    coll = chroma_client.get_or_create_collection(name=collection_name, embedding_function=emb_chroma)
    # Create stable ids (helpful to later delete by ids/source)
    ids = [f"{collection_name}::{uuid.uuid4()}" for _ in final_docs]
    coll.add(ids=ids, documents=[d.page_content for d in final_docs], metadatas=[d.metadata for d in final_docs])
    logger_all.info("Collections: %s", chroma_client.list_collections())
    cols = chroma_client.get_collection(collection_name)
    logger_local.info("Count: %d", cols.count())


    files = len(st.session_state.uploaders)

    logger_all.info("Created/Updated Chroma collection with %d files, %d documents and %d chunks.", files, len(raw_docs), len(docs))

    # Optionally clear uploaders after processing to avoid reprocessing duplicates
    st.session_state.uploaders = []

    return f"Processed {files} files and updated vector DB with {len(raw_docs)} documents and {len(docs)} chunks."


# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="Enterprise RAG Agent", page_icon="📄", layout="wide")
st.title("📄 Enterprise PDFs-RAG Agent: LangGraph + Chroma", anchor=False)

body="""
**Document Intelligence Agent - Powered by Agentic RAG**

This application demonstrates a **Retrieval-Augmented Generation (RAG)** system wrapped in an Agentic framework.
Unlike linear RAG pipelines, this agent **autonomously decides** when to query the knowledge base based on user intent.

**Key Capabilities:**
*   **🧠 Agentic Decision Making:** The system uses a **ReAct** loop. It doesn't just retrieve; it *thinks* first. If you say "Hi", it replies instantly. If you ask "What's in the contract?", it calls the retrieval tool.
*   **🔍 Session-Scoped Vector Store:** Uses **ChromaDB** to create isolated, ephemeral embeddings for each user session.
*   **📏 Granular Observability:** Tracks cost separately for the *Agent's Reasoning* vs. the *Retrieval Process*, allowing for precise ROI calculation of RAG operations.
"""
with st.expander('About this demo (Read me)', expanded=False, ):
    st.markdown(body)

# Chat submission (note: using agent.invoke recommended to extract content)
user_query = st.chat_input("Type your message here...")
if user_query:

    # Get user info as soon as they submit a query
    user_details = get_user_info(logger_all)
    logger_all.info(
        f"New query received from IP: {user_details['ip']} "
        f"with User-Agent: {user_details['user_agent']}"
    )

    st.session_state.chat_history.append(HumanMessage(content=user_query))

    thread_id = st.session_state.conversation_thread_id 
    token_callback = TokenUsageCallbackHandler()

    config = {
        "configurable": {"thread_id": thread_id},
        "callbacks": [token_callback]
        }
    inputs = {"messages": st.session_state.chat_history}

    final_answer_obj = None
    ai_content = ""

    # ------------------------------
    # Collect events using background loop helper
    # - this internally runs agent.astream(inputs, config=config, stream_mode="updates")
    # ------------------------------

    start = perf_counter()
    with st.spinner("Thinking..."):
        try:
            # 1. Collect all events from the agent in the background
            events = collect_events_from_agent(st.session_state['agent_for_session'], inputs, config=config, timeout=120)

            logger_local.info("COLLECTED EVENTS: %s", events)
            
            # 2. Process the events with our new, focused parser
            final_answer_obj, trace, usage_last = process_agent_events(events)

            # 3. Update session state for UI
            st.session_state.trace = trace
            update_token_usage(usage_last, token_callback) 

            if final_answer_obj:
                ai_content = final_answer_obj.content
            else:
                ai_content = "I'm sorry, I couldn't find a final answer."
                st.error("Could not extract a final answer from the agent's response.")

        except Exception as e:
            ai_content = f"An error occurred: {e}"
            st.error(ai_content)
            st.session_state.trace = []

    st.session_state.latency = perf_counter() - start

    # Calculate costs for UI display
    try:
        st.session_state.usd = compute_cost(st.session_state.total_input_tokens, st.session_state.total_output_tokens, COST_PER_1K_INPUT, COST_PER_1K_OUTPUT)
    except Exception:
        logger_all.exception("Error computing total cost: %s", e)
        st.session_state.usd = 0.0
    
    try:
        st.session_state.usd_last = compute_cost(st.session_state.input_tokens_last, st.session_state.output_tokens_last, COST_PER_1K_INPUT, COST_PER_1K_OUTPUT)
    except Exception:
        logger_all.exception("Error computing last cost: %s", e)
        st.session_state.usd_last = 0.0

    st.session_state.chat_history.append(AIMessage(content=ai_content))

    # --- LOGGING ---
    logger_all.info("Latency for full response: %.2f seconds", st.session_state.latency)
    logger_all.info("Last interaction tokens: %d (input), %d (output), %d (total)", st.session_state.input_tokens_last, st.session_state.output_tokens_last, st.session_state.total_tokens_last)
    logger_all.info("Estimated last interaction cost: $%.5f", st.session_state.usd_last)
    logger_all.info("Total tokens so far: %d (input), %d (output), %d (total)", st.session_state.total_input_tokens, st.session_state.total_output_tokens, st.session_state.total_tokens)
    logger_all.info("Estimated total cost so far: $%.5f", st.session_state.usd)

    data_dict={
        "ts": datetime.now(timezone.utc).isoformat(),
        "model": MODEL,
        "messages": [{"role": m.__class__.__name__, "content": m.content} for m in st.session_state.chat_history],
        "trace": st.session_state.trace,
    }
    logger_all.info(json.dumps(data_dict, indent=2, ensure_ascii=False))

# ------------------------------
# Sidebar
# ------------------------------
with st.sidebar:
    st.header("⚙️ System Architecture", divider="rainbow")
    st.info("This PoC demonstrates **Agentic RAG** with **Session-Scoped Vector Storage**.")

    # ------------------------------
    # Info Section
    # ------------------------------
    with st.expander("🛠️ Architecture & Tech Stack"):
        st.markdown("""
        **Core Orchestration:**
        - `LangGraph`: Agentic reasoning loop.
        - `LangChain`: Ingestion pipelines.
        
        **Data & Retrieval:**
        - ***Ingestion:*** `PyPDFLoader` + `RecursiveCharacterTextSplitter`.
        - ***Vector Store:*** `ChromaDB` (Ephemeral/In-Memory) for session isolation.
        - ***Embedding Model:*** Google Gemini `text-embedding-004`.
        
        **Observability & FinOps:**
        - ***Cost Attribution:*** Splits token usage between the *Agent* (Reasoning) and the *RAG Chain* (Retrieval).
        - ***Protocol Trace:*** JSON-level visibility into Tool Inputs/Outputs.
        - ***Distributed Logging:***  Structured logs (Redis + BetterStack) for remote monitoring.
        """)
    with st.expander("🧪 How to Test (Scenarios)"):
        st.caption("Upload PDFs and try similar inputs to trigger the tool usage:")
        st.markdown("**1. Specific Extraction (*needs `research` tool*)**:")
        st.markdown("> *In the PDF I uploaded, what does XXX say about YYY?*")
        
        st.markdown("**2. Summarization (*needs `research` tool*)**:")
        st.markdown("> *Summarize the concept of ZZZ as explained in the document.*")

        st.markdown("**3. General Chat (*no tool*)**:")
        st.markdown("> *Hello, how are you?*")
        st.caption("👉 If you ask a general question unrelated to the uploaded docs, the agent should respond directly without invoking the `research` tool.")
        st.write("#### *👇 Or you can Load the Example PDF below and try questions like these:*")
        st.markdown("""
            > *What is a Neural Network, according to the document?*
            
            > *Summarize the concept of Transformers as explained in the document*
            """
            )

        with st.expander("👀 What to watch"):
            st.markdown("**1. The Tool Trigger**:")
            st.caption("👇 *Check the **🧠 Agent's Reasoning Steps** section below*")
            st.markdown("""
                        Notice how the agent **decides** whether to call the `research` tool or answer directly. This saves costs on general chit-chat.
                        """)

            st.markdown("**2. RAG Cost Attribution**:")
            st.caption("👇 *Check the **📊 Live Metrics** section below*")
            st.markdown("""
                        Observe the **(RAG tokens: ...)** metric. This shows exactly how much "context overhead" the document retrieval added to the conversation.
                        """)
            
    st.markdown("---")

    # ------------------------------
    # Upload PDF Section
    # ------------------------------
    st.markdown("### 📂 Knowledge Base")
    st.session_state.uploaders = st.file_uploader("Upload & click the *Process* button", type=["pdf", "txt"], accept_multiple_files=True)        
    if st.session_state.uploaders:
        if st.button("Process and update vector DB"):
            result_msg = update_vector_db()
            st.success(result_msg)
            clear_db()
    
    if not st.session_state.get("example_loaded"):
        st.markdown("##### or load sample data:")
        if st.button("Load Example PDF"):
            bio = example_pdf()
            st.session_state["example_loaded"] = True
            st.session_state["example_file"] = bio
            st.success(f"Loaded example: {bio.name}")
            time.sleep(1.5)
            st.rerun()
    if st.session_state.get("example_loaded"):
        with st.expander("You can now *\"chat\"* with the example PDF! ✅"):
            st.markdown("Try asking:")
            st.markdown("""
                        > *What is a Neural Network, according to the document?*
                        
                        > *Summarize the concept of Transformers as explained in the document*
                        """
                        )
        if st.button("Preview example pdf"):
            st.markdown(f"##### Preview of {st.session_state['example_file'].name}")
            try:
                show_pdf_preview_with_fallback(st.session_state["example_file"])
            except Exception as e:
                st.warning("Preview unavailable.")
        
        clear_db()
    
    st.markdown("---")

    # ------------------------------
    # Metrics Section
    # ------------------------------
    st.subheader("📊 Live Metrics")

    col1, col2 = st.columns(2)
    with col1:
        st.metric(label=":blue[Latency]", value=f"{st.session_state.latency:.2f}s", help="Time taken to generate the LAST response.")
    with col2:
        st.metric(label=":blue[Session Cost]", value=f"${st.session_state.usd:.4f}", help="Estimated cost for the entire session calculated using %.4f per 1K input tokens and %.4f per 1K output tokens" % (COST_PER_1K_INPUT, COST_PER_1K_OUTPUT))
    
    st.caption("💡 *Metrics update in real-time to track API costs.*")
    st.caption(f"Token Usage: :blue[{st.session_state.total_tokens}] total tokens used.", help="Total number of tokens consumed in the entire session.")

    with st.expander(":blue[🔎 Token Usage Breakdown:]"):        
        st.markdown(f"#### :blue[Last Interaction Tokens:] {st.session_state.total_tokens_last} (RAG tokens: {st.session_state.RAG_total_tokens_last})")
        col1, col2 = st.columns(2)
        col1.metric("Input Tokens", f"{st.session_state.input_tokens_last} ({st.session_state.RAG_input_tokens_last})")
        col2.metric("Output Tokens", f"{st.session_state.output_tokens_last} ({st.session_state.RAG_output_tokens_last})")
        st.metric("Last Est. Cost (USD)", f"${st.session_state.usd_last:.4f}", help="Estimated cost for the last interaction.")

        st.markdown(f"#### :blue[Session Total Tokens:] {st.session_state.total_tokens} (RAG tokens: {st.session_state.RAG_total_tokens})")
        col3, col4 = st.columns(2)
        col3.metric("Total Input", f"{st.session_state.total_input_tokens} ({st.session_state.RAG_input_tokens})")
        col4.metric("Total Output", f"{st.session_state.total_output_tokens} ({st.session_state.RAG_output_tokens})")
        st.metric("Total Est. Cost (USD)", f"${st.session_state.usd:.4f}", help="Estimated cost for the entire session.")

    st.markdown("---")

    # ------------------------------
    # Reasoning Steps Section
    # ------------------------------
    st.subheader("🧠 Agent's Reasoning Steps:")
    if not st.session_state.trace:
        st.caption("No tool usage in the last turn.")
    else:
        for step in st.session_state.trace:
            if step["type"] == "tool_call":
                with st.expander(f"🛠️ Calling Tool: `{step['tool']}`"):
                    st.markdown("**Tool Input:**")
                    st.code(json.dumps(step['tool_input'], indent=2), language="json")
            elif step["type"] == "tool_output":
                with st.expander(f"👀 Observation from `{step['tool']}`"):
                    obs = str(step['observation'])
                    if len(obs) > 1000:
                        st.markdown(obs[:1000] + "...")
                    else:
                        st.markdown(obs)
    
    st.markdown("---")

    st.markdown("[View Source Code](https://github.com/danielecelsa/pdf-researcher) • Developed by **[Daniele Celsa](https://www.domenicodanielecelsa.com)**")

# ------------------------------
# Render chat
# ------------------------------
for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.write(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.write(msg.content)
    else:
        content = getattr(msg, "content", None) or str(msg)
        with st.chat_message("assistant"):
            st.write(content)