# agent_core.py
# Core Agentic RAG logic (LangGraph agent + hybrid retrieval + ingestion),
# decoupled from Streamlit so it can be imported by the app shell (app.py),
# the eval runner (workstream B) and tests (C1) WITHOUT a Streamlit runtime.
#
# The resources that used to be @st.cache_resource (Chroma client, BM25 store,
# LLM, prompt, checkpointer) are process-level singletons here (functools.lru_cache):
# same semantics as st.cache_resource for these no-arg resources (one instance per
# process, shared across sessions and threads), so ingestion and the `research`
# tool share the same in-memory Chroma client + BM25 store even when research runs
# on the background asyncio thread.

import os
import asyncio
import tempfile
import time
import uuid
from functools import lru_cache

from dotenv import load_dotenv

# LangGraph / LangChain Core
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import StructuredTool
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    PromptTemplate,
)
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.callbacks import Callbacks

# Vector DB: Chroma integration
import chromadb
from langchain_chroma import Chroma
from chromadb.utils.embedding_functions.chroma_langchain_embedding_function import create_langchain_embedding

from logging_config import get_logger

from prompts import RAG_AGENT_SYSTEM_PROMPT, RAG_RETRIEVAL_PROMPT

# Load environment variables from .env file if not in a rendering environment
if os.getenv("RENDER") != "true":
    load_dotenv()

# ------------------------------
# Configuration
# ------------------------------
MODEL = os.environ.get("GENAI_MODEL", "gemini-2.5-flash")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY_2")

SAMPLE_PDF_PATH = "example_docs/llm_introduction.pdf"

# ------------------------------
# Loggers
# ------------------------------
logger_local = get_logger("local")
logger_all = get_logger("all")


# ------------------------------
# Process-level singletons (were @st.cache_resource in the monolith)
# ------------------------------
@lru_cache(maxsize=None)
def get_chroma_client():  # default: in-process, no persistence
    """Get or create a Chroma client instance (singleton, shared across sessions/threads)."""
    client = chromadb.Client()
    logger_local.info("Collections_lru: %s", client.list_collections())
    return client


@lru_cache(maxsize=None)
def get_bm25_store():
    """Thread-safe global BM25 doc store: { session_id: [List of Documents] } (singleton)."""
    # This dictionary lives in the global memory of the server
    return {}


@lru_cache(maxsize=None)
def get_prompt():
    system = SystemMessagePromptTemplate.from_template(RAG_AGENT_SYSTEM_PROMPT)

    hist = MessagesPlaceholder(variable_name="messages")
    prompt = ChatPromptTemplate.from_messages([system, hist])
    return prompt


@lru_cache(maxsize=None)
def get_llm():
    try:
        llm = ChatGoogleGenerativeAI(
            model=MODEL,
            google_api_key=GOOGLE_API_KEY,
            temperature=0.2,
            safety_settings=None,
            transport="rest"
        )
    except Exception as e:
        logger_all.exception("Could not initialize LLM: %s", e)
        llm = None
    return llm


@lru_cache(maxsize=None)
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


# ------------------------------
# Retrieval pipeline (single source of truth: research tool AND eval runner)
# ------------------------------
# Useful to have many sessions with isolated collections. We are not using persistence here, but you could.

def _ensure_event_loop():
    """Google GenAI uses async clients and expects an event loop in the current thread."""
    try:
        asyncio.get_event_loop()
    except Exception:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)


def build_retrievers(collection_name: str, session_id: str):
    """Build the session's retrieval pipeline: the hybrid BM25 + Chroma ensemble (pre-rerank)
    and the FlashRank-reranked compression retriever (top_n=5).

    This is the single source of truth for retrieval: the `research` tool and the offline
    eval runner (eval/run_eval.py) both go through here, so evaluation exercises exactly the
    app's retrieval pipeline.

    Returns:
        (base_retriever, compression_retriever). base_retriever is the ensemble (10 candidates:
        BM25 k=5 + Chroma k=5), or a pure-vector fallback if the session's BM25 store is empty.
        compression_retriever wraps it with FlashRank, keeping the top 5.
    """
    _ensure_event_loop()

    # Embeddings (explicit key)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=GOOGLE_API_KEY)

    # Vector retriever (Chroma), k=5
    client = get_chroma_client()  # in-process ephemeral
    logger_all.info("Collections_tool: %s ", collection_name)
    logger_local.info("Collections_tool_list: %s", client.list_collections())
    vect = Chroma(collection_name=collection_name, embedding_function=embeddings, client=client)
    chroma_retriever = vect.as_retriever(search_kwargs={"k": 5})

    # Hybrid search: BM25 (raw text from the global store, keyed by the closure's session_id)
    # ensembled 50/50 with the vector retriever.
    bm25_store = get_bm25_store()
    current_docs = bm25_store.get(session_id, [])  # Read from the global dictionary

    if current_docs:
        logger_local.info("Building BM25 Index from %d chunks (Global Store)...", len(current_docs))
        bm25_retriever = BM25Retriever.from_documents(current_docs)
        bm25_retriever.k = 5

        base_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, chroma_retriever],
            weights=[0.5, 0.5],
        )
        logger_local.info("✅ Hybrid Search Activated (BM25 + Chroma)")
    else:
        # If no docs for BM25, fallback to pure vector search
        base_retriever = chroma_retriever
        logger_local.warning(f"⚠️ BM25 docs missing for session {session_id}. Using pure Vector Search.")

    # Rerank: FlashRank cross-encoder, compress the ensemble candidates down to top 5.
    compressor = FlashrankRerank(
        model="ms-marco-MiniLM-L-12-v2",
        top_n=5)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever,
    )
    return base_retriever, compression_retriever


def answer_from_retriever(compression_retriever, query: str, callbacks: Callbacks = None) -> dict:
    """Run retrieval + synthesis over a prebuilt compression retriever and return the raw
    LangChain dict ({input, context, answer}). `context` = the reranked documents actually used.

    Kept separate from build_retrievers so callers (e.g. the eval runner) can also read the
    pre-rerank candidates from the base retriever without rebuilding the pipeline.
    """
    rag_prompt = PromptTemplate.from_template(RAG_RETRIEVAL_PROMPT)
    llm = ChatGoogleGenerativeAI(model=MODEL, google_api_key=GOOGLE_API_KEY, temperature=0.2, transport="rest")
    doc_chain = create_stuff_documents_chain(llm, rag_prompt)
    rag_chain = create_retrieval_chain(compression_retriever, doc_chain)
    return rag_chain.invoke({"input": query}, config={"callbacks": callbacks})


def run_rag_chain(collection_name: str, session_id: str, query: str, callbacks: Callbacks = None) -> dict:
    """Full app retrieval + synthesis path: build the session retrievers and answer `query`.
    Returns the raw dict {input, context, answer}. The `research` tool returns only `answer`;
    the eval runner also consumes `context` for deterministic recall@k / precision@k.
    """
    _, compression_retriever = build_retrievers(collection_name, session_id)
    return answer_from_retriever(compression_retriever, query, callbacks=callbacks)


# ------------------------------
# TOOL: research (uses Chroma) — thin wrapper over run_rag_chain
# ------------------------------
def research_factory(collection_name: str, session_id: str):
    """Factory to create a research tool bound to a specific Chroma collection/session of the user."""
    def research(query: str, callbacks: Callbacks = None) -> str:
        """
        Use this tool to retrieve and summarize information from the documents (PDFs or TXTs) uploaded by the user, and answer the user's question.

        Use this tool whenever the user's question involves the uploaded documents,
        even if the question is only partially related to their content.
        Do not use this tool for general knowledge questions unrelated to the uploaded documents.

        Uses Hybrid Search (Vector + Keyword) and Reranking for high accuracy.

        Args:
            query (str): The user question to be answered using only the uploaded documents.

        Returns:
            str: A concise, evidence-based answer derived exclusively from the uploaded documents.
        """
        logger_all.info("TOOL CALLED with: %s", query)

        try:
            resp = run_rag_chain(collection_name, session_id, query, callbacks=callbacks)

            # Log the final reranked context
            if isinstance(resp, dict) and "context" in resp:
                logger_local.info("--- FINAL RERANKED CONTEXT ---")
                for i, doc in enumerate(resp["context"]):
                    logger_local.info(f"RERANKED #{i+1}: {doc.page_content[:60]}...")

            # Normalize response
            if isinstance(resp, dict):
                result = resp.get("answer") or resp.get("text") or str(resp)
                logger_local.info("Tool's answer: %s", result)
                return result
            logger_local.info("Tool's answer (fallback): %s", resp)
            return str(resp)

        except Exception as e:
            logger_all.exception("RAG chain invocation failed: %s", e)
            return "I could not run the retrieval chain due to an internal error."

    return research


# ------------------------------
# Build agent
# ------------------------------
def build_agent(collection_name, session_id):

    research_tool_function = research_factory(collection_name, session_id)

    tools = [
        StructuredTool.from_function(
            research_tool_function,
            name="research",
            description=(
                "Use this tool to answer questions that require information from the uploaded PDF/text documents. "
                "Always call this tool when the user's question refers to facts, dates, quotes, or content contained in the uploaded files."
                "The tool accepts a single string question and returns a concise, evidence-based answer."
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


# ------------------------------
# Ingestion (Streamlit-free core of the old update_vector_db)
# ------------------------------
def ingest_documents(files, collection_name, session_id, progress_cb=None, on_file_error=None):
    """Pure ingestion: load PDF/TXT -> chunk -> BM25 global store + Chroma. No Streamlit.

    Args:
        files: iterable of file-like objects exposing .name and .read()
               (Streamlit UploadedFile, or any shim with the same surface).
        collection_name: Chroma collection to write into.
        session_id: key used for the per-session BM25 doc store.
        progress_cb: optional callable(fraction: float) invoked during the Chroma
                     write phase (0.0 first, then per batch up to 1.0).
        on_file_error: optional callable(filename, exc, kind) where kind is
                       "load" (PDF parse error) or "generic" (per-file error);
                       lets the caller surface UI feedback without coupling to it.

    Returns:
        (files_count, chunk_count) on success, or None if there were no files /
        no valid documents extracted.
    """
    if not files:
        return None

    # ensure event loop for embeddings init (Google GenAI uses async clients)
    try:
        asyncio.get_event_loop()
    except Exception:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=GOOGLE_API_KEY)

    raw_docs = []

    # Iterate on files passed as argument
    for f in files:
        try:
            # write temp file and load via PyPDFLoader
            if f.name.lower().endswith(".pdf"):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(f.read())
                    tmp_path = tmp.name

                try:
                    loader = PyPDFLoader(tmp_path)
                    docs = loader.load()
                    # attach source metadata
                    for d in docs:
                        d.metadata = d.metadata or {}
                        d.metadata["source"] = f.name
                    raw_docs.extend(docs)
                except Exception as e:
                    # PDF Corrupted: report the error but do not interrupt everything
                    if on_file_error:
                        on_file_error(f.name, e, "load")
                    continue
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)

            elif f.name.lower().endswith(".txt"):
                content = f.read().decode("utf-8") if hasattr(f, "read") else f
                raw_docs.append(Document(page_content=content, metadata={"source": f.name}))

        except Exception as e:
            if on_file_error:
                on_file_error(f.name, e, "generic")

    if not raw_docs:
        # Either there were no files, or they were all corrupted
        return None

    # Split and Load into Chroma
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    final_docs = text_splitter.split_documents(raw_docs)

    # Save docs for BM25
    bm25_store = get_bm25_store()

    if session_id not in bm25_store:
        bm25_store[session_id] = []

    bm25_store[session_id].extend(final_docs)

    # STORE FOR VECTORS (Chroma)
    chroma_client = get_chroma_client()
    emb_chroma = create_langchain_embedding(embeddings)
    coll = chroma_client.get_or_create_collection(name=collection_name, embedding_function=emb_chroma)

    # Batch processing to avoid rate limits (even with tier 1, good practice)
    batch_size = 20  # Bigger because of Tier 1
    total_docs = len(final_docs)

    if progress_cb:
        progress_cb(0.0)

    for i in range(0, total_docs, batch_size):
        batch = final_docs[i : i + batch_size]
        ids = [f"{collection_name}::{uuid.uuid4()}" for _ in batch]
        coll.add(ids=ids, documents=[d.page_content for d in batch], metadatas=[d.metadata for d in batch])
        # Update progress
        if progress_cb:
            progress_cb(min((i + batch_size) / total_docs, 1.0))
        time.sleep(0.3)

    files_count = len(files)
    logger_all.info("Updated Chroma & BM25 List with %d files, %d chunks.", files_count, len(final_docs))

    return files_count, len(final_docs)
