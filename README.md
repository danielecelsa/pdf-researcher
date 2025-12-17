# Enterprise Document Intelligence Agent 🧠

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Orchestration-orange.svg)](https://langchain-ai.github.io/langgraph/)
[![RAG-Hybrid](https://img.shields.io/badge/RAG-Hybrid%20%2B%20Rerank-green)]
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)](https://streamlit.io/)

A production-grade **Agentic RAG System** designed for granular document analysis. Unlike linear RAG pipelines, this system utilizes a ReAct agent loop to autonomously decide when to query the knowledge base, employing a State-of-the-Art retrieval stack (**Hybrid Search + Cross-Encoder Reranking**) to ensure high-precision answers with zero hallucinations.

---

## 🚀 Key Features

*   **Agentic Reasoning (ReAct Loop):** The system does not blindly retrieve data for every query. Using **LangGraph**, the agent evaluates the user intent:
    *   **General Chit-Chat:** Answered directly (Zero Retrieval Cost).
    *   **Document Queries:** Triggers the research tool only when necessary.
*   **SOTA Retrieval Pipeline:** Implements a sophisticated "*Retrieve & Rerank*" architecture to overcome the limitations of standard vector search:
    *   **Hybrid Search:** Combines **ChromaDB** (Semantic/Dense Vector Search) with **BM25** (Keyword/Sparse Search) via EnsembleRetriever. This ensures retrieval of both conceptual matches and specific alphanumeric codes/acronyms.
    *   **Reranking:** Applies **FlashRank** (Cross-Encoder) to the top 10 candidates to semantically re-score and extract the "Top 5" most relevant chunks, dramatically improving accuracy.
*   **FinOps & Observability:** Engineered for transparency in cost and performance:
    *   **Granular Token Tracking:** Distinguishes between Reasoning Cost (Agent thought process) and Retrieval Context Cost (RAG overhead).
    *   **Live Metrics:** Real-time tracking of Latency, Input/Output Tokens, and USD Cost per interaction.
    *   **Structured Logging:** Integration with Redis and BetterStack for remote monitoring.
*   **Enterprise-Grade Privacy:** Session-Scoped Storage: Uses ephemeral, in-memory vector stores isolated by `session_id`. Data is strictly segregated between users and wiped upon session termination.

## 🏗️ Architecture

graph TD
    User(User Query) --> Agent{Agent Router}
    
    %% Path 1: General Chat
    Agent -- "General Intent" --> LLM_Direct[LLM Response]
    
    %% Path 2: RAG
    Agent -- "Document Intent" --> Tool[Research Tool]
    
    subgraph "SOTA Retrieval Pipeline"
        Tool --> Hybrid[Ensemble Retriever]
        Hybrid --> Chroma[(ChromaDB\nVector Search)]
        Hybrid --> BM25[(BM25\nKeyword Search)]
        
        Chroma & BM25 --> Candidates[Top 10 Candidates]
        Candidates --> Rerank[FlashRank\nCross-Encoder]
        Rerank --> Top5[Top 5 Contexts]
    end
    
    Top5 --> Synthesis[LLM Synthesis]
    Synthesis --> Response(Final Answer)

## 🛠️ Tech Stack

*   **LLM:** Google Gemini 2.5 Flash (via langchain-google-genai).
*   **Orchestration:** LangGraph (Stateful Agentic Workflow).
*   **Vector Store:** ChromaDB (In-memory/Ephemeral).
*   **Retrieval:** rank_bm25 (Sparse), FlashRank (Reranking/Compression).
*   **App Framework:** Streamlit.
*   **Concurrency:** asyncio background loops for non-blocking agent execution.

## 🧪 Intallation & Usage

### Installation
```bash
git clone https://github.com/danielecelsa/pdf-researchert.git
cd pdf-researcher
pip install -r requirements.txt
```

### Environment Setup
Create a .env file:
```bash
GOOGLE_API_KEY=your_key_here
REDIS_URL=your_redis_url (optional)
```

### Running the App
```bash
streamlit run app.py
```
## 🧪 Test Scenarios

To validate the **Hybrid Search** capabilities, load the provided `example_docs/llm_introduction.pdf` and try these queries:

1.   **The "Semantic" Test (Vector Search):**
    *"Summarize the concept of Transformers."*

*Result:* The embedding model successfully retrieves conceptual definitions.

2.   **The "Needle in a Haystack" Test (BM25 + Rerank):**
*"What is the 'laryngeal descent theory' mentioned in the preface?"*

*Result:* Vector search often misses this minor detail. BM25 catches the specific keyword, and FlashRank promotes it to the top.

3.   **The "Specific Data" Test:**
*"How many parameters does the model 'Cohere xlarge v20220609' have?"*

*Result:* The system identifies the exact alphanumeric code v20220609 which pure vector search treats as noise.

## ⚙️ Engineering Highlights
*   **Thread-Safe Global Store Pattern:** One of the main challenges in using **Streamlit** with background agents involves the volatility of `session_state` across threads.
This project implements a **Thread-Safe Global Store** pattern using `@st.cache_resource` to bridge the gap between the UI thread (file upload) and the Agent thread (retrieval tool).

# From app.py
@st.cache_resource
def get_bm25_store():
    # A persistent, thread-safe dictionary living in global memory
    return {}

This ensures that the **BM25 index** (which requires raw text access) remains accessible to the background agent without raising `ContextMissing` errors or race conditions.

*   **Precise Cost Attribution:** A custom `TokenUsageCallbackHandler` intercepts the LLM streams to separate the cost of the *"Agent's Brain"* from the cost of the *"RAG Context"*. This allows for precise ROI calculation of the retrieval feature.

---

## 👨‍💻 Author
Daniele Celsa

*   [Portfolio Website](https://danielecelsa.github.io/portfolio/)
*   [LinkedIn](https://www.linkedin.com/in/domenico-daniele-celsa-518b758b/)