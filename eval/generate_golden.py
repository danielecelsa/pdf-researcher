"""
B.1 — Golden set generation via RAGAS TestsetGenerator (workstream B: RAG eval).

Phase 1a: single corpus document (example_docs/llm_introduction.pdf), single-hop
queries only. Multi-hop needs cross-document relationships in the knowledge graph,
so it comes later once 1-2 topic-related docs are added.

Notes:
- ragas is pinned to 0.2.15 (see requirements-eval.txt): ragas >= 0.3 requires
  langchain 1.x, which is incompatible with the app's pinned langchain 0.3.x stack.
- Generator model = gemini-2.5-flash (same family as the system; the self-judging
  caveat is a known, deliberate trade-off for this learning exercise).
- Output: eval/golden_raw.jsonl (RAW, pre-curation). The curated set -> eval/golden.jsonl.
- Cost is tracked locally via a token callback AND traced to the LangSmith project
  'pdf-researcher-eval' (EU) so the eval spend stays separate from the app project.

Usage: .venv/bin/python eval/generate_golden.py [testset_size]
"""
import os
import sys
import glob
import json
from pathlib import Path

# Route eval traces to a SEPARATE LangSmith project (keep the app's project clean).
# Must be set before anything triggers load_dotenv (python-dotenv does not override
# existing env vars), so this value wins over LANGSMITH_PROJECT in .env.
os.environ["LANGSMITH_PROJECT"] = "pdf-researcher-eval"

# eval/ lives under the repo root; put the root on sys.path so agent_core/helpers import.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import nest_asyncio
nest_asyncio.apply()

from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.tracers.langchain import wait_for_all_tracers

from ragas.testset import TestsetGenerator
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.testset.synthesizers import default_query_distribution
from ragas.run_config import RunConfig

# Reuse the single source of config (model id, api key, sample path)
from agent_core import MODEL, GOOGLE_API_KEY, SAMPLE_PDF_PATH
from helpers import TokenUsageCallbackHandler, compute_cost

REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(REPO_ROOT)  # PyPDFLoader uses the repo-relative sample path
EVAL_DIR = REPO_ROOT / "eval"
EVAL_DIR.mkdir(exist_ok=True)
RAW_OUT = EVAL_DIR / "golden_raw.jsonl"

# Real Gemini 2.5-flash prices per 1K tokens (same values used in the app's A.4 FinOps).
COST_IN, COST_OUT = 0.0003, 0.0025


CORPUS_DIR = REPO_ROOT / "eval" / "corpus"


def _is_content_page(text):
    """Generic, multi-doc content filter for question GENERATION.

    Drops near-empty covers, tables of contents, and the NVIDIA legal boilerplate
    page (Notice / Terms of Sale / Trademarks) — that legal page is dense with strong
    NER entities and otherwise hijacks RAGAS' entity-driven synthesizer. Substantive
    content of every corpus doc is kept. (The app still ingests full PDFs; only
    question generation is filtered.)
    """
    t = text.strip()
    if len(t) < 200:
        return False  # cover / near-empty page
    if "Table of Contents" in text:
        return False
    if "Notice" in text and "Terms of Sale" in text:
        return False  # NVIDIA legal notice / terms of sale / trademarks
    return True


def main():
    testset_size = int(sys.argv[1]) if len(sys.argv) > 1 else 20

    # Multi-doc corpus: related docs (LLM guide + Attention + BERT) share entities
    # (Transformer, BERT, attention) so the RAGAS knowledge graph forms clusters,
    # which single-hop needs for entities and multi-hop needs for relationships.
    pdf_paths = sorted(glob.glob(str(CORPUS_DIR / "*.pdf")))
    if not pdf_paths:
        raise SystemExit(f"No PDFs found in {CORPUS_DIR}")

    docs = []
    total_pages = 0
    for p in pdf_paths:
        pages = PyPDFLoader(p).load()
        kept = [d for d in pages if _is_content_page(d.page_content)]
        total_pages += len(pages)
        docs.extend(kept)
        print(f"  {os.path.basename(p)}: {len(pages)} pages -> {len(kept)} content pages")
    print(f"Corpus: {len(pdf_paths)} docs, {total_pages} pages, {len(docs)} content pages kept")

    token_cb = TokenUsageCallbackHandler()
    chat = ChatGoogleGenerativeAI(
        model=MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=0.2,
        transport="rest",
        callbacks=[token_cb],  # captures token usage across ALL ragas internal calls
    )
    emb = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=GOOGLE_API_KEY,
    )
    gen_llm = LangchainLLMWrapper(chat)
    gen_emb = LangchainEmbeddingsWrapper(emb)

    generator = TestsetGenerator(llm=gen_llm, embedding_model=gen_emb)

    # RAGAS default mix: single-hop-specific (entities) + multi-hop abstract (themes)
    # + multi-hop specific. The theme-based abstract synthesizer suits the conceptual
    # content (transformers, RNNs, BERT...), where pure entity-driven single-hop finds
    # too few named entities. Multi-hop stays within this one doc via KG relationships.
    query_distribution = default_query_distribution(gen_llm)
    print("Query distribution:", [(s.__class__.__name__, round(w, 3)) for s, w in query_distribution])

    run_config = RunConfig(max_workers=4)  # bound concurrency -> gentler on rate limits

    print(f"Generating {testset_size} cases over the corpus (single + multi-hop; several minutes)...")
    testset = generator.generate_with_langchain_docs(
        docs,
        testset_size=testset_size,
        query_distribution=query_distribution,
        run_config=run_config,
        raise_exceptions=False,  # skip problematic nodes instead of aborting the whole run
    )

    df = testset.to_pandas()
    print(f"Generated {len(df)} rows. Columns: {list(df.columns)}")

    n = 0
    with open(RAW_OUT, "w", encoding="utf-8") as f:
        for i, row in df.iterrows():
            rec = {
                "id": f"golden_raw_{i:03d}",
                "question": row.get("user_input"),
                "reference_answer": row.get("reference"),
                "reference_contexts": list(row.get("reference_contexts") or []),
                "synthesizer": row.get("synthesizer_name"),
                "source": "eval/corpus (llm_introduction + attention + bert)",
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    print(f"Wrote {n} raw cases -> {RAW_OUT}")

    usage = token_cb.get_usage_dict()
    cost = compute_cost(usage.get("input_tokens", 0), usage.get("output_tokens", 0), COST_IN, COST_OUT)
    print("\n=== COST (LLM calls, via token callback) ===")
    print(f"  input_tokens : {usage.get('input_tokens', 0)}")
    print(f"  output_tokens: {usage.get('output_tokens', 0)}")
    print(f"  total_tokens : {usage.get('total_tokens', 0)}")
    print(f"  est. cost    : ${cost:.4f}  (Gemini 2.5-flash 0.30/2.50 per 1M; embeddings excluded, negligible)")
    print("  (also traced to LangSmith project 'pdf-researcher-eval', EU)")

    wait_for_all_tracers()


if __name__ == "__main__":
    main()
