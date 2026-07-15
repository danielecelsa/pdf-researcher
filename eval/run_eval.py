"""
B.2 — RAG evaluation runner (two axes) via langsmith.evaluate().

Runs the app's real retrieval pipeline (agent_core.build_retrievers / answer_from_retriever)
over the LangSmith Dataset 'pdf-researcher-golden' and records the results as a comparable
Experiment in the separate LangSmith project 'pdf-researcher-eval' (EU).

Two axes, measured separately (a RAG has two independent failure points):

  RETRIEVAL (deterministic, no LLM):
    - recall@5              : fraction of the golden reference_contexts covered by >=1 of the
                             top-5 reranked chunks (a missed gold passage = unanswerable).
    - precision@5          : fraction of the 5 reranked chunks that hit any reference_context.
    - recall@10_prererank  : recall over the 10 ensemble candidates BEFORE FlashRank — the gap
                             vs recall@5 shows what reranking keeps/drops.
    Coverage is text overlap (rapidfuzz partial_ratio) between an 800-char app chunk and the
    (longer, page-sized) golden reference_context, threshold OVERLAP_THRESHOLD. This is why the
    golden set stores reference_contexts as passages, not chunk-ids: the metric survives a
    chunking A/B (workstream B.3).

  GENERATION (LLM-judge, RAGAS 0.2.15, gemini-2.5-flash):
    - faithfulness       : answer grounded in the retrieved contexts (reference-free).
    - answer_relevancy   : answer addresses the question (reference-free).
    - answer_correctness : answer vs the golden reference_answer (reference-based).
    Evaluated on the `research` tool's answer (the RAG output proper), grounded in exactly the
    contexts retrieved in the same invocation.

Caveat (deliberate for this exercise): the system and the judge are the same model
(gemini-2.5-flash) -> known self-judging bias; in production the judge is a different/stronger
model. Stated openly.

Usage:
  .venv/bin/python eval/run_eval.py --limit 3   # dry-run on N cases (cheap; validate wiring/cost)
  .venv/bin/python eval/run_eval.py             # full run (all 17 cases) [spends API]
"""
import os
import sys
import glob
import argparse
import asyncio
from pathlib import Path
from collections import defaultdict, Counter

# Route eval traces/experiments to a SEPARATE LangSmith project (keep the app's project clean).
# Must be set before anything triggers load_dotenv (python-dotenv does not override existing
# env vars), so this wins over LANGSMITH_PROJECT in .env.
os.environ["LANGSMITH_PROJECT"] = "pdf-researcher-eval"

# eval/ lives under the repo root; put the root on sys.path so agent_core/helpers import.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

import nest_asyncio
nest_asyncio.apply()

from rapidfuzz import fuzz
from langsmith import Client, evaluate

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import Faithfulness, ResponseRelevancy, AnswerCorrectness
from ragas.dataset_schema import SingleTurnSample
from ragas.run_config import RunConfig

from agent_core import (
    MODEL,
    GOOGLE_API_KEY,
    build_retrievers,
    answer_from_retriever,
    ingest_documents,
)
from helpers import TokenUsageCallbackHandler, compute_cost

# ------------------------------
# Config
# ------------------------------
DATASET_NAME = "pdf-researcher-golden"
CORPUS_DIR = REPO_ROOT / "eval" / "corpus"
EVAL_COLLECTION = "pdf_researcher_eval"
EVAL_SESSION = "eval"

OVERLAP_THRESHOLD = 90  # partial_ratio; calibrated on the golden set (see eval/README.md)

# Real Gemini 2.5-flash prices per 1K tokens (same as the app's A.4 FinOps).
COST_IN, COST_OUT = 0.0003, 0.0025

# Shared, in-process accumulators (runner uses max_concurrency=1 -> sequential, race-free).
SCORES = defaultdict(dict)   # example_id -> {metric_key: score}
META = {}                    # example_id -> {"question", "synthesizer"}
TOKEN_CB = TokenUsageCallbackHandler()


# ------------------------------
# Corpus ingestion (fresh in-memory Chroma per process = reproducible)
# ------------------------------
class _DiskFile:
    """Minimal file-shim (.name + .read()) so ingest_documents can read corpus PDFs from disk."""
    def __init__(self, path: Path):
        self._p = Path(path)
        self.name = self._p.name

    def read(self) -> bytes:
        return self._p.read_bytes()


def ingest_corpus():
    pdfs = sorted(glob.glob(str(CORPUS_DIR / "*.pdf")))
    if not pdfs:
        raise SystemExit(f"No corpus PDFs in {CORPUS_DIR}. Run: .venv/bin/python eval/fetch_corpus.py")
    files = [_DiskFile(p) for p in pdfs]
    print(f"Ingesting {len(files)} corpus docs into Chroma '{EVAL_COLLECTION}' (in-memory)...")
    res = ingest_documents(files, collection_name=EVAL_COLLECTION, session_id=EVAL_SESSION)
    if not res:
        raise SystemExit("Ingestion produced no chunks.")
    n_files, n_chunks = res
    print(f"  ingested {n_files} docs -> {n_chunks} chunks")


# ------------------------------
# Target: run the RAG for one golden question
# ------------------------------
def target(inputs: dict) -> dict:
    """Return {answer, contexts (top-5 reranked), candidates (10 pre-rerank)} for a question."""
    question = inputs["question"]
    base_retriever, compression_retriever = build_retrievers(EVAL_COLLECTION, EVAL_SESSION)

    # One invocation drives both axes: reranked context (retrieval) + synthesized answer (generation).
    resp = answer_from_retriever(compression_retriever, question, callbacks=[TOKEN_CB])
    contexts = [d.page_content for d in resp.get("context", [])]

    # Pre-rerank ensemble candidates (deterministic; for recall@10_prererank).
    candidates = [d.page_content for d in base_retriever.invoke(question)]

    return {"answer": resp.get("answer", ""), "contexts": contexts, "candidates": candidates}


# ------------------------------
# Retrieval evaluators (deterministic overlap)
# ------------------------------
def _norm(s: str) -> str:
    return " ".join((s or "").lower().split())


def _covers(chunk: str, ref_ctx: str) -> bool:
    """A chunk 'covers' a reference_context if the (shorter) chunk is ~contained in it."""
    return fuzz.partial_ratio(_norm(chunk), _norm(ref_ctx)) >= OVERLAP_THRESHOLD


def _record(example, key, score):
    eid = str(example.id)
    SCORES[eid][key] = score
    if eid not in META:
        META[eid] = {
            "question": (example.inputs or {}).get("question", ""),
            "synthesizer": (example.metadata or {}).get("synthesizer", "?"),
        }


def _recall(retrieved, refs):
    if not refs:
        return None
    covered = sum(1 for g in refs if any(_covers(c, g) for c in retrieved))
    return covered / len(refs)


def recall_at_5(run, example):
    refs = (example.outputs or {}).get("reference_contexts", [])
    score = _recall(run.outputs.get("contexts", []), refs)
    _record(example, "recall@5", score)
    return {"key": "recall@5", "score": score}


def precision_at_5(run, example):
    contexts = run.outputs.get("contexts", [])
    refs = (example.outputs or {}).get("reference_contexts", [])
    score = None if not contexts else sum(1 for c in contexts if any(_covers(c, g) for g in refs)) / len(contexts)
    _record(example, "precision@5", score)
    return {"key": "precision@5", "score": score}


def recall_at_10_prererank(run, example):
    refs = (example.outputs or {}).get("reference_contexts", [])
    score = _recall(run.outputs.get("candidates", []), refs)
    _record(example, "recall@10_prererank", score)
    return {"key": "recall@10_prererank", "score": score}


# ------------------------------
# Generation evaluators (RAGAS LLM-judge)
# ------------------------------
def _build_ragas_metrics():
    # Judge at temperature 0 for stability; same model family as the system (self-judging caveat).
    judge_chat = ChatGoogleGenerativeAI(
        model=MODEL, google_api_key=GOOGLE_API_KEY, temperature=0, transport="rest",
        callbacks=[TOKEN_CB],  # capture judge token usage alongside synthesis usage
    )
    judge_emb = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=GOOGLE_API_KEY)
    r_llm = LangchainLLMWrapper(judge_chat)
    r_emb = LangchainEmbeddingsWrapper(judge_emb)
    metrics = {
        "faithfulness": Faithfulness(llm=r_llm),
        "answer_relevancy": ResponseRelevancy(llm=r_llm, embeddings=r_emb),
        "answer_correctness": AnswerCorrectness(llm=r_llm, embeddings=r_emb),
    }
    # init() wires up prompts/run_config and builds sub-metrics (AnswerCorrectness needs its
    # AnswerSimilarity component built from the embeddings; single_turn_ascore alone won't).
    run_config = RunConfig()
    for m in metrics.values():
        m.init(run_config)
    return metrics


_METRICS = None


def _metrics():
    global _METRICS
    if _METRICS is None:
        _METRICS = _build_ragas_metrics()
    return _METRICS


def _ascore(metric, sample: SingleTurnSample):
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return float(loop.run_until_complete(metric.single_turn_ascore(sample)))


def faithfulness_eval(run, example):
    sample = SingleTurnSample(
        user_input=(example.inputs or {}).get("question", ""),
        response=run.outputs.get("answer", ""),
        retrieved_contexts=run.outputs.get("contexts", []),
    )
    score = _ascore(_metrics()["faithfulness"], sample)
    _record(example, "faithfulness", score)
    return {"key": "faithfulness", "score": score}


def answer_relevancy_eval(run, example):
    sample = SingleTurnSample(
        user_input=(example.inputs or {}).get("question", ""),
        response=run.outputs.get("answer", ""),
        retrieved_contexts=run.outputs.get("contexts", []),
    )
    score = _ascore(_metrics()["answer_relevancy"], sample)
    _record(example, "answer_relevancy", score)
    return {"key": "answer_relevancy", "score": score}


def answer_correctness_eval(run, example):
    sample = SingleTurnSample(
        user_input=(example.inputs or {}).get("question", ""),
        response=run.outputs.get("answer", ""),
        reference=(example.outputs or {}).get("reference_answer", ""),
    )
    score = _ascore(_metrics()["answer_correctness"], sample)
    _record(example, "answer_correctness", score)
    return {"key": "answer_correctness", "score": score}


RETRIEVAL_KEYS = ["recall@5", "precision@5", "recall@10_prererank"]
GENERATION_KEYS = ["faithfulness", "answer_relevancy", "answer_correctness"]
ALL_KEYS = RETRIEVAL_KEYS + GENERATION_KEYS


# ------------------------------
# Report
# ------------------------------
def _mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def _fmt(v):
    return " n/a " if v is None else f"{v:5.2f}"


def print_report(experiment_name):
    ids = list(SCORES.keys())
    print("\n" + "=" * 100)
    print("RAG EVAL — 2 axes  |  overlap threshold =", OVERLAP_THRESHOLD, " | judge = gemini-2.5-flash (self-judging caveat)")
    print("=" * 100)

    header = f"{'id':<11}{'synth':<26}" + "".join(f"{k:>19}" for k in ALL_KEYS)
    print(header)
    print("-" * len(header))
    for eid in ids:
        m = META.get(eid, {})
        synth = m.get("synthesizer", "?").replace("_query_synthesizer", "")
        row = f"{eid[:10]:<11}{synth[:25]:<26}" + "".join(f"{_fmt(SCORES[eid].get(k)):>19}" for k in ALL_KEYS)
        print(row)

    print("-" * len(header))
    overall = f"{'MEAN (all)':<37}" + "".join(f"{_fmt(_mean([SCORES[e].get(k) for e in ids])):>19}" for k in ALL_KEYS)
    print(overall)

    # Breakdown by synthesizer family (single-hop vs multi-hop is where recall discriminates).
    groups = defaultdict(list)
    for eid in ids:
        groups[META.get(eid, {}).get("synthesizer", "?")].append(eid)
    print("\nBy synthesizer:")
    for synth, eids in sorted(groups.items()):
        label = synth.replace("_query_synthesizer", "")
        line = f"  {label:<33}(n={len(eids):>2})  " + "  ".join(
            f"{k}={_fmt(_mean([SCORES[e].get(k) for e in eids]))}" for k in ALL_KEYS
        )
        print(line)

    # Cost (local estimate; LangSmith has the authoritative per-run cost with real prices).
    usage = TOKEN_CB.get_usage_dict()
    cost = compute_cost(usage.get("input_tokens", 0), usage.get("output_tokens", 0), COST_IN, COST_OUT)
    print("\n=== COST (synthesis + RAGAS judge, via token callback; embeddings excluded) ===")
    print(f"  input_tokens : {usage.get('input_tokens', 0)}")
    print(f"  output_tokens: {usage.get('output_tokens', 0)}")
    print(f"  total_tokens : {usage.get('total_tokens', 0)}")
    print(f"  est. cost    : ${cost:.4f}  (Gemini 2.5-flash 0.30/2.50 per 1M)")
    print(f"\nExperiment: {experiment_name}  (LangSmith project 'pdf-researcher-eval', EU)")
    print("=" * 100)


# ------------------------------
# Main
# ------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="run only the first N golden cases (dry-run)")
    args = ap.parse_args()

    ingest_corpus()

    client = Client()
    if args.limit:
        data = list(client.list_examples(dataset_name=DATASET_NAME, limit=args.limit))
        print(f"DRY-RUN on {len(data)} case(s).")
    else:
        data = DATASET_NAME
        print("FULL run on the whole dataset.")

    evaluators = [
        recall_at_5, precision_at_5, recall_at_10_prererank,
        faithfulness_eval, answer_relevancy_eval, answer_correctness_eval,
    ]

    results = evaluate(
        target,
        data=data,
        evaluators=evaluators,
        experiment_prefix="pdf-researcher-eval",
        metadata={
            "axes": "retrieval+generation",
            "judge": MODEL,
            "overlap_threshold": OVERLAP_THRESHOLD,
            "k_rerank": 5,
        },
        max_concurrency=1,  # sequential -> deterministic ordering + accurate local cost tally
        client=client,
        blocking=True,
    )

    experiment_name = getattr(results, "experiment_name", "(see LangSmith)")
    print_report(experiment_name)


if __name__ == "__main__":
    main()
