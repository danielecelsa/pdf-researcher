"""
CI eval-gate for the RAG. Fails (exit 1) when quality drops below thresholds.

Lightweight and self-contained: reads the versioned golden set (eval/golden.jsonl), ingests the
fixed corpus into a fresh in-memory Chroma, runs the app's real retrieval pipeline, and checks
metrics against floors. No LangSmith writes, no rich Experiment (that's eval/run_eval.py) — this
is meant to run fast and cheap in CI.

Modes:
  retrieval  (default): mean recall@5 over the golden set (deterministic; embeddings only).
                        Runs on every push.
  generation:           + mean faithfulness (RAGAS LLM-judge, gemini-2.5-flash). Runs on PRs.

Usage:
  python eval/gate.py --mode retrieval  --recall-floor 0.40
  python eval/gate.py --mode generation --recall-floor 0.40 --faithfulness-floor 0.80

Requires GOOGLE_API_KEY_2 (embeddings; and the judge in generation mode).
"""
import argparse
import asyncio
import glob
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from retrieval_metrics import recall_at_k

from agent_core import answer_from_retriever, build_retrievers, ingest_documents

GOLDEN = REPO_ROOT / "eval" / "golden.jsonl"
CORPUS_DIR = REPO_ROOT / "eval" / "corpus"
EVAL_COLLECTION = "pdf_researcher_gate"
EVAL_SESSION = "gate"
OVERLAP_THRESHOLD = 90


class _DiskFile:
    """Minimal file-shim (.name + .read()) so ingest_documents can read corpus PDFs from disk."""
    def __init__(self, path):
        self._p = Path(path)
        self.name = self._p.name

    def read(self):
        return self._p.read_bytes()


def _ingest_corpus():
    pdfs = sorted(glob.glob(str(CORPUS_DIR / "*.pdf")))
    if not pdfs:
        raise SystemExit(f"No corpus PDFs in {CORPUS_DIR}. Run eval/fetch_corpus.py first.")
    res = ingest_documents([_DiskFile(p) for p in pdfs],
                           collection_name=EVAL_COLLECTION, session_id=EVAL_SESSION)
    if not res:
        raise SystemExit("Ingestion produced no chunks.")
    print(f"Ingested {res[0]} docs -> {res[1]} chunks")


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else 0.0


def _ascore(metric, sample):
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return float(loop.run_until_complete(metric.single_turn_ascore(sample)))


def _faithfulness_mean(samples):
    """samples: list of (question, answer, contexts). Mean RAGAS faithfulness."""
    import nest_asyncio
    nest_asyncio.apply()
    from langchain_google_genai import ChatGoogleGenerativeAI
    from ragas.dataset_schema import SingleTurnSample
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import Faithfulness
    from ragas.run_config import RunConfig

    from agent_core import GOOGLE_API_KEY, MODEL

    judge = ChatGoogleGenerativeAI(model=MODEL, google_api_key=GOOGLE_API_KEY,
                                   temperature=0, transport="rest")
    metric = Faithfulness(llm=LangchainLLMWrapper(judge))
    metric.init(RunConfig())

    scores = []
    for question, answer, contexts in samples:
        sample = SingleTurnSample(user_input=question, response=answer, retrieved_contexts=contexts)
        scores.append(_ascore(metric, sample))
    return _mean(scores)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["retrieval", "generation"], default="retrieval")
    ap.add_argument("--recall-floor", type=float, default=0.40)
    ap.add_argument("--faithfulness-floor", type=float, default=0.80)
    args = ap.parse_args()

    rows = [json.loads(line) for line in open(GOLDEN, encoding="utf-8") if line.strip()]
    _ingest_corpus()

    recalls, gen_samples = [], []
    for r in rows:
        question = r["question"]
        base_retriever, compression_retriever = build_retrievers(EVAL_COLLECTION, EVAL_SESSION)
        if args.mode == "generation":
            resp = answer_from_retriever(compression_retriever, question)
            contexts = [d.page_content for d in resp.get("context", [])]
            gen_samples.append((question, resp.get("answer", ""), contexts))
        else:
            contexts = [d.page_content for d in compression_retriever.invoke(question)]
        recalls.append(recall_at_k(contexts, r.get("reference_contexts", []), OVERLAP_THRESHOLD))

    mean_recall = _mean(recalls)
    print(f"mean recall@5     = {mean_recall:.3f}   (floor {args.recall_floor})")

    failed = mean_recall < args.recall_floor
    if failed:
        print(f"FAIL: recall@5 {mean_recall:.3f} < floor {args.recall_floor}")

    if args.mode == "generation":
        mean_faith = _faithfulness_mean(gen_samples)
        print(f"mean faithfulness = {mean_faith:.3f}   (floor {args.faithfulness_floor})")
        if mean_faith < args.faithfulness_floor:
            print(f"FAIL: faithfulness {mean_faith:.3f} < floor {args.faithfulness_floor}")
            failed = True

    if failed:
        sys.exit(1)
    print("GATE PASSED")


if __name__ == "__main__":
    main()
