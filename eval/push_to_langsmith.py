"""
B.1 — Push the curated golden set (eval/golden.jsonl) to a LangSmith Dataset.

The jsonl in git stays the SSoT; LangSmith is the mirror/container so the B.2 runner
can drive `langsmith.evaluate()` against it and store each run as a comparable Experiment.

Idempotent: if the dataset already exists it is left untouched (re-runs won't duplicate).

Usage: .venv/bin/python eval/push_to_langsmith.py
"""
import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(REPO_ROOT)

from dotenv import load_dotenv

load_dotenv()  # LANGSMITH_API_KEY + LANGSMITH_ENDPOINT (EU) from .env

from langsmith import Client

GOLDEN = REPO_ROOT / "eval" / "golden.jsonl"
DATASET_NAME = "pdf-researcher-golden"
DESCRIPTION = (
    "RAG eval golden set — 17 curated cases (5 single-hop + 6 multi-hop-abstract + "
    "6 multi-hop-specific) generated with RAGAS over a 3-doc corpus (LLM beginner guide "
    "+ Attention Is All You Need + BERT). Each example: question -> reference_answer + "
    "reference_contexts. Model: gemini-2.5-flash."
)


def main():
    rows = [json.loads(line) for line in open(GOLDEN, encoding="utf-8") if line.strip()]
    client = Client()

    if client.has_dataset(dataset_name=DATASET_NAME):
        ds = client.read_dataset(dataset_name=DATASET_NAME)
        print(f"Dataset '{DATASET_NAME}' already exists (id={ds.id}). Skipping to avoid duplicates.")
        return

    ds = client.create_dataset(dataset_name=DATASET_NAME, description=DESCRIPTION)
    client.create_examples(
        dataset_id=ds.id,
        inputs=[{"question": r["question"]} for r in rows],
        outputs=[
            {"reference_answer": r["reference_answer"], "reference_contexts": r["reference_contexts"]}
            for r in rows
        ],
        metadata=[
            {"id": r["id"], "synthesizer": r["synthesizer"], "source": r["source"], "raw_id": r.get("raw_id")}
            for r in rows
        ],
    )
    print(f"Pushed {len(rows)} examples to LangSmith dataset '{DATASET_NAME}' "
          f"(id={ds.id}) on {os.environ.get('LANGSMITH_ENDPOINT')}")


if __name__ == "__main__":
    main()
