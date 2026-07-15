# RAG Evaluation (`eval/`)

Offline evaluation harness for the Agentic RAG pipeline: a versioned **golden set**
plus scripts to (re)generate it, curate it, and push it to LangSmith as a Dataset for
tracked, comparable evaluation runs.

## Contents
- **`golden.jsonl`** — the curated golden set (source of truth): 17 cases, each
  `question -> reference_answer + reference_contexts`. Generated with RAGAS over a
  3-document corpus and hand-curated. Mix: single-hop + multi-hop (abstract & specific).
- `generate_golden.py` — synthesize a raw golden set with RAGAS `TestsetGenerator`
  (`gemini-2.5-flash`) over the corpus → `golden_raw.jsonl`.
- `curate_golden.py` — drop low-quality / duplicate cases (each drop documented) →
  `golden.jsonl`.
- `push_to_langsmith.py` — push `golden.jsonl` to a LangSmith Dataset
  (`pdf-researcher-golden`), idempotent.
- `fetch_corpus.py` — (re)build `corpus/` (the arXiv PDFs are not committed).

## Corpus
`corpus/` holds the documents the golden set is derived from:
- `llm_introduction.pdf` — *A Beginner's Guide to Large Language Models* (also the app's sample doc)
- `attention_is_all_you_need.pdf` — Vaswani et al., 2017 ([arXiv:1706.03762](https://arxiv.org/abs/1706.03762))
- `bert.pdf` — Devlin et al., 2018 ([arXiv:1810.04805](https://arxiv.org/abs/1810.04805))

The two arXiv PDFs are fetched at setup time, not committed. Run `fetch_corpus.py` first.

## Dependencies
Eval dependencies are kept separate from the app (they are **not** in the Docker image):

```bash
uv pip install -r requirements-eval.txt
```

## Workflow
```bash
uv pip install -r requirements-eval.txt
python eval/fetch_corpus.py          # rebuild corpus/
python eval/generate_golden.py 20    # -> golden_raw.jsonl   (RAGAS; consumes API credits)
python eval/curate_golden.py         # -> golden.jsonl
python eval/push_to_langsmith.py     # -> LangSmith Dataset 'pdf-researcher-golden'
```

The evaluation **metrics** — retrieval `recall@k` / `precision@k` (deterministic, chunk vs
`reference_contexts` overlap) and generation `faithfulness` / `answer_relevance` /
`answer_correctness` (RAGAS LLM-judge) — are produced by the evaluation **runner**, which
drives the RAG over this golden set and records each run as a LangSmith Experiment.
