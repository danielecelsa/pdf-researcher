"""
B.1 — Curation of the RAGAS-generated golden set.

Reads eval/golden_raw.jsonl, drops the cases flagged below (with reasons), re-ids
the survivors sequentially, and writes eval/golden.jsonl (the versioned SSoT).

Curation is manual judgement encoded here so it is transparent and reproducible:
each dropped id carries a reason. Re-run after regenerating the raw set (ids change,
so revisit the DROP map when the raw set changes).

Usage: .venv/bin/python eval/curate_golden.py
"""
import json
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
RAW = EVAL_DIR / "golden_raw.jsonl"
OUT = EVAL_DIR / "golden.jsonl"

# id (in golden_raw.jsonl) -> reason for dropping
DROP = {
    "golden_raw_002": "Q/A mismatch: verbose persona question, answer is only the BERT code URL",
    "golden_raw_004": "broken grammar ('How BERT pre-train?'), redundant with 008",
    "golden_raw_011": "near-duplicate of 010 (BERT hyperparameters: pre-training vs fine-tuning)",
    "golden_raw_020": "broken compound grammar, redundant with 014 (BERT MLM masking)",
}


def main():
    with open(RAW, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    kept, dropped = [], []
    for r in rows:
        if r["id"] in DROP:
            dropped.append(r["id"])
        else:
            kept.append(r)

    with open(OUT, "w", encoding="utf-8") as f:
        for i, r in enumerate(kept):
            rec = {
                "id": f"golden_{i:03d}",
                "raw_id": r["id"],
                "question": r["question"],
                "reference_answer": r["reference_answer"],
                "reference_contexts": r["reference_contexts"],
                "synthesizer": r["synthesizer"],
                "source": r["source"],
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    from collections import Counter
    by_syn = Counter(r["synthesizer"] for r in kept)
    print(f"Raw: {len(rows)}  ->  kept: {len(kept)}  (dropped {len(dropped)})")
    print(f"Dropped: {dropped}")
    print("Kept by synthesizer:")
    for s, n in by_syn.items():
        print(f"  {s}: {n}")
    print(f"Wrote -> {OUT}")


if __name__ == "__main__":
    main()
