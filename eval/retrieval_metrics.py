"""
Deterministic retrieval metrics for the RAG eval (workstream B / CI eval-gate).

No LLM, no heavy imports: the overlap between a retrieved chunk and a golden reference_context
is pure text matching (rapidfuzz.partial_ratio). Kept side-effect-free and dependency-light so
it is unit-testable in isolation — the eval runner imports these; the tests import them directly.

A retrieved chunk "covers" a reference_context when the (shorter) chunk is ~contained in the
(longer, page-sized) passage: partial_ratio >= threshold on normalized text. This is why the
golden set stores reference_contexts as text passages (not chunk-ids) — the metric survives a
chunking A/B (the chunk boundaries change, the passages don't).
"""
from rapidfuzz import fuzz

DEFAULT_THRESHOLD = 90


def normalize(text: str) -> str:
    """Lowercase + collapse whitespace (robust to PDF-extraction artifacts)."""
    return " ".join((text or "").lower().split())


def covers(chunk: str, ref_context: str, threshold: int = DEFAULT_THRESHOLD) -> bool:
    """True if `chunk` is ~contained in `ref_context` (partial_ratio >= threshold)."""
    return fuzz.partial_ratio(normalize(chunk), normalize(ref_context)) >= threshold


def recall_at_k(retrieved, reference_contexts, threshold: int = DEFAULT_THRESHOLD):
    """Fraction of reference_contexts covered by >=1 retrieved chunk.

    Denominator = number of reference_contexts (the gold passages), NOT the number of retrieved
    chunks. Returns None when there are no reference_contexts (undefined, excluded from means).
    """
    if not reference_contexts:
        return None
    covered = sum(1 for ref in reference_contexts if any(covers(c, ref, threshold) for c in retrieved))
    return covered / len(reference_contexts)


def precision_at_k(retrieved, reference_contexts, threshold: int = DEFAULT_THRESHOLD):
    """Fraction of retrieved chunks that hit any reference_context.

    Denominator = number of retrieved chunks (k). Returns None when nothing was retrieved.
    """
    if not retrieved:
        return None
    hits = sum(1 for c in retrieved if any(covers(c, ref, threshold) for ref in reference_contexts))
    return hits / len(retrieved)
