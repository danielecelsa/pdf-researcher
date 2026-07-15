"""Unit tests for the deterministic retrieval metrics (no LLM, no API)."""
from retrieval_metrics import covers, normalize, precision_at_k, recall_at_k

REF_A = ("recurrent neural networks have long been firmly established as state of the art "
         "approaches in sequence modeling and transduction problems")
REF_B = ("we propose a new simple network architecture based solely on attention mechanisms "
         "dispensing with recurrence and convolutions entirely")
UNRELATED = "the quick brown fox jumps over the lazy dog near the river bank at dawn while birds sing"


def test_normalize_lowercases_and_collapses_whitespace():
    assert normalize("  Hello\n  WORLD  ") == "hello world"
    assert normalize(None) == ""


def test_covers_true_when_chunk_is_substring_of_passage():
    chunk = "firmly established as state of the art approaches in sequence modeling"
    assert covers(chunk, REF_A) is True


def test_covers_false_when_unrelated():
    assert covers(UNRELATED, REF_A) is False


def test_recall_full_partial_zero():
    refs = [REF_A, REF_B]
    assert recall_at_k([REF_A, REF_B], refs) == 1.0          # both gold passages covered
    assert recall_at_k([REF_A, UNRELATED], refs) == 0.5      # only REF_A covered
    assert recall_at_k([UNRELATED], refs) == 0.0             # none covered


def test_recall_none_when_no_reference_contexts():
    assert recall_at_k(["anything"], []) is None


def test_precision_hits_over_k():
    refs = [REF_A]
    assert precision_at_k([REF_A, UNRELATED], refs) == 0.5   # 1 of 2 retrieved on-target


def test_precision_none_when_nothing_retrieved():
    assert precision_at_k([], [REF_A]) is None
