"""Unit tests for the cost helper (pure math, no API)."""
import pytest

from helpers import compute_cost

# Real Gemini 2.5-flash prices per 1K tokens.
COST_IN, COST_OUT = 0.0003, 0.0025


def test_compute_cost_zero():
    assert compute_cost(0, 0, COST_IN, COST_OUT) == 0.0


def test_compute_cost_known_values():
    # 1000 in @0.0003/1k + 1000 out @0.0025/1k
    assert compute_cost(1000, 1000, COST_IN, COST_OUT) == pytest.approx(0.0028)


def test_output_token_costs_more_than_input_token():
    input_only = compute_cost(1000, 0, COST_IN, COST_OUT)
    output_only = compute_cost(0, 1000, COST_IN, COST_OUT)
    assert output_only > input_only


def test_compute_cost_scales_linearly():
    assert compute_cost(2000, 0, COST_IN, COST_OUT) == pytest.approx(2 * compute_cost(1000, 0, COST_IN, COST_OUT))
