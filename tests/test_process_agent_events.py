"""Unit tests for the LangGraph event-stream parser (no API)."""
from langchain_core.messages import AIMessage, ToolMessage

from helpers import process_agent_events


def test_parses_tool_call_output_final_answer_and_sums_agent_usage():
    ai_toolcall = AIMessage(
        content="",
        tool_calls=[{"name": "research", "args": {"query": "what is X"}, "id": "call_1", "type": "tool_call"}],
        usage_metadata={"input_tokens": 100, "output_tokens": 20, "total_tokens": 120},
    )
    tool_msg = ToolMessage(content="X is a thing.", name="research", tool_call_id="call_1")
    ai_final = AIMessage(
        content="X is a thing, in short.",
        usage_metadata={"input_tokens": 50, "output_tokens": 30, "total_tokens": 80},
    )
    events = [
        {"agent": {"messages": [ai_toolcall]}},
        {"tools": {"messages": [tool_msg]}},
        {"agent": {"messages": [ai_final]}},
    ]

    final_answer, trace, usage = process_agent_events(events)

    assert final_answer is ai_final
    # usage sums ONLY the agent AIMessages (tool message usage is not double-counted)
    assert usage == {"input_tokens": 150, "output_tokens": 50, "total_tokens": 200}
    assert {"type": "tool_call", "tool": "research", "tool_input": {"query": "what is X"}} in trace
    assert any(t["type"] == "tool_output" and t["tool"] == "research" for t in trace)


def test_no_tool_call_means_empty_trace_and_direct_answer():
    ai = AIMessage(content="Hello!", usage_metadata={"input_tokens": 5, "output_tokens": 3, "total_tokens": 8})
    events = [{"agent": {"messages": [ai]}}]

    final_answer, trace, usage = process_agent_events(events)

    assert final_answer is ai
    assert trace == []
    assert usage["total_tokens"] == 8
