
from unittest.mock import patch

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.prebuilt import ToolNode

from vision_agent.tools import remember_memory

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage


def invoke_remember_memory(
    text,
    category,
    owner_id="owner-1",
    original_request="Remember that I prefer concise answers.",
):
    from langgraph.graph import START, StateGraph
    from typing import Annotated, TypedDict
    from langchain_core.messages import AnyMessage
    from langgraph.graph.message import add_messages

    class TestState(TypedDict):
        messages: Annotated[list[AnyMessage], add_messages]
        owner_id: str

    tool_node = ToolNode([remember_memory])
    graph = StateGraph(TestState)
    graph.add_node("tools", tool_node)
    graph.add_edge(START, "tools")
    test_graph = graph.compile()

    tool_call = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "remember_memory",
                "args": {"text": text, "category": category},
                "id": "call-1",
                "type": "tool_call",
            }
        ],
    )

    return test_graph.invoke(
        {
            "messages": [
                HumanMessage(content=original_request),
                tool_call,
            ],
            "owner_id": owner_id,
        }
    )


def test_remember_memory_uses_owner_from_graph_state():
    with patch(
        "vision_agent.memory.store_memory_candidate",
        return_value=True,
    ) as mock_store:
        result = invoke_remember_memory(
            "I prefer concise answers.",
            "preference",
            owner_id="owner-1",
        )

    tool_result = result["messages"][-1]

    assert isinstance(tool_result, ToolMessage)
    assert "stored successfully" in tool_result.content.lower()
    mock_store.assert_called_once_with(
        text="I prefer concise answers.",
        category="preference",
        owner_id="owner-1",
    )



def test_remember_memory_reports_policy_rejection():
    with patch(
        "vision_agent.memory.store_memory_candidate",
        return_value=False,
    ) as mock_store:
        result = invoke_remember_memory(
            "password is: test-secret",
            "preference",
            original_request=(
                "Remember that my password is: test-secret."
            ),
        )

    tool_result = result["messages"][-1]

    assert "rejected" in tool_result.content.lower()
    assert "not stored" in tool_result.content.lower()
    mock_store.assert_not_called()



def test_remember_memory_schema_hides_owner_id():
    schema = remember_memory.tool_call_schema.model_json_schema()

    assert "text" in schema["properties"]
    assert "category" in schema["properties"]
    assert "owner_id" not in schema["properties"]



def test_remember_memory_rejects_sensitive_original_request():
    with patch(
        "vision_agent.memory.store_memory_candidate",
    ) as mock_store:
        result = invoke_remember_memory(
            "test-secret-123",
            "preference",
            owner_id="owner-1",
            original_request=(
                "Remember that my password is: test-secret-123."
            ),
        )

    tool_result = result["messages"][-1]

    assert isinstance(tool_result, ToolMessage)
    assert "rejected" in tool_result.content.lower()
    assert "not stored" in tool_result.content.lower()
    mock_store.assert_not_called()