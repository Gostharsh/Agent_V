
import os

import pytest

from vision_agent.vision import describe_image
from test_vision import make_red_rectangle_png
from langchain_core.messages import HumanMessage


@pytest.mark.skipif(
    os.getenv("RUN_OLLAMA_TESTS") != "1",
    reason="Set RUN_OLLAMA_TESTS=1 to run live Ollama tests",
)
def test_describe_image_identifies_red_rectangle(tmp_path):
    import base64

    image_path = tmp_path / "red_rectangle.png"
    image_path.write_bytes(base64.b64decode(make_red_rectangle_png()))

    answer = describe_image(
        str(image_path),
        "What is the main color of the rectangle? Answer with just the color name.",
    )

    assert "red" in answer.lower(), f"Unexpected vision response: {answer!r}"






@pytest.mark.skipif(
    os.getenv("RUN_OLLAMA_TESTS") != "1",
    reason="Set RUN_OLLAMA_TESTS=1 to run live Ollama tests",
)
def test_agent_selects_and_executes_analyze_image(tmp_path):
    import base64
    from unittest.mock import patch
    from langchain_core.messages import ToolMessage
    from vision_agent.graph import build_graph
    from langgraph.checkpoint.memory import InMemorySaver

    image_path = tmp_path / "red_rectangle.png"
    image_path.write_bytes(base64.b64decode(make_red_rectangle_png()))

    graph = build_graph(checkpointer=InMemorySaver())

    with patch(
        "vision_agent.memory.retrieve_memories",
        return_value=[],
    ):
        result = graph.invoke(
            {
                "messages": [
                    HumanMessage(
                        content=(
                            "You must use the analyze_image tool on the local "
                            f"image at {image_path}. Ask it: "
                            "'What is the main color of the rectangle?'. "
                            "Then report the tool's result."
                        )
                    )
                ],
                "owner_id": "vision-integration-test",
            },
            config={
                "recursion_limit": 10,
                "configurable": {
                    "thread_id": "vision-integration-test",
                    "owner_id": "vision-integration-test",
                },
            },
        )

    tool_messages = [
        message
        for message in result["messages"]
        if isinstance(message, ToolMessage)
    ]

    # assert any(
    #     message.name == "analyze_image"
    #     for message in tool_messages
    # ), (
    #     "The agent did not execute analyze_image. Tool messages: "
    #     f"{[(message.name, message.content) for message in tool_messages]!r}"
    # )
    # print("\n--- Agent messages ---")
    # for message in result["messages"]:
    #     print(
    #         "type =", getattr(message, "type", None),
    #         "content =", getattr(message, "content", None),
    #         "tool_calls =", getattr(message, "tool_calls", None),
    #     )

    vision_tool_messages = [
        message
        for message in tool_messages
        if message.name == "analyze_image"
    ]

    assert len(vision_tool_messages) == 1, (
        f"Expected exactly one analyze_image call, got "
        f"{len(vision_tool_messages)}"
    )
    vision_result = vision_tool_messages[0].content

    assert "red" in vision_result.lower(), (
        f"Unexpected analyze_image result: {vision_result!r}"
    )
