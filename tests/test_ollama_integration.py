import os

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from vision_agent.graph import agent


@pytest.mark.skipif(
    os.getenv("RUN_OLLAMA_TESTS") != "1",
    reason="Set RUN_OLLAMA_TESTS=1 to run live Ollama tests",
)
def test_real_qwen_model_responds():
    result = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content="Reply with exactly: Vision Agent is online."
                )
            ]
        },
        config={
            "configurable": {"thread_id": "ollama-respond-test"}
        },
    )

    assistant_messages = [
        message
        for message in result["messages"]
        if isinstance(message, AIMessage) and not message.tool_calls
    ]

    assert assistant_messages
    assert isinstance(assistant_messages[-1].content, str)
    assert assistant_messages[-1].content.strip()

@pytest.mark.skipif(
    os.getenv("RUN_OLLAMA_TESTS") != "1",
    reason="Set RUN_OLLAMA_TESTS=1 to run live Ollama tests",
)

def test_real_qwen_uses_calculator_tool():
    result = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content=(
                        "Use the calculator tool to calculate 137 * 239. "
                        "After the tool returns, tell me the numerical result."
                    )
                )
            ]
        },
        config={"recursion_limit": 10, 
                "configurable": {"thread_id": "ollama-calculator-test"},
                },
    )

    tool_messages = [
        message
        for message in result["messages"]
        if getattr(message, "type", None) == "tool"
        and getattr(message, "name", None) == "calculator"
    ]

    assistant_messages = [
        message
        for message in result["messages"]
        if isinstance(message, AIMessage) and not message.tool_calls
    ]

    assert tool_messages, "The model did not execute the calculator tool"
    assert tool_messages[0].content == "32743"
    assert assistant_messages, "The model did not produce a final answer"
    assert "32743" in str(assistant_messages[-1].content)
