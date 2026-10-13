from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from vision_agent.graph import build_graph, call_model, calculator
from vision_agent.tools import TOOLS
import pytest
from unittest.mock import patch

@pytest.fixture(autouse=True)
def mock_memory_retrieval():
    with patch(
        "vision_agent.memory.retrieve_memories",
        return_value=[],
    ):
        yield

def test_call_model_returns_assistant_message():
    """The model node should append the assistant's response."""
    expected = AIMessage(content="Hello from Vision Agent")

    with patch("vision_agent.graph.invoke_model", return_value=expected):
        result = call_model(
            {"messages": [HumanMessage(content="Hello")]}
        )

    assert result["messages"] == [expected]


# def test_graph_passes_state_to_model():
#     """The compiled graph should preserve conversation history."""
#     expected = AIMessage(content="Graph executed successfully")

#     with patch("vision_agent.graph.invoke_model", return_value=expected):
#         graph = build_graph()
#         result = graph.invoke(
#             {"messages": [HumanMessage(content="Test Vision Agent")]}
#         )

#     assert result["messages"][0].content == "Test Vision Agent"
#     assert result["messages"][-1].content == "Graph executed successfully"


def test_graph_passes_state_to_model():
    """The compiled graph should preserve conversation history."""
    expected = AIMessage(content="Graph executed successfully")

    with patch(
        "vision_agent.graph.invoke_model",
        return_value=expected,
    ):
        graph = build_graph()
        result = graph.invoke(
            {"messages": [HumanMessage(content="Test Vision Agent")]}
        )

    assert result["messages"][0].content == "Test Vision Agent"
    assert result["messages"][-1].content == "Graph executed successfully"

def test_call_model_propagates_model_errors():
    """Model errors should not be silently swallowed."""
    with patch(
        "vision_agent.graph.invoke_model",
        side_effect=RuntimeError("Ollama unavailable"),
    ):
        try:
            call_model({"messages": [HumanMessage(content="Hello")]})
        except RuntimeError as exc:
            assert str(exc) == "Ollama unavailable"
        else:
            raise AssertionError("Expected model error to propagate")


def test_graph_executes_calculator_tool():
    """The graph should execute a requested tool and return to the model."""
    tool_request = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "calculator",
                "args": {"expression": "125 * 48"},
                "id": "call_1",
                "type": "tool_call",
            }
        ],
    )
    final_answer = AIMessage(content="125 × 48 = 6000")

    with patch(
        "vision_agent.graph.invoke_model",
        side_effect=[tool_request, final_answer],
    ) as mock_invoke:
        graph = build_graph()
        result = graph.invoke(
            {"messages": [HumanMessage(content="Calculate 125 * 48")]}
        )

    tool_messages = [
        message
        for message in result["messages"]
        if isinstance(message, ToolMessage)
    ]

    assert len(tool_messages) == 1
    assert tool_messages[0].content == "6000"
    assert result["messages"][-1].content == "125 × 48 = 6000"
    assert mock_invoke.call_count == 2



def test_graph_executes_analyze_image_tool(tmp_path):
    """The graph should execute the vision tool and return to the model."""
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
    from unittest.mock import patch

    image_path = tmp_path / "test.png"
    image_path.write_bytes(b"test image placeholder")

    tool_request = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "analyze_image",
                "args": {
                    "image_path": str(image_path),
                    "question": "What color is the rectangle?",
                },
                "id": "analyze-image-call-1",
                "type": "tool_call",
            }
        ],
    )
    final_answer = AIMessage(content="The rectangle is red.")

    with patch(
        "vision_agent.graph.invoke_model",
        side_effect=[tool_request, final_answer],
    ) as mock_invoke, patch(
        "vision_agent.tools.describe_image",
        return_value="The rectangle is red.",
    ) as mock_describe:
        graph = build_graph()
        result = graph.invoke(
            {
                "messages": [
                    HumanMessage(
                        content=f"Analyze this image: {image_path}"
                    )
                ]
            }
        )

    tool_messages = [
        message
        for message in result["messages"]
        if isinstance(message, ToolMessage)
    ]

    assert len(tool_messages) == 1
    assert tool_messages[0].name == "analyze_image"
    assert tool_messages[0].content == "The rectangle is red."
    mock_describe.assert_called_once_with(
        str(image_path),
        "What color is the rectangle?",
    )
    assert result["messages"][-1].content == "The rectangle is red."
    assert mock_invoke.call_count == 2



def test_tool_registry_contains_expected_tools():
    names = {registered_tool.name for registered_tool in TOOLS}

    assert names == {
        "calculator",
        "get_current_time",
        "remember_memory",
        "analyze_image",
        "analyze_current_screen",
        "open_notepad_and_type_text",
    }




def test_graph_preserves_conversation_across_turns():
    """The same thread should retain messages across separate invocations."""
    from langgraph.checkpoint.memory import InMemorySaver

    first_response = AIMessage(content="I'll remember that your favorite color is blue.")
    second_response = AIMessage(content="Your favorite color is blue.")

    with patch(
        "vision_agent.graph.invoke_model",
        side_effect=[first_response, second_response],
    ):
        graph = build_graph(checkpointer=InMemorySaver())
        config = {
            "configurable": {"thread_id": "memory-test"},
        }

        graph.invoke(
            {"messages": [HumanMessage(content="My favorite color is blue.")]},
            config=config,
        )
        result = graph.invoke(
            {"messages": [HumanMessage(content="What is my favorite color?")]},
            config=config,
        )

    messages = result["messages"]
    human_messages = [
        message for message in messages
        if isinstance(message, HumanMessage)
    ]

    assert len(human_messages) == 2
    assert human_messages[0].content == "My favorite color is blue."
    assert human_messages[1].content == "What is my favorite color?"
    assert messages[-1].content == "Your favorite color is blue."



def test_graph_keeps_separate_threads_isolated():
    from langgraph.checkpoint.memory import InMemorySaver

    responses = [
        AIMessage(content="Blue."),
        AIMessage(content="I don't know your favorite color yet."),
    ]

    with patch(
        "vision_agent.graph.invoke_model",
        side_effect=responses,
    ):
        graph = build_graph(checkpointer=InMemorySaver())

        graph.invoke(
            {"messages": [HumanMessage(content="My favorite color is blue.")]},
            config={"configurable": {"thread_id": "thread-a"}},
        )

        result = graph.invoke(
            {"messages": [HumanMessage(content="What is my favorite color?")]},
            config={"configurable": {"thread_id": "thread-b"}},
        )

    human_messages = [
        message
        for message in result["messages"]
        if isinstance(message, HumanMessage)
    ]

    assert len(human_messages) == 1
    assert human_messages[0].content == "What is my favorite color?"




def test_run_agent_passes_owner_id_and_thread_id():
    from unittest.mock import patch
    from vision_agent.graph import run_agent

    with patch("vision_agent.graph.agent") as mock_agent:
        mock_agent.invoke.return_value = {
            "messages": [
                __import__(
                    "langchain_core.messages",
                    fromlist=["AIMessage"],
                ).AIMessage(content="Hello!")
            ]
        }

        result = run_agent(
            "Hello",
            thread_id="thread-1",
            owner_id="owner-1",
        )

    assert result == "Hello!"
    config = mock_agent.invoke.call_args.kwargs["config"]
    assert config["configurable"]["thread_id"] == "thread-1"
    assert config["configurable"]["owner_id"] == "owner-1"


def test_run_agent_passes_owner_id_in_state():
    from unittest.mock import patch
    from langchain_core.messages import AIMessage
    from vision_agent.graph import run_agent

    with patch("vision_agent.graph.agent") as mock_agent:
        mock_agent.invoke.return_value = {
            "messages": [AIMessage(content="Hello!")]
        }

        result = run_agent(
            "Hello",
            thread_id="thread-1",
            owner_id="owner-1",
        )

    assert result == "Hello!"

    input_state = mock_agent.invoke.call_args.args[0]
    assert input_state["owner_id"] == "owner-1"




def test_load_memory_retrieves_for_latest_user_message():
    from unittest.mock import patch
    from langchain_core.messages import AIMessage, HumanMessage
    from vision_agent.graph import load_memory

    state = {
        "messages": [
            HumanMessage(content="Earlier question"),
            AIMessage(content="Earlier answer"),
            HumanMessage(content="What are my preferences?"),
        ],
        "owner_id": "owner-1",
    }

    expected = ["I prefer concise explanations."]

    with patch(
        "vision_agent.memory.retrieve_memories",
        return_value=expected,
    ) as mock_retrieve:
        result = load_memory(state)

    assert result == {"memories": expected}
    mock_retrieve.assert_called_once_with(
        "What are my preferences?",
        owner_id="owner-1",
        limit=5,
    )


def test_load_memory_returns_empty_when_no_user_message():
    from vision_agent.graph import load_memory
    from langchain_core.messages import AIMessage

    result = load_memory(
        {
            "messages": [AIMessage(content="Hello")],
            "owner_id": "owner-1",
        }
    )

    assert result == {"memories": []}




def test_call_model_includes_retrieved_memories_in_context():
    from unittest.mock import patch
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
    from vision_agent.graph import call_model

    state = {
        "messages": [HumanMessage(content="How should you explain things to me?")],
        "owner_id": "owner-1",
        "memories": ["I prefer concise explanations."],
    }

    with patch(
        "vision_agent.graph.invoke_model",
        return_value=AIMessage(content="I'll keep it concise."),
    ) as mock_invoke:
        result = call_model(state)

    assert result["messages"][0].content == "I'll keep it concise."

    model_messages = mock_invoke.call_args.args[0]
    memory_messages = [
        message
        for message in model_messages
        if isinstance(message, SystemMessage)
        and "I prefer concise explanations." in str(message.content)
    ]

    assert len(memory_messages) == 1
    assert state["messages"] == [
        HumanMessage(content="How should you explain things to me?")
    ]



def test_graph_executes_remember_memory_tool():
    """An explicit memory tool call should execute and return its result."""
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
    from unittest.mock import patch

    tool_request = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "remember_memory",
                "args": {
                    "text": "I prefer concise answers.",
                    "category": "preference",
                },
                "id": "remember-call-1",
                "type": "tool_call",
            }
        ],
    )
    final_answer = AIMessage(content="I've saved your preference.")

    with patch(
        "vision_agent.graph.invoke_model",
        side_effect=[tool_request, final_answer],
    ) as mock_invoke, patch(
        "vision_agent.memory.store_memory_candidate",
        return_value=True,
    ) as mock_store:
        graph = build_graph()
        result = graph.invoke(
            {
                "messages": [
                    HumanMessage(
                        content="Remember that I prefer concise answers."
                    )
                ],
                "owner_id": "owner-1",
            }
        )

    tool_messages = [
        message for message in result["messages"]
        if isinstance(message, ToolMessage)
    ]

    assert len(tool_messages) == 1
    assert tool_messages[0].name == "remember_memory"
    assert "stored successfully" in tool_messages[0].content.lower()
    mock_store.assert_called_once_with(
        text="I prefer concise answers.",
        category="preference",
        owner_id="owner-1",
    )
    assert result["messages"][-1].content == "I've saved your preference."
    assert mock_invoke.call_count == 2


def test_graph_does_not_save_without_memory_tool_call():
    """An ordinary statement must not save memory unless the model calls the tool."""
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
    from unittest.mock import patch

    with patch(
        "vision_agent.graph.invoke_model",
        return_value=AIMessage(
            content="Thanks for letting me know."
        ),
    ), patch(
        "vision_agent.memory.store_memory_candidate"
    ) as mock_store:
        graph = build_graph()
        result = graph.invoke(
            {
                "messages": [
                    HumanMessage(content="I prefer concise answers.")
                ],
                "owner_id": "owner-1",
            }
        )

    assert not any(
        isinstance(message, ToolMessage)
        for message in result["messages"]
    )
    mock_store.assert_not_called()
    assert result["messages"][-1].content == "Thanks for letting me know."


def test_graph_exposes_memory_rejection_to_model():
    """The model should receive the actual rejection result before replying."""
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
    from unittest.mock import patch

    tool_request = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "remember_memory",
                "args": {
                    "text": "test-secret-123",
                    "category": "preference",
                },
                "id": "remember-call-2",
                "type": "tool_call",
            }
        ],
    )
    final_answer = AIMessage(
        content="That information was not saved."
    )

    with patch(
        "vision_agent.graph.invoke_model",
        side_effect=[tool_request, final_answer],
    ) as mock_invoke, patch(
        "vision_agent.memory.store_memory_candidate",
        side_effect=AssertionError(
            "Sensitive request must be rejected before storage."
        ),
    ):
        graph = build_graph()
        result = graph.invoke(
            {
                "messages": [
                    HumanMessage(
                        content=(
                            "Remember that my password is: test-secret-123."
                        )
                    )
                ],
                "owner_id": "owner-1",
            }
        )

    tool_messages = [
        message for message in result["messages"]
        if isinstance(message, ToolMessage)
    ]

    assert len(tool_messages) == 1
    assert "rejected" in tool_messages[0].content.lower()
    assert "not stored" in tool_messages[0].content.lower()

    # The second model call must receive the tool result in its history.
    second_model_messages = mock_invoke.call_args_list[1].args[0]
    assert any(
        isinstance(message, ToolMessage)
        and "rejected" in message.content.lower()
        for message in second_model_messages
    )
    assert result["messages"][-1].content == "That information was not saved."




def test_call_model_includes_tool_use_instructions():
    """The model receives clear instructions for tool use and memory saving."""
    from unittest.mock import patch
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
    from vision_agent.graph import call_model

    with patch(
        "vision_agent.graph.invoke_model",
        return_value=AIMessage(content="Understood."),
    ) as mock_invoke:
        call_model({"messages": [HumanMessage(content="Hello")]})

    model_messages = mock_invoke.call_args.args[0]
    system_messages = [
        message for message in model_messages
        if isinstance(message, SystemMessage)
    ]

    assert len(system_messages) >= 1

    system_content = str(system_messages[0].content)
    assert "calculator" in system_content
    assert "get_current_time" in system_content
    assert "remember_memory" in system_content
    assert "explicitly asks" in system_content
    assert "tell the user it was not saved" in system_content



def test_calculator_handles_deeply_nested_input():
    """Malformed, deeply nested input should return an error, not crash."""
    from vision_agent.tools import calculator

    expression = "(" * 1500 + "1" + ")" * 1500
    result = calculator.invoke({"expression": expression})

    assert result.startswith("Calculator error:")






@pytest.mark.parametrize(
    "expression",
    [
        "",
        "   ",
        "2 +",
        "2 ** 100",
        "1 / 0",
        "10 ** 20",
        "1e100",
        "-1e100"
    ],
)
def test_calculator_rejects_invalid_or_unsupported_expressions(expression):
    """Invalid, unsupported, or out-of-range input returns a controlled error."""
    result = calculator.invoke({"expression": expression})

    assert result.startswith("Calculator error:")


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("2 + 3", "5"),
        ("125 * 48", "6000"),
        ("-5 + 2", "-3"),
        ("8 / 2", "4.0"),
        ("-(3 + 2)", "-5"),
    ],
)
def test_calculator_returns_expected_results(expression, expected):
    """Supported basic arithmetic produces the expected result."""
    result = calculator.invoke({"expression": expression})

    assert result == expected


def test_calculator_rejects_expression_longer_than_limit():
    """Expressions exceeding the input length limit are rejected."""
    expression = "1" * 201

    result = calculator.invoke({"expression": expression})

    assert result.startswith("Calculator error:")
    assert "too long" in result.lower()




def test_build_graph_declares_screen_context():
    from vision_agent.graph import AgentContext, build_graph

    graph = build_graph()

    assert "screen_observation_controller" in AgentContext.__annotations__
    assert graph is not None


def test_run_agent_passes_screen_controller_in_context():
    from unittest.mock import patch
    from langchain_core.messages import AIMessage
    from vision_agent.graph import run_agent

    controller = object()

    with patch("vision_agent.graph.agent") as mock_agent:
        mock_agent.invoke.return_value = {
            "messages": [AIMessage(content="Screen analysis complete.")]
        }

        result = run_agent(
            "Describe my screen",
            screen_observation_controller=controller,
        )

    assert result == "Screen analysis complete."
    kwargs = mock_agent.invoke.call_args.kwargs
    assert kwargs["context"]["screen_observation_controller"] is controller


def test_run_agent_uses_empty_context_without_controller():
    from unittest.mock import patch
    from langchain_core.messages import AIMessage
    from vision_agent.graph import run_agent

    with patch("vision_agent.graph.agent") as mock_agent:
        mock_agent.invoke.return_value = {
            "messages": [AIMessage(content="Hello.")]
        }

        result = run_agent("Hello")

    assert result == "Hello."
    assert mock_agent.invoke.call_args.kwargs["context"] == {}
