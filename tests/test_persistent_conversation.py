
from langchain_core.messages import AIMessage, HumanMessage

from vision_agent.graph import build_graph
from vision_agent.persistence import PersistentCheckpointer


def test_conversation_survives_graph_reconstruction(tmp_path):
    from unittest.mock import patch

    database_path = tmp_path / "conversation.sqlite3"
    thread_id = "restart-persistence-test"

    with PersistentCheckpointer(database_path) as first_store:
        first_graph = build_graph(checkpointer=first_store.saver)

        with patch(
            "vision_agent.graph.invoke_model",
            return_value=AIMessage(content="Your favorite color is blue."),
        ):
            first_graph.invoke(
                {
                    "messages": [
                        HumanMessage(content="My favorite color is blue.")
                    ]
                },
                config={"configurable": {"thread_id": thread_id}},
            )

    with PersistentCheckpointer(database_path) as second_store:
        second_graph = build_graph(checkpointer=second_store.saver)

        with patch(
            "vision_agent.graph.invoke_model",
            return_value=AIMessage(content="It is blue."),
        ):
            result = second_graph.invoke(
                {
                    "messages": [
                        HumanMessage(content="What is my favorite color?")
                    ]
                },
                config={"configurable": {"thread_id": thread_id}},
            )

    human_messages = [
        message
        for message in result["messages"]
        if isinstance(message, HumanMessage)
    ]

    assert len(human_messages) == 2
    assert human_messages[0].content == "My favorite color is blue."
    assert human_messages[1].content == "What is my favorite color?"
