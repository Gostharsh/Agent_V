import os
import uuid

import pytest
from neo4j import GraphDatabase

from vision_agent.memory import store_memory


@pytest.mark.skipif(
    os.getenv("RUN_NEO4J_TESTS") != "1",
    reason="Set RUN_NEO4J_TESTS=1 to run Neo4j integration tests",
)
def test_store_memory_persists_to_neo4j():
    owner_id = f"test-{uuid.uuid4()}"
    text = "Vision integration memory test"

    query = """
    MATCH (m:Memory {owner_id: $owner_id, text: $text})
    RETURN m.embedding AS embedding,
           m.created_at IS NOT NULL AS has_created_at
    """

    with GraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(
            os.environ["NEO4J_USERNAME"],
            os.environ["NEO4J_PASSWORD"],
        ),
    ) as driver:
        try:
            store_memory(text, owner_id)

            records, _, _ = driver.execute_query(
                query,
                owner_id=owner_id,
                text=text,
                database_=os.environ["NEO4J_DATABASE"],
            )

            assert len(records) == 1
            assert len(records[0]["embedding"]) == 768
            assert records[0]["has_created_at"] is True
        finally:
            driver.execute_query(
                """
                MATCH (m:Memory {owner_id: $owner_id})
                DELETE m
                """,
                owner_id=owner_id,
                database_=os.environ["NEO4J_DATABASE"],
            )


@pytest.mark.skipif(
    os.getenv("RUN_NEO4J_TESTS") != "1",
    reason="Set RUN_NEO4J_TESTS=1 to run Neo4j integration tests",
)
def test_retrieve_memories_returns_only_matching_owner():
    from vision_agent.memory import retrieve_memories

    owner_id = f"retrieval-test-{uuid.uuid4()}"
    other_owner_id = f"other-owner-{uuid.uuid4()}"

    own_text = "I prefer concise explanations in technical discussions."
    other_text = "I prefer very detailed explanations in technical discussions."

    with GraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(
            os.environ["NEO4J_USERNAME"],
            os.environ["NEO4J_PASSWORD"],
        ),
    ) as driver:
        try:
            store_memory(own_text, owner_id)
            store_memory(other_text, other_owner_id)

            results = retrieve_memories(
                "How should technical explanations be written?",
                owner_id=owner_id,
                limit=5,
            )

            assert own_text in results
            assert other_text not in results
        finally:
            driver.execute_query(
                """
                MATCH (m:Memory)
                WHERE m.owner_id IN $owner_ids
                DELETE m
                """,
                owner_ids=[owner_id, other_owner_id],
                database_=os.environ["NEO4J_DATABASE"],
            )



@pytest.mark.skipif(
    os.getenv("RUN_NEO4J_TESTS") != "1",
    reason="Set RUN_NEO4J_TESTS=1 to run Neo4j integration tests",
)
def test_graph_uses_memory_across_new_thread():
    from unittest.mock import patch
    from langchain_core.messages import AIMessage, SystemMessage
    from vision_agent.graph import agent

    owner_id = f"graph-memory-{uuid.uuid4()}"
    thread_id = f"new-thread-{uuid.uuid4()}"
    memory_text = (
        f"Test preference {uuid.uuid4()}: "
        "I prefer concise answers with a short summary."
    )

    with GraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(
            os.environ["NEO4J_USERNAME"],
            os.environ["NEO4J_PASSWORD"],
        ),
    ) as driver:
        try:
            store_memory(memory_text, owner_id)

            captured = {}

            def fake_invoke(messages):
                captured["messages"] = messages
                return AIMessage(content="Test response")

            with patch("vision_agent.graph.invoke_model", side_effect=fake_invoke):
                result = agent.invoke(
                    {
                        "messages": [
                            __import__(
                                "langchain_core.messages",
                                fromlist=["HumanMessage"],
                            ).HumanMessage(
                                content="How should you answer me?"
                            )
                        ],
                        "owner_id": owner_id,
                    },
                    config={
                        "recursion_limit": 10,
                        "configurable": {"thread_id": thread_id},
                    },
                )

            assert result["messages"][-1].content == "Test response"
            assert any(
                isinstance(message, SystemMessage)
                and memory_text in str(message.content)
                for message in captured["messages"]
            )
        finally:
            driver.execute_query(
                """
                MATCH (m:Memory {owner_id: $owner_id})
                DELETE m
                """,
                owner_id=owner_id,
                database_=os.environ["NEO4J_DATABASE"],
            )




@pytest.mark.skipif(
    os.getenv("RUN_NEO4J_TESTS") != "1",
    reason="Set RUN_NEO4J_TESTS=1 to run Neo4j integration tests",
)
def test_store_memory_candidate_persists_accepted_candidate():
    from vision_agent.memory import retrieve_memories, store_memory_candidate

    owner_id = f"policy-accepted-{uuid.uuid4()}"
    text = f"Policy integration test {uuid.uuid4()}: concise answers preferred."

    with GraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]),
    ) as driver:
        try:
            result = store_memory_candidate(
                f"  {text}  ",
                "preference",
                owner_id,
            )

            assert result is True
            memories = retrieve_memories(
                "concise answers",
                owner_id=owner_id,
                limit=5,
            )
            assert text in memories
        finally:
            driver.execute_query(
                "MATCH (m:Memory {owner_id: $owner_id}) DELETE m",
                owner_id=owner_id,
                database_=os.environ["NEO4J_DATABASE"],
            )


@pytest.mark.skipif(
    os.getenv("RUN_NEO4J_TESTS") != "1",
    reason="Set RUN_NEO4J_TESTS=1 to run Neo4j integration tests",
)
def test_store_memory_candidate_does_not_persist_rejected_candidate():
    from unittest.mock import patch

    from vision_agent.memory import store_memory_candidate

    owner_id = f"policy-rejected-{uuid.uuid4()}"

    with GraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]),
    ) as driver:
        try:
            with patch("vision_agent.memory.embed_text") as mock_embed:
                result = store_memory_candidate(
                    "password is: integration-test-secret",
                    "preference",
                    owner_id,
                )

            assert result is False
            mock_embed.assert_not_called()

            records, _, _ = driver.execute_query(
                "MATCH (m:Memory {owner_id: $owner_id}) RETURN count(m) AS count",
                owner_id=owner_id,
                database_=os.environ["NEO4J_DATABASE"],
            )
            assert records[0]["count"] == 0
        finally:
            driver.execute_query(
                "MATCH (m:Memory {owner_id: $owner_id}) DELETE m",
                owner_id=owner_id,
                database_=os.environ["NEO4J_DATABASE"],
            )
