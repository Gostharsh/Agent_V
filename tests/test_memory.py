
from unittest.mock import patch

import pytest

from vision_agent.memory import embed_text


def test_embed_text_returns_numeric_vector():
    with patch(
        "vision_agent.memory._embed_query",
        return_value=[0.1, 0.2, 0.3],
    ) as mock_embed:
        result = embed_text("Vision memory test")

    assert result == [0.1, 0.2, 0.3]
    mock_embed.assert_called_once_with("Vision memory test")


@pytest.mark.parametrize("text", ["", "   "])
def test_embed_text_rejects_empty_input(text):
    with pytest.raises(ValueError, match="non-empty string"):
        embed_text(text)


def test_store_memory_saves_text_embedding_and_owner():
    from vision_agent.memory import store_memory

    embedding = [0.1, 0.2, 0.3]

    with (
        patch("vision_agent.memory.embed_text", return_value=embedding),
        patch("vision_agent.memory._store_memory") as mock_store,
    ):
        store_memory("I prefer concise explanations.", owner_id="user-1")

    mock_store.assert_called_once_with(
        text="I prefer concise explanations.",
        embedding=embedding,
        owner_id="user-1",
    )





def test_retrieve_memories_returns_matching_texts():
    from vision_agent.memory import retrieve_memories

    embedding = [0.1, 0.2, 0.3]
    expected = [
        "I prefer concise explanations.",
        "I am building Vision Agent.",
    ]

    with (
        patch("vision_agent.memory.embed_text", return_value=embedding),
        patch(
            "vision_agent.memory._retrieve_memories",
            return_value=expected,
        ) as mock_retrieve,
    ):
        result = retrieve_memories(
            "What are my preferences?",
            owner_id="user-1",
        )

    assert result == expected
    mock_retrieve.assert_called_once_with(
        embedding=embedding,
        owner_id="user-1",
        limit=5,
    )


def test_retrieve_memories_passes_custom_limit():
    from vision_agent.memory import retrieve_memories

    with (
        patch("vision_agent.memory.embed_text", return_value=[0.1]),
        patch(
            "vision_agent.memory._retrieve_memories",
            return_value=["A memory"],
        ) as mock_retrieve,
    ):
        retrieve_memories(
            "Find a memory",
            owner_id="user-1",
            limit=3,
        )

    assert mock_retrieve.call_args.kwargs["limit"] == 3


@pytest.mark.parametrize(
    ("query", "owner_id", "limit"),
    [
        ("", "user-1", 5),
        ("   ", "user-1", 5),
        ("query", "", 5),
        ("query", "   ", 5),
        ("query", "user-1", 0),
        ("query", "user-1", -1),
        ("query", "user-1", 1.5),
        ("query", "user-1", True),
    ],
)
def test_retrieve_memories_rejects_invalid_arguments(
    query,
    owner_id,
    limit,
):
    from vision_agent.memory import retrieve_memories

    with pytest.raises(ValueError):
        retrieve_memories(query, owner_id=owner_id, limit=limit)



def test_retrieve_memories_scopes_database_query_to_owner():
    from vision_agent.memory import _retrieve_memories

    # with patch("vision_agent.memory.GraphDatabase.driver") as mock_driver:
    #     mock_session = mock_driver.return_value.__enter__.return_value
    #     mock_session.execute_query.return_value = (
    #         [{"text": "Owner-specific memory"}],
    #         None,
    #         None,
    #     )

    #     result = _retrieve_memories(
    #         embedding=[0.1, 0.2, 0.3],
    #         owner_id="user-1",
    #         limit=5,
    #     )
    
    with (
        patch.dict(
            "os.environ",
            {
                "NEO4J_URI": "bolt://localhost:7687",
                "NEO4J_USERNAME": "test-user",
                "NEO4J_PASSWORD": "test-password",
                "NEO4J_DATABASE": "neo4j",
            },
        ),
        patch("vision_agent.memory.GraphDatabase.driver") as mock_driver,
    ):
        mock_session = mock_driver.return_value.__enter__.return_value
        mock_session.execute_query.return_value = (
            [{"text": "Owner-specific memory"}],
            None,
            None,
        )

        result = _retrieve_memories(
            embedding=[0.1, 0.2, 0.3],
            owner_id="user-1",
            limit=5,
        )


    assert result == ["Owner-specific memory"]
    query = mock_session.execute_query.call_args.args[0]
    assert "owner_id" in query


    
def test_store_memory_candidate_stores_accepted_text():
    from unittest.mock import patch

    from vision_agent.memory import store_memory_candidate

    with (
        patch(
            "vision_agent.memory.embed_text",
            return_value=[0.1, 0.2, 0.3],
        ) as mock_embed,
        patch("vision_agent.memory._store_memory") as mock_store,
    ):
        result = store_memory_candidate(
            "  I prefer concise answers.  ",
            "preference",
            "owner-1",
        )

    assert result is True
    mock_embed.assert_called_once_with("I prefer concise answers.")
    mock_store.assert_called_once_with(
        text="I prefer concise answers.",
        embedding=[0.1, 0.2, 0.3],
        owner_id="owner-1",
    )


def test_store_memory_candidate_rejects_without_embedding():
    from unittest.mock import patch

    from vision_agent.memory import store_memory_candidate

    with (
        patch("vision_agent.memory.embed_text") as mock_embed,
        patch("vision_agent.memory._store_memory") as mock_store,
    ):
        result = store_memory_candidate(
            "password is: hunter2",
            "preference",
            "owner-1",
        )

    assert result is False
    mock_embed.assert_not_called()
    mock_store.assert_not_called()


def test_store_memory_candidate_rejects_invalid_category():
    from unittest.mock import patch

    from vision_agent.memory import store_memory_candidate

    with (
        patch("vision_agent.memory.embed_text") as mock_embed,
        patch("vision_agent.memory._store_memory") as mock_store,
    ):
        result = store_memory_candidate(
            "I prefer concise answers.",
            "unknown",
            "owner-1",
        )

    assert result is False
    mock_embed.assert_not_called()
    mock_store.assert_not_called()


def test_store_memory_candidate_rejects_invalid_owner():
    import pytest
    from unittest.mock import patch

    from vision_agent.memory import store_memory_candidate

    with (
        patch("vision_agent.memory.embed_text") as mock_embed,
        patch("vision_agent.memory._store_memory") as mock_store,
        pytest.raises(ValueError, match="Owner ID"),
    ):
        store_memory_candidate(
            "I prefer concise answers.",
            "preference",
            " ",
        )

    mock_embed.assert_not_called()
    mock_store.assert_not_called()
