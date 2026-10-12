
import os

from langchain_ollama import OllamaEmbeddings
from neo4j import GraphDatabase
from vision_agent.memory_policy import filter_memory_candidate

EMBEDDING_MODEL = "nomic-embed-text:latest"

_embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)


def _embed_query(text: str) -> list[float]:
    """Call the configured embedding model."""
    return _embeddings.embed_query(text)


def embed_text(text: str) -> list[float]:
    """Convert text into an embedding vector using Ollama."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Text must be a non-empty string")

    return _embed_query(text)


def _store_memory(
    *,
    text: str,
    embedding: list[float],
    owner_id: str,
) -> None:
    """Persist a memory and its embedding in Neo4j."""
    uri = os.environ["NEO4J_URI"]
    username = os.environ["NEO4J_USERNAME"]
    password = os.environ["NEO4J_PASSWORD"]
    database = os.environ["NEO4J_DATABASE"]

    query = """
    CREATE (m:Memory {
        text: $text,
        embedding: $embedding,
        owner_id: $owner_id,
        created_at: datetime()
    })
    RETURN elementId(m) AS id
    """

    with GraphDatabase.driver(
        uri,
        auth=(username, password),
    ) as driver:
        driver.execute_query(
            query,
            text=text,
            embedding=embedding,
            owner_id=owner_id,
            database_=database,
        )


def store_memory(text: str, owner_id: str) -> None:
    """Embed and persist a memory for a specific owner."""
    if not isinstance(owner_id, str) or not owner_id.strip():
        raise ValueError("Owner ID must be a non-empty string")

    embedding = embed_text(text)

    _store_memory(
        text=text,
        embedding=embedding,
        owner_id=owner_id,
    )



def store_memory_candidate(
    text: str,
    category: str,
    owner_id: str,
) -> bool:
    """Store a candidate only if it passes the memory policy."""
    if not isinstance(owner_id, str) or not owner_id.strip():
        raise ValueError("Owner ID must be a non-empty string")

    accepted_text = filter_memory_candidate(text, category)
    if accepted_text is None:
        return False

    embedding = embed_text(accepted_text)
    _store_memory(
        text=accepted_text,
        embedding=embedding,
        owner_id=owner_id,
    )
    return True



def _retrieve_memories(
    *,
    embedding: list[float],
    owner_id: str,
    limit: int,
) -> list[str]:
    """Find similar memories belonging only to the specified owner."""
    uri = os.environ["NEO4J_URI"]
    username = os.environ["NEO4J_USERNAME"]
    password = os.environ["NEO4J_PASSWORD"]
    database = os.environ["NEO4J_DATABASE"]

    query = """
    MATCH (m:Memory)
    WHERE m.owner_id = $owner_id
      AND m.embedding IS NOT NULL
    WITH m,
         vector.similarity.cosine(m.embedding, $embedding) AS score
    RETURN m.text AS text, score
    ORDER BY score DESC
    LIMIT $limit
    """

    with GraphDatabase.driver(
        uri,
        auth=(username, password),
    ) as driver:
        records, _, _ = driver.execute_query(
            query,
            embedding=embedding,
            owner_id=owner_id,
            candidate_limit=limit * 10,
            limit=limit,
            database_=database,
        )

    return [record["text"] for record in records]

def retrieve_memories(
    query: str,
    owner_id: str,
    limit: int = 5,
) -> list[str]:
    """Retrieve relevant memories belonging to one owner."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query must be a non-empty string")

    if not isinstance(owner_id, str) or not owner_id.strip():
        raise ValueError("Owner ID must be a non-empty string")

    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("Limit must be a positive integer")

    embedding = embed_text(query)

    return _retrieve_memories(
        embedding=embedding,
        owner_id=owner_id,
        limit=limit,
    )
