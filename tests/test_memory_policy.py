
import pytest

from vision_agent.memory_policy import should_store_memory, filter_memory_candidate


@pytest.mark.parametrize(
    ("text", "category"),
    [
        ("I prefer concise answers.", "preference"),
        ("Vision Agent uses LangGraph.", "project_fact"),
        ("I prefer Python examples.", "preference"),
    ],
)
def test_accepts_valid_memory_candidates(text, category):
    assert should_store_memory(text, category) is True


@pytest.mark.parametrize(
    ("text", "category"),
    [
        ("", "preference"),
        ("   ", "project_fact"),
        ("What is 17 times 24?", "unknown"),
        ("I prefer concise answers.", "unknown"),
    ],
)
def test_rejects_empty_or_unsupported_candidates(text, category):
    assert should_store_memory(text, category) is False


@pytest.mark.parametrize(
    "text",
    [
        "My password is: hunter2",
        "password=abc123",
        "api_key=sk-example-secret-value",
        "token: ghp_example123456789",
    ],
)
def test_rejects_sensitive_looking_candidates(text):
    assert should_store_memory(text, "preference") is False
    assert should_store_memory(text, "project_fact") is False


def test_rejects_non_string_text():
    assert should_store_memory(None, "preference") is False


def test_rejects_non_string_category():
    assert should_store_memory("I prefer concise answers.", None) is False



from vision_agent.memory_policy import filter_memory_candidate


def test_filter_returns_trimmed_text_for_accepted_candidate():
    result = filter_memory_candidate(
        "  I prefer concise answers.  ",
        "preference",
    )
    assert result == "I prefer concise answers."


def test_filter_returns_none_for_rejected_candidate():
    result = filter_memory_candidate(
        "password is: hunter2",
        "preference",
    )
    assert result is None


def test_filter_returns_none_for_unsupported_category():
    result = filter_memory_candidate(
        "I prefer concise answers.",
        "unknown",
    )
    assert result is None
