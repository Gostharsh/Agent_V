
import pytest

from vision_agent.graph import _evaluate_expression
from datetime import datetime

from vision_agent.tools import get_current_time, calculator


def test_calculator_rejects_empty_input():
    assert calculator.invoke({"expression": ""}) == (
        "Calculator error: expression cannot be empty"
    )


def test_calculator_rejects_whitespace_input():
    assert calculator.invoke({"expression": "   "}) == (
        "Calculator error: expression cannot be empty"
    )


@pytest.mark.parametrize(
    "expression",
    [
        "1 / 0",
        "import os",
        "__import__('os').system('whoami')",
        "2 ** 100",
        "open('secret.txt')",
    ],
)
def test_calculator_rejects_unsupported_expressions(expression):
    with pytest.raises((ValueError, SyntaxError, ZeroDivisionError)):
        _evaluate_expression(expression)

def test_get_current_time_returns_timezone_aware_time():
    result = get_current_time.invoke({})

    prefix = "Local system time: "
    assert result.startswith(prefix)

    parsed = datetime.fromisoformat(result.removeprefix(prefix))
    assert parsed.tzinfo is not None
    assert parsed.utcoffset() is not None

