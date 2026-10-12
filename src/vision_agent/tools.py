
import ast
import operator
import math
from datetime import datetime
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from vision_agent.vision import describe_image
from vision_agent.memory import store_memory_candidate
from langgraph.prebuilt import ToolRuntime



# Explicitly permit only basic arithmetic operations.
_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _evaluate_expression(expression: str) -> int | float:
    """Evaluate a restricted arithmetic expression without eval()."""
    if len(expression) > 200:
        raise ValueError("Expression is too long")

    tree = ast.parse(expression, mode="eval")

    def visit(node):
        if isinstance(node, ast.Expression):
            return visit(node.body)

        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            value = node.value
            if not math.isfinite(value) or abs(value) > 1e12:
                raise ValueError("Number is outside the permitted range")
            return value

        if isinstance(node, ast.BinOp):
            operation = _ALLOWED_OPERATORS.get(type(node.op))
            if operation is None:
                raise ValueError("Unsupported operator")

            left = visit(node.left)
            right = visit(node.right)

            if abs(left) > 1e12 or abs(right) > 1e12:
                raise ValueError("Operand is too large")

            if isinstance(node.op, ast.Div) and right == 0:
                raise ValueError("Division by zero")

            result = operation(left, right)

            if (
                not isinstance(result, (int, float))
                or not math.isfinite(result)
                or abs(result) > 1e12
            ):
                raise ValueError("Result is outside the permitted range")

            return result

        if isinstance(node, ast.UnaryOp):
            operation = _ALLOWED_OPERATORS.get(type(node.op))
            if operation is None:
                raise ValueError("Unsupported unary operator")
            return operation(visit(node.operand))

        raise ValueError("Only basic arithmetic is supported")

    return visit(tree)

@tool
def calculator(expression: str) -> str:
    """Calculate a basic arithmetic expression such as 125 * 48."""
    if not isinstance(expression, str):
        return "Calculator error: expression must be a string"

    if not expression.strip():
        return "Calculator error: expression cannot be empty"

    try:
        return str(_evaluate_expression(expression))
    except (
        ValueError,
        SyntaxError,
        ZeroDivisionError,
        OverflowError,
        RecursionError,
    ) as exc:
        return f"Calculator error: {exc}"
    
@tool
def get_current_time() -> str:
    """Get the current local system time and UTC offset."""
    current_time = datetime.now().astimezone()
    return (
        "Local system time: "
        f"{current_time.isoformat(timespec='seconds')}"
    )



@tool
def remember_memory(
    text: str,
    category: Literal["preference", "project_fact"],
    owner_id: Annotated[str, InjectedState("owner_id")],
    messages: Annotated[list[AnyMessage], InjectedState("messages")],
) -> str:
    """Save information only when the user explicitly asks to remember it.
    Reject candidates if the user's original request contains sensitive-looking
    information, even if the model omits that context from the candidate text.
    """
    from vision_agent.memory_policy import filter_memory_candidate
    from vision_agent.memory import store_memory_candidate

    latest_user_text = next(
        (
            message.content
            for message in reversed(messages)
            if getattr(message, "type", None) == "human"
            and isinstance(message.content, str)
        ),
        "",
    )

    # Check the original request as well as the model-generated candidate.
    if filter_memory_candidate(latest_user_text, category) is None:
        return "Memory rejected by the memory policy and was not stored."

    try:
        stored = store_memory_candidate(
            text=text,
            category=category,
            owner_id=owner_id,
        )
    except ValueError as exc:
        return f"Memory was not stored: {exc}"

    if stored:
        return "Memory stored successfully."

    return "Memory rejected by the memory policy and was not stored."




@tool
def analyze_image(image_path: str, question: str) -> str:
    """Analyze a local image to answer a question about its visual contents.

    Use when the user asks about an image stored at a local file path.
    Supported formats: PNG, JPEG, and WebP.
    """
    try:
        return describe_image(image_path, question)
    except (ValueError, FileNotFoundError) as exc:
        return f"Image analysis error: {exc}"
    except Exception:
        return "Image analysis failed due to an unexpected error."


class ScreenObservationContext(TypedDict, total=False):
    screen_observation_controller: object




@tool
def analyze_current_screen(
    question: str,
    runtime: ToolRuntime[ScreenObservationContext, dict],
) -> str:
    """Answer a question about the current screen using an existing observation.

    Use only when screen observation has already been explicitly activated.
    This tool never starts or polls screen capture.
    """
    context = runtime.context
    controller = (
        context.get("screen_observation_controller")
        if isinstance(context, dict)
        else None
    )

    if controller is None:
        return "Screen analysis error: no screen observation controller is configured."

    try:
        from vision_agent.screen_vision import analyze_current_screen as analyze

        return analyze(controller, question)
    except RuntimeError as exc:
        return f"Screen analysis error: {exc}"
    except ValueError as exc:
        return f"Screen analysis error: {exc}"
    except Exception:
        return "Screen analysis failed due to an unexpected error."




# The single registry used for model tool binding and graph execution.
TOOLS = [
    calculator,
    get_current_time,
    remember_memory,
    analyze_image,
    analyze_current_screen,
]
