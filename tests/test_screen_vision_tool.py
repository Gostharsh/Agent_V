from datetime import datetime, timezone
from typing import TypedDict

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import ToolNode

from vision_agent.screen_capture import ScreenFrame
from vision_agent.screen_observation import ScreenObservationController
from vision_agent.tools import analyze_current_screen


class ScreenToolTestState(TypedDict):
    messages: list

class ScreenToolContext(TypedDict, total=False):
    screen_observation_controller: ScreenObservationController


class FakeCaptureService:
    def __init__(self):
        self.active = False
        self.starts = 0
        self.captures = 0

    def start(self):
        self.starts += 1
        self.active = True

    def stop(self):
        self.active = False

    def capture_once(self):
        if not self.active:
            raise RuntimeError("Screen capture is not active")
        self.captures += 1
        return ScreenFrame(
            data=b"\x89PNG\r\n\x1a\nfake-frame",
            width=16,
            height=12,
            captured_at=datetime.now(timezone.utc),
        )


def invoke_screen_tool(controller, *, include_controller=True):
    graph = StateGraph(ScreenToolTestState,context_schema=ScreenToolContext,)
    graph.add_node("tools", ToolNode([analyze_current_screen]))
    graph.add_edge(START, "tools")
    compiled = graph.compile()

    context: ScreenToolContext = {}
    if include_controller:
        context["screen_observation_controller"] = controller

    state = {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "analyze_current_screen",
                        "args": {"question": "Describe the screen"},
                        "id": "screen-call-1",
                        "type": "tool_call",
                    }
                ],
            )
        ]
    }

    return compiled.invoke(state, context=context)


def test_screen_tool_rejects_missing_controller():
    service = FakeCaptureService()
    controller = ScreenObservationController(service)

    result = invoke_screen_tool(controller, include_controller=False)

    assert "error" in result["messages"][-1].content.lower()
    assert service.starts == 0
    assert service.captures == 0


def test_screen_tool_does_not_start_inactive_controller():
    service = FakeCaptureService()
    controller = ScreenObservationController(service)

    result = invoke_screen_tool(controller)

    assert "error" in result["messages"][-1].content.lower()
    assert service.starts == 0
    assert service.captures == 0


def test_screen_tool_requires_existing_observation():
    service = FakeCaptureService()
    controller = ScreenObservationController(service)
    controller.start()

    try:
        result = invoke_screen_tool(controller)

        assert "error" in result["messages"][-1].content.lower()
        assert service.captures == 0
    finally:
        controller.stop()


def test_screen_tool_analyzes_latest_observation():
    service = FakeCaptureService()
    controller = ScreenObservationController(service)
    controller.start()

    try:
        observation = controller.poll()
        assert observation is not None

        with __import__("unittest.mock", fromlist=["patch"]).patch(
            "vision_agent.screen_vision.describe_image_bytes",
            return_value="A blue desktop.",
        ) as describe:
            result = invoke_screen_tool(controller)

        content = result["messages"][-1].content
        assert content.startswith("A blue desktop.\n")
        assert f"Observation fingerprint: {observation.fingerprint}" in content
        describe.assert_called_once_with(
            observation.frame.data,
            "Describe the screen",
        )
        assert service.captures == 1
    finally:
        controller.stop()


def test_screen_tool_rejects_controller_after_stop():
    service = FakeCaptureService()
    controller = ScreenObservationController(service)
    controller.start()
    controller.poll()
    controller.stop()

    result = invoke_screen_tool(controller)

    assert "error" in result["messages"][-1].content.lower()
    assert service.captures == 1
