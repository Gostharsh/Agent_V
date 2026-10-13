
from datetime import datetime, timezone, timedelta
from typing import TypedDict
from unittest.mock import patch

from langchain_core.messages import AIMessage
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import ToolNode

from vision_agent.screen_capture import ScreenFrame
from vision_agent.screen_observation import ScreenObservationController
from vision_agent.tools import click_screen_position_tool


class MouseToolState(TypedDict):
    messages: list


class MouseToolContext(TypedDict, total=False):
    screen_observation_controller: ScreenObservationController


class FakeCaptureService:
    def __init__(self):
        self.active = False
        self.captures = 0

    def start(self):
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


def invoke_click_tool(controller, *, include_controller=True):
    graph = StateGraph(
        MouseToolState,
        context_schema=MouseToolContext,
    )
    graph.add_node("tools", ToolNode([click_screen_position_tool]))
    graph.add_edge(START, "tools")
    compiled = graph.compile()

    context: MouseToolContext = {}
    if include_controller:
        context["screen_observation_controller"] = controller

    state = {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "click_screen_position_tool",
                        "args": {"x": 100, "y": 200},
                        "id": "mouse-call-1",
                        "type": "tool_call",
                    }
                ],
            )
        ]
    }
    return compiled.invoke(state, context=context)


def test_click_tool_rejects_missing_controller():
    service = FakeCaptureService()
    controller = ScreenObservationController(service)

    with patch("vision_agent.tools.click_screen_position") as click:
        result = invoke_click_tool(controller, include_controller=False)

    assert "error" in result["messages"][-1].content.lower()
    click.assert_not_called()
    assert service.captures == 0


def test_click_tool_rejects_inactive_observation():
    service = FakeCaptureService()
    controller = ScreenObservationController(service)

    with patch("vision_agent.tools.click_screen_position") as click:
        result = invoke_click_tool(controller)

    assert "error" in result["messages"][-1].content.lower()
    click.assert_not_called()
    assert service.captures == 0


def test_click_tool_allows_click_with_fresh_observation():
    service = FakeCaptureService()
    controller = ScreenObservationController(service)
    controller.start()

    try:
        controller.poll()

        with patch(
            "vision_agent.tools.click_screen_position",
            return_value="Clicked screen position (100, 200).",
        ) as click:
            result = invoke_click_tool(controller)

        assert result["messages"][-1].content == (
            "Clicked screen position (100, 200)."
        )
        click.assert_called_once_with(100, 200)
        assert service.captures == 1
    finally:
        controller.stop()



def test_click_tool_rejects_stale_observation():
    from dataclasses import replace

    from vision_agent.screen_observation import ScreenObservation

    now = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    service = FakeCaptureService()
    controller = ScreenObservationController(
        service,
        wall_clock=lambda: now,
    )
    controller.start()

    try:
        controller.poll()
        observation = controller.latest_observation
        assert observation is not None

        controller._latest_observation = replace(
            observation,
            frame=replace(
                observation.frame,
                captured_at=now - timedelta(seconds=10),
            ),
        )

        with patch("vision_agent.tools.click_screen_position") as click:
            result = invoke_click_tool(controller)

        assert "stale" in result["messages"][-1].content.lower()
        click.assert_not_called()
        assert service.captures == 1
    finally:
        controller.stop()
