from vision_agent.screen_observation import ScreenObservationController
from vision_agent.vision import describe_image_bytes


def analyze_current_screen(
    controller: ScreenObservationController,
    question: str,
) -> str:
    """Analyze the latest screen observation without initiating capture."""
    if not controller.active:
        raise RuntimeError("Screen observation is not active")

    observation = controller.latest_observation
    if observation is None:
        raise RuntimeError("No screen observation is available")

    return describe_image_bytes(observation.frame.data, question)
