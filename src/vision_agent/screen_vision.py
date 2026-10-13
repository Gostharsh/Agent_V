from vision_agent.screen_observation import ScreenObservationController
from vision_agent.vision import describe_image_bytes





def analyze_current_screen(
    controller: ScreenObservationController,
    question: str,
) -> str:
    """Analyze a fresh screen observation without initiating capture."""
    observation = controller.get_fresh_observation()
    return describe_image_bytes(observation.frame.data, question)
