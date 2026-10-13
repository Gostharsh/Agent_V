from vision_agent.screen_observation import ScreenObservationController
from vision_agent.vision import describe_image_bytes





def analyze_current_screen(
    controller: ScreenObservationController,
    question: str,
) -> str:
    """Analyze a fresh screen observation without initiating capture."""
    observation = controller.get_fresh_observation()
    description = describe_image_bytes(
        observation.frame.data,
        question,
    )
    return (
        f"{description}\n"
        f"Observation fingerprint: {observation.fingerprint}"
    )