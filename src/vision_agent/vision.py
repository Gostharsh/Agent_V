
import base64
import mimetypes
from pathlib import Path

from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama


MODEL_NAME = "qwen3-vl:latest"

SUPPORTED_IMAGE_TYPES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
}


def describe_image(image_path: str, question: str) -> str:
    """Answer a question about a local image using Qwen3-VL."""
    if not isinstance(image_path, str) or not image_path.strip():
        raise ValueError("Image path must be a non-empty string")

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must be a non-empty string")

    path = Path(image_path).expanduser()

    if not path.is_file():
        raise FileNotFoundError(f"Image file not found: {path}")

    if path.suffix.lower() not in SUPPORTED_IMAGE_TYPES:
        raise ValueError("Unsupported image format")

    mime_type, _ = mimetypes.guess_type(path.name)
    if mime_type not in {
        "image/png",
        "image/jpeg",
        "image/webp",
    }:
        raise ValueError("Unsupported image MIME type")

    image_data = base64.b64encode(path.read_bytes()).decode("ascii")

    model = ChatOllama(model=MODEL_NAME, temperature=0)

    message = HumanMessage(
        content=[
            {"type": "text", "text": question},
            {
                "type": "image",
                "base64": image_data,
                "mime_type": mime_type,
            },
        ]
    )

    response = model.invoke([message])

    if not isinstance(response.content, str) or not response.content.strip():
        raise RuntimeError("The vision model returned an empty response")

    return response.content.strip()
