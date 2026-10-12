
import base64
import mimetypes
from io import BytesIO
from pathlib import Path

from PIL import Image, UnidentifiedImageError
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama


MODEL_NAME = "qwen3-vl:latest"

SUPPORTED_IMAGE_TYPES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
}


def _validate_image_signature(image_data: bytes, suffix: str) -> None:
    """Reject image files whose contents don't match their extension."""
    valid = False

    if suffix == ".png":
        valid = image_data.startswith(b"\x89PNG\r\n\x1a\n")
    elif suffix in {".jpg", ".jpeg"}:
        valid = (
            len(image_data) >= 3
            and image_data.startswith(b"\xff\xd8\xff")
        )
    elif suffix == ".webp":
        valid = (
            len(image_data) >= 12
            and image_data.startswith(b"RIFF")
            and image_data[8:12] == b"WEBP"
        )

    if not valid:
        raise ValueError("Invalid image data")


def _validate_image_decoding(image_data: bytes) -> None:
    """Reject image data that cannot be verified as a valid image."""
    try:
        with Image.open(BytesIO(image_data)) as image:
            image.verify()
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        raise ValueError("Invalid or corrupted image data") from exc




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

    # image_data = base64.b64encode(path.read_bytes()).decode("ascii")
    raw_image_data = path.read_bytes()
    _validate_image_signature(raw_image_data, path.suffix.lower())
    _validate_image_decoding(raw_image_data)

    image_data = base64.b64encode(raw_image_data).decode("ascii")

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



def describe_image_bytes(image_data: bytes, question: str) -> str:
    """Answer a question about validated PNG image bytes without disk I/O."""
    if not isinstance(image_data, bytes) or not image_data:
        raise ValueError("Image data must be non-empty bytes")

    if len(image_data) > 20 * 1024 * 1024:
        raise ValueError("Image data exceeds the 20 MiB limit")

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Image question must be a non-empty string")

    if not image_data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("Image data must be PNG")

    _validate_image_signature(image_data, ".png")
    _validate_image_decoding(image_data)

    # Verify that the decoded image is actually PNG, not merely data
    # with a PNG signature.
    try:
        with Image.open(BytesIO(image_data)) as image:
            if image.format != "PNG":
                raise ValueError("Image data must be PNG")
            image.verify()
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        raise ValueError("Invalid or corrupted PNG image data") from exc

    encoded_image = base64.b64encode(image_data).decode("ascii")
    model = ChatOllama(model=MODEL_NAME, temperature=0)

    message = HumanMessage(
        content=[
            {"type": "text", "text": question.strip()},
            {
                "type": "image",
                "base64": encoded_image,
                "mime_type": "image/png",
            },
        ]
    )

    response = model.invoke([message])

    if not isinstance(response.content, str) or not response.content.strip():
        raise RuntimeError("The vision model returned an empty response")

    return response.content.strip()
