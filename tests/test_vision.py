
import base64
import struct
import zlib

from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama


def make_red_rectangle_png():
    width = height = 32
    rows = []

    for y in range(height):
        row = bytearray([0])  # PNG filter: None
        for x in range(width):
            if 4 <= x < 28 and 4 <= y < 28:
                row.extend((255, 0, 0))  # Red
            else:
                row.extend((255, 255, 255))  # White
        rows.append(bytes(row))

    def chunk(kind, data):
        payload = kind + data
        crc = zlib.crc32(payload) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + payload + struct.pack(">I", crc)

    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">2I5B", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(b"".join(rows)))
        + chunk(b"IEND", b"")
    )
    return base64.b64encode(png).decode("ascii")


def test_qwen_vl_identifies_red_in_image():
    image_data = make_red_rectangle_png()
    model = ChatOllama(model="qwen3-vl:latest", temperature=0)

    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": (
                    "What is the main color of the large rectangle? "
                    "Answer with just the color name."
                ),
            },
            {
                "type": "image",
                "base64": image_data,
                "mime_type": "image/png",
            },
        ]
    )

    response = model.invoke([message])

    assert isinstance(response.content, str)
    assert "red" in response.content.lower(), (
        f"Expected the model to identify red, got: {response.content!r}"
    )
