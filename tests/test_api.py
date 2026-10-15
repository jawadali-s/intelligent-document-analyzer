"""API tests for the Intelligent Image Document Analyzer."""

import random
from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app

MAX_UPLOAD_SIZE_BYTES = 10 * 1024 * 1024  # The documented 10 MB API limit


@pytest.fixture
def client() -> TestClient:
    """Provide a fresh test client for each test."""
    return TestClient(app)


def make_image_bytes(
    image_format: str,
    size: tuple[int, int] = (120, 80),
    mode: str = "RGB",
) -> bytes:
    """Create a small image in memory and return its encoded bytes."""
    buffer = BytesIO()
    Image.new(mode, size).save(buffer, format=image_format)
    return buffer.getvalue()


def make_noisy_jpeg_bytes() -> bytes:
    """Create a JPEG of random pixels so its data is large enough to truncate."""
    rng = random.Random(1234)  # Fixed seed keeps the test deterministic
    width, height = 64, 64
    noise = rng.randbytes(width * height * 3)
    buffer = BytesIO()
    Image.frombytes("RGB", (width, height), noise).save(buffer, format="JPEG")
    return buffer.getvalue()


def upload(
    client: TestClient,
    content: bytes,
    filename: str = "invoice.png",
    content_type: str = "image/png",
):
    """Send a file to POST /upload using the form field name `file`."""
    return client.post("/upload", files={"file": (filename, content, content_type)})


# --- Basic endpoints ---------------------------------------------------------


def test_root_returns_project_info(client: TestClient) -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "project": "Intelligent Image Document Analyzer",
        "message": "API is running",
    }


def test_health_returns_healthy_status(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


# --- Successful uploads ------------------------------------------------------


def test_upload_valid_png_returns_expected_details(client: TestClient) -> None:
    png_bytes = make_image_bytes("PNG", size=(300, 200))

    response = upload(client, png_bytes, filename="invoice.png")

    assert response.status_code == 200
    body = response.json()
    assert body["filename"] == "invoice.png"
    assert body["format"] == "PNG"
    assert body["width"] == 300
    assert body["height"] == 200
    assert body["size_bytes"] == len(png_bytes)
    assert body["status"] == "valid"
    assert body["message"]  # A confirmation message is present


def test_upload_valid_jpeg_is_accepted(client: TestClient) -> None:
    jpeg_bytes = make_image_bytes("JPEG", size=(160, 90))

    response = upload(client, jpeg_bytes, filename="scan.jpg", content_type="image/jpeg")

    assert response.status_code == 200
    body = response.json()
    assert body["format"] == "JPEG"
    assert body["width"] == 160
    assert body["height"] == 90
    assert body["status"] == "valid"


# --- Rejected uploads --------------------------------------------------------


def test_upload_empty_file_returns_400(client: TestClient) -> None:
    response = upload(client, b"", filename="empty.png")

    assert response.status_code == 400


def test_upload_plain_text_named_as_png_returns_400(client: TestClient) -> None:
    response = upload(
        client,
        b"This is just text, not an image.",
        filename="fake.png",
        content_type="image/png",
    )

    assert response.status_code == 400


def test_upload_truncated_jpeg_is_rejected(client: TestClient) -> None:
    jpeg_bytes = make_noisy_jpeg_bytes()
    truncated_bytes = jpeg_bytes[: len(jpeg_bytes) // 2]

    response = upload(
        client, truncated_bytes, filename="broken.jpg", content_type="image/jpeg"
    )

    assert response.status_code == 400


def test_upload_unsupported_gif_returns_415(client: TestClient) -> None:
    gif_bytes = make_image_bytes("GIF", mode="P")

    response = upload(client, gif_bytes, filename="image.gif", content_type="image/gif")

    assert response.status_code == 415


def test_upload_over_10_mb_returns_413(client: TestClient) -> None:
    oversized_bytes = b"\x00" * (MAX_UPLOAD_SIZE_BYTES + 1)

    response = upload(client, oversized_bytes, filename="huge.png")

    assert response.status_code == 413


# --- Contents matter, not metadata -------------------------------------------


def test_upload_valid_png_with_wrong_filename_and_content_type_is_accepted(
    client: TestClient,
) -> None:
    png_bytes = make_image_bytes("PNG", size=(50, 40))

    response = upload(
        client, png_bytes, filename="notes.txt", content_type="text/plain"
    )

    assert response.status_code == 200
    body = response.json()
    assert body["format"] == "PNG"  # Detected from the bytes
    assert body["filename"] == "notes.txt"  # Original name is echoed back
    assert body["width"] == 50
    assert body["height"] == 40


def test_upload_text_with_image_filename_and_content_type_is_rejected(
    client: TestClient,
) -> None:
    response = upload(
        client,
        b"not an image at all",
        filename="photo.jpg",
        content_type="image/jpeg",
    )

    assert response.status_code == 400