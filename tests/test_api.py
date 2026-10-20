"""API tests for the Intelligent Image Document Analyzer.

The invoice pipeline is replaced by a fake in every test (see ``pipeline_spy``),
so these tests never start Tesseract.
"""

import asyncio
import copy
import logging
import random
from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import main as main_module
from app.main import app
from app.ocr import OCREngineError, OCREngineNotFoundError, OCRLanguageDataError
from app.pipeline import ImageDecodeError

MAX_UPLOAD_SIZE_BYTES = 10 * 1024 * 1024  # The documented 10 MB API limit

FAKE_PIPELINE_RESULT = {
    "raw_ocr_text": "  Acme   Traders \r\nInvoice No:\tINV-001  \r\nTotal: Rs. 1,500.00\f",
    "cleaned_text": "Acme Traders\nInvoice No: INV-001\nTotal: Rs. 1,500.00",
    "invoice_fields": {
        "invoice_number": "INV-001",
        "invoice_date": None,
        "vendor_name": "Acme Traders",
        "subtotal": None,
        "tax_amount": None,
        "total_amount": "1500.00",
    },
}
EXPECTED_FIELD_NAMES = {
    "invoice_number",
    "invoice_date",
    "vendor_name",
    "subtotal",
    "tax_amount",
    "total_amount",
}
EXPECTED_UPLOAD_KEYS = {
    "filename",
    "format",
    "width",
    "height",
    "size_bytes",
    "status",
    "message",
    "raw_ocr_text",
    "cleaned_text",
    "invoice_fields",
}


class PipelineSpy:
    """Fake for ``process_invoice_image`` that records how it was called."""

    def __init__(self) -> None:
        self.calls: list[bytes] = []
        self.result = copy.deepcopy(FAKE_PIPELINE_RESULT)
        self.error: Exception | None = None
        self.ran_inside_event_loop: bool | None = None

    def __call__(self, image_bytes: bytes) -> dict:
        self.calls.append(image_bytes)
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            self.ran_inside_event_loop = False  # Worker thread: no running loop
        else:
            self.ran_inside_event_loop = True
        if self.error is not None:
            raise self.error
        return self.result


@pytest.fixture(autouse=True)
def pipeline_spy(monkeypatch: pytest.MonkeyPatch) -> PipelineSpy:
    """Replace the pipeline for every test so Tesseract is never needed."""
    spy = PipelineSpy()
    monkeypatch.setattr(main_module, "process_invoice_image", spy)
    return spy


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


# --- Successful uploads and pipeline integration -----------------------------


def test_upload_valid_png_returns_metadata_and_pipeline_results(
    client: TestClient,
) -> None:
    png_bytes = make_image_bytes("PNG", size=(300, 200))

    response = upload(client, png_bytes, filename="invoice.png")

    assert response.status_code == 200
    body = response.json()
    assert set(body) == EXPECTED_UPLOAD_KEYS
    assert body["filename"] == "invoice.png"
    assert body["format"] == "PNG"
    assert body["width"] == 300
    assert body["height"] == 200
    assert body["size_bytes"] == len(png_bytes)
    assert body["status"] == "processed"
    assert body["message"]
    assert body["raw_ocr_text"] == FAKE_PIPELINE_RESULT["raw_ocr_text"]
    assert body["cleaned_text"] == FAKE_PIPELINE_RESULT["cleaned_text"]
    assert body["invoice_fields"] == FAKE_PIPELINE_RESULT["invoice_fields"]


def test_upload_response_has_all_six_invoice_fields_with_nulls_for_missing_ones(
    client: TestClient, pipeline_spy: PipelineSpy
) -> None:
    pipeline_spy.result["invoice_fields"] = dict.fromkeys(EXPECTED_FIELD_NAMES)

    response = upload(client, make_image_bytes("PNG"))

    assert response.status_code == 200
    assert response.json()["invoice_fields"] == dict.fromkeys(EXPECTED_FIELD_NAMES)


def test_upload_passes_the_uploaded_bytes_to_the_pipeline(
    client: TestClient, pipeline_spy: PipelineSpy
) -> None:
    png_bytes = make_image_bytes("PNG", size=(90, 60))

    response = upload(client, png_bytes)

    assert response.status_code == 200
    assert pipeline_spy.calls == [png_bytes]
    assert type(pipeline_spy.calls[0]) is bytes  # Not a BytesIO or Pillow image


def test_pipeline_runs_in_a_worker_thread_not_in_the_event_loop(
    client: TestClient, pipeline_spy: PipelineSpy
) -> None:
    upload(client, make_image_bytes("PNG"))

    assert pipeline_spy.ran_inside_event_loop is False


def test_upload_valid_jpeg_is_accepted_and_processed(
    client: TestClient, pipeline_spy: PipelineSpy
) -> None:
    jpeg_bytes = make_image_bytes("JPEG", size=(160, 90))

    response = upload(client, jpeg_bytes, filename="scan.jpg", content_type="image/jpeg")

    assert response.status_code == 200
    body = response.json()
    assert body["format"] == "JPEG"
    assert body["width"] == 160
    assert body["height"] == 90
    assert body["status"] == "processed"
    assert pipeline_spy.calls == [jpeg_bytes]


# --- Rejected uploads never reach the pipeline -------------------------------


def test_upload_empty_file_returns_400(
    client: TestClient, pipeline_spy: PipelineSpy
) -> None:
    response = upload(client, b"", filename="empty.png")

    assert response.status_code == 400
    assert pipeline_spy.calls == []


def test_upload_plain_text_named_as_png_returns_400(
    client: TestClient, pipeline_spy: PipelineSpy
) -> None:
    response = upload(
        client,
        b"This is just text, not an image.",
        filename="fake.png",
        content_type="image/png",
    )

    assert response.status_code == 400
    assert pipeline_spy.calls == []


def test_upload_truncated_jpeg_is_rejected(
    client: TestClient, pipeline_spy: PipelineSpy
) -> None:
    jpeg_bytes = make_noisy_jpeg_bytes()
    truncated_bytes = jpeg_bytes[: len(jpeg_bytes) // 2]

    response = upload(
        client, truncated_bytes, filename="broken.jpg", content_type="image/jpeg"
    )

    assert response.status_code == 400
    assert pipeline_spy.calls == []


def test_upload_unsupported_gif_returns_415(
    client: TestClient, pipeline_spy: PipelineSpy
) -> None:
    gif_bytes = make_image_bytes("GIF", mode="P")

    response = upload(client, gif_bytes, filename="image.gif", content_type="image/gif")

    assert response.status_code == 415
    assert pipeline_spy.calls == []


def test_upload_over_10_mb_returns_413(
    client: TestClient, pipeline_spy: PipelineSpy
) -> None:
    oversized_bytes = b"\x00" * (MAX_UPLOAD_SIZE_BYTES + 1)

    response = upload(client, oversized_bytes, filename="huge.png")

    assert response.status_code == 413
    assert pipeline_spy.calls == []


def test_upload_image_over_the_pixel_limit_returns_413(
    client: TestClient, pipeline_spy: PipelineSpy, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(main_module, "MAX_IMAGE_PIXELS", 100)  # 50x50 = 2500 pixels

    response = upload(client, make_image_bytes("PNG", size=(50, 50)))

    assert response.status_code == 413
    assert pipeline_spy.calls == []


# --- Contents matter, not metadata -------------------------------------------


def test_upload_valid_png_with_wrong_filename_and_content_type_is_accepted(
    client: TestClient, pipeline_spy: PipelineSpy
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
    assert pipeline_spy.calls == [png_bytes]


def test_upload_text_with_image_filename_and_content_type_is_rejected(
    client: TestClient, pipeline_spy: PipelineSpy
) -> None:
    response = upload(
        client,
        b"not an image at all",
        filename="photo.jpg",
        content_type="image/jpeg",
    )

    assert response.status_code == 400
    assert pipeline_spy.calls == []


# --- Pipeline failures -------------------------------------------------------


def test_upload_without_a_file_returns_422(client: TestClient) -> None:
    response = client.post("/upload")

    assert response.status_code == 422


def test_image_decode_error_returns_422_after_validation_passed(
    client: TestClient, pipeline_spy: PipelineSpy
) -> None:
    pipeline_spy.error = ImageDecodeError("OpenCV could not decode: internal detail")

    response = upload(client, make_image_bytes("PNG"))

    assert response.status_code == 422
    assert "could not be decoded" in response.json()["detail"]
    assert len(pipeline_spy.calls) == 1  # Validation passed, decoding failed
    assert "internal detail" not in response.text


@pytest.mark.parametrize(
    "error_type",
    [OCREngineNotFoundError, OCRLanguageDataError, OCREngineError],
)
def test_ocr_failure_returns_500_with_a_safe_message(
    client: TestClient, pipeline_spy: PipelineSpy, error_type: type[Exception]
) -> None:
    pipeline_spy.error = error_type("failed at /usr/bin/tesseract with secret details")

    response = upload(client, make_image_bytes("PNG"))

    assert response.status_code == 500
    assert response.json() == {
        "detail": "Text recognition failed on the server. Please try again later."
    }
    assert "tesseract" not in response.text
    assert "secret" not in response.text


def test_ocr_failure_details_are_logged_for_the_operator(
    client: TestClient,
    pipeline_spy: PipelineSpy,
    caplog: pytest.LogCaptureFixture,
) -> None:
    pipeline_spy.error = OCREngineNotFoundError("tesseract executable not found")

    with caplog.at_level(logging.ERROR, logger="app.main"):
        upload(client, make_image_bytes("PNG"))

    assert "tesseract executable not found" in caplog.text


def test_unexpected_pipeline_error_is_not_hidden_by_the_endpoint(
    client: TestClient, pipeline_spy: PipelineSpy
) -> None:
    pipeline_spy.error = RuntimeError("programming error")

    with pytest.raises(RuntimeError, match="programming error"):
        upload(client, make_image_bytes("PNG"))


# --- OpenAPI documentation ---------------------------------------------------


def test_openapi_documents_the_upload_response_and_nullable_invoice_fields(
    client: TestClient,
) -> None:
    schema = client.get("/openapi.json").json()["components"]["schemas"]

    assert EXPECTED_UPLOAD_KEYS == set(schema["UploadResponse"]["properties"])
    invoice_fields = schema["InvoiceFields"]
    assert set(invoice_fields["properties"]) == EXPECTED_FIELD_NAMES
    assert set(invoice_fields["required"]) == EXPECTED_FIELD_NAMES
    for name in EXPECTED_FIELD_NAMES:
        types = {option.get("type") for option in invoice_fields["properties"][name]["anyOf"]}
        assert types == {"string", "null"}