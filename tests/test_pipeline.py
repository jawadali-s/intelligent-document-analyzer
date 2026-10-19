"""Unit tests for app.pipeline.

``decode_image`` is tested with images encoded in memory by OpenCV. In the
``process_invoice_image`` tests, preprocessing, OCR, cleaning and extraction
are replaced with recording fakes, so these tests never start Tesseract.
"""

import cv2
import numpy as np
import pytest

from app import pipeline
from app.extraction import InvalidExtractionInputError
from app.image_processing import InvalidImageError
from app.ocr import OCREngineNotFoundError
from app.pipeline import (
    ImageDecodeError,
    InvalidImageBytesInputError,
    decode_image,
    process_invoice_image,
)
from app.text_cleaning import InvalidTextInputError

EXPECTED_RESULT_KEYS = {"raw_ocr_text", "cleaned_text", "invoice_fields"}
EXPECTED_FIELD_NAMES = {
    "invoice_number",
    "invoice_date",
    "vendor_name",
    "subtotal",
    "tax_amount",
    "total_amount",
}
EXPECTED_STEP_ORDER = ["decode", "preprocess", "ocr", "clean", "extract"]

RAW_TEXT = "  Acme   Traders \r\nInvoice No:\tINV-001  \r\n\r\n\r\nTotal: Rs. 1,500.00\f"
CLEANED_TEXT = "Acme Traders\nInvoice No: INV-001\n\nTotal: Rs. 1,500.00"
STUB_FIELDS = {
    "invoice_number": "INV-001",
    "invoice_date": None,
    "vendor_name": "Acme Traders",
    "subtotal": None,
    "tax_amount": None,
    "total_amount": "1500.00",
}


def make_bgr_image(height: int = 30, width: int = 40) -> np.ndarray:
    """Deterministic colour image with a white square."""
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :] = (10, 120, 240)  # BGR
    image[5:15, 5:15] = (255, 255, 255)
    return image


def encode_image(image: np.ndarray, extension: str = ".png") -> bytes:
    """Encode an image array into file bytes with OpenCV."""
    success, buffer = cv2.imencode(extension, image)
    assert success
    return buffer.tobytes()


PNG_BYTES = encode_image(make_bgr_image())


class StubbedPipeline:
    """Replaces the four processing functions in app.pipeline with fakes.

    ``decode_image`` stays real but is wrapped so its call is recorded too.
    Every call is appended to ``calls`` and its argument is stored in
    ``received``. To make a step fail, put an exception in ``errors``.
    """

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.calls: list[str] = []
        self.received: dict[str, object] = {}
        self.errors: dict[str, Exception] = {}
        self.preprocessed = np.zeros((30, 40), dtype=np.uint8)
        self.fields = dict(STUB_FIELDS)

        real_decode = pipeline.decode_image

        def recording_decode(image_bytes: object) -> np.ndarray:
            self.calls.append("decode")
            self.received["decode"] = image_bytes
            return real_decode(image_bytes)  # type: ignore[arg-type]

        monkeypatch.setattr(pipeline, "decode_image", recording_decode)
        monkeypatch.setattr(pipeline, "preprocess_image", self._preprocess)
        monkeypatch.setattr(pipeline, "extract_text", self._ocr)
        monkeypatch.setattr(pipeline, "clean_ocr_text", self._clean)
        monkeypatch.setattr(pipeline, "extract_invoice_fields", self._extract)

    def _step(self, name: str, argument: object, result: object) -> object:
        self.calls.append(name)
        self.received[name] = argument
        if name in self.errors:
            raise self.errors[name]
        return result

    def _preprocess(self, image: np.ndarray) -> object:
        return self._step("preprocess", image, self.preprocessed)

    def _ocr(self, image: np.ndarray) -> object:
        return self._step("ocr", image, RAW_TEXT)

    def _clean(self, text: str) -> object:
        return self._step("clean", text, CLEANED_TEXT)

    def _extract(self, text: str) -> object:
        return self._step("extract", text, self.fields)


@pytest.fixture
def stubs(monkeypatch: pytest.MonkeyPatch) -> StubbedPipeline:
    """Fakes for every processing step, with call recording."""
    return StubbedPipeline(monkeypatch)


# --- decode_image: valid input -----------------------------------------------


def test_decode_valid_png_returns_identical_bgr_pixels() -> None:
    original = make_bgr_image(height=30, width=40)

    result = decode_image(encode_image(original))

    assert isinstance(result, np.ndarray)
    assert result.dtype == np.uint8
    assert result.shape == (30, 40, 3)
    assert np.array_equal(result, original)  # PNG is lossless


@pytest.mark.parametrize("extension", [".png", ".jpg", ".tiff"])
def test_decode_supported_formats_return_three_channel_uint8_array(
    extension: str,
) -> None:
    result = decode_image(encode_image(make_bgr_image(24, 36), extension))

    assert result.dtype == np.uint8
    assert result.shape == (24, 36, 3)


def test_decode_grayscale_png_is_expanded_to_three_channels() -> None:
    gray = np.full((20, 30), 128, dtype=np.uint8)

    result = decode_image(encode_image(gray))

    assert result.shape == (20, 30, 3)
    assert np.all(result == 128)


def test_decode_png_with_alpha_channel_drops_alpha() -> None:
    bgra = np.dstack([make_bgr_image(20, 30), np.full((20, 30), 255, np.uint8)])

    result = decode_image(encode_image(bgra))

    assert result.shape == (20, 30, 3)


@pytest.mark.parametrize("wrapper", [bytes, bytearray, memoryview])
def test_decode_accepts_bytes_like_inputs(wrapper: type) -> None:
    original = make_bgr_image()

    result = decode_image(wrapper(encode_image(original)))

    assert np.array_equal(result, original)


def test_decode_does_not_modify_the_input_bytes() -> None:
    data = bytearray(PNG_BYTES)

    result = decode_image(data)
    result[:] = 0  # Changing the result must not touch the input

    assert bytes(data) == PNG_BYTES


# --- decode_image: invalid input ---------------------------------------------


@pytest.mark.parametrize("empty", [b"", bytearray(), memoryview(b"")])
def test_decode_empty_bytes_raises_image_decode_error(empty: object) -> None:
    with pytest.raises(ImageDecodeError, match="empty"):
        decode_image(empty)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "bad_bytes",
    [
        b"This is plain text, not an image.",
        b"\x00" * 200,
        b"\x89PNG\r\n\x1a\n",  # PNG signature with no image data
        PNG_BYTES[:12],  # Truncated header
    ],
    ids=["text", "zeros", "signature-only", "truncated-header"],
)
def test_decode_invalid_image_bytes_raises_image_decode_error(
    bad_bytes: bytes,
) -> None:
    with pytest.raises(ImageDecodeError, match="could not be decoded"):
        decode_image(bad_bytes)


@pytest.mark.parametrize(
    "bad_input",
    [
        None,
        "image.png",
        123,
        1.5,
        [1, 2, 3],
        {"image": b"data"},
        np.zeros((4, 4), dtype=np.uint8),
    ],
    ids=["none", "str", "int", "float", "list", "dict", "ndarray"],
)
def test_decode_non_bytes_input_raises_invalid_input_error(bad_input: object) -> None:
    with pytest.raises(InvalidImageBytesInputError, match="image_bytes must be"):
        decode_image(bad_input)  # type: ignore[arg-type]


def test_decode_input_error_message_names_the_received_type() -> None:
    with pytest.raises(InvalidImageBytesInputError, match="NoneType"):
        decode_image(None)  # type: ignore[arg-type]


def test_decode_exception_types_are_distinct_and_specific() -> None:
    assert issubclass(InvalidImageBytesInputError, TypeError)
    assert issubclass(ImageDecodeError, ValueError)
    assert not issubclass(ImageDecodeError, InvalidImageBytesInputError)
    assert not issubclass(InvalidImageBytesInputError, ImageDecodeError)


def test_decode_wraps_opencv_error_and_keeps_it_as_cause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = cv2.error("simulated OpenCV failure")

    def failing_imdecode(*args: object, **kwargs: object) -> np.ndarray:
        raise original

    monkeypatch.setattr(pipeline.cv2, "imdecode", failing_imdecode)

    with pytest.raises(ImageDecodeError, match="OpenCV could not decode") as exc_info:
        decode_image(PNG_BYTES)

    assert exc_info.value.__cause__ is original


# --- process_invoice_image: order and data flow ------------------------------


def test_pipeline_runs_each_step_once_in_the_documented_order(
    stubs: StubbedPipeline,
) -> None:
    process_invoice_image(PNG_BYTES)

    assert stubs.calls == EXPECTED_STEP_ORDER


def test_pipeline_decodes_the_bytes_it_was_given(stubs: StubbedPipeline) -> None:
    process_invoice_image(PNG_BYTES)

    assert stubs.received["decode"] is PNG_BYTES


def test_preprocessing_receives_the_decoded_image(stubs: StubbedPipeline) -> None:
    process_invoice_image(PNG_BYTES)

    image = stubs.received["preprocess"]
    assert isinstance(image, np.ndarray)
    assert image.dtype == np.uint8
    assert np.array_equal(image, make_bgr_image())


def test_ocr_receives_the_preprocessed_image(stubs: StubbedPipeline) -> None:
    process_invoice_image(PNG_BYTES)

    assert stubs.received["ocr"] is stubs.preprocessed


def test_cleaning_receives_the_raw_ocr_text(stubs: StubbedPipeline) -> None:
    process_invoice_image(PNG_BYTES)

    assert stubs.received["clean"] == RAW_TEXT


def test_extraction_receives_the_cleaned_text_not_the_raw_text(
    stubs: StubbedPipeline,
) -> None:
    process_invoice_image(PNG_BYTES)

    assert stubs.received["extract"] == CLEANED_TEXT
    assert stubs.received["extract"] != RAW_TEXT


@pytest.mark.parametrize("wrapper", [bytearray, memoryview])
def test_pipeline_accepts_bytearray_and_memoryview(
    stubs: StubbedPipeline, wrapper: type
) -> None:
    result = process_invoice_image(wrapper(PNG_BYTES))

    assert set(result) == EXPECTED_RESULT_KEYS
    assert stubs.calls == EXPECTED_STEP_ORDER


# --- process_invoice_image: result structure ---------------------------------


def test_result_contains_exactly_the_three_documented_keys(
    stubs: StubbedPipeline,
) -> None:
    result = process_invoice_image(PNG_BYTES)

    assert set(result) == EXPECTED_RESULT_KEYS


def test_result_keeps_raw_text_before_cleaning_and_cleaned_text_separate(
    stubs: StubbedPipeline,
) -> None:
    result = process_invoice_image(PNG_BYTES)

    assert result["raw_ocr_text"] == RAW_TEXT  # Not cleaned
    assert result["cleaned_text"] == CLEANED_TEXT
    assert result["raw_ocr_text"] != result["cleaned_text"]


def test_result_invoice_fields_have_all_six_keys_and_the_extracted_values(
    stubs: StubbedPipeline,
) -> None:
    result = process_invoice_image(PNG_BYTES)

    fields = result["invoice_fields"]
    assert isinstance(fields, dict)
    assert set(fields) == EXPECTED_FIELD_NAMES
    assert fields == STUB_FIELDS


def test_pipeline_does_not_modify_the_input_bytes(stubs: StubbedPipeline) -> None:
    data = bytearray(PNG_BYTES)

    process_invoice_image(data)

    assert bytes(data) == PNG_BYTES


# --- process_invoice_image: real cleaning and extraction ---------------------


def test_real_cleaning_and_extraction_with_mocked_preprocessing_and_ocr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_text = (
        "Acme Traders\r\nInvoice No:  INV-001\r\nDate: 2026-10-17\r\n"
        "Total:\tRs. 1,500.00\r\n"
    )
    monkeypatch.setattr(
        pipeline, "preprocess_image", lambda image: np.zeros((10, 10), np.uint8)
    )
    monkeypatch.setattr(pipeline, "extract_text", lambda image: raw_text)

    result = process_invoice_image(PNG_BYTES)

    assert result["raw_ocr_text"] == raw_text
    assert result["cleaned_text"] == (
        "Acme Traders\nInvoice No: INV-001\nDate: 2026-10-17\nTotal: Rs. 1,500.00"
    )
    fields = result["invoice_fields"]
    assert set(fields) == EXPECTED_FIELD_NAMES
    assert fields["invoice_number"] == "INV-001"
    assert fields["invoice_date"] == "2026-10-17"
    assert fields["vendor_name"] == "Acme Traders"
    assert fields["total_amount"] == "1500.00"
    assert fields["subtotal"] is None
    assert fields["tax_amount"] is None


def test_empty_ocr_output_gives_empty_texts_and_all_fields_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        pipeline, "preprocess_image", lambda image: np.zeros((10, 10), np.uint8)
    )
    monkeypatch.setattr(pipeline, "extract_text", lambda image: "")

    result = process_invoice_image(PNG_BYTES)

    assert result["raw_ocr_text"] == ""
    assert result["cleaned_text"] == ""
    assert result["invoice_fields"] == dict.fromkeys(EXPECTED_FIELD_NAMES)


# --- process_invoice_image: invalid image data -------------------------------


@pytest.mark.parametrize(
    "bad_bytes",
    [b"", b"not an image", b"\x00" * 100],
    ids=["empty", "text", "zeros"],
)
def test_invalid_image_bytes_raise_decode_error_before_any_other_step(
    stubs: StubbedPipeline, bad_bytes: bytes
) -> None:
    with pytest.raises(ImageDecodeError):
        process_invoice_image(bad_bytes)

    assert stubs.calls == ["decode"]


@pytest.mark.parametrize("bad_input", [None, "image.png", 42, [1, 2]])
def test_non_bytes_input_raises_input_error_before_any_other_step(
    stubs: StubbedPipeline, bad_input: object
) -> None:
    with pytest.raises(InvalidImageBytesInputError):
        process_invoice_image(bad_input)  # type: ignore[arg-type]

    assert stubs.calls == ["decode"]


# --- process_invoice_image: errors are not swallowed -------------------------


@pytest.mark.parametrize(
    ("step", "exception_type"),
    [
        ("preprocess", InvalidImageError),
        ("ocr", OCREngineNotFoundError),
        ("clean", InvalidTextInputError),
        ("extract", InvalidExtractionInputError),
    ],
)
def test_step_error_propagates_unchanged_and_stops_the_pipeline(
    stubs: StubbedPipeline, step: str, exception_type: type[Exception]
) -> None:
    error = exception_type(f"{step} failed")
    stubs.errors[step] = error

    with pytest.raises(exception_type, match=f"{step} failed") as exc_info:
        process_invoice_image(PNG_BYTES)

    assert exc_info.value is error  # Same object: not wrapped or replaced
    last_call = EXPECTED_STEP_ORDER.index(step) + 1
    assert stubs.calls == EXPECTED_STEP_ORDER[:last_call]  # Later steps never ran


@pytest.mark.parametrize("step", ["preprocess", "ocr", "clean", "extract"])
def test_unexpected_exception_in_any_step_is_not_swallowed(
    stubs: StubbedPipeline, step: str
) -> None:
    stubs.errors[step] = RuntimeError("programming error")

    with pytest.raises(RuntimeError, match="programming error"):
        process_invoice_image(PNG_BYTES)


def test_original_cause_of_a_step_error_is_preserved(
    stubs: StubbedPipeline,
) -> None:
    cause = OSError("tesseract binary missing")
    error = OCREngineNotFoundError("OCR engine not found")
    error.__cause__ = cause
    stubs.errors["ocr"] = error

    with pytest.raises(OCREngineNotFoundError) as exc_info:
        process_invoice_image(PNG_BYTES)

    assert exc_info.value.__cause__ is cause