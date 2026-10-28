"""Unit tests for app.pipeline.

``decode_image`` is tested with images encoded in memory by OpenCV. In the
``process_invoice_image`` tests, preprocessing, OCR, cleaning and extraction
are replaced with recording fakes, so these tests never start Tesseract.
"""

import logging

import cv2
import numpy as np
import pytest

from app import pipeline
from app.extraction import InvalidExtractionInputError
from app.image_processing import InvalidImageError
from app.ocr import OCRConfigError, OCREngineError, OCREngineNotFoundError, OCRError
from app.pipeline import (
    ImageDecodeError,
    InvalidImageBytesInputError,
    decode_image,
    process_invoice_image,
)
from app.text_cleaning import InvalidTextInputError, clean_ocr_text
from invoice_ocr_samples import (
    BLUEPEAK_EXPECTED,
    BLUEPEAK_PSM4,
    BLUEPEAK_PSM6,
    GREENFIELD_EXPECTED,
    GREENFIELD_PSM4,
    GREENFIELD_PSM6,
    NORTHSTAR_EXPECTED,
    NORTHSTAR_PSM4,
    NORTHSTAR_PSM6,
)

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


# --- process_invoice_image: opt-in PSM fallback ------------------------------

COMPLETE_TEXT = (
    "Acme Traders\nInvoice No: INV-001\nInvoice Date: 2026-10-17\n"
    "Subtotal: 100.00\nTax: 17.00\nTotal: 117.00"
)


class TwoPassOCR:
    """Fake preprocessing and OCR that returns different text for each PSM.

    Cleaning and extraction stay real. Every OCR call is recorded as
    ``(image, psm)``. Put an exception in ``errors[psm]`` to make that pass fail.
    """

    def __init__(self, monkeypatch: pytest.MonkeyPatch, texts: dict[int, str]) -> None:
        self.texts = texts
        self.calls: list[tuple[object, int]] = []
        self.errors: dict[int, Exception] = {}
        self.preprocess_calls = 0
        self.preprocessed = np.zeros((10, 10), dtype=np.uint8)
        monkeypatch.setattr(pipeline, "preprocess_image", self._preprocess)
        monkeypatch.setattr(pipeline, "extract_text", self._ocr)

    @property
    def psms(self) -> list[int]:
        return [psm for _, psm in self.calls]

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        self.preprocess_calls += 1
        return self.preprocessed

    def _ocr(self, image: object, psm: int = 6) -> str:
        self.calls.append((image, psm))
        if psm in self.errors:
            raise self.errors[psm]
        return self.texts[psm]


@pytest.fixture
def make_ocr(monkeypatch: pytest.MonkeyPatch):
    """Factory for a TwoPassOCR with the given text for each PSM."""

    def factory(primary: str, fallback: str = "") -> TwoPassOCR:
        return TwoPassOCR(monkeypatch, {6: primary, 4: fallback})

    return factory


def test_fallback_is_off_by_default_even_when_fields_are_missing(make_ocr) -> None:
    ocr = make_ocr(GREENFIELD_PSM6, GREENFIELD_PSM4)

    result = process_invoice_image(PNG_BYTES)

    assert ocr.psms == [6]
    assert result["invoice_fields"]["invoice_number"] is None
    assert result["invoice_fields"]["total_amount"] is None


def test_fallback_none_behaves_like_no_fallback(make_ocr) -> None:
    ocr = make_ocr(GREENFIELD_PSM6, GREENFIELD_PSM4)

    result = process_invoice_image(PNG_BYTES, fallback_psm=None)

    assert ocr.psms == [6]
    assert result == process_invoice_image(PNG_BYTES)


def test_fallback_runs_after_the_primary_pass_when_fields_are_missing(make_ocr) -> None:
    ocr = make_ocr(GREENFIELD_PSM6, GREENFIELD_PSM4)

    process_invoice_image(PNG_BYTES, fallback_psm=4)

    assert ocr.psms == [6, 4]


def test_fallback_reuses_the_preprocessed_image(make_ocr) -> None:
    ocr = make_ocr(GREENFIELD_PSM6, GREENFIELD_PSM4)

    process_invoice_image(PNG_BYTES, fallback_psm=4)

    assert ocr.preprocess_calls == 1
    assert all(image is ocr.preprocessed for image, _ in ocr.calls)


def test_fallback_does_not_run_when_every_field_was_extracted(make_ocr) -> None:
    ocr = make_ocr(COMPLETE_TEXT, GREENFIELD_PSM4)

    result = process_invoice_image(PNG_BYTES, fallback_psm=4)

    assert ocr.psms == [6]
    assert all(value is not None for value in result["invoice_fields"].values())


@pytest.mark.parametrize(
    "missing_line",
    ["Invoice No: INV-001", "Invoice Date: 2026-10-17", "Acme Traders",
     "Subtotal: 100.00", "Tax: 17.00", "Total: 117.00"],
)
def test_fallback_runs_when_any_single_field_is_missing(
    make_ocr, missing_line: str
) -> None:
    ocr = make_ocr(COMPLETE_TEXT.replace(missing_line + "\n", "").replace(missing_line, ""))

    process_invoice_image(PNG_BYTES, fallback_psm=4)

    assert ocr.psms == [6, 4]


def test_fallback_runs_when_the_primary_pass_finds_no_text(make_ocr) -> None:
    ocr = make_ocr("", GREENFIELD_PSM4)

    result = process_invoice_image(PNG_BYTES, fallback_psm=4)

    assert ocr.psms == [6, 4]
    assert result["invoice_fields"] == GREENFIELD_EXPECTED
    assert result["raw_ocr_text"] == ""


def test_fallback_fills_only_fields_that_were_none(make_ocr) -> None:
    primary = "Acme Traders\nInvoice No: A-1\nSubtotal: 100.00"
    fallback = (
        "Other Vendor\nInvoice No: B-2\nInvoice Date: 2026-10-17\n"
        "Subtotal: 999.00\nTax: 17.00\nTotal: 117.00"
    )
    make_ocr(primary, fallback)

    fields = process_invoice_image(PNG_BYTES, fallback_psm=4)["invoice_fields"]

    assert fields == {
        "invoice_number": "A-1",  # Primary value kept although the fallback differs
        "invoice_date": "2026-10-17",
        "vendor_name": "Acme Traders",
        "subtotal": "100.00",
        "tax_amount": "17.00",
        "total_amount": "117.00",
    }


def test_primary_values_are_never_overwritten_by_different_fallback_values(make_ocr) -> None:
    primary_fields = process_invoice_image_with(make_ocr, NORTHSTAR_PSM6, None)
    make_ocr_result = process_invoice_image_with(make_ocr, NORTHSTAR_PSM6, GREENFIELD_PSM4)

    for name, value in primary_fields.items():
        if value is not None:
            assert make_ocr_result[name] == value


def process_invoice_image_with(make_ocr, primary: str, fallback: str | None) -> dict:
    """Run the pipeline with the fallback enabled only when fallback text is given."""
    make_ocr(primary, fallback or "")
    psm = None if fallback is None else 4
    return process_invoice_image(PNG_BYTES, fallback_psm=psm)["invoice_fields"]


def test_fallback_that_finds_nothing_leaves_the_primary_result_unchanged(make_ocr) -> None:
    primary_only = process_invoice_image_with(make_ocr, NORTHSTAR_PSM6, None)

    with_empty_fallback = process_invoice_image_with(make_ocr, NORTHSTAR_PSM6, "")

    assert with_empty_fallback == primary_only


def test_result_text_and_structure_come_from_the_primary_pass_only(make_ocr) -> None:
    make_ocr(GREENFIELD_PSM6, GREENFIELD_PSM4)

    result = process_invoice_image(PNG_BYTES, fallback_psm=4)

    assert set(result) == EXPECTED_RESULT_KEYS
    assert result["raw_ocr_text"] == GREENFIELD_PSM6
    assert result["cleaned_text"] == clean_ocr_text(GREENFIELD_PSM6)
    assert list(result["invoice_fields"]) == [
        "invoice_number", "invoice_date", "vendor_name",
        "subtotal", "tax_amount", "total_amount",
    ]


@pytest.mark.parametrize(
    ("primary", "fallback", "expected", "passes"),
    [
        (NORTHSTAR_PSM6, NORTHSTAR_PSM4, NORTHSTAR_EXPECTED, [6, 4]),
        (BLUEPEAK_PSM6, BLUEPEAK_PSM4, BLUEPEAK_EXPECTED, [6]),
        (GREENFIELD_PSM6, GREENFIELD_PSM4, GREENFIELD_EXPECTED, [6, 4]),
    ],
    ids=["northstar", "bluepeak", "greenfield"],
)
def test_three_test_invoices_are_fully_extracted_with_the_psm_4_fallback(
    make_ocr, primary: str, fallback: str, expected: dict, passes: list[int]
) -> None:
    ocr = make_ocr(primary, fallback)

    result = process_invoice_image(PNG_BYTES, fallback_psm=4)

    assert result["invoice_fields"] == expected
    assert ocr.psms == passes


def test_northstar_without_the_fallback_keeps_unlabelled_values_as_none(make_ocr) -> None:
    fields = process_invoice_image_with(make_ocr, NORTHSTAR_PSM6, None)

    assert fields == {
        **NORTHSTAR_EXPECTED,
        "invoice_number": None,
        "subtotal": None,
        "total_amount": None,
    }


def test_fallback_ocr_error_returns_the_primary_result(make_ocr) -> None:
    ocr = make_ocr(GREENFIELD_PSM6, GREENFIELD_PSM4)
    ocr.errors[4] = OCREngineNotFoundError("Tesseract vanished")

    result = process_invoice_image(PNG_BYTES, fallback_psm=4)

    assert ocr.psms == [6, 4]
    assert result["raw_ocr_text"] == GREENFIELD_PSM6
    assert result["invoice_fields"] == {
        **GREENFIELD_EXPECTED,
        "invoice_number": None,
        "invoice_date": None,
        "total_amount": None,
    }


@pytest.mark.parametrize(
    "error",
    [OCREngineNotFoundError("missing"), OCREngineError("failed")],
    ids=["engine-not-found", "engine-error"],
)
def test_fallback_ocr_error_is_logged(
    make_ocr, caplog: pytest.LogCaptureFixture, error: Exception
) -> None:
    ocr = make_ocr(GREENFIELD_PSM6, GREENFIELD_PSM4)
    ocr.errors[4] = error

    with caplog.at_level(logging.WARNING, logger="app.pipeline"):
        process_invoice_image(PNG_BYTES, fallback_psm=4)

    assert any(
        record.levelno >= logging.WARNING and "fallback" in record.getMessage().lower()
        and str(error) in record.getMessage()
        for record in caplog.records
    )


def test_unexpected_exception_in_the_fallback_pass_is_not_swallowed(make_ocr) -> None:
    ocr = make_ocr(GREENFIELD_PSM6, GREENFIELD_PSM4)
    ocr.errors[4] = KeyError("programming error")

    with pytest.raises(KeyError):
        process_invoice_image(PNG_BYTES, fallback_psm=4)


def test_primary_ocr_error_propagates_and_the_fallback_is_not_tried(make_ocr) -> None:
    ocr = make_ocr(GREENFIELD_PSM6, GREENFIELD_PSM4)
    ocr.errors[6] = OCREngineNotFoundError("Tesseract is missing")

    with pytest.raises(OCREngineNotFoundError):
        process_invoice_image(PNG_BYTES, fallback_psm=4)

    assert ocr.psms == [6]


@pytest.mark.parametrize("bad_psm", [99, 0, -1, True, "4", 4.0])
def test_invalid_fallback_psm_is_rejected_before_any_ocr(make_ocr, bad_psm: object) -> None:
    ocr = make_ocr(GREENFIELD_PSM6, GREENFIELD_PSM4)

    with pytest.raises(OCRConfigError):
        process_invoice_image(PNG_BYTES, fallback_psm=bad_psm)  # type: ignore[arg-type]

    assert ocr.calls == []


def test_fallback_log_names_the_filled_fields_but_not_their_values(
    make_ocr, caplog: pytest.LogCaptureFixture
) -> None:
    make_ocr(GREENFIELD_PSM6, GREENFIELD_PSM4)

    with caplog.at_level(logging.INFO, logger="app.pipeline"):
        process_invoice_image(PNG_BYTES, fallback_psm=4)

    messages = " ".join(record.getMessage() for record in caplog.records)
    assert "invoice_number" in messages
    assert "GS-26-000731" not in messages


def test_fallback_is_keyword_only() -> None:
    with pytest.raises(TypeError):
        process_invoice_image(PNG_BYTES, 4)  # type: ignore[misc]


# --- process_invoice_image: opt-in rescaled third pass -----------------------

PARTIAL_TEXT = "Acme Traders\nInvoice No: A-1\nSubtotal PKR 1.250,000.00"


class ScriptedOCR:
    """Fake OCR that returns one prepared result per call, in order.

    A result that is an exception is raised. Each call is recorded as
    ``(image shape, psm)``.
    """

    def __init__(self, monkeypatch: pytest.MonkeyPatch, results: list) -> None:
        self.results = list(results)
        self.calls: list[tuple[tuple[int, ...], int]] = []
        self.preprocessed = np.zeros((40, 80), dtype=np.uint8)
        monkeypatch.setattr(pipeline, "preprocess_image", lambda _image: self.preprocessed)
        monkeypatch.setattr(pipeline, "extract_text", self._ocr)

    def _ocr(self, image: np.ndarray, psm: int = 6) -> str:
        self.calls.append((image.shape, psm))
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def test_rescale_is_off_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    ocr = ScriptedOCR(monkeypatch, [PARTIAL_TEXT, PARTIAL_TEXT])

    process_invoice_image(PNG_BYTES, fallback_psm=4)

    assert [psm for _, psm in ocr.calls] == [6, 4]


def test_rescale_runs_last_with_the_fallback_mode_on_a_resized_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ocr = ScriptedOCR(monkeypatch, [PARTIAL_TEXT, PARTIAL_TEXT, COMPLETE_TEXT])

    fields = process_invoice_image(PNG_BYTES, fallback_psm=4, rescue_scale=0.75)["invoice_fields"]

    assert ocr.calls == [((40, 80), 6), ((40, 80), 4), ((30, 60), 4)]
    assert fields["total_amount"] == "117.00"
    assert ocr.preprocessed.shape == (40, 80)  # the preprocessed image was not changed


def test_rescale_does_not_run_when_the_fallback_filled_every_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ocr = ScriptedOCR(monkeypatch, [PARTIAL_TEXT, COMPLETE_TEXT])

    process_invoice_image(PNG_BYTES, fallback_psm=4, rescue_scale=0.75)

    assert len(ocr.calls) == 2


def test_rescale_does_not_run_when_the_primary_pass_is_complete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ocr = ScriptedOCR(monkeypatch, [COMPLETE_TEXT])

    process_invoice_image(PNG_BYTES, fallback_psm=4, rescue_scale=0.75)

    assert len(ocr.calls) == 1


def test_rescale_fills_only_fields_that_are_still_none(monkeypatch: pytest.MonkeyPatch) -> None:
    other = (
        "Other Vendor\nInvoice No: Z-9\nInvoice Date: 2026-10-17\n"
        "Subtotal: 999.00\nTax: 17.00\nTotal: 117.00"
    )
    fallback = "Acme Traders\nInvoice No: A-1\nInvoice Date: 2026-01-02"
    ScriptedOCR(monkeypatch, [PARTIAL_TEXT, fallback, other])

    fields = process_invoice_image(PNG_BYTES, fallback_psm=4, rescue_scale=0.75)["invoice_fields"]

    assert fields == {
        "invoice_number": "A-1",  # first pass
        "invoice_date": "2026-01-02",  # second pass, not the third pass's 2026-10-17
        "vendor_name": "Acme Traders",
        "subtotal": "999.00",  # only the third pass could fill these
        "tax_amount": "17.00",
        "total_amount": "117.00",
    }


def test_rescale_error_is_logged_and_the_previous_result_is_returned(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    ScriptedOCR(monkeypatch, [PARTIAL_TEXT, PARTIAL_TEXT, OCRError("boom")])

    with caplog.at_level(logging.WARNING, logger="app.pipeline"):
        result = process_invoice_image(PNG_BYTES, fallback_psm=4, rescue_scale=0.75)

    assert result["invoice_fields"]["invoice_number"] == "A-1"
    assert result["invoice_fields"]["total_amount"] is None
    assert "boom" in caplog.text


def test_rescale_needs_the_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    ocr = ScriptedOCR(monkeypatch, [PARTIAL_TEXT])

    with pytest.raises(ValueError):
        process_invoice_image(PNG_BYTES, rescue_scale=0.75)

    assert ocr.calls == []


@pytest.mark.parametrize(
    "bad_scale", [0, -0.5, 1, 1.0, 5, float("nan"), float("inf"), "0.75", True],
    ids=repr,
)
def test_invalid_rescale_is_rejected_before_any_ocr(
    monkeypatch: pytest.MonkeyPatch, bad_scale: object
) -> None:
    ocr = ScriptedOCR(monkeypatch, [PARTIAL_TEXT])

    with pytest.raises(ValueError):
        process_invoice_image(PNG_BYTES, fallback_psm=4, rescue_scale=bad_scale)  # type: ignore[arg-type]

    assert ocr.calls == []


def test_rescale_is_keyword_only() -> None:
    with pytest.raises(TypeError):
        process_invoice_image(PNG_BYTES, 4, 0.75)  # type: ignore[misc]


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