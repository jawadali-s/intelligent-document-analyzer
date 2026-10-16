"""Unit tests for app.ocr. Tesseract is mocked unless stated otherwise."""

import os
import shutil
from unittest.mock import patch

import cv2
import numpy as np
import pytesseract
import pytest

from app.ocr import (
    OCRConfigError,
    OCREngineError,
    OCREngineNotFoundError,
    OCRError,
    OCRInputError,
    OCRLanguageDataError,
    build_tesseract_config,
    extract_text,
)


def make_gray() -> np.ndarray:
    """Deterministic 40x60 grayscale image."""
    return np.arange(40 * 60, dtype=np.uint32).reshape(40, 60).astype(np.uint8)


def mock_tesseract(**kwargs):
    """Patch pytesseract.image_to_string as used by app.ocr."""
    return patch("app.ocr.pytesseract.image_to_string", **kwargs)


# --- Valid images ------------------------------------------------------------


def test_valid_grayscale_image_returns_mocked_text() -> None:
    with mock_tesseract(return_value="Invoice 123") as mocked:
        result = extract_text(make_gray())

    assert result == "Invoice 123"
    mocked.assert_called_once()


def test_valid_three_channel_image_is_accepted() -> None:
    color = np.dstack([make_gray()] * 3)

    with mock_tesseract(return_value="text") as mocked:
        result = extract_text(color)

    assert result == "text"
    assert mocked.call_args.args[0].shape == color.shape


def test_valid_four_channel_image_is_accepted() -> None:
    bgra = np.dstack([make_gray()] * 4)

    with mock_tesseract(return_value="text") as mocked:
        result = extract_text(bgra)

    assert result == "text"
    assert mocked.call_args.args[0].shape == bgra.shape


def test_single_channel_3d_image_is_passed_as_2d() -> None:
    image = make_gray()[:, :, np.newaxis]

    with mock_tesseract(return_value="text") as mocked:
        extract_text(image)

    assert mocked.call_args.args[0].shape == (40, 60)


# --- Invalid images ----------------------------------------------------------


@pytest.mark.parametrize(
    "bad_input", [None, "image.png", [[1, 2], [3, 4]]], ids=["none", "string", "list"]
)
def test_non_array_input_raises_ocr_input_error(bad_input: object) -> None:
    with mock_tesseract() as mocked:
        with pytest.raises(OCRInputError):
            extract_text(bad_input)  # type: ignore[arg-type]

    mocked.assert_not_called()


@pytest.mark.parametrize(
    "empty_image",
    [np.zeros((0, 10), dtype=np.uint8), np.zeros((10, 0, 3), dtype=np.uint8)],
    ids=["zero-rows", "zero-columns"],
)
def test_empty_array_raises_ocr_input_error(empty_image: np.ndarray) -> None:
    with mock_tesseract() as mocked:
        with pytest.raises(OCRInputError, match="empty"):
            extract_text(empty_image)

    mocked.assert_not_called()


@pytest.mark.parametrize(
    "bad_shape",
    [(10,), (2, 2, 2, 3), (10, 10, 2), (10, 10, 5)],
    ids=["1d", "4d", "two-channels", "five-channels"],
)
def test_invalid_dimensions_or_channels_raise_ocr_input_error(
    bad_shape: tuple[int, ...],
) -> None:
    with mock_tesseract() as mocked:
        with pytest.raises(OCRInputError):
            extract_text(np.zeros(bad_shape, dtype=np.uint8))

    mocked.assert_not_called()


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.uint16])
def test_unsupported_dtype_raises_ocr_input_error(dtype: type) -> None:
    with mock_tesseract() as mocked:
        with pytest.raises(OCRInputError, match="uint8"):
            extract_text(np.zeros((10, 10), dtype=dtype))

    mocked.assert_not_called()


def test_input_errors_are_distinct_from_engine_errors() -> None:
    with pytest.raises(OCRInputError) as exc_info:
        extract_text(None)  # type: ignore[arg-type]

    assert not isinstance(exc_info.value, OCREngineError)
    assert isinstance(exc_info.value, OCRError)
    assert isinstance(exc_info.value, ValueError)


# --- Forwarding configuration -----------------------------------------------


def test_default_language_and_config_are_forwarded() -> None:
    with mock_tesseract(return_value="") as mocked:
        extract_text(make_gray())

    assert mocked.call_args.kwargs == {"lang": "eng", "config": "--psm 6"}


def test_custom_language_psm_and_extra_config_are_forwarded() -> None:
    with mock_tesseract(return_value="") as mocked:
        extract_text(
            make_gray(),
            language="eng+fra",
            psm=4,
            extra_config="-c preserve_interword_spaces=1",
        )

    assert mocked.call_args.kwargs == {
        "lang": "eng+fra",
        "config": "--psm 4 -c preserve_interword_spaces=1",
    }


def test_image_array_is_forwarded_unchanged() -> None:
    image = make_gray()

    with mock_tesseract(return_value="") as mocked:
        extract_text(image)

    assert np.array_equal(mocked.call_args.args[0], image)


def test_build_tesseract_config_without_extra_config() -> None:
    assert build_tesseract_config() == "--psm 6"
    assert build_tesseract_config(psm=11) == "--psm 11"


def test_build_tesseract_config_ignores_blank_extra_config() -> None:
    assert build_tesseract_config(psm=6, extra_config="   ") == "--psm 6"


# --- Output handling ---------------------------------------------------------


def test_text_is_returned_without_cleaning() -> None:
    raw = "  Invoice  No: 42 \n\n\tTotal:   $10.00  \n\f"

    with mock_tesseract(return_value=raw):
        assert extract_text(make_gray()) == raw


def test_empty_ocr_output_returns_empty_string() -> None:
    with mock_tesseract(return_value=""):
        assert extract_text(make_gray()) == ""


def test_non_string_engine_output_raises_engine_error() -> None:
    with mock_tesseract(return_value=b"bytes"):
        with pytest.raises(OCREngineError, match="bytes"):
            extract_text(make_gray())


# --- Original array ----------------------------------------------------------


@pytest.mark.parametrize(
    "image",
    [make_gray(), np.dstack([make_gray()] * 3), make_gray()[:, :, np.newaxis]],
    ids=["gray", "bgr", "single-channel-3d"],
)
def test_original_image_is_not_modified(image: np.ndarray) -> None:
    snapshot = image.copy()

    with mock_tesseract(return_value="text"):
        extract_text(image)

    assert image.dtype == snapshot.dtype
    assert np.array_equal(image, snapshot)


# --- Invalid configuration ---------------------------------------------------


@pytest.mark.parametrize(
    "language", ["", "en g", "eng;rm", "eng+", "+eng", None, 5]
)
def test_invalid_language_raises_config_error(language: object) -> None:
    with mock_tesseract() as mocked:
        with pytest.raises(OCRConfigError, match="language"):
            extract_text(make_gray(), language=language)  # type: ignore[arg-type]

    mocked.assert_not_called()


@pytest.mark.parametrize("psm", [0, 2, 14, -1, "6", 6.0, True, None])
def test_invalid_psm_raises_config_error(psm: object) -> None:
    with mock_tesseract() as mocked:
        with pytest.raises(OCRConfigError, match="psm"):
            extract_text(make_gray(), psm=psm)  # type: ignore[arg-type]

    mocked.assert_not_called()


@pytest.mark.parametrize(
    "extra_config", ["--psm 7", "-l fra", "'unclosed", 5, ["-c", "x=1"]]
)
def test_invalid_extra_config_raises_config_error(extra_config: object) -> None:
    with mock_tesseract() as mocked:
        with pytest.raises(OCRConfigError):
            extract_text(make_gray(), extra_config=extra_config)  # type: ignore[arg-type]

    mocked.assert_not_called()


def test_config_errors_are_value_errors() -> None:
    with pytest.raises(ValueError):
        extract_text(make_gray(), psm=99)


# --- Engine failures ---------------------------------------------------------


def test_missing_tesseract_executable_raises_not_found_error() -> None:
    original = pytesseract.TesseractNotFoundError()

    with mock_tesseract(side_effect=original):
        with pytest.raises(OCREngineNotFoundError, match="apt install") as exc_info:
            extract_text(make_gray())

    assert exc_info.value.__cause__ is original


def test_missing_language_data_raises_language_data_error() -> None:
    original = pytesseract.TesseractError(
        1, "Error opening data file xyz.traineddata. Failed loading language 'xyz'"
    )

    with mock_tesseract(side_effect=original):
        with pytest.raises(OCRLanguageDataError, match="xyz") as exc_info:
            extract_text(make_gray(), language="xyz")

    assert exc_info.value.__cause__ is original


def test_other_tesseract_failure_raises_engine_error() -> None:
    original = pytesseract.TesseractError(1, "Image too small to scale")

    with mock_tesseract(side_effect=original):
        with pytest.raises(OCREngineError, match="Image too small") as exc_info:
            extract_text(make_gray())

    assert not isinstance(exc_info.value, OCRLanguageDataError)
    assert exc_info.value.__cause__ is original


def test_engine_errors_are_distinct_from_input_errors() -> None:
    with mock_tesseract(side_effect=pytesseract.TesseractNotFoundError()):
        with pytest.raises(OCREngineError) as exc_info:
            extract_text(make_gray())

    assert not isinstance(exc_info.value, OCRInputError)
    assert isinstance(exc_info.value, OCRError)


def test_unexpected_exceptions_are_not_swallowed() -> None:
    with mock_tesseract(side_effect=KeyError("programming error")):
        with pytest.raises(KeyError):
            extract_text(make_gray())


# --- Optional real-engine test ----------------------------------------------


@pytest.mark.skipif(
    os.environ.get("RUN_OCR_INTEGRATION") != "1" or shutil.which("tesseract") is None,
    reason="Set RUN_OCR_INTEGRATION=1 and install Tesseract to run this test.",
)
def test_real_tesseract_reads_simple_rendered_text() -> None:
    canvas = np.full((120, 480), 255, dtype=np.uint8)
    cv2.putText(canvas, "INVOICE 2026", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, 0, 4)

    result = extract_text(canvas, psm=7)

    assert "INVOICE" in result.upper()