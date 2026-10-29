"""Document processing pipeline for the Intelligent Image Document Analyzer.

The public entry point is :func:`process_invoice_image`. It coordinates the
existing components in this order:

1. Decode the image bytes with OpenCV (:func:`decode_image`).
2. Preprocess the image (:func:`app.image_processing.preprocess_image`).
3. Run OCR (:func:`app.ocr.extract_text`).
4. Clean the recognized text (:func:`app.text_cleaning.clean_ocr_text`).
5. Extract invoice fields (:func:`app.extraction.extract_invoice_fields`).

Optionally, when ``fallback_psm`` is given and the first pass leaves any field
empty, the same preprocessed image is read again with that Tesseract page
segmentation mode, and only the empty fields are filled from the second pass.
Optionally, when ``rescue_scale`` is also given and a field is still empty, a
resized copy of the preprocessed image is read once more with the same mode and
again only the empty fields are filled. Tesseract sometimes misreads the
punctuation of an amount at one image scale and reads it correctly at another.

This module contains no preprocessing, OCR, cleaning or extraction logic of
its own, and it does not depend on FastAPI.
"""

import logging

import cv2
import numpy as np
from numpy.typing import NDArray

from app.extraction import extract_invoice_fields
from app.image_processing import preprocess_image
from app.ocr import OCRError, build_tesseract_config, extract_text
from app.text_cleaning import clean_ocr_text

logger = logging.getLogger(__name__)

_MAX_RESCUE_SCALE = 4.0


class InvalidImageBytesInputError(TypeError):
    """Raised when the image data argument is not a bytes-like object."""


class ImageDecodeError(ValueError):
    """Raised when image bytes are empty or cannot be decoded as an image."""


def decode_image(image_bytes: bytes | bytearray | memoryview) -> NDArray[np.uint8]:
    """Decode encoded image bytes into a 3-channel BGR image array.

    OpenCV decides the format from the bytes themselves, not from any filename.
    The result is always a ``uint8`` array of shape (H, W, 3) in OpenCV's BGR
    order. Any alpha channel is dropped, and grayscale images are expanded to
    three channels.

    Args:
        image_bytes: The contents of an image file, for example JPEG or PNG.

    Returns:
        A new ``uint8`` array of shape (H, W, 3).

    Raises:
        InvalidImageBytesInputError: If ``image_bytes`` is not ``bytes``,
            ``bytearray`` or ``memoryview``. It is a ``TypeError`` subclass.
        ImageDecodeError: If the data is empty or OpenCV cannot decode it. It
            is a ``ValueError`` subclass. If OpenCV itself raised an error, it
            is kept as ``__cause__``.
    """
    if not isinstance(image_bytes, (bytes, bytearray, memoryview)):
        raise InvalidImageBytesInputError(
            "image_bytes must be bytes, bytearray or memoryview, "
            f"got {type(image_bytes).__name__}."
        )

    buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    if buffer.size == 0:
        raise ImageDecodeError("Image data is empty.")

    try:
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    except cv2.error as exc:
        raise ImageDecodeError(f"OpenCV could not decode the image: {exc}") from exc

    if image is None:
        raise ImageDecodeError(
            "The data could not be decoded as an image. "
            "It may be corrupt or in an unsupported format."
        )
    return image


def _validate_rescue_scale(rescue_scale: float) -> None:
    """Raise ``ValueError`` unless ``rescue_scale`` is a usable resize factor."""
    if (
        isinstance(rescue_scale, bool)
        or not isinstance(rescue_scale, (int, float))
        or not 0 < rescue_scale <= _MAX_RESCUE_SCALE
        or rescue_scale == 1
    ):
        raise ValueError(
            f"rescue_scale must be a number above 0 and up to {_MAX_RESCUE_SCALE}, "
            f"other than 1, got {rescue_scale!r}."
        )


def _rescale(image: NDArray[np.uint8], scale: float) -> NDArray[np.uint8]:
    """Return a resized copy of ``image`` (area averaging when shrinking)."""
    height, width = image.shape[:2]
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    return cv2.resize(image, size, interpolation=interpolation)


def _fill_missing_fields(
    primary_fields: dict[str, str | None],
    preprocessed: NDArray[np.uint8],
    fallback_psm: int,
    pass_name: str = "Fallback OCR",
) -> dict[str, str | None]:
    """Fill the ``None`` fields of ``primary_fields`` from another OCR pass.

    A value that is already set is never replaced. If the extra OCR pass fails
    with an :class:`app.ocr.OCRError`, the error is logged and the primary
    fields are returned unchanged.
    """
    try:
        fallback_raw_text = extract_text(preprocessed, psm=fallback_psm)
    except OCRError as exc:
        logger.warning(
            "%s (PSM %s) failed; keeping the previous result: %s",
            pass_name,
            fallback_psm,
            exc,
            exc_info=exc,
        )
        return primary_fields

    fallback_fields = extract_invoice_fields(clean_ocr_text(fallback_raw_text))
    merged = {
        name: value if value is not None else fallback_fields[name]
        for name, value in primary_fields.items()
    }
    filled = [name for name, value in primary_fields.items() if value is None and merged[name]]
    logger.info(
        "%s (PSM %s) filled %d field(s): %s",
        pass_name,
        fallback_psm,
        len(filled),
        ", ".join(filled) or "none",
    )
    return merged


def process_invoice_image(
    image_bytes: bytes | bytearray | memoryview,
    *,
    fallback_psm: int | None = None,
    rescue_scale: float | None = None,
) -> dict[str, str | dict[str, str | None]]:
    """Run the full pipeline on an invoice image and return the results.

    Args:
        image_bytes: The contents of an invoice image file.
        fallback_psm: Optional Tesseract page segmentation mode for a second
            OCR pass. The default ``None`` disables it. If given, the second
            pass runs only when the first pass leaves at least one field
            ``None``. It reads the same preprocessed image, fills only the
            empty fields, and never replaces a value from the first pass.
            ``raw_ocr_text`` and ``cleaned_text`` always come from the first
            pass. A failing second pass is logged and ignored.
        rescue_scale: Optional resize factor (for example ``0.75``) for a third
            OCR pass. It needs ``fallback_psm`` and runs, with that mode, only
            if a field is still ``None`` after the second pass. It reads a
            resized copy of the preprocessed image and, like the second pass,
            only fills empty fields and never replaces a value. The default
            ``None`` disables it.

    Returns:
        A new dictionary with these keys:

        * ``raw_ocr_text`` - the text exactly as Tesseract returned it.
        * ``cleaned_text`` - the text after :func:`clean_ocr_text`.
        * ``invoice_fields`` - the dictionary from
          :func:`extract_invoice_fields`, with the keys ``invoice_number``,
          ``invoice_date``, ``vendor_name``, ``subtotal``, ``tax_amount`` and
          ``total_amount``. Fields that were not found are ``None``.

    Raises:
        app.ocr.OCRConfigError: If ``fallback_psm`` is not a supported mode. It
            is raised before any OCR runs.
        ValueError: If ``rescue_scale`` is not a usable number, or is given
            without ``fallback_psm``. It is raised before any OCR runs.
        InvalidImageBytesInputError: If ``image_bytes`` is not bytes-like.
        ImageDecodeError: If the bytes are empty or not a decodable image.
        app.image_processing.ImagePreprocessingError: If preprocessing fails
            (propagated unchanged).
        app.ocr.OCRError: If OCR fails, for example because Tesseract is
            missing (propagated unchanged).

    Errors from the later steps are not caught here, so their original type
    and message reach the caller.
    """
    if fallback_psm is not None:
        build_tesseract_config(fallback_psm)  # Reject an unsupported mode up front
    if rescue_scale is not None:
        _validate_rescue_scale(rescue_scale)
        if fallback_psm is None:
            raise ValueError("rescue_scale needs fallback_psm to be set.")

    image = decode_image(image_bytes)
    preprocessed = preprocess_image(image)
    raw_ocr_text = extract_text(preprocessed)
    cleaned_text = clean_ocr_text(raw_ocr_text)
    invoice_fields = extract_invoice_fields(cleaned_text)

    if fallback_psm is not None and any(value is None for value in invoice_fields.values()):
        invoice_fields = _fill_missing_fields(invoice_fields, preprocessed, fallback_psm)
        if rescue_scale is not None and any(value is None for value in invoice_fields.values()):
            invoice_fields = _fill_missing_fields(
                invoice_fields,
                _rescale(preprocessed, rescue_scale),
                fallback_psm,
                pass_name=f"Rescaled OCR (x{rescue_scale})",
            )

    return {
        "raw_ocr_text": raw_ocr_text,
        "cleaned_text": cleaned_text,
        "invoice_fields": invoice_fields,
    }