"""Document processing pipeline for the Intelligent Image Document Analyzer.

The public entry point is :func:`process_invoice_image`. It coordinates the
existing components in this order:

1. Decode the image bytes with OpenCV (:func:`decode_image`).
2. Preprocess the image (:func:`app.image_processing.preprocess_image`).
3. Run OCR (:func:`app.ocr.extract_text`).
4. Clean the recognized text (:func:`app.text_cleaning.clean_ocr_text`).
5. Extract invoice fields (:func:`app.extraction.extract_invoice_fields`).

This module contains no preprocessing, OCR, cleaning or extraction logic of
its own, and it does not depend on FastAPI.
"""

import cv2
import numpy as np
from numpy.typing import NDArray

from app.extraction import extract_invoice_fields
from app.image_processing import preprocess_image
from app.ocr import extract_text
from app.text_cleaning import clean_ocr_text


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


def process_invoice_image(
    image_bytes: bytes | bytearray | memoryview,
) -> dict[str, str | dict[str, str | None]]:
    """Run the full pipeline on an invoice image and return the results.

    Args:
        image_bytes: The contents of an invoice image file.

    Returns:
        A new dictionary with these keys:

        * ``raw_ocr_text`` - the text exactly as Tesseract returned it.
        * ``cleaned_text`` - the text after :func:`clean_ocr_text`.
        * ``invoice_fields`` - the dictionary from
          :func:`extract_invoice_fields`, with the keys ``invoice_number``,
          ``invoice_date``, ``vendor_name``, ``subtotal``, ``tax_amount`` and
          ``total_amount``. Fields that were not found are ``None``.

    Raises:
        InvalidImageBytesInputError: If ``image_bytes`` is not bytes-like.
        ImageDecodeError: If the bytes are empty or not a decodable image.
        app.image_processing.ImagePreprocessingError: If preprocessing fails
            (propagated unchanged).
        app.ocr.OCRError: If OCR fails, for example because Tesseract is
            missing (propagated unchanged).

    Errors from the later steps are not caught here, so their original type
    and message reach the caller.
    """
    image = decode_image(image_bytes)
    preprocessed = preprocess_image(image)
    raw_ocr_text = extract_text(preprocessed)
    cleaned_text = clean_ocr_text(raw_ocr_text)
    invoice_fields = extract_invoice_fields(cleaned_text)

    return {
        "raw_ocr_text": raw_ocr_text,
        "cleaned_text": cleaned_text,
        "invoice_fields": invoice_fields,
    }