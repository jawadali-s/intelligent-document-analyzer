"""OCR for the Intelligent Image Document Analyzer.

The public entry point is :func:`extract_text`, which runs Tesseract on an
image array and returns the recognized text exactly as Tesseract produced it.
No cleaning or normalization happens here.

Errors are split so callers can tell them apart:

* :class:`OCRInputError` - the image argument is invalid.
* :class:`OCRConfigError` - language, PSM or extra config is invalid.
* :class:`OCREngineError` - Tesseract is missing or failed.
"""

import re
import shlex

import numpy as np
import pytesseract
from numpy.typing import NDArray

from app.image_processing import InvalidImageError, validate_image

DEFAULT_LANGUAGE = "eng"
DEFAULT_PSM = 6  # Assume a single uniform block of text

# 0 (orientation detection only) and 2 (not implemented) do not return text.
SUPPORTED_PSM_VALUES = frozenset({1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13})

_LANGUAGE_PATTERN = re.compile(r"[A-Za-z0-9_]+(\+[A-Za-z0-9_]+)*")
_RESERVED_FLAGS = ("-l", "--psm")
_MISSING_LANGUAGE_MARKERS = (
    "failed loading language",
    "error opening data file",
    "couldn't load any languages",
)


class OCRError(Exception):
    """Base class for all errors raised by this module."""


class OCRInputError(OCRError, ValueError):
    """Raised when the image argument is not a valid image array."""


class OCRConfigError(OCRError, ValueError):
    """Raised when the language, PSM or extra configuration is invalid."""


class OCREngineError(OCRError, RuntimeError):
    """Raised when Tesseract cannot run or fails while processing an image."""


class OCREngineNotFoundError(OCREngineError):
    """Raised when the Tesseract executable is not installed or not on PATH."""


class OCRLanguageDataError(OCREngineError):
    """Raised when Tesseract cannot load the requested language data."""


def validate_language(language: object) -> str:
    """Return ``language`` if it is a valid Tesseract language string.

    Valid values look like ``"eng"`` or ``"eng+fra"``.

    Raises:
        OCRConfigError: If ``language`` is not a string in that form.
    """
    if not isinstance(language, str) or not _LANGUAGE_PATTERN.fullmatch(language):
        raise OCRConfigError(
            "language must be a string such as 'eng' or 'eng+fra' "
            f"(letters, digits and underscores, joined by '+'), got {language!r}."
        )
    return language


def build_tesseract_config(
    psm: int = DEFAULT_PSM, extra_config: str | None = None
) -> str:
    """Build the Tesseract configuration string.

    Args:
        psm: Page segmentation mode, one of ``SUPPORTED_PSM_VALUES``.
        extra_config: Optional extra Tesseract options, for example
            ``"-c preserve_interword_spaces=1"``. It must not contain
            ``--psm`` or ``-l``; use the dedicated parameters instead.

    Returns:
        A string such as ``"--psm 6"`` or ``"--psm 6 -c key=value"``.

    Raises:
        OCRConfigError: If ``psm`` or ``extra_config`` is invalid.
    """
    if isinstance(psm, bool) or not isinstance(psm, int):
        raise OCRConfigError(f"psm must be an integer, got {psm!r}.")
    if psm not in SUPPORTED_PSM_VALUES:
        raise OCRConfigError(
            f"psm must be one of {sorted(SUPPORTED_PSM_VALUES)}, got {psm}."
        )

    config = f"--psm {psm}"
    if extra_config is None:
        return config

    if not isinstance(extra_config, str):
        raise OCRConfigError(
            f"extra_config must be a string or None, got {extra_config!r}."
        )
    try:
        tokens = shlex.split(extra_config)
    except ValueError as exc:
        raise OCRConfigError(
            f"extra_config could not be parsed: {extra_config!r}."
        ) from exc
    for flag in _RESERVED_FLAGS:
        if flag in tokens:
            raise OCRConfigError(
                f"extra_config must not contain {flag!r}; "
                "use the language and psm parameters instead."
            )

    extra = extra_config.strip()
    return f"{config} {extra}" if extra else config


def extract_text(
    image: NDArray[np.uint8],
    language: str = DEFAULT_LANGUAGE,
    psm: int = DEFAULT_PSM,
    extra_config: str | None = None,
) -> str:
    """Run Tesseract OCR on an image and return the recognized text.

    The text is returned exactly as Tesseract produced it, including line
    breaks and spacing. The input array is never modified.

    Args:
        image: ``uint8`` array of shape (H, W), (H, W, 1), (H, W, 3) or
            (H, W, 4).
        language: Tesseract language code, for example ``"eng"`` or
            ``"eng+fra"``. The language data must be installed.
        psm: Tesseract page segmentation mode (see ``SUPPORTED_PSM_VALUES``).
            The default, 6, assumes a single uniform block of text.
        extra_config: Optional extra Tesseract options, appended after
            ``--psm``.

    Returns:
        The recognized text. It may be an empty string if nothing was found.

    Raises:
        OCRInputError: If ``image`` is not a valid image array.
        OCRConfigError: If ``language``, ``psm`` or ``extra_config`` is invalid.
        OCREngineNotFoundError: If the Tesseract executable cannot be found.
        OCRLanguageDataError: If the language data cannot be loaded.
        OCREngineError: If Tesseract fails for any other reason.
    """
    validate_language(language)
    config = build_tesseract_config(psm, extra_config)

    try:
        validate_image(image)
    except InvalidImageError as exc:
        raise OCRInputError(str(exc)) from exc

    # Pillow cannot handle (H, W, 1) arrays, so pass a 2-D view instead.
    ocr_image = image[:, :, 0] if image.ndim == 3 and image.shape[2] == 1 else image

    try:
        text = pytesseract.image_to_string(ocr_image, lang=language, config=config)
    except pytesseract.TesseractNotFoundError as exc:
        raise OCREngineNotFoundError(
            "The Tesseract executable was not found. Install it (on Ubuntu: "
            "'sudo apt install tesseract-ocr') and make sure 'tesseract' is "
            "on your PATH."
        ) from exc
    except pytesseract.TesseractError as exc:
        message = str(exc)
        if any(marker in message.lower() for marker in _MISSING_LANGUAGE_MARKERS):
            raise OCRLanguageDataError(
                f"Tesseract could not load language data for {language!r}. "
                "Install the matching package (on Ubuntu: "
                f"'sudo apt install tesseract-ocr-<code>'). Details: {message}"
            ) from exc
        raise OCREngineError(f"Tesseract failed: {message}") from exc

    if not isinstance(text, str):
        raise OCREngineError(
            f"Tesseract returned {type(text).__name__} instead of text."
        )
    return text