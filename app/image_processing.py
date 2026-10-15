"""Image preprocessing for the Intelligent Image Document Analyzer.

The public entry point is :func:`preprocess_image`, which converts an image to
grayscale, reduces noise, and applies thresholding to produce a black-and-white
image. Each step is also available as its own function so it can be tested or
reused separately.

Images are NumPy arrays of dtype ``uint8``. Colour images are assumed to be in
OpenCV's BGR channel order unless ``color_order="RGB"`` is given (for example
when the array came from ``np.array(PIL_image)``).
"""

from typing import Literal

import cv2
import numpy as np
from numpy.typing import NDArray

GrayImage = NDArray[np.uint8]

SUPPORTED_COLOR_ORDERS = ("BGR", "RGB")
SUPPORTED_DENOISE_METHODS = ("gaussian", "median")
SUPPORTED_THRESHOLD_METHODS = ("otsu", "adaptive")

ColorOrder = Literal["BGR", "RGB"]
DenoiseMethod = Literal["gaussian", "median"]
ThresholdMethod = Literal["otsu", "adaptive"]

DEFAULT_KERNEL_SIZE = 3
DEFAULT_BLOCK_SIZE = 31
DEFAULT_ADAPTIVE_C = 10.0


class ImagePreprocessingError(ValueError):
    """Base class for all errors raised by this module."""


class InvalidImageError(ImagePreprocessingError):
    """Raised when the input is not a valid image array."""


class InvalidParameterError(ImagePreprocessingError):
    """Raised when a processing option has an unsupported or invalid value."""


# --- Validation helpers ------------------------------------------------------


def validate_image(image: object) -> None:
    """Check that ``image`` is a non-empty ``uint8`` NumPy image array.

    Accepted shapes are ``(H, W)`` for grayscale and ``(H, W, C)`` with
    ``C`` equal to 1, 3 or 4.

    Raises:
        InvalidImageError: If the input is not a NumPy array, does not have
            dtype ``uint8``, has an unsupported shape, or is empty.
    """
    if not isinstance(image, np.ndarray):
        raise InvalidImageError(
            f"Image must be a NumPy array, got {type(image).__name__}."
        )
    if image.dtype != np.uint8:
        raise InvalidImageError(
            f"Image must have dtype uint8, got {image.dtype}."
        )
    if image.ndim not in (2, 3):
        raise InvalidImageError(
            "Image must be 2-D (grayscale) or 3-D (height x width x channels), "
            f"got {image.ndim} dimension(s) with shape {image.shape}."
        )
    if image.size == 0:
        raise InvalidImageError(f"Image is empty (shape {image.shape}).")
    if image.ndim == 3 and image.shape[2] not in (1, 3, 4):
        raise InvalidImageError(
            f"Image must have 1, 3 or 4 channels, got {image.shape[2]}."
        )


def _require_grayscale(image: object) -> None:
    """Check that ``image`` is a valid 2-D grayscale image."""
    validate_image(image)
    if image.ndim != 2:  # type: ignore[union-attr]
        raise InvalidImageError(
            "Expected a 2-D grayscale image, got shape "
            f"{image.shape}. Use to_grayscale() first."  # type: ignore[union-attr]
        )


def _validate_choice(value: object, allowed: tuple[str, ...], name: str) -> None:
    """Check that ``value`` is one of the allowed option strings."""
    if value not in allowed:
        raise InvalidParameterError(
            f"{name} must be one of {allowed}, got {value!r}."
        )


def _validate_int(value: object, name: str, minimum: int, odd: bool = False) -> None:
    """Check that ``value`` is an integer >= ``minimum`` (and odd if required)."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise InvalidParameterError(f"{name} must be an integer, got {value!r}.")
    if value < minimum:
        raise InvalidParameterError(
            f"{name} must be at least {minimum}, got {value}."
        )
    if odd and value % 2 == 0:
        raise InvalidParameterError(f"{name} must be odd, got {value}.")


# --- Processing steps --------------------------------------------------------


def to_grayscale(image: NDArray[np.uint8], color_order: ColorOrder = "BGR") -> GrayImage:
    """Convert an image to a single-channel grayscale image.

    Args:
        image: ``uint8`` array of shape (H, W), (H, W, 1), (H, W, 3) or (H, W, 4).
            Four-channel images are treated as having an alpha channel, which
            is ignored.
        color_order: Channel order of 3- and 4-channel images, ``"BGR"``
            (OpenCV default) or ``"RGB"`` (Pillow default).

    Returns:
        A new 2-D ``uint8`` array of shape (H, W).

    Raises:
        InvalidImageError: If ``image`` is not a valid image array.
        InvalidParameterError: If ``color_order`` is not supported.
    """
    validate_image(image)
    _validate_choice(color_order, SUPPORTED_COLOR_ORDERS, "color_order")

    if image.ndim == 2:
        return image.copy()

    channels = image.shape[2]
    if channels == 1:
        return image[:, :, 0].copy()

    is_bgr = color_order == "BGR"
    if channels == 3:
        code = cv2.COLOR_BGR2GRAY if is_bgr else cv2.COLOR_RGB2GRAY
    else:
        code = cv2.COLOR_BGRA2GRAY if is_bgr else cv2.COLOR_RGBA2GRAY
    return cv2.cvtColor(image, code)


def reduce_noise(
    gray: GrayImage,
    method: DenoiseMethod = "gaussian",
    kernel_size: int = DEFAULT_KERNEL_SIZE,
) -> GrayImage:
    """Reduce noise in a grayscale image.

    Args:
        gray: 2-D ``uint8`` grayscale image.
        method: ``"gaussian"`` smooths the whole image with a Gaussian blur.
            ``"median"`` replaces each pixel with the median of its
            neighbourhood and removes isolated speckles (salt-and-pepper
            noise) while keeping edges sharper.
        kernel_size: Odd integer >= 1 giving the neighbourhood width in
            pixels. A value of 1 returns an unchanged copy.

    Returns:
        A new 2-D ``uint8`` array with the same shape as ``gray``.

    Raises:
        InvalidImageError: If ``gray`` is not a valid 2-D grayscale image.
        InvalidParameterError: If ``method`` or ``kernel_size`` is invalid.
    """
    _require_grayscale(gray)
    _validate_choice(method, SUPPORTED_DENOISE_METHODS, "method")
    _validate_int(kernel_size, "kernel_size", minimum=1, odd=True)

    kernel_size = int(kernel_size)
    if kernel_size == 1:
        return gray.copy()
    if method == "gaussian":
        return cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
    return cv2.medianBlur(gray, kernel_size)


def threshold_otsu(gray: GrayImage) -> GrayImage:
    """Binarize a grayscale image with Otsu's global threshold.

    Otsu's method picks one threshold for the whole image from its histogram.
    It works best when lighting is even.

    Args:
        gray: 2-D ``uint8`` grayscale image.

    Returns:
        A new 2-D ``uint8`` array containing only 0 (dark) and 255 (light).

    Raises:
        InvalidImageError: If ``gray`` is not a valid 2-D grayscale image.
    """
    _require_grayscale(gray)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def threshold_adaptive(
    gray: GrayImage,
    block_size: int = DEFAULT_BLOCK_SIZE,
    c: float = DEFAULT_ADAPTIVE_C,
) -> GrayImage:
    """Binarize a grayscale image with a local (adaptive) Gaussian threshold.

    Each pixel is compared with a Gaussian-weighted mean of its neighbourhood
    minus ``c``, so it copes better with uneven lighting than Otsu's method.

    Args:
        gray: 2-D ``uint8`` grayscale image.
        block_size: Odd integer >= 3 giving the neighbourhood width in pixels.
        c: Constant subtracted from the local mean. Larger values make more
            pixels white.

    Returns:
        A new 2-D ``uint8`` array containing only 0 (dark) and 255 (light).

    Raises:
        InvalidImageError: If ``gray`` is not a valid 2-D grayscale image.
        InvalidParameterError: If ``block_size`` or ``c`` is invalid.
    """
    _require_grayscale(gray)
    _validate_int(block_size, "block_size", minimum=3, odd=True)
    if isinstance(c, bool) or not isinstance(c, (int, float, np.integer, np.floating)):
        raise InvalidParameterError(f"c must be a number, got {c!r}.")

    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        int(block_size),
        float(c),
    )


# --- Public pipeline ---------------------------------------------------------


def preprocess_image(
    image: NDArray[np.uint8],
    *,
    color_order: ColorOrder = "BGR",
    denoise_method: DenoiseMethod = "gaussian",
    kernel_size: int = DEFAULT_KERNEL_SIZE,
    threshold_method: ThresholdMethod = "otsu",
    block_size: int = DEFAULT_BLOCK_SIZE,
    c: float = DEFAULT_ADAPTIVE_C,
) -> GrayImage:
    """Convert an image to grayscale, reduce noise, and binarize it.

    Args:
        image: ``uint8`` array of shape (H, W), (H, W, 1), (H, W, 3) or
            (H, W, 4).
        color_order: Channel order of colour input, ``"BGR"`` or ``"RGB"``.
        denoise_method: ``"gaussian"`` or ``"median"`` (see :func:`reduce_noise`).
        kernel_size: Odd integer >= 1 used by the denoising step.
        threshold_method: ``"otsu"`` for a global threshold or ``"adaptive"``
            for a local threshold.
        block_size: Odd integer >= 3. Used only when ``threshold_method`` is
            ``"adaptive"``.
        c: Constant subtracted from the local mean. Used only when
            ``threshold_method`` is ``"adaptive"``.

    Returns:
        A new 2-D ``uint8`` array of shape (H, W) containing only 0 and 255.
        The input array is never modified.

    Raises:
        InvalidImageError: If ``image`` is not a valid image array.
        InvalidParameterError: If any option has an unsupported value.
    """
    _validate_choice(threshold_method, SUPPORTED_THRESHOLD_METHODS, "threshold_method")

    gray = to_grayscale(image, color_order=color_order)
    denoised = reduce_noise(gray, method=denoise_method, kernel_size=kernel_size)

    if threshold_method == "otsu":
        return threshold_otsu(denoised)
    return threshold_adaptive(denoised, block_size=block_size, c=c)