"""Unit tests for app.image_processing using synthetic NumPy images."""

import numpy as np
import pytest

from app.image_processing import (
    ImagePreprocessingError,
    InvalidImageError,
    InvalidParameterError,
    preprocess_image,
    reduce_noise,
    threshold_adaptive,
    threshold_otsu,
    to_grayscale,
    validate_image,
)

BACKGROUND = 200
INK = 50


def make_text_like_gray() -> np.ndarray:
    """100x100 light background with a dark 40x40 block."""
    image = np.full((100, 100), BACKGROUND, dtype=np.uint8)
    image[30:70, 30:70] = INK
    return image


def make_noisy(clean: np.ndarray, fraction: float = 0.05, seed: int = 42) -> np.ndarray:
    """Replace a fixed fraction of pixels with random 0 or 255 values."""
    rng = np.random.default_rng(seed)
    mask = rng.random(clean.shape) < fraction
    speckles = rng.choice([0, 255], size=clean.shape)
    return np.where(mask, speckles, clean).astype(np.uint8)


def make_uneven_lighting_gray() -> np.ndarray:
    """Horizontal brightness gradient with a dark horizontal bar."""
    gradient = np.tile(np.linspace(120, 230, 200), (100, 1))
    gradient[40:45, 20:180] -= 100
    return gradient.astype(np.uint8)


def assert_binary(image: np.ndarray) -> None:
    assert set(np.unique(image)).issubset({0, 255})


# --- Valid inputs ------------------------------------------------------------


def test_preprocess_grayscale_input_returns_binary_image_of_same_size() -> None:
    gray = make_text_like_gray()

    result = preprocess_image(gray)

    assert isinstance(result, np.ndarray)
    assert result.dtype == np.uint8
    assert result.shape == gray.shape
    assert_binary(result)


def test_preprocess_color_input_returns_2d_binary_image() -> None:
    color = np.dstack([make_text_like_gray()] * 3)

    result = preprocess_image(color)

    assert result.ndim == 2
    assert result.shape == color.shape[:2]
    assert result.dtype == np.uint8
    assert_binary(result)


def test_color_and_grayscale_versions_of_same_image_give_same_result() -> None:
    gray = make_text_like_gray()
    color = np.dstack([gray] * 3)

    assert np.array_equal(preprocess_image(color), preprocess_image(gray))


def test_preprocess_four_channel_input_is_accepted() -> None:
    gray = make_text_like_gray()
    bgra = np.dstack([gray, gray, gray, np.full_like(gray, 255)])

    result = preprocess_image(bgra)

    assert result.shape == gray.shape
    assert_binary(result)


def test_preprocess_single_channel_3d_input_is_accepted() -> None:
    gray = make_text_like_gray()

    result = preprocess_image(gray[:, :, np.newaxis])

    assert result.shape == gray.shape
    assert_binary(result)


def test_threshold_output_marks_dark_content_black_and_background_white() -> None:
    result = preprocess_image(make_text_like_gray())

    assert result[50, 50] == 0  # Inside the dark block
    assert result[5, 5] == 255  # Background


def test_preprocess_does_not_modify_input_array() -> None:
    original = make_text_like_gray()
    snapshot = original.copy()

    preprocess_image(original)

    assert np.array_equal(original, snapshot)


# --- Individual steps --------------------------------------------------------


def test_to_grayscale_returns_new_2d_array_for_grayscale_input() -> None:
    gray = make_text_like_gray()

    result = to_grayscale(gray)

    assert np.array_equal(result, gray)
    assert result is not gray


def test_to_grayscale_converts_equal_channels_to_same_values() -> None:
    gray = make_text_like_gray()

    result = to_grayscale(np.dstack([gray] * 3))

    assert result.shape == gray.shape
    assert np.allclose(result, gray, atol=1)


def test_to_grayscale_respects_color_order() -> None:
    # One pixel whose array values are (0, 0, 255).
    pixel = np.zeros((1, 1, 3), dtype=np.uint8)
    pixel[0, 0] = (0, 0, 255)

    as_bgr = to_grayscale(pixel, color_order="BGR")  # Red -> about 76
    as_rgb = to_grayscale(pixel, color_order="RGB")  # Blue -> about 29

    assert as_bgr[0, 0] == pytest.approx(76, abs=1)
    assert as_rgb[0, 0] == pytest.approx(29, abs=1)


@pytest.mark.parametrize("method", ["gaussian", "median"])
def test_reduce_noise_keeps_shape_and_dtype(method: str) -> None:
    gray = make_text_like_gray()

    result = reduce_noise(gray, method=method, kernel_size=3)

    assert result.shape == gray.shape
    assert result.dtype == np.uint8


def test_reduce_noise_with_kernel_size_one_returns_unchanged_copy() -> None:
    gray = make_text_like_gray()

    result = reduce_noise(gray, kernel_size=1)

    assert np.array_equal(result, gray)
    assert result is not gray


def test_threshold_otsu_separates_two_gray_levels() -> None:
    result = threshold_otsu(make_text_like_gray())

    assert_binary(result)
    assert np.all(result[30:70, 30:70] == 0)
    assert result[0, 0] == 255


def test_threshold_adaptive_handles_uneven_lighting() -> None:
    result = threshold_adaptive(make_uneven_lighting_gray(), block_size=31, c=10)

    assert_binary(result)
    assert result[42, 100] == 0  # Dark bar in the middle of the image
    assert result[10, 100] == 255  # Background, away from the bar
    assert result[80, 100] == 255


# --- Noise -------------------------------------------------------------------


def test_median_denoising_reduces_errors_caused_by_speckle_noise() -> None:
    clean = make_text_like_gray()
    noisy = make_noisy(clean, fraction=0.05)
    expected = threshold_otsu(clean)

    without_denoising = threshold_otsu(noisy)
    with_denoising = preprocess_image(
        noisy, denoise_method="median", kernel_size=3, threshold_method="otsu"
    )

    errors_without = int(np.count_nonzero(without_denoising != expected))
    errors_with = int(np.count_nonzero(with_denoising != expected))

    assert errors_without > 0  # The noise really does cause errors
    assert errors_with < errors_without / 4


# --- Configurable options ----------------------------------------------------


@pytest.mark.parametrize("denoise_method", ["gaussian", "median"])
@pytest.mark.parametrize("threshold_method", ["otsu", "adaptive"])
def test_all_option_combinations_produce_binary_output(
    denoise_method: str, threshold_method: str
) -> None:
    gray = make_text_like_gray()

    result = preprocess_image(
        gray,
        denoise_method=denoise_method,
        threshold_method=threshold_method,
        kernel_size=5,
        block_size=21,
        c=5,
    )

    assert result.shape == gray.shape
    assert_binary(result)


def test_preprocess_accepts_rgb_color_order() -> None:
    gray = make_text_like_gray()
    rgb = np.dstack([gray] * 3)

    result = preprocess_image(rgb, color_order="RGB")

    assert result.shape == gray.shape
    assert_binary(result)


# --- Invalid images ----------------------------------------------------------


@pytest.mark.parametrize(
    "bad_input",
    [None, "image.png", [[1, 2], [3, 4]], 42],
    ids=["none", "string", "list", "int"],
)
def test_non_array_input_raises_invalid_image_error(bad_input: object) -> None:
    with pytest.raises(InvalidImageError, match="NumPy array"):
        preprocess_image(bad_input)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "empty_image",
    [np.zeros((0, 10), dtype=np.uint8), np.zeros((10, 0, 3), dtype=np.uint8)],
    ids=["zero-rows", "zero-columns"],
)
def test_empty_array_raises_invalid_image_error(empty_image: np.ndarray) -> None:
    with pytest.raises(InvalidImageError, match="empty"):
        preprocess_image(empty_image)


def test_wrong_dtype_raises_invalid_image_error() -> None:
    float_image = np.full((10, 10), 0.5, dtype=np.float64)

    with pytest.raises(InvalidImageError, match="uint8"):
        preprocess_image(float_image)


@pytest.mark.parametrize(
    "bad_shape",
    [(10,), (2, 2, 2, 3), (10, 10, 2), (10, 10, 5)],
    ids=["1d", "4d", "two-channels", "five-channels"],
)
def test_unsupported_shape_raises_invalid_image_error(bad_shape: tuple[int, ...]) -> None:
    image = np.zeros(bad_shape, dtype=np.uint8)

    with pytest.raises(InvalidImageError):
        preprocess_image(image)


def test_color_image_passed_to_grayscale_only_step_raises_invalid_image_error() -> None:
    color = np.zeros((10, 10, 3), dtype=np.uint8)

    with pytest.raises(InvalidImageError, match="grayscale"):
        reduce_noise(color)
    with pytest.raises(InvalidImageError, match="grayscale"):
        threshold_otsu(color)


def test_validate_image_accepts_valid_arrays_and_returns_none() -> None:
    assert validate_image(np.zeros((5, 5), dtype=np.uint8)) is None
    assert validate_image(np.zeros((5, 5, 3), dtype=np.uint8)) is None


# --- Invalid options ---------------------------------------------------------


def test_unknown_denoise_method_raises_invalid_parameter_error() -> None:
    with pytest.raises(InvalidParameterError, match="method"):
        preprocess_image(make_text_like_gray(), denoise_method="blur")  # type: ignore[arg-type]


def test_unknown_threshold_method_raises_invalid_parameter_error() -> None:
    with pytest.raises(InvalidParameterError, match="threshold_method"):
        preprocess_image(make_text_like_gray(), threshold_method="magic")  # type: ignore[arg-type]


def test_unknown_color_order_raises_invalid_parameter_error() -> None:
    with pytest.raises(InvalidParameterError, match="color_order"):
        to_grayscale(np.zeros((5, 5, 3), dtype=np.uint8), color_order="GBR")  # type: ignore[arg-type]


@pytest.mark.parametrize("kernel_size", [0, -3, 2, 4])
def test_invalid_kernel_size_raises_invalid_parameter_error(kernel_size: int) -> None:
    with pytest.raises(InvalidParameterError, match="kernel_size"):
        reduce_noise(make_text_like_gray(), kernel_size=kernel_size)


@pytest.mark.parametrize("kernel_size", [3.0, "3", None, True])
def test_non_integer_kernel_size_raises_invalid_parameter_error(kernel_size: object) -> None:
    with pytest.raises(InvalidParameterError, match="kernel_size"):
        reduce_noise(make_text_like_gray(), kernel_size=kernel_size)  # type: ignore[arg-type]


@pytest.mark.parametrize("block_size", [1, 2, 30, -5])
def test_invalid_block_size_raises_invalid_parameter_error(block_size: int) -> None:
    with pytest.raises(InvalidParameterError, match="block_size"):
        threshold_adaptive(make_text_like_gray(), block_size=block_size)


def test_non_numeric_c_raises_invalid_parameter_error() -> None:
    with pytest.raises(InvalidParameterError, match="c must be a number"):
        threshold_adaptive(make_text_like_gray(), c="ten")  # type: ignore[arg-type]


def test_module_exceptions_share_a_common_base_class() -> None:
    assert issubclass(InvalidImageError, ImagePreprocessingError)
    assert issubclass(InvalidParameterError, ImagePreprocessingError)
    assert issubclass(ImagePreprocessingError, ValueError)