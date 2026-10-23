"""Experimental OCR configuration comparison for one image.

This is an experiment, not an accuracy benchmark. It runs the project's own
preprocessing and OCR functions on one image with a few settings and prints
the raw and cleaned text of each run, so the outputs can be compared by eye.
It makes no judgement about which configuration is better.

Usage (from the project root, with the virtual environment active):

    python scripts/evaluate_ocr.py path/to/invoice.png
    python scripts/evaluate_ocr.py path/to/invoice.png --experiment 2

Experiment 1 (blur and page segmentation):

* Preprocessing "current defaults": ``preprocess_image(image)``
  (grayscale, Gaussian blur, Otsu threshold, with the module's default options).
* Preprocessing "no blur": ``preprocess_image(image, kernel_size=1)``.
* Tesseract page segmentation modes (PSM) 4, 6 and 11 for each of the above.

Experiment 2 (scale and thresholding, 12 configurations, no blur throughout):

* Scale factors 1x, 2x and 3x, resized with ``cv2.resize`` and ``cv2.INTER_CUBIC``
  (1x keeps the original pixels).
* "Grayscale, no threshold": ``to_grayscale`` then ``reduce_noise(kernel_size=1)``.
  ``preprocess_image`` cannot be used here because it always thresholds.
* "Otsu threshold": ``preprocess_image(kernel_size=1, threshold_method="otsu")``.
* PSM 4 and 6 for each combination.

Nothing is written to disk and no production module is modified.
"""

import argparse
import inspect
import sys
from collections.abc import Callable
from pathlib import Path

# Make "import app" work when this file is run as a script from any directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

try:
    from app.image_processing import (
        ImagePreprocessingError,
        preprocess_image,
        reduce_noise,
        to_grayscale,
    )
    from app.ocr import OCREngineNotFoundError, OCRError, extract_text
    from app.text_cleaning import clean_ocr_text
except ImportError as exc:
    sys.exit(
        f"Could not import the project modules: {exc}\n"
        "Run this script from the project root with the virtual environment active."
    )

EXPERIMENT_1_PSM_MODES = (4, 6, 11)
EXPERIMENT_2_SCALE_FACTORS = (1, 2, 3)
EXPERIMENT_2_PSM_MODES = (4, 6)
SEPARATOR = "=" * 72


def load_image(path_text: str) -> np.ndarray:
    """Read an image from disk with OpenCV, or exit with a clear message."""
    path = Path(path_text).expanduser()
    if not path.exists():
        sys.exit(f"Error: the path does not exist: {path}")
    if not path.is_file():
        sys.exit(f"Error: the path is not a file: {path}")

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        sys.exit(
            f"Error: OpenCV could not read {path} as an image. "
            "It may be empty, corrupt, or not an image file."
        )
    return image


def require_parameter(function: object, parameter: str) -> None:
    """Exit with a clear message if ``function`` has no ``parameter`` argument."""
    if parameter not in inspect.signature(function).parameters:  # type: ignore[arg-type]
        sys.exit(
            f"Error: {function.__module__}.{function.__name__}() has no "  # type: ignore[attr-defined]
            f"'{parameter}' parameter, so this experiment cannot vary it. "
            "Check the function signature in the repository."
        )


def describe_size(image: np.ndarray) -> str:
    """Return the image size as 'width x height pixels'."""
    height, width = image.shape[:2]
    return f"{width} x {height} pixels"


def print_run(name: str, prepared: np.ndarray, raw_text: str, cleaned_text: str) -> None:
    """Print one configuration's name, image size, raw output and cleaned output."""
    print(SEPARATOR)
    print(f"Configuration: {name}")
    print(f"Image passed to OCR: {describe_size(prepared)}")
    print("-- Raw OCR output (repr, so whitespace is visible):")
    print(repr(raw_text))
    print("-- Cleaned OCR output:")
    print(cleaned_text if cleaned_text else "(empty)")


def prepare_image(
    label: str, preprocess: Callable[[np.ndarray], np.ndarray], image: np.ndarray
) -> np.ndarray:
    """Run a preprocessing function, or exit with a clear message if it fails."""
    try:
        return preprocess(image)
    except ImagePreprocessingError as exc:
        sys.exit(f"Error: preprocessing '{label}' failed: {exc}")


def run_ocr(name: str, prepared: np.ndarray, psm: int) -> None:
    """Run OCR on a prepared image and print the result.

    Exits if Tesseract is missing. Any other OCR error is printed for this
    configuration and the experiment continues.
    """
    try:
        raw_text = extract_text(prepared, psm=psm)
    except OCREngineNotFoundError as exc:
        sys.exit(f"Error: Tesseract is not available: {exc}")
    except OCRError as exc:
        print(SEPARATOR)
        print(f"Configuration: {name}")
        print(f"Image passed to OCR: {describe_size(prepared)}")
        print(f"OCR failed for this configuration: {exc}")
        return
    print_run(name, prepared, raw_text, clean_ocr_text(raw_text))


# --- Experiment 1: blur and page segmentation ---------------------------------


def run_experiment_1(image: np.ndarray) -> None:
    """Compare default and no-blur preprocessing at PSM 4, 6 and 11."""
    print(SEPARATOR)
    print("EXPERIMENT 1: blur and page segmentation mode")

    preprocessing_settings = (
        ("current defaults", {}),
        ("no blur (kernel_size=1)", {"kernel_size": 1}),
    )
    for preprocessing_name, options in preprocessing_settings:
        prepared = prepare_image(
            preprocessing_name,
            lambda img, options=options: preprocess_image(img, **options),
            image,
        )
        for psm in EXPERIMENT_1_PSM_MODES:
            name = f"preprocessing: {preprocessing_name} | psm {psm}"
            run_ocr(name, prepared, psm)


# --- Experiment 2: scale and thresholding -------------------------------------


def scale_image(image: np.ndarray, factor: int) -> np.ndarray:
    """Enlarge an image by ``factor`` with cubic interpolation (1 keeps it as is)."""
    if factor == 1:
        return image
    return cv2.resize(image, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)


def to_gray_without_threshold(image: np.ndarray) -> np.ndarray:
    """Grayscale with an unchanged (kernel_size=1) denoising step, no threshold."""
    return reduce_noise(to_grayscale(image), kernel_size=1)


def to_otsu_without_blur(image: np.ndarray) -> np.ndarray:
    """Grayscale, unchanged denoising (kernel_size=1), then Otsu threshold."""
    return preprocess_image(image, kernel_size=1, threshold_method="otsu")


def run_experiment_2(image: np.ndarray) -> None:
    """Compare scale factors and thresholding at PSM 4 and 6 (12 configurations)."""
    print(SEPARATOR)
    print("EXPERIMENT 2: scale factor and thresholding (no blur in any run)")

    thresholding_methods = (
        ("grayscale, no threshold", to_gray_without_threshold),
        ("Otsu threshold", to_otsu_without_blur),
    )
    for factor in EXPERIMENT_2_SCALE_FACTORS:
        scaled = scale_image(image, factor)
        for method_name, preprocess in thresholding_methods:
            prepared = prepare_image(f"{factor}x {method_name}", preprocess, scaled)
            for psm in EXPERIMENT_2_PSM_MODES:
                name = f"scale {factor}x | {method_name} | psm {psm}"
                run_ocr(name, prepared, psm)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare OCR settings on one image (an experiment, not a benchmark)."
    )
    parser.add_argument("image_path", help="Path to the invoice image to analyze.")
    parser.add_argument(
        "--experiment",
        choices=("1", "2", "all"),
        default="all",
        help="Which experiment to run (default: all).",
    )
    arguments = parser.parse_args()

    image = load_image(arguments.image_path)

    # Confirm the options these experiments vary really exist before using them.
    require_parameter(preprocess_image, "kernel_size")
    require_parameter(preprocess_image, "threshold_method")
    require_parameter(reduce_noise, "kernel_size")
    require_parameter(extract_text, "psm")

    print(f"Image: {arguments.image_path} ({describe_size(image)})")
    if arguments.experiment in ("1", "all"):
        run_experiment_1(image)
    if arguments.experiment in ("2", "all"):
        run_experiment_2(image)
    print(SEPARATOR)
    print(
        "Done. These outputs are for manual comparison only; this script does "
        "not score or rank the configurations."
    )


if __name__ == "__main__":
    main()