"""Ten-invoice regression tests (no Tesseract needed).

Each case in ``invoice_regression_cases.CASES`` holds the raw PSM 6 and PSM 4 OCR
text of a real test invoice and the fields the application must return. The
pipeline is run with OCR stubbed to return that saved text, so the tests check
the extraction and the PSM 6 / PSM 4 merge exactly as the API uses them.

Real Tesseract on the invoice images is not run here; see the optional test in
``test_ocr.py`` (``RUN_OCR_INTEGRATION=1``).
"""

import logging

import numpy as np
import pytest

from app import pipeline
from app.extraction import FIELD_NAMES, extract_invoice_fields
from app.ocr import OCREngineError
from app.text_cleaning import clean_ocr_text
from invoice_regression_cases import (
    ALPINE_PSM4,
    CASES,
    CRESCENT_PSM6,
    HARBORVIEW_PSM6,
)

CASE_IDS = [case.name for case in CASES]
BLANK_IMAGE = np.zeros((40, 40, 3), dtype=np.uint8)


def primary_fields(case) -> dict[str, str | None]:
    return extract_invoice_fields(clean_ocr_text(case.psm6))


def fallback_fields(case) -> dict[str, str | None]:
    return extract_invoice_fields(clean_ocr_text(case.psm4))


class FakeOCR:
    """Stands in for Tesseract: returns saved text for each page segmentation mode."""

    def __init__(self, monkeypatch, texts: dict[int, str | Exception]) -> None:
        self.texts = texts
        self.calls: list[int] = []
        monkeypatch.setattr(pipeline, "decode_image", lambda _data: BLANK_IMAGE)
        monkeypatch.setattr(pipeline, "preprocess_image", lambda image: image)
        monkeypatch.setattr(pipeline, "extract_text", self.extract_text)

    def extract_text(self, image, psm: int = 6, **_kwargs) -> str:
        # The third pass reads a resized copy, so it is told apart by the image size.
        rescaled = image.shape != BLANK_IMAGE.shape
        self.calls.append(-psm if rescaled else psm)
        result = self.texts[-psm if rescaled else psm]
        if isinstance(result, Exception):
            raise result
        return result


def run_pipeline(monkeypatch, case, *, fallback_psm: int | None = 4, rescue_scale=None):
    ocr = FakeOCR(monkeypatch, {6: case.psm6, 4: case.psm4, -4: case.psm4_rescaled})
    result = pipeline.process_invoice_image(
        b"image", fallback_psm=fallback_psm, rescue_scale=rescue_scale
    )
    return result["invoice_fields"], ocr


# --- All seven invoices -------------------------------------------------------


def test_there_are_ten_cases_with_all_six_fields() -> None:
    assert len(CASES) == 10
    for case in CASES:
        assert list(case.expected) == list(FIELD_NAMES)


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_pipeline_with_fallback_and_rescale_returns_the_expected_fields(monkeypatch, case) -> None:
    fields, _ = run_pipeline(monkeypatch, case, rescue_scale=0.75)

    assert fields == case.expected


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_pipeline_with_fallback_only_is_correct_except_what_needs_the_rescale(
    monkeypatch, case
) -> None:
    fields, _ = run_pipeline(monkeypatch, case)

    for name, value in fields.items():
        if case.needs_rescale and value is None:
            assert case.expected[name] is not None
        else:
            assert value == case.expected[name], name


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_primary_pass_never_returns_a_wrong_value(case) -> None:
    primary = primary_fields(case)

    for name, value in primary.items():
        assert value is None or value == case.expected[name], name


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_fallback_pass_never_returns_a_wrong_value(case) -> None:
    fallback = fallback_fields(case)

    for name, value in fallback.items():
        assert value is None or value == case.expected[name], name


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_fallback_runs_only_when_the_primary_pass_misses_a_field(monkeypatch, case) -> None:
    _, ocr = run_pipeline(monkeypatch, case, rescue_scale=0.75)

    primary_has_gap = any(value is None for value in primary_fields(case).values())
    assert primary_has_gap is case.needs_fallback
    expected_calls = [6, 4, -4] if case.needs_rescale else [6, 4] if primary_has_gap else [6]
    assert ocr.calls == expected_calls  # -4 is the rescaled PSM 4 pass


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_without_a_fallback_only_psm6_runs(monkeypatch, case) -> None:
    fields, ocr = run_pipeline(monkeypatch, case, fallback_psm=None)

    assert ocr.calls == [6]
    assert fields == primary_fields(case)


# --- Invoices 1-3 and 6 stay correct -----------------------------------------


@pytest.mark.parametrize(
    "case", [case for case in CASES if case.name[0] in "1236"], ids=lambda case: case.name
)
def test_earlier_invoices_still_extract_every_field(monkeypatch, case) -> None:
    fields, ocr = run_pipeline(monkeypatch, case, rescue_scale=0.75)

    assert all(value is not None for value in fields.values())
    assert fields == case.expected
    assert -4 not in ocr.calls  # nothing was left for the rescaled pass


# --- Fallback rules -----------------------------------------------------------


def test_fallback_never_overwrites_a_primary_value(monkeypatch) -> None:
    """Crescent's PSM 6 text is combined with a different invoice's PSM 4 text."""
    other = next(case for case in CASES if case.name == "5_alpine")
    crescent = next(case for case in CASES if case.name == "7_crescent")
    ocr = FakeOCR(monkeypatch, {6: crescent.psm6, 4: other.psm4})

    fields = pipeline.process_invoice_image(b"image", fallback_psm=4)["invoice_fields"]

    primary = primary_fields(crescent)
    for name, value in primary.items():
        if value is not None:
            assert fields[name] == value, name
    assert fields["invoice_number"] == "CS-INV-2026-019"
    assert fields["invoice_date"] == "24/10/2026"
    assert fields["vendor_name"] == "Crescent Industrial Solutions"
    assert fields["tax_amount"] == "225000.00"
    # Only the empty fields were filled, and from the second pass.
    assert fields["subtotal"] == "5190.00"
    assert fields["total_amount"] == "6176.10"
    assert ocr.calls == [6, 4]


def test_complete_primary_result_does_not_start_the_fallback(monkeypatch) -> None:
    harborview = next(case for case in CASES if case.name == "4_harborview")
    ocr = FakeOCR(monkeypatch, {6: HARBORVIEW_PSM6, 4: OCREngineError("must not run")})

    fields = pipeline.process_invoice_image(b"image", fallback_psm=4)["invoice_fields"]

    assert fields == harborview.expected
    assert ocr.calls == [6]


@pytest.mark.parametrize("case", [c for c in CASES if c.needs_fallback], ids=lambda c: c.name)
def test_fallback_error_returns_the_primary_result(monkeypatch, caplog, case) -> None:
    FakeOCR(monkeypatch, {6: case.psm6, 4: OCREngineError("tesseract crashed")})

    with caplog.at_level(logging.WARNING, logger="app.pipeline"):
        result = pipeline.process_invoice_image(b"image", fallback_psm=4)

    assert result["invoice_fields"] == primary_fields(case)
    assert result["raw_ocr_text"] == case.psm6
    assert "tesseract crashed" in caplog.text


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_response_text_and_field_names_do_not_change(monkeypatch, case) -> None:
    FakeOCR(monkeypatch, {6: case.psm6, 4: case.psm4})

    result = pipeline.process_invoice_image(b"image", fallback_psm=4)

    assert list(result) == ["raw_ocr_text", "cleaned_text", "invoice_fields"]
    assert list(result["invoice_fields"]) == list(FIELD_NAMES)
    assert result["raw_ocr_text"] == case.psm6
    assert result["cleaned_text"] == clean_ocr_text(case.psm6)


# --- Specific failures that were found by testing -----------------------------


def test_harborview_psm6_gives_all_six_fields() -> None:
    harborview = next(case for case in CASES if case.name == "4_harborview")

    assert primary_fields(harborview) == harborview.expected


def test_alpine_total_misread_in_psm6_is_left_empty_then_read_from_psm4() -> None:
    alpine = next(case for case in CASES if case.name == "5_alpine")

    assert "6.176.10" in alpine.psm6  # the OCR misread
    assert primary_fields(alpine)["total_amount"] is None
    assert fallback_fields(alpine)["total_amount"] == "6176.10"


def test_crescent_subtotal_stays_none_in_the_full_size_passes() -> None:
    crescent = next(case for case in CASES if case.name == "7_crescent")

    assert "1.250,000.00" in crescent.psm6
    assert "1.250,000.00" in crescent.psm4
    assert primary_fields(crescent)["subtotal"] is None
    assert fallback_fields(crescent)["subtotal"] is None


def test_crescent_subtotal_is_read_by_the_rescaled_pass() -> None:
    crescent = next(case for case in CASES if case.name == "7_crescent")

    rescaled = extract_invoice_fields(clean_ocr_text(crescent.psm4_rescaled))
    assert "PKR 1,250,000.00" in crescent.psm4_rescaled
    assert rescaled == crescent.expected


def test_rescaled_pass_never_overwrites_values_from_earlier_passes(monkeypatch) -> None:
    crescent = next(case for case in CASES if case.name == "7_crescent")
    other = next(case for case in CASES if case.name == "5_alpine")
    # The rescaled pass returns a different invoice entirely: every field differs.
    FakeOCR(monkeypatch, {6: crescent.psm6, 4: crescent.psm4, -4: other.psm4})

    fields = pipeline.process_invoice_image(b"image", fallback_psm=4, rescue_scale=0.75)[
        "invoice_fields"
    ]

    for name, value in fallback_after_two_passes(crescent).items():
        if value is not None:
            assert fields[name] == value, name
    assert fields["subtotal"] == "5190.00"  # only the empty field was filled


def fallback_after_two_passes(case) -> dict[str, str | None]:
    return {
        name: value if value is not None else fallback_fields(case)[name]
        for name, value in primary_fields(case).items()
    }


def test_rescaled_pass_error_returns_the_previous_result(monkeypatch, caplog) -> None:
    crescent = next(case for case in CASES if case.name == "7_crescent")
    FakeOCR(monkeypatch, {6: crescent.psm6, 4: crescent.psm4, -4: OCREngineError("rescale boom")})

    with caplog.at_level(logging.WARNING, logger="app.pipeline"):
        result = pipeline.process_invoice_image(b"image", fallback_psm=4, rescue_scale=0.75)

    assert result["invoice_fields"] == fallback_after_two_passes(crescent)
    assert result["invoice_fields"]["subtotal"] is None
    assert "rescale boom" in caplog.text


def test_crescent_date_comes_from_date_not_due() -> None:
    fields = extract_invoice_fields(clean_ocr_text(CRESCENT_PSM6))

    assert fields["invoice_date"] == "24/10/2026"
    assert fields["invoice_date"] != "23/11/2026"


def test_alpine_example_line_is_not_used_as_an_amount() -> None:
    """The invoice explains the number format with 'Example: 1.234,56 = ...'."""
    fields = extract_invoice_fields(clean_ocr_text(ALPINE_PSM4))

    assert fields["subtotal"] == "5190.00"
    assert fields["tax_amount"] == "986.10"
    assert fields["total_amount"] == "6176.10"
