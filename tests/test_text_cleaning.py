"""Unit tests for app.text_cleaning."""

import pytest

from app.text_cleaning import InvalidTextInputError, clean_ocr_text


# --- Normal text -------------------------------------------------------------


def test_normal_invoice_text_is_preserved() -> None:
    text = "INVOICE\nInvoice No: INV-001\nDate: 2026-10-17\nTotal: Rs. 2,500.00"

    assert clean_ocr_text(text) == text


def test_trailing_newline_is_removed() -> None:
    assert clean_ocr_text("Invoice No: INV-001\n") == "Invoice No: INV-001"


# --- Line endings ------------------------------------------------------------


@pytest.mark.parametrize("line_ending", ["\n", "\r\n", "\r"])
def test_all_line_ending_styles_become_newline(line_ending: str) -> None:
    text = line_ending.join(["Invoice No: INV-001", "Total: $125.50"])

    assert clean_ocr_text(text) == "Invoice No: INV-001\nTotal: $125.50"


def test_mixed_line_endings_are_normalized() -> None:
    assert clean_ocr_text("a\r\nb\rc\nd") == "a\nb\nc\nd"


def test_windows_blank_line_is_kept_as_one_blank_line() -> None:
    assert clean_ocr_text("a\r\n\r\nb") == "a\n\nb"


# --- Whitespace --------------------------------------------------------------


def test_leading_and_trailing_whitespace_of_whole_text_is_removed() -> None:
    assert clean_ocr_text("  \n\t Invoice No: INV-001 \n  \n") == "Invoice No: INV-001"


def test_leading_and_trailing_spaces_of_each_line_are_removed() -> None:
    assert clean_ocr_text("  A  \n\tB\t\n   C") == "A\nB\nC"


def test_tabs_and_repeated_spaces_become_single_space() -> None:
    assert clean_ocr_text("Invoice\t\tNo:   INV-001") == "Invoice No: INV-001"


def test_tab_between_label_and_value_becomes_space() -> None:
    assert clean_ocr_text("Total:\t$125.50") == "Total: $125.50"


def test_non_breaking_space_is_treated_as_whitespace() -> None:
    assert clean_ocr_text("Total:\u00a0\u00a0$10.00") == "Total: $10.00"


def test_form_feed_added_by_tesseract_is_removed() -> None:
    assert clean_ocr_text("Total: 10\n\f") == "Total: 10"


# --- Blank lines -------------------------------------------------------------


def test_multiple_blank_lines_reduce_to_one() -> None:
    assert clean_ocr_text("A\n\n\n\nB") == "A\n\nB"


def test_single_blank_line_is_preserved() -> None:
    assert clean_ocr_text("A\n\nB") == "A\n\nB"


def test_whitespace_only_lines_count_as_blank_lines() -> None:
    assert clean_ocr_text("A\n   \n\t\n \nB") == "A\n\nB"


def test_leading_and_trailing_blank_lines_are_removed() -> None:
    assert clean_ocr_text("\n\nA\n\n\n") == "A"


# --- Empty input -------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    ["", " ", "\n", "\r\n", "\t \r\n \f \n"],
    ids=["empty", "space", "newline", "crlf", "mixed-whitespace"],
)
def test_empty_and_whitespace_only_input_returns_empty_string(text: str) -> None:
    assert clean_ocr_text(text) == ""


# --- Unicode -----------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "رقم الفاتورة: ١٢٣",
        "کل رقم: Rs. 2,500",
        "Café Müller € 1.234,50",
        "請求書番号 INV-001",
        "Счёт № 42",
    ],
)
def test_unicode_text_is_preserved(text: str) -> None:
    assert clean_ocr_text(text) == text


def test_whitespace_is_normalized_around_unicode_text() -> None:
    assert clean_ocr_text("  Café \t Müller  \n  €  99  ") == "Café Müller\n€ 99"


def test_zero_width_space_is_not_removed() -> None:
    assert clean_ocr_text("A\u200bB") == "A\u200bB"


# --- Invoice data is preserved exactly --------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        "INV-001",
        "#A-00123/B",
        "00123",
        "007",
        "Rs. 2,500.00",
        "$125.50",
        "€99.99",
        "1,234,567.89",
        "1.234,56",
        "2026-10-17",
        "17/10/2026",
        "+92 300 1234567",
        "100%",
    ],
)
def test_invoice_values_are_preserved_exactly(value: str) -> None:
    assert clean_ocr_text(f"Label: {value}") == f"Label: {value}"
    assert clean_ocr_text(f"  Label:\t  {value}  \r\n") == f"Label: {value}"


def test_leading_zeros_are_not_removed() -> None:
    assert clean_ocr_text("Account: 00123\nRef: 000") == "Account: 00123\nRef: 000"


def test_no_space_is_inserted_or_removed_inside_values() -> None:
    assert clean_ocr_text("Total:Rs.2,500.00") == "Total:Rs.2,500.00"


# --- Line breaks -------------------------------------------------------------


def test_label_and_value_on_separate_lines_stay_separate() -> None:
    text = "Invoice No:\nINV-001\nDate:\n2026-10-17"

    assert clean_ocr_text(text) == text


def test_lines_are_never_joined() -> None:
    assert clean_ocr_text("Item A\nItem B\nItem C") == "Item A\nItem B\nItem C"


# --- Invalid input -----------------------------------------------------------


@pytest.mark.parametrize(
    "bad_input",
    [None, 123, 1.5, ["a", "b"], b"bytes", {"a": 1}, object()],
    ids=["none", "int", "float", "list", "bytes", "dict", "object"],
)
def test_non_string_input_raises_invalid_text_input_error(bad_input: object) -> None:
    with pytest.raises(InvalidTextInputError, match="must be a str"):
        clean_ocr_text(bad_input)  # type: ignore[arg-type]


def test_invalid_input_error_is_a_type_error() -> None:
    with pytest.raises(TypeError):
        clean_ocr_text(None)  # type: ignore[arg-type]


def test_error_message_names_the_received_type() -> None:
    with pytest.raises(InvalidTextInputError, match="int"):
        clean_ocr_text(42)  # type: ignore[arg-type]


# --- Idempotence and determinism --------------------------------------------


MESSY_SAMPLES = [
    "",
    "  \n\n ",
    "Invoice\t No:   INV-001 \r\n\r\n\r\nTotal:  Rs. 2,500.00\r",
    "\n\nA\u00a0\u00a0B\n\n\n\nC\f\n",
    "  Café \t Müller  \n   \n  €  99  ",
]


@pytest.mark.parametrize("text", MESSY_SAMPLES)
def test_cleaning_is_idempotent(text: str) -> None:
    once = clean_ocr_text(text)

    assert clean_ocr_text(once) == once


@pytest.mark.parametrize("text", MESSY_SAMPLES)
def test_cleaning_is_deterministic(text: str) -> None:
    assert clean_ocr_text(text) == clean_ocr_text(text)


def test_input_string_is_not_affected_by_cleaning() -> None:
    original = "  A \t B  \r\n"
    snapshot = str(original)

    clean_ocr_text(original)

    assert original == snapshot


# --- str subclass ------------------------------------------------------------


def test_str_subclass_is_accepted_and_returns_plain_str() -> None:
    class MyStr(str):
        pass

    result = clean_ocr_text(MyStr("  A  B  "))

    assert result == "A B"
    assert type(result) is str