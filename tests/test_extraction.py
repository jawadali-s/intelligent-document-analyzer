"""Unit tests for app.extraction."""

from decimal import Decimal

import pytest

from app.extraction import InvalidExtractionInputError, extract_invoice_fields

EXPECTED_FIELD_NAMES = [
    "invoice_number",
    "invoice_date",
    "vendor_name",
    "subtotal",
    "tax_amount",
    "total_amount",
]

FULL_INVOICE = """Acme Traders (Pvt.) Ltd.
123 Main Street, Lahore
Phone: +92 300 1234567

Invoice No: INV-00123
Invoice Date: 2026-10-17

Widget A 2 50.00 100.00
Widget B 1 1,200.00 1,200.00

Subtotal: Rs. 1,300.00
Sales Tax (17%): Rs. 221.00
Grand Total: Rs. 1,521.00"""


def all_none() -> dict[str, None]:
    """A fresh dictionary with all six fields set to None."""
    return dict.fromkeys(EXPECTED_FIELD_NAMES)


def field(text: str, name: str) -> str | None:
    """Extract one field from the text."""
    return extract_invoice_fields(text)[name]


# --- Normal invoice and schema ----------------------------------------------


def test_normal_invoice_extracts_all_six_fields() -> None:
    assert extract_invoice_fields(FULL_INVOICE) == {
        "invoice_number": "INV-00123",
        "invoice_date": "2026-10-17",
        "vendor_name": "Acme Traders (Pvt.) Ltd.",
        "subtotal": "1300.00",
        "tax_amount": "221.00",
        "total_amount": "1521.00",
    }


def test_result_has_exactly_the_six_documented_keys_in_order() -> None:
    assert list(extract_invoice_fields(FULL_INVOICE)) == EXPECTED_FIELD_NAMES
    assert list(extract_invoice_fields("")) == EXPECTED_FIELD_NAMES


def test_all_values_are_strings_or_none() -> None:
    for value in extract_invoice_fields(FULL_INVOICE).values():
        assert value is None or isinstance(value, str)


# --- Invoice numbers ---------------------------------------------------------

INVOICE_NUMBER_LABELS = [
    "Invoice No",
    "Invoice No.",
    "Invoice Number",
    "Invoice #",
    "Invoice#",
    "Bill No",
    "Bill No.",
    "Bill Number",
    "Bill #",
    "Inv No",
    "Inv. No",
    "Inv #",
]


@pytest.mark.parametrize("label", INVOICE_NUMBER_LABELS)
def test_invoice_number_labels_with_colon(label: str) -> None:
    assert field(f"{label}: INV-001", "invoice_number") == "INV-001"


@pytest.mark.parametrize("label", INVOICE_NUMBER_LABELS)
def test_invoice_number_labels_without_colon(label: str) -> None:
    assert field(f"{label} INV-001", "invoice_number") == "INV-001"


@pytest.mark.parametrize("transform", [str.lower, str.upper, str.swapcase])
def test_invoice_number_labels_are_case_insensitive(transform) -> None:
    assert field(f"{transform('Invoice Number')}: INV-001", "invoice_number") == "INV-001"


@pytest.mark.parametrize(
    "value",
    ["INV-001", "00123", "007", "A-00123/B", "2026/INV/0007", "INV_2026.10.17", "7"],
)
def test_invoice_number_is_preserved_exactly_as_a_string(value: str) -> None:
    assert field(f"Invoice No: {value}", "invoice_number") == value


def test_hash_before_invoice_number_is_treated_as_part_of_the_label() -> None:
    assert field("Invoice No: #A-00123/B", "invoice_number") == "A-00123/B"


@pytest.mark.parametrize(
    "line",
    [
        "Invoice No: Date",  # No digit
        "Invoice No: INV 001",  # Contains a space
        "Invoice No: INV-001.",  # Trailing punctuation
        "Invoice No: ---",
        "Invoice No:",  # Nothing follows
        "Invoice No: " + "A1" * 21,  # Longer than 40 characters
        "Invoice Note: 12345",  # Not an invoice-number label
    ],
)
def test_malformed_or_missing_invoice_number_returns_none(line: str) -> None:
    assert field(line, "invoice_number") is None


# --- Dates -------------------------------------------------------------------


@pytest.mark.parametrize("label", ["Date", "Invoice Date", "Bill Date", "invoice date", "DATE"])
def test_date_labels(label: str) -> None:
    assert field(f"{label}: 2026-10-17", "invoice_date") == "2026-10-17"


@pytest.mark.parametrize(
    "value",
    [
        "2026-10-17",
        "17/10/2026",
        "17-10-2026",
        "17 Oct 2026",
        "17 October 2026",
        "17 oct 2026",
        "17 Oct. 2026",
        "17 Sept 2026",
        "5 Sep 2026",
        "1/2/2026",
        "03/04/2026",  # Valid as DD/MM and as MM/DD
        "10/17/2026",  # Valid only as MM/DD; preserved as written
        "29 Feb 2028",  # Leap year
    ],
)
def test_supported_date_formats_are_preserved_exactly(value: str) -> None:
    assert field(f"Date: {value}", "invoice_date") == value


@pytest.mark.parametrize(
    "value",
    [
        "2026-13-01",
        "2026-02-30",
        "31/02/2026",
        "31/04/2026",
        "13/13/2026",
        "29 Feb 2027",
        "17/10/26",  # Two-digit year
        "2026-1-5",  # ISO needs two-digit month and day
        "2026/10/17",  # Not a supported format
        "17 Foo 2026",
        "17 Oct",
        "tomorrow",
        "17/10/2026 18:30",
    ],
)
def test_invalid_or_unsupported_dates_return_none(value: str) -> None:
    assert field(f"Date: {value}", "invoice_date") is None


def test_due_date_is_not_mistaken_for_invoice_date() -> None:
    assert field("Due Date: 2026-11-17", "invoice_date") is None


# --- Money formats and precision --------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Rs. 2,500.00", "2500.00"),
        ("Rs.2,500", "2500"),
        ("Rs 2500", "2500"),
        ("PKR 2500", "2500"),
        ("2500 PKR", "2500"),
        ("$125.50", "125.50"),
        ("$ 1,000", "1000"),
        ("€99.99", "99.99"),
        ("1,234.56", "1234.56"),
        ("1,234,567.89", "1234567.89"),
        ("Rs. 1,23,456.00", "123456.00"),  # Indian digit grouping
        ("0.50", "0.50"),
        ("125.5", "125.5"),
        ("00500", "500"),
    ],
)
def test_supported_currency_formats(raw: str, expected: str) -> None:
    assert field(f"Total: {raw}", "total_amount") == expected


@pytest.mark.parametrize(
    ("label", "name"),
    [("Subtotal", "subtotal"), ("Sales Tax", "tax_amount"), ("Grand Total", "total_amount")],
)
def test_all_money_fields_use_the_same_representation(label: str, name: str) -> None:
    assert field(f"{label}: Rs. 2,500.00", name) == "2500.00"


def test_amount_without_colon_after_label() -> None:
    assert field("Total Rs. 2,500.00", "total_amount") == "2500.00"
    assert field("Grand Total $125.50", "total_amount") == "125.50"


def test_money_values_are_decimal_strings_without_float_errors() -> None:
    result = extract_invoice_fields("Subtotal: 0.10\nTax: 0.20\nTotal: 0.30")

    assert result["subtotal"] == "0.10"
    assert result["tax_amount"] == "0.20"
    assert result["total_amount"] == "0.30"
    assert Decimal(result["subtotal"]) + Decimal(result["tax_amount"]) == Decimal(
        result["total_amount"]
    )


@pytest.mark.parametrize(
    "raw",
    [
        "12,50.00",
        "1,23",
        "1,2345.00",
        "1.234,56",  # European format is not supported
        "100.000",  # Three decimal places
        "-100.00",
        "abc",
        "Rs.",
        "$$100",
        "12 34",
        "100.00 or 200.00",
    ],
)
@pytest.mark.parametrize(
    ("label", "name"),
    [("Subtotal", "subtotal"), ("Tax", "tax_amount"), ("Total", "total_amount")],
)
def test_malformed_amounts_return_none(label: str, name: str, raw: str) -> None:
    assert field(f"{label}: {raw}", name) is None


# --- Case-insensitive labels -------------------------------------------------

LABELLED_LINES = [
    ("Invoice No", "INV-9"),
    ("Invoice Date", "2026-10-17"),
    ("Subtotal", "100.00"),
    ("Sales Tax", "17.00"),
    ("Grand Total", "117.00"),
]


@pytest.mark.parametrize("transform", [str.lower, str.upper, str.swapcase])
def test_all_labels_are_case_insensitive(transform) -> None:
    text = "\n".join(f"{transform(label)}: {value}" for label, value in LABELLED_LINES)

    assert extract_invoice_fields(text) == {
        "invoice_number": "INV-9",
        "invoice_date": "2026-10-17",
        "vendor_name": None,  # The first line is a field, not a vendor
        "subtotal": "100.00",
        "tax_amount": "17.00",
        "total_amount": "117.00",
    }


# --- Subtotal, tax and total disambiguation ----------------------------------


def test_subtotal_tax_and_total_are_kept_apart() -> None:
    result = extract_invoice_fields("Subtotal: 1,000.00\nTax: 170.00\nTotal: 1,170.00")

    assert result["subtotal"] == "1000.00"
    assert result["tax_amount"] == "170.00"
    assert result["total_amount"] == "1170.00"


@pytest.mark.parametrize("label", ["Subtotal", "Sub Total", "Sub-Total", "SUBTOTAL"])
def test_subtotal_labels_never_produce_a_total(label: str) -> None:
    result = extract_invoice_fields(f"{label}: 1,000.00")

    assert result["subtotal"] == "1000.00"
    assert result["total_amount"] is None


def test_total_alone_does_not_produce_subtotal_or_tax() -> None:
    result = extract_invoice_fields("Total: 1,170.00")

    assert result["total_amount"] == "1170.00"
    assert result["subtotal"] is None
    assert result["tax_amount"] is None


@pytest.mark.parametrize(
    "label", ["Tax", "Sales Tax", "VAT", "GST", "Tax Amount", "Total Tax", "TAX"]
)
def test_tax_labels_produce_tax_and_never_a_total(label: str) -> None:
    result = extract_invoice_fields(f"{label}: 17.00")

    assert result["tax_amount"] == "17.00"
    assert result["total_amount"] is None
    assert result["subtotal"] is None


def test_total_tax_line_is_not_read_as_the_total() -> None:
    result = extract_invoice_fields("Total Tax: 17.00\nGrand Total: 117.00")

    assert result["tax_amount"] == "17.00"
    assert result["total_amount"] == "117.00"


@pytest.mark.parametrize(
    "label",
    ["Grand Total", "Total Due", "Amount Due", "Total Amount", "Total Amount Due", "Total"],
)
def test_total_labels(label: str) -> None:
    assert field(f"{label}: 100.00", "total_amount") == "100.00"


@pytest.mark.parametrize(
    "line", ["Tax 17%: 170.00", "VAT (17%): Rs. 170.00", "GST 17.0%: 170.00", "Tax 17% 170.00"]
)
def test_tax_rate_next_to_label_is_not_read_as_the_amount(line: str) -> None:
    assert field(line, "tax_amount") == "170.00"


def test_tax_rate_without_amount_returns_none() -> None:
    assert field("Tax: 17%", "tax_amount") is None


# --- Values on the line after the label -------------------------------------


def test_values_on_the_line_after_a_label_are_extracted() -> None:
    text = (
        "Invoice No:\nINV-001\n"
        "Invoice Date:\n2026-10-17\n"
        "Subtotal:\nRs. 100.00\n"
        "Sales Tax:\nRs. 17.00\n"
        "Total:\nRs. 117.00"
    )

    assert extract_invoice_fields(text) == {
        "invoice_number": "INV-001",
        "invoice_date": "2026-10-17",
        "vendor_name": None,
        "subtotal": "100.00",
        "tax_amount": "17.00",
        "total_amount": "117.00",
    }


def test_label_alone_on_the_last_line_returns_none() -> None:
    assert field("Total:", "total_amount") is None


def test_label_followed_by_another_label_returns_none() -> None:
    assert field("Invoice No:\nInvoice Date: 2026-10-17", "invoice_number") is None


# --- Vendor name -------------------------------------------------------------


@pytest.mark.parametrize(
    "vendor",
    [
        "Acme Corp",
        "Ahmed & Sons (Pvt.) Ltd.",
        "O'Brien-Smith, Inc.",
        "Café Müller GmbH",
        "شركة النور للتجارة",
        "Zhang's Store 张记商店",
        "Date Palm Trading Co",  # Starts with a label word
        "Total Solutions Ltd",
        "Tax Consultants Ltd",
    ],
)
def test_vendor_name_is_the_first_line_and_preserved_exactly(vendor: str) -> None:
    assert field(f"{vendor}\nInvoice No: INV-1", "vendor_name") == vendor


@pytest.mark.parametrize("title", ["INVOICE", "Tax Invoice", "Sales Invoice", "Receipt"])
def test_generic_title_line_is_skipped_before_the_vendor(title: str) -> None:
    assert field(f"{title}\nAcme Corp\nInvoice No: INV-1", "vendor_name") == "Acme Corp"


def test_leading_blank_lines_are_ignored_for_the_vendor() -> None:
    assert field("\n\nAcme Corp\nInvoice No: INV-1", "vendor_name") == "Acme Corp"


def test_vendor_name_keeps_spacing_inside_the_line() -> None:
    assert field("Acme   Traders\nInvoice No: INV-1", "vendor_name") == "Acme   Traders"


@pytest.mark.parametrize(
    "text",
    [
        "Invoice No: INV-1\nAcme Corp",  # The first line is metadata
        "INVOICE\nInvoice No: INV-1",  # Only a title before the metadata
        "Tax Invoice\nInvoice No: INV-1",
        "Date: 2026-10-17\nTotal: 10.00",
        "Total\nInvoice No: INV-1",  # A bare label is not a vendor
        "Tax ID: 1234567\nInvoice No: INV-1",  # Contains a colon
        "7-Eleven\nInvoice No: INV-1",  # Starts with a digit
        "A" * 81 + "\nInvoice No: INV-1",  # Too long
        "billing@acme.com\nInvoice No: INV-1",
        "www.acme.com\nInvoice No: INV-1",
        "2026-10-17\nInvoice No: INV-1",  # No letters
        "Call 0300 1234567\nInvoice No: INV-1",  # More digits than letters
        "Acme Corp\nThank you for your business",  # No invoice fields follow
        "Acme Corp",
    ],
)
def test_vendor_name_is_none_when_not_clearly_identifiable(text: str) -> None:
    assert field(text, "vendor_name") is None


# --- Unrelated numbers -------------------------------------------------------


def test_phone_numbers_addresses_and_line_items_are_not_extracted() -> None:
    text = (
        "Fresh Mart\n"
        "House 12, Street 5, Sector F-7, Islamabad 44000\n"
        "Phone: +92 51 1234567\n"
        "NTN: 1234567-8\n"
        "Invoice No: INV-77\n"
        "Date: 2026-10-17\n"
        "Item 3 x 400.00 1,200.00\n"
        "Total: Rs. 1,200.00"
    )

    assert extract_invoice_fields(text) == {
        "invoice_number": "INV-77",
        "invoice_date": "2026-10-17",
        "vendor_name": "Fresh Mart",
        "subtotal": None,
        "tax_amount": None,
        "total_amount": "1200.00",
    }


def test_text_without_labelled_fields_returns_all_none() -> None:
    text = "Call 0300 1234567\nHouse 12, Street 5 44000\nWidget 2 x 50.00 100.00\n1,200.00"

    assert extract_invoice_fields(text) == all_none()


def test_registration_labels_with_text_before_the_number_are_not_tax_amounts() -> None:
    text = "Tax ID: 1234567\nVAT No: 98765\nGST No: 555\nTotal: 100.00"

    result = extract_invoice_fields(text)

    assert result["tax_amount"] is None
    assert result["total_amount"] == "100.00"


# --- Missing fields ----------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "present"),
    [
        ("Invoice No: INV-1", {"invoice_number": "INV-1"}),
        ("Date: 2026-10-17", {"invoice_date": "2026-10-17"}),
        ("Subtotal: 10.00", {"subtotal": "10.00"}),
        ("VAT: 1.70", {"tax_amount": "1.70"}),
        ("Total: 11.70", {"total_amount": "11.70"}),
    ],
)
def test_only_the_present_field_is_filled(text: str, present: dict[str, str]) -> None:
    assert extract_invoice_fields(text) == {**all_none(), **present}


# --- Empty and whitespace-only text -----------------------------------------


@pytest.mark.parametrize("text", ["", " ", "\n", "\r\n", "\t \n  \n"])
def test_empty_and_whitespace_only_text_returns_all_none(text: str) -> None:
    assert extract_invoice_fields(text) == all_none()


# --- Invalid input -----------------------------------------------------------


@pytest.mark.parametrize(
    "bad_input",
    [None, 123, 1.5, ["Invoice No: 1"], {"text": "x"}, b"Invoice No: 1", object()],
    ids=["none", "int", "float", "list", "dict", "bytes", "object"],
)
def test_non_string_input_raises_invalid_extraction_input_error(bad_input: object) -> None:
    with pytest.raises(InvalidExtractionInputError, match="must be a str"):
        extract_invoice_fields(bad_input)  # type: ignore[arg-type]


def test_invalid_input_error_is_a_type_error_naming_the_type() -> None:
    with pytest.raises(TypeError, match="NoneType"):
        extract_invoice_fields(None)  # type: ignore[arg-type]


# --- Multiple occurrences (documented behaviour) -----------------------------


@pytest.mark.parametrize(
    ("text", "name", "expected"),
    [
        # Same label twice: the first occurrence wins.
        ("Invoice No: A-1\nInvoice No: B-2", "invoice_number", "A-1"),
        ("Total: 100.00\nTotal: 200.00", "total_amount", "100.00"),
        ("Invoice Date: 2026-02-02\nInvoice Date: 2026-03-03", "invoice_date", "2026-02-02"),
        # Different labels: the higher-priority label wins, wherever it appears.
        ("Bill No: B-2\nInvoice No: A-1", "invoice_number", "A-1"),
        ("Date: 2026-01-01\nInvoice Date: 2026-02-02", "invoice_date", "2026-02-02"),
        ("Total: 100.00\nGrand Total: 118.00", "total_amount", "118.00"),
        ("Total: 100.00\nAmount Due: 40.00", "total_amount", "40.00"),
        ("Tax: 1.00\nSales Tax: 2.00", "tax_amount", "2.00"),
        ("GST: 5.00\nVAT: 6.00", "tax_amount", "6.00"),
        # A malformed value is skipped in favour of the next valid one.
        ("Total: abc\nTotal: 100.00", "total_amount", "100.00"),
        ("Grand Total: abc\nTotal: 100.00", "total_amount", "100.00"),
        ("Date: 2026-13-45\nDate: 2026-10-17", "invoice_date", "2026-10-17"),
    ],
)
def test_multiple_occurrences_follow_documented_priority(
    text: str, name: str, expected: str
) -> None:
    assert field(text, name) == expected


# --- Input is not modified; results are deterministic -----------------------


def test_input_text_is_not_modified_or_cleaned() -> None:
    text = "  Acme   Traders  \r\nInvoice No:  INV-1  \r\n"
    snapshot = "".join(list(text))

    result = extract_invoice_fields(text)

    assert text == snapshot
    assert result["vendor_name"] == "Acme   Traders"
    assert result["invoice_number"] == "INV-1"


def test_repeated_calls_return_equal_results() -> None:
    results = [extract_invoice_fields(FULL_INVOICE) for _ in range(3)]

    assert results[0] == results[1] == results[2]


def test_each_call_returns_a_new_dictionary() -> None:
    first = extract_invoice_fields(FULL_INVOICE)
    first["vendor_name"] = "Changed"

    second = extract_invoice_fields(FULL_INVOICE)

    assert second["vendor_name"] == "Acme Traders (Pvt.) Ltd."