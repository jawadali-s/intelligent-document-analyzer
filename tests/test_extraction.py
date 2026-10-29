"""Unit tests for app.extraction."""

from decimal import Decimal

import pytest

from app.extraction import InvalidExtractionInputError, extract_invoice_fields
from app.text_cleaning import clean_ocr_text
from invoice_ocr_samples import (
    BLUEPEAK_EXPECTED,
    BLUEPEAK_PSM4,
    BLUEPEAK_PSM6,
    GREENFIELD_EXPECTED,
    GREENFIELD_PSM4,
    GREENFIELD_PSM6,
    NORTHSTAR_EXPECTED,
    NORTHSTAR_PSM4,
    NORTHSTAR_PSM6,
)

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
        "1,2345.00",
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


# --- Vendor name: decorative leading lines -----------------------------------

DECORATIVE_LINES = ["——", "-----", "=====", "*****", "• • •", "~~~ ~~~", "—— ——", "___"]


@pytest.mark.parametrize("decoration", DECORATIVE_LINES)
def test_decorative_leading_line_is_skipped_before_the_vendor(decoration: str) -> None:
    text = f"{decoration}\nStudio Name\nInvoice No: INV-1"

    assert field(text, "vendor_name") == "Studio Name"


def test_several_decorative_and_blank_leading_lines_are_skipped() -> None:
    text = "\n——\n\n=====\nStudio Name\nInvoice No: INV-1"

    assert field(text, "vendor_name") == "Studio Name"


def test_decorative_line_after_a_generic_title_is_skipped() -> None:
    text = "INVOICE\n=====\nAcme Corp\nInvoice No: INV-1"

    assert field(text, "vendor_name") == "Acme Corp"


def test_only_decorative_lines_before_the_fields_give_no_vendor() -> None:
    assert field("——\nInvoice No: INV-1", "vendor_name") is None
    assert field("——\n=====", "vendor_name") is None


@pytest.mark.parametrize(
    "first_line",
    ["12345", "2026-10-17", "-- 1 --", "7-Eleven", "0300 1234567"],
)
def test_numeric_leading_line_is_not_skipped_when_looking_for_the_vendor(
    first_line: str,
) -> None:
    text = f"{first_line}\nStudio Name\nInvoice No: INV-1"

    assert field(text, "vendor_name") is None


def test_leading_line_with_letters_is_not_skipped_as_decoration() -> None:
    text = "OO —__ ar\nStudio Name\nInvoice No: INV-1"

    assert field(text, "vendor_name") == "OO —__ ar"


def test_decorative_line_does_not_confirm_a_vendor_without_invoice_fields() -> None:
    assert field("——\nStudio Name\nThank you for your business", "vendor_name") is None


def test_garbled_ocr_text_with_a_decorative_first_line_still_returns_all_none() -> None:
    text = (
        "——\n"
        "FROM SALTO\n"
        "Studio Name Client Name\n"
        "Description Qty Rate Amount\n"
        "Photography Session 1 $450 $450\n"
        "Photo Editing 25 $20 $500\n"
        "Prints Package 1 $150 $150\n"
        "prorat 58100)\n"
        "OO —__ ar"
    )

    assert extract_invoice_fields(text) == all_none()


# --- Vendor name: explicit FROM label ----------------------------------------


def test_from_label_gives_the_vendor_name() -> None:
    assert field("FROM: Studio Name\nInvoice No: INV-1", "vendor_name") == "Studio Name"


@pytest.mark.parametrize(
    "line",
    [
        "FROM: Studio Name",
        "From: Studio Name",
        "from: Studio Name",
        "FROM : Studio Name",
        "FROM:Studio Name",
        "FROM:   Studio Name",
        "FROM： Studio Name",  # Full-width colon, as accepted for other labels
    ],
)
def test_from_label_is_case_and_spacing_tolerant(line: str) -> None:
    assert field(f"{line}\nInvoice No: INV-1", "vendor_name") == "Studio Name"


@pytest.mark.parametrize(
    "vendor",
    ["Ahmed & Sons (Pvt.) Ltd.", "O'Brien-Smith, Inc.", "Café Müller GmbH", "شركة النور للتجارة"],
)
def test_from_label_vendor_is_preserved_exactly(vendor: str) -> None:
    assert field(f"FROM: {vendor}\nInvoice No: INV-1", "vendor_name") == vendor


def test_from_label_works_without_other_invoice_fields() -> None:
    assert field("FROM: Studio Name", "vendor_name") == "Studio Name"


def test_from_label_after_a_title_returns_the_sender_not_the_customer() -> None:
    text = "INVOICE\nFROM: Acme Ltd\nBILL TO: Client Co\nInvoice No: INV-1"

    assert field(text, "vendor_name") == "Acme Ltd"


def test_explicit_from_label_wins_over_the_first_line() -> None:
    text = "Header Text\nFROM: Studio Name\nInvoice No: INV-1"

    assert field(text, "vendor_name") == "Studio Name"


def test_first_valid_from_label_wins() -> None:
    assert field("FROM: Acme Ltd\nFROM: Other Ltd", "vendor_name") == "Acme Ltd"
    assert field("FROM: BILL TO:\nFROM: Acme Ltd", "vendor_name") == "Acme Ltd"


def test_from_label_does_not_change_the_other_fields() -> None:
    text = "FROM: Studio Name\nInvoice No: INV-1\nDate: 2026-10-17\nTotal: $1,100"

    assert extract_invoice_fields(text) == {
        "invoice_number": "INV-1",
        "invoice_date": "2026-10-17",
        "vendor_name": "Studio Name",
        "subtotal": None,
        "tax_amount": None,
        "total_amount": "1100",
    }


@pytest.mark.parametrize(
    "line",
    [
        "FROM:",  # Nothing after the label
        "FROM:   ",
        "FROM: BILL TO:",  # Two labels, no name
        "FROM: Studio Name BILL TO: Client Name",  # Two parties with a second label
        "FROM: Studio Name Bill To Client Name",  # Customer label without a colon
        "FROM: Studio Name Ship To Client Name",
        "FROM: Studio Name Sold To Client Name",
        "FROM: billing@acme.com",
        "FROM: www.acme.com",
        "FROM: 12345",  # Starts with a digit
        "FROM: ----",  # No letters
        "FROM: " + "A" * 81,  # Too long
        "FROM: Invoice No: INV-1",  # Another label inside
    ],
)
def test_invalid_from_value_gives_no_vendor(line: str) -> None:
    assert field(f"{line}\nInvoice No: INV-1", "vendor_name") is None


def test_from_alone_on_its_line_is_not_returned_as_the_vendor() -> None:
    assert field("FROM:\nStudio Name\nInvoice No: INV-1", "vendor_name") is None


@pytest.mark.parametrize("line", ["FROM:", "FROM: BILL TO:", "From:"])
def test_vendor_is_never_the_word_from(line: str) -> None:
    assert field(f"{line}\nInvoice No: INV-1", "vendor_name") not in {"FROM", "From", "FROM:"}


def test_invalid_from_label_falls_back_to_the_normal_vendor_rule() -> None:
    text = "Acme Corp\nFROM: billing@acme.com\nInvoice No: INV-1"

    assert field(text, "vendor_name") == "Acme Corp"


@pytest.mark.parametrize(
    "vendor", ["From Scratch Bakery", "Fromage Dairy Ltd", "From the Farm Foods"]
)
def test_vendor_names_that_merely_start_with_from_use_the_normal_rule(vendor: str) -> None:
    assert field(f"{vendor}\nInvoice No: INV-1", "vendor_name") == vendor


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


# --- Labels inside a line: real OCR text -------------------------------------

# OCR output of a fictional test invoice, kept exactly as produced (noise included).
LARKFIELD_OCR_TEXT = """Larkfield Stationers TAX INVOICE
. A fictional company Invoice no. INV-2026-0142
22 Example Market, Sampleton 000 000 Invoice date 18 September 2026
Tonnes SPECIMEN aoe Due date 18 October 2026
no.: * 7
»  1eKn0 ; Order ref, PO-77310
BILL TO ; :
Example Traders (fictional) . .
Unit 5, Specimen Industrial Estate
Sampleton 000 000° . .
Tax no.: SPECIMEN-TAX-0002 .
# = ~~ Description - . Ay Unit price Amount
1° A4 copier paper, 75 gsm, ream of 500 20 _ *285.00 %5,700.00
2 Ballpoint pens, blue, box of 50 . 4 420.00 ™1,680.00
3 Lever-arch files, A4, pack of 10 . 3 ‘899.00 %2,697.00
4 Whiteboard markers, assorted, pack of 4 6 7160.00 %960.00
5 Desk organiser, mesh, black 5 %349.00 71,745.00 .
6 Sticky notes, 76 x 76 mm, pack of 12 — 8 7240.00 %1,920.00
; Subtotal %14,702.00
. , GST at 18% %2,646.36 .
. Total due 17,348.36
Payment
Bank transfer to Specimen Bank, account SPECIMEN-0000-111 1, within .
30 days. Please quote the invoice number with your payment.
Every amount is quantity times unit price; the subtotal, GST and total add
up to the paisa.
: SPECIMEN - fictional document made by filedone.ai for testing. Free to use (CC0 1.0). Page 1 of 1"""

LARKFIELD_EXPECTED = {
    "invoice_number": "INV-2026-0142",
    "invoice_date": "18 September 2026",
    "vendor_name": "Larkfield Stationers",
    "subtotal": "14702.00",
    "tax_amount": "2646.36",
    "total_amount": "17348.36",
}


def test_larkfield_ocr_text_extracts_all_six_fields() -> None:
    assert extract_invoice_fields(LARKFIELD_OCR_TEXT) == LARKFIELD_EXPECTED


def test_larkfield_result_keeps_the_six_keys_in_order() -> None:
    assert list(extract_invoice_fields(LARKFIELD_OCR_TEXT)) == EXPECTED_FIELD_NAMES


def test_larkfield_values_do_not_come_from_unrelated_lines() -> None:
    result = extract_invoice_fields(LARKFIELD_OCR_TEXT)

    assert result["invoice_date"] != "18 October 2026"  # Due date
    assert result["tax_amount"] != "18"  # Tax rate
    assert result["total_amount"] != result["subtotal"]
    assert "BILL" not in (result["vendor_name"] or "").upper()
    assert "Example Traders" not in (result["vendor_name"] or "")


def test_larkfield_without_the_invoice_date_line_does_not_use_the_due_date() -> None:
    text = LARKFIELD_OCR_TEXT.replace(
        "22 Example Market, Sampleton 000 000 Invoice date 18 September 2026",
        "22 Example Market, Sampleton 000 000",
    )

    assert extract_invoice_fields(text) == {**LARKFIELD_EXPECTED, "invoice_date": None}


def test_larkfield_without_the_total_line_does_not_guess_a_total() -> None:
    text = LARKFIELD_OCR_TEXT.replace(". Total due 17,348.36\n", "")

    assert extract_invoice_fields(text) == {**LARKFIELD_EXPECTED, "total_amount": None}


def test_larkfield_without_the_subtotal_line_does_not_guess_a_subtotal() -> None:
    text = LARKFIELD_OCR_TEXT.replace("; Subtotal %14,702.00\n", "")

    assert extract_invoice_fields(text) == {**LARKFIELD_EXPECTED, "subtotal": None}


def test_larkfield_body_without_header_and_totals_returns_all_none() -> None:
    lines = LARKFIELD_OCR_TEXT.split("\n")
    body = "\n".join(lines[11:18] + lines[21:])  # Line items, payment note, footer

    assert extract_invoice_fields(body) == all_none()


# --- Labels inside a line: invoice number ------------------------------------


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        (". A fictional company Invoice no. INV-2026-0142", "INV-2026-0142"),
        ("22 Example Market, Lahore Invoice # A-17", "A-17"),
        ("Phone 0300 1234567 Invoice Number: 00123", "00123"),
        ("Sampleton 000 000 INVOICE NO INV-9", "INV-9"),
        ("Page 1 Bill No. B-5", "B-5"),
        ("Lahore Inv. No 77", "77"),
        ("Invoice no. INV-2026-0142 .", "INV-2026-0142"),  # Stray punctuation after the value
    ],
)
def test_invoice_number_label_inside_a_line(line: str, expected: str) -> None:
    assert field(line, "invoice_number") == expected


@pytest.mark.parametrize(
    "line",
    [
        "30 days. Please quote the invoice number with your payment.",
        "Please quote invoice no. 12 and 13",
        "Phone 0300 1234567 Invoice No:",
        "Phone 0300 1234567 Invoice No: ---",
        "Lahore Invoice No. INV-1 Invoice date 18 September 2026",  # Two fields on one line
        "Invoice noted: INV-5",
        "Lahore Invoice Notes 12345",
    ],
)
def test_unclear_invoice_number_inside_a_line_returns_none(line: str) -> None:
    assert field(line, "invoice_number") is None


def test_invoice_number_on_its_own_line_still_wins_over_the_relaxed_search() -> None:
    text = "Acme Ltd Invoice no. X-1\nInvoice No: A-2"

    assert field(text, "invoice_number") == "A-2"


# --- Labels inside a line: invoice date and due date ------------------------


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("22 Example Market, Sampleton 000 000 Invoice date 18 September 2026", "18 September 2026"),
        ("Page 1 Invoice Date: 2026-10-17", "2026-10-17"),
        ("Lahore INVOICE DATE 17/10/2026", "17/10/2026"),
        ("Lahore Bill Date 17 Oct 2026", "17 Oct 2026"),
        (". Date: 18 September 2026", "18 September 2026"),  # Bare label after stray punctuation
    ],
)
def test_invoice_date_label_inside_a_line(line: str, expected: str) -> None:
    assert field(line, "invoice_date") == expected


@pytest.mark.parametrize(
    "line",
    [
        "Tonnes SPECIMEN aoe Due date 18 October 2026",
        "Net 30 Payment Due Date: 2026-11-17",
        "Lahore Order date 5 Oct 2026",
        "Issued Date 2026-10-17",  # A bare "Date" inside a line is too ambiguous
        "Lahore Invoice date soon",
        "Lahore Invoice date 31/04/2026",  # Not a real date
    ],
)
def test_other_dates_and_unclear_dates_inside_a_line_return_none(line: str) -> None:
    assert field(line, "invoice_date") is None


def test_invoice_date_is_chosen_over_a_due_date_in_either_order() -> None:
    due_first = "Sampleton Due date 18 October 2026\nSampleton Invoice date 18 September 2026"
    due_last = "Sampleton Invoice date 18 September 2026\nSampleton Due date 18 October 2026"

    assert field(due_first, "invoice_date") == "18 September 2026"
    assert field(due_last, "invoice_date") == "18 September 2026"


# --- Labels inside a line: subtotal, tax and total ---------------------------


@pytest.mark.parametrize(
    ("line", "name", "expected"),
    [
        ("; Subtotal 14,702.00", "subtotal", "14702.00"),
        (". Total due 17,348.36", "total_amount", "17348.36"),
        (". , GST 2,646.36 .", "tax_amount", "2646.36"),
        ("Qty 3 Subtotal 14,702.00", "subtotal", "14702.00"),
        ("Qty 3 Grand Total Rs. 17,348.36", "total_amount", "17348.36"),
        ("Qty 3 Amount Due 117.00", "total_amount", "117.00"),
        ("Qty 3 Sales Tax 17.00", "tax_amount", "17.00"),
        ("Qty 3 VAT 17.00", "tax_amount", "17.00"),
        ("Qty 3 Total Amount 117.00", "total_amount", "117.00"),
    ],
)
def test_money_label_inside_a_line_or_after_stray_punctuation(
    line: str, name: str, expected: str
) -> None:
    assert field(line, name) == expected


@pytest.mark.parametrize(
    ("line", "name"),
    [
        ("Item total 100.00", "total_amount"),  # A bare "Total" inside a line is a line total
        ("Sub total 100.00 Qty", "total_amount"),
        ("Late Tax 17.00", "tax_amount"),  # A bare "Tax" inside a line is too ambiguous
        ("Seller VAT 1234567", "tax_amount"),  # No decimals: could be a registration number
        ("Seller GST 29", "tax_amount"),
        ("Customer Tax no.: SPECIMEN-TAX-0002 .", "tax_amount"),
        ("Qty 3 Subtotal 14,702", "subtotal"),  # Inside a line, amounts need two decimals
        ("the subtotal, GST and total add", "subtotal"),
        ("the subtotal, GST and total add", "tax_amount"),
        ("the subtotal, GST and total add", "total_amount"),
        ("Qty 3 Total due within 30 days", "total_amount"),
    ],
)
def test_ambiguous_money_text_inside_a_line_returns_none(line: str, name: str) -> None:
    assert field(line, name) is None


def test_total_due_after_stray_punctuation_is_not_a_subtotal_or_tax() -> None:
    result = extract_invoice_fields(". Total due 17,348.36")

    assert result["total_amount"] == "17348.36"
    assert result["subtotal"] is None
    assert result["tax_amount"] is None


# --- Tax lines with a rate: "GST at 18%" ------------------------------------


@pytest.mark.parametrize(
    "line",
    [
        "GST at 18% 2,646.36",
        "GST at 18.0% 2,646.36",
        "GST AT 18% 2,646.36",
        "GST @ 18% 2,646.36",
        "GST @18% 2,646.36",
        "GST at (18%) 2,646.36",
        "GST at 18%: 2,646.36",
        "GST at 18% Rs. 2,646.36",
    ],
)
def test_tax_rate_with_the_word_at_is_not_read_as_the_amount(line: str) -> None:
    assert field(line, "tax_amount") == "2646.36"


@pytest.mark.parametrize("line", ["GST at 18%", "GST at 18% only", "GST at 18", "GST at %", "Tax at 5%"])
def test_tax_rate_without_an_amount_returns_none(line: str) -> None:
    assert field(line, "tax_amount") is None


def test_tax_rate_line_followed_by_a_total_line_does_not_borrow_the_total() -> None:
    result = extract_invoice_fields("GST at 18%\nTotal due 100.00")

    assert result["tax_amount"] is None
    assert result["total_amount"] == "100.00"


# --- OCR currency errors: a leading "%" --------------------------------------


@pytest.mark.parametrize(
    ("line", "name", "expected"),
    [
        ("Subtotal %14,702.00", "subtotal", "14702.00"),
        ("Subtotal: %14,702.00", "subtotal", "14702.00"),
        ("Sales Tax: %2,646.36", "tax_amount", "2646.36"),
        ("Grand Total %17,348.36", "total_amount", "17348.36"),
        ("Total %0.50", "total_amount", "0.50"),
        ("Total %1,23,456.00", "total_amount", "123456.00"),  # Indian digit grouping
        ("Tax 18% %2,646.36", "tax_amount", "2646.36"),  # The rate is skipped, not the amount
        ("Tax (18%) %2,646.36", "tax_amount", "2646.36"),
    ],
)
def test_percent_sign_directly_before_a_printed_amount_is_treated_as_a_currency_symbol(
    line: str, name: str, expected: str
) -> None:
    assert field(line, name) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "18%",  # A rate
        "100.00%",  # A percent after the number
        "100%",
        "% 100.00",  # Detached from the number
        "%100",  # No decimals: could be a rate
        "%18",
        "%100.5",
        "%100.000",
        "%%100.00",
        "12%100.00",  # Percent in the middle of a number
        "%1,2.00",  # Bad digit grouping
        "%-100.00",
        "%",
        "100.00 %",
    ],
)
@pytest.mark.parametrize(
    ("label", "name"),
    [("Subtotal", "subtotal"), ("Tax", "tax_amount"), ("Total", "total_amount")],
)
def test_percent_sign_in_other_positions_is_never_turned_into_an_amount(
    label: str, name: str, raw: str
) -> None:
    assert field(f"{label}: {raw}", name) is None


@pytest.mark.parametrize("raw", ["*100.00", "™100.00", "‘100.00", "?100.00", "¥100.00"])
def test_other_possible_currency_misreads_are_not_guessed(raw: str) -> None:
    assert field(f"Total: {raw}", "total_amount") is None


def test_percent_sign_is_not_treated_as_a_currency_symbol_for_other_fields() -> None:
    assert field("Invoice No: %123", "invoice_number") is None
    assert field("Invoice Date: %2026-10-17", "invoice_date") is None


# --- Stray punctuation around values and labels ------------------------------


@pytest.mark.parametrize(
    "line",
    [
        "Total: 100.00 .",
        "Total: 100.00 ,",
        "Total: 100.00 ; .",
        "Total: 100.00 '",
        "Total: 100.00 |",
        "Total: 100.00 *",
        "; Total: 100.00",
        ". , Total 100.00 .",
    ],
)
def test_stray_punctuation_tokens_around_a_labelled_amount_are_ignored(line: str) -> None:
    assert field(line, "total_amount") == "100.00"


@pytest.mark.parametrize(
    "line",
    [
        "Total: 100.00.",  # Punctuation attached to the number is still malformed
        "Total: 100.00,",
        "Total: 100.00 abc",
        "Total: 100.00 200.00",
        "Total: 100.00 .5",
    ],
)
def test_attached_or_meaningful_trailing_text_is_still_rejected(line: str) -> None:
    assert field(line, "total_amount") is None


def test_a_trailing_currency_symbol_is_still_a_valid_existing_format() -> None:
    assert field("Total: 100.00 $", "total_amount") == "100.00"


# --- Vendor name with a title on the same line -------------------------------


@pytest.mark.parametrize(
    ("first_line", "expected"),
    [
        ("Larkfield Stationers TAX INVOICE", "Larkfield Stationers"),
        ("Larkfield Stationers  TAX   INVOICE", "Larkfield Stationers"),
        ("Larkfield Stationers INVOICE", "Larkfield Stationers"),
        ("Larkfield Stationers SALES INVOICE", "Larkfield Stationers"),
        ("Ahmed & Sons (Pvt.) Ltd. COMMERCIAL INVOICE", "Ahmed & Sons (Pvt.) Ltd."),
        ("Café Müller GmbH PROFORMA INVOICE", "Café Müller GmbH"),
        ("Larkfield Stationers RECEIPT", "Larkfield Stationers"),
        ("ACME TRADERS TAX INVOICE", "ACME TRADERS"),  # Whole line in capitals, two-word title
    ],
)
def test_generic_title_at_the_end_of_the_vendor_line_is_removed(
    first_line: str, expected: str
) -> None:
    assert field(f"{first_line}\nInvoice No: INV-1", "vendor_name") == expected


@pytest.mark.parametrize(
    "first_line",
    [
        "Fast Invoice",  # Title case: could be a company name
        "Pro Bill",
        "Receipt Rolls Ltd",  # Title word is not at the end
        "Invoice Solutions Ltd",
        "ACME INVOICE",  # One-word title with nothing to tell it from a name
        "Quick Estimate Co",
    ],
)
def test_title_like_words_inside_a_vendor_name_are_kept(first_line: str) -> None:
    assert field(f"{first_line}\nInvoice No: INV-1", "vendor_name") == first_line


def test_title_removal_only_applies_when_a_name_remains() -> None:
    assert field("TAX INVOICE\nInvoice No: INV-1", "vendor_name") is None
    assert field("INVOICE\nAcme Corp\nInvoice No: INV-1", "vendor_name") == "Acme Corp"


def test_vendor_with_a_title_suffix_is_not_reported_without_confirming_fields() -> None:
    assert field("Larkfield Stationers TAX INVOICE\nThank you", "vendor_name") is None


def test_vendor_is_confirmed_by_a_label_inside_a_later_line() -> None:
    text = "Larkfield Stationers TAX INVOICE\n22 Example Market Invoice no. INV-9"

    assert field(text, "vendor_name") == "Larkfield Stationers"


def test_vendor_is_not_confirmed_by_a_sentence_that_mentions_a_label() -> None:
    text = "Larkfield Stationers\nPlease quote the invoice number with your payment."

    assert field(text, "vendor_name") is None


# --- Vendor name: never the customer or a field line -------------------------


@pytest.mark.parametrize(
    "text",
    [
        "BILL TO\nExample Traders (fictional)\nInvoice No: INV-1",
        "INVOICE\nBILL TO Example Traders\nInvoice No: INV-1",
        "Bill To Example Traders\nInvoice No: INV-1",
        "Larkfield Stationers TAX INVOICE BILL TO Example Traders\nInvoice No: INV-1",
        "Larkfield Stationers Ship To Example Traders\nInvoice No: INV-1",
    ],
)
def test_customer_named_after_bill_to_is_never_the_vendor(text: str) -> None:
    assert field(text, "vendor_name") is None


@pytest.mark.parametrize("vendor", ["Billing Services Ltd", "Bill Tomlinson & Co", "Soldo Traders"])
def test_vendor_names_that_only_resemble_a_customer_label_are_kept(vendor: str) -> None:
    assert field(f"{vendor}\nInvoice No: INV-1", "vendor_name") == vendor


def test_first_line_holding_an_invoice_number_is_not_returned_as_the_vendor() -> None:
    text = ". A fictional company Invoice no. INV-2026-0142\nInvoice Date: 2026-10-17"

    assert extract_invoice_fields(text) == {
        **all_none(),
        "invoice_number": "INV-2026-0142",
        "invoice_date": "2026-10-17",
    }


def test_first_line_holding_an_invoice_date_is_not_returned_as_the_vendor() -> None:
    text = "Sampleton Invoice date 18 September 2026\nInvoice No: INV-1"

    assert field(text, "vendor_name") is None


# --- Three test invoices: real OCR output at PSM 6 and PSM 4 -----------------


def extract_from_ocr(raw_text: str) -> dict[str, str | None]:
    """Clean raw OCR text like the pipeline does, then extract the fields."""
    return extract_invoice_fields(clean_ocr_text(raw_text))


@pytest.mark.parametrize(
    ("raw_text", "expected"),
    [
        (NORTHSTAR_PSM4, NORTHSTAR_EXPECTED),
        (BLUEPEAK_PSM4, BLUEPEAK_EXPECTED),
        (BLUEPEAK_PSM6, BLUEPEAK_EXPECTED),
        (GREENFIELD_PSM4, GREENFIELD_EXPECTED),
    ],
    ids=["northstar-psm4", "bluepeak-psm4", "bluepeak-psm6", "greenfield-psm4"],
)
def test_invoices_whose_ocr_text_has_every_label_extract_all_six_fields(
    raw_text: str, expected: dict[str, str]
) -> None:
    assert extract_from_ocr(raw_text) == expected


def test_northstar_psm6_keeps_missing_fields_as_none_instead_of_guessing() -> None:
    # The invoice number label is corrupted ("ice No"), the subtotal row is absent
    # and the total has no label: none of these may be guessed.
    assert extract_from_ocr(NORTHSTAR_PSM6) == {
        **NORTHSTAR_EXPECTED,
        "invoice_number": None,
        "subtotal": None,
        "total_amount": None,
    }


def test_greenfield_psm6_extracts_only_what_the_text_supports() -> None:
    # The header table is missing from this OCR text and the total has no label.
    assert extract_from_ocr(GREENFIELD_PSM6) == {
        **GREENFIELD_EXPECTED,
        "invoice_number": None,
        "invoice_date": None,
        "total_amount": None,
    }


@pytest.mark.parametrize(
    "raw_text",
    [NORTHSTAR_PSM4, NORTHSTAR_PSM6, BLUEPEAK_PSM4, GREENFIELD_PSM4, GREENFIELD_PSM6],
)
def test_sample_invoices_never_return_a_wrong_value(raw_text: str) -> None:
    expected_by_vendor = {
        "Northstar": NORTHSTAR_EXPECTED,
        "BluePeak": BLUEPEAK_EXPECTED,
        "Greenfield": GREENFIELD_EXPECTED,
    }
    expected = next(value for key, value in expected_by_vendor.items() if key in raw_text)

    result = extract_from_ocr(raw_text)

    assert list(result) == EXPECTED_FIELD_NAMES
    for name, value in result.items():
        assert value is None or value == expected[name]


# --- Pipe-separated table cells ----------------------------------------------


@pytest.mark.parametrize(
    ("line", "name", "expected"),
    [
        ("| Subtotal | 56,500.00", "subtotal", "56500.00"),
        ("| Subtotal | 56,500.00 |", "subtotal", "56500.00"),
        ("Subtotal | 56,500.00", "subtotal", "56500.00"),
        ("|Subtotal|56,500.00", "subtotal", "56500.00"),
        ("| GST (18%) | 10,170.00", "tax_amount", "10170.00"),
        ("Please quote the invoice number. | GST (18%) | 10,170.00", "tax_amount", "10170.00"),
        ("| Sales Tax | Rs. 17.00", "tax_amount", "17.00"),
        ("| Total | 66,670.00", "total_amount", "66670.00"),
        ("Bank: Example | A/C No: 0011 2233 4455 | Total | 66,670.00", "total_amount", "66670.00"),
        ("Cheques payable to: X | Total PKR 239,400.00", "total_amount", "239400.00"),
        ("| Grand Total | $ 1,000", "total_amount", "1000"),
        ("| Invoice No. | GS-26-000731", "invoice_number", "GS-26-000731"),
        ("| Invoice No. | GS-26-000731 |", "invoice_number", "GS-26-000731"),
        ("Plot 21 | Invoice No: INV-2026-00125", "invoice_number", "INV-2026-00125"),
        ("; | Date | 12-Oct-2026", "invoice_date", "12-Oct-2026"),
        ("| Invoice Date | 2026-10-17", "invoice_date", "2026-10-17"),
    ],
)
def test_label_and_value_in_separate_pipe_cells(line: str, name: str, expected: str) -> None:
    assert field(line, name) == expected


@pytest.mark.parametrize(
    ("line", "name"),
    [
        ("| Total | abc", "total_amount"),
        ("| Total | 1 | 2", "total_amount"),
        ("| Total | 100.00 | 200.00", "total_amount"),
        ("| Total |", "total_amount"),
        ("| Total ||", "total_amount"),
        ("|| Total || 5", "total_amount"),
        ("| Total | Subtotal", "total_amount"),
        ("| Subtotal | Total | 100.00", "subtotal"),
        ("| Total Qty | 105", "total_amount"),
        ("Items: 7 | Total Qty: 105 | Delivery: Same day", "total_amount"),
        ("| Total | 12,50.00", "total_amount"),
        ("| Total | -100.00", "total_amount"),
        ("| Invoice No. | INV 001", "invoice_number"),
        ("| Invoice No. | ---", "invoice_number"),
        ("| Invoice No. |", "invoice_number"),
        ("| Date | 31-Feb-2026", "invoice_date"),
        ("| Date | tomorrow", "invoice_date"),
        ("| Terms | Net 30", "invoice_date"),
        ("| Due Date | 2026-11-17", "invoice_date"),
        ("Payment | Due Date | 2026-11-17", "invoice_date"),
        ("| Seller VAT | 1234567", "tax_amount"),
    ],
)
def test_malformed_or_unrelated_pipe_cells_give_no_value(line: str, name: str) -> None:
    assert field(line, name) is None


def test_pipe_cell_amounts_from_the_same_table_stay_in_their_own_fields() -> None:
    text = "| Subtotal | 56,500.00\n| GST (18%) | 10,170.00\n| Total | 66,670.00"

    assert extract_invoice_fields(text) == {
        **all_none(),
        "subtotal": "56500.00",
        "tax_amount": "10170.00",
        "total_amount": "66670.00",
    }


def test_a_pipe_inside_a_value_is_never_ignored() -> None:
    assert field("Total: 100.00 | 200.00", "total_amount") is None
    assert field("Invoice No: INV-1 | INV-2", "invoice_number") is None


# --- Label row followed by a value row ---------------------------------------


def test_label_row_and_value_row_give_number_and_issue_date_not_due_date() -> None:
    text = "INVOICE NUMBER ISSUE DATE DUE DATE\nBP/2026/0842 2026-09-15 2026-10-15"

    assert extract_invoice_fields(text) == {
        **all_none(),
        "invoice_number": "BP/2026/0842",
        "invoice_date": "2026-09-15",
    }


@pytest.mark.parametrize("separator", ["\n", "\n\n", "\n\n\n", "\n \n"])
def test_blank_lines_between_label_row_and_value_row_are_skipped(separator: str) -> None:
    text = f"INVOICE NUMBER ISSUE DATE DUE DATE{separator}BP/2026/0842 2026-09-15 2026-10-15"

    assert field(text, "invoice_number") == "BP/2026/0842"
    assert field(text, "invoice_date") == "2026-09-15"


@pytest.mark.parametrize(
    "text",
    [
        "Invoice Number Issue Date Due Date\nBP-1 2026-09-15 2026-10-15",
        "Invoice No. Invoice Date Due Date\nBP-1 2026-09-15 2026-10-15",
        "INVOICE NUMBER  ISSUE DATE   DUE DATE\nBP-1   2026-09-15   2026-10-15",
        "INVOICE # ISSUE DATE DUE DATE\nBP-1 2026-09-15 2026-10-15",
    ],
)
def test_label_row_spelling_and_spacing_variants(text: str) -> None:
    assert field(text, "invoice_number") == "BP-1"
    assert field(text, "invoice_date") == "2026-09-15"


@pytest.mark.parametrize(
    "text",
    [
        "DUE DATE ISSUE DATE INVOICE NUMBER\n2026-10-15 2026-09-15 BP-1",
        "DUE DATE INVOICE NUMBER ISSUE DATE\n2026-10-15 BP-1 2026-09-15",
    ],
)
def test_label_row_values_follow_the_label_order(text: str) -> None:
    assert field(text, "invoice_number") == "BP-1"
    assert field(text, "invoice_date") == "2026-09-15"


def test_issue_date_and_due_date_together_return_the_issue_date() -> None:
    assert field("ISSUE DATE DUE DATE\n2026-09-15 2026-10-15", "invoice_date") == "2026-09-15"
    assert field("Issue Date: 2026-09-15\nDue Date: 2026-10-15", "invoice_date") == "2026-09-15"
    assert field("Due Date: 2026-10-15\nIssue Date: 2026-09-15", "invoice_date") == "2026-09-15"


def test_due_date_alone_in_a_label_row_is_never_the_invoice_date() -> None:
    assert field("INVOICE NUMBER DUE DATE\nBP-1 2026-10-15", "invoice_date") is None
    assert field("INVOICE NUMBER DUE DATE\nBP-1 2026-10-15", "invoice_number") == "BP-1"
    assert field("TERMS DUE DATE\nNet30 2026-10-15", "invoice_date") is None


@pytest.mark.parametrize(
    "text",
    [
        "INVOICE NUMBER ISSUE DATE DUE DATE\nBP-1 2026-09-15",  # One value too few
        "INVOICE NUMBER ISSUE DATE\nBP-1 2026-09-15 2026-10-15",  # One value too many
        "INVOICE NUMBER ISSUE DATE\nBP-1 15 Sep 2026",  # A date with spaces cannot be split
        "INVOICE NUMBER SHIPPED VIA ISSUE DATE\nBP-1 DHL 2026-09-15",  # Unknown label
        "INVOICE NUMBER ISSUE DATE\n",  # No value row
        "INVOICE NUMBER ISSUE DATE\n\n\nThank you",  # Value row is text
        "INVOICE NUMBER ISSUE DATE",  # Last line
    ],
)
def test_unmatched_label_and_value_rows_give_no_values(text: str) -> None:
    assert field(text, "invoice_number") is None
    assert field(text, "invoice_date") is None


def test_label_row_values_are_validated_individually() -> None:
    text = "INVOICE NUMBER ISSUE DATE DUE DATE\nabc 2026-09-15 2026-10-15"

    assert field(text, "invoice_number") is None  # No digit
    assert field(text, "invoice_date") == "2026-09-15"
    assert field("INVOICE NUMBER ISSUE DATE\nBP-1 2026-13-45", "invoice_date") is None


def test_a_label_row_with_a_value_on_its_own_line_still_uses_the_labelled_value() -> None:
    text = "INVOICE NUMBER ISSUE DATE\nBP-1 2026-09-15\nInvoice No: A-2"

    assert field(text, "invoice_number") == "A-2"  # An explicit label has priority


@pytest.mark.parametrize(
    ("text", "name", "expected"),
    [
        ("Invoice No:\n\nINV-001", "invoice_number", "INV-001"),
        ("Total\n\nRs. 117.00", "total_amount", "117.00"),
        ("Invoice Date:\n\n2026-10-17", "invoice_date", "2026-10-17"),
        ("INVOICE NUMBER\n\nBP/2026/0842", "invoice_number", "BP/2026/0842"),
    ],
)
def test_a_single_label_may_be_separated_from_its_value_by_a_blank_line(
    text: str, name: str, expected: str
) -> None:
    assert field(text, name) == expected


@pytest.mark.parametrize(
    ("text", "name"),
    [
        ("Invoice No:\n\nThank you for your business", "invoice_number"),
        ("Total:\n\nInvoice Date: 2026-10-17", "total_amount"),
        ("Invoice No:\n\n\n", "invoice_number"),
    ],
)
def test_a_blank_line_does_not_make_unrelated_text_a_value(text: str, name: str) -> None:
    assert field(text, name) is None


# --- Issue Date label --------------------------------------------------------


@pytest.mark.parametrize("label", ["Issue Date", "ISSUE DATE", "issue date", "Issue  Date"])
def test_issue_date_is_an_invoice_date_label(label: str) -> None:
    assert field(f"{label}: 2026-09-15", "invoice_date") == "2026-09-15"
    assert field(f"{label} 2026-09-15", "invoice_date") == "2026-09-15"


def test_issue_date_inside_a_line_is_an_invoice_date() -> None:
    assert field("Plot 21, Lahore Issue Date: 15 September 2026", "invoice_date") == (
        "15 September 2026"
    )


def test_invoice_date_label_has_priority_over_issue_date() -> None:
    assert field("Issue Date: 2026-01-01\nInvoice Date: 2026-02-02", "invoice_date") == (
        "2026-02-02"
    )
    assert field("Date: 2026-01-01\nIssue Date: 2026-02-02", "invoice_date") == "2026-02-02"


def test_issue_date_label_does_not_match_longer_words() -> None:
    assert field("Issued Date: 2026-09-15", "invoice_date") is None
    assert field("Issue Dates: 2026-09-15", "invoice_date") is None


# --- Hyphenated dates --------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    ["12-Oct-2026", "5-Oct-2026", "12-October-2026", "12-oct-2026", "12-OCT-2026", "29-Feb-2028",
     "12 - Oct - 2026", "12-Sept-2026", "12-Oct.-2026"],
)
def test_hyphenated_dates_are_preserved_exactly(value: str) -> None:
    assert field(f"Date: {value}", "invoice_date") == value


@pytest.mark.parametrize(
    "value",
    ["31-Feb-2026", "29-Feb-2027", "12-Foo-2026", "12-Oct", "12-Oct-26", "12--Oct--2026",
     "Oct-12-2026", "12-Oct-2026-01", "12-Octo-2026", "-Oct-2026"],
)
def test_invalid_hyphenated_dates_return_none(value: str) -> None:
    assert field(f"Date: {value}", "invoice_date") is None


def test_existing_date_formats_still_work_after_hyphenated_dates() -> None:
    for value in ("2026-10-17", "17/10/2026", "17-10-2026", "17 Oct 2026", "17 October 2026"):
        assert field(f"Date: {value}", "invoice_date") == value


# --- Vendor name: table rows are not vendors ---------------------------------


def test_table_row_before_the_vendor_is_skipped() -> None:
    text = "| Invoice No. | GS-26-000731\n\nGreenfield Trading\n\n; | Date | 12-Oct-2026"

    assert extract_invoice_fields(text) == {
        **all_none(),
        "invoice_number": "GS-26-000731",
        "invoice_date": "12-Oct-2026",
        "vendor_name": "Greenfield Trading",
    }


@pytest.mark.parametrize(
    "first_line",
    ["| Invoice No. | GS-26-000731", "; | Date | 12-Oct-2026", "| Acme Traders |", "|Acme Traders",
     "• | Acme Traders"],
)
def test_lines_with_a_leading_pipe_are_never_the_vendor(first_line: str) -> None:
    text = f"{first_line}\nInvoice No: INV-1"

    assert field(text, "vendor_name") is None


def test_only_table_rows_before_the_fields_give_no_vendor() -> None:
    assert field("| Invoice No. | X-1\n| Date | 12-Oct-2026", "vendor_name") is None


def test_a_vendor_line_containing_a_pipe_further_along_is_unchanged() -> None:
    assert field("Acme | Traders\nInvoice No: INV-1", "vendor_name") == "Acme | Traders"


def test_a_metadata_first_line_still_gives_no_vendor() -> None:
    assert field("Invoice No: INV-1\nAcme Corp", "vendor_name") is None


# --- Tax lines with a rate in the middle of a line ---------------------------


@pytest.mark.parametrize(
    "line",
    [
        "Bank: Example Commercial Bank, Lahore | Acct 0123456789 Tax (5%) PKR 11,400.00",
        "Notes Acct 0123456789 Tax (5%) PKR 11,400.00",
        "Notes Tax 5% PKR 11,400.00",
        "Notes Tax @ 5% PKR 11,400.00",
        "Notes tax at 5% 11,400.00",
        "Tax (5%) PKR 11,400.00",
    ],
)
def test_tax_with_a_rate_is_found_inside_a_line(line: str) -> None:
    assert field(line, "tax_amount") == "11400.00"


@pytest.mark.parametrize(
    "line",
    [
        "Notes Tax (5%)",  # A rate is not an amount
        "Notes Tax 5%",
        "Notes Tax (5%) PKR 11,400",  # Inside a line, amounts need two decimals
        "Notes Tax (5%) PKR 1234567",
        "Notes Tax (5%) PKR 11,400.00 extra",
        "Notes Tax (5%) PKR 11,400.00 and 5.00",
        "Notes Tax 11,400.00",  # No rate: a bare Tax inside a line is still ignored
        "Notes Late Tax 17.00",
        "Notes Tax ID 5% 11,400.00",
        "Notes Tax: 11,400.00",
    ],
)
def test_tax_inside_a_line_without_a_clear_rate_and_amount_gives_none(line: str) -> None:
    assert field(line, "tax_amount") is None


def test_a_tax_rate_is_never_returned_as_the_tax_amount() -> None:
    for line in ("Notes Tax (5%) PKR 11,400.00", "GST at 18% 2,646.36", "| GST (18%) | 10,170.00"):
        assert field(line, "tax_amount") not in {"5", "18", "5%", "18%"}


# --- Corrupted labels are not guessed ----------------------------------------


@pytest.mark.parametrize(
    "line",
    [
        "ice No: INV-2026-00125",
        "Plot 21, Sector I-9 Industrial Area | ice No: INV-2026-00125",
        "Islamabad 44790, Pakistan noice 0: : .",
        "Police No: ABC-123",
        "Office No: 12",
        "nvoice No: INV-1",
        "Invoce Number: INV-1",
        "Invoi ce No: INV-1",
        "lnvoice No INV-1 ",
        "| ice No | INV-2026-00125",
    ],
)
def test_corrupted_invoice_number_labels_are_not_matched(line: str) -> None:
    assert field(line, "invoice_number") is None


# --- Totals are never inferred -----------------------------------------------


def test_an_unlabelled_amount_is_not_used_as_the_total() -> None:
    text = "Bank: Example Commercial Bank | A/C No: 0011 2233 4455 66,670.00"

    assert field(text, "total_amount") is None


def test_total_is_not_calculated_from_subtotal_and_tax() -> None:
    text = "| Subtotal | 56,500.00\n| GST (18%) | 10,170.00\nBank: X 66,670.00"

    result = extract_invoice_fields(text)

    assert result["subtotal"] == "56500.00"
    assert result["tax_amount"] == "10170.00"
    assert result["total_amount"] is None


def test_subtotal_and_tax_are_not_calculated_from_a_total() -> None:
    result = extract_invoice_fields("| Total | 66,670.00")

    assert result["total_amount"] == "66670.00"
    assert result["subtotal"] is None
    assert result["tax_amount"] is None


# --- Existing behaviour that must not change ----------------------------------


def test_label_row_logic_does_not_change_a_normal_invoice() -> None:
    assert extract_invoice_fields(FULL_INVOICE) == {
        "invoice_number": "INV-00123",
        "invoice_date": "2026-10-17",
        "vendor_name": "Acme Traders (Pvt.) Ltd.",
        "subtotal": "1300.00",
        "tax_amount": "221.00",
        "total_amount": "1521.00",
    }


def test_due_date_is_still_not_the_invoice_date_in_any_layout() -> None:
    assert field("Due Date: 2026-11-17", "invoice_date") is None
    assert field("Net 30 Due date 18 October 2026", "invoice_date") is None
    assert field("| Due Date | 2026-11-17", "invoice_date") is None
    assert field("DUE DATE\n2026-11-17", "invoice_date") is None


def test_subtotal_tax_and_total_are_still_kept_apart_in_pipe_tables() -> None:
    result = extract_invoice_fields("| Subtotal | 1,000.00\n| Tax | 170.00\n| Total | 1,170.00")

    assert result["subtotal"] == "1000.00"
    assert result["tax_amount"] == "170.00"
    assert result["total_amount"] == "1170.00"


def test_unrelated_numbers_next_to_pipes_are_still_ignored() -> None:
    text = "Fresh Mart\nPhone: +92 51 1234567 | NTN: 1234567-8 | Items: 7\nInvoice No: INV-77"

    assert extract_invoice_fields(text) == {
        **all_none(),
        "invoice_number": "INV-77",
        "vendor_name": "Fresh Mart",
    }


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

# --- Dates: "Mon DD, YYYY" and "DD.MM.YYYY" -----------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        "Oct 03, 2026",
        "October 3, 2026",
        "Oct. 03, 2026",
        "Oct 03 2026",
        "19.10.2026",
        "1.2.2026",
        # Formats that worked before.
        "24 October 2026",
        "2026-09-15",
        "12-Oct-2026",
        "24/10/2026",
        "24-10-2026",
    ],
)
def test_supported_date_formats_are_returned_as_written(raw: str) -> None:
    assert field(f"Invoice Date: {raw}", "invoice_date") == raw
    assert field(f"Invoice Date {raw}", "invoice_date") == raw


@pytest.mark.parametrize(
    "raw",
    [
        "Feb 30, 2026",  # Not a real date
        "Octo 03, 2026",
        "Oct 03, 26",
        "Oct 32, 2026",
        "31.02.2026",
        "19.13.2026",
        "10.19.2026",  # Dotted dates are day first, so month 19 is invalid
        "19.10.26",
        "19..10.2026",
        "2026.10.19",
    ],
)
def test_invalid_or_unsupported_dates_return_none(raw: str) -> None:
    assert field(f"Invoice Date: {raw}", "invoice_date") is None


def test_date_after_a_pipe_on_the_same_line_is_found_and_due_is_ignored() -> None:
    text = "Invoice No: CS-1 | Date: 24/10/2026 | Due: 23/11/2026"

    assert field(text, "invoice_date") == "24/10/2026"
    assert field("Due Date: 23/11/2026", "invoice_date") is None
    assert field("Due: 23/11/2026", "invoice_date") is None


# --- European and international amounts ---------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("5.190,00", "5190.00"),
        ("986,10", "986.10"),
        ("6.176,10", "6176.10"),
        ("1.234,56", "1234.56"),
        ("1.234.567,89", "1234567.89"),
        ("12.345,60", "12345.60"),
        ("100,00", "100.00"),
        ("1,23", "1.23"),
        ("EUR 5.190,00", "5190.00"),
        ("5.190,00 EUR", "5190.00"),
        ("€ 986,10", "986.10"),
        # Existing styles are unchanged.
        ("56,500.00", "56500.00"),
        ("10,170.00", "10170.00"),
        ("2,430.00", "2430.00"),
        ("239,400.00", "239400.00"),
        ("1,23,456.00", "123456.00"),
        ("1,234", "1234"),
        ("2500", "2500"),
        ("0.30", "0.30"),
    ],
)
def test_european_and_international_amounts(raw: str, expected: str) -> None:
    assert field(f"Total: {raw}", "total_amount") == expected
    assert field(f"Subtotal: {raw}", "subtotal") == expected


@pytest.mark.parametrize(
    "raw",
    [
        "1.250,000.00",  # Mixed styles (the Crescent OCR misread)
        "1.250.000.00",
        "6.176.10",  # The Alpine OCR misread: periods only, last group not three digits
        "1.234",  # Could be 1234 or 1.234
        "1.234.567",  # Thousands groups with no decimal comma
        "1.234,5",  # One decimal
        "1.234,567",  # Three decimals
        "1,234,56",
        "1.23,45",  # Bad thousands group
        "12.34,56",
        "1.2345,00",
        "1,250.000,00",
        "12,50.00",
        "1.234,56,78",
        "1.234,56.78",
        ",50",
        "1.,50",
    ],
)
def test_malformed_or_ambiguous_amounts_stay_none(raw: str) -> None:
    for label, name in (("Subtotal", "subtotal"), ("Tax", "tax_amount"), ("Total", "total_amount")):
        assert field(f"{label}: {raw}", name) is None


def test_european_amounts_are_decimal_strings_without_float_errors() -> None:
    result = extract_invoice_fields("Subtotal: 0,10\nTax: 0,20\nTotal: 0,30")

    assert Decimal(result["subtotal"]) + Decimal(result["tax_amount"]) == Decimal(
        result["total_amount"]
    )


def test_european_amount_inside_a_line_after_a_label() -> None:
    text = "Number format on this invoice: Net Amount EUR 5.190,00"

    assert field(text, "subtotal") == "5190.00"
    assert field("VAT 19% EUR 986,10", "tax_amount") == "986.10"
    assert field("Example: 1.234,56 = 1234 euros. Total Amount EUR 6.176,10", "total_amount") == (
        "6176.10"
    )


# --- "Net Amount" as a subtotal label -----------------------------------------


def test_net_amount_is_a_subtotal_label() -> None:
    assert field("Net Amount: USD 1,755.00", "subtotal") == "1755.00"
    assert field("NET AMOUNT 1,755.00", "subtotal") == "1755.00"
    assert field("Net Amount\n1,755.00", "subtotal") == "1755.00"


def test_subtotal_label_wins_over_net_amount() -> None:
    text = "Net Amount: 10.00\nSubtotal: 20.00"

    assert field(text, "subtotal") == "20.00"


def test_net_amount_is_never_read_as_tax_or_total() -> None:
    text = "Net Amount: 1,755.00"

    assert field(text, "tax_amount") is None
    assert field(text, "total_amount") is None


@pytest.mark.parametrize("line", ["Net 30 days", "Payment Terms Net 30", "Net Amounts", "Net: 100.00"])
def test_other_net_text_is_not_a_subtotal(line: str) -> None:
    assert field(line, "subtotal") is None


def test_net_amount_column_header_is_not_a_subtotal() -> None:
    text = "VAT Rate | Net Amount | VAT Amount |\n20% | 7,950.00 | 1,590.00 |"

    assert field(text, "subtotal") is None


# --- Several labelled fields on one pipe-separated line -----------------------

HEADER_LINE = "Invoice No: CS-INV-2026-019 | Date: 24/10/2026 | Due: 23/11/2026"


def test_each_field_on_a_pipe_separated_header_line_is_read_separately() -> None:
    result = extract_invoice_fields(HEADER_LINE)

    assert result["invoice_number"] == "CS-INV-2026-019"
    assert result["invoice_date"] == "24/10/2026"


@pytest.mark.parametrize(
    "line",
    [
        "Invoice No: CS-1 | Date: 24/10/2026 | Due: 23/11/2026",
        "Invoice No CS-1 | Date 24/10/2026 | Due 23/11/2026",
        "| Invoice No: CS-1 | Date: 24/10/2026 |",
        "Order Ref: SO-1 | Invoice No: CS-1 | Date: 24/10/2026",
        "Invoice Date: 24/10/2026 | Invoice No: CS-1",
        "Invoice No: CS-1 | Invoice Date: 24/10/2026",
        "Invoice No: CS-1 | Due: 23/11/2026 | Date: 24/10/2026",
    ],
)
def test_field_order_and_extra_cells_do_not_matter(line: str) -> None:
    result = extract_invoice_fields(line)

    assert result["invoice_number"] == "CS-1"
    assert result["invoice_date"] == "24/10/2026"


def test_a_value_never_swallows_the_next_label_or_value() -> None:
    result = extract_invoice_fields("Invoice No: INV-1 | Date: 2026-10-01")

    assert result["invoice_number"] == "INV-1"
    assert result["invoice_date"] == "2026-10-01"
    assert field("Invoice No: INV-1 | something else", "invoice_number") is None
    assert field("Invoice No: INV-1 | Date:", "invoice_number") == "INV-1"
    assert field("Invoice No: | Date: 2026-10-01", "invoice_number") is None


def test_amounts_on_one_pipe_separated_line_are_read_separately() -> None:
    result = extract_invoice_fields("Subtotal: 100.00 | Tax: 17.00 | Total: 117.00")

    assert result["subtotal"] == "100.00"
    assert result["tax_amount"] == "17.00"
    assert result["total_amount"] == "117.00"


def test_a_pipe_that_is_not_followed_by_a_label_does_not_end_the_value() -> None:
    assert field("Invoice No: INV | 1", "invoice_number") is None


# --- Label row over a value row: totals ---------------------------------------

TOTALS_LABELS = "Net Amount Tax Amount Due"
TOTALS_VALUES = "USD 1755.00 USD 105.30 USD 1860.30"


def totals(text: str) -> tuple[str | None, str | None, str | None]:
    result = extract_invoice_fields(text)
    return result["subtotal"], result["tax_amount"], result["total_amount"]


@pytest.mark.parametrize(
    "text",
    [
        f"{TOTALS_LABELS}\n{TOTALS_VALUES}",
        f"{TOTALS_LABELS}\n\n{TOTALS_VALUES}",
        f"{TOTALS_LABELS}\n\n\n\n{TOTALS_VALUES}",
        f"  {TOTALS_LABELS}  \n   {TOTALS_VALUES}  ",
        "Net   Amount    Tax     Amount   Due\nUSD  1755.00   USD   105.30  USD   1860.30",
        "Net Amount | Tax | Amount Due\nUSD 1755.00 | USD 105.30 | USD 1860.30",
        "| Net Amount | Tax | Amount Due |\n| USD 1755.00 | USD 105.30 | USD 1860.30 |",
        "Net Amount | Tax | Amount Due\nUSD 1755.00 USD 105.30 USD 1860.30",
        "Net Amount Tax Amount Due\n$1755.00 $105.30 $1860.30",
        "Net Amount Tax Amount Due\n1755.00 105.30 1860.30",
        "NET AMOUNT TAX AMOUNT DUE\nusd 1755.00 usd 105.30 usd 1860.30",
        "Item Description Quantity\nTax rate applied: 6%\n"
        f"{TOTALS_LABELS}\n{TOTALS_VALUES}\nTax rate applied: 6%",
    ],
)
def test_totals_row_is_mapped_by_position(text: str) -> None:
    assert totals(text) == ("1755.00", "105.30", "1860.30")


@pytest.mark.parametrize(
    ("labels", "values", "expected"),
    [
        ("Subtotal Tax Total", "100.00 17.00 117.00", ("100.00", "17.00", "117.00")),
        ("Subtotal | VAT | Grand Total", "Rs. 100.00 | Rs. 17.00 | Rs. 117.00", ("100.00", "17.00", "117.00")),
        ("Net Amount VAT Total Due", "EUR 5.190,00 EUR 986,10 EUR 6.176,10", ("5190.00", "986.10", "6176.10")),
        ("Subtotal GST Total Amount", "PKR 228,000.00 PKR 11,400.00 PKR 239,400.00", ("228000.00", "11400.00", "239400.00")),
    ],
)
def test_other_totals_rows(labels: str, values: str, expected: tuple[str, str, str]) -> None:
    assert totals(f"{labels}\n{values}") == expected


@pytest.mark.parametrize(
    "text",
    [
        f"{TOTALS_LABELS}\nUSD 1755.00 USD 105.30",  # One value missing
        f"{TOTALS_LABELS}\nUSD 1755.00 USD 105.30 USD 1860.30 USD 5.00",  # One value too many
        f"{TOTALS_LABELS} Notes\n{TOTALS_VALUES}",  # Unknown cell: not a label row
        f"{TOTALS_LABELS}\n\n",  # No value row
        "Net Amount Tax\nUSD 1755.00",
    ],
)
def test_totals_row_with_the_wrong_number_of_values_or_cells_is_not_used(text: str) -> None:
    assert totals(text) == (None, None, None)


def test_a_value_that_is_not_an_amount_is_not_used() -> None:
    assert totals("Net Amount Tax Amount Due\nabc USD 105.30 USD 1860.30") == (
        None,
        "105.30",
        "1860.30",
    )


def test_ambiguous_label_row_is_skipped_rather_than_guessed() -> None:
    # "Tax Amount Due" can be Tax + Amount Due, but with two values both
    # "Net Amount" + "Tax" and "Net Amount" + "Tax Amount Due"-less readings are
    # not possible, so only the reading with two cells is used.
    assert totals("Net Amount Tax\n1755.00 105.30") == ("1755.00", "105.30", None)


def test_labels_in_a_value_row_position_are_never_read_as_a_header_row_twice() -> None:
    text = f"{TOTALS_LABELS}\n{TOTALS_VALUES}\nSubtotal: 9.00"

    assert totals(text)[0] == "9.00"  # An explicit "Subtotal:" line takes priority


def test_invoice_number_and_date_row_still_works_with_pipes_and_currency_free_values() -> None:
    text = "INVOICE NUMBER ISSUE DATE DUE DATE\nBP/2026/0842 2026-09-15 2026-10-15"
    result = extract_invoice_fields(text)

    assert result["invoice_number"] == "BP/2026/0842"
    assert result["invoice_date"] == "2026-09-15"


def test_header_row_with_pipes_allows_dates_with_spaces() -> None:
    text = "Invoice No | Invoice Date | Due Date\nHV-1 | Oct 03, 2026 | Nov 02, 2026"
    result = extract_invoice_fields(text)

    assert result["invoice_number"] == "HV-1"
    assert result["invoice_date"] == "Oct 03, 2026"


# --- Vendor name is not taken from table headers or label rows ---------------


def test_vendor_is_not_taken_from_a_table_header_or_totals_row() -> None:
    text = "\n".join(
        [
            "Harborview Wholesale",
            "Invoice No. HV-1",
            "Item Description Quantity Rate (USD) Amount (USD)",
            TOTALS_LABELS,
            TOTALS_VALUES,
        ]
    )

    assert field(text, "vendor_name") == "Harborview Wholesale"


# --- "Total Payable", "Balance Due", "Taxable Amount" -------------------------


@pytest.mark.parametrize(
    "line",
    [
        "TOTAL PAYABLE $6,910.06",
        "Total Payable: $6,910.06",
        "Balance Due GBP 6,910.06",
        "Balance Due: 6,910.06",
    ],
)
def test_total_payable_and_balance_due_are_total_labels(line: str) -> None:
    assert field(line, "total_amount") == "6910.06"


def test_total_wins_over_balance_due_when_both_are_present() -> None:
    text = "Total 1,000.00\nBalance Due 400.00"

    assert field(text, "total_amount") == "1000.00"


def test_grand_total_wins_over_total_payable() -> None:
    text = "Total Payable 900.00\nGrand Total 1,000.00"

    assert field(text, "total_amount") == "1000.00"


def test_balance_due_is_not_read_as_a_subtotal_or_tax() -> None:
    fields = extract_invoice_fields("Balance Due GBP 3,029.94")

    assert fields["subtotal"] is None
    assert fields["tax_amount"] is None


def test_taxable_amount_is_the_subtotal_and_not_the_tax() -> None:
    fields = extract_invoice_fields("Taxable Amount GBP 2,524.95\nVAT (20%) GBP 504.99")

    assert fields["subtotal"] == "2524.95"
    assert fields["tax_amount"] == "504.99"


def test_subtotal_label_still_wins_over_taxable_amount() -> None:
    text = "Taxable Amount 900.00\nSubtotal 1,000.00"

    assert field(text, "subtotal") == "1000.00"


# --- Date label sharing a line with another column ----------------------------


def test_date_on_the_next_line_when_the_label_line_holds_another_column() -> None:
    text = "INVOICE DATE Item Qty Price (GBP) Amount (GBP)\n05 OCT 2026\nWireless Router 5 189.99"

    assert field(text, "invoice_date") == "05 OCT 2026"


def test_due_date_next_to_a_column_is_never_the_invoice_date() -> None:
    text = "DUE DATE ;\n19 OCT 2026 24-Port Network Switch 3 245.00 735.00"

    assert field(text, "invoice_date") is None


def test_due_date_value_is_not_taken_when_the_invoice_date_value_is_missing() -> None:
    text = "INVOICE DATE Item Qty Price\nDUE DATE\n19 OCT 2026"

    assert field(text, "invoice_date") is None


def test_next_line_is_used_only_when_the_whole_line_is_a_date() -> None:
    text = "Invoice Date Item Qty\n05 OCT 2026 Wireless Router 5"

    assert field(text, "invoice_date") is None


def test_bare_date_label_never_takes_the_next_line() -> None:
    text = "Date Received: ____\n12 Oct 2026"

    assert field(text, "invoice_date") is None


def test_a_valid_date_on_the_label_line_wins_over_the_next_line() -> None:
    text = "Invoice Date: 01 Jan 2026\n05 OCT 2026"

    assert field(text, "invoice_date") == "01 Jan 2026"


def test_other_fields_do_not_take_their_value_from_the_next_line_when_it_is_malformed() -> None:
    assert field("Invoice No. Item Qty\nSEL-2610-A0733", "invoice_number") is None
    assert field("Subtotal Item Qty\n$6,458.00", "subtotal") is None


# --- Vendor with a logo read as a stray bracket token -------------------------


@pytest.mark.parametrize(
    ("first_line", "vendor"),
    [
        ("M1) Mapleline Furniture INVOICE", "Mapleline Furniture"),
        ("iM) Mapleline Furniture", "Mapleline Furniture"),
        ("M] Mapleline Furniture", "Mapleline Furniture"),
        ("(M) Mapleline Furniture", "(M) Mapleline Furniture"),
        ("Acme (Pvt) Ltd", "Acme (Pvt) Ltd"),
    ],
)
def test_stray_bracket_token_before_the_vendor_name_is_removed(
    first_line: str, vendor: str
) -> None:
    text = f"{first_line}\nInvoice No: INV-1\nInvoice Date: 2026-01-02"

    assert field(text, "vendor_name") == vendor
