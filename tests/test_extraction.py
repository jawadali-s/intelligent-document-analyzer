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