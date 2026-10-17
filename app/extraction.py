"""Rule-based invoice field extraction for the Intelligent Image Document Analyzer.

The public entry point is :func:`extract_invoice_fields`. It reads already
cleaned OCR text and returns a dictionary with exactly these keys, in this
order. A field that cannot be identified reliably is ``None``.

* ``invoice_number`` - string, exactly as written (leading zeros kept).
* ``invoice_date``   - string, exactly as written; only plausibility-checked.
* ``vendor_name``    - string, the first line after any generic title.
* ``subtotal``, ``tax_amount``, ``total_amount`` - decimal strings such as
  ``"2500.00"``: no currency symbol, no thousands separators, decimal places
  kept as written. Convert with ``decimal.Decimal`` (never ``float``).

Rules shared by all labelled fields:

* A label must start a line (case-insensitive), so ``Subtotal`` is never read
  as ``Total`` and ``Due Date`` is never read as ``Date``.
* The value is the rest of the line, or the whole next line when the label
  stands alone. Lines holding several fields are not split, so such a field
  returns ``None`` rather than a guess.
* If a field has several candidates, labels are tried in priority order and,
  within one label, occurrences are tried from the top of the text. The first
  valid value wins. A malformed value is skipped in favour of the next one.

This is a heuristic extractor. It makes no claim of complete accuracy.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation

FIELD_NAMES = (
    "invoice_number",
    "invoice_date",
    "vendor_name",
    "subtotal",
    "tax_amount",
    "total_amount",
)

_MAX_INVOICE_NUMBER_LENGTH = 40
_MAX_VENDOR_NAME_LENGTH = 80


class InvalidExtractionInputError(TypeError):
    """Raised when the text argument is not a ``str``."""


# --- Label patterns (tried in priority order, matched at the start of a line) -

# A label must not run into a longer word: "Total" matches, "Totals" does not.
_WORD_END = r"(?![A-Za-z])"

_INVOICE_NUMBER_LABELS = (
    rf"Invoice\s*(?:No{_WORD_END}\.?|Number{_WORD_END}|#)",
    rf"Bill\s*(?:No{_WORD_END}\.?|Number{_WORD_END}|#)",
    rf"Inv\.?\s*(?:No{_WORD_END}\.?|#)",
)
_DATE_LABELS = (
    rf"Invoice\s*Date{_WORD_END}",
    rf"Bill\s*Date{_WORD_END}",
    rf"Date{_WORD_END}",
)
_SUBTOTAL_LABELS = (rf"Sub[\s\-]*total{_WORD_END}",)
_TAX_LABELS = (
    rf"Sales\s*Tax{_WORD_END}",
    rf"Tax\s*Amount{_WORD_END}",
    rf"Total\s*Tax{_WORD_END}",
    rf"VAT{_WORD_END}",
    rf"GST{_WORD_END}",
    rf"Tax{_WORD_END}",
)
# An optional tax rate may follow a tax label: "Tax 17%", "VAT (17%)".
_TAX_RATE_SUFFIX = r"(?:\s*\(?\s*\d+(?:\.\d+)?\s*%\s*\)?)?"
_TOTAL_LABELS = (
    rf"Grand\s*Total{_WORD_END}",
    rf"Total\s*(?:Amount\s*)?Due{_WORD_END}",
    rf"Amount\s*Due{_WORD_END}",
    rf"Total\s*Amount{_WORD_END}",
    rf"Total{_WORD_END}",
)

# --- Value patterns -----------------------------------------------------------

# Optional currency before or after the number: "Rs. 2,500", "$ 10", "2500 PKR".
_CURRENCY = r"(?:Rs\.?|PKR|USD|EUR|GBP|INR|AED|SAR|[$€£₹])"
# Integer part: international grouping (1,234,567), Indian grouping (1,23,456)
# or plain digits. The decimal part, if present, has one or two digits.
_NUMBER = r"(?:\d{1,3}(?:,\d{3})+|\d{1,2}(?:,\d{2})+,\d{3}|\d+)(?:\.\d{1,2})?"
_AMOUNT_RE = re.compile(
    rf"(?:{_CURRENCY}\s*)?(?P<number>{_NUMBER})(?:\s*{_CURRENCY})?",
    re.IGNORECASE,
)

# Letters and digits, optionally joined by - _ / . and ending on a letter/digit.
_INVOICE_NUMBER_RE = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9\-_/.]*[A-Za-z0-9])?")

_MONTH_NAMES = (
    r"Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|"
    r"Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?"
)
_MONTH_NUMBERS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}
_ISO_DATE_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2})")  # YYYY-MM-DD
_NUMERIC_DATE_RE = re.compile(r"(\d{1,2})([/-])(\d{1,2})\2(\d{4})")  # DD/MM/YYYY, DD-MM-YYYY
_TEXT_DATE_RE = re.compile(  # DD Mon YYYY, DD Month YYYY
    r"(\d{1,2})\s+(" + _MONTH_NAMES + r")\.?,?\s+(\d{4})", re.IGNORECASE
)

# Lines that are document titles rather than vendor names.
_GENERIC_HEADINGS = frozenset(
    {
        "invoice",
        "tax invoice",
        "sales invoice",
        "commercial invoice",
        "proforma invoice",
        "bill",
        "receipt",
        "cash memo",
        "credit note",
        "quotation",
        "estimate",
    }
)


# --- Value parsers (each returns the value to report, or None) ---------------


def _parse_invoice_number(candidate: str) -> str | None:
    """Return ``candidate`` if it looks like an invoice number, else ``None``."""
    if len(candidate) > _MAX_INVOICE_NUMBER_LENGTH:
        return None
    if _INVOICE_NUMBER_RE.fullmatch(candidate) is None:
        return None
    if re.search(r"[0-9]", candidate) is None:  # Must contain at least one digit
        return None
    return candidate


def _is_real_date(year: int, month: int, day: int) -> bool:
    """Return True if the year, month and day form a real calendar date."""
    try:
        date(year, month, day)
    except ValueError:
        return False
    return True


def _parse_date(candidate: str) -> str | None:
    """Return ``candidate`` unchanged if it is a plausible date, else ``None``."""
    iso = _ISO_DATE_RE.fullmatch(candidate)
    if iso is not None:
        year, month, day = (int(part) for part in iso.groups())
        return candidate if _is_real_date(year, month, day) else None

    numeric = _NUMERIC_DATE_RE.fullmatch(candidate)
    if numeric is not None:
        first, _, second, year_text = numeric.groups()
        year = int(year_text)
        # The string is not converted, so accept it if either DD/MM or MM/DD is valid.
        valid = _is_real_date(year, int(second), int(first)) or _is_real_date(
            year, int(first), int(second)
        )
        return candidate if valid else None

    textual = _TEXT_DATE_RE.fullmatch(candidate)
    if textual is not None:
        day_text, month_text, year_text = textual.groups()
        month = _MONTH_NUMBERS[month_text[:3].lower()]
        valid = _is_real_date(int(year_text), month, int(day_text))
        return candidate if valid else None

    return None


def _parse_amount(candidate: str) -> str | None:
    """Return a decimal string for a money amount, or ``None`` if malformed."""
    match = _AMOUNT_RE.fullmatch(candidate)
    if match is None:
        return None
    digits = match.group("number").replace(",", "")
    try:
        amount = Decimal(digits)
    except InvalidOperation:
        return None
    return format(amount, "f")


# --- Labelled-field search ----------------------------------------------------


@dataclass(frozen=True)
class _FieldSpec:
    """How to find one labelled field."""

    labels: tuple[re.Pattern[str], ...]
    parser: Callable[[str], str | None]
    strip_hash: bool = False  # Treat a leading '#' before the value as part of the label


def _compile_labels(labels: tuple[str, ...], suffix: str = "") -> tuple[re.Pattern[str], ...]:
    """Compile label patterns, keeping their priority order."""
    return tuple(re.compile(rf"(?:{label}){suffix}", re.IGNORECASE) for label in labels)


_FIELD_SPECS = {
    "invoice_number": _FieldSpec(
        _compile_labels(_INVOICE_NUMBER_LABELS), _parse_invoice_number, strip_hash=True
    ),
    "invoice_date": _FieldSpec(_compile_labels(_DATE_LABELS), _parse_date),
    "subtotal": _FieldSpec(_compile_labels(_SUBTOTAL_LABELS), _parse_amount),
    "tax_amount": _FieldSpec(_compile_labels(_TAX_LABELS, _TAX_RATE_SUFFIX), _parse_amount),
    "total_amount": _FieldSpec(_compile_labels(_TOTAL_LABELS), _parse_amount),
}


def _drop_leading_hash(text: str) -> str:
    """Remove one leading '#' (and following spaces), if present."""
    return text[1:].lstrip() if text.startswith("#") else text


def _value_after_label(line: str, label_end: int, strip_hash: bool) -> str:
    """Return the text after a label and its optional ':' separator."""
    remainder = line[label_end:].lstrip()
    if remainder.startswith((":", "\uff1a")):
        remainder = remainder[1:].lstrip()
    if strip_hash:
        remainder = _drop_leading_hash(remainder)
    return remainder.strip()


def _extract_labelled_field(lines: list[str], spec: _FieldSpec) -> str | None:
    """Find the first valid value for a field (see the module docstring)."""
    for label in spec.labels:
        for index, line in enumerate(lines):
            match = label.match(line)
            if match is None:
                continue
            value_text = _value_after_label(line, match.end(), spec.strip_hash)
            if not value_text and index + 1 < len(lines):
                value_text = lines[index + 1]  # Label alone: value is on the next line
                if spec.strip_hash:
                    value_text = _drop_leading_hash(value_text)
            if not value_text:
                continue
            parsed = spec.parser(value_text)
            if parsed is not None:
                return parsed
    return None


# --- Vendor name --------------------------------------------------------------


def _looks_like_field_line(line: str) -> bool:
    """Return True if the line starts with a field label and has no other text.

    The rest of the line must be empty or a valid value for that field, so a
    vendor called "Date Palm Trading Co" is not mistaken for a date label.
    """
    for spec in _FIELD_SPECS.values():
        for label in spec.labels:
            match = label.match(line)
            if match is None:
                continue
            value_text = _value_after_label(line, match.end(), spec.strip_hash)
            if not value_text or spec.parser(value_text) is not None:
                return True
    return False


def _is_plausible_vendor(line: str) -> bool:
    """Return True if a line could be a vendor name."""
    if len(line) > _MAX_VENDOR_NAME_LENGTH or line[0].isdigit():
        return False
    if ":" in line or "@" in line:
        return False
    lowered = line.lower()
    if "http" in lowered or "www." in lowered:
        return False
    letters = sum(character.isalpha() for character in line)
    digits = sum(character.isdigit() for character in line)
    if letters < 2 or letters < digits:
        return False
    return not _looks_like_field_line(line)


def _find_vendor_name(lines: list[str]) -> str | None:
    """Return the vendor name, or ``None`` if it is not clearly identifiable.

    Only the first non-empty line after any generic title is considered, and
    it is reported only if labelled invoice fields follow it.
    """
    for index, line in enumerate(lines):
        if not line or " ".join(line.lower().split()) in _GENERIC_HEADINGS:
            continue
        if not _is_plausible_vendor(line):
            return None
        if any(_looks_like_field_line(later) for later in lines[index + 1 :]):
            return line
        return None
    return None


# --- Public function ----------------------------------------------------------


def extract_invoice_fields(text: str) -> dict[str, str | None]:
    """Extract invoice fields from cleaned OCR text.

    The text is only read, never modified or cleaned. Leading and trailing
    whitespace of each line is ignored while searching, and the returned vendor
    name keeps any spacing inside the line.

    Args:
        text: Cleaned invoice text, for example from
            :func:`app.text_cleaning.clean_ocr_text`.

    Returns:
        A new dictionary with the keys ``invoice_number``, ``invoice_date``,
        ``vendor_name``, ``subtotal``, ``tax_amount`` and ``total_amount``.
        Values are strings (see the module docstring for their formats) or
        ``None`` when a field is missing, malformed or ambiguous. Empty and
        whitespace-only text returns all six fields as ``None``.

    Raises:
        InvalidExtractionInputError: If ``text`` is not a ``str``. It is a
            ``TypeError`` subclass. Other types are never converted.
    """
    if not isinstance(text, str):
        raise InvalidExtractionInputError(
            f"text must be a str, got {type(text).__name__}."
        )

    fields: dict[str, str | None] = dict.fromkeys(FIELD_NAMES)
    lines = [line.strip() for line in text.split("\n")]
    if not any(lines):
        return fields

    fields["vendor_name"] = _find_vendor_name(lines)
    for name, spec in _FIELD_SPECS.items():
        fields[name] = _extract_labelled_field(lines, spec)
    return fields