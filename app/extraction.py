"""Rule-based invoice field extraction for the Intelligent Image Document Analyzer.

The public entry point is :func:`extract_invoice_fields`. It reads already
cleaned OCR text and returns a dictionary with exactly these keys, in this
order. A field that cannot be identified reliably is ``None``.

* ``invoice_number`` - string, exactly as written (leading zeros kept).
* ``invoice_date``   - string, exactly as written; only plausibility-checked.
* ``vendor_name``    - string, from an explicit ``FROM:`` line, otherwise the
  first line after any generic title or decorative line, without a generic
  title written at its end (``Acme Ltd TAX INVOICE`` gives ``Acme Ltd``).
* ``subtotal``, ``tax_amount``, ``total_amount`` - decimal strings such as
  ``"2500.00"``: no currency symbol, no thousands separators, decimal places
  kept as written. Convert with ``decimal.Decimal`` (never ``float``).

Rules shared by all labelled fields:

* A label normally starts a line (case-insensitive), so ``Subtotal`` is never
  read as ``Total`` and ``Due Date`` is never read as ``Date``.
* The value is the rest of the line, or the whole next line when the label
  stands alone. Lines holding several fields are not split, so such a field
  returns ``None`` rather than a guess.
* Only if no label at the start of a line gives a valid value, a second,
  stricter search looks at labels after stray punctuation (``; Subtotal ...``)
  and, for specific labels only, inside a line (``... Invoice no. INV-1``).
  The value must still be the whole rest of the line, ignoring stray
  punctuation tokens such as a trailing `` .``. Inside a line, the bare labels
  ``Date``, ``Tax`` and ``Total`` are never used, and amounts need two decimals.
* A ``%`` written directly before a printed amount with two decimals
  (``%2,646.36``) is read as an OCR misread of a currency symbol. A tax rate
  such as ``18%`` is never an amount, and no other symbol is guessed.
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

# "FROM: Studio Name". The colon is required, so "From Scratch Bakery" is not a label.
_FROM_LABEL_RE = re.compile(r"From\s*[:\N{FULLWIDTH COLON}]\s*(?P<value>.*)", re.IGNORECASE)
# Words showing that a second party (the customer) is named on the same line.
_OTHER_PARTY_RE = re.compile(r"\b(?:bill(?:ed)?|ship|sold)\s*to\b", re.IGNORECASE)
# Stray OCR punctuation standing alone at the end of a line, such as " ." or " ;".
_TRAILING_NOISE_RE = re.compile(r"(?:\s+[.,;:|'\"\u2018\u2019\u201c\u201d*~_])+$")


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
# "GST at 18%" and "GST @ 18%" are accepted too.
_TAX_RATE_SUFFIX = r"(?:\s*(?:at\b|@)?\s*\(?\s*\d+(?:\.\d+)?\s*%\s*\)?)?"
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
_INTEGER = r"(?:\d{1,3}(?:,\d{3})+|\d{1,2}(?:,\d{2})+,\d{3}|\d+)"
_NUMBER = rf"{_INTEGER}(?:\.\d{{1,2}})?"
_AMOUNT_RE = re.compile(
    rf"(?:{_CURRENCY}\s*)?(?P<number>{_NUMBER})(?:\s*{_CURRENCY})?",
    re.IGNORECASE,
)
# OCR often reads a currency symbol as "%". Accepted only when it is attached to
# the front of a printed amount with exactly two decimals, e.g. "%2,646.36".
_PERCENT_AMOUNT_RE = re.compile(rf"%(?P<number>{_INTEGER}\.\d{{2}})")

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


# "<name> <generic title>" at the end of a line, e.g. "Acme Ltd TAX INVOICE". The
# non-greedy name makes the longest title win ("TAX INVOICE", not just "INVOICE").
_TITLE_SUFFIX_RE = re.compile(
    r"(?P<name>.*?\S)\s+(?P<title>"
    + "|".join(r"\s+".join(map(re.escape, title.split())) for title in sorted(_GENERIC_HEADINGS))
    + r")",
    re.IGNORECASE,
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
    match = _AMOUNT_RE.fullmatch(candidate) or _PERCENT_AMOUNT_RE.fullmatch(candidate)
    if match is None:
        return None
    digits = match.group("number").replace(",", "")
    try:
        amount = Decimal(digits)
    except InvalidOperation:
        return None
    return format(amount, "f")


def _parse_printed_amount(candidate: str) -> str | None:
    """Like :func:`_parse_amount`, but the amount must have two decimals.

    Used for labels found inside a line, where a bare integer such as a
    registration number must not be mistaken for money.
    """
    amount = _parse_amount(candidate)
    if amount is None or re.fullmatch(r"\d+\.\d{2}", amount) is None:
        return None
    return amount


# --- Labelled-field search ----------------------------------------------------


@dataclass(frozen=True)
class _FieldSpec:
    """How to find one labelled field."""

    labels: tuple[re.Pattern[str], ...]
    parser: Callable[[str], str | None]
    strip_hash: bool = False  # Treat a leading '#' before the value as part of the label
    # The same labels, allowed to start after other text (see _extract_relaxed_field).
    relaxed_labels: tuple[re.Pattern[str], ...] = ()
    # How many of the first (highest-priority) labels are specific enough to be
    # used in the middle of a line. The remaining bare labels (Date, Tax, Total)
    # are only used at the start of a line.
    inside_line_label_count: int = 0
    inside_line_parser: Callable[[str], str | None] | None = None


def _compile_labels(labels: tuple[str, ...], suffix: str = "") -> tuple[re.Pattern[str], ...]:
    """Compile label patterns, keeping their priority order."""
    return tuple(re.compile(rf"(?:{label}){suffix}", re.IGNORECASE) for label in labels)


def _compile_relaxed_labels(
    labels: tuple[str, ...], suffix: str = ""
) -> tuple[re.Pattern[str], ...]:
    """Compile labels that may start after other text, but not inside a word."""
    return tuple(
        re.compile(rf"(?<![A-Za-z0-9])(?:{label}){suffix}", re.IGNORECASE) for label in labels
    )


_FIELD_SPECS = {
    "invoice_number": _FieldSpec(
        _compile_labels(_INVOICE_NUMBER_LABELS),
        _parse_invoice_number,
        strip_hash=True,
        relaxed_labels=_compile_relaxed_labels(_INVOICE_NUMBER_LABELS),
        inside_line_label_count=len(_INVOICE_NUMBER_LABELS),
    ),
    "invoice_date": _FieldSpec(
        _compile_labels(_DATE_LABELS),
        _parse_date,
        relaxed_labels=_compile_relaxed_labels(_DATE_LABELS),
        inside_line_label_count=len(_DATE_LABELS) - 1,  # Not bare "Date": "Due date" ends in it
    ),
    "subtotal": _FieldSpec(
        _compile_labels(_SUBTOTAL_LABELS),
        _parse_amount,
        relaxed_labels=_compile_relaxed_labels(_SUBTOTAL_LABELS),
        inside_line_label_count=len(_SUBTOTAL_LABELS),
        inside_line_parser=_parse_printed_amount,
    ),
    "tax_amount": _FieldSpec(
        _compile_labels(_TAX_LABELS, _TAX_RATE_SUFFIX),
        _parse_amount,
        relaxed_labels=_compile_relaxed_labels(_TAX_LABELS, _TAX_RATE_SUFFIX),
        inside_line_label_count=len(_TAX_LABELS) - 1,  # Not bare "Tax"
        inside_line_parser=_parse_printed_amount,
    ),
    "total_amount": _FieldSpec(
        _compile_labels(_TOTAL_LABELS),
        _parse_amount,
        relaxed_labels=_compile_relaxed_labels(_TOTAL_LABELS),
        inside_line_label_count=len(_TOTAL_LABELS) - 1,  # Not bare "Total": "Sub total"
        inside_line_parser=_parse_printed_amount,
    ),
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


def _first_alphanumeric_index(line: str) -> int:
    """Return the index of the first letter or digit in a line, or -1 if none."""
    for index, character in enumerate(line):
        if character.isalnum():
            return index
    return -1


def _relaxed_value_in_line(
    line: str, rank: int, label: re.Pattern[str], spec: _FieldSpec
) -> str | None:
    """Return the valid value after one label in a line, allowing text before it.

    A label that is the first thing on the line (after any stray punctuation)
    may be any label of the field. A label further along must be one of the
    specific ones, and its value must satisfy the stricter inside-line parser.
    The value is the whole rest of the line, minus trailing stray punctuation.
    """
    first = _first_alphanumeric_index(line)
    if first < 0:
        return None
    for match in label.finditer(line):
        at_line_start = match.start() == first
        if not at_line_start and rank >= spec.inside_line_label_count:
            continue
        value_text = _value_after_label(line, match.end(), spec.strip_hash)
        value_text = _TRAILING_NOISE_RE.sub("", value_text)
        if not value_text:
            continue
        parser = spec.parser if at_line_start else spec.inside_line_parser or spec.parser
        parsed = parser(value_text)
        if parsed is not None:
            return parsed
    return None


def _extract_relaxed_field(lines: list[str], spec: _FieldSpec) -> str | None:
    """Second-chance search for labels after stray punctuation or inside a line.

    Labels are tried in priority order, and lines from the top. Unlike the main
    search, a label alone on its line does not take its value from the next line.
    """
    for rank, label in enumerate(spec.relaxed_labels):
        for line in lines:
            value = _relaxed_value_in_line(line, rank, label, spec)
            if value is not None:
                return value
    return None


def _has_relaxed_value(line: str) -> bool:
    """Return True if a line holds any field label followed by a valid value."""
    return any(
        _relaxed_value_in_line(line, rank, label, spec) is not None
        for spec in _FIELD_SPECS.values()
        for rank, label in enumerate(spec.relaxed_labels)
    )


def _extract_labelled_field(lines: list[str], spec: _FieldSpec) -> str | None:
    """Find the first valid value for a field (see the module docstring)."""
    value = _extract_strict_field(lines, spec)
    if value is None:
        value = _extract_relaxed_field(lines, spec)
    return value


def _extract_strict_field(lines: list[str], spec: _FieldSpec) -> str | None:
    """Find a value after a label that starts a line."""
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


def _is_decorative_line(line: str) -> bool:
    """Return True if a line has no letters or digits, such as '-----' or '——'."""
    return not any(character.isalnum() for character in line)


def _find_vendor_from_label(lines: list[str]) -> str | None:
    """Return the vendor named on an explicit ``FROM:`` line, or ``None``.

    The label needs a colon, so a vendor called "From Scratch Bakery" is not
    mistaken for a label. The name must pass the usual vendor checks and must
    not mention another party ("Bill To", "Ship To", "Sold To"). If the value is
    not valid, that line is ignored and the next ``FROM:`` line is tried.
    """
    for line in lines:
        match = _FROM_LABEL_RE.fullmatch(line)
        if match is None:
            continue
        candidate = match.group("value").strip()
        if (
            candidate
            and _is_plausible_vendor(candidate)
            and _OTHER_PARTY_RE.search(candidate) is None
        ):
            return candidate
    return None


def _without_title_suffix(line: str) -> str:
    """Remove a generic title written at the end of a vendor line.

    "Acme Ltd TAX INVOICE" gives "Acme Ltd". The title must be in capitals, and
    a one-word title ("INVOICE", "RECEIPT") is removed only if the rest of the
    line has lowercase letters. So "Fast Invoice" and "ACME INVOICE" are kept:
    they could be company names. The line is returned unchanged when no clear
    title is found.
    """
    match = _TITLE_SUFFIX_RE.fullmatch(line)
    if match is None:
        return line
    title = match.group("title")
    name = match.group("name")
    if not title.isupper():
        return line
    if " " not in " ".join(title.split()) and not any(c.islower() for c in name):
        return line
    return name


def _find_vendor_name(lines: list[str]) -> str | None:
    """Return the vendor name, or ``None`` if it is not clearly identifiable.

    An explicit ``FROM:`` line wins. Otherwise only the first non-empty line
    after any generic title or decorative line is considered. A generic title
    at the end of that line is removed. The line is never used if it names
    another party ("Bill To ...") or itself holds a labelled field value, and
    it is reported only if labelled invoice fields follow it.
    """
    from_label_vendor = _find_vendor_from_label(lines)
    if from_label_vendor is not None:
        return from_label_vendor

    for index, line in enumerate(lines):
        if (
            not line
            or _is_decorative_line(line)
            or " ".join(line.lower().split()) in _GENERIC_HEADINGS
        ):
            continue
        if _OTHER_PARTY_RE.search(line) is not None or _has_relaxed_value(line):
            return None
        candidate = _without_title_suffix(line)
        if not _is_plausible_vendor(candidate):
            candidate = line  # Keep the original rule for lines the suffix rule cannot clean
            if not _is_plausible_vendor(candidate):
                return None
        if any(
            _looks_like_field_line(later) or _has_relaxed_value(later)
            for later in lines[index + 1 :]
        ):
            return candidate
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