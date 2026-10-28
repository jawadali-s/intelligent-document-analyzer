"""Rule-based invoice field extraction for the Intelligent Image Document Analyzer.

The public entry point is :func:`extract_invoice_fields`. It reads already
cleaned OCR text and returns a dictionary with exactly these keys, in this
order. A field that cannot be identified reliably is ``None``.

* ``invoice_number`` - string, exactly as written (leading zeros kept).
* ``invoice_date``   - string, exactly as written; only plausibility-checked.
* ``vendor_name``    - string, from an explicit ``FROM:`` line, otherwise the
  first line after any generic title, decorative line or table row, without a
  generic title written at its end (``Acme Ltd TAX INVOICE`` gives ``Acme Ltd``).
* ``subtotal``, ``tax_amount``, ``total_amount`` - decimal strings such as
  ``"2500.00"``: no currency symbol, no thousands separators, decimal places
  kept as written. Convert with ``decimal.Decimal`` (never ``float``).

Rules shared by all labelled fields:

* A label normally starts a line (case-insensitive), so ``Subtotal`` is never
  read as ``Total`` and ``Due Date`` is never read as ``Date``.
* The value is the rest of the line, or the whole next line when the label
  stands alone. Lines holding several fields are not split, so such a field
  returns ``None`` rather than a guess.
* A value may follow its label after ``:`` or after a table-cell separator
  ``|``, and a label alone on its line takes the next non-empty line.
* Only if no label at the start of a line gives a valid value, a second,
  stricter search looks at labels after stray punctuation (``; Subtotal ...``),
  at labels that start a table cell (``... | Total | 66,670.00``) and, for
  specific labels only, inside a line (``... Invoice no. INV-1``).
  The value must still be the whole rest of the line, ignoring stray
  punctuation tokens such as a trailing `` .``. Inside a line, the bare labels
  ``Date``, ``Tax`` and ``Total`` are never used and amounts need two decimals.
  The exception is ``Tax`` followed by a rate (``Tax (5%) PKR 11,400.00``).
* Last, a row of labels followed by a row of values is read by position
  (``INVOICE NUMBER ISSUE DATE DUE DATE`` then ``BP-1 2026-09-15 2026-10-15``).
  The row is used only if every cell is a known label and the value count
  matches. ``Due Date`` is a known cell that is never used as the invoice date.
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
    rf"Issue\s*Date{_WORD_END}",
    rf"Bill\s*Date{_WORD_END}",
    rf"Date{_WORD_END}",
)
_SUBTOTAL_LABELS = (
    rf"Sub[\s\-]*total{_WORD_END}",
    rf"Net\s*Amount{_WORD_END}",
    rf"Taxable\s*Amount{_WORD_END}",
)
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
_TAX_RATE = r"\s*(?:at\b|@)?\s*\(?\s*\d+(?:\.\d+)?\s*%\s*\)?"
_TAX_RATE_SUFFIX = rf"(?:{_TAX_RATE})?"
_TOTAL_LABELS = (
    rf"Grand\s*Total{_WORD_END}",
    rf"Total\s*(?:Amount\s*)?Due{_WORD_END}",
    rf"Amount\s*Due{_WORD_END}",
    rf"Total\s*Amount{_WORD_END}",
    rf"Total\s*Payable{_WORD_END}",
    rf"Total{_WORD_END}",
)
# Used only when no label above gives a value: a balance can be lower than the
# total once payments are deducted, so a real "Total" always wins over it.
_TOTAL_FALLBACK_LABELS = (rf"Balance\s*Due{_WORD_END}",)

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
# European format: "." groups thousands in threes and "," starts exactly two decimals
# ("5.190,00", "986,10"). It needs the decimal comma, so "6.176" or "100.000"
# (three decimals, could be either style) and mixed forms such as "1.250,000.00"
# or "12,50.00" are never accepted.
_EUROPEAN_NUMBER = r"(?:\d{1,3}(?:\.\d{3})+|\d+),\d{2}"
_EUROPEAN_AMOUNT_RE = re.compile(
    rf"(?:{_CURRENCY}\s*)?(?P<number>{_EUROPEAN_NUMBER})(?:\s*{_CURRENCY})?",
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
_DATE_PART_SEPARATOR = r"(?:\s+|\s*-\s*)"
_TEXT_DATE_RE = re.compile(  # DD Mon YYYY, DD Month YYYY, DD-Mon-YYYY
    rf"(\d{{1,2}}){_DATE_PART_SEPARATOR}({_MONTH_NAMES})\.?,?{_DATE_PART_SEPARATOR}(\d{{4}})",
    re.IGNORECASE,
)
_DOTTED_DATE_RE = re.compile(r"(\d{1,2})\.(\d{1,2})\.(\d{4})")  # DD.MM.YYYY (day first)
_MONTH_FIRST_DATE_RE = re.compile(  # Mon DD, YYYY and Month DD, YYYY (comma optional)
    rf"({_MONTH_NAMES})\.?\s+(\d{{1,2}}),?\s+(\d{{4}})",
    re.IGNORECASE,
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

    month_first = _MONTH_FIRST_DATE_RE.fullmatch(candidate)
    if month_first is not None:
        month_text, day_text, year_text = month_first.groups()
        month = _MONTH_NUMBERS[month_text[:3].lower()]
        valid = _is_real_date(int(year_text), month, int(day_text))
        return candidate if valid else None

    dotted = _DOTTED_DATE_RE.fullmatch(candidate)
    if dotted is not None:
        day_text, month_text, year_text = dotted.groups()
        valid = _is_real_date(int(year_text), int(month_text), int(day_text))
        return candidate if valid else None

    return None


def _parse_amount(candidate: str) -> str | None:
    """Return a decimal string for a money amount, or ``None`` if malformed.

    International (``1,234.56``), Indian (``1,23,456.00``) and plain amounts are
    read as before. A European amount (``1.234,56``, ``986,10``) is read only
    when it has a decimal comma followed by exactly two digits.
    """
    match = _AMOUNT_RE.fullmatch(candidate)
    if match is not None:
        digits = match.group("number").replace(",", "")
    else:
        european = _EUROPEAN_AMOUNT_RE.fullmatch(candidate)
        if european is not None:
            digits = european.group("number").replace(".", "").replace(",", ".")
        else:
            match = _PERCENT_AMOUNT_RE.fullmatch(candidate)
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
    # The same labels in priority order, allowed to start after other text, each
    # with a flag saying whether it is specific enough to use in the middle of a
    # line. Bare labels (Date, Tax, Total) are only used where a line or table
    # cell starts. See _extract_relaxed_field.
    relaxed_labels: tuple[tuple[re.Pattern[str], bool], ...] = ()
    inside_line_parser: Callable[[str], str | None] | None = None
    # How many of the first labels may take their value from the next line when the
    # rest of their own line is not a valid value. See _extract_strict_field.
    next_line_label_count: int = 0


def _compile_labels(labels: tuple[str, ...], suffix: str = "") -> tuple[re.Pattern[str], ...]:
    """Compile label patterns, keeping their priority order."""
    return tuple(re.compile(rf"(?:{label}){suffix}", re.IGNORECASE) for label in labels)


def _compile_relaxed_labels(
    labels: tuple[str, ...],
    inside_line_count: int,
    suffix: str = "",
    extra_inside_line: tuple[str, ...] = (),
) -> tuple[tuple[re.Pattern[str], bool], ...]:
    """Compile labels that may start after other text, but not inside a word.

    The first ``inside_line_count`` labels (the specific, highest-priority ones)
    may be used in the middle of a line. ``extra_inside_line`` adds further
    patterns, tried last, that may also be used in the middle of a line.
    """
    relaxed = [
        (
            re.compile(rf"(?<![A-Za-z0-9])(?:{label}){suffix}", re.IGNORECASE),
            rank < inside_line_count,
        )
        for rank, label in enumerate(labels)
    ]
    relaxed += [
        (re.compile(rf"(?<![A-Za-z0-9])(?:{label})", re.IGNORECASE), True)
        for label in extra_inside_line
    ]
    return tuple(relaxed)


_FIELD_SPECS = {
    "invoice_number": _FieldSpec(
        _compile_labels(_INVOICE_NUMBER_LABELS),
        _parse_invoice_number,
        strip_hash=True,
        relaxed_labels=_compile_relaxed_labels(
            _INVOICE_NUMBER_LABELS, len(_INVOICE_NUMBER_LABELS)
        ),
    ),
    "invoice_date": _FieldSpec(
        _compile_labels(_DATE_LABELS),
        _parse_date,
        # Not the bare "Date" inside a line: "Due date" ends in it.
        relaxed_labels=_compile_relaxed_labels(_DATE_LABELS, len(_DATE_LABELS) - 1),
        # Not the bare "Date": "Date Received: ____" must not take the next line.
        next_line_label_count=len(_DATE_LABELS) - 1,
    ),
    "subtotal": _FieldSpec(
        _compile_labels(_SUBTOTAL_LABELS),
        _parse_amount,
        relaxed_labels=_compile_relaxed_labels(_SUBTOTAL_LABELS, len(_SUBTOTAL_LABELS)),
        inside_line_parser=_parse_printed_amount,
    ),
    "tax_amount": _FieldSpec(
        _compile_labels(_TAX_LABELS, _TAX_RATE_SUFFIX),
        _parse_amount,
        # Not the bare "Tax" inside a line, unless a rate follows it: "Tax (5%) PKR 11,400.00".
        relaxed_labels=_compile_relaxed_labels(
            _TAX_LABELS,
            len(_TAX_LABELS) - 1,
            _TAX_RATE_SUFFIX,
            extra_inside_line=(rf"Tax{_WORD_END}{_TAX_RATE}",),
        ),
        inside_line_parser=_parse_printed_amount,
    ),
    "total_amount": _FieldSpec(
        _compile_labels(_TOTAL_LABELS + _TOTAL_FALLBACK_LABELS),
        _parse_amount,
        # Not the bare "Total" inside a line: "Sub total" ends in it.
        relaxed_labels=_compile_relaxed_labels(
            _TOTAL_LABELS,
            len(_TOTAL_LABELS) - 1,
            extra_inside_line=_TOTAL_FALLBACK_LABELS,
        ),
        inside_line_parser=_parse_printed_amount,
    ),
}


def _drop_leading_hash(text: str) -> str:
    """Remove one leading '#' (and following spaces), if present."""
    return text[1:].lstrip() if text.startswith("#") else text


def _value_after_label(line: str, label_end: int, strip_hash: bool) -> str:
    """Return the text after a label and its optional ':' separator."""
    remainder = line[label_end:].lstrip()
    if remainder.startswith((":", "\uff1a", "|")):  # "|" separates table cells
        remainder = remainder[1:].lstrip()
    if strip_hash:
        remainder = _drop_leading_hash(remainder)
    return remainder.strip()


# A table cell that starts a new labelled field: "Date: ..." or "Due: ...".
_NEXT_CELL_LABEL_RE = re.compile(r"[A-Za-z][A-Za-z .#/&()\-]{0,30}?\s*[:\uff1a]")
# "Due" alone ends a value too (``Date 24/10/2026 | Due 23/11/2026``). It is not a
# label row cell, because "Tax Amount Due" must stay readable as Tax + Amount Due.
_BARE_DUE_RE = re.compile(rf"Due{_WORD_END}", re.IGNORECASE)


def _starts_with_label(text: str) -> bool:
    """Return True if the text starts with a field label or a ``Label:`` cell."""
    if _NEXT_CELL_LABEL_RE.match(text) is not None or _BARE_DUE_RE.match(text) is not None:
        return True
    return any(
        label.match(text) is not None
        for spec in _FIELD_SPECS.values()
        for label in spec.labels
    ) or any(pattern.match(text) is not None for _, pattern in _HEADER_CELLS)


def _parse_up_to_next_cell(value_text: str, parser: Callable[[str], str | None]) -> str | None:
    """Parse only the text before the next ``|`` that starts another labelled cell.

    Used when the whole rest of the line is not a valid value because other
    fields follow it: ``CS-1 | Date: 24/10/2026 | Due: 23/11/2026``. The value
    may never swallow the next label, and a ``|`` that is not followed by a
    label does not end it.
    """
    position = value_text.find("|")
    while position != -1:
        if _starts_with_label(value_text[position + 1 :].lstrip()):
            head = _TRAILING_NOISE_RE.sub("", value_text[:position].strip())
            return parser(head) if head else None
        position = value_text.find("|", position + 1)
    return None


def _first_alphanumeric_index(line: str) -> int:
    """Return the index of the first letter or digit in a line, or -1 if none."""
    for index, character in enumerate(line):
        if character.isalnum():
            return index
    return -1


def _starts_table_cell(line: str, position: int) -> bool:
    """Return True if only a ``|`` separates ``position`` from the previous text."""
    return line[:position].rstrip().endswith("|")


def _relaxed_value_in_line(
    line: str, label: re.Pattern[str], inside_line: bool, spec: _FieldSpec
) -> str | None:
    """Return the valid value after one label in a line, allowing text before it.

    A label that is the first thing on the line (after any stray punctuation)
    or that starts a table cell may be any label of the field. A label further
    along must be one of the specific ones (``inside_line``), and its value must
    satisfy the stricter inside-line parser. The value is the whole rest of the
    line, minus trailing stray punctuation.
    """
    first = _first_alphanumeric_index(line)
    if first < 0:
        return None
    for match in label.finditer(line):
        at_start = match.start() == first or _starts_table_cell(line, match.start())
        if not at_start and not inside_line:
            continue
        value_text = _value_after_label(line, match.end(), spec.strip_hash)
        value_text = _TRAILING_NOISE_RE.sub("", value_text)
        if not value_text:
            continue
        parser = spec.parser if at_start else spec.inside_line_parser or spec.parser
        parsed = parser(value_text)
        if parsed is None:
            parsed = _parse_up_to_next_cell(value_text, parser)
        if parsed is not None:
            return parsed
    return None


def _extract_relaxed_field(lines: list[str], spec: _FieldSpec) -> str | None:
    """Second-chance search for labels after stray punctuation or inside a line.

    Labels are tried in priority order, and lines from the top. Unlike the main
    search, a label alone on its line does not take its value from the next line.
    """
    for label, inside_line in spec.relaxed_labels:
        for line in lines:
            value = _relaxed_value_in_line(line, label, inside_line, spec)
            if value is not None:
                return value
    return None


def _has_relaxed_value(line: str) -> bool:
    """Return True if a line holds any field label followed by a valid value."""
    return any(
        _relaxed_value_in_line(line, label, inside_line, spec) is not None
        for spec in _FIELD_SPECS.values()
        for label, inside_line in spec.relaxed_labels
    )


# --- Label row followed by a value row ----------------------------------------

# Every cell of a label row must be one of these. ``None`` marks a known cell that
# is never extracted, so "Due Date" cannot be mistaken for the invoice date. The
# ignored cells are listed first so that "Due Date" is not read as a bare "Date".
_HEADER_CELLS: tuple[tuple[str | None, re.Pattern[str]], ...] = (
    (None, re.compile(rf"Due\s*Date{_WORD_END}", re.IGNORECASE)),
    (None, re.compile(rf"Terms{_WORD_END}", re.IGNORECASE)),
    (
        None,
        re.compile(
            rf"(?:PO|Purchase\s*Order)\s*(?:No{_WORD_END}\.?|Number{_WORD_END}|#)",
            re.IGNORECASE,
        ),
    ),
    *(("invoice_number", re.compile(label, re.IGNORECASE)) for label in _INVOICE_NUMBER_LABELS),
    *(("invoice_date", re.compile(label, re.IGNORECASE)) for label in _DATE_LABELS),
    *(("subtotal", re.compile(label, re.IGNORECASE)) for label in _SUBTOTAL_LABELS),
    *(("tax_amount", re.compile(label, re.IGNORECASE)) for label in _TAX_LABELS),
    *(
        ("total_amount", re.compile(label, re.IGNORECASE))
        for label in (*_TOTAL_LABELS, *_TOTAL_FALLBACK_LABELS)
    ),
)
_CELL_SEPARATORS = " \t|"
_MAX_LABEL_ROW_PARSES = 16
# A lone currency token such as "USD" or "Rs." that belongs to the amount after it.
_CURRENCY_ONLY_RE = re.compile(_CURRENCY, re.IGNORECASE)


def _label_row_parses(line: str) -> list[tuple[str | None, ...]]:
    """Return every way to read the line as a row of known label cells.

    Cells may be separated by spaces or ``|``. A line can be read in more than
    one way ("Tax Amount Due" is ``Tax`` + ``Amount Due``, or the unknown
    ``Tax Amount`` + ``Due``), so all complete readings are returned. ``None``
    entries are known cells that are never extracted. A reading needs at least
    two cells, and every part of the line must be a known label. An empty list
    means the line is not a label row.
    """
    results: list[tuple[str | None, ...]] = []

    def walk(position: int, cells: tuple[str | None, ...]) -> None:
        if len(results) >= _MAX_LABEL_ROW_PARSES:
            return
        while position < len(line) and line[position] in _CELL_SEPARATORS:
            position += 1
        if position >= len(line):
            if len(cells) >= 2 and cells not in results:
                results.append(cells)
            return
        for field_name, pattern in _HEADER_CELLS:
            match = pattern.match(line, position)
            if match is not None and match.end() > position:
                walk(match.end(), (*cells, field_name))

    walk(0, ())
    return results


def _value_row_splits(value_row: str) -> list[list[str]]:
    """Return the ways to split a value row into one value per label cell.

    A row with ``|`` is split on it, so a value may contain spaces
    (``15 Sep 2026``). Otherwise, or if that count does not fit, the row is split
    on whitespace and a lone currency (``USD``) joins the amount after it.
    """
    splits: list[list[str]] = []
    if "|" in value_row:
        cells = [cell.strip() for cell in value_row.split("|")]
        if cells and not cells[0]:
            cells = cells[1:]
        if cells and not cells[-1]:
            cells = cells[:-1]
        splits.append(cells)
    tokens = value_row.replace("|", " ").split()
    merged: list[str] = []
    index = 0
    while index < len(tokens):
        if index + 1 < len(tokens) and _CURRENCY_ONLY_RE.fullmatch(tokens[index]):
            merged.append(f"{tokens[index]} {tokens[index + 1]}")
            index += 2
        else:
            merged.append(tokens[index])
            index += 1
    splits.append(merged)
    return splits


def _extract_from_label_rows(lines: list[str], name: str, spec: _FieldSpec) -> str | None:
    """Read ``name`` from a label row and the next non-empty line, by position.

    The value row must have exactly one value per label, and the label row must
    have only one reading with that many cells; otherwise it is skipped. Each
    value is validated like any other, and a currency written before an amount
    is not part of it.
    """
    for index, line in enumerate(lines):
        parses = _label_row_parses(line)
        if not parses:
            continue
        value_row = next((later for later in lines[index + 1 :] if later), "")
        for values in _value_row_splits(value_row):
            matching = {parse for parse in parses if len(parse) == len(values)}
            if len(matching) != 1:
                continue
            for cell, value in zip(next(iter(matching)), values):
                if cell == name:
                    parsed = spec.parser(value)
                    if parsed is not None:
                        return parsed
            break
    return None


def _extract_labelled_field(lines: list[str], name: str, spec: _FieldSpec) -> str | None:
    """Find the first valid value for a field (see the module docstring)."""
    value = _extract_strict_field(lines, spec)
    if value is None:
        value = _extract_relaxed_field(lines, spec)
    if value is None:
        value = _extract_from_label_rows(lines, name, spec)
    return value


def _extract_strict_field(lines: list[str], spec: _FieldSpec) -> str | None:
    """Find a value after a label that starts a line.

    For the first ``spec.next_line_label_count`` labels there is one more chance.
    In a side-by-side layout OCR can put the label on the same line as an unrelated
    column (``INVOICE DATE Item Qty Price``) and the value alone on the next line.
    If the rest of the label's line is not a valid value, the next non-empty line is
    used, but only when that whole line is a valid value.
    """
    for rank, label in enumerate(spec.labels):
        for index, line in enumerate(lines):
            match = label.match(line)
            if match is None:
                continue
            value_text = _value_after_label(line, match.end(), spec.strip_hash)
            if not value_text:
                # Label alone: the value is on the next non-empty line.
                value_text = next((later for later in lines[index + 1 :] if later), "")
                if spec.strip_hash:
                    value_text = _drop_leading_hash(value_text)
            if not value_text:
                continue
            parsed = spec.parser(value_text)
            if parsed is None:
                parsed = _parse_up_to_next_cell(value_text, spec.parser)
            if parsed is None and rank < spec.next_line_label_count:
                next_line = next((later for later in lines[index + 1 :] if later), "")
                parsed = spec.parser(next_line) if next_line else None
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


def _is_table_row_line(line: str) -> bool:
    """Return True if a ``|`` comes before the first letter or digit of the line."""
    first = _first_alphanumeric_index(line)
    return "|" in (line if first < 0 else line[:first])


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


# A logo or icon that OCR reads as a short token ending in a lone closing bracket:
# "M1) Mapleline Furniture". An opening bracket in the token ("(M)") keeps it.
_STRAY_BRACKET_PREFIX_RE = re.compile(r"[^\s(\[]{1,4}[)\]]\s+(?=[A-Za-z])")


def _without_stray_bracket_prefix(line: str) -> str:
    """Remove a short leading token that ends in an unmatched ``)`` or ``]``."""
    match = _STRAY_BRACKET_PREFIX_RE.match(line)
    return line[match.end() :] if match is not None else line


def _find_vendor_name(lines: list[str]) -> str | None:
    """Return the vendor name, or ``None`` if it is not clearly identifiable.

    An explicit ``FROM:`` line wins. Otherwise only the first non-empty line
    after any generic title, decorative line or table row is considered. A generic title
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
            or _is_table_row_line(line)
            or " ".join(line.lower().split()) in _GENERIC_HEADINGS
        ):
            continue
        if _OTHER_PARTY_RE.search(line) is not None or _has_relaxed_value(line):
            return None
        line = _without_stray_bracket_prefix(line)
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
        fields[name] = _extract_labelled_field(lines, name, spec)
    return fields