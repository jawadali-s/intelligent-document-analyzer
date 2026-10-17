"""Conservative OCR text cleaning for the Intelligent Image Document Analyzer.

The public entry point is :func:`clean_ocr_text`. It only normalizes
whitespace and line endings. It never changes letters, digits, punctuation,
currency symbols or any other visible character, so invoice identifiers,
dates and amounts keep their exact values.
"""


class InvalidTextInputError(TypeError):
    """Raised when the text argument is not a ``str``."""


def clean_ocr_text(text: str) -> str:
    """Normalize whitespace and line endings in OCR output.

    Rules:

    * ``\\r\\n`` and ``\\r`` line endings become ``\\n``.
    * Within each line, runs of whitespace (spaces, tabs, form feeds,
      non-breaking spaces and other Unicode whitespace) become a single
      space, and leading and trailing whitespace is removed.
    * Consecutive blank lines are reduced to a single blank line.
    * Leading and trailing blank lines and whitespace are removed.

    Line breaks are otherwise preserved, so a label on one line and its value
    on the next stay on separate lines. Characters that are not whitespace
    are never altered.

    Args:
        text: Raw text, for example from :func:`app.ocr.extract_text`.

    Returns:
        The cleaned text. Empty and whitespace-only input returns ``""``.
        Cleaning already-cleaned text returns it unchanged.

    Raises:
        InvalidTextInputError: If ``text`` is not a ``str``. It is a
            ``TypeError`` subclass. Other types are never converted.
    """
    if not isinstance(text, str):
        raise InvalidTextInputError(
            f"text must be a str, got {type(text).__name__}."
        )

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")

    cleaned_lines: list[str] = []
    previous_line_blank = False
    for line in normalized.split("\n"):
        cleaned_line = " ".join(line.split())
        is_blank = cleaned_line == ""
        if is_blank and previous_line_blank:
            continue
        cleaned_lines.append(cleaned_line)
        previous_line_blank = is_blank

    return "\n".join(cleaned_lines).strip()