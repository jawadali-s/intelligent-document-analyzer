"""Response schemas for the Intelligent Image Document Analyzer API."""

from typing import Literal

from pydantic import BaseModel, Field


class InvoiceFields(BaseModel):
    """The six invoice fields extracted by the rule-based extractor.

    Every field is always present in the response. A field that could not be
    identified reliably is ``null``.
    """

    invoice_number: str | None = Field(
        description="Invoice number exactly as written, so leading zeros are kept."
    )
    invoice_date: str | None = Field(
        description="Invoice date exactly as written on the document."
    )
    vendor_name: str | None = Field(description="Name of the vendor or seller.")
    subtotal: str | None = Field(
        description="Subtotal as a decimal string without currency or separators, e.g. '1300.00'."
    )
    tax_amount: str | None = Field(
        description="Tax amount as a decimal string without currency or separators."
    )
    total_amount: str | None = Field(
        description="Total amount as a decimal string without currency or separators."
    )


class UploadResponse(BaseModel):
    """Result of validating and processing an uploaded invoice image."""

    filename: str = Field(description="Original filename sent by the client.")
    format: str = Field(description="Image format detected from the file contents.")
    width: int = Field(description="Image width in pixels.")
    height: int = Field(description="Image height in pixels.")
    size_bytes: int = Field(description="Size of the uploaded file in bytes.")
    status: Literal["processed"] = Field(
        description="'processed' when validation and the full pipeline succeeded."
    )
    message: str = Field(description="Human-readable summary of the result.")
    raw_ocr_text: str = Field(description="Text recognized by OCR, before cleaning.")
    cleaned_text: str = Field(
        description="Recognized text after whitespace and line-ending cleanup."
    )
    invoice_fields: InvoiceFields