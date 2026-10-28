import logging
import warnings
from io import BytesIO
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from PIL import Image, UnidentifiedImageError

from app.ocr import OCRError
from app.pipeline import ImageDecodeError, process_invoice_image
from app.schemas import UploadResponse

logger = logging.getLogger(__name__)

# Page segmentation mode for the second OCR pass, used only when the first pass
# leaves an invoice field empty. PSM 4 reads text in columns of variable size,
# which recovers table cells that PSM 6 can drop.
FALLBACK_PSM = 4
# Third pass, only for fields still empty: a 0.75x copy of the page. Tesseract can
# misread the commas of an amount at one scale and read them correctly at another.
RESCUE_SCALE = 0.75

MAX_UPLOAD_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB
READ_CHUNK_SIZE_BYTES = 1024 * 1024  # 1 MB
MAX_IMAGE_PIXELS = 40_000_000  # Images larger than this are rejected
ALLOWED_IMAGE_FORMATS = {"JPEG", "PNG", "WEBP", "TIFF"}

DECODE_FAILURE_MESSAGE = (
    "The image passed validation but could not be decoded for processing."
)
OCR_FAILURE_MESSAGE = "Text recognition failed on the server. Please try again later."

# Pillow warns above this limit and raises DecompressionBombError at twice
# this limit. Both cases are handled inside validate_image().
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS

app = FastAPI(
    title="Intelligent Image Document Analyzer",
    description=(
        "A backend API for analyzing invoice images. An uploaded image is "
        "validated, preprocessed with OpenCV, read with Tesseract OCR, cleaned, "
        "and searched with rule-based extraction for the invoice number, date, "
        "vendor name, subtotal, tax amount and total amount."
    ),
    version="0.3.0",
)


async def read_upload_with_limit(upload: UploadFile, destination: BytesIO) -> int:
    """Copy an upload into `destination` in chunks and return its size in bytes.

    Raises HTTP 413 as soon as the upload exceeds the size limit.
    """
    if upload.size is not None and upload.size > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail="File is too large. Maximum allowed size is 10 MB.",
        )

    total_bytes = 0
    while chunk := await upload.read(READ_CHUNK_SIZE_BYTES):
        total_bytes += len(chunk)
        if total_bytes > MAX_UPLOAD_SIZE_BYTES:
            raise HTTPException(
                status_code=413,
                detail="File is too large. Maximum allowed size is 10 MB.",
            )
        destination.write(chunk)

    destination.seek(0)
    return total_bytes


def validate_image(image_stream: BytesIO) -> tuple[str, int, int]:
    """Validate image contents with Pillow and return (format, width, height).

    This is blocking, CPU-bound work, so call it from a worker thread.
    """
    try:
        # Treat Pillow's decompression-bomb warning as an error, but only
        # inside this block. The previous warning filters are restored on exit.
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)

            # First pass: read the header, check format and size, then verify.
            with Image.open(image_stream) as image:
                if image.format not in ALLOWED_IMAGE_FORMATS:
                    raise HTTPException(
                        status_code=415,
                        detail=(
                            f"Unsupported image format: {image.format}. "
                            "Allowed formats: JPEG, PNG, WebP, TIFF."
                        ),
                    )

                width, height = image.size
                if width * height > MAX_IMAGE_PIXELS:
                    raise HTTPException(
                        status_code=413,
                        detail="Image dimensions are too large.",
                    )

                image.verify()  # The image object cannot be used after this.

            # Second pass: verify() does not fully decode the pixel data, so
            # reopen the image and decode it to catch truncated or corrupt files.
            image_stream.seek(0)
            with Image.open(image_stream) as image:
                detected_format = image.format
                if detected_format not in ALLOWED_IMAGE_FORMATS:
                    raise HTTPException(
                        status_code=415,
                        detail=(
                            f"Unsupported image format: {detected_format}. "
                            "Allowed formats: JPEG, PNG, WebP, TIFF."
                        ),
                    )

                width, height = image.size
                if width * height > MAX_IMAGE_PIXELS:
                    raise HTTPException(
                        status_code=413,
                        detail="Image dimensions are too large.",
                    )

                image.load()  # Fully decode the pixel data.

    except (Image.DecompressionBombWarning, Image.DecompressionBombError):
        raise HTTPException(
            status_code=413,
            detail="Image dimensions are too large.",
        )
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=400,
            detail="File is not a recognizable image.",
        )
    except (OSError, SyntaxError, ValueError, EOFError):
        raise HTTPException(
            status_code=400,
            detail="Image is corrupt or incomplete.",
        )

    return detected_format, width, height


@app.get("/")
def read_root() -> dict[str, str]:
    """Identify the project and confirm the API is running."""
    return {
        "project": "Intelligent Image Document Analyzer",
        "message": "API is running",
    }


@app.get("/health")
def health_check() -> dict[str, str]:
    """Report the health status of the API."""
    return {"status": "healthy"}


@app.post(
    "/upload",
    responses={
        400: {"description": "Empty file, not an image, or corrupt image."},
        413: {"description": "File over 10 MB, or image dimensions too large."},
        415: {"description": "Image format is not JPEG, PNG, WebP or TIFF."},
        422: {"description": "Image passed validation but could not be decoded."},
        500: {"description": "Text recognition (OCR) failed on the server."},
    },
)
async def upload_invoice_image(
    file: Annotated[UploadFile, File(description="Invoice image (JPEG, PNG, WebP, TIFF)")],
) -> UploadResponse:
    """Validate an invoice image, run OCR on it, and extract invoice fields.

    The upload is read in chunks (maximum 10 MB) and validated from its actual
    contents. The image is then preprocessed, read with Tesseract OCR, cleaned,
    and searched for the invoice number, date, vendor name, subtotal, tax amount
    and total amount. Fields that cannot be identified reliably are ``null``.
    Nothing is saved to disk.
    """
    image_buffer = BytesIO()
    try:
        size_bytes = await read_upload_with_limit(file, image_buffer)

        if size_bytes == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        image_format, width, height = await run_in_threadpool(
            validate_image, image_buffer
        )

        # The pipeline takes the file's raw bytes, not the BytesIO or a Pillow image.
        image_bytes = image_buffer.getvalue()
        try:
            result = await run_in_threadpool(
                process_invoice_image,
                image_bytes,
                fallback_psm=FALLBACK_PSM,
                rescue_scale=RESCUE_SCALE,
            )
        except ImageDecodeError as exc:
            raise HTTPException(status_code=422, detail=DECODE_FAILURE_MESSAGE) from exc
        except OCRError as exc:
            # Log the details for the operator, but keep them out of the response.
            logger.error("OCR failed while processing an upload: %s", exc, exc_info=exc)
            raise HTTPException(status_code=500, detail=OCR_FAILURE_MESSAGE) from exc
    finally:
        image_buffer.close()  # Release the in-memory image data.
        await file.close()

    return UploadResponse(
        filename=file.filename or "unknown",
        format=image_format,
        width=width,
        height=height,
        size_bytes=size_bytes,
        status="processed",
        message="Image was validated and processed successfully.",
        raw_ocr_text=result["raw_ocr_text"],
        cleaned_text=result["cleaned_text"],
        invoice_fields=result["invoice_fields"],
    )