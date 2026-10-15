import warnings
from io import BytesIO
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from PIL import Image, UnidentifiedImageError

MAX_UPLOAD_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB
READ_CHUNK_SIZE_BYTES = 1024 * 1024  # 1 MB
MAX_IMAGE_PIXELS = 40_000_000  # Images larger than this are rejected
ALLOWED_IMAGE_FORMATS = {"JPEG", "PNG", "WEBP", "TIFF"}

# Pillow warns above this limit and raises DecompressionBombError at twice
# this limit. Both cases are handled inside validate_image().
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS

app = FastAPI(
    title="Intelligent Image Document Analyzer",
    description=(
        "A backend API for analyzing image-based documents. "
        "Currently supports validated image upload; analysis features "
        "will be added in later steps."
    ),
    version="0.2.0",
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


@app.post("/upload")
async def upload_invoice_image(
    file: Annotated[UploadFile, File(description="Invoice image (JPEG, PNG, WebP, TIFF)")],
) -> dict[str, str | int]:
    """Validate an uploaded invoice image and return its basic properties."""
    image_buffer = BytesIO()
    try:
        size_bytes = await read_upload_with_limit(file, image_buffer)

        if size_bytes == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        image_format, width, height = await run_in_threadpool(
            validate_image, image_buffer
        )
    finally:
        image_buffer.close()  # Release the in-memory image data.
        await file.close()

    return {
        "filename": file.filename or "unknown",
        "format": image_format,
        "width": width,
        "height": height,
        "size_bytes": size_bytes,
        "status": "valid",
        "message": "Image was successfully validated.",
    }