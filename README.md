# Intelligent Image Document Analyzer

A Python-based application that extracts structured invoice information from images using Optical Character Recognition (OCR), image preprocessing, and rule-based field extraction. The application exposes its functionality through a FastAPI REST API.

## Overview

Invoices contain important business information, including invoice numbers, dates, vendor names, and monetary amounts. Extracting this information manually can be time-consuming.

The Intelligent Image Document Analyzer automates part of this process by processing invoice images and returning recognized text alongside extracted invoice fields in a structured JSON response.

The application uses Tesseract OCR for text recognition and Python-based rules to identify invoice information.

## Features

- **Image validation:** Checks image format, dimensions, and file size before processing.
- **Image preprocessing:** Uses OpenCV-based image processing to prepare images for OCR.
- **Optical Character Recognition:** Uses Tesseract OCR to recognize text in invoice images.
- **Text cleaning:** Normalizes recognized text before extraction.
- **Invoice field extraction:** Identifies invoice numbers, dates, vendor names, and monetary amounts.
- **OCR fallback processing:** Uses additional OCR processing when invoice fields remain missing.
- **Structured responses:** Returns image metadata, recognized text, and extracted fields as JSON.
- **Input validation and error handling:** Handles empty, invalid, corrupted, oversized, and unsupported image uploads.
- **REST API:** Provides invoice processing through FastAPI.
- **Interactive documentation:** Makes API endpoints accessible through Swagger UI.
- **Automated testing:** Includes unit, API, pipeline, and invoice regression tests.
- **OCR experiments:** Includes a script for comparing OCR configurations on an individual image.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Application logic |
| FastAPI | REST API |
| OpenCV | Image processing |
| Tesseract OCR | Text recognition |
| pytesseract | Python interface to Tesseract |
| Pillow | Image validation and decoding |
| Pydantic | Structured request and response data |
| pytest | Automated testing |

Python dependencies and their pinned versions are listed in `requirements.txt`.

## Project Structure

```text
intelligent-document-analyzer/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── pipeline.py
│   ├── image_processing.py
│   ├── ocr.py
│   ├── text_cleaning.py
│   ├── extraction.py
│   └── schemas.py
├── docs/
├── samples/
│   └── invoice_testing_dataset/
├── scripts/
│   └── evaluate_ocr.py
├── tests/
│   ├── invoice_ocr_samples.py
│   ├── invoice_regression_cases.py
│   ├── test_api.py
│   ├── test_extraction.py
│   ├── test_image_processing.py
│   ├── test_invoice_regression.py
│   ├── test_ocr.py
│   ├── test_pipeline.py
│   └── test_text_cleaning.py
├── .gitignore
├── requirements.txt
└── README.md
```

### Main Components

- `app/main.py`: Defines the FastAPI application, validates uploads, and exposes API endpoints.
- `app/pipeline.py`: Coordinates invoice image processing.
- `app/image_processing.py`: Handles image preprocessing.
- `app/ocr.py`: Integrates with Tesseract OCR.
- `app/text_cleaning.py`: Cleans and normalizes OCR text.
- `app/extraction.py`: Extracts structured invoice fields from recognized text.
- `app/schemas.py`: Defines structured API response models.
- `samples/`: Contains sample invoice images.
- `tests/`: Contains automated and regression tests.
- `scripts/evaluate_ocr.py`: Runs OCR experiments on a supplied invoice image.

## Prerequisites

You need:

- Python with compatible package versions.
- Git.
- Tesseract OCR installed and available to the application.

On Ubuntu, install Tesseract using:

```bash
sudo apt update
sudo apt install tesseract-ocr
```

Verify the installation:

```bash
tesseract --version
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/jawadali-s/intelligent-document-analyzer.git
cd intelligent-document-analyzer
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
```

### 3. Activate the environment

On Linux:

```bash
source .venv/bin/activate
```

### 4. Install Python dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Running the Application

Start the FastAPI development server:

```bash
python -m uvicorn app.main:app --reload
```

By default, the local server is available at:

`http://127.0.0.1:8000`

### Interactive API Documentation

Open:

`http://127.0.0.1:8000/docs`

Swagger UI lets you inspect endpoints, review request requirements, upload test images, and examine API responses.

## API Endpoints

### 1. Root Endpoint

**Method:** `GET`

**Path:** `/`

Returns information identifying the project and indicating that the API is running.

### 2. Health Check

**Method:** `GET`

**Path:** `/health`

Example response:

```json
{
  "status": "healthy"
}
```

### 3. Upload and Process an Invoice

**Method:** `POST`

**Path:** `/upload`

Submit an invoice image using the multipart form field `file`.

Example using cURL:

```bash
curl -X POST "http://127.0.0.1:8000/upload" \
  -F "file=@samples/invoice_testing_dataset/invoice_test_01.png"
```

The response includes the uploaded filename, detected image format, image dimensions, file size, processing status, raw OCR text, cleaned text, and extracted invoice fields.

The application processes uploads in memory and does not save uploaded images to disk through this endpoint.

### Upload Restrictions

| Restriction | Behavior |
|---|---|
| Maximum upload size | 10 MB |
| Maximum image dimensions | 40 million pixels |
| Supported formats | JPEG, PNG, WebP, TIFF |
| Empty file | Rejected |
| Corrupted image | Rejected |
| Unsupported image format | Rejected |

### HTTP Error Responses

| Status code | Meaning |
|---|---|
| `400` | Empty file, unrecognized image, or corrupt image |
| `413` | Upload exceeds the size limit or image dimensions are too large |
| `415` | Unsupported image format |
| `422` | Image passed validation but could not be decoded by the processing pipeline |
| `500` | OCR processing failed on the server |

## Invoice Extraction

The application extracts invoice information from OCR text using rule-based processing.

The fields identified by the application include:

- Invoice number
- Invoice date
- Vendor name
- Subtotal
- Tax amount
- Total amount

The exact values returned depend on the input image and the extraction rules.

When a field cannot be identified reliably, it may be returned as `null`. OCR output and extracted fields should be reviewed before being used for financial or business decisions.

The processing pipeline also includes additional OCR attempts to recover fields that remain missing after the initial pass.

## Running the Tests

Activate the virtual environment and run:

```bash
python -m pytest -v
```

Run a specific test module:

```bash
python -m pytest tests/test_extraction.py -v
```

Run invoice regression tests:

```bash
python -m pytest tests/test_invoice_regression.py -v
```

### Testing Areas

The repository contains tests for:

- API behavior and upload validation.
- Image processing.
- OCR functionality.
- Text cleaning.
- Invoice field extraction.
- End-to-end pipeline processing.
- Invoice regression cases.

Test results may change as the code and test suite evolve. Run the current tests to verify the latest implementation.

## OCR Experiments

The repository includes `scripts/evaluate_ocr.py` for comparing OCR configurations on an individual image.

Display its supported options:

```bash
python scripts/evaluate_ocr.py --help
```

Run all available experiments on a sample invoice:

```bash
python scripts/evaluate_ocr.py samples/invoice_testing_dataset/invoice_test_01.png
```

To run a specific experiment:

```bash
python scripts/evaluate_ocr.py samples/invoice_testing_dataset/invoice_test_01.png --experiment 1
```

The script is an experimental tool for examining OCR behavior on an image. It should not be treated as a formal accuracy benchmark unless evaluation against verified ground-truth data is implemented.

## Limitations

- OCR quality depends on image resolution, clarity, orientation, and layout.
- Rule-based extraction depends on recognizable invoice labels and supported formats.
- Unusual layouts or ambiguous text may result in missing or incorrectly identified fields.
- Successful processing does not guarantee that every extracted value is correct.
- The supported image formats and upload restrictions apply to the current upload endpoint.

## Future Improvements

Potential areas for improvement include:

- Expanding support for additional invoice layouts.
- Improving extraction accuracy on challenging images.
- Evaluating extraction against manually verified ground-truth data.
- Increasing regression-test coverage with representative invoice examples.
- Adding continuous integration to run tests automatically.
- Improving extraction for additional languages and document types.

## Contributing

When changing invoice extraction behavior, add or update regression tests for the affected cases.

Before committing changes, run the complete test suite:

```bash
python -m pytest -v
```

## License

No license has been specified in this repository. Add an appropriate license if you intend to grant others permission to reuse or redistribute the project.