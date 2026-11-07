# Intelligent Image Document Analyzer

A Python-based application that extracts structured invoice information from images using Optical Character Recognition (OCR), image preprocessing, and rule-based extraction. It provides invoice processing through a FastAPI REST API.

## Features

- **Image validation:** Validates image formats, dimensions, and file sizes.
- **Image preprocessing:** Prepares invoice images for OCR using OpenCV.
- **Text recognition:** Extracts text using Tesseract OCR.
- **Text cleaning:** Normalizes recognized text.
- **Invoice extraction:** Identifies invoice number, date, vendor name, subtotal, tax, and total amount.
- **OCR fallback:** Performs additional OCR attempts when fields are missing.
- **REST API:** Provides invoice processing through FastAPI.
- **Automated testing:** Includes unit, API, pipeline, and regression tests.

## Technology Stack

- Python
- FastAPI
- OpenCV
- Tesseract OCR and pytesseract
- Pillow
- Pydantic
- pytest

## Project Structure

```text
intelligent-document-analyzer/
├── app/                 # Application source code
├── docs/                # Technical documentation
├── samples/             # Sample invoice images
├── scripts/             # OCR evaluation scripts
├── tests/               # Automated tests
├── requirements.txt     # Python dependencies
├── .gitignore
└── README.md
```

## Prerequisites

- Python 3
- Git
- Tesseract OCR

On Ubuntu, install Tesseract:

```bash
sudo apt update
sudo apt install tesseract-ocr
```

## Installation

Clone the repository:

```bash
git clone https://github.com/jawadali-s/intelligent-document-analyzer.git
cd intelligent-document-analyzer
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Running the Application

Start the FastAPI server:

```bash
python -m uvicorn app.main:app --reload
```

The application will be available at:

- **API:** http://127.0.0.1:8000
- **Swagger UI:** http://127.0.0.1:8000/docs

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | API information |
| GET | `/health` | Health check |
| POST | `/upload` | Upload and process an invoice image |

To process an invoice, send an image using the multipart form field `file`.

Example:

```bash
curl -X POST "http://127.0.0.1:8000/upload" \
  -F "file=@samples/invoice_testing_dataset/invoice_test_01.png"
```

The response contains image metadata, raw OCR text, cleaned text, and extracted invoice fields.

### Upload Restrictions

- Maximum file size: 10 MB
- Maximum image dimensions: 40 million pixels
- Supported formats: JPEG, PNG, WebP, and TIFF

## Running the Tests

Run the complete test suite:

```bash
python -m pytest -v
```

Run a specific test module:

```bash
python -m pytest tests/test_extraction.py -v
```

## OCR Experiments

The project includes an experimental script for comparing OCR configurations:

```bash
python scripts/evaluate_ocr.py --help
```

This script helps examine OCR behavior and is not a formal accuracy benchmark.

## Limitations

- OCR accuracy depends on image quality, resolution, and layout.
- Rule-based extraction may not identify fields in unusual invoice layouts.
- Extracted values may be missing or incorrect and should be verified before business use.
- The application currently focuses on invoice image processing.

## Technical Documentation

For detailed information about the system architecture, implementation, processing pipeline, API, testing, and application demonstrations, see:

[**View Final Project Technical Documentation (PDF)**](docs/Intelligent_Image_Document_Analyzer_Documentation.pdf)
