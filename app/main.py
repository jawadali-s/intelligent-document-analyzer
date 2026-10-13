from fastapi import FastAPI

app = FastAPI(
    title="Intelligent Image Document Analyzer",
    description=(
        "A backend API for analyzing image-based documents. "
        "This is the initial project skeleton; analysis features will be added in later steps."
    ),
    version="0.1.0",
)


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