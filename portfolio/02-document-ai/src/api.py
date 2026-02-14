"""
FastAPI application for document data extraction.

Accepts document images (invoices, receipts) and returns structured
JSON with extracted fields such as date, vendor, total, and line items.
"""

import io
import os
import tempfile
import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from extractor import InvoiceExtractor
from ocr_engine import OCREngine


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------

class LineItemResponse(BaseModel):
    """Schema for a single extracted line item."""
    description: str
    quantity: Optional[float] = None
    unit_price: Optional[float] = None
    amount: Optional[float] = None


class ExtractionResponse(BaseModel):
    """Schema for the /extract endpoint response."""
    invoice_number: Optional[str] = None
    date: Optional[str] = None
    vendor_name: Optional[str] = None
    total_amount: Optional[float] = None
    subtotal: Optional[float] = None
    tax_amount: Optional[float] = None
    line_items: List[LineItemResponse] = []
    confidence: float = 0.0
    field_confidences: Dict[str, float] = {}
    processing_time_ms: float = 0.0


class HealthResponse(BaseModel):
    """Schema for the /health endpoint response."""
    status: str
    ocr_backend: str
    languages: List[str]


# ---------------------------------------------------------------------------
# Application setup
# ---------------------------------------------------------------------------

OCR_BACKEND = os.getenv("OCR_BACKEND", "easyocr")
OCR_LANGUAGES = os.getenv("OCR_LANGUAGES", "es,en").split(",")

app = FastAPI(
    title="Document AI - Data Extraction API",
    description="Automatic extraction of structured data from invoices and documents.",
    version="1.0.0",
)

# Initialize the OCR engine and extractor
ocr_engine = OCREngine(backend=OCR_BACKEND, languages=OCR_LANGUAGES)
extractor = InvoiceExtractor()


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Return the current health status of the service."""
    return HealthResponse(
        status="healthy",
        ocr_backend=OCR_BACKEND,
        languages=OCR_LANGUAGES,
    )


@app.post("/extract", response_model=ExtractionResponse)
async def extract_document(file: UploadFile = File(...)):
    """
    Extract structured data from an uploaded document image.

    Accepts JPEG or PNG images of invoices, receipts, or similar
    business documents. Returns extracted fields as structured JSON.
    """
    # Validate content type
    allowed_types = ("image/jpeg", "image/png", "image/tiff", "image/bmp")
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type: {file.content_type}. "
                f"Allowed types: {', '.join(allowed_types)}"
            ),
        )

    try:
        start_time = time.time()

        # Save uploaded file to a temporary location for OCR processing
        contents = await file.read()
        suffix = os.path.splitext(file.filename or ".jpg")[1]

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        try:
            # Step 1: Run OCR
            ocr_results = ocr_engine.extract_text(tmp_path, preprocess=True)

            # Step 2: Extract structured fields
            extraction = extractor.extract(ocr_results)
        finally:
            # Clean up temp file
            os.unlink(tmp_path)

        processing_time = (time.time() - start_time) * 1000

        # Build response
        line_items = [
            LineItemResponse(
                description=item.description,
                quantity=item.quantity,
                unit_price=item.unit_price,
                amount=item.amount,
            )
            for item in extraction.line_items
        ]

        return ExtractionResponse(
            invoice_number=extraction.invoice_number,
            date=extraction.date,
            vendor_name=extraction.vendor_name,
            total_amount=extraction.total_amount,
            subtotal=extraction.subtotal,
            tax_amount=extraction.tax_amount,
            line_items=line_items,
            confidence=extraction.confidence,
            field_confidences=extraction.field_confidences,
            processing_time_ms=round(processing_time, 2),
        )

    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Extraction failed: {str(exc)}",
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
