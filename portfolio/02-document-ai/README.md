# Automated Data Extraction from Documents

## Business Problem

Manual document processing (invoices, delivery notes, purchase orders, contracts) is a task that consumes time and resources in any organization:

- **Tedious**: an employee takes 3-5 minutes per invoice to manually enter data into the ERP.
- **Error-prone**: the human data entry error rate is 1-4%, generating accounting discrepancies and audit issues.
- **Not scalable**: during activity peaks (monthly closing, campaigns), unprocessed documents pile up.
- **Hidden cost**: time spent on repetitive tasks prevents staff from focusing on higher-value tasks.

## Proposed Solution

Automated information extraction pipeline that combines OCR (Optical Character Recognition) with NLP (Natural Language Processing) to automatically extract key fields from documents.

### Architecture

```
Document (image/PDF)
        |
        v
  Preprocessing
  (deskew, denoise, binarization)
        |
        v
  OCR Engine (EasyOCR / PaddleOCR)
        |
        v
  Text + Bounding Boxes + Confidence
        |
        v
  Field Extractor (regex + heuristics)
        |
        v
  Structured Data (JSON)
  - Date
  - Invoice number
  - Vendor
  - Total amount
  - Line items / detail lines
```

### Key Components

1. **Image preprocessing**: skew correction, noise removal, and binarization to maximize OCR accuracy.
2. **Dual OCR engine**: support for EasyOCR (simpler) and PaddleOCR (more accurate on complex documents).
3. **Smart extractor**: combination of regex patterns for standard fields (dates, amounts) with positional heuristics (the total is usually at the bottom, the vendor in the header).

## Expected Results

| Metric | Value |
|---------|-------|
| Field extraction accuracy | >90% |
| Time per document | <3 seconds |
| Supported document types | Invoices, delivery notes, purchase orders |
| OCR languages | Spanish, English, Catalan |

## Technologies

- **EasyOCR**: open-source OCR engine with multi-language support
- **PaddleOCR**: high-performance OCR engine from Baidu
- **OpenCV**: image preprocessing
- **FastAPI**: REST API for integration
- **Pydantic**: structured data validation

## How to Run

### 1. Installation

```bash
cd portfolio/02-document-ai
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Launch the API

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8001
```

### 3. Extract Data from a Document

```bash
curl -X POST "http://localhost:8001/extract" \
    -F "file=@invoice_example.jpg"
```

Expected response:
```json
{
    "invoice_number": "FAC-2024-001234",
    "date": "2024-03-15",
    "vendor_name": "Suministros Industriales S.L.",
    "total_amount": 1542.30,
    "line_items": [
        {"description": "Tornillos M8x40", "quantity": 500, "unit_price": 0.12, "amount": 60.00},
        {"description": "Tuercas M8", "quantity": 500, "unit_price": 0.08, "amount": 40.00}
    ],
    "confidence": 0.92
}
```

### 4. Docker

```bash
docker build -t document-ai .
docker run -p 8001:8001 document-ai
```

## How to Present It: Client Pitch

### Value Proposition

> "Turn piles of invoices into structured data in seconds, not hours. Our system automatically extracts key information from any document, reducing errors and freeing your team for strategic tasks."

### Estimated ROI

**Scenario**: company processing 2,000 invoices/month with 2 dedicated employees.

| Item | Before | After |
|----------|-------|---------|
| Time per invoice | 3-5 minutes | <10 seconds |
| Monthly hours dedicated | ~130 hours | ~15 hours (review) |
| Error rate | 2-4% | <0.5% |
| Monthly processing cost | ~4,000 EUR | ~1,200 EUR |

**Estimated savings: ~33,600 EUR/year** in processing costs, plus the elimination of error-related costs (discrepancies, claims, rework).

### Key Points for the Presentation

1. **Demo with real documents**: ask the client to bring 2-3 invoices from their daily operations and process them live.
2. **ERP integration**: extracted data can be sent directly to SAP, Navision, Sage, etc.
3. **Continuous learning**: the system improves with user feedback (corrections).
4. **Compliance**: full processing traceability for audits.
5. **Multi-format**: works with scans, mobile phone photos, and PDFs.

### Frequently Asked Client Questions

- **"Does it work with our invoices?"** - Yes, the system adapts to any format. During the pilot phase, it is calibrated with your specific documents.
- **"What happens if the OCR fails?"** - Documents with low confidence (<80%) are flagged for human review. The system prioritizes accuracy over coverage.
- **"Can it integrate with our ERP?"** - Yes, the API returns structured JSON that maps to any ERP fields via standard integration.
