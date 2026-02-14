"""
Document field extractor for invoices and similar business documents.

Combines regex patterns with positional heuristics to extract structured
fields (date, total, vendor, line items) from OCR output.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ocr_engine import OCRResult


@dataclass
class LineItem:
    """A single line item from an invoice."""
    description: str
    quantity: Optional[float] = None
    unit_price: Optional[float] = None
    amount: Optional[float] = None


@dataclass
class ExtractionResult:
    """Structured extraction result from a document."""
    invoice_number: Optional[str] = None
    date: Optional[str] = None
    vendor_name: Optional[str] = None
    total_amount: Optional[float] = None
    subtotal: Optional[float] = None
    tax_amount: Optional[float] = None
    line_items: List[LineItem] = field(default_factory=list)
    raw_text: str = ""
    confidence: float = 0.0
    field_confidences: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the result to a JSON-serializable dictionary."""
        return {
            "invoice_number": self.invoice_number,
            "date": self.date,
            "vendor_name": self.vendor_name,
            "total_amount": self.total_amount,
            "subtotal": self.subtotal,
            "tax_amount": self.tax_amount,
            "line_items": [
                {
                    "description": item.description,
                    "quantity": item.quantity,
                    "unit_price": item.unit_price,
                    "amount": item.amount,
                }
                for item in self.line_items
            ],
            "confidence": self.confidence,
            "field_confidences": self.field_confidences,
        }


class InvoiceExtractor:
    """
    Extract structured fields from invoice OCR results.

    Uses a combination of regex patterns for standard fields and
    positional heuristics (e.g. totals at the bottom, vendor at the top)
    to locate and parse document fields.
    """

    # Regex patterns for common invoice fields

    # Date patterns: DD/MM/YYYY, DD-MM-YYYY, DD.MM.YYYY, YYYY-MM-DD
    DATE_PATTERNS = [
        r"(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})",
        r"(\d{4}[/\-\.]\d{1,2}[/\-\.]\d{1,2})",
        r"(\d{1,2}\s+de\s+\w+\s+de\s+\d{4})",  # "15 de marzo de 2024"
    ]

    # Invoice number patterns
    INVOICE_PATTERNS = [
        r"(?:factura|invoice|fra|fac|n[uú]mero|no\.?|num\.?|#)\s*[:\-]?\s*([A-Z0-9\-/]+)",
        r"([A-Z]{2,4}[\-/]\d{4}[\-/]\d{3,6})",  # FAC-2024-001234
        r"(?:n[uú]m\.?\s*factura)\s*[:\-]?\s*(\S+)",
    ]

    # Amount patterns (European format with comma decimal separator)
    AMOUNT_PATTERNS = [
        r"(\d{1,3}(?:\.\d{3})*,\d{2})\s*(?:EUR|€)?",  # 1.234,56
        r"(?:EUR|€)\s*(\d{1,3}(?:\.\d{3})*,\d{2})",    # EUR 1.234,56
        r"(\d+,\d{2})\s*(?:EUR|€)?",                     # 1234,56
        r"(\d{1,3}(?:,\d{3})*\.\d{2})\s*(?:USD|\$)?",   # 1,234.56 (US format)
    ]

    # Total amount keywords
    TOTAL_KEYWORDS = [
        "total", "importe total", "total factura", "total a pagar",
        "amount due", "total due", "grand total", "total general",
    ]

    # Subtotal keywords
    SUBTOTAL_KEYWORDS = [
        "subtotal", "base imponible", "importe bruto", "neto",
    ]

    # Tax keywords
    TAX_KEYWORDS = [
        "iva", "i.v.a", "tax", "impuesto", "igic", "irpf",
    ]

    def __init__(self):
        """Initialize the extractor with compiled regex patterns."""
        self._date_re = [re.compile(p, re.IGNORECASE) for p in self.DATE_PATTERNS]
        self._invoice_re = [
            re.compile(p, re.IGNORECASE) for p in self.INVOICE_PATTERNS
        ]
        self._amount_re = [re.compile(p, re.IGNORECASE) for p in self.AMOUNT_PATTERNS]

    def extract(self, ocr_results: List[OCRResult]) -> ExtractionResult:
        """
        Extract all fields from a list of OCR results.

        Args:
            ocr_results: List of OCRResult objects from the OCR engine.

        Returns:
            ExtractionResult with all extracted fields and confidence scores.
        """
        result = ExtractionResult()

        # Combine all text for full-document regex matching
        full_text = "\n".join(r.text for r in ocr_results)
        result.raw_text = full_text

        # Sort results by vertical position for positional heuristics
        sorted_results = sorted(ocr_results, key=lambda r: r.bbox[0][1])

        # Extract individual fields
        result.date, result.field_confidences["date"] = self._extract_date(
            full_text, sorted_results
        )
        result.invoice_number, result.field_confidences["invoice_number"] = (
            self._extract_invoice_number(full_text, sorted_results)
        )
        result.vendor_name, result.field_confidences["vendor_name"] = (
            self._extract_vendor_name(sorted_results)
        )
        result.total_amount, result.field_confidences["total_amount"] = (
            self._extract_total_amount(full_text, sorted_results)
        )
        result.subtotal, result.field_confidences["subtotal"] = (
            self._extract_amount_near_keyword(full_text, self.SUBTOTAL_KEYWORDS)
        )
        result.tax_amount, result.field_confidences["tax_amount"] = (
            self._extract_amount_near_keyword(full_text, self.TAX_KEYWORDS)
        )
        result.line_items = self._extract_line_items(sorted_results)

        # Calculate overall confidence as the average of field confidences
        valid_confidences = [
            v for v in result.field_confidences.values() if v > 0
        ]
        result.confidence = (
            round(sum(valid_confidences) / len(valid_confidences), 4)
            if valid_confidences
            else 0.0
        )

        return result

    def _extract_date(
        self,
        full_text: str,
        sorted_results: List[OCRResult],
    ) -> Tuple[Optional[str], float]:
        """
        Extract the document date using regex patterns.

        Prioritizes dates found near keywords like 'fecha', 'date'.
        """
        # First try to find dates near a 'fecha' or 'date' keyword
        date_keyword_re = re.compile(
            r"(?:fecha|date)\s*[:\-]?\s*(.+)", re.IGNORECASE
        )
        for result in sorted_results:
            match = date_keyword_re.search(result.text)
            if match:
                date_str = match.group(1).strip()
                # Validate the extracted string looks like a date
                for pattern in self._date_re:
                    date_match = pattern.search(date_str)
                    if date_match:
                        return date_match.group(1), result.confidence

        # Fallback: search for any date pattern in the full text
        for pattern in self._date_re:
            match = pattern.search(full_text)
            if match:
                return match.group(1), 0.7

        return None, 0.0

    def _extract_invoice_number(
        self,
        full_text: str,
        sorted_results: List[OCRResult],
    ) -> Tuple[Optional[str], float]:
        """
        Extract the invoice number using regex patterns.

        Looks for patterns like 'Factura: FAC-2024-001234'.
        """
        for pattern in self._invoice_re:
            match = pattern.search(full_text)
            if match:
                invoice_num = match.group(1).strip()
                # Find the OCR confidence for this region
                for result in sorted_results:
                    if invoice_num in result.text:
                        return invoice_num, result.confidence
                return invoice_num, 0.8

        return None, 0.0

    def _extract_vendor_name(
        self,
        sorted_results: List[OCRResult],
    ) -> Tuple[Optional[str], float]:
        """
        Extract the vendor name using positional heuristics.

        Assumes the vendor name is one of the first text blocks in the
        document (top area), and typically the longest text in the header.
        """
        if not sorted_results:
            return None, 0.0

        # Look at the top 30% of the document
        all_y = [r.bbox[0][1] for r in sorted_results]
        max_y = max(all_y) if all_y else 1
        header_results = [
            r for r in sorted_results if r.bbox[0][1] < max_y * 0.3
        ]

        if not header_results:
            return None, 0.0

        # Skip results that look like dates, invoice numbers, or short text
        candidates = []
        for r in header_results:
            text = r.text.strip()
            # Filter out dates, numbers-only, and very short strings
            if len(text) < 5:
                continue
            if re.match(r"^\d+[/\-\.]\d+", text):
                continue
            if re.match(r"^(?:factura|invoice|fecha|date|tel|fax|cif|nif)", text, re.I):
                continue
            candidates.append(r)

        if not candidates:
            return None, 0.0

        # Take the longest candidate as the most likely vendor name
        best = max(candidates, key=lambda r: len(r.text))
        return best.text.strip(), best.confidence

    def _extract_total_amount(
        self,
        full_text: str,
        sorted_results: List[OCRResult],
    ) -> Tuple[Optional[float], float]:
        """
        Extract the total amount from the document.

        Searches for amounts near total-related keywords, with preference
        for amounts found in the bottom half of the document.
        """
        return self._extract_amount_near_keyword(full_text, self.TOTAL_KEYWORDS)

    def _extract_amount_near_keyword(
        self,
        full_text: str,
        keywords: List[str],
    ) -> Tuple[Optional[float], float]:
        """
        Find a monetary amount near one of the given keywords.

        Args:
            full_text: Full OCR text.
            keywords: List of keywords to search near.

        Returns:
            Tuple of (amount, confidence).
        """
        for keyword in keywords:
            # Build a pattern that finds the keyword followed by an amount
            pattern = re.compile(
                rf"{re.escape(keyword)}\s*[:\-]?\s*"
                rf"(?:EUR|€)?\s*(\d{{1,3}}(?:[\.\,]\d{{3}})*[\.\,]\d{{2}})",
                re.IGNORECASE,
            )
            match = pattern.search(full_text)
            if match:
                amount_str = match.group(1)
                amount = self._parse_amount(amount_str)
                if amount is not None:
                    return amount, 0.85

        return None, 0.0

    def _parse_amount(self, amount_str: str) -> Optional[float]:
        """
        Parse a monetary amount string to a float.

        Handles both European (1.234,56) and US (1,234.56) formats.
        """
        try:
            # European format: dots as thousands separator, comma as decimal
            if "," in amount_str and "." in amount_str:
                if amount_str.rindex(",") > amount_str.rindex("."):
                    # European: 1.234,56
                    cleaned = amount_str.replace(".", "").replace(",", ".")
                else:
                    # US: 1,234.56
                    cleaned = amount_str.replace(",", "")
            elif "," in amount_str:
                # Could be European decimal: 1234,56
                cleaned = amount_str.replace(",", ".")
            else:
                cleaned = amount_str

            return round(float(cleaned), 2)
        except (ValueError, TypeError):
            return None

    def _extract_line_items(
        self,
        sorted_results: List[OCRResult],
    ) -> List[LineItem]:
        """
        Extract line items from the middle section of the document.

        Uses heuristics: line items typically contain a description
        followed by quantity, unit price, and amount values.
        """
        items = []

        # Look for rows that contain amounts (potential line items)
        amount_pattern = re.compile(
            r"(\d+(?:[,\.]\d+)?)\s+(\d+(?:[,\.]\d{2}))\s+(\d+(?:[,\.]\d{2}))"
        )

        for result in sorted_results:
            match = amount_pattern.search(result.text)
            if match:
                # Try to extract the description (text before the numbers)
                desc_end = match.start()
                description = result.text[:desc_end].strip()

                if len(description) > 2:
                    quantity = self._parse_amount(match.group(1))
                    unit_price = self._parse_amount(match.group(2))
                    amount = self._parse_amount(match.group(3))

                    items.append(
                        LineItem(
                            description=description,
                            quantity=quantity,
                            unit_price=unit_price,
                            amount=amount,
                        )
                    )

        return items
