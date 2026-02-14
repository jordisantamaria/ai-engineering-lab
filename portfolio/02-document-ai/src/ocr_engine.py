"""
OCR engine wrapper with support for EasyOCR and PaddleOCR.

Provides a unified interface for text extraction from document images,
including preprocessing steps to improve OCR accuracy.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Optional, Tuple

import cv2
import numpy as np
from PIL import Image


@dataclass
class OCRResult:
    """Single text detection result from the OCR engine."""
    text: str
    bbox: List[List[int]]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    confidence: float


class OCREngine:
    """
    Unified OCR wrapper supporting EasyOCR and PaddleOCR backends.

    Handles image preprocessing (deskew, denoise, thresholding) and
    provides a consistent output format regardless of the backend.
    """

    def __init__(
        self,
        backend: Literal["easyocr", "paddleocr"] = "easyocr",
        languages: Optional[List[str]] = None,
    ):
        """
        Initialize the OCR engine.

        Args:
            backend: Which OCR library to use ('easyocr' or 'paddleocr').
            languages: List of language codes. Defaults to ['es', 'en'].
        """
        self.backend = backend
        self.languages = languages or ["es", "en"]
        self._reader = None
        self._initialize_backend()

    def _initialize_backend(self) -> None:
        """Lazy-load the selected OCR backend."""
        if self.backend == "easyocr":
            import easyocr

            self._reader = easyocr.Reader(
                self.languages,
                gpu=False,  # Set to True if GPU is available
            )
        elif self.backend == "paddleocr":
            from paddleocr import PaddleOCR

            # Map language codes for PaddleOCR (uses different codes)
            lang = "es" if "es" in self.languages else "en"
            self._reader = PaddleOCR(
                use_angle_cls=True,
                lang=lang,
                show_log=False,
            )
        else:
            raise ValueError(f"Unsupported OCR backend: {self.backend}")

    def preprocess_image(
        self,
        image_path: str,
        deskew: bool = True,
        denoise: bool = True,
        threshold: bool = True,
    ) -> np.ndarray:
        """
        Apply preprocessing steps to improve OCR accuracy.

        Args:
            image_path: Path to the input image.
            deskew: Whether to correct image rotation/skew.
            denoise: Whether to apply noise reduction.
            threshold: Whether to apply adaptive thresholding.

        Returns:
            Preprocessed image as a numpy array (grayscale or BGR).
        """
        # Read the image
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        # Convert to grayscale for preprocessing
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Step 1: Deskew - correct rotation using Hough transform
        if deskew:
            gray = self._deskew(gray)

        # Step 2: Denoise - reduce noise while preserving edges
        if denoise:
            gray = cv2.fastNlMeansDenoising(gray, h=10)

        # Step 3: Adaptive thresholding for better binarization
        if threshold:
            gray = cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                blockSize=11,
                C=2,
            )

        return gray

    def _deskew(self, image: np.ndarray) -> np.ndarray:
        """
        Correct image skew using the minimum area rectangle of contours.

        Args:
            image: Grayscale image as numpy array.

        Returns:
            Deskewed image.
        """
        # Detect edges
        edges = cv2.Canny(image, 50, 150, apertureSize=3)

        # Find lines using Hough transform
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, threshold=100,
            minLineLength=100, maxLineGap=10,
        )

        if lines is None:
            return image

        # Calculate the median angle of detected lines
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if abs(angle) < 45:  # Only consider near-horizontal lines
                angles.append(angle)

        if not angles:
            return image

        median_angle = np.median(angles)

        # Rotate the image to correct the skew
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        rotated = cv2.warpAffine(
            image, rotation_matrix, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )

        return rotated

    def extract_text(
        self,
        image_path: str,
        preprocess: bool = True,
    ) -> List[OCRResult]:
        """
        Extract text from a document image.

        Args:
            image_path: Path to the document image.
            preprocess: Whether to apply preprocessing before OCR.

        Returns:
            List of OCRResult objects, each containing the detected text,
            bounding box coordinates, and confidence score.
        """
        if preprocess:
            processed = self.preprocess_image(image_path)
        else:
            processed = cv2.imread(str(image_path))

        if self.backend == "easyocr":
            return self._extract_easyocr(processed)
        elif self.backend == "paddleocr":
            return self._extract_paddleocr(processed)

    def _extract_easyocr(self, image: np.ndarray) -> List[OCRResult]:
        """Run EasyOCR on the preprocessed image."""
        raw_results = self._reader.readtext(image)

        results = []
        for bbox, text, confidence in raw_results:
            # EasyOCR returns bbox as [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
            results.append(
                OCRResult(
                    text=text.strip(),
                    bbox=[[int(p[0]), int(p[1])] for p in bbox],
                    confidence=round(float(confidence), 4),
                )
            )

        return results

    def _extract_paddleocr(self, image: np.ndarray) -> List[OCRResult]:
        """Run PaddleOCR on the preprocessed image."""
        raw_results = self._reader.ocr(image, cls=True)

        results = []
        if raw_results and raw_results[0]:
            for line in raw_results[0]:
                bbox = line[0]
                text = line[1][0]
                confidence = line[1][1]

                results.append(
                    OCRResult(
                        text=text.strip(),
                        bbox=[[int(p[0]), int(p[1])] for p in bbox],
                        confidence=round(float(confidence), 4),
                    )
                )

        return results

    def extract_text_raw(self, image_path: str) -> str:
        """
        Extract all text from a document as a single string.

        Convenience method that concatenates all detected text blocks
        sorted by vertical position (top to bottom).

        Args:
            image_path: Path to the document image.

        Returns:
            Full text content as a single string.
        """
        results = self.extract_text(image_path)

        # Sort by vertical position (top of bounding box)
        results.sort(key=lambda r: r.bbox[0][1])

        return "\n".join(r.text for r in results)
