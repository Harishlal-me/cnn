"""
OCR text extractor for image-only samples.
Uses EasyOCR to extract readable text from meme/image content.
Falls back gracefully if easyocr is not installed.
"""
import torch

try:
    import easyocr
    _HAS_EASYOCR = True
except ImportError:
    _HAS_EASYOCR = False


class OCRExtractor:
    def __init__(self):
        if _HAS_EASYOCR:
            self.reader = easyocr.Reader(
                ["en"], gpu=torch.cuda.is_available(), verbose=False
            )
        else:
            self.reader = None

    def extract(self, image_path: str) -> str:
        """Extract text from an image. Returns empty string on failure."""
        if self.reader is None:
            return ""
        try:
            results = self.reader.readtext(image_path)
            tokens = [text for (_, text, conf) in results if conf > 0.6]
            return " ".join(tokens)
        except Exception:
            return ""


# Singleton — initialised lazily on first use
_extractor = None

def get_ocr_extractor():
    global _extractor
    if _extractor is None:
        _extractor = OCRExtractor()
    return _extractor
