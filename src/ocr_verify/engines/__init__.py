"""OCR 引擎子包。"""

from .base import OCREngine, MockEngine
from .paddle import PaddleEngine
from .tesseract import TesseractEngine
from .vlm import VLMEngine

__all__ = [
    "OCREngine",
    "MockEngine",
    "PaddleEngine",
    "TesseractEngine",
    "VLMEngine",
]
