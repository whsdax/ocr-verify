"""智能 OCR 测试验证系统。

双层 OCR 引擎(快速 PaddleOCR + 多模态模型复核)+ 指纹缓存,
用于提升客户端 UI 自动化中 OCR 断言的可靠性。
"""

from .config import AppConfig, get_config
from .engines.base import OCREngine
from .engines.paddle import PaddleEngine
from .engines.tesseract import TesseractEngine
from .engines.vlm import VLMEngine
from .router import TwoLayerRouter
from .cache import OCRResultCache
from .types import OCRResult, TextBox, EngineType, EscalationReason
from .verifier import OCRVerifier

__version__ = "0.1.0"

__all__ = [
    "AppConfig",
    "get_config",
    "OCREngine",
    "PaddleEngine",
    "TesseractEngine",
    "VLMEngine",
    "TwoLayerRouter",
    "OCRResultCache",
    "OCRResult",
    "TextBox",
    "EngineType",
    "EscalationReason",
    "OCRVerifier",
]
