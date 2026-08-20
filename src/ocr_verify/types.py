"""跨模块共享的核心数据类型。

所有引擎返回统一的 OCRResult,这是"依赖倒置"的落点:
上层 router / verifier / 断言库都只依赖这个契约,不关心底层是
Tesseract、PaddleOCR 还是多模态模型。换引擎只需实现同一接口。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class EngineType(str, Enum):
    """识别结果的来源层。继承 str 使其可直接 JSON 序列化。"""

    TESSERACT = "tesseract"       # 基线对照组
    PADDLE = "paddle"             # 第一层:快速通道
    VLM = "vlm"                   # 第二层:多模态复核
    CACHE = "cache"               # 缓存命中,未实际推理
    NONE = "none"                 # 全部失败时的空结果


class EscalationReason(str, Enum):
    """触发第二层复核的原因。

    刻意区分多种原因而非只用置信度,因为置信度衡量的是
    "模型对自己输出的确信程度",而不是"输出是否正确" ——
    文本被完全遮挡时,模型可能对一个错误结果给出很高的置信度。
    """

    LOW_CONFIDENCE = "low_confidence"        # 置信度低于阈值
    EMPTY_RESULT = "empty_result"            # 检测到文本框但识别为空
    PATTERN_MISMATCH = "pattern_mismatch"    # 不符合调用方指定的预期格式
    BOX_OVERLAP = "box_overlap"              # 文本框大面积重叠,疑似弹窗遮挡
    FORCED = "forced"                        # 调用方强制要求走二层


@dataclass
class TextBox:
    """单个文本框的识别结果。

    box 为四点坐标 [(x1,y1), (x2,y2), (x3,y3), (x4,y4)],
    顺序为左上->右上->右下->左下。之所以不用简单的矩形 (x,y,w,h),
    是因为 OCR 检测出的文本区域可能是倾斜的四边形。
    """

    text: str
    confidence: float
    box: Optional[list[tuple[float, float]]] = None

    def area(self) -> float:
        """用鞋带公式(Shoelace formula)计算多边形面积。

        面积 = |Σ(x_i * y_{i+1} - x_{i+1} * y_i)| / 2
        用于后续判断文本框之间的重叠程度。
        """
        if not self.box or len(self.box) < 3:
            return 0.0
        pts = self.box
        n = len(pts)
        total = 0.0
        for i in range(n):
            x1, y1 = pts[i]
            x2, y2 = pts[(i + 1) % n]
            total += x1 * y2 - x2 * y1
        return abs(total) / 2.0

    def bbox(self) -> tuple[float, float, float, float]:
        """返回外接矩形 (x_min, y_min, x_max, y_max)。"""
        if not self.box:
            return (0.0, 0.0, 0.0, 0.0)
        xs = [p[0] for p in self.box]
        ys = [p[1] for p in self.box]
        return (min(xs), min(ys), max(xs), max(ys))

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "confidence": round(self.confidence, 4),
            "box": self.box,
        }


@dataclass
class OCRResult:
    """统一的识别结果契约。

    这个类是整个系统的"窄腰" —— 所有引擎向上输出它,
    所有上层逻辑向下只消费它。
    """

    text: str = ""
    confidence: float = 0.0
    boxes: list[TextBox] = field(default_factory=list)
    engine: EngineType = EngineType.NONE
    elapsed_ms: float = 0.0

    # --- 链路元信息:用于可观测性与问题定位 ---
    from_cache: bool = False
    cache_kind: Optional[str] = None          # "exact"(MD5) | "perceptual"(dHash)
    escalated: bool = False                   # 是否走了第二层
    escalation_reason: Optional[EscalationReason] = None
    degraded: bool = False                    # 二层失败回落到一层
    first_layer_text: Optional[str] = None    # 升级前一层的结果,便于对比分析
    first_layer_confidence: Optional[float] = None
    error: Optional[str] = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()

    @property
    def succeeded(self) -> bool:
        return self.error is None and not self.is_empty

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "confidence": round(self.confidence, 4),
            "engine": self.engine.value,
            "elapsed_ms": round(self.elapsed_ms, 2),
            "from_cache": self.from_cache,
            "cache_kind": self.cache_kind,
            "escalated": self.escalated,
            "escalation_reason": (
                self.escalation_reason.value if self.escalation_reason else None
            ),
            "degraded": self.degraded,
            "first_layer_text": self.first_layer_text,
            "first_layer_confidence": (
                round(self.first_layer_confidence, 4)
                if self.first_layer_confidence is not None
                else None
            ),
            "error": self.error,
            "box_count": len(self.boxes),
            "extra": self.extra,
        }

    def summary(self) -> str:
        """单行摘要,用于日志和断言失败信息。"""
        parts = [f"[{self.engine.value}]"]
        if self.from_cache:
            parts.append(f"cache:{self.cache_kind}")
        if self.escalated:
            parts.append(f"escalated:{self.escalation_reason.value}")  # type: ignore
        if self.degraded:
            parts.append("DEGRADED")
        parts.append(f"conf={self.confidence:.2f}")
        parts.append(f"{self.elapsed_ms:.0f}ms")
        text_preview = self.text[:60].replace("\n", " ⏎ ")
        if len(self.text) > 60:
            text_preview += "..."
        parts.append(f'text="{text_preview}"')
        return " ".join(parts)


class OCREngineError(Exception):
    """引擎层面的可恢复错误。

    单独定义异常类型,使 router 能够区分
    "引擎坏了(应降级)" 和 "程序 bug(应抛出)"。
    """

    def __init__(self, message: str, engine: str = "", retryable: bool = True) -> None:
        super().__init__(message)
        self.engine = engine
        self.retryable = retryable
