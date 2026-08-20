"""OCR 引擎抽象基类。

为什么要抽象这一层
------------------
1. **可替换**:Tesseract / PaddleOCR / 多模态模型 / Mock 对上层完全同构,
   router 不需要知道自己调的是谁。
2. **可测试**:单元测试注入 MockEngine,不消耗 API 额度、不依赖模型文件,
   CI 可以秒级跑完。
3. **可扩展**:未来加第三层(比如人工兜底队列)只需实现同一接口。

这就是依赖倒置:高层模块(router)依赖抽象(OCREngine),
而不是依赖具体实现(PaddleOCR)。
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, Optional

from ..types import EngineType, OCREngineError, OCRResult, TextBox

# 避免循环导入:fingerprint 中的 ImageInput 类型别名在此重新声明
from ..cache.fingerprint import ImageInput, load_image  # noqa: F401


class OCREngine(ABC):
    """所有 OCR 引擎的统一契约。

    子类只需实现 :meth:`_recognize_impl`,
    计时、异常包装、结果规范化由基类的 :meth:`recognize` 统一处理 ——
    这样每个引擎不必重复这些样板代码,也保证了指标口径一致。
    """

    #: 该引擎在结果中的标识
    engine_type: EngineType = EngineType.NONE

    #: 人类可读名称,用于报告展示
    display_name: str = "base"

    def __init__(self, **kwargs: Any) -> None:
        self._initialized = False
        self._init_error: Optional[str] = None

    # ------------------------------------------------------------------
    # 子类必须实现
    # ------------------------------------------------------------------

    @abstractmethod
    def _recognize_impl(self, image: ImageInput, **kwargs: Any) -> OCRResult:
        """真正的识别逻辑。允许抛异常,基类会捕获并转为带 error 的结果。"""

    # ------------------------------------------------------------------
    # 子类可选覆盖
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """引擎是否可用(依赖已安装、模型已就绪、密钥已配置)。

        router 用它来决定是否跳过该层,而不是等到调用时才失败 ——
        提前判断能省掉一次无谓的超时等待。
        """
        return True

    def warmup(self) -> None:
        """预热。首次推理通常包含模型加载,耗时远高于稳态。

        评测前调用一次,避免第一张图的耗时污染延迟统计 ——
        这是性能测试的基本规范,否则 P95 数据会失真。
        """
        return None

    def close(self) -> None:
        """释放资源(HTTP 连接池、模型显存等)。"""
        return None

    # ------------------------------------------------------------------
    # 模板方法:统一计时与异常处理
    # ------------------------------------------------------------------

    def recognize(self, image: ImageInput, **kwargs: Any) -> OCRResult:
        """执行识别。**本方法不抛异常**,失败信息写入 result.error。

        不抛异常是刻意设计:OCR 是自动化链路中的一环,
        一个引擎挂掉不应该让整条用例崩溃,而应该让 router 有机会降级。
        真正的编程错误(如参数类型错)仍会以 error 形式暴露,不会被静默吞掉。
        """
        start = time.perf_counter()
        try:
            result = self._recognize_impl(image, **kwargs)
        except OCREngineError as exc:
            return OCRResult(
                engine=self.engine_type,
                elapsed_ms=(time.perf_counter() - start) * 1000,
                error=f"{type(exc).__name__}: {exc}",
            )
        except Exception as exc:  # noqa: BLE001 - 兜底,防止未知异常炸穿调用链
            return OCRResult(
                engine=self.engine_type,
                elapsed_ms=(time.perf_counter() - start) * 1000,
                error=f"{type(exc).__name__}: {exc}",
            )

        # 统一补齐元信息,子类无需关心
        result.elapsed_ms = (time.perf_counter() - start) * 1000
        if result.engine is EngineType.NONE:
            result.engine = self.engine_type
        return result

    # ------------------------------------------------------------------
    # 供子类复用的工具方法
    # ------------------------------------------------------------------

    @staticmethod
    def _aggregate(boxes: list[TextBox], joiner: str = "\n") -> tuple[str, float]:
        """将多个文本框聚合为整体文本与整体置信度。

        整体置信度取**最小值**而非平均值。
        理由:自动化断言关心的是"这段文本是否可信",
        只要有一个字段识别得心虚,整体就应该被判为不可信。
        取平均会让一个 0.2 的错误框被三个 0.95 的正确框稀释掉,
        从而错过本该触发的第二层复核 —— 这是漏检,比误检代价更高。
        """
        if not boxes:
            return "", 0.0
        text = joiner.join(b.text for b in boxes if b.text)
        confidence = min(b.confidence for b in boxes)
        return text, confidence

    def __repr__(self) -> str:
        status = "ready" if self.is_available() else "unavailable"
        return f"<{type(self).__name__} name={self.display_name} status={status}>"


class MockEngine(OCREngine):
    """测试专用的假引擎。

    支持三种模式:
      - 固定返回:每次都返回同一结果
      - 序列返回:按调用次序依次返回预设结果(用于测试降级、重试路径)
      - 抛异常:模拟引擎故障

    有了它,单元测试完全不需要真实模型或网络,CI 可以在几秒内跑完全部用例。
    """

    engine_type = EngineType.NONE
    display_name = "mock"

    def __init__(
        self,
        text: str = "mock text",
        confidence: float = 0.9,
        engine_type: EngineType = EngineType.NONE,
        results: Optional[list[OCRResult]] = None,
        raise_error: Optional[Exception] = None,
        delay_ms: float = 0.0,
        available: bool = True,
    ) -> None:
        super().__init__()
        self._text = text
        self._confidence = confidence
        self.engine_type = engine_type
        self._results = list(results) if results else None
        self._raise_error = raise_error
        self._delay_ms = delay_ms
        self._available = available
        self.call_count = 0

    def is_available(self) -> bool:
        return self._available

    def _recognize_impl(self, image: ImageInput, **kwargs: Any) -> OCRResult:
        self.call_count += 1

        if self._delay_ms:
            time.sleep(self._delay_ms / 1000)

        if self._raise_error is not None:
            raise self._raise_error

        if self._results:
            # 序列耗尽后重复最后一个,避免测试写起来太啰嗦
            idx = min(self.call_count - 1, len(self._results) - 1)
            return self._results[idx]

        return OCRResult(
            text=self._text,
            confidence=self._confidence,
            boxes=[TextBox(text=self._text, confidence=self._confidence)],
            engine=self.engine_type,
        )
