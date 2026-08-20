"""双层路由决策 —— 系统的大脑。

核心职责:判断第一层的结果是否可信,不可信则升级到第二层。

为什么升级条件不能只看置信度
----------------------------
这是本项目最重要的设计洞察。

置信度衡量的是"模型对自己输出的确信程度",而**不是**"输出是否正确"。
两者在正常场景下高度相关,但在异常场景下会解耦:

  场景 A(遮挡):弹窗盖住了"确认删除"的后两个字,OCR 只看到"确认",
    它会以 0.98 的高置信度返回"确认" —— 模型对自己看到的部分很确信,
    但这个结果对断言而言是错的。

  场景 B(检测失败):检测器找到了文本框,但识别器输出空串。
    此时置信度可能是 0(会触发)也可能因为框被过滤而根本不产生记录(不会触发)。

  场景 C(格式错乱):期望是金额 "¥128.00",识别成 "¥12800"。
    置信度很高,但小数点丢了 —— 对金额断言是致命错误。

所以本路由器实现了四条独立的升级条件,任一满足即升级:
  1. 低置信度      —— 覆盖常规不确定场景
  2. 空结果        —— 覆盖检测成功但识别失败
  3. 格式不匹配    —— 覆盖高置信度的结构性错误(需调用方提供 expected_pattern)
  4. 文本框重叠    —— 覆盖弹窗遮挡(遮挡层与底层文字的框会大面积交叠)

降级哲学
--------
第二层是外部依赖,一定会失败。设计原则是:
**第二层的失败不能传导为整条自动化用例的失败。**
失败时回落第一层结果并打上 degraded=True 标记,
让调用方知道"这个结果质量存疑"但仍然拿得到数据。
可观测性比可用性洁癖更重要。
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional, Pattern

from .engines.base import OCREngine
from .types import (
    EngineType,
    EscalationReason,
    OCRResult,
    TextBox,
)

logger = logging.getLogger(__name__)


class RoutingDecision:
    """一次路由决策的结果,便于测试与日志追溯。"""

    __slots__ = ("should_escalate", "reason", "detail")

    def __init__(
        self,
        should_escalate: bool,
        reason: Optional[EscalationReason] = None,
        detail: str = "",
    ) -> None:
        self.should_escalate = should_escalate
        self.reason = reason
        self.detail = detail

    def __repr__(self) -> str:
        if not self.should_escalate:
            return "RoutingDecision(keep_first_layer)"
        return f"RoutingDecision(escalate, reason={self.reason.value}, {self.detail})"  # type: ignore


class TwoLayerRouter:
    """双层 OCR 路由器。

    Parameters
    ----------
    first_layer:
        快速引擎(PaddleOCR)。必需。
    second_layer:
        复核引擎(多模态模型)。可为 None —— 此时退化为纯单层,
        便于在没有 API Key 的环境下依然能跑通全流程。
    """

    def __init__(
        self,
        first_layer: OCREngine,
        second_layer: Optional[OCREngine] = None,
        confidence_threshold: float = 0.7,
        escalate_on_empty: bool = True,
        escalate_on_pattern_mismatch: bool = True,
        escalate_on_box_overlap: bool = True,
        box_overlap_threshold: float = 0.35,
        fallback_to_first_layer: bool = True,
    ) -> None:
        self.first_layer = first_layer
        self.second_layer = second_layer
        self.confidence_threshold = confidence_threshold
        self.escalate_on_empty = escalate_on_empty
        self.escalate_on_pattern_mismatch = escalate_on_pattern_mismatch
        self.escalate_on_box_overlap = escalate_on_box_overlap
        self.box_overlap_threshold = box_overlap_threshold
        self.fallback_to_first_layer = fallback_to_first_layer

        # 路由统计:分原因计数,用于评测报告中分析"升级主要由什么触发"
        self.total_requests = 0
        self.escalation_counts: dict[str, int] = {r.value: 0 for r in EscalationReason}
        self.degraded_count = 0

    # ------------------------------------------------------------------
    # 决策逻辑
    # ------------------------------------------------------------------

    def decide(
        self,
        result: OCRResult,
        expected_pattern: Optional[str | Pattern[str]] = None,
        force_escalate: bool = False,
    ) -> RoutingDecision:
        """判断第一层结果是否需要升级。纯函数,无副作用,便于单测。"""

        if force_escalate:
            return RoutingDecision(True, EscalationReason.FORCED, "调用方强制指定")

        # 条件 1:第一层直接报错 —— 按低置信度处理,交给第二层兜底
        if result.error:
            return RoutingDecision(
                True, EscalationReason.LOW_CONFIDENCE, f"第一层报错: {result.error}"
            )

        # 条件 2:空结果。区分两种情况:
        #   - 检测到框但文本为空 -> 识别器失效,第二层大概率能救回来
        #   - 连框都没有         -> 图上可能真的没字,升级也是浪费
        # 但保守起见仍然升级:UI 自动化中"整屏无字"极其罕见,
        # 更可能是检测器被异常渲染干扰了。
        if self.escalate_on_empty and result.is_empty:
            return RoutingDecision(
                True,
                EscalationReason.EMPTY_RESULT,
                f"识别结果为空(检测到 {len(result.boxes)} 个文本框)",
            )

        # 条件 3:格式不匹配。仅在调用方明确给出预期格式时生效。
        # 这条的价值在于捕获"高置信度的错误",是纯置信度方案的盲区。
        if self.escalate_on_pattern_mismatch and expected_pattern is not None:
            pattern = (
                re.compile(expected_pattern)
                if isinstance(expected_pattern, str)
                else expected_pattern
            )
            if not pattern.search(result.text):
                return RoutingDecision(
                    True,
                    EscalationReason.PATTERN_MISMATCH,
                    f"不匹配预期格式 {pattern.pattern!r}",
                )

        # 条件 4:文本框重叠 —— 弹窗遮挡的几何特征。
        # 正常 UI 布局中文本框互不重叠;一旦出现大面积交叠,
        # 通常意味着有半透明蒙层、悬浮弹窗或渲染错位。
        if self.escalate_on_box_overlap and len(result.boxes) >= 2:
            overlap = self._max_overlap_ratio(result.boxes)
            if overlap > self.box_overlap_threshold:
                return RoutingDecision(
                    True,
                    EscalationReason.BOX_OVERLAP,
                    f"最大文本框重叠率 {overlap:.1%} 超过阈值 {self.box_overlap_threshold:.1%}",
                )

        # 条件 5:置信度不足 —— 放在最后判断,因为前面几条能给出更具体的原因,
        # 对问题定位更有帮助。
        if result.confidence < self.confidence_threshold:
            return RoutingDecision(
                True,
                EscalationReason.LOW_CONFIDENCE,
                f"置信度 {result.confidence:.3f} < 阈值 {self.confidence_threshold}",
            )

        return RoutingDecision(False)

    @staticmethod
    def _max_overlap_ratio(boxes: list[TextBox]) -> float:
        """计算所有文本框两两之间的最大重叠比例。

        比例定义为 **交集面积 / 较小框的面积**,而非标准 IoU。
        原因:IoU 在"小框完全落在大框内"时数值很低(因为并集很大),
        但这恰恰是遮挡最典型的形态 —— 一个小弹窗盖在大段文字上。
        用较小框做分母能准确捕获这种包含关系(此时比例接近 1)。

        复杂度 O(n²)。UI 截图的文本框通常在几十个量级,
        实测 50 个框耗时 < 1ms,无需空间索引优化。
        """
        rects: list[tuple[float, float, float, float]] = []
        for b in boxes:
            if b.box:
                rects.append(b.bbox())

        if len(rects) < 2:
            return 0.0

        max_ratio = 0.0
        for i in range(len(rects)):
            for j in range(i + 1, len(rects)):
                ratio = TwoLayerRouter._overlap_ratio(rects[i], rects[j])
                if ratio > max_ratio:
                    max_ratio = ratio
        return max_ratio

    @staticmethod
    def _overlap_ratio(
        a: tuple[float, float, float, float], b: tuple[float, float, float, float]
    ) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b

        inter_w = min(ax2, bx2) - max(ax1, bx1)
        inter_h = min(ay2, by2) - max(ay1, by1)
        if inter_w <= 0 or inter_h <= 0:
            return 0.0

        inter_area = inter_w * inter_h
        area_a = max((ax2 - ax1) * (ay2 - ay1), 1e-9)
        area_b = max((bx2 - bx1) * (by2 - by1), 1e-9)
        return inter_area / min(area_a, area_b)

    # ------------------------------------------------------------------
    # 执行
    # ------------------------------------------------------------------

    def route(
        self,
        image: Any,
        expected_pattern: Optional[str | Pattern[str]] = None,
        force_escalate: bool = False,
        **kwargs: Any,
    ) -> OCRResult:
        """执行完整的双层识别流程。"""
        self.total_requests += 1

        # ---- 第一层 ----
        first = self.first_layer.recognize(image, **kwargs)
        decision = self.decide(first, expected_pattern, force_escalate)

        if not decision.should_escalate:
            logger.debug("第一层结果可信,不升级 | %s", first.summary())
            return first

        reason = decision.reason
        self.escalation_counts[reason.value] += 1  # type: ignore[union-attr]
        logger.info("触发第二层复核 | %s", decision.detail)

        # ---- 第二层不可用:直接返回第一层,标记降级 ----
        if self.second_layer is None or not self.second_layer.is_available():
            first.escalated = False
            first.degraded = True
            first.extra["escalation_skipped"] = (
                "第二层未配置或不可用,已回落第一层结果"
            )
            first.extra["would_escalate_reason"] = reason.value  # type: ignore[union-attr]
            self.degraded_count += 1
            logger.warning("第二层不可用,回落第一层结果")
            return first

        # ---- 第二层 ----
        second = self.second_layer.recognize(
            image, expected_pattern=expected_pattern, **kwargs
        )

        # 保留第一层信息用于对比分析 —— 这是评测时判断
        # "第二层到底救回了多少" 的关键数据
        second.escalated = True
        second.escalation_reason = reason
        second.first_layer_text = first.text
        second.first_layer_confidence = first.confidence
        # 累加两层耗时,反映用户实际感知的延迟
        second.elapsed_ms += first.elapsed_ms

        # ---- 第二层失败:降级 ----
        if second.error or second.is_empty:
            self.degraded_count += 1
            if self.fallback_to_first_layer:
                logger.warning(
                    "第二层失败(%s),回落第一层结果", second.error or "返回空"
                )
                first.escalated = True
                first.escalation_reason = reason
                first.degraded = True
                first.error = None  # 第一层本身是成功的,不应把二层的错误挂上去
                first.extra["second_layer_error"] = second.error or "empty_result"
                first.elapsed_ms += second.elapsed_ms
                return first

            second.degraded = True
            return second

        logger.debug("第二层复核完成 | %s", second.summary())
        return second

    # ------------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        total_escalations = sum(self.escalation_counts.values())
        rate = (
            total_escalations / self.total_requests if self.total_requests else 0.0
        )
        return {
            "total_requests": self.total_requests,
            "total_escalations": total_escalations,
            "escalation_rate": round(rate, 4),
            "escalation_by_reason": dict(self.escalation_counts),
            "degraded_count": self.degraded_count,
            "confidence_threshold": self.confidence_threshold,
        }

    def reset_stats(self) -> None:
        self.total_requests = 0
        self.escalation_counts = {r.value: 0 for r in EscalationReason}
        self.degraded_count = 0
