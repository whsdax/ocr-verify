"""路由器决策逻辑的单元测试。"""

import re

import pytest

from ocr_verify.engines.base import MockEngine
from ocr_verify.router import TwoLayerRouter
from ocr_verify.types import EngineType, EscalationReason, OCRResult, TextBox


def _textbox(text: str, conf: float, box: list[tuple[float, float]]) -> TextBox:
    return TextBox(text=text, confidence=conf, box=box)


def test_high_confidence_no_escalation():
    first = MockEngine(text="登录成功", confidence=0.95, engine_type=EngineType.PADDLE)
    router = TwoLayerRouter(first_layer=first, second_layer=None)
    res = router.route("img")
    assert not res.escalated
    assert res.text == "登录成功"


def test_low_confidence_triggers_escalation():
    first = MockEngine(text="登录成功", confidence=0.5, engine_type=EngineType.PADDLE)
    second = MockEngine(text="登录成功", confidence=0.95, engine_type=EngineType.VLM)
    router = TwoLayerRouter(first_layer=first, second_layer=second)
    res = router.route("img")
    assert res.escalated
    assert res.escalation_reason == EscalationReason.LOW_CONFIDENCE
    assert res.engine == EngineType.VLM


def test_empty_result_triggers_escalation():
    first = MockEngine(text="", confidence=0.0, engine_type=EngineType.PADDLE)
    second = MockEngine(text="登录成功", confidence=0.9, engine_type=EngineType.VLM)
    router = TwoLayerRouter(
        first_layer=first,
        second_layer=second,
        confidence_threshold=0.0,  # 让空结果优先触发
    )
    res = router.route("img")
    assert res.escalated
    assert res.escalation_reason == EscalationReason.EMPTY_RESULT


def test_pattern_mismatch_triggers_escalation():
    first = MockEngine(text="余额12800", confidence=0.92, engine_type=EngineType.PADDLE)
    second = MockEngine(text="余额128.00", confidence=0.95, engine_type=EngineType.VLM)
    router = TwoLayerRouter(
        first_layer=first,
        second_layer=second,
        confidence_threshold=0.95,  # 让模式不匹配优先触发
    )
    res = router.route("img", expected_pattern=r"\d+\.\d{2}")
    assert res.escalated
    assert res.escalation_reason == EscalationReason.PATTERN_MISMATCH


def test_box_overlap_triggers_escalation():
    # 两个文本框高度重叠
    boxes = [
        _textbox("确认删除", 0.85, [(0, 0), (100, 0), (100, 30), (0, 30)]),
        _textbox("删除文件", 0.88, [(5, 5), (105, 5), (105, 35), (5, 35)]),
    ]
    first = MockEngine(text="确认删除\n删除文件", confidence=0.85, engine_type=EngineType.PADDLE)
    first._results = [OCRResult(text="确认删除\n删除文件", confidence=0.85, boxes=boxes, engine=EngineType.PADDLE)]
    second = MockEngine(text="确认删除该文件", confidence=0.95, engine_type=EngineType.VLM)
    router = TwoLayerRouter(
        first_layer=first,
        second_layer=second,
        confidence_threshold=0.95,
    )
    res = router.route("img")
    assert res.escalated
    assert res.escalation_reason == EscalationReason.BOX_OVERLAP


def test_second_layer_failure_fallback():
    first = MockEngine(text="登录成功", confidence=0.5, engine_type=EngineType.PADDLE)
    second = MockEngine(
        text="",
        confidence=0.0,
        engine_type=EngineType.VLM,
        raise_error=RuntimeError("API 超时"),
    )
    router = TwoLayerRouter(first_layer=first, second_layer=second)
    res = router.route("img")
    assert res.degraded
    assert res.text == "登录成功"  # 回落第一层
    assert res.escalation_reason == EscalationReason.LOW_CONFIDENCE


def test_second_layer_unavailable_fallback():
    first = MockEngine(text="登录成功", confidence=0.5, engine_type=EngineType.PADDLE)
    second = MockEngine(text="登录成功", confidence=0.95, engine_type=EngineType.VLM, available=False)
    router = TwoLayerRouter(first_layer=first, second_layer=second)
    res = router.route("img")
    assert res.degraded
    assert not res.escalated
    assert res.text == "登录成功"


def test_stats_counts():
    first = MockEngine(text="ok", confidence=0.5, engine_type=EngineType.PADDLE)
    second = MockEngine(text="ok", confidence=0.9, engine_type=EngineType.VLM)
    router = TwoLayerRouter(first_layer=first, second_layer=second)
    router.route("a")
    router.route("b")
    s = router.stats()
    assert s["total_requests"] == 2
    assert s["total_escalations"] == 2
    assert s["escalation_by_reason"][EscalationReason.LOW_CONFIDENCE.value] == 2
