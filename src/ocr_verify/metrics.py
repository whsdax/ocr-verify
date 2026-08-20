"""评测指标 —— 准确率如何定义与计算。

指标口径是评测的生命线
----------------------
"识别成功率 95%" 这句话本身是没有意义的,除非说清楚:
  - 什么算"成功"?完全一字不差,还是允许部分正确?
  - 空格、换行、标点算不算?
  - 大小写敏感吗?

本模块定义三个层次的指标,分别回答不同的问题:

1. **精确匹配率 (Exact Match)**
   识别结果与标注**完全一致**的样本占比。
   最严格,最贴近"UI 断言是否通过"的真实业务语义 ——
   自动化脚本里 `assert text == "登录成功"` 就是精确匹配。

2. **字符准确率 (1 - CER)**
   基于编辑距离。CER = 编辑距离 / 标注长度。
   能反映"错了多少",而非只有对错二元。
   一张图错 1 个字和错 20 个字,精确匹配都判 0 分,但 CER 能区分。

3. **包含匹配率 (Contains)**
   标注文本是否作为子串出现在识别结果中。
   对应 `assert "登录成功" in text` 这类宽松断言。

三个指标一起看才完整:
  精确匹配高 = 可以直接做严格断言
  CER 低但精确匹配低 = 大方向对,可能是标点/空格差异,可通过归一化改善
  包含匹配高但精确匹配低 = 识别出了多余内容(常见于第二层模型加了解释性文字)
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional


# ----------------------------------------------------------------------
# 文本归一化
# ----------------------------------------------------------------------

# 中英文标点映射。UI 文本里全角半角混用极其常见,
# 这类差异对断言语义无影响,但会严重拉低精确匹配率,属于"假失败"。
_PUNCT_MAP = str.maketrans({
    "，": ",", "。": ".", "！": "!", "？": "?", "；": ";", "：": ":",
    "（": "(", "）": ")", "【": "[", "】": "]", "《": "<", "》": ">",
    "“": '"', "”": '"', "‘": "'", "’": "'", "、": ",", "～": "~",
    "％": "%", "＃": "#", "＠": "@", "－": "-", "　": " ",
})


def normalize_text(
    text: str,
    strip_space: bool = True,
    lowercase: bool = False,
    unify_punct: bool = True,
) -> str:
    """文本归一化。

    Parameters
    ----------
    strip_space:
        移除所有空白字符。中文 OCR 中空格位置极不稳定
        (同一张图不同引擎可能给出 "确定 取消" 或 "确定取消"),
        且对语义无影响,默认移除。
    lowercase:
        默认**不**转小写。UI 中大小写常有语义(如 "OK" vs "Ok" 是不同控件),
        且英文品牌名的大小写是识别质量的一部分。
    unify_punct:
        全角标点转半角。

    注意:归一化策略必须在评测报告中明确写出。
    不同的归一化会得到不同的数字,不说清楚就是耍流氓。
    """
    if not text:
        return ""

    # NFKC 规范化:处理全角字母数字(ＡＢＣ -> ABC)、兼容字符等
    result = unicodedata.normalize("NFKC", text)

    if unify_punct:
        result = result.translate(_PUNCT_MAP)

    if strip_space:
        result = re.sub(r"\s+", "", result)
    else:
        result = re.sub(r"\s+", " ", result).strip()

    if lowercase:
        result = result.lower()

    return result


# ----------------------------------------------------------------------
# 编辑距离
# ----------------------------------------------------------------------

def levenshtein(a: str, b: str) -> int:
    """Levenshtein 编辑距离(动态规划,滚动数组优化)。

    朴素 DP 需要 O(m*n) 空间,但转移只依赖上一行,
    因此用两个一维数组滚动即可,空间降到 O(min(m,n))。
    对 OCR 场景(文本通常 < 500 字符)性能完全够用,
    无需引入 python-Levenshtein 这类 C 扩展依赖。

    时间 O(m*n),空间 O(min(m,n))。
    """
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    # 让 b 是较短的那个,减少内层数组长度
    if len(a) < len(b):
        a, b = b, a

    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            replace_cost = previous[j - 1] + (ca != cb)
            current[j] = min(insert_cost, delete_cost, replace_cost)
        previous = current

    return previous[-1]


def char_error_rate(prediction: str, ground_truth: str) -> float:
    """字符错误率 CER = 编辑距离 / 标注长度。

    标注为空时:预测也为空返回 0(完全正确),否则返回 1(全错)。
    CER 可能 > 1(预测比标注长很多时),这是正常的,不做截断 ——
    截断会掩盖"模型输出了大段无关内容"这个严重问题。
    """
    if not ground_truth:
        return 0.0 if not prediction else 1.0
    return levenshtein(prediction, ground_truth) / len(ground_truth)


def char_accuracy(prediction: str, ground_truth: str) -> float:
    """字符准确率 = 1 - CER,下限截断到 0。"""
    return max(0.0, 1.0 - char_error_rate(prediction, ground_truth))


# ----------------------------------------------------------------------
# 单样本评估
# ----------------------------------------------------------------------

@dataclass
class SampleEvaluation:
    """单个样本的评测结果。"""

    sample_id: str
    prediction: str
    ground_truth: str
    exact_match: bool
    contains_match: bool
    cer: float
    char_acc: float
    confidence: float = 0.0
    elapsed_ms: float = 0.0
    engine: str = ""
    escalated: bool = False
    from_cache: bool = False
    degraded: bool = False
    perturbation: str = "none"      # 扰动类型,用于分维度统计
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "prediction": self.prediction,
            "ground_truth": self.ground_truth,
            "exact_match": self.exact_match,
            "contains_match": self.contains_match,
            "cer": round(self.cer, 4),
            "char_acc": round(self.char_acc, 4),
            "confidence": round(self.confidence, 4),
            "elapsed_ms": round(self.elapsed_ms, 2),
            "engine": self.engine,
            "escalated": self.escalated,
            "from_cache": self.from_cache,
            "degraded": self.degraded,
            "perturbation": self.perturbation,
            "error": self.error,
        }


def evaluate_sample(
    sample_id: str,
    prediction: str,
    ground_truth: str,
    normalize: bool = True,
    **meta: Any,
) -> SampleEvaluation:
    """评估单个样本。"""
    pred = normalize_text(prediction) if normalize else prediction
    truth = normalize_text(ground_truth) if normalize else ground_truth

    cer = char_error_rate(pred, truth)
    return SampleEvaluation(
        sample_id=sample_id,
        prediction=prediction,          # 保留原文便于人工复查
        ground_truth=ground_truth,
        exact_match=(pred == truth),
        contains_match=(truth in pred) if truth else False,
        cer=cer,
        char_acc=max(0.0, 1.0 - cer),
        **meta,
    )


# ----------------------------------------------------------------------
# 聚合统计
# ----------------------------------------------------------------------

@dataclass
class AggregateMetrics:
    """一组样本的聚合指标。"""

    name: str
    sample_count: int = 0
    exact_match_rate: float = 0.0
    contains_match_rate: float = 0.0
    mean_char_acc: float = 0.0
    mean_cer: float = 0.0
    mean_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    escalation_count: int = 0
    cache_hit_count: int = 0
    degraded_count: int = 0
    error_count: int = 0
    by_perturbation: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "sample_count": self.sample_count,
            "exact_match_rate": round(self.exact_match_rate, 4),
            "contains_match_rate": round(self.contains_match_rate, 4),
            "mean_char_acc": round(self.mean_char_acc, 4),
            "mean_cer": round(self.mean_cer, 4),
            "mean_latency_ms": round(self.mean_latency_ms, 2),
            "p50_latency_ms": round(self.p50_latency_ms, 2),
            "p95_latency_ms": round(self.p95_latency_ms, 2),
            "escalation_count": self.escalation_count,
            "cache_hit_count": self.cache_hit_count,
            "degraded_count": self.degraded_count,
            "error_count": self.error_count,
            "by_perturbation": self.by_perturbation,
        }


def percentile(values: list[float], p: float) -> float:
    """线性插值百分位数。

    不用 numpy.percentile 是为了让 metrics 模块保持零依赖,
    便于单独复用。样本量小时两者结果一致。
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]

    k = (len(ordered) - 1) * p
    lower = int(k)
    upper = min(lower + 1, len(ordered) - 1)
    weight = k - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def aggregate(name: str, evaluations: Iterable[SampleEvaluation]) -> AggregateMetrics:
    """聚合多个样本的评测结果,并按扰动类型分组统计。

    分维度统计是本项目的关键论证手段:
    如果双层方案的收益主要来自"遮挡"和"模糊"两个维度,
    而在"正常"维度上与单层持平 —— 那就精确验证了设计假设,
    比一个笼统的总分有说服力得多。
    """
    evals = list(evaluations)
    if not evals:
        return AggregateMetrics(name=name)

    n = len(evals)
    latencies = [e.elapsed_ms for e in evals]

    metrics = AggregateMetrics(
        name=name,
        sample_count=n,
        exact_match_rate=sum(e.exact_match for e in evals) / n,
        contains_match_rate=sum(e.contains_match for e in evals) / n,
        mean_char_acc=sum(e.char_acc for e in evals) / n,
        mean_cer=sum(e.cer for e in evals) / n,
        mean_latency_ms=sum(latencies) / n,
        p50_latency_ms=percentile(latencies, 0.50),
        p95_latency_ms=percentile(latencies, 0.95),
        escalation_count=sum(e.escalated for e in evals),
        cache_hit_count=sum(e.from_cache for e in evals),
        degraded_count=sum(e.degraded for e in evals),
        error_count=sum(e.error is not None for e in evals),
    )

    # 按扰动类型分组
    groups: dict[str, list[SampleEvaluation]] = {}
    for e in evals:
        groups.setdefault(e.perturbation, []).append(e)

    for key, items in sorted(groups.items()):
        k = len(items)
        metrics.by_perturbation[key] = {
            "count": k,
            "exact_match_rate": round(sum(i.exact_match for i in items) / k, 4),
            "mean_char_acc": round(sum(i.char_acc for i in items) / k, 4),
            "escalation_count": sum(i.escalated for i in items),
            "mean_latency_ms": round(sum(i.elapsed_ms for i in items) / k, 2),
        }

    return metrics
