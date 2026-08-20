"""OCR 结果缓存 —— 指纹 + LRU + 感知近似查找。

这一层挂在所有引擎之前(而不是只挡 VLM),
因此它能同时省掉 PaddleOCR 和 VLM 的推理开销。
在 UI 自动化中,同一页面被反复截图断言是常态,
"同图重复计算"是隐性成本的主要来源,缓存的边际收益非常大。

两级查找策略
------------
1. **精确查找(MD5)**:O(1),零误判。字节完全相同的图直接命中。
2. **感知查找(dHash)**:在精确 Miss 后,线性扫描缓存中所有条目,
   寻找汉明距离 ≤ threshold 的近似图。命中表示"视觉几乎一样",
   直接复用结果。

为什么要线性扫描做感知查找
--------------------------
感知哈希的近似查找本质是个近邻搜索问题。精确哈希可以 O(1) 索引,
但近似哈希没有好的 O(1) 索引结构(除非引入 LSH 这类额外复杂度)。
考虑到缓存容量通常在几百到几千,线性扫描是 O(n) 但 n 很小,
实测在 capacity=512 时单次扫描 < 0.2ms,完全够用,且实现简单、无外部依赖。
如果未来容量扩展到万级,再升级到 LSH 也不迟 —— 过早优化是万恶之源。

命中即"省了一次调用"
--------------------
统计里的 `saved_vlm_calls` / `saved_cost` 是给老板看的关键指标:
缓存每命中一次,就少花一次最高成本的 VLM 调用。
这部分节省构成了双层方案成本可控的核心论据之一。
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from ..types import OCRResult
from .fingerprint import ImageFingerprint
from .lru import LRUCache

logger = logging.getLogger(__name__)


class OCRResultCache:
    """图片 -> 识别结果的双层指纹缓存。"""

    def __init__(
        self,
        capacity: int = 512,
        use_perceptual: bool = True,
        hamming_threshold: int = 3,
        dhash_size: int = 8,
        vlm_cost_per_call: float = 0.003,
        enabled: bool = True,
    ) -> None:
        self.enabled = enabled
        self.use_perceptual = use_perceptual
        self._hamming_threshold = hamming_threshold
        self._dhash_size = dhash_size
        self.vlm_cost_per_call = vlm_cost_per_call

        # 精确查找。key 用 ImageFingerprint 对象(其 __hash__ 基于 MD5)。
        # 但这里 value 存的是 (fingerprint, result) 元组,
        # 因为感知查找需要知道每个条目的 dhash 才能做比较。
        self._store: LRUCache[ImageFingerprint, "tuple[ImageFingerprint, OCRResult]"] = (
            LRUCache(capacity=capacity)
        )

        # 统计
        self.exact_hits = 0
        self.perceptual_hits = 0
        self.misses = 0

    # ------------------------------------------------------------------

    def get(self, image: Any) -> Optional[OCRResult]:
        """查询缓存。命中返回结果(已打上 from_cache 标记),未命中返回 None。"""
        if not self.enabled:
            self.misses += 1
            return None

        fp = self._fingerprint_of(image)

        # 第一级:精确查找
        entry = self._store.get(fp)
        if entry is not None:
            stored_fp, result = entry
            self.exact_hits += 1
            return self._decorate(result, "exact")

        # 第二级:感知近似查找
        if self.use_perceptual:
            near = self._find_perceptual(fp)
            if near is not None:
                stored_fp, result = near
                self.perceptual_hits += 1
                logger.debug(
                    "感知命中:汉明距离 %d <= %d",
                    self._hamming(fp, stored_fp),
                    self._hamming_threshold,
                )
                return self._decorate(result, "perceptual")

        self.misses += 1
        return None

    def put(self, image: Any, result: OCRResult) -> None:
        """写入缓存。"""
        if not self.enabled:
            return
        fp = self._fingerprint_of(image)
        self._store.put(fp, (fp, result))

    def _find_perceptual(
        self, fp: ImageFingerprint
    ) -> Optional["tuple[ImageFingerprint, OCRResult]"]:
        """线性扫描寻找感知近似条目。

        注意 LRUCache.__iter__ 返回的是 keys,不是 key-value 对,
        因此这里用 peek(key) 取 value。
        """
        for stored_fp in self._store:
            entry = self._store.peek(stored_fp)
            if entry is None:
                continue
            if fp.similar_to(stored_fp):
                return entry
        return None

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------

    @staticmethod
    def _hamming(fp_a: ImageFingerprint, fp_b: ImageFingerprint) -> int:
        return (fp_a.dhash ^ fp_b.dhash).bit_count()

    def _decorate(self, result: OCRResult, kind: str) -> OCRResult:
        """命中后给结果打标记。返回副本,避免污染缓存中的原始对象。

        这里做浅拷贝即可:缓存中存的是不可变风格的数据,
        boxes 列表不会在命中后被上层修改(如果修改了就是调用方的 bug)。
        深拷贝反而会引入不必要的性能开销。
        """
        clone = OCRResult(
            text=result.text,
            confidence=result.confidence,
            boxes=result.boxes,  # 共享引用,命中场景只读不写
            engine=result.engine,
            elapsed_ms=0.0,  # 缓存命中感知延迟应记为 0
        )
        clone.from_cache = True
        clone.cache_kind = kind
        clone.escalated = result.escalated
        clone.degraded = result.degraded
        clone.extra = dict(result.extra)
        clone.extra["cache_lookup"] = kind
        return clone

    def _fingerprint_of(self, image: Any) -> ImageFingerprint:
        return ImageFingerprint.from_image(
            image,
            threshold=self._hamming_threshold,
            hash_size=self._dhash_size,
        )

    # ------------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        cs = self._store.stats()
        total_hits = self.exact_hits + self.perceptual_hits
        total_lookups = total_hits + self.misses
        hit_rate = total_hits / total_lookups if total_lookups else 0.0

        # 成本节省估算:感知 + 精确命中都省了一次 VLM 调用(最贵的那层)
        saved_cost = total_hits * self.vlm_cost_per_call

        return {
            "enabled": self.enabled,
            "exact_hits": self.exact_hits,
            "perceptual_hits": self.perceptual_hits,
            "misses": self.misses,
            "total_hits": total_hits,
            "total_lookups": total_lookups,
            "hit_rate": round(hit_rate, 4),
            "cache_size": cs.size,
            "cache_capacity": cs.capacity,
            "evictions": cs.evictions,
            "saved_vlm_calls_est": total_hits,
            "saved_cost_est": round(saved_cost, 4),
            "saved_cost_currency": "CNY",
        }

    def clear(self) -> None:
        self._store.clear()
        self.exact_hits = self.perceptual_hits = self.misses = 0

    def __len__(self) -> int:
        return len(self._store)
