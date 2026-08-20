"""对外门面 OCRVerifier —— 调用方唯一需要关心的入口。

设计目标:调用方只需 `verify = OCRVerifier(config)`,
然后 `verify.recognize(image)` 拿结果,完全不需要知道
背后是几层引擎、是否走了缓存、哪层失败降级了。

这是"门面模式(Facade)":把一组复杂子系统的接口收敛成一个简洁高层接口。
对测开同学尤其重要 —— 他们要的是"给我识别结果",
而不是"加载模型、查缓存、路由、降级"这一堆编排细节。
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Pattern

from .cache import OCRResultCache
from .cache.fingerprint import ImageInput
from .config import AppConfig, get_config
from .engines.base import OCREngine
from .engines.paddle import PaddleEngine
from .engines.tesseract import TesseractEngine
from .engines.vlm import VLMEngine
from .router import TwoLayerRouter
from .types import EngineType, OCRResult

logger = logging.getLogger(__name__)


class OCRVerifier:
    """智能 OCR 验证器:双层引擎 + 指纹缓存的统一入口。"""

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        self.config = config or get_config()

        problems = self.config.validate()
        if problems:
            # 警告而非崩溃:让"没有 API Key 只想用第一层"的场景也能跑
            for p in problems:
                logger.warning("配置问题: %s", p)

        # ---- 装配各引擎 ----
        self.first_layer = PaddleEngine(
            backend=self.config.paddle.backend,
            lang=self.config.paddle.lang,
            use_gpu=self.config.paddle.use_gpu,
            det_db_thresh=self.config.paddle.det_db_thresh,
            drop_score=self.config.paddle.drop_score,
        )

        self.second_layer: Optional[VLMEngine] = None
        if self.config.vlm.is_ready:
            self.second_layer = VLMEngine(
                api_key=self.config.vlm.api_key,
                base_url=self.config.vlm.base_url,
                model=self.config.vlm.model,
                protocol=self.config.vlm.protocol,
                timeout_s=self.config.vlm.timeout_s,
                max_retries=self.config.vlm.max_retries,
                retry_backoff_s=self.config.vlm.retry_backoff_s,
                temperature=self.config.vlm.temperature,
                max_output_tokens=self.config.vlm.max_output_tokens,
                max_image_width=self.config.vlm.max_image_width,
                jpeg_quality=self.config.vlm.jpeg_quality,
            )
        else:
            logger.warning(
                "第二层未启用(缺少 API Key)。系统将以单 PaddleOCR 模式运行,"
                "无法处理复杂遮挡场景。"
            )

        # ---- 路由器 ----
        self.router = TwoLayerRouter(
            first_layer=self.first_layer,
            second_layer=self.second_layer,
            confidence_threshold=self.config.router.confidence_threshold,
            escalate_on_empty=self.config.router.escalate_on_empty,
            escalate_on_pattern_mismatch=self.config.router.escalate_on_pattern_mismatch,
            escalate_on_box_overlap=self.config.router.escalate_on_box_overlap,
            box_overlap_threshold=self.config.router.box_overlap_threshold,
            fallback_to_first_layer=self.config.router.fallback_to_first_layer,
        )

        # ---- 缓存 ----
        self.cache = OCRResultCache(
            capacity=self.config.cache.capacity,
            use_perceptual=self.config.cache.use_perceptual,
            hamming_threshold=self.config.cache.hamming_threshold,
            dhash_size=self.config.cache.dhash_size,
            vlm_cost_per_call=self.config.cache.vlm_cost_per_call,
            enabled=self.config.cache.enabled,
        )

    # ------------------------------------------------------------------

    def recognize(
        self,
        image: ImageInput,
        expected_pattern: Optional[str | Pattern[str]] = None,
        force_escalate: bool = False,
        use_cache: bool = True,
        **kwargs: Any,
    ) -> OCRResult:
        """识别图片文本。

        Parameters
        ----------
        image:
            图片路径 / 字节流 / numpy 数组。
        expected_pattern:
            可选的正则表达式。提供后可触发"格式不匹配"升级条件,
            显著提升结构化字段(金额、手机号、订单号)的识别准确率。
        force_escalate:
            强制走第二层。用于对关键断言做"二次确认"。
        use_cache:
            是否查询指纹缓存。调试时可设为 False 强制重新推理。

        Returns
        -------
        OCRResult:统一结果。调用方应检查 .confidence / .degraded /
            .from_cache 来决策自己的断言策略。
        """
        # 1) 缓存优先 —— 命中则直接返回,省掉两层引擎的开销
        if use_cache:
            cached = self.cache.get(image)
            if cached is not None:
                return cached

        # 2) 双层路由识别
        result = self.router.route(
            image,
            expected_pattern=expected_pattern,
            force_escalate=force_escalate,
            **kwargs,
        )

        # 3) 回写缓存(即使降级结果也缓存:同一张图再次出错时,重试无益,
        #    不如直接命中并让调用方快速拿到已知结果)
        if use_cache and not result.error:
            self.cache.put(image, result)

        return result

    def warmup(self) -> None:
        """预热各引擎。评测前调用,避免首图加载开销污染延迟统计。"""
        logger.info("预热第一层引擎...")
        self.first_layer.warmup()
        if self.second_layer is not None:
            logger.info("预热第二层连接...")
            try:
                self.second_layer.is_available()
            except Exception as exc:  # noqa: BLE001
                logger.debug("第二层预热检查跳过: %s", exc)

    def stats(self) -> dict[str, Any]:
        """汇总统计:路由 + 缓存 + 二层调用,用于报告。"""
        return {
            "router": self.router.stats(),
            "cache": self.cache.stats(),
            "vlm": self.second_layer.stats() if self.second_layer else None,
            "config": {
                "confidence_threshold": self.config.router.confidence_threshold,
                "cache_capacity": self.config.cache.capacity,
                "second_layer": (
                    self.second_layer.model if self.second_layer else None
                ),
            },
        }

    def close(self) -> None:
        if self.second_layer is not None:
            self.second_layer.close()

    # 上下文管理器支持:with OCRVerifier() as v: ...
    def __enter__(self) -> "OCRVerifier":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
