"""第一层引擎:PaddleOCR(PP-OCRv4/v5),带 RapidOCR 兜底。

为什么第一层用 PaddleOCR
------------------------
- 中文场景准确率显著优于 Tesseract(Tesseract 的中文模型基于传统 LSTM,
  对 UI 界面这种短文本、多字号、有背景色的场景适应性差)
- 本地推理,零 API 成本,毫秒级延迟
- 输出带每个文本框的置信度 —— 这是双层路由能够工作的前提,
  没有可靠的置信度就无法判断"什么时候该升级"

版本兼容问题
------------
PaddleOCR 在 3.0 做了破坏性 API 变更:
  2.x:  ocr.ocr(img, cls=True) -> [[[box, (text, score)], ...]]
  3.x:  ocr.predict(img)       -> [{"rec_texts": [...], "rec_scores": [...], "dt_polys": [...]}]
构造函数参数也不同(use_angle_cls 在 3.x 被拆成多个开关)。
本模块用运行时探测的方式同时兼容两者,而不是把版本号写死 ——
写死版本号意味着用户升级依赖就会崩,这在工程上不可接受。

RapidOCR 兜底
-------------
RapidOCR 是 PP-OCR 模型的 ONNXRuntime 移植版,识别效果基本一致,
但依赖极轻(不需要 paddlepaddle),在部分环境下装不上 Paddle 时是可靠替代。
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

from ..cache.fingerprint import ImageInput, load_image
from ..types import EngineType, OCREngineError, OCRResult, TextBox
from .base import OCREngine

logger = logging.getLogger(__name__)


class PaddleEngine(OCREngine):
    """第一层快速通道。

    Parameters
    ----------
    backend:
        "auto" | "paddleocr" | "rapidocr"。auto 会优先尝试 paddleocr。
    lang:
        "ch" 模型同时覆盖中英文混排,UI 场景通常够用。
    """

    engine_type = EngineType.PADDLE
    display_name = "PaddleOCR"

    def __init__(
        self,
        backend: str = "auto",
        lang: str = "ch",
        use_gpu: bool = False,
        det_db_thresh: float = 0.3,
        drop_score: float = 0.3,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.backend = backend
        self.lang = lang
        self.use_gpu = use_gpu
        self.det_db_thresh = det_db_thresh
        self.drop_score = drop_score

        self._engine: Any = None
        self._actual_backend: Optional[str] = None
        self._api_style: Optional[str] = None   # "predict"(3.x) | "ocr"(2.x) | "rapid"

    # ------------------------------------------------------------------
    # 延迟初始化
    # ------------------------------------------------------------------

    def _ensure_engine(self) -> None:
        """首次使用时才加载模型。

        延迟加载的意义:import 本模块不应该触发几百 MB 的模型下载。
        很多场景(只跑单测、只用 Tesseract 基线)根本用不到 Paddle。
        """
        if self._engine is not None:
            return
        if self._init_error is not None:
            raise OCREngineError(self._init_error, engine="paddle", retryable=False)

        order = (
            ["paddleocr", "rapidocr"]
            if self.backend == "auto"
            else [self.backend]
        )

        errors: list[str] = []
        for name in order:
            try:
                if name == "paddleocr":
                    self._init_paddleocr()
                else:
                    self._init_rapidocr()
                # 关键:某些后端(典型如 PaddleOCR 3.x)会把依赖缺失延迟到
                # 首次 predict 时才抛出 "dependency error during pipeline creation"。
                # 仅构造对象成功不足以证明该后端可用,必须真正跑一次推理。
                # 否则 auto 模式会被假"成功"的 paddle 卡住,永远回退不到 rapidocr。
                self._raw_infer(np.full((64, 192, 3), 255, dtype=np.uint8))
                self._actual_backend = name
                logger.info("第一层引擎已就绪: %s (API 风格: %s)", name, self._api_style)
                return
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{name}: {type(exc).__name__}: {exc}")
                logger.warning("后端 %s 初始化失败: %s", name, exc)
                # 清理残留的失败引擎,避免影响下一个后端的初始化
                self._engine = None
                self._api_style = None

        self._init_error = "所有第一层后端均不可用 -> " + " | ".join(errors)
        raise OCREngineError(self._init_error, engine="paddle", retryable=False)

    def _init_paddleocr(self) -> None:
        from paddleocr import PaddleOCR  # 局部导入,避免无谓的启动开销

        # 3.x 与 2.x 构造参数不兼容,逐个尝试而不是判断版本号。
        # 判断版本号看似更"干净",但 PaddleOCR 的 __version__ 在部分
        # 发行版中缺失或格式不一致,实测不如直接试构造来得可靠。
        attempts: list[dict[str, Any]] = [
            # PaddleOCR 3.x:关闭文档方向分类和图像矫正,UI 截图用不上且拖慢速度
            {
                "lang": self.lang,
                "use_doc_orientation_classify": False,
                "use_doc_unwarping": False,
                "use_textline_orientation": False,
            },
            # PaddleOCR 2.6~2.9
            {"lang": self.lang, "use_angle_cls": False, "show_log": False},
            # 最保守的兜底
            {"lang": self.lang},
        ]

        last_exc: Optional[Exception] = None
        for params in attempts:
            try:
                self._engine = PaddleOCR(**params)
                break
            except (TypeError, ValueError) as exc:
                last_exc = exc
                continue
        if self._engine is None:
            raise OCREngineError(
                f"PaddleOCR 构造失败,已尝试全部参数组合。最后错误: {last_exc}",
                engine="paddle",
                retryable=False,
            )

        # 探测可用的推理方法
        self._api_style = "predict" if hasattr(self._engine, "predict") else "ocr"

    def _init_rapidocr(self) -> None:
        try:
            from rapidocr_onnxruntime import RapidOCR  # type: ignore
        except ImportError:
            from rapidocr import RapidOCR  # type: ignore

        self._engine = RapidOCR()
        self._api_style = "rapid"

    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        if self._engine is not None:
            return True
        if self._init_error is not None:
            return False
        try:
            self._ensure_engine()
            return True
        except Exception:  # noqa: BLE001
            return False

    def warmup(self) -> None:
        """用一张小白图触发模型加载,把首次推理的加载开销从统计中剥离。"""
        try:
            self._ensure_engine()
            dummy = np.full((64, 192, 3), 255, dtype=np.uint8)
            self._raw_infer(dummy)
        except Exception as exc:  # noqa: BLE001
            logger.debug("预热失败(不影响主流程): %s", exc)

    # ------------------------------------------------------------------

    def _raw_infer(self, img: np.ndarray) -> Any:
        if self._api_style == "predict":
            return self._engine.predict(img)
        if self._api_style == "rapid":
            return self._engine(img)
        # 2.x 的 ocr() 在部分版本要求显式传 cls
        try:
            return self._engine.ocr(img, cls=False)
        except TypeError:
            return self._engine.ocr(img)

    def _recognize_impl(self, image: ImageInput, **kwargs: Any) -> OCRResult:
        self._ensure_engine()
        img = load_image(image)
        raw = self._raw_infer(img)

        if self._api_style == "predict":
            boxes = self._parse_v3(raw)
        elif self._api_style == "rapid":
            boxes = self._parse_rapid(raw)
        else:
            boxes = self._parse_v2(raw)

        # 过滤低分框。注意:这里丢弃的是"检测器都不确定有没有字"的框,
        # 与后续路由的置信度阈值是两回事,不要混淆。
        boxes = [b for b in boxes if b.confidence >= self.drop_score]

        text, confidence = self._aggregate(boxes)
        return OCRResult(
            text=text,
            confidence=confidence,
            boxes=boxes,
            engine=self.engine_type,
            extra={"backend": self._actual_backend, "raw_box_count": len(boxes)},
        )

    # ------------------------------------------------------------------
    # 各版本输出格式的解析
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_v3(raw: Any) -> list[TextBox]:
        """PaddleOCR 3.x: [{"rec_texts": [...], "rec_scores": [...], "dt_polys": [...]}]"""
        boxes: list[TextBox] = []
        if not raw:
            return boxes

        for page in raw:
            # 3.x 返回的是 OCRResult 对象,支持 dict 式访问;部分版本直接是 dict
            data = page if isinstance(page, dict) else getattr(page, "json", None) or {}
            if isinstance(data, dict) and "res" in data:
                data = data["res"]

            texts = data.get("rec_texts") or []
            scores = data.get("rec_scores") or []
            polys = data.get("dt_polys") or data.get("rec_polys") or []

            for i, txt in enumerate(texts):
                score = float(scores[i]) if i < len(scores) else 0.0
                poly = polys[i] if i < len(polys) else None
                boxes.append(
                    TextBox(
                        text=str(txt),
                        confidence=score,
                        box=PaddleEngine._normalize_poly(poly),
                    )
                )
        return boxes

    @staticmethod
    def _parse_v2(raw: Any) -> list[TextBox]:
        """PaddleOCR 2.x: [[[box, (text, score)], ...]],单图时最外层只有一个元素。"""
        boxes: list[TextBox] = []
        if not raw:
            return boxes

        page = raw[0] if isinstance(raw[0], list) else raw
        if page is None:
            return boxes

        for item in page:
            if not item or len(item) < 2:
                continue
            poly, rec = item[0], item[1]
            if isinstance(rec, (list, tuple)) and len(rec) >= 2:
                text, score = str(rec[0]), float(rec[1])
            else:
                text, score = str(rec), 0.0
            boxes.append(
                TextBox(
                    text=text,
                    confidence=score,
                    box=PaddleEngine._normalize_poly(poly),
                )
            )
        return boxes

    @staticmethod
    def _parse_rapid(raw: Any) -> list[TextBox]:
        """RapidOCR: (result, elapse),result = [[box, text, score], ...]"""
        boxes: list[TextBox] = []
        if not raw:
            return boxes

        result = raw[0] if isinstance(raw, tuple) else raw
        # 新版 RapidOCR 返回带 .boxes/.txts/.scores 属性的对象
        if hasattr(result, "txts") and result.txts is not None:
            txts = result.txts
            scores = result.scores or []
            polys = result.boxes if result.boxes is not None else []
            for i, txt in enumerate(txts):
                boxes.append(
                    TextBox(
                        text=str(txt),
                        confidence=float(scores[i]) if i < len(scores) else 0.0,
                        box=PaddleEngine._normalize_poly(
                            polys[i] if i < len(polys) else None
                        ),
                    )
                )
            return boxes

        if not isinstance(result, list):
            return boxes

        for item in result:
            if not item or len(item) < 3:
                continue
            poly, text, score = item[0], item[1], item[2]
            boxes.append(
                TextBox(
                    text=str(text),
                    confidence=float(score),
                    box=PaddleEngine._normalize_poly(poly),
                )
            )
        return boxes

    @staticmethod
    def _normalize_poly(poly: Any) -> Optional[list[tuple[float, float]]]:
        """把各种形态的坐标(ndarray / 嵌套 list)统一成 [(x, y), ...]。"""
        if poly is None:
            return None
        try:
            arr = np.asarray(poly, dtype=float).reshape(-1, 2)
            return [(float(x), float(y)) for x, y in arr]
        except Exception:  # noqa: BLE001 - 坐标解析失败不应影响文本结果
            return None
