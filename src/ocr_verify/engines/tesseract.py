"""基线引擎:Tesseract。

这个引擎**不参与生产链路**,只作为评测的对照组存在。

为什么必须保留基线
------------------
"识别率从 80% 提升到 95%" 这句话如果没有基线,就是空话。
评测的本质是对比,没有对照组的绝对数字没有说服力 ——
面试官会问"95% 是高还是低",只有拿出"同一份数据上旧方案是 XX%"才能回答。

同时它也解释了本项目的立项动机:Tesseract 在中文 UI 截图上的短板
(检测器对彩色背景、小字号、抗锯齿文字适应差)正是问题的起点。

置信度换算
----------
Tesseract 输出 0-100 的整数置信度,且对识别失败的块返回 -1。
本模块统一归一化到 [0,1] 区间,与其他引擎口径一致 ——
指标口径不统一是评测中最容易出现的隐蔽错误。
"""

from __future__ import annotations

import logging
import shutil
from typing import Any, Optional

from ..cache.fingerprint import ImageInput, load_image
from ..types import EngineType, OCREngineError, OCRResult, TextBox
from .base import OCREngine

logger = logging.getLogger(__name__)


class TesseractEngine(OCREngine):
    """Tesseract OCR 基线。

    需要系统级安装 tesseract 可执行文件,以及中文语言包 chi_sim。
    Windows: https://github.com/UB-Mannheim/tesseract/wiki
    安装后若不在 PATH 中,可通过 cmd_path 参数指定。
    """

    engine_type = EngineType.TESSERACT
    display_name = "Tesseract"

    def __init__(
        self,
        lang: str = "chi_sim+eng",
        cmd_path: Optional[str] = None,
        psm: int = 6,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.lang = lang
        self.cmd_path = cmd_path
        # PSM 6 = 假设图像是一个统一的文本块。
        # UI 截图通常是分散的多个文本区域,理论上 PSM 11(稀疏文本)更合适,
        # 但实测 PSM 6 在多数界面上综合表现更稳,故设为默认。
        self.psm = psm
        self._pytesseract: Any = None

    def _ensure(self) -> None:
        if self._pytesseract is not None:
            return
        try:
            import pytesseract
        except ImportError as exc:
            raise OCREngineError(
                "未安装 pytesseract。执行: pip install pytesseract",
                engine="tesseract",
                retryable=False,
            ) from exc

        if self.cmd_path:
            pytesseract.pytesseract.tesseract_cmd = self.cmd_path
        elif shutil.which("tesseract") is None:
            raise OCREngineError(
                "未找到 tesseract 可执行文件。请先安装 Tesseract-OCR 并加入 PATH,"
                "或通过 cmd_path 参数指定路径。"
                "Windows 下载: https://github.com/UB-Mannheim/tesseract/wiki",
                engine="tesseract",
                retryable=False,
            )

        self._pytesseract = pytesseract

    def is_available(self) -> bool:
        try:
            self._ensure()
            return True
        except Exception:  # noqa: BLE001
            return False

    def _recognize_impl(self, image: ImageInput, **kwargs: Any) -> OCRResult:
        self._ensure()
        pt = self._pytesseract

        img = load_image(image)
        # Tesseract 期望 RGB,OpenCV 载入的是 BGR
        import cv2

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        config = f"--psm {self.psm}"
        data = pt.image_to_data(
            rgb, lang=self.lang, config=config, output_type=pt.Output.DICT
        )

        boxes: list[TextBox] = []
        n = len(data.get("text", []))
        for i in range(n):
            text = (data["text"][i] or "").strip()
            if not text:
                continue

            raw_conf = float(data["conf"][i])
            if raw_conf < 0:
                continue  # -1 表示该块无有效识别结果

            boxes.append(
                TextBox(
                    text=text,
                    confidence=raw_conf / 100.0,  # 统一归一化到 [0,1]
                    box=[
                        (float(data["left"][i]), float(data["top"][i])),
                        (
                            float(data["left"][i] + data["width"][i]),
                            float(data["top"][i]),
                        ),
                        (
                            float(data["left"][i] + data["width"][i]),
                            float(data["top"][i] + data["height"][i]),
                        ),
                        (
                            float(data["left"][i]),
                            float(data["top"][i] + data["height"][i]),
                        ),
                    ],
                )
            )

        # Tesseract 按词切分,中文场景下用空字符串拼接更接近原文
        text = "".join(b.text for b in boxes)
        confidence = min((b.confidence for b in boxes), default=0.0)

        return OCRResult(
            text=text,
            confidence=confidence,
            boxes=boxes,
            engine=self.engine_type,
            extra={"psm": self.psm, "lang": self.lang},
        )
