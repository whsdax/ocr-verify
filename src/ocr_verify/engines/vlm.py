"""第二层引擎:多模态大模型复核。

职责边界
--------
本层**只在第一层不可信时才被调用**,负责啃硬骨头:
弹窗半透明遮挡、运动模糊、低对比度、艺术字体、异形排版。
这些场景传统 OCR 的检测器本身就会失效,而多模态模型依靠语义先验
("这个位置应该是个按钮,文字大概率是'确定'")能够恢复出正确结果。

三个关键工程问题
----------------
1. **输出不稳定**:大模型天然爱加解释性文字("这张图片中的文字是:...")。
   对策:强制 JSON 输出 + 提示词明确约束 + 解析层容错剥离 markdown 围栏。

2. **成本与延迟**:每次调用都要钱,且延迟是本地推理的百倍量级。
   对策:(a) 上游缓存拦截重复图;(b) 上传前压缩图片降低 token;
        (c) 只在必要时才升级到本层。

3. **可用性**:外部 API 会超时、限流、偶发 5xx。
   对策:指数退避重试 + 最终失败时向 router 报错,由 router 决定降级。
   **绝不能因为外部 API 抖动就让整条自动化用例失败。**

协议抽象
--------
同时支持 Gemini 原生协议、OpenAI 兼容协议、Anthropic 兼容协议。
把差异收敛在 _build_request / _extract_content 两个方法里,
换厂商只需改配置文件的一行 protocol。

说明:base_url 在 openai / gemini 协议下会被追加固定路径
(/chat/completions、/v1beta/models/...:generateContent);
而在 anthropic 协议下 base_url 被**原样使用**——因为有些中转代理
(例如把 Claude 接口挂在某路径下的网关)路径已经包含在 base_url 里,
强行拼 /v1/messages 反而 404。
"""

from __future__ import annotations

import base64
import json
import logging
import re
import time
from typing import Any, Optional

import cv2
import numpy as np
import requests

from ..cache.fingerprint import ImageInput, load_image
from ..types import EngineType, OCREngineError, OCRResult, TextBox
from .base import OCREngine

logger = logging.getLogger(__name__)


# 提示词设计说明:
# 1. 明确角色与场景(UI 截图),让模型建立正确的先验
# 2. 强制 JSON 输出,并给出 schema
# 3. 显式禁止解释性文字 —— 这是实测中最容易出问题的地方
# 4. 要求模型自评置信度,用于与第一层结果做对比
# 5. 保留原始排版顺序,因为 UI 断言经常依赖文本的相对位置
DEFAULT_PROMPT = """你是一个精确的 OCR 文字识别引擎,正在处理软件界面截图。

任务:识别图片中所有可见的文字内容。

要求:
1. 严格按照原图的视觉排版顺序输出,从上到下、从左到右
2. 不同区域的文字用换行符分隔
3. 如果文字被弹窗、遮罩或其他元素部分遮挡,尽最大可能推断完整内容
4. 如果图片模糊,根据字形轮廓和上下文语义做最合理的推断
5. 保留原文的大小写、标点和数字格式,不要做任何修正或翻译
6. 忽略纯装饰性图标,但保留图标旁边的文字标签

只返回如下 JSON,不要输出任何解释、前言或 markdown 代码块标记:
{"text": "识别到的完整文字", "confidence": 0.95, "occluded": false}

字段说明:
- text: 识别出的全部文字,多行用 \\n 分隔
- confidence: 你对本次识别准确性的自评,0 到 1 之间的小数
- occluded: 图片中是否存在明显的遮挡或严重模糊,布尔值"""


class VLMEngine(OCREngine):
    """多模态模型复核引擎。"""

    engine_type = EngineType.VLM
    display_name = "VLM"

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "https://api.claudecode.net.cn/api/gemini",
        model: str = "gemini-3.5-flash",
        protocol: str = "gemini",
        timeout_s: float = 30.0,
        max_retries: int = 2,
        retry_backoff_s: float = 1.0,
        temperature: float = 0.0,
        max_output_tokens: int = 2048,
        max_image_width: int = 1600,
        jpeg_quality: int = 85,
        prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.protocol = protocol.lower()
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.retry_backoff_s = retry_backoff_s
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.max_image_width = max_image_width
        self.jpeg_quality = jpeg_quality
        self.prompt = prompt or DEFAULT_PROMPT

        # 复用 Session:保持 TCP 连接,省掉每次请求的 TLS 握手(实测省 100~300ms)
        self._session = requests.Session()

        # 调用统计,用于成本核算
        self.call_count = 0
        self.error_count = 0
        self.retry_count = 0
        self.total_latency_ms = 0.0

    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        return bool(self.api_key)

    def close(self) -> None:
        self._session.close()

    # ------------------------------------------------------------------
    # 图片预处理
    # ------------------------------------------------------------------

    def _encode_image(self, image: ImageInput) -> tuple[str, str]:
        """压缩并编码为 base64。返回 (base64_str, mime_type)。

        为什么要压缩:多模态模型按图片尺寸切分 token 计费,
        一张 4K 截图的成本可能是 1080p 的数倍,但对文字识别精度的提升微乎其微
        (文字区域的有效分辨率才是关键,整图分辨率不是)。
        实测 1600px 宽度对绝大多数 UI 截图已经足够。

        为什么用 JPEG 而非 PNG:同等视觉质量下 JPEG 体积约为 PNG 的 1/3。
        quality=85 时压缩伪影对 OCR 的影响可忽略。
        """
        img = load_image(image)
        h, w = img.shape[:2]

        if w > self.max_image_width:
            scale = self.max_image_width / w
            new_size = (self.max_image_width, int(h * scale))
            # INTER_AREA 是缩小图像的最佳选择,能有效抗锯齿
            img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)

        ok, buf = cv2.imencode(
            ".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        )
        if not ok:
            raise OCREngineError("图片编码失败", engine="vlm", retryable=False)

        return base64.b64encode(buf.tobytes()).decode("ascii"), "image/jpeg"

    # ------------------------------------------------------------------
    # 协议适配
    # ------------------------------------------------------------------

    def _build_request(
        self, b64: str, mime: str, prompt: str
    ) -> tuple[str, dict[str, str], dict[str, Any]]:
        """构造请求。返回 (url, headers, json_body)。"""
        if self.protocol == "gemini":
            url = f"{self.base_url}/v1beta/models/{self.model}:generateContent"
            headers = {
                "Content-Type": "application/json",
                # 用 header 传 key 而非 query string,避免密钥出现在日志和 URL 中
                "x-goog-api-key": self.api_key,
            }
            body = {
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            {"text": prompt},
                            {"inline_data": {"mime_type": mime, "data": b64}},
                        ],
                    }
                ],
                "generationConfig": {
                    "temperature": self.temperature,
                    "maxOutputTokens": self.max_output_tokens,
                    "responseMimeType": "application/json",
                },
            }
            return url, headers, body

        if self.protocol == "anthropic":
            # base_url 原样使用(中转代理常已含路径),不追加 /v1/messages。
            # 常见 Claude 兼容网关期望 x-api-key + anthropic-version 头。
            url = self.base_url
            headers = {
                "Content-Type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
            }
            body = {
                "model": self.model,
                "max_tokens": self.max_output_tokens,
                "temperature": self.temperature,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": mime,
                                    "data": b64,
                                },
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
            }
            return url, headers, body

        # OpenAI 兼容协议
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        body = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{b64}"},
                        },
                    ],
                }
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_output_tokens,
            "response_format": {"type": "json_object"},
        }
        return url, headers, body

    def _extract_content(self, payload: dict[str, Any]) -> str:
        """从响应体中提取模型输出的纯文本。"""
        if self.protocol == "gemini":
            candidates = payload.get("candidates") or []
            if not candidates:
                # 触发安全过滤时没有 candidates,但有 promptFeedback
                feedback = payload.get("promptFeedback", {})
                raise OCREngineError(
                    f"响应中无 candidates,promptFeedback={feedback}",
                    engine="vlm",
                    retryable=False,
                )
            parts = candidates[0].get("content", {}).get("parts") or []
            return "".join(p.get("text", "") for p in parts)

        if self.protocol == "anthropic":
            content = payload.get("content") or []
            pieces = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    pieces.append(block.get("text", ""))
            text = "".join(pieces)
            if not text:
                raise OCREngineError(
                    "响应中无文本块", engine="vlm", retryable=False
                )
            return text

        choices = payload.get("choices") or []
        if not choices:
            raise OCREngineError("响应中无 choices", engine="vlm", retryable=False)
        return choices[0].get("message", {}).get("content", "") or ""

    # ------------------------------------------------------------------
    # 输出解析
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_model_output(content: str) -> tuple[str, float, dict[str, Any]]:
        """解析模型输出,返回 (text, confidence, meta)。

        三级容错策略,依次尝试:
          1. 直接 JSON 解析(理想情况)
          2. 剥离 markdown 代码块围栏后再解析(模型最常见的"不听话"形式)
          3. 正则提取第一个 JSON 对象
          4. 全部失败则把原始输出当作纯文本使用,并降低置信度

        第 4 步很重要:即使模型没按格式返回,它的输出通常仍包含正确的文字。
        直接丢弃太浪费,标记为低置信度交给上层判断更合理。
        """
        content = (content or "").strip()
        if not content:
            return "", 0.0, {"parse": "empty"}

        # 尝试 1:直接解析
        parsed = VLMEngine._try_json(content)

        # 尝试 2:剥离 ```json ... ``` 围栏
        if parsed is None:
            fenced = re.sub(
                r"^```(?:json)?\s*|\s*```$", "", content, flags=re.MULTILINE
            ).strip()
            parsed = VLMEngine._try_json(fenced)

        # 尝试 3:正则捞出第一个 JSON 对象(非贪婪匹配到最后一个右括号)
        if parsed is None:
            match = re.search(r"\{.*\}", content, flags=re.DOTALL)
            if match:
                parsed = VLMEngine._try_json(match.group(0))

        if isinstance(parsed, dict):
            text = str(parsed.get("text", "")).strip()
            try:
                conf = float(parsed.get("confidence", 0.85))
            except (TypeError, ValueError):
                conf = 0.85
            conf = max(0.0, min(1.0, conf))  # 夹到合法区间,防止模型返回 95 这种
            meta = {
                "parse": "json",
                "occluded": bool(parsed.get("occluded", False)),
            }
            return text, conf, meta

        # 尝试 4:降级为纯文本。置信度给 0.6 —— 能识别出内容但格式失控,
        # 说明模型状态不太稳定,不应给高分。
        return content, 0.6, {"parse": "raw_text"}

    @staticmethod
    def _try_json(s: str) -> Optional[dict[str, Any]]:
        try:
            value = json.loads(s)
            return value if isinstance(value, dict) else None
        except (json.JSONDecodeError, TypeError):
            return None

    # ------------------------------------------------------------------
    # 主流程
    # ------------------------------------------------------------------

    def _recognize_impl(self, image: ImageInput, **kwargs: Any) -> OCRResult:
        if not self.api_key:
            raise OCREngineError(
                "未配置 API Key。请在 config.yaml 中填写 vlm.api_key,"
                "或设置环境变量 OCR_VERIFY_API_KEY",
                engine="vlm",
                retryable=False,
            )

        prompt = kwargs.get("prompt") or self.prompt

        # 调用方可传入预期格式,拼进提示词能显著提升结构化字段的识别准确率
        expected_pattern = kwargs.get("expected_pattern")
        if expected_pattern:
            prompt += f"\n\n补充提示:图中的目标文本预期符合正则 {expected_pattern},请特别留意该部分。"

        b64, mime = self._encode_image(image)
        url, headers, body = self._build_request(b64, mime, prompt)

        content = self._request_with_retry(url, headers, body)
        text, confidence, meta = self._parse_model_output(content)

        return OCRResult(
            text=text,
            confidence=confidence,
            boxes=[TextBox(text=text, confidence=confidence)] if text else [],
            engine=self.engine_type,
            extra={
                "model": self.model,
                "protocol": self.protocol,
                "image_bytes": len(b64) * 3 // 4,  # base64 膨胀率约 4/3,反推原始大小
                **meta,
            },
        )

    def _request_with_retry(
        self, url: str, headers: dict[str, str], body: dict[str, Any]
    ) -> str:
        """带指数退避的请求。

        重试策略的取舍:
          - 429(限流)和 5xx(服务端错误)重试 —— 这些是暂时性故障
          - 4xx(除 429)不重试 —— 请求本身有问题,重试多少次都一样
          - 网络超时重试 —— 可能是瞬时抖动

        退避时间 = base * 2^attempt,避免在服务端已经过载时雪上加霜。
        """
        last_error: Optional[str] = None

        for attempt in range(self.max_retries + 1):
            if attempt > 0:
                wait = self.retry_backoff_s * (2 ** (attempt - 1))
                logger.info("第 %d 次重试,等待 %.1fs", attempt, wait)
                time.sleep(wait)
                self.retry_count += 1

            start = time.perf_counter()
            try:
                resp = self._session.post(
                    url, headers=headers, json=body, timeout=self.timeout_s
                )
                self.total_latency_ms += (time.perf_counter() - start) * 1000
                self.call_count += 1

                if resp.status_code == 200:
                    return self._extract_content(resp.json())

                snippet = resp.text[:300]
                last_error = f"HTTP {resp.status_code}: {snippet}"

                if resp.status_code == 429 or resp.status_code >= 500:
                    logger.warning("可重试的错误 %s", last_error)
                    continue

                # 不可重试的客户端错误,立刻失败
                self.error_count += 1
                raise OCREngineError(last_error, engine="vlm", retryable=False)

            except requests.Timeout:
                self.total_latency_ms += (time.perf_counter() - start) * 1000
                last_error = f"请求超时(>{self.timeout_s}s)"
                logger.warning(last_error)
            except requests.RequestException as exc:
                last_error = f"网络异常: {type(exc).__name__}: {exc}"
                logger.warning(last_error)
            except json.JSONDecodeError as exc:
                last_error = f"响应不是合法 JSON: {exc}"
                logger.warning(last_error)

        self.error_count += 1
        raise OCREngineError(
            f"已重试 {self.max_retries} 次仍失败。最后错误: {last_error}",
            engine="vlm",
            retryable=True,
        )

    # ------------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        """调用统计,用于成本核算与报告。"""
        avg = self.total_latency_ms / self.call_count if self.call_count else 0.0
        return {
            "call_count": self.call_count,
            "error_count": self.error_count,
            "retry_count": self.retry_count,
            "avg_latency_ms": round(avg, 2),
            "total_latency_ms": round(self.total_latency_ms, 2),
        }

    def reset_stats(self) -> None:
        self.call_count = self.error_count = self.retry_count = 0
        self.total_latency_ms = 0.0
