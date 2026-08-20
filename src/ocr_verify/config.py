"""配置加载 —— YAML 文件 + 环境变量覆盖。

设计原则
--------
1. **密钥永不进代码库**:config.yaml 已加入 .gitignore,
   仓库中只提交 config.example.yaml 模板。
2. **环境变量优先级最高**:CI 环境无法放置配置文件,
   通过 OCR_VERIFY_API_KEY 等环境变量注入,便于 GitHub Actions 使用。
3. **配置即文档**:每个字段都有默认值和类型标注,
   新人不看文档也能通过 dataclass 定义理解可调项。

优先级: 环境变量 > config.yaml > 代码内默认值
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Optional

import yaml

# 项目根目录:本文件位于 <root>/src/ocr_verify/config.py,故上溯三级
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"
EXAMPLE_CONFIG_PATH = PROJECT_ROOT / "config.example.yaml"


@dataclass
class VLMConfig:
    """第二层多模态模型配置。

    protocol 支持三种取值:
      - "gemini":   Google Gemini 原生协议
                    POST {base_url}/v1beta/models/{model}:generateContent,x-goog-api-key 鉴权
      - "openai":   OpenAI 兼容协议(通义千问、智谱、DeepSeek 等大多支持)
                    POST {base_url}/chat/completions,Bearer 鉴权
      - "anthropic":Claude 兼容网关(常见于中转代理)
                    base_url 原样使用,x-api-key + anthropic-version 鉴权

    做成可切换是刻意的:模型厂商的性价比变化很快,
    把协议差异隔离在一个配置项里,换厂商时业务代码零改动。
    """

    enabled: bool = True
    protocol: str = "gemini"
    base_url: str = "https://api.claudecode.net.cn/api/gemini"
    model: str = "gemini-3.5-flash"
    api_key: str = ""

    timeout_s: float = 30.0
    max_retries: int = 2
    retry_backoff_s: float = 1.0     # 指数退避基数:第 n 次重试等待 base * 2^n 秒
    temperature: float = 0.0         # 识别任务要确定性输出,温度固定为 0
    max_output_tokens: int = 2048

    # 图片预处理:超过此宽度先等比缩放。
    # 目的有二:降低 token 消耗(成本与图片尺寸正相关);规避部分接口的体积上限。
    max_image_width: int = 1600
    jpeg_quality: int = 85           # 上传前重编码质量,在体积与清晰度间取平衡

    @property
    def is_ready(self) -> bool:
        return self.enabled and bool(self.api_key)


@dataclass
class PaddleConfig:
    """第一层 OCR 配置。

    backend 可选:
      - "auto":     优先 paddleocr,不可用时回落 rapidocr(同为 PP-OCRv4 模型)
      - "paddleocr": 强制使用 PaddleOCR
      - "rapidocr":  强制使用 RapidOCR(ONNXRuntime 推理,依赖更轻,跨平台更稳)
    """

    backend: str = "auto"
    lang: str = "ch"                 # ch 模型同时支持中英文混排
    use_gpu: bool = False
    det_db_thresh: float = 0.3       # 文本检测二值化阈值,越低召回越多(也越容易误检)
    drop_score: float = 0.3          # 低于此分数的文本框直接丢弃


@dataclass
class RouterConfig:
    """双层路由策略。

    confidence_threshold 是本项目最关键的超参,
    默认 0.7 仅为起点 —— 必须用 benchmark/calibrate.py 在自己的
    数据集上跑校准曲线来确定,否则面试时无法回答"为什么是 0.7"。
    """

    confidence_threshold: float = 0.7

    # 除低置信度外的补充升级条件(见 EscalationReason)
    escalate_on_empty: bool = True          # 检测到框但识别为空
    escalate_on_pattern_mismatch: bool = True
    escalate_on_box_overlap: bool = True
    box_overlap_threshold: float = 0.35     # 重叠面积占比超过此值判定为疑似遮挡

    # 容错:二层失败时是否回落一层结果。
    # 默认 True —— 宁可给一个低质量结果并打上 degraded 标记,
    # 也不要让整条自动化用例因为外部 API 抖动而失败。
    fallback_to_first_layer: bool = True


@dataclass
class CacheConfig:
    enabled: bool = True
    capacity: int = 512
    use_perceptual: bool = False     # 是否启用 dHash 近似命中。
    # 默认关闭:它只在"同一页面反复截图、仅状态栏等局部像素变化"时有效,
    # 跨页面数据集容易误命中(合成 UI 结构相似时尤甚)。确有同图复现需求再开启。
    hamming_threshold: int = 3       # dHash 判定近似的汉明距离上限
    dhash_size: int = 8              # 生成 dhash_size^2 = 64 位哈希

    # 成本估算参数,用于报告中折算"省了多少钱"。
    # 需按实际计费标准填写,默认值仅为量级示意。
    vlm_cost_per_call: float = 0.003
    currency: str = "CNY"


@dataclass
class BenchmarkConfig:
    dataset_dir: str = "datasets"
    report_dir: str = "reports"
    ground_truth_file: str = "datasets/ground_truth.json"
    random_seed: int = 42            # 固定种子保证扰动样本可复现
    include_tesseract: bool = True   # 基线对照,若未安装 Tesseract 可关掉


@dataclass
class AppConfig:
    vlm: VLMConfig = field(default_factory=VLMConfig)
    paddle: PaddleConfig = field(default_factory=PaddleConfig)
    router: RouterConfig = field(default_factory=RouterConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    log_level: str = "INFO"

    # ------------------------------------------------------------------

    @classmethod
    def load(cls, path: Optional[Path | str] = None) -> "AppConfig":
        """从 YAML 加载配置,随后应用环境变量覆盖。

        配置文件不存在时不报错,直接使用默认值 + 环境变量 ——
        这让"只设一个环境变量就能跑"成为可能,降低上手门槛。
        """
        cfg = cls()

        config_path = Path(path) if path else DEFAULT_CONFIG_PATH
        if config_path.exists():
            raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
            cfg = _merge_dataclass(cfg, raw)

        cfg._apply_env_overrides()
        return cfg

    def _apply_env_overrides(self) -> None:
        """环境变量覆盖。命名规范: OCR_VERIFY_<SECTION>_<FIELD>。

        API Key 额外支持不带 section 的简写 OCR_VERIFY_API_KEY,
        因为这是最常被单独设置的一项。
        """
        env_key = os.getenv("OCR_VERIFY_API_KEY") or os.getenv("GEMINI_API_KEY")
        if env_key:
            self.vlm.api_key = env_key

        mapping: list[tuple[str, Any, str, type]] = [
            ("OCR_VERIFY_VLM_BASE_URL", self.vlm, "base_url", str),
            ("OCR_VERIFY_VLM_MODEL", self.vlm, "model", str),
            ("OCR_VERIFY_VLM_PROTOCOL", self.vlm, "protocol", str),
            ("OCR_VERIFY_VLM_ENABLED", self.vlm, "enabled", bool),
            ("OCR_VERIFY_ROUTER_THRESHOLD", self.router, "confidence_threshold", float),
            ("OCR_VERIFY_CACHE_ENABLED", self.cache, "enabled", bool),
            ("OCR_VERIFY_CACHE_CAPACITY", self.cache, "capacity", int),
            ("OCR_VERIFY_PADDLE_BACKEND", self.paddle, "backend", str),
            ("OCR_VERIFY_LOG_LEVEL", self, "log_level", str),
        ]

        for env_name, target, attr, caster in mapping:
            raw = os.getenv(env_name)
            if raw is None:
                continue
            try:
                value = _cast_env(raw, caster)
            except ValueError:
                continue  # 环境变量格式错误时静默忽略,不阻断启动
            setattr(target, attr, value)

    def validate(self) -> list[str]:
        """返回配置问题列表。空列表表示配置健康。

        刻意返回列表而非抛异常:允许调用方决定是"警告后继续"
        还是"直接退出",不同场景需求不同(单测 vs 正式评测)。
        """
        problems: list[str] = []

        if not 0.0 <= self.router.confidence_threshold <= 1.0:
            problems.append(
                f"router.confidence_threshold 应在 [0,1] 区间,"
                f"当前为 {self.router.confidence_threshold}"
            )
        if self.cache.capacity <= 0:
            problems.append(f"cache.capacity 必须为正数,当前为 {self.cache.capacity}")
        if self.cache.hamming_threshold < 0:
            problems.append("cache.hamming_threshold 不能为负数")
        if self.vlm.enabled and not self.vlm.api_key:
            problems.append(
                "vlm.enabled 为 true 但 api_key 为空 —— "
                "请在 config.yaml 填写,或设置环境变量 OCR_VERIFY_API_KEY。"
                "第二层将被自动跳过。"
            )
        if self.vlm.protocol not in ("gemini", "openai", "anthropic"):
            problems.append(
                f"vlm.protocol 仅支持 'gemini' / 'openai' / 'anthropic',"
                f"当前为 {self.vlm.protocol!r}"
            )
        if self.paddle.backend not in ("auto", "paddleocr", "rapidocr"):
            problems.append(f"paddle.backend 取值非法: {self.paddle.backend!r}")

        return problems

    def to_dict(self) -> dict[str, Any]:
        """导出为字典。API Key 会被脱敏,可安全写入报告。"""
        result = _dataclass_to_dict(self)
        key = result.get("vlm", {}).get("api_key", "")
        if key:
            result["vlm"]["api_key"] = f"{key[:4]}***{key[-4:]}" if len(key) > 8 else "***"
        return result


# ----------------------------------------------------------------------
# 内部工具
# ----------------------------------------------------------------------

def _cast_env(raw: str, caster: type) -> Any:
    if caster is bool:
        return raw.strip().lower() in ("1", "true", "yes", "on")
    return caster(raw)


def _merge_dataclass(instance: Any, data: dict[str, Any]) -> Any:
    """将字典递归合并进 dataclass 实例。未知字段直接忽略。

    忽略未知字段而非报错,是为了让旧版配置文件能在新版代码上继续工作,
    减少升级摩擦。
    """
    if not is_dataclass(instance) or not isinstance(data, dict):
        return instance

    for f in fields(instance):
        if f.name not in data:
            continue
        value = data[f.name]
        current = getattr(instance, f.name)
        if is_dataclass(current) and isinstance(value, dict):
            _merge_dataclass(current, value)
        elif value is not None:
            setattr(instance, f.name, value)
    return instance


def _dataclass_to_dict(instance: Any) -> Any:
    if is_dataclass(instance):
        return {f.name: _dataclass_to_dict(getattr(instance, f.name)) for f in fields(instance)}
    if isinstance(instance, (list, tuple)):
        return [_dataclass_to_dict(v) for v in instance]
    return instance


_cached_config: Optional[AppConfig] = None


def get_config(path: Optional[Path | str] = None, reload: bool = False) -> AppConfig:
    """获取全局配置单例。

    用单例是因为配置在进程生命周期内不变,
    避免每次调用都读盘解析 YAML。测试中可用 reload=True 强制重载。
    """
    global _cached_config
    if _cached_config is None or reload:
        _cached_config = AppConfig.load(path)
    return _cached_config
