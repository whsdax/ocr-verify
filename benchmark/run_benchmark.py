"""OCR 三方案横向评测。

运行三条独立的识别链路,用同一套 ground truth 计算指标:
  1. Tesseract 基线(可选)
  2. 纯 PaddleOCR(第一层无复核)
  3. 双层方案(第一层 + VLM 复核 + 缓存)

最后生成 JSON 结果,供 report.py 渲染 HTML 报告。

这条脚本本身就是项目最有说服力的证据之一:
"识别率从 XX% 提升到 XX%" 数字是否真实,跑一遍即可复现。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# 必须先把 src 挂进 sys.path,再 import ocr_verify ——
# 本脚本是以 `python benchmark/run_benchmark.py` 方式直接执行的,
# 此时 sys.path[0] 是 benchmark/ 而不是项目根目录,包不在默认搜索路径上。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ocr_verify.cache import OCRResultCache
from ocr_verify.config import AppConfig
from ocr_verify.engines.base import MockEngine
from ocr_verify.engines.paddle import PaddleEngine
from ocr_verify.engines.tesseract import TesseractEngine
from ocr_verify.engines.vlm import VLMEngine
from ocr_verify.metrics import AggregateMetrics, evaluate_sample, aggregate
from ocr_verify.router import TwoLayerRouter
from ocr_verify.types import EngineType, OCRResult, TextBox
from ocr_verify.verifier import OCRVerifier

DATASETS_DIR = PROJECT_ROOT / "datasets"
REPORTS_DIR = PROJECT_ROOT / "reports"
GROUND_TRUTH = DATASETS_DIR / "ground_truth.json"


def load_ground_truth(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"未找到标注文件 {path}。请先运行: python benchmark/build_dataset.py"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_image_path(relpath: str) -> Path:
    return DATASETS_DIR / relpath


def run_single(
    image_path: Path,
    meta: dict[str, Any],
    runner: Any,
    runner_name: str,
    use_cache: bool = False,
) -> dict[str, Any]:
    """对一张图片跑识别,并评估。runner 可以是 OCREngine 或 OCRVerifier。"""
    t0 = time.perf_counter()
    result = runner.recognize(str(image_path))
    wall_ms = (time.perf_counter() - t0) * 1000

    # 如果 runner 自身没有耗时,用 wall clock 补齐
    if not result.elapsed_ms:
        result.elapsed_ms = wall_ms

    ev = evaluate_sample(
        sample_id=image_path.name,
        prediction=result.text,
        ground_truth=meta["text"],
        confidence=result.confidence,
        elapsed_ms=result.elapsed_ms,
        engine=result.engine.value,
        escalated=result.escalated,
        from_cache=result.from_cache,
        degraded=result.degraded,
        perturbation=meta.get("perturbation", "none"),
        error=result.error,
    )
    return {"runner": runner_name, "result": result, "eval": ev}


def run_paddle_only(config: AppConfig, gt: dict[str, dict[str, Any]]) -> AggregateMetrics:
    """纯第一层,无复核。"""
    engine = PaddleEngine(
        backend=config.paddle.backend,
        lang=config.paddle.lang,
        det_db_thresh=config.paddle.det_db_thresh,
        drop_score=config.paddle.drop_score,
    )
    engine.warmup()
    evals = []
    print("\n[1/3] 评估:纯 PaddleOCR ...")
    for name, meta in gt.items():
        img_path = resolve_image_path(meta["relpath"])
        row = run_single(img_path, meta, engine, "paddle")
        evals.append(row["eval"])
    return aggregate("纯 PaddleOCR", evals)


def run_tesseract(config: AppConfig, gt: dict[str, dict[str, Any]]) -> AggregateMetrics | None:
    if not config.benchmark.include_tesseract:
        print("\n[基线] 跳过 Tesseract(include_tesseract=false)")
        return None

    engine = TesseractEngine(lang="chi_sim+eng")
    if not engine.is_available():
        print("\n[基线] 本机未安装 Tesseract,跳过")
        return None

    evals = []
    print("\n[基线] 评估:Tesseract ...")
    for name, meta in gt.items():
        img_path = resolve_image_path(meta["relpath"])
        row = run_single(img_path, meta, engine, "tesseract")
        evals.append(row["eval"])
    return aggregate("Tesseract 基线", evals)


def run_two_layer(config: AppConfig, gt: dict[str, dict[str, Any]]) -> AggregateMetrics:
    """双层方案 + 缓存。"""
    print("\n[2/3] 评估:双层方案(第一层 + VLM 复核 + 缓存) ...")
    verifier = OCRVerifier(config)
    verifier.warmup()

    evals = []
    for name, meta in gt.items():
        img_path = resolve_image_path(meta["relpath"])
        row = run_single(img_path, meta, verifier, "two_layer", use_cache=True)
        evals.append(row["eval"])

    return aggregate("双层方案", evals)


def run_two_layer_no_cache(config: AppConfig, gt: dict[str, dict[str, Any]]) -> AggregateMetrics:
    """双层方案但禁用缓存,用于计算"缓存带来的额外收益"。"""
    print("\n[3/3] 评估:双层方案(禁用缓存) ...")
    verifier = OCRVerifier(config)
    verifier.cache.enabled = False
    verifier.warmup()

    evals = []
    for name, meta in gt.items():
        img_path = resolve_image_path(meta["relpath"])
        row = run_single(img_path, meta, verifier, "two_layer_no_cache", use_cache=False)
        evals.append(row["eval"])

    return aggregate("双层方案(无缓存)", evals)


def _build_dry_verifier(config: AppConfig, gt: dict[str, dict[str, Any]], cache_enabled: bool) -> OCRVerifier:
    """构造一个使用 Mock VLM 的 Verifier,VLM 固定返回 ground truth。

    用于 dry-run:不消耗真实 API,但能完整验证缓存、路由、评测链路。
    """
    from ocr_verify.engines.base import MockEngine
    from ocr_verify.types import EngineType

    verifier = OCRVerifier(config)
    # 用 Mock 替换第二层:对每个样本返回标注文本作为"完美复核结果"
    class GroundTruthVLM(MockEngine):
        def _recognize_impl(self, image, **kwargs):
            # image 参数在这里是路径字符串
            path = Path(image)
            meta = gt.get(path.name, {})
            text = meta.get("text", "")
            return OCRResult(
                text=text,
                confidence=0.95,
                boxes=[TextBox(text=text, confidence=0.95)],
                engine=EngineType.VLM,
            )

    verifier.second_layer = GroundTruthVLM(engine_type=EngineType.VLM)
    verifier.router.second_layer = verifier.second_layer
    verifier.cache.enabled = cache_enabled
    return verifier


def run_two_layer_dry(config: AppConfig, gt: dict[str, dict[str, Any]]) -> AggregateMetrics:
    print("\n[2/3] 评估:双层方案(MOCK VLM + 缓存) ...")
    verifier = _build_dry_verifier(config, gt, cache_enabled=True)
    verifier.warmup()

    evals = []
    for name, meta in gt.items():
        img_path = resolve_image_path(meta["relpath"])
        row = run_single(img_path, meta, verifier, "two_layer_dry", use_cache=True)
        evals.append(row["eval"])
    return aggregate("双层方案(dry-run)", evals)


def run_two_layer_no_cache_dry(config: AppConfig, gt: dict[str, dict[str, Any]]) -> AggregateMetrics:
    print("\n[3/3] 评估:双层方案(MOCK VLM,无缓存) ...")
    verifier = _build_dry_verifier(config, gt, cache_enabled=False)
    verifier.warmup()

    evals = []
    for name, meta in gt.items():
        img_path = resolve_image_path(meta["relpath"])
        row = run_single(img_path, meta, verifier, "two_layer_dry_no_cache", use_cache=False)
        evals.append(row["eval"])
    return aggregate("双层方案(dry-run,无缓存)", evals)


def main() -> None:
    parser = argparse.ArgumentParser(description="运行 OCR 三方案横向评测")
    parser.add_argument(
        "--config", type=str, default=str(PROJECT_ROOT / "config.yaml"), help="配置文件路径"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出 JSON 路径,默认 reports/benchmark_{timestamp}.json",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="用 Mock VLM 跑通流程,不消耗真实 API,用于冒烟测试",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(GROUND_TRUTH),
        help="ground truth JSON 文件路径",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="只评测前 N 个样本。非 dry-run 时每个升级样本都会真实调用一次 API,"
        "先用小样本(如 --limit 20)确认链路和成本,再跑全量",
    )
    args = parser.parse_args()

    config = AppConfig.load(args.config)
    # 校验:提示 API Key 缺失,但不阻止运行
    for p in config.validate():
        print(f"配置提醒: {p}")

    gt = load_ground_truth(Path(args.dataset))
    if not gt:
        print("评测集为空,请先运行 build_dataset.py 生成数据")
        return

    if args.limit is not None and args.limit > 0:
        # dict 保持插入顺序,截断即"取前 N 个样本",可复现
        gt = dict(list(gt.items())[: args.limit])
        print(f"\n*** 已限制样本数为 {len(gt)}(--limit {args.limit}) ***")

    # dry-run:用 Mock 引擎替换 VLM,不消耗额度,但跑通完整链路
    original_vlm: Any = None
    if args.dry_run:
        print("\n*** DRY RUN 模式:使用 Mock VLM,不消耗 API ***")

    metrics_list: list[AggregateMetrics | None] = []

    tesseract_metrics = run_tesseract(config, gt)
    metrics_list.append(tesseract_metrics)

    paddle_metrics = run_paddle_only(config, gt)
    metrics_list.append(paddle_metrics)

    # 双层方案会真实调 API,成本较高,因此放在最后
    if args.dry_run:
        two_metrics = run_two_layer_dry(config, gt)
        metrics_list.append(two_metrics)
        two_no_cache = run_two_layer_no_cache_dry(config, gt)
        metrics_list.append(two_no_cache)
    else:
        two_metrics = run_two_layer(config, gt)
        metrics_list.append(two_metrics)
        two_no_cache = run_two_layer_no_cache(config, gt)
        metrics_list.append(two_no_cache)

    # 聚合报告
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output) if args.output else _default_report_path()

    report_payload: dict[str, Any] = {
        "meta": {
            "sample_count": len(gt),
            "config": config.to_dict(),
            "dry_run": args.dry_run,
        },
        "metrics": [m.to_dict() for m in metrics_list if m is not None],
    }

    output_path.write_text(
        json.dumps(report_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\n结果已保存: {output_path}")
    print("\n=== 汇总 ===")
    for m in metrics_list:
        if m is None:
            continue
        print(
            f"{m.name:20s} 精确匹配 {m.exact_match_rate:.1%}  "
            f"字符准确 {m.mean_char_acc:.1%}  平均延迟 {m.mean_latency_ms:.0f}ms  "
            f"升级 {m.escalation_count}"
        )


def _default_report_path() -> Path:
    import datetime

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return REPORTS_DIR / f"benchmark_{ts}.json"


if __name__ == "__main__":
    main()
