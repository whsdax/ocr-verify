"""命令行入口。

提供两个子命令:
  - recognize: 单张图片识别,快速验证效果
  - benchmark: 运行评测并生成报告
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .cache.fingerprint import ImageInput
from .config import AppConfig
from .verifier import OCRVerifier


def _load_config(path: str | None) -> AppConfig:
    return AppConfig.load(path)


def cmd_recognize(args: argparse.Namespace) -> int:
    config = _load_config(args.config)
    problems = config.validate()
    for p in problems:
        print(f"配置提醒: {p}")

    verifier = OCRVerifier(config)
    result = verifier.recognize(
        args.image,
        expected_pattern=args.pattern,
        force_escalate=args.force_vlm,
        use_cache=not args.no_cache,
    )

    print(result.summary())
    if args.json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    # benchmark/run_benchmark.py 作为独立脚本执行,这里仅做转发
    benchmark_script = Path(__file__).resolve().parents[2] / "benchmark" / "run_benchmark.py"
    if not benchmark_script.exists():
        print(f"未找到评测脚本: {benchmark_script}")
        return 1

    import subprocess

    cmd = [sys.executable, str(benchmark_script)]
    if args.config:
        cmd.extend(["--config", args.config])
    if args.dry_run:
        cmd.append("--dry-run")
    if args.output:
        cmd.extend(["--output", args.output])
    if args.dataset:
        cmd.extend(["--dataset", args.dataset])

    return subprocess.call(cmd)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ocr-verify",
        description="智能 OCR 测试验证系统 CLI",
    )
    parser.add_argument(
        "--config", type=str, default=None, help="配置文件路径,默认 config.yaml"
    )

    sub = parser.add_subparsers(dest="command", required=True)

    rec = sub.add_parser("recognize", help="识别单张图片")
    rec.add_argument("image", type=str, help="图片路径")
    rec.add_argument(
        "--pattern", type=str, default=None, help="预期文本正则(可选)"
    )
    rec.add_argument(
        "--force-vlm", action="store_true", help="强制走多模态模型复核"
    )
    rec.add_argument(
        "--no-cache", action="store_true", help="不使用缓存"
    )
    rec.add_argument(
        "--json", action="store_true", help="输出完整 JSON"
    )

    bench = sub.add_parser("benchmark", help="运行评测")
    bench.add_argument(
        "--dry-run", action="store_true", help="用 Mock VLM 跑,不消耗 API"
    )
    bench.add_argument("--output", type=str, default=None, help="输出 JSON 路径")
    bench.add_argument(
        "--dataset", type=str, default=None, help="ground truth JSON 路径"
    )

    args = parser.parse_args(argv)

    if args.command == "recognize":
        return cmd_recognize(args)
    if args.command == "benchmark":
        return cmd_benchmark(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
