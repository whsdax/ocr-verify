"""置信度阈值校准 —— 回答面试官"0.7 是怎么来的"。

算法
----
在评测集上只跑第一层(PaddleOCR),记录每个样本:
  - 第一层置信度 confidence
  - 是否正确(exact_match)

然后扫描阈值 τ 从 0 到 1(步长 0.02):
  - 对于每个 τ,把所有 confidence < τ 的样本交给"完美的第二层"
    (即用 ground truth 替代) → 这些样本全部判对
  - confidence >= τ 的样本使用第一层结果,有的对有的错
  - 计算最终准确率

同时计算 VLM 调用率 = confidence < τ 的样本比例。

绘制两条曲线:
  - 最终准确率 vs τ
  - VLM 调用率 vs τ

拐点定义
--------
在准确率已接近饱和的位置,选择 VLM 调用率仍然较低的那个阈值。
例如:τ=0.68 时准确率 95.2%,τ=0.70 时准确率 95.3%,τ=0.72 时准确率 95.3%,
则选择 0.70 而非 0.72 —— 因为进一步提升阈值的收益几乎为零,
却会让更多样本走昂贵的第二层。

注意事项
--------
这里用 ground truth 模拟"完美第二层",得到的是阈值选择的**理论上限**。
真实 VLM 并非完美,所以实际最终准确率会比这条曲线低一些。
但它仍然能回答"阈值如何权衡准确率与成本"这个核心问题。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ocr_verify.config import AppConfig
from ocr_verify.engines.paddle import PaddleEngine
from ocr_verify.metrics import normalize_text
from ocr_verify.cache.fingerprint import load_image

DATASETS_DIR = PROJECT_ROOT / "datasets"


def load_ground_truth(path: Path) -> dict[str, dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def gather_first_layer_scores(
    config: AppConfig, gt: dict[str, dict[str, Any]]
) -> list[tuple[float, bool, str, str]]:
    """返回 [(confidence, is_correct, sample_id, perturbation), ...]。"""
    engine = PaddleEngine(
        backend=config.paddle.backend,
        lang=config.paddle.lang,
        det_db_thresh=config.paddle.det_db_thresh,
        drop_score=config.paddle.drop_score,
    )
    engine.warmup()

    rows: list[tuple[float, bool, str, str]] = []
    for name, meta in gt.items():
        img = load_image(str(DATASETS_DIR / meta["relpath"]))
        res = engine.recognize(img)

        pred = normalize_text(res.text)
        truth = normalize_text(meta["text"])
        is_correct = (pred == truth)
        rows.append((
            res.confidence,
            is_correct,
            name,
            meta.get("perturbation", "none"),
        ))
    return rows


def calibrate(rows: list[tuple[float, bool, str, str]]) -> list[dict[str, Any]]:
    """扫描阈值,返回每个阈值点的指标。"""
    results = []
    thresholds = [round(i * 0.02, 3) for i in range(51)]  # 0.00 ~ 1.00

    for tau in thresholds:
        correct = 0
        vlm_calls = 0
        by_perturbation: dict[str, dict[str, int]] = {}

        for conf, is_correct, sample_id, pert in rows:
            # 低于阈值 → 升级(模拟完美 VLM),判对
            if conf < tau:
                correct += 1
                vlm_calls += 1
                by_perturbation.setdefault(pert, {"vlm": 0, "total": 0})
                by_perturbation[pert]["vlm"] += 1
            else:
                # 不低于阈值 → 用第一层结果
                if is_correct:
                    correct += 1

            by_perturbation.setdefault(pert, {"vlm": 0, "total": 0})
            by_perturbation[pert]["total"] += 1

        total = len(rows)
        results.append({
            "threshold": tau,
            "final_accuracy": round(correct / total, 4),
            "vlm_call_rate": round(vlm_calls / total, 4),
            "vlm_calls": vlm_calls,
            "first_layer_kept": total - vlm_calls,
            "by_perturbation": by_perturbation,
        })

    return results


def suggest_threshold(results: list[dict[str, Any]]) -> dict[str, Any]:
    """根据"准确率收益递减"原则推荐阈值。

    策略:从 τ=1.0 往低走,找到准确率首次达到最大准确率 - 1% 范围内的点,
    且 VLM 调用率不过高的位置。这是一个启发式,最终仍应人工确认。
    """
    max_acc = max(r["final_accuracy"] for r in results)
    target_acc = max_acc - 0.01

    # 优先找准确率 >= target_acc 且 vlm_call_rate 最小的点(即尽量省 VLM)
    candidates = [r for r in results if r["final_accuracy"] >= target_acc]
    if not candidates:
        candidates = results

    best = min(candidates, key=lambda r: r["vlm_call_rate"])
    return {
        "recommended_threshold": best["threshold"],
        "expected_accuracy": best["final_accuracy"],
        "vlm_call_rate": best["vlm_call_rate"],
        "max_accuracy": max_acc,
    }


def plot_curve(results: list[dict[str, Any]], output_path: Path) -> None:
    """绘制阈值-准确率-调用率曲线。"""
    try:
        import matplotlib
        import matplotlib.pyplot as plt
    except ImportError:
        print("未安装 matplotlib,跳过绘图。可执行 pip install matplotlib 后重试。")
        return

    # 尝试使用系统常见中文字体,避免中文标签显示为方块
    for font_name in ["Microsoft YaHei", "SimHei", "PingFang SC", "Noto Sans CJK SC"]:
        try:
            matplotlib.rcParams["font.family"] = [font_name, "sans-serif"]
            matplotlib.rcParams["axes.unicode_minus"] = False
            break
        except Exception:  # noqa: BLE001
            continue

    xs = [r["threshold"] for r in results]
    accs = [r["final_accuracy"] for r in results]
    rates = [r["vlm_call_rate"] for r in results]

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(xs, accs, "b-", label="最终准确率", linewidth=2)
    ax1.set_xlabel("置信度阈值 τ")
    ax1.set_ylabel("准确率", color="b")
    ax1.tick_params(axis="y", labelcolor="b")
    ax1.set_ylim(0, 1.05)

    ax2 = ax1.twinx()
    ax2.plot(xs, rates, "r--", label="VLM 调用率", linewidth=2)
    ax2.set_ylabel("VLM 调用率", color="r")
    ax2.tick_params(axis="y", labelcolor="r")
    ax2.set_ylim(0, 1.05)

    fig.legend(loc="upper right", bbox_to_anchor=(0.88, 0.92))
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"校准曲线已保存: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="校准双层 OCR 置信度阈值")
    parser.add_argument(
        "--config", type=str, default=str(PROJECT_ROOT / "config.yaml"), help="配置文件"
    )
    parser.add_argument(
        "--dataset", type=str, default=str(PROJECT_ROOT / "datasets/ground_truth.json"), help="标注文件"
    )
    parser.add_argument(
        "--output", type=str, default=str(PROJECT_ROOT / "reports/calibration.json"), help="输出 JSON"
    )
    args = parser.parse_args()

    config = AppConfig.load(args.config)
    gt = load_ground_truth(Path(args.dataset))

    print("采集第一层置信度与正确性...")
    rows = gather_first_layer_scores(config, gt)

    print("扫描阈值 0.00 ~ 1.00 ...")
    results = calibrate(rows)

    suggestion = suggest_threshold(results)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {"suggestion": suggestion, "curve": results},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n=== 推荐阈值 ===")
    print(json.dumps(suggestion, ensure_ascii=False, indent=2))

    plot_curve(results, out_path.with_suffix(".png"))


if __name__ == "__main__":
    main()
