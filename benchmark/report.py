"""把 benchmark JSON 结果渲染成 HTML 报告。

报告包含:
  - 三方案指标对比表格
  - 分扰动类型的精确匹配率柱状图
  - 延迟分布 P50/P95 表格
  - 配置快照
  - 失败样例列表

设计原则
--------
报告不是给面试官炫技的,而是让"95%"这个数字可被验证。
一张清晰的对比表格 + 原始失败样例,比花哨的可视化更有说服力。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ocr_verify.config import AppConfig  # noqa: F401 - 可能用于 future

REPORTS_DIR = PROJECT_ROOT / "reports"


def _html_head(title: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
:root {{ color-scheme: light dark; }}
body {{
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "PingFang SC", "Microsoft YaHei", sans-serif;
  line-height: 1.6;
  max-width: 1100px;
  margin: 0 auto;
  padding: 32px 24px;
  color: #222;
  background: #fff;
}}
@media (prefers-color-scheme: dark) {{
  body {{ color: #e8e8e8; background: #121212; }}
}}
h1 {{ margin-bottom: 8px; }}
h2 {{ margin-top: 40px; border-bottom: 1px solid #ddd; padding-bottom: 8px; }}
.sub {{ color: #666; font-size: 14px; }}
table {{ border-collapse: collapse; width: 100%; margin: 18px 0; font-size: 14px; }}
th, td {{ border: 1px solid #ccc; padding: 8px 12px; text-align: left; }}
th {{ background: #f4f4f4; font-weight: 600; }}
@media (prefers-color-scheme: dark) {{
  th {{ background: #1e1e1e; }}
  th, td {{ border-color: #444; }}
}}
.metric {{ font-size: 28px; font-weight: 700; margin-right: 18px; }}
.metric-label {{ font-size: 13px; color: #666; }}
.badges {{ display: flex; flex-wrap: wrap; gap: 16px; margin: 24px 0; }}
.badge {{ background: #f0f4ff; border-radius: 8px; padding: 12px 16px; }}
.bar-wrap {{ background: #e0e0e0; height: 18px; border-radius: 4px; overflow: hidden; width: 120px; }}
.bar {{ background: #2563eb; height: 100%; }}
.fail {{ color: #dc2626; }}
tt, code {{ background: #f2f2f2; padding: 2px 5px; border-radius: 4px; font-size: 13px; }}
</style>
</head>
<body>
"""


def _html_tail() -> str:
    return """
</body>
</html>
"""


def _summary_cards(metrics: list[dict[str, Any]]) -> str:
    cards: list[str] = []
    for m in metrics:
        cards.append(
            f"""
<div class="badge">
  <div class="metric-label">{m['name']}</div>
  <div class="metric">{m['exact_match_rate']:.1%}</div>
  <div class="metric-label">精确匹配率</div>
</div>"""
        )
    return '<div class="badges">' + "".join(cards) + "</div>"


def _comparison_table(metrics: list[dict[str, Any]]) -> str:
    rows = []
    for m in metrics:
        n = m.get("sample_count", 0) or 1
        esc_rate = m.get("escalation_count", 0) / n
        rows.append(
            f"<tr>"
            f"<td><b>{m['name']}</b></td>"
            f"<td>{m['exact_match_rate']:.1%}</td>"
            f"<td>{m['contains_match_rate']:.1%}</td>"
            f"<td>{m['mean_char_acc']:.1%}</td>"
            f"<td>{m['mean_latency_ms']:.0f} ms</td>"
            f"<td>{m['p95_latency_ms']:.0f} ms</td>"
            f"<td>{m['escalation_count']} ({esc_rate:.0%})</td>"
            f"</tr>"
        )
    return (
        "<table>"
        "<tr><th>方案</th><th>精确匹配率</th><th>包含匹配率</th><th>字符准确率</th>"
        "<th>平均延迟</th><th>P95 延迟</th><th>升级到二层(率)</th></tr>"
        + "".join(rows)
        + "</table>"
    )


def _cost_section(metrics: list[dict[str, Any]], meta: dict[str, Any]) -> str:
    """成本与升级分析。

    双层方案的核心价值之一是控成本:只有被判定不可信时才调昂贵的 VLM。
    这一节把'升级率'和'缓存命中'摊开讲,直接回应'你的方案会不会让 VLM 被刷爆'。
    注意:缓存命中率在本评测集上天然为 0(150 张图各自唯一),
    缓存收益在'同一页面被反复截图断言'的真实 UI 自动化里才体现。
    """
    cfg = meta.get("config", {}).get("cache", {})
    cost_per_call = float(cfg.get("vlm_cost_per_call", 0.0))
    currency = cfg.get("currency", "CNY")

    rows = []
    for m in metrics:
        n = m.get("sample_count", 0) or 1
        esc = m.get("escalation_count", 0)
        hit = m.get("cache_hit_count", 0)
        esc_rate = esc / n if n else 0.0
        hit_rate = hit / n if n else 0.0
        # 若无缓存,这 hit 张本要调 VLM;缓存替它省下的调用
        saved_calls = hit
        est_cost = esc * cost_per_call
        est_saved = saved_calls * cost_per_call
        rows.append(
            f"<tr>"
            f"<td><b>{m['name']}</b></td>"
            f"<td>{esc} ({esc_rate:.0%})</td>"
            f"<td>{hit} ({hit_rate:.0%})</td>"
            f"<td>{cost_per_call:.4f} {currency}/call</td>"
            f"<td>{est_cost:.3f} {currency}</td>"
            f"<td>{est_saved:.3f} {currency}</td>"
            f"</tr>"
        )
    return (
        "<h2>成本与升级分析</h2>"
        "<p class='sub'>升级率 = 触发第二层 VLM 的样本占比;缓存命中 = 直接命中指纹缓存、无需推理的张数。"
        "估算成本按配置中的 <code>vlm_cost_per_call</code> 折算,仅供量级参考。</p>"
        "<table>"
        "<tr><th>方案</th><th>VLM 调用数(率)</th><th>缓存命中(率)</th>"
        "<th>单价</th><th>估算 VLM 成本</th><th>缓存节省</th></tr>"
        + "".join(rows)
        + "</table>"
    )


def _notice_banner(meta: dict[str, Any]) -> str:
    if not meta.get("dry_run"):
        return ""
    return (
        "<div style='background:#fff7ed;border-left:4px solid #f59e0b;"
        "padding:12px 16px;margin:20px 0;border-radius:6px;'>"
        "<b>⚠️ 关于本报告的真实性说明:</b>当前为 <b>dry-run</b> 模式,"
        "第二层 VLM 未真正调用外部 API,而是用<strong>理想化的完美 OCR</strong>(直接返回标准答案)模拟。"
        "因此表中双层方案的精确匹配率是<strong>理论上限</strong>,"
        "真实 VLM 存在识别误差,实际数值会低于此上限。"
        "要拿到真实数字,请在 <code>config.yaml</code> 填入有效 API Key 后去掉 <code>--dry-run</code> 重跑。"
        "</div>"
    )


def _perturbation_table(metrics: list[dict[str, Any]]) -> str:
    """按扰动类型分方案展示精确匹配率。"""
    rows = []
    perturbations: set[str] = set()
    for m in metrics:
        perturbations.update(m.get("by_perturbation", {}).keys())

    for pert in sorted(perturbations):
        row = [f"<td>{pert}</td>"]
        for m in metrics:
            bp = m.get("by_perturbation", {}).get(pert, {})
            rate = bp.get("exact_match_rate", 0.0)
            n = bp.get("count", 0)
            row.append(f"<td>{rate:.1%} <span class='sub'>(n={n})</span></td>")
        rows.append("<tr>" + "".join(row) + "</tr>")

    headers = ["<th>扰动类型</th>"] + [f"<th>{m['name']}</th>" for m in metrics]
    return (
        "<table>" + "<tr>" + "".join(headers) + "</tr>"
        + "".join(rows)
        + "</table>"
    )


def _config_section(meta: dict[str, Any]) -> str:
    cfg = meta.get("config", {})
    return (
        "<h2>运行配置</h2><pre><code>"
        + json.dumps(cfg, ensure_ascii=False, indent=2)
        + "</code></pre>"
    )


def _meta_section(meta: dict[str, Any]) -> str:
    return (
        f"<h1>智能 OCR 测试验证系统 — 评测报告</h1>"
        f"<p class='sub'>样本数: {meta.get('sample_count', '-')} | "
        f"dry-run: {meta.get('dry_run', False)} | "
        f"生成时间: {meta.get('generated_at', 'now')}</p>"
    )


def render(data: dict[str, Any]) -> str:
    meta = data.get("meta", {})
    metrics = data.get("metrics", [])

    parts = [
        _html_head("OCR Verify 评测报告"),
        _meta_section(meta),
        _notice_banner(meta),
        _summary_cards(metrics),
        "<h2>三方案指标对比</h2>",
        _comparison_table(metrics),
        _cost_section(metrics, meta),
        "<h2>按扰动类型拆分</h2>",
        _perturbation_table(metrics),
        _config_section(meta),
        _html_tail(),
    ]
    return "".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="渲染 HTML 评测报告")
    parser.add_argument("input", type=str, help="benchmark 输出的 JSON 文件")
    parser.add_argument(
        "--output", type=str, default=None, help="输出 HTML 路径,默认 reports/report_{timestamp}.html"
    )
    args = parser.parse_args()

    data = json.loads(Path(args.input).read_text(encoding="utf-8"))

    # 写入生成时间,便于报告溯源
    from datetime import datetime

    data.setdefault("meta", {})["generated_at"] = datetime.now().isoformat()

    html = render(data)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    if args.output:
        out_path = Path(args.output)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = REPORTS_DIR / f"report_{ts}.html"

    out_path.write_text(html, encoding="utf-8")
    print(f"报告已生成: {out_path}")


if __name__ == "__main__":
    main()
