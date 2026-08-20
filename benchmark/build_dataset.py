"""评测数据集构建 —— 合成 UI 截图 + 程序化扰动。

为什么用合成截图而不是真实业务截图
----------------------------------
1. **合规**:真实业务截图属于雇主资产,个人项目使用存在风险。
2. **可复现**:任何人 clone 仓库后跑一条命令就能得到完全相同的数据集,
   评测结果可被第三方验证 —— 这是实验可信度的基础。
3. **标注免费且零误差**:文字是我们自己画上去的,ground truth 天然精确。
   人工标注 50 张图不仅费时,还会引入标注噪声。
4. **扰动可控**:能精确控制"遮挡率 30%"这样的变量,
   做单变量对照实验。真实截图做不到这一点。

这套方法本身就是测试用例设计能力的体现:
把不可控的真实样本,转化为可控、可复现、可量化的实验设计。

扰动类型对应的真实故障
----------------------
  occlusion  -> 弹窗、Toast、加载遮罩盖住文字
  blur       -> 页面切换动画未完成时截图、视频流截帧
  lowcontrast-> 深色模式适配不良、背景图干扰
  jpeg       -> 截图经过有损压缩传输(如通过 IM 回传日志)
  scale      -> 不同 DPI 设备上的渲染差异

用法
----
    python benchmark/build_dataset.py --count 50
    python benchmark/build_dataset.py --count 50 --seed 42 --clean
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

DATASETS_DIR = PROJECT_ROOT / "datasets"
RAW_DIR = DATASETS_DIR / "raw"
PERTURBED_DIR = DATASETS_DIR / "perturbed"
GROUND_TRUTH = DATASETS_DIR / "ground_truth.json"

# 中文字体候选。微软雅黑是 Windows UI 的默认字体,最贴近真实界面。
FONT_CANDIDATES = [
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simhei.ttf",
    "C:/Windows/Fonts/simsun.ttc",
    "/System/Library/Fonts/PingFang.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
]

# ----------------------------------------------------------------------
# 语料:模拟真实 App 界面的文本
# ----------------------------------------------------------------------

UI_TEMPLATES: list[dict[str, Any]] = [
    {"kind": "dialog", "lines": ["确认删除该文件?", "删除后无法恢复", "取消", "确定"]},
    {"kind": "toast", "lines": ["保存成功"]},
    {"kind": "form", "lines": ["用户名", "admin@example.com", "密码", "登录"]},
    {"kind": "list", "lines": ["设置", "账号与安全", "消息通知", "隐私", "关于"]},
    {"kind": "order", "lines": ["订单号 202608110037", "实付金额 ¥128.50", "待发货"]},
    {"kind": "status", "lines": ["设备已连接", "固件版本 v2.14.3", "电量 87%"]},
    {"kind": "error", "lines": ["网络连接失败", "错误码 -1009", "请检查网络设置", "重试"]},
    {"kind": "profile", "lines": ["个人中心", "会员等级 黄金", "积分 3280", "有效期至 2027-06-30"]},
    {"kind": "progress", "lines": ["正在上传", "45.2 MB / 128.0 MB", "剩余时间 00:32"]},
    {"kind": "search", "lines": ["搜索结果", "共找到 156 条记录", "按相关度排序"]},
    {"kind": "camera", "lines": ["拍摄模式", "4K 60fps", "剩余可拍 01:24:35", "ISO 400"]},
    {"kind": "battery", "lines": ["电池健康度 94%", "充电循环 218 次", "预计续航 6 小时"]},
]

# 深浅两套配色,模拟浅色/深色模式
THEMES = [
    {"bg": (250, 250, 252), "fg": (32, 33, 36), "accent": (24, 96, 200), "sub": (110, 112, 118)},
    {"bg": (30, 31, 34), "fg": (232, 234, 237), "accent": (120, 180, 255), "sub": (154, 160, 166)},
]


def find_font(size: int) -> ImageFont.FreeTypeFont:
    for path in FONT_CANDIDATES:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except Exception:  # noqa: BLE001
                continue
    raise RuntimeError(
        "未找到可用的中文字体。请在 FONT_CANDIDATES 中添加系统字体路径。"
    )


# ----------------------------------------------------------------------
# 合成原图
# ----------------------------------------------------------------------

def render_ui(template: dict[str, Any], rng: random.Random) -> tuple[np.ndarray, str]:
    """渲染一张合成 UI 截图,返回 (BGR 图像, ground truth 文本)。"""
    theme = rng.choice(THEMES)
    width = rng.choice([420, 480, 540])
    lines: list[str] = template["lines"]

    title_size = rng.choice([22, 24, 26])
    body_size = rng.choice([16, 17, 18])

    padding = 28
    line_gap = rng.choice([16, 20, 24])
    height = padding * 2 + title_size + line_gap
    height += (body_size + line_gap) * (len(lines) - 1)
    height = max(height, 180)

    img = Image.new("RGB", (width, height), theme["bg"])
    draw = ImageDraw.Draw(img)

    title_font = find_font(title_size)
    body_font = find_font(body_size)

    # 顶部装饰条,模拟真实界面的标题栏分隔
    draw.rectangle([0, 0, width, 4], fill=theme["accent"])

    y = padding
    for idx, line in enumerate(lines):
        if idx == 0:
            font, color = title_font, theme["fg"]
        elif line in ("确定", "登录", "重试"):
            # 主按钮:用强调色绘制,模拟真实的可点击控件
            font, color = body_font, theme["accent"]
        else:
            font, color = body_font, theme["sub"]

        draw.text((padding, y), line, font=font, fill=color)
        y += (title_size if idx == 0 else body_size) + line_gap

    bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    # ground truth 用换行连接,与引擎的聚合方式保持一致
    return bgr, "\n".join(lines)


# ----------------------------------------------------------------------
# 扰动算子
# ----------------------------------------------------------------------

def apply_occlusion(img: np.ndarray, ratio: float, rng: random.Random) -> np.ndarray:
    """半透明矩形遮挡,模拟弹窗 / Toast / 加载遮罩。

    用 addWeighted 做 alpha 混合而非直接覆盖,因为真实弹窗
    多数是半透明的 —— 底层文字仍有微弱可见性,这正是
    "PaddleOCR 会给出高置信度错误结果"的典型场景。
    """
    out = img.copy()
    h, w = out.shape[:2]

    # 遮挡区域的高度按比例计算,横向覆盖大部分宽度(模拟横幅式弹窗)
    occ_h = max(int(h * ratio), 12)
    occ_w = int(w * rng.uniform(0.6, 0.95))
    y0 = rng.randint(0, max(h - occ_h, 1))
    x0 = rng.randint(0, max(w - occ_w, 1))

    overlay = out.copy()
    color = (60, 60, 60) if out.mean() > 127 else (200, 200, 200)
    cv2.rectangle(overlay, (x0, y0), (x0 + occ_w, y0 + occ_h), color, thickness=-1)

    alpha = rng.uniform(0.75, 0.92)  # 不完全不透明,保留一丝底层痕迹
    cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0, out)
    return out


def apply_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    """高斯模糊,模拟动画未完成 / 对焦失败时的截图。"""
    # 核大小取 sigma 的 6 倍并保证为奇数,覆盖高斯分布的主要能量
    ksize = int(sigma * 6) | 1
    return cv2.GaussianBlur(img, (ksize, ksize), sigma)


def apply_low_contrast(img: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """线性压缩对比度,模拟深色模式适配不良 / 屏幕反光。

    out = img * alpha + beta。alpha < 1 压缩动态范围,beta 抬高整体亮度,
    两者叠加后前景与背景的差异变小,这是 OCR 检测器失效的常见诱因。
    """
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)


def apply_jpeg(img: np.ndarray, quality: int) -> np.ndarray:
    """JPEG 有损压缩,模拟截图经 IM / 日志系统传输后的质量损失。"""
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return img
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def apply_scale(img: np.ndarray, factor: float) -> np.ndarray:
    """先缩小再放大,模拟低 DPI 设备渲染 / 图片被二次缩放。

    这个操作会不可逆地丢失高频细节(文字笔画),
    比单纯缩放更接近真实的画质劣化。
    """
    h, w = img.shape[:2]
    small = cv2.resize(
        img, (max(int(w * factor), 8), max(int(h * factor), 8)),
        interpolation=cv2.INTER_AREA,
    )
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


# 扰动配置表:(名称, 函数, 参数字典)
def build_perturbations(rng: random.Random) -> list[tuple[str, Any]]:
    return [
        ("none", lambda im: im),
        ("occlusion_light", lambda im: apply_occlusion(im, 0.12, rng)),
        ("occlusion_heavy", lambda im: apply_occlusion(im, 0.32, rng)),
        ("blur_light", lambda im: apply_blur(im, 1.2)),
        ("blur_heavy", lambda im: apply_blur(im, 2.4)),
        ("lowcontrast", lambda im: apply_low_contrast(im, 0.45, 95)),
        ("jpeg_low", lambda im: apply_jpeg(im, 28)),
        ("scale_down", lambda im: apply_scale(im, 0.45)),
    ]


# ----------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------

def build(count: int, seed: int, clean: bool, perturb: bool = True) -> dict[str, Any]:
    rng = random.Random(seed)
    np.random.seed(seed)  # 保证任何用到 numpy 随机的地方也可复现

    if clean:
        for d in (RAW_DIR, PERTURBED_DIR):
            if d.exists():
                shutil.rmtree(d)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PERTURBED_DIR.mkdir(parents=True, exist_ok=True)

    ground_truth: dict[str, dict[str, Any]] = {}
    perturbations = build_perturbations(rng)

    print(f"生成 {count} 张原图(随机种子 {seed})...")
    for i in range(count):
        template = UI_TEMPLATES[i % len(UI_TEMPLATES)]
        img, text = render_ui(template, rng)

        raw_name = f"ui_{i:03d}_{template['kind']}.png"
        raw_path = RAW_DIR / raw_name
        # imencode + tofile:规避 OpenCV 在中文路径下写文件失败的问题
        cv2.imencode(".png", img)[1].tofile(str(raw_path))

        ground_truth[raw_name] = {
            "text": text,
            "perturbation": "none",
            "source": "synthetic",
            "kind": template["kind"],
            "relpath": f"raw/{raw_name}",
        }

        if not perturb:
            continue

        # 每张原图随机挑 2 种扰动(排除 none),控制数据集规模不爆炸
        chosen = rng.sample(perturbations[1:], k=2)
        for pname, pfunc in chosen:
            perturbed = pfunc(img)
            p_name = f"ui_{i:03d}_{template['kind']}__{pname}.png"
            p_path = PERTURBED_DIR / p_name
            cv2.imencode(".png", perturbed)[1].tofile(str(p_path))

            ground_truth[p_name] = {
                "text": text,          # 扰动不改变文字内容,GT 与原图相同
                "perturbation": pname,
                "source": "synthetic",
                "kind": template["kind"],
                "relpath": f"perturbed/{p_name}",
            }

    GROUND_TRUTH.write_text(
        json.dumps(ground_truth, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 统计各扰动类型的样本数,便于确认分布合理
    dist: dict[str, int] = {}
    for meta in ground_truth.values():
        dist[meta["perturbation"]] = dist.get(meta["perturbation"], 0) + 1

    print(f"\n完成。共 {len(ground_truth)} 个样本")
    print(f"  原图:      {RAW_DIR}")
    print(f"  扰动样本:  {PERTURBED_DIR}")
    print(f"  标注文件:  {GROUND_TRUTH}")
    print("\n扰动分布:")
    for k, v in sorted(dist.items()):
        print(f"  {k:20} {v:4} 张")

    return ground_truth


def main() -> None:
    parser = argparse.ArgumentParser(description="构建 OCR 评测数据集")
    parser.add_argument("--count", type=int, default=50, help="生成的原图数量")
    parser.add_argument("--seed", type=int, default=42, help="随机种子,保证可复现")
    parser.add_argument("--clean", action="store_true", help="生成前清空旧数据")
    parser.add_argument("--no-perturb", action="store_true", help="只生成原图")
    args = parser.parse_args()

    build(
        count=args.count,
        seed=args.seed,
        clean=args.clean,
        perturb=not args.no_perturb,
    )


if __name__ == "__main__":
    main()
