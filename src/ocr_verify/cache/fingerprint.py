"""图片指纹 —— MD5 精确指纹 + dHash 感知指纹。

为什么需要两种指纹
------------------
**MD5(精确)**:字节级完全一致才命中。
  优点:零误判,计算极快(50KB 图片约 0.05ms)。
  局限:UI 截图中极常见的场景 —— 页面布局完全一样,只有状态栏时间、
        电量图标、光标位置发生变化 —— MD5 会完全失效。

**dHash(感知)**:对图像内容做低频降采样,视觉相近即命中。
  优点:能吸收上述像素级微小扰动。
  局限:存在误判可能,因此仅在"近似命中"时使用,且阈值需实验校准。

为什么选 MD5 而不是 SHA-256
---------------------------
这里只需要"唯一标识",不需要抗恶意碰撞的密码学强度。
MD5 输出 128 位,在本项目的数据量级(单次运行万级图片)下,
生日碰撞概率约为 10^-30 量级,完全可忽略;而速度约为 SHA-256 的 2 倍。
此外我们额外拼接了图片字节长度做二次校验,进一步降低碰撞风险。

为什么 dHash 优于 aHash
-----------------------
aHash(均值哈希)比较每个像素与全图均值,对整体亮度变化敏感。
dHash(差值哈希)比较**相邻像素的相对大小**,只关心梯度方向,
因此对亮度、对比度的整体变化天然免疫 —— 这正是 UI 截图在不同
主题/亮度设置下的主要差异来源。
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Union

import cv2
import numpy as np

ImageInput = Union[str, Path, bytes, np.ndarray]


# ----------------------------------------------------------------------
# 图片载入:统一各种输入形态
# ----------------------------------------------------------------------

def load_image(source: ImageInput) -> np.ndarray:
    """将路径 / 字节流 / ndarray 统一载入为 BGR 格式的 ndarray。

    使用 cv2.imdecode 而非 cv2.imread,因为后者在 Windows 上
    无法处理包含中文的路径(OpenCV 内部用 ANSI 编码打开文件)。
    """
    if isinstance(source, np.ndarray):
        return source

    if isinstance(source, bytes):
        buf = np.frombuffer(source, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("无法解码图片字节流,可能不是合法的图片格式")
        return img

    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"图片不存在: {path}")

    # 中文路径安全的读法:先用 Python 读字节,再交给 OpenCV 解码
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"无法解码图片: {path}")
    return img


def to_bytes(source: ImageInput, ext: str = ".png") -> bytes:
    """将任意输入转为原始字节。ndarray 会被编码为 PNG(无损)。"""
    if isinstance(source, bytes):
        return source
    if isinstance(source, np.ndarray):
        ok, buf = cv2.imencode(ext, source)
        if not ok:
            raise ValueError("图片编码失败")
        return buf.tobytes()
    return Path(source).read_bytes()


# ----------------------------------------------------------------------
# 精确指纹
# ----------------------------------------------------------------------

def md5_fingerprint(source: ImageInput) -> str:
    """计算图片的 MD5 指纹。

    返回格式: ``"{md5_hex}-{byte_length}"``
    附带字节长度作为二次校验 —— 两张图既要 MD5 相同又要长度相同才判为同一张,
    实际上把碰撞空间又收窄了一个维度。

    注意:ndarray 输入会先编码为 PNG。由于 PNG 编码是确定性的,
    同一 ndarray 每次得到的指纹一致。但**同一张图**经过
    "读入 JPEG -> 解码 -> 重编码 PNG" 后,指纹与原 JPEG 文件不同,
    这是预期行为(内容相同但字节不同)。
    """
    data = to_bytes(source)
    digest = hashlib.md5(data, usedforsecurity=False).hexdigest()
    return f"{digest}-{len(data)}"


# ----------------------------------------------------------------------
# 感知指纹
# ----------------------------------------------------------------------

def dhash_fingerprint(source: ImageInput, hash_size: int = 8) -> int:
    """计算差值哈希(dHash),返回一个整数形式的位串。

    算法步骤:
      1. 转灰度 —— 丢弃颜色信息,只保留结构
      2. 缩放到 (hash_size+1) x hash_size —— 宽度多 1 列用于做水平差分
      3. 逐行比较相邻像素: left > right ? 1 : 0
      4. 拼成 hash_size^2 位的整数(默认 64 位)

    Parameters
    ----------
    hash_size:
        默认 8,产生 64 位哈希。增大可提升区分度,但对微小变化更敏感。
    """
    img = load_image(source)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # INTER_AREA 在缩小图像时抗锯齿效果最好,能更好保留低频结构
    resized = cv2.resize(
        gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA
    )

    # 向量化的水平差分:比逐像素 Python 循环快约 50 倍
    diff = resized[:, 1:] > resized[:, :-1]

    # 将布尔矩阵打包成整数。flatten 后按位左移累加。
    bits = diff.flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def hamming_distance(hash_a: int, hash_b: int) -> int:
    """两个哈希的汉明距离 = 不同位的个数。

    ``a ^ b`` 让不同的位变成 1,再数 1 的个数即可。
    Python 3.10+ 的 int.bit_count() 是 CPython 内建的 popcount,速度最优。
    """
    return (hash_a ^ hash_b).bit_count()


def is_similar(hash_a: int, hash_b: int, threshold: int = 3) -> bool:
    """判断两张图是否视觉近似。

    threshold 的含义:64 位哈希中允许多少位不同。
      0     完全一致
      1-3   肉眼几乎无差别(推荐,可吸收时间戳、光标等局部变化)
      4-8   较相似(可能已经是不同页面,风险偏高)
      >10   基本不相关

    默认 3 是经验起点,**应当在自己的数据集上跑校准实验确定**,
    benchmark/calibrate_dhash.py 提供了该实验。
    """
    return hamming_distance(hash_a, hash_b) <= threshold


# ----------------------------------------------------------------------
# 组合指纹:先精确后感知
# ----------------------------------------------------------------------

class ImageFingerprint:
    """图片的双重指纹,作为缓存 key 使用。

    比较策略(两级):
      1. MD5 完全一致  -> 判定为同一张图(零误判)
      2. dHash 汉明距离 <= threshold -> 判定为视觉近似

    ``__hash__`` 只用 MD5,因此可以直接作为 dict 的 key 用于精确查找;
    近似查找需要走 :meth:`similar_to` 的线性扫描(见 fingerprint_index)。
    """

    __slots__ = ("md5", "dhash", "_threshold")

    def __init__(self, md5: str, dhash: int, threshold: int = 3) -> None:
        self.md5 = md5
        self.dhash = dhash
        self._threshold = threshold

    @classmethod
    def from_image(
        cls, source: ImageInput, threshold: int = 3, hash_size: int = 8
    ) -> "ImageFingerprint":
        # 只读取一次字节,避免路径输入时重复 IO
        data = to_bytes(source)
        return cls(
            md5=md5_fingerprint(data),
            dhash=dhash_fingerprint(data, hash_size=hash_size),
            threshold=threshold,
        )

    def similar_to(self, other: "ImageFingerprint") -> bool:
        """是否视觉近似(不要求字节一致)。"""
        if self.md5 == other.md5:
            return True
        return is_similar(self.dhash, other.dhash, self._threshold)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ImageFingerprint):
            return NotImplemented
        return self.md5 == other.md5

    def __hash__(self) -> int:
        return hash(self.md5)

    def __repr__(self) -> str:
        return f"ImageFingerprint(md5={self.md5[:8]}..., dhash={self.dhash:016x})"
