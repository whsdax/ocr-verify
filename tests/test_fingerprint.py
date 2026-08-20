"""图片指纹单元测试。"""

import numpy as np

from ocr_verify.cache.fingerprint import (
    dhash_fingerprint,
    hamming_distance,
    is_similar,
    md5_fingerprint,
)


def test_md5_same_image():
    img = np.full((60, 120, 3), 255, dtype=np.uint8)
    fp1 = md5_fingerprint(img)
    fp2 = md5_fingerprint(img)
    assert fp1 == fp2


def test_md5_different_images():
    img1 = np.full((60, 120, 3), 255, dtype=np.uint8)
    img2 = np.full((60, 120, 3), 200, dtype=np.uint8)
    assert md5_fingerprint(img1) != md5_fingerprint(img2)


def test_md5_includes_size():
    # 两张空图,不同尺寸,指纹应不同
    img1 = np.full((60, 120, 3), 0, dtype=np.uint8)
    img2 = np.full((61, 120, 3), 0, dtype=np.uint8)
    assert md5_fingerprint(img1) != md5_fingerprint(img2)


def test_dhash_consistent():
    img = np.full((80, 160, 3), 128, dtype=np.uint8)
    h1 = dhash_fingerprint(img)
    h2 = dhash_fingerprint(img)
    assert h1 == h2
    assert isinstance(h1, int)


def test_dhash_similar_for_minor_changes():
    # 单像素亮度变化应产生很小的汉明距离
    img1 = np.full((80, 160, 3), 128, dtype=np.uint8)
    img2 = np.full((80, 160, 3), 130, dtype=np.uint8)
    h1 = dhash_fingerprint(img1)
    h2 = dhash_fingerprint(img2)
    assert is_similar(h1, h2, threshold=3)


def test_hamming_distance_same():
    assert hamming_distance(0b1010, 0b1010) == 0
    assert hamming_distance(0b0000, 0b1111) == 4
