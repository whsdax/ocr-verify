"""LRU 缓存单元测试。"""

import threading

import pytest

from ocr_verify.cache.lru import LRUCache


def test_basic_get_put():
    c = LRUCache[str, int](capacity=3)
    c.put("a", 1)
    assert c.get("a") == 1
    assert c.get("b") is None


def test_eviction_order():
    c = LRUCache[str, int](capacity=3)
    c.put("a", 1)
    c.put("b", 2)
    c.put("c", 3)
    c.get("a")  # 把 a 提升为最近使用
    c.put("d", 4)  # 应淘汰最久未使用的 b
    assert c.get("b") is None
    assert set(c.keys_mru_to_lru()) == {"a", "c", "d"}
    # 新插入的 d 是 MRU;a 刚被访问过,是次新的;c 未被访问,是 LRU
    assert c.keys_mru_to_lru() == ["d", "a", "c"]


def test_update_moves_to_front():
    c = LRUCache[str, int](capacity=2)
    c.put("a", 1)
    c.put("b", 2)
    c.put("a", 10)  # 更新 a,应提升
    c.put("c", 3)
    assert c.get("a") == 10
    assert c.get("b") is None


def test_stats():
    c = LRUCache[str, int](capacity=2)
    c.get("x")  # miss
    c.put("y", 1)
    c.get("y")  # hit
    s = c.stats()
    assert s.hits == 1
    assert s.misses == 1
    assert s.hit_rate == 0.5


def test_on_evict_callback():
    evicted = []

    def cb(k, v):
        evicted.append((k, v))

    c = LRUCache[str, int](capacity=2, on_evict=cb)
    c.put("a", 1)
    c.put("b", 2)
    c.put("c", 3)
    assert ("a", 1) in evicted


def test_thread_safety():
    """并发 put/get 不应抛异常或死锁。"""
    c = LRUCache[int, int](capacity=50)
    errors = []

    def worker(start):
        try:
            for i in range(start, start + 200):
                c.put(i, i)
                c.get(i)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i * 200,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert len(c) <= 50


def test_get_or_compute():
    c = LRUCache[str, int](capacity=2)
    v1, cached1 = c.get_or_compute("a", lambda: 42)
    v2, cached2 = c.get_or_compute("a", lambda: 99)
    assert v1 == v2 == 42
    assert cached1 is False
    assert cached2 is True


def test_invalid_capacity():
    with pytest.raises(ValueError):
        LRUCache[str, int](capacity=0)
    with pytest.raises(ValueError):
        LRUCache[str, int](capacity=-1)
