"""线程安全的 LRU 缓存 —— 哈希表 + 双向链表手写实现。

设计要点
--------
1. **为什么是 哈希表 + 双向链表**
   LRU 需要两个 O(1) 操作:
     - 按 key 随机访问   -> 哈希表 (dict: key -> Node)
     - 任意节点摘除/头插 -> 双向链表 (Node 持有 prev/next)
   单用哈希表无法维护访问顺序;单用链表查找是 O(n)。两者结合才能都做到 O(1)。

2. **为什么用哨兵节点 (sentinel)**
   head/tail 两个空节点常驻,使得插入、删除无需判断"是否为首/尾节点",
   消除了全部边界分支,代码更短且不易出错。

3. **为什么不用 collections.OrderedDict / functools.lru_cache**
   标准库当然能用,但本项目刻意手写,原因有二:
     - 需要嵌入自定义统计(命中率、淘汰数、节省的模型调用次数)
     - 需要支持"淘汰回调"以便统计被丢弃的高成本条目
   同时这也是数据结构功底的直接体现。

4. **链表方向约定**
   head 侧 = 最近使用 (MRU),tail 侧 = 最久未使用 (LRU)。
   淘汰时永远从 tail.prev 摘除。

复杂度
------
get / put / 淘汰 均为 O(1) 平均时间;空间 O(capacity)。
"""

from __future__ import annotations

import threading
from typing import Any, Callable, Generic, Iterator, Optional, TypeVar

K = TypeVar("K")
V = TypeVar("V")

_MISSING = object()


class _Node(Generic[K, V]):
    """双向链表节点。

    使用 __slots__ 避免每个节点创建 __dict__,在缓存条目较多时显著降低内存占用。
    """

    __slots__ = ("key", "value", "prev", "next")

    def __init__(self, key: Optional[K] = None, value: Optional[V] = None) -> None:
        self.key = key
        self.value = value
        self.prev: Optional[_Node[K, V]] = None
        self.next: Optional[_Node[K, V]] = None

    def __repr__(self) -> str:  # pragma: no cover - 仅调试用
        return f"_Node(key={self.key!r})"


class LRUCache(Generic[K, V]):
    """容量受限的 LRU 缓存,线程安全。

    Parameters
    ----------
    capacity:
        最大条目数,必须为正整数。
    on_evict:
        可选的淘汰回调 ``fn(key, value)``,在条目因容量不足被淘汰时触发。
        用于统计"被丢弃的昂贵结果",帮助判断容量是否设置过小。

    Examples
    --------
    >>> cache = LRUCache[str, int](capacity=2)
    >>> cache.put("a", 1)
    >>> cache.put("b", 2)
    >>> cache.get("a")            # 访问 a,使其成为最近使用
    1
    >>> cache.put("c", 3)         # 容量满,淘汰最久未使用的 b
    >>> cache.get("b") is None
    True
    >>> cache.stats().hits
    1
    """

    def __init__(
        self,
        capacity: int = 256,
        on_evict: Optional[Callable[[K, V], None]] = None,
    ) -> None:
        if not isinstance(capacity, int) or capacity <= 0:
            raise ValueError(f"capacity 必须为正整数,收到: {capacity!r}")

        self._capacity = capacity
        self._on_evict = on_evict

        # 哈希表:key -> 链表节点。提供 O(1) 定位。
        self._map: dict[K, _Node[K, V]] = {}

        # 双向链表哨兵。head <-> tail 初始互指,表示空链表。
        # 约定:head.next 是最近使用,tail.prev 是最久未使用。
        self._head: _Node[K, V] = _Node()
        self._tail: _Node[K, V] = _Node()
        self._head.next = self._tail
        self._tail.prev = self._head

        # 可重入锁:允许同一线程在回调中再次调用本缓存的方法而不死锁。
        self._lock = threading.RLock()

        # 统计计数
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    # ------------------------------------------------------------------
    # 链表基本操作(均在持锁状态下调用,自身不加锁)
    # ------------------------------------------------------------------

    def _unlink(self, node: _Node[K, V]) -> None:
        """将节点从链表中摘除。O(1)。"""
        prev_node, next_node = node.prev, node.next
        # 有哨兵保证 prev/next 一定非空,无需判空
        prev_node.next = next_node  # type: ignore[union-attr]
        next_node.prev = prev_node  # type: ignore[union-attr]
        node.prev = node.next = None

    def _push_front(self, node: _Node[K, V]) -> None:
        """将节点插入到链表头部(标记为最近使用)。O(1)。"""
        first = self._head.next
        node.prev = self._head
        node.next = first
        self._head.next = node
        first.prev = node  # type: ignore[union-attr]

    def _move_to_front(self, node: _Node[K, V]) -> None:
        """已存在节点提升为最近使用 = 摘除 + 头插。O(1)。"""
        if self._head.next is node:
            return  # 已在头部,省去两次指针改写
        self._unlink(node)
        self._push_front(node)

    def _evict_lru(self) -> None:
        """淘汰最久未使用的条目(tail.prev)。O(1)。"""
        victim = self._tail.prev
        if victim is self._head:
            return  # 空链表,理论上不会走到
        self._unlink(victim)
        del self._map[victim.key]  # type: ignore[arg-type]
        self._evictions += 1
        if self._on_evict is not None:
            self._on_evict(victim.key, victim.value)  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------

    def get(self, key: K, default: Optional[V] = None) -> Optional[V]:
        """读取并将该 key 提升为最近使用。未命中返回 default。"""
        with self._lock:
            node = self._map.get(key)
            if node is None:
                self._misses += 1
                return default
            self._hits += 1
            self._move_to_front(node)
            return node.value

    def put(self, key: K, value: V) -> None:
        """写入或更新。超出容量时淘汰最久未使用的条目。"""
        with self._lock:
            node = self._map.get(key)
            if node is not None:
                # 已存在:更新值并提升
                node.value = value
                self._move_to_front(node)
                return

            # 新键:先插入再检查容量,逻辑比"先腾位再插"更直观
            node = _Node(key, value)
            self._map[key] = node
            self._push_front(node)

            if len(self._map) > self._capacity:
                self._evict_lru()

    def get_or_compute(self, key: K, factory: Callable[[], V]) -> tuple[V, bool]:
        """读取;未命中则调用 factory 计算并写入。

        Returns
        -------
        (value, from_cache)
            from_cache 为 True 表示本次命中缓存,未执行 factory。

        Note
        ----
        factory 在**释放锁之后**执行,避免昂贵的模型调用长时间独占锁而阻塞其他线程。
        代价是并发场景下同一 key 可能被重复计算一次(缓存击穿),
        对本项目而言这是可接受的权衡:重复计算只是浪费一次调用,不影响正确性。
        若需严格单飞(single-flight),应引入 per-key 锁。
        """
        cached = self.get(key, default=_MISSING)  # type: ignore[arg-type]
        if cached is not _MISSING:
            return cached, True  # type: ignore[return-value]

        value = factory()
        self.put(key, value)
        return value, False

    def peek(self, key: K) -> Optional[V]:
        """查看值但**不**改变访问顺序,也不计入命中统计。用于测试与调试。"""
        with self._lock:
            node = self._map.get(key)
            return node.value if node is not None else None

    def pop(self, key: K) -> Optional[V]:
        """删除并返回指定 key 的值。"""
        with self._lock:
            node = self._map.pop(key, None)
            if node is None:
                return None
            self._unlink(node)
            return node.value

    def clear(self) -> None:
        """清空所有条目,但保留统计计数。"""
        with self._lock:
            self._map.clear()
            self._head.next = self._tail
            self._tail.prev = self._head

    def reset_stats(self) -> None:
        """重置统计计数,不影响已缓存的数据。"""
        with self._lock:
            self._hits = self._misses = self._evictions = 0

    # ------------------------------------------------------------------
    # 内省
    # ------------------------------------------------------------------

    def keys_mru_to_lru(self) -> list[K]:
        """按最近使用 -> 最久未使用的顺序返回全部 key。主要用于测试断言。"""
        with self._lock:
            result: list[K] = []
            cur = self._head.next
            while cur is not self._tail:
                result.append(cur.key)  # type: ignore[arg-type]
                cur = cur.next  # type: ignore[assignment]
            return result

    def stats(self) -> "CacheStats":
        with self._lock:
            return CacheStats(
                hits=self._hits,
                misses=self._misses,
                evictions=self._evictions,
                size=len(self._map),
                capacity=self._capacity,
            )

    @property
    def capacity(self) -> int:
        return self._capacity

    def __len__(self) -> int:
        with self._lock:
            return len(self._map)

    def __contains__(self, key: object) -> bool:
        with self._lock:
            return key in self._map

    def __iter__(self) -> Iterator[K]:
        return iter(self.keys_mru_to_lru())

    def __repr__(self) -> str:
        s = self.stats()
        return (
            f"LRUCache(size={s.size}/{s.capacity}, "
            f"hits={s.hits}, misses={s.misses}, hit_rate={s.hit_rate:.1%})"
        )


class CacheStats:
    """缓存统计快照(不可变)。"""

    __slots__ = ("hits", "misses", "evictions", "size", "capacity")

    def __init__(
        self, hits: int, misses: int, evictions: int, size: int, capacity: int
    ) -> None:
        self.hits = hits
        self.misses = misses
        self.evictions = evictions
        self.size = size
        self.capacity = capacity

    @property
    def total(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        return self.hits / self.total if self.total else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "size": self.size,
            "capacity": self.capacity,
            "total_lookups": self.total,
            "hit_rate": round(self.hit_rate, 4),
        }

    def __repr__(self) -> str:
        return (
            f"CacheStats(hits={self.hits}, misses={self.misses}, "
            f"hit_rate={self.hit_rate:.1%}, size={self.size}/{self.capacity})"
        )
