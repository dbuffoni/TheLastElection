import numpy as np
from tle.engines.protocol import LabelContext
from tle.models.similarity import SimilarityCache, distances, features


def ctx(rows):
    m = len(rows)
    z = lambda *s: np.zeros(s, dtype=int)  # noqa: E731
    T = np.array([r.get("T", [3, 3, 3, 3]) for r in rows])
    O = np.array([r.get("O", [3] * 6) for r in rows])
    return LabelContext(
        ids=np.arange(m), age_band=np.array([r.get("age", 2) for r in rows]), social_class=z(m) + 1,
        trade=np.array([r.get("trade", 0) for r in rows]), circle=z(m), T=T, C=z(m, 3) + 3, I=z(m, 3) + 3, O=O,
        fear=z(m) + 2, risk=z(m),
    )  # fmt: skip


def test_distance_zero_for_identical_and_grows_with_gaps():
    c = ctx([{}, {}, {"O": [0, 3, 3, 3, 3, 3]}, {"trade": 4}])
    fo, fc = features(c)
    d = distances(fo[0], fc[0], fo, fc)
    assert d[0] == 0 and d[1] == 0 and d[2] > 0 and d[3] > 0 and d[2] < 0.1


def test_cache_asks_only_representatives_and_is_deterministic():
    c = ctx([{}, {}, {"O": [1, 3, 3, 3, 3, 3]}, {"O": [6, 6, 6, 6, 6, 6]}])
    asked = []

    def ask(idx):
        asked.append(list(idx))
        return [100 + i for i in idx]

    out = SimilarityCache(0.0).resolve("k", c, ask)
    assert asked == [[0, 2, 3]] and out == [100, 100, 102, 103]  # the exact duplicate follows citizen 0
    asked.clear()
    cache = SimilarityCache(0.03)
    out = cache.resolve("k", c, ask)
    assert asked == [[0, 3]] and out == [100, 100, 100, 103]  # 2 is within the threshold of 0
    asked.clear()
    assert cache.resolve("k", c, ask) == out and asked == []  # same question later: everything from the cache
    assert cache.hits == 2 + 4 and cache.asked == 2
    asked.clear()
    cache.resolve("other question", c, ask)
    assert asked == [[0, 3]]  # another key does not share answers


def test_cache_off():
    c = ctx([{}, {}])
    cache = SimilarityCache(None)
    assert not cache.on
    assert cache.resolve("k", c, lambda idx: list(idx)) == [0, 1]
