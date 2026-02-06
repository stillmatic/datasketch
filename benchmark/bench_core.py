"""
Benchmarks for MinHash, MinHashLSH, and HyperLogLog.

Run:
    source .venv/bin/activate
    pytest benchmark/bench_core.py -v --benchmark-sort=fullname

Compare across runs:
    pytest benchmark/bench_core.py --benchmark-save=baseline
    # ... make changes ...
    pytest benchmark/bench_core.py --benchmark-compare=0001_baseline

Disable GC for more stable results:
    pytest benchmark/bench_core.py --benchmark-disable-gc
"""

import pickle

import numpy as np
import pytest

from datasketch import HyperLogLog, MinHash, MinHashLSH
from datasketch.hyperloglog import HyperLogLogPlusPlus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bytes(i):
    return f"item-{i}".encode()


def _make_bytes_list(n):
    return [f"item-{i}".encode() for i in range(n)]


def _make_minhash(num_perm, n_items):
    m = MinHash(num_perm=num_perm)
    for i in range(n_items):
        m.update(_make_bytes(i))
    return m


def _make_minhash_batch(num_perm, n_items):
    m = MinHash(num_perm=num_perm)
    m.update_batch(_make_bytes_list(n_items))
    return m


# ---------------------------------------------------------------------------
# MinHash benchmarks
# ---------------------------------------------------------------------------

class TestMinHashBench:
    """Benchmarks for MinHash operations."""

    # -- init --

    @pytest.mark.parametrize("num_perm", [64, 128, 256])
    def test_init(self, benchmark, num_perm):
        benchmark(MinHash, num_perm=num_perm)

    # -- update (single element) --

    @pytest.mark.parametrize("num_perm", [64, 128, 256])
    def test_update_single(self, benchmark, num_perm):
        m = MinHash(num_perm=num_perm)
        data = _make_bytes(42)
        benchmark(m.update, data)

    # -- update_batch --

    @pytest.mark.parametrize(
        "num_perm,n_items",
        [(128, 100), (128, 1_000), (128, 10_000), (256, 10_000)],
    )
    def test_update_batch(self, benchmark, num_perm, n_items):
        items = _make_bytes_list(n_items)
        m = MinHash(num_perm=num_perm)
        benchmark(m.update_batch, items)

    # -- jaccard --

    @pytest.mark.parametrize("num_perm", [64, 128, 256])
    def test_jaccard(self, benchmark, num_perm):
        m1 = _make_minhash(num_perm, 500)
        m2 = _make_minhash(num_perm, 500)
        benchmark(m1.jaccard, m2)

    # -- merge --

    @pytest.mark.parametrize("num_perm", [128, 256])
    def test_merge(self, benchmark, num_perm):
        m1 = _make_minhash(num_perm, 500)
        m2 = _make_minhash(num_perm, 500)
        benchmark(m1.merge, m2)

    # -- copy --

    def test_copy(self, benchmark):
        m = _make_minhash(128, 500)
        benchmark(m.copy)

    # -- pickle round-trip --

    def test_pickle_roundtrip(self, benchmark):
        m = _make_minhash(128, 500)
        data = pickle.dumps(m)
        benchmark(pickle.loads, data)

    # -- bulk creation --

    @pytest.mark.parametrize(
        "n_minhashes,set_size",
        [(100, 50), (1_000, 200), (5_000, 200)],
    )
    def test_bulk(self, benchmark, n_minhashes, set_size):
        rng = np.random.RandomState(42)
        data = [
            [_make_bytes(x) for x in rng.randint(0, n_minhashes * 10, size=set_size)]
            for _ in range(n_minhashes)
        ]
        benchmark(MinHash.bulk, data, num_perm=128)


# ---------------------------------------------------------------------------
# MinHashLSH benchmarks
# ---------------------------------------------------------------------------

class TestMinHashLSHBench:
    """Benchmarks for MinHashLSH operations."""

    # -- init (includes parameter optimization) --

    @pytest.mark.parametrize(
        "threshold,num_perm",
        [(0.5, 128), (0.8, 128), (0.9, 128), (0.5, 256)],
    )
    def test_init(self, benchmark, threshold, num_perm):
        benchmark(MinHashLSH, threshold=threshold, num_perm=num_perm)

    # -- init with pre-computed params (skips optimization) --

    def test_init_with_params(self, benchmark):
        benchmark(MinHashLSH, num_perm=128, params=(5, 25))

    # -- insert --

    @pytest.mark.parametrize("num_perm", [128, 256])
    def test_insert(self, benchmark, num_perm):
        lsh = MinHashLSH(threshold=0.5, num_perm=num_perm)
        m = _make_minhash(num_perm, 200)
        counter = [0]

        def do_insert():
            key = f"key-{counter[0]}"
            counter[0] += 1
            lsh.insert(key, m, check_duplication=True)

        benchmark(do_insert)

    # -- insert without duplication check --

    def test_insert_no_dup_check(self, benchmark):
        lsh = MinHashLSH(threshold=0.5, num_perm=128)
        m = _make_minhash(128, 200)
        counter = [0]

        def do_insert():
            key = f"key-{counter[0]}"
            counter[0] += 1
            lsh.insert(key, m, check_duplication=False)

        benchmark(do_insert)

    # -- query (small index) --

    def test_query_small(self, benchmark):
        num_perm = 128
        lsh = MinHashLSH(threshold=0.5, num_perm=num_perm)
        minhashes = []
        for i in range(100):
            m = MinHash(num_perm=num_perm)
            m.update_batch([_make_bytes(j) for j in range(i, i + 50)])
            minhashes.append(m)
            lsh.insert(f"key-{i}", m, check_duplication=False)
        query = minhashes[0]
        benchmark(lsh.query, query)

    # -- query (medium index) --

    def test_query_medium(self, benchmark):
        num_perm = 128
        lsh = MinHashLSH(threshold=0.5, num_perm=num_perm)
        rng = np.random.RandomState(42)
        minhashes = []
        for i in range(1_000):
            m = MinHash(num_perm=num_perm)
            items = [_make_bytes(x) for x in rng.randint(0, 5000, size=50)]
            m.update_batch(items)
            minhashes.append(m)
            lsh.insert(i, m, check_duplication=False)
        query = minhashes[0]
        benchmark(lsh.query, query)

    # -- query (large index) --

    def test_query_large(self, benchmark):
        num_perm = 128
        lsh = MinHashLSH(threshold=0.5, num_perm=num_perm)
        rng = np.random.RandomState(42)
        minhashes = []
        for i in range(10_000):
            m = MinHash(num_perm=num_perm)
            items = [_make_bytes(x) for x in rng.randint(0, 50000, size=50)]
            m.update_batch(items)
            minhashes.append(m)
            lsh.insert(i, m, check_duplication=False)
        query = minhashes[0]
        benchmark(lsh.query, query)

    # -- __contains__ --

    def test_contains(self, benchmark):
        num_perm = 128
        lsh = MinHashLSH(threshold=0.5, num_perm=num_perm)
        m = _make_minhash(num_perm, 200)
        for i in range(1_000):
            lsh.insert(i, m, check_duplication=False)
        benchmark(lambda: 500 in lsh)

    # -- remove --

    def test_remove(self, benchmark):
        num_perm = 128
        counter = [0]

        def setup_and_remove():
            # Build a fresh index each time with enough keys
            lsh = MinHashLSH(threshold=0.5, num_perm=num_perm)
            m = _make_minhash(num_perm, 200)
            for i in range(500):
                lsh.insert(i, m, check_duplication=False)
            return lsh, m

        lsh, m = setup_and_remove()
        keys = list(range(500))

        def do_remove():
            nonlocal lsh, keys
            if not keys:
                lsh, m = setup_and_remove()
                keys = list(range(500))
            lsh.remove(keys.pop())

        benchmark(do_remove)

    # -- end-to-end: insert N + query --

    @pytest.mark.parametrize(
        "n_items,set_size",
        [(1_000, 20), (1_000, 200), (1_000, 1_000), (5_000, 20), (5_000, 200)],
    )
    def test_end_to_end(self, benchmark, n_items, set_size):
        num_perm = 128
        # Pre-generate data outside the benchmark loop
        rng = np.random.RandomState(42)
        all_items = [
            [_make_bytes(x) for x in rng.randint(0, n_items * 10, size=set_size)]
            for _ in range(n_items)
        ]

        def workflow():
            lsh = MinHashLSH(threshold=0.5, num_perm=num_perm)
            minhashes = []
            for i, items in enumerate(all_items):
                m = MinHash(num_perm=num_perm)
                m.update_batch(items)
                minhashes.append(m)
                lsh.insert(i, m, check_duplication=False)
            # Query with first 10
            results = []
            for m in minhashes[:10]:
                results.append(lsh.query(m))
            return results

        benchmark(workflow)

    # -- end-to-end using MinHash.bulk: bulk create + insert + query --

    @pytest.mark.parametrize(
        "n_items,set_size",
        [(1_000, 20), (1_000, 200), (1_000, 1_000), (5_000, 20), (5_000, 200)],
    )
    def test_end_to_end_bulk(self, benchmark, n_items, set_size):
        num_perm = 128
        # Pre-generate data outside the benchmark loop
        rng = np.random.RandomState(42)
        all_items = [
            [_make_bytes(x) for x in rng.randint(0, n_items * 10, size=set_size)]
            for _ in range(n_items)
        ]

        def workflow():
            lsh = MinHashLSH(threshold=0.5, num_perm=num_perm)
            minhashes = MinHash.bulk(all_items, num_perm=num_perm)
            for i, m in enumerate(minhashes):
                lsh.insert(i, m, check_duplication=False)
            # Query with first 10
            results = []
            for m in minhashes[:10]:
                results.append(lsh.query(m))
            return results

        benchmark(workflow)


# ---------------------------------------------------------------------------
# HyperLogLog benchmarks
# ---------------------------------------------------------------------------

class TestHyperLogLogBench:
    """Benchmarks for HyperLogLog operations."""

    # -- init --

    @pytest.mark.parametrize("p", [8, 12, 16])
    def test_init(self, benchmark, p):
        benchmark(HyperLogLog, p=p)

    # -- update (single element) --

    @pytest.mark.parametrize("p", [8, 12, 16])
    def test_update_single(self, benchmark, p):
        h = HyperLogLog(p=p)
        data = _make_bytes(42)
        benchmark(h.update, data)

    # -- update N elements --

    @pytest.mark.parametrize(
        "p,n_items",
        [(8, 1_000), (8, 10_000), (12, 10_000), (16, 10_000)],
    )
    def test_update_n(self, benchmark, p, n_items):
        items = _make_bytes_list(n_items)

        def feed():
            h = HyperLogLog(p=p)
            for item in items:
                h.update(item)
            return h

        benchmark(feed)

    # -- update_batch --

    @pytest.mark.parametrize(
        "p,n_items",
        [(8, 1_000), (8, 10_000), (12, 10_000), (16, 10_000)],
    )
    def test_update_batch(self, benchmark, p, n_items):
        items = _make_bytes_list(n_items)

        def feed():
            h = HyperLogLog(p=p)
            h.update_batch(items)
            return h

        benchmark(feed)

    # -- count --

    @pytest.mark.parametrize("p", [8, 12, 16])
    def test_count(self, benchmark, p):
        h = HyperLogLog(p=p)
        for i in range(5000):
            h.update(_make_bytes(i))
        benchmark(h.count)

    # -- merge --

    @pytest.mark.parametrize("p", [8, 12, 16])
    def test_merge(self, benchmark, p):
        h1 = HyperLogLog(p=p)
        h2 = HyperLogLog(p=p)
        for i in range(5000):
            h1.update(_make_bytes(i))
            h2.update(_make_bytes(i + 2500))
        benchmark(h1.merge, h2)

    # -- serialize / deserialize --

    def test_serialize(self, benchmark):
        h = HyperLogLog(p=12)
        for i in range(5000):
            h.update(_make_bytes(i))
        buf = bytearray(h.bytesize())
        benchmark(h.serialize, buf)

    def test_deserialize(self, benchmark):
        h = HyperLogLog(p=12)
        for i in range(5000):
            h.update(_make_bytes(i))
        buf = bytearray(h.bytesize())
        h.serialize(buf)
        benchmark(HyperLogLog.deserialize, buf)

    # -- pickle round-trip --

    def test_pickle_roundtrip(self, benchmark):
        h = HyperLogLog(p=12)
        for i in range(5000):
            h.update(_make_bytes(i))
        data = pickle.dumps(h)
        benchmark(pickle.loads, data)

    # -- union --

    def test_union(self, benchmark):
        hlls = []
        for seed in range(10):
            h = HyperLogLog(p=12)
            for i in range(1000):
                h.update(_make_bytes(seed * 1000 + i))
            hlls.append(h)
        benchmark(HyperLogLog.union, *hlls)

    # -- copy --

    def test_copy(self, benchmark):
        h = HyperLogLog(p=12)
        for i in range(5000):
            h.update(_make_bytes(i))
        benchmark(h.copy)


# ---------------------------------------------------------------------------
# HyperLogLog++ benchmarks
# ---------------------------------------------------------------------------

class TestHyperLogLogPlusPlusBench:
    """Benchmarks for HyperLogLog++ (64-bit variant)."""

    @pytest.mark.parametrize("p", [8, 12, 16])
    def test_update_single(self, benchmark, p):
        h = HyperLogLogPlusPlus(p=p)
        data = _make_bytes(42)
        benchmark(h.update, data)

    @pytest.mark.parametrize("p", [8, 12, 16])
    def test_count(self, benchmark, p):
        h = HyperLogLogPlusPlus(p=p)
        for i in range(5000):
            h.update(_make_bytes(i))
        benchmark(h.count)

    def test_update_n(self, benchmark):
        items = _make_bytes_list(10_000)

        def feed():
            h = HyperLogLogPlusPlus(p=12)
            for item in items:
                h.update(item)
            return h

        benchmark(feed)

    def test_update_batch(self, benchmark):
        items = _make_bytes_list(10_000)

        def feed():
            h = HyperLogLogPlusPlus(p=12)
            h.update_batch(items)
            return h

        benchmark(feed)
