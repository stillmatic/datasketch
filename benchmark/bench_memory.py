"""
Memory usage benchmarks for MinHash, MinHashLSH, and HyperLogLog.

Also compares Python-only vs Rust-accelerated paths where applicable.

Run:
    uv run python benchmark/bench_memory.py
"""

import gc
import sys
import time
import tracemalloc

import numpy as np


def measure(fn):
    """Run fn(), return (result, current_bytes, peak_bytes, elapsed_seconds)."""
    gc.collect()
    gc.disable()
    tracemalloc.start()
    t0 = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - t0
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    gc.enable()
    return result, current, peak, elapsed


def fmt(nbytes):
    if nbytes < 1024:
        return f"{nbytes:,} B"
    if nbytes < 1024 * 1024:
        return f"{nbytes / 1024:,.1f} KB"
    return f"{nbytes / (1024 * 1024):,.2f} MB"


def fmt_time(s):
    if s < 0.001:
        return f"{s * 1e6:,.0f} us"
    if s < 1:
        return f"{s * 1e3:,.1f} ms"
    return f"{s:,.2f} s"


def make_bytes_list(n):
    return [f"item-{i}".encode() for i in range(n)]


def section(title):
    print(f"\n{'=' * 72}")
    print(f"  {title}")
    print(f"{'=' * 72}")


def row(label, current, peak, elapsed=None):
    time_str = f"  time={fmt_time(elapsed):>10s}" if elapsed is not None else ""
    print(f"  {label:<42s}  cur={fmt(current):>10s}  peak={fmt(peak):>10s}{time_str}")


# -----------------------------------------------------------------------
# MinHash
# -----------------------------------------------------------------------

def bench_minhash():
    from datasketch import MinHash

    section("MinHash Memory Usage")

    for num_perm in [64, 128, 256, 512]:
        _, cur, peak, t = measure(lambda np=num_perm: MinHash(num_perm=np))
        row(f"MinHash(num_perm={num_perm})", cur, peak, t)

    # Single object total size
    print()
    for num_perm in [128, 256]:
        m = MinHash(num_perm=num_perm)
        hv_bytes = m.hashvalues.nbytes
        perm_bytes = m.permutations[0].nbytes + m.permutations[1].nbytes
        total = hv_bytes + perm_bytes
        print(f"  MinHash(num_perm={num_perm}) array footprint:")
        print(f"    hashvalues:    {fmt(hv_bytes)}")
        print(f"    permutations:  {fmt(perm_bytes)}")
        print(f"    total arrays:  {fmt(total)}")

    # Batch creation memory
    print()
    items = make_bytes_list(100)
    for count in [100, 1_000, 10_000]:
        data = [items] * count
        _, cur, peak, t = measure(lambda d=data: MinHash.bulk(d, num_perm=128))
        row(f"MinHash.bulk({count:,} x 100 items)", cur, peak, t)

    # update_batch memory overhead
    print()
    for n_items in [1_000, 10_000, 100_000]:
        items = make_bytes_list(n_items)
        m = MinHash(num_perm=128)
        _, cur, peak, t = measure(lambda m=m, it=items: m.update_batch(it))
        row(f"update_batch({n_items:,}, 128 perm)", cur, peak, t)


# -----------------------------------------------------------------------
# MinHashLSH
# -----------------------------------------------------------------------

def bench_lsh():
    from datasketch import MinHash, MinHashLSH

    section("MinHashLSH Memory Usage")

    for n_items in [1_000, 10_000, 50_000]:
        def build_index(n=n_items):
            lsh = MinHashLSH(threshold=0.5, num_perm=128)
            rng = np.random.RandomState(42)
            for i in range(n):
                m = MinHash(num_perm=128)
                items = [f"item-{x}".encode() for x in rng.randint(0, n * 10, size=20)]
                m.update_batch(items)
                lsh.insert(i, m, check_duplication=False)
            return lsh

        _, cur, peak, t = measure(build_index)
        row(f"LSH index {n_items:,} items", cur, peak, t)


# -----------------------------------------------------------------------
# HyperLogLog
# -----------------------------------------------------------------------

def bench_hll():
    from datasketch import HyperLogLog
    from datasketch.hyperloglog import HyperLogLogPlusPlus

    section("HyperLogLog Memory Usage")

    for p in [8, 12, 16]:
        _, cur, peak, t = measure(lambda p=p: HyperLogLog(p=p))
        m = 1 << p
        row(f"HyperLogLog(p={p}, m={m:,})", cur, peak, t)

    # Object footprint
    print()
    for p in [8, 12, 16]:
        h = HyperLogLog(p=p)
        reg_bytes = h.reg.nbytes
        print(f"  HyperLogLog(p={p}) register: {fmt(reg_bytes)} ({1 << p} x int8)")

    # update_batch memory overhead
    print()
    for n_items in [10_000, 100_000, 1_000_000]:
        items = make_bytes_list(n_items)
        h = HyperLogLog(p=12)
        _, cur, peak, t = measure(lambda h=h, it=items: h.update_batch(it))
        row(f"HLL.update_batch({n_items:,}, p=12)", cur, peak, t)

    # HLL++
    print()
    for p in [8, 12, 16]:
        _, cur, peak, t = measure(lambda p=p: HyperLogLogPlusPlus(p=p))
        row(f"HyperLogLogPlusPlus(p={p})", cur, peak, t)


# -----------------------------------------------------------------------
# Pipeline: many sketches
# -----------------------------------------------------------------------

def bench_pipeline():
    from datasketch import HyperLogLog, MinHash

    section("Pipeline Pattern: Many Sketches")

    for count in [1_000, 10_000]:
        _, cur, peak, t = measure(
            lambda n=count: [MinHash(num_perm=128) for _ in range(n)]
        )
        row(f"{count:,} x MinHash(128 perm)", cur, peak, t)

    for count in [1_000, 10_000]:
        _, cur, peak, t = measure(
            lambda n=count: [HyperLogLog(p=12) for _ in range(n)]
        )
        row(f"{count:,} x HyperLogLog(p=12)", cur, peak, t)


# -----------------------------------------------------------------------
# Python vs Rust comparison
# -----------------------------------------------------------------------

def bench_python_vs_rust():
    from datasketch.hashfunc import sha1_hash32, sha1_hash64

    section("Python vs Rust: Side-by-Side Comparison")

    has_rs = False
    try:
        from datasketch._rs import (
            hll64_update_batch,
            hll_update_batch,
            minhash_update,
            minhash_update_batch,
        )
        has_rs = True
    except ImportError:
        print("  Rust extension not available — skipping comparison")
        return

    from datasketch import HyperLogLog, MinHash
    from datasketch.hyperloglog import HyperLogLogPlusPlus

    # -- MinHash update_batch --
    print("\n  --- MinHash update_batch (128 perm, 10k items) ---")
    items = make_bytes_list(10_000)
    num_perm = 128

    # Python path
    m_py = MinHash(num_perm=num_perm)
    m_py._use_rs = False  # force Python path
    _, cur_py, peak_py, t_py = measure(lambda: m_py.update_batch(items))

    # Rust path
    m_rs = MinHash(num_perm=num_perm)
    m_rs._use_rs = True
    _, cur_rs, peak_rs, t_rs = measure(lambda: m_rs.update_batch(items))

    row("Python path", cur_py, peak_py, t_py)
    row("Rust path", cur_rs, peak_rs, t_rs)
    if t_py > 0:
        print(f"  Speedup: {t_py / t_rs:.1f}x    Peak mem ratio: {peak_rs / max(peak_py, 1):.2f}x")

    # -- MinHash update single (loop of 1000) --
    print("\n  --- MinHash update x1000 (128 perm) ---")

    m_py = MinHash(num_perm=num_perm)
    m_py._use_rs = False
    loop_items = make_bytes_list(1000)
    def py_update_loop():
        for item in loop_items:
            m_py.update(item)
    _, cur_py, peak_py, t_py = measure(py_update_loop)

    m_rs = MinHash(num_perm=num_perm)
    m_rs._use_rs = True
    def rs_update_loop():
        for item in loop_items:
            m_rs.update(item)
    _, cur_rs, peak_rs, t_rs = measure(rs_update_loop)

    row("Python path", cur_py, peak_py, t_py)
    row("Rust path", cur_rs, peak_rs, t_rs)
    if t_py > 0:
        print(f"  Speedup: {t_py / t_rs:.1f}x    Peak mem ratio: {peak_rs / max(peak_py, 1):.2f}x")

    # -- HLL update_batch --
    print("\n  --- HLL update_batch (p=12, 100k items) ---")
    hll_items = make_bytes_list(100_000)

    h_py = HyperLogLog(p=12)
    h_py._use_rs = False
    _, cur_py, peak_py, t_py = measure(lambda: h_py.update_batch(hll_items))

    h_rs = HyperLogLog(p=12)
    h_rs._use_rs = True
    _, cur_rs, peak_rs, t_rs = measure(lambda: h_rs.update_batch(hll_items))

    row("Python path", cur_py, peak_py, t_py)
    row("Rust path", cur_rs, peak_rs, t_rs)
    if t_py > 0:
        print(f"  Speedup: {t_py / t_rs:.1f}x    Peak mem ratio: {peak_rs / max(peak_py, 1):.2f}x")

    # -- HLL update loop --
    print("\n  --- HLL update x10000 (p=12) ---")
    hll_loop = make_bytes_list(10_000)

    h_py = HyperLogLog(p=12)
    h_py._use_rs = False
    def py_hll_loop():
        for item in hll_loop:
            h_py.update(item)
    _, cur_py, peak_py, t_py = measure(py_hll_loop)

    h_rs = HyperLogLog(p=12)
    h_rs._use_rs = True
    def rs_hll_loop():
        for item in hll_loop:
            h_rs.update(item)
    _, cur_rs, peak_rs, t_rs = measure(rs_hll_loop)

    row("Python path", cur_py, peak_py, t_py)
    row("Rust path", cur_rs, peak_rs, t_rs)
    if t_py > 0:
        print(f"  Speedup: {t_py / t_rs:.1f}x    Peak mem ratio: {peak_rs / max(peak_py, 1):.2f}x")

    # -- HLL++ update_batch --
    print("\n  --- HLL++ update_batch (p=12, 100k items) ---")

    hpp_py = HyperLogLogPlusPlus(p=12)
    hpp_py._use_rs = False
    _, cur_py, peak_py, t_py = measure(lambda: hpp_py.update_batch(hll_items))

    hpp_rs = HyperLogLogPlusPlus(p=12)
    hpp_rs._use_rs = True
    _, cur_rs, peak_rs, t_rs = measure(lambda: hpp_rs.update_batch(hll_items))

    row("Python path", cur_py, peak_py, t_py)
    row("Rust path", cur_rs, peak_rs, t_rs)
    if t_py > 0:
        print(f"  Speedup: {t_py / t_rs:.1f}x    Peak mem ratio: {peak_rs / max(peak_py, 1):.2f}x")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

if __name__ == "__main__":
    print("Memory Usage Benchmarks")
    print(f"Python {sys.version}")
    print(f"NumPy {np.__version__}")

    try:
        from datasketch._rs import xxhash32
        print("Rust extension: AVAILABLE")
    except ImportError:
        print("Rust extension: NOT AVAILABLE (pure Python)")

    bench_minhash()
    bench_hll()
    bench_lsh()
    bench_pipeline()
    bench_python_vs_rust()

    print(f"\n{'=' * 72}")
    print("  Done.")
    print(f"{'=' * 72}")
