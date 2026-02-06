# Performance Optimization Iterations

Tracking each optimization step with benchmark results.

**Environment:** Linux (WSL2), Python 3.12, NumPy, pytest-benchmark
**Benchmark file:** `benchmark/bench_core.py` (66 tests)
**Saved runs:** `.benchmarks/Linux-*/`

---

## Iteration 0: Baseline (`0001_baseline`)

**Commit:** `a1c1e29` (upstream master, v1.9.0)

No changes. Baseline measurements on stock datasketch.

Key timings (mean):
| Benchmark | Time |
|---|---|
| MinHash.update (128 perm) | ~3.6 us |
| MinHash.update_batch (128 perm, 10k) | ~17 ms |
| MinHash.init (128 perm) | ~234 us |
| LSH init (0.5, 128) | ~6.8 ms |
| LSH init (0.5, 256) | ~22.4 ms |
| LSH insert (128 perm) | ~22 us |
| LSH query (medium) | ~14.5 us |
| HLL.update (p=12) | ~640 ns |
| end_to_end (1000) | ~421 ms |
| end_to_end (5000) | ~2,205 ms |

---

## Iteration 1: Pure Python Optimizations (`0002_phase1_optimized`)

**Commit:** `fd3ea6c`

Changes:
1. **Vectorized `_init_permutations`** (`minhash.py`) — replaced per-element `randint` loop with bulk `gen.randint(..., size=num_perm)` calls
2. **Cached `_optimal_param`** (`lsh.py`) — added `@functools.lru_cache` to the grid-search integration function
3. **`tobytes()` instead of `byteswap`** (`lsh.py`) — replaced `bytes(hs.byteswap().data)` with `hs.data.tobytes()` for dict keys
4. **Inlined `_get_rank`** (`hyperloglog.py`) — eliminated method call overhead in hot `update()` path
5. **Added `HyperLogLog.update_batch`** (`hyperloglog.py`) — vectorized numpy batch update with `np.maximum.at`

Impact:
| Benchmark | Before | After | Speedup |
|---|---|---|---|
| LSH init (0.5, 128) | 6.8 ms | 42 us | **~160x** |
| LSH init (0.5, 256) | 22.4 ms | 140 us | **~160x** |
| MinHash.init (128) | 234 us | 77 us | **~3x** |
| LSH insert (128 perm) | 22 us | 15 us | **~1.5x** |
| end_to_end (1000) | 421 ms | ~145 ms | **~2.9x** |
| end_to_end (5000) | 2,205 ms | ~700 ms | **~3.1x** |

The huge LSH init speedup is from caching `_optimal_param` — it was running
O(num_perm^2) scipy integrations on every construction.

---

## Iteration 2: Rust Core via Maturin (`0003_iteration2_current`)

**Commit:** `fd3ea6c` + `a1d080e` (HLL rank fix)

Changes:
1. **Maturin/PyO3 build system** — switched `pyproject.toml` to maturin, added `Cargo.toml` and `src/lib.rs`
2. **xxHash replacing SHA1** — `xxhash32` and `xxhash64` via `xxhash-rust` crate, ~10x faster hashing
3. **Fused MinHash update in Rust** — hash + permute + min-reduce in one call, eliminates Python-C boundary and temp arrays
4. **Fused HyperLogLog update in Rust** — hash + bit extraction + register update, both 32-bit and 64-bit variants
5. **Auto-detection with fallback** — `_use_rs` flag set at init, pure Python remains the fallback when Rust extension unavailable
6. **HLL rank overflow fix** — changed Rust rank arithmetic to `i16` to prevent u8 underflow when `bits == 0`

Integration pattern:
```python
try:
    from datasketch._rs import minhash_update as _rs_minhash_update
    _HAS_RS = True
except ImportError:
    _HAS_RS = False
```

Impact (from memory benchmark, Python vs Rust on same data):
| Operation | Python | Rust | Speedup | Peak Mem Ratio |
|---|---|---|---|---|
| MinHash update_batch (10k, 128 perm) | ~10 ms | ~1.8 ms | **~5.5x** | 0.12x |
| MinHash update x1000 (128 perm) | ~10 ms | ~1.8 ms | **~5.5x** | 0.12x |
| HLL update x10000 (p=12) | ~23 ms | ~0.85 ms | **~27x** | 0.31x |
| HLL++ update_batch (100k, p=12) | ~94 ms | ~2.5 ms | **~38x** | 0.01x |

Core benchmark impact (vs baseline):
| Benchmark | Baseline | After Rust | Speedup |
|---|---|---|---|
| end_to_end (1000) | 421 ms | 145 ms | **~2.9x** |
| end_to_end (5000) | 2,205 ms | 655 ms | **~3.4x** |

---

## Iteration 3: Low-Hanging Fruit One-Liners (`0004_iteration3_oneliners`)

**Commit:** (this commit)

Changes:
1. **`set.update()` in LSH query** (`lsh.py:430`) — replaced `for key in ...: candidates.add(key)` with `candidates.update(hashtable.get(H))`
2. **`tobytes()` in LSHForest** (`lshforest.py:172`) — same pattern as LSH, replaced `bytes(hs.byteswap().data)` with `hs.data.tobytes()`
3. **Removed matching `byteswap()` on read** (`lshforest.py:145`) — `get_minhash_hashvalues` no longer needs to un-swap

Impact (vs baseline):
| Benchmark | Baseline | Now | Speedup |
|---|---|---|---|
| LSH query small | 17.4 us | 8.1 us | **2.1x** |
| LSH query medium | 14.5 us | 7.8 us | **1.9x** |
| LSH query large | 13.1 us | 7.9 us | **1.7x** |
| LSH insert (128) | 22.4 us | 15.1 us | **1.5x** |
| LSH insert (256) | 33.8 us | 23.2 us | **1.5x** |
| end_to_end (1000) | 421 ms | 149 ms | **2.8x** |
| end_to_end (5000) | 2,205 ms | 701 ms | **3.1x** |

---

## Cumulative Speedup Summary (Baseline to Iteration 3)

| Benchmark | Baseline | Current | Total Speedup |
|---|---|---|---|
| LSH init (0.5, 128) | 6,845 us | 42 us | **163x** |
| LSH init (0.5, 256) | 22,426 us | 140 us | **160x** |
| MinHash init (128 perm) | 234 us | 77 us | **3x** |
| LSH query (small) | 17.4 us | 8.1 us | **2.1x** |
| LSH query (medium) | 14.5 us | 7.8 us | **1.9x** |
| LSH insert (128 perm) | 22.4 us | 15.1 us | **1.5x** |
| HLL update (p=12) | 640 ns | ~50 ns (Rust) | **~13x** |
| end_to_end (1000 items) | 421 ms | 149 ms | **2.8x** |
| end_to_end (5000 items) | 2,205 ms | 701 ms | **3.1x** |

---

## What's Next

Remaining optimization opportunities (roughly ordered by impact/effort):

| Opportunity | Module | Est. Impact | Effort |
|---|---|---|---|
| SIMD for Rust permutation loops | `src/lib.rs` | 30-50% | High |
| Rayon parallelism for batch updates | `src/lib.rs` | 2-4x multi-core | High |
| Vectorize WeightedMinHash `minhash()` loop | `weighted_minhash.py` | 40-60% | Medium |
| Redis `mget` pipeline in storage | `storage.py` | 50-70% (Redis only) | Medium |
| NumPy `searchsorted` in LSHForest | `lshforest.py` | 20-30% | Low |
| LeanMinHash numpy-native serialization | `lean_minhash.py` | 30-40% serialize | Medium |
| Vectorize b-bit MinHash packing | `b_bit_minhash.py` | 20-30% | Medium |
