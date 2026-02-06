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

## Iteration 4: Fast Mersenne Reduction + Rayon Parallelism (`0006_iteration4_final`)

**Commit:** (this commit)

Changes:
1. **Fast Mersenne prime reduction** (`src/lib.rs`) — replaced `% (MERSENNE_PRIME as u128)` (software division, ~50-100ns via `__udivti3`) with bit manipulation: `(x >> 61) + (x & (2^61 - 1))` with one conditional subtract (~1-2ns)
2. **Unchecked indexing in permutation kernel** — `get_unchecked`/`get_unchecked_mut` eliminates bounds checks, helps LLVM auto-vectorize
3. **Rayon parallelism for MinHash batch** — thread-local accumulators merged by element-wise min, GIL released via `py.allow_threads()`
4. **Split sequential/parallel paths** — small batches (<2048 items) operate directly on numpy arrays with zero copies; large batches copy to owned Vecs and use Rayon
5. **HLL stays sequential** — per-item HLL work (~20ns) is too light to amortize register array cloning (up to 64KB for p=16); Rayon was counterproductive here

Key design decisions:
- MinHash Rayon threshold = 2048 items (below this, fast Mersenne alone is sufficient)
- Chunk size = 512 items (balances per-chunk work vs merge overhead)
- HLL: no Rayon, but benefits from unchecked indexing and consistent i16 rank arithmetic

Impact (vs iteration 3):
| Benchmark | Before | After | Speedup |
|---|---|---|---|
| MinHash update_batch (128, 100) | 29.9 us | 22.9 us | **23%** |
| MinHash update_batch (128, 1k) | 284 us | 226 us | **20%** |
| MinHash update_batch (128, 10k) | 2,863 us | 2,515 us | **12%** |
| MinHash update_batch (256, 10k) | 4,788 us | 2,561 us | **47%** |
| HLL update_batch (p=8, 10k) | 1,072 us | 1,098 us | same |
| HLL update_batch (p=16, 10k) | 1,068 us | 1,110 us | same |
| end_to_end (1000) | 145 ms | 142 ms | same |
| end_to_end (5000) | 655 ms | 707 ms | same |

The 47% speedup on `[256-10000]` is the Rayon parallel reduction. The 20-23%
speedup on smaller batches is purely from fast Mersenne elimination of the
`__udivti3` software division.

---

## Iteration 5: Permutation Cache + Rayon Bulk API (`0007_iteration5_bulk`)

**Commit:** (this commit)

Changes:
1. **Cached `_init_permutations`** (`minhash.py`) — `@functools.lru_cache` on `(num_perm, seed)`. Since all MinHashes with the same seed share identical permutations, this avoids recreating `np.random.RandomState` (~77 us) on every init
2. **`minhash_bulk_from_hashes` Rust function** (`src/lib.rs`) — takes pre-hashed u32 values + offsets array, applies permutations in parallel with Rayon across N independent MinHashes
3. **Updated `MinHash.bulk` to use Rust fast path** (`minhash.py`) — pre-hashes all items with xxhash32, passes hash+offset arrays to Rust, builds MinHash objects from flat result
4. **Fixed Rust path type guards** (`minhash.py`) — `update()`, `update_batch()`, and `bulk()` now check for `bytes`/`bytearray` before using Rust path, falling back to Python for other types (numpy arrays, etc.)
5. **Improved end-to-end benchmark** (`bench_core.py`) — moved data generation outside timed section, added `set_size` parameter axis, added `test_end_to_end_bulk` variant

Time breakdown analysis (5000 items × 200 per set, profiled):
| Component | Before (iter 4) | After |
|---|---|---|
| MinHash init (5000x) | 458 ms (39%) | 12 ms (2%) |
| update_batch (5000x) | 344 ms (63%) | — (folded into bulk) |
| MinHash.bulk (5000×200) | — | 184 ms total |
| LSH insert (5000x) | 193 ms | ~38 ms |

Impact — end-to-end (data generation excluded from timing):

| Benchmark | Sequential | Bulk | Speedup |
|---|---|---|---|
| end_to_end (1000, 20) | 22.8 ms | 20.4 ms | 1.1x |
| end_to_end (1000, 200) | 75.4 ms | 36.8 ms | **2.0x** |
| end_to_end (1000, 1000) | 314.9 ms | 109.1 ms | **2.9x** |
| end_to_end (5000, 20) | 131.8 ms | 110.1 ms | 1.2x |
| end_to_end (5000, 200) | 417.7 ms | 184.3 ms | **2.3x** |

The permutation cache alone dropped MinHash init from ~92 us to ~1.5 us
(~60x). The Rayon bulk API then parallelizes the permutation step across
all N MinHashes — this is most impactful at larger set sizes where per-
MinHash work is significant.

---

## Cumulative Speedup Summary (Baseline to Iteration 5)

| Benchmark | Baseline | Current | Total Speedup |
|---|---|---|---|
| LSH init (0.5, 128) | 6,845 us | 8 us | **~850x** |
| LSH init (0.5, 256) | 22,426 us | 13 us | **~1700x** |
| MinHash init (128 perm) | 234 us | 1.5 us | **~156x** |
| MinHash update_batch (128, 1k) | 465 us | 225 us | **2.1x** |
| MinHash update_batch (256, 10k) | 8,520 us | 2,372 us | **3.6x** |
| MinHash.bulk (1000, 200) | — | 28 ms | new API |
| MinHash.bulk (5000, 200) | — | 146 ms | new API |
| LSH query (small) | 17.4 us | 8.1 us | **2.1x** |
| LSH query (medium) | 14.5 us | 7.4 us | **2.0x** |
| LSH insert (128 perm) | 22.4 us | 11.0 us | **2.0x** |
| HLL update (p=12) | 640 ns | ~91 ns (Rust) | **~7x** |
| end_to_end seq (1000, 200) | — | 75 ms | new config |
| end_to_end bulk (1000, 200) | — | 37 ms | new config |
| end_to_end bulk (5000, 200) | — | 184 ms | new config |

Note: end_to_end timings now exclude data generation (was included in baseline).
The original baseline end_to_end (1000, 20) was 421 ms including data gen;
the equivalent sequential benchmark now measures 23 ms without data gen.

---

## What's Next

Remaining optimization opportunities (roughly ordered by impact/effort):

| Opportunity | Module | Est. Impact | Effort |
|---|---|---|---|
| Fuse bulk create + LSH insert in Rust | `src/lib.rs`, `lsh.py` | 30-50% e2e | High |
| Vectorize WeightedMinHash `minhash()` loop | `weighted_minhash.py` | 40-60% | Medium |
| Redis `mget` pipeline in storage | `storage.py` | 50-70% (Redis only) | Medium |
| NumPy `searchsorted` in LSHForest | `lshforest.py` | 20-30% | Low |
| LeanMinHash numpy-native serialization | `lean_minhash.py` | 30-40% serialize | Medium |
| Vectorize b-bit MinHash packing | `b_bit_minhash.py` | 20-30% | Medium |
| Explicit SIMD intrinsics (AVX2) | `src/lib.rs` | 10-20% | High |
