use numpy::ndarray::ArrayViewMut1;
use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1};
use pyo3::prelude::*;
use rayon::prelude::*;
use xxhash_rust::xxh32::xxh32 as xxhash32_impl;
use xxhash_rust::xxh64::xxh64 as xxhash64_impl;

const MERSENNE_PRIME: u64 = (1u64 << 61) - 1;
const MAX_HASH: u64 = (1u64 << 32) - 1;

/// Minimum items before Rayon parallelism kicks in for MinHash.
/// At 1k items the Rayon overhead roughly breaks even; at 2k+ it wins.
const RAYON_MINHASH_THRESHOLD: usize = 2048;
/// Items per Rayon chunk for MinHash batch updates.
const MINHASH_CHUNK_SIZE: usize = 512;

// --------------------------------------------------------------------------
// Fast Mersenne prime reduction
// --------------------------------------------------------------------------

/// Fast modular reduction by the Mersenne prime 2^61 - 1.
///
/// Uses the identity: x mod (2^61 - 1) = (x >> 61) + (x & (2^61 - 1))
/// with at most one conditional subtraction.
///
/// Valid for inputs where hi + lo < 2 * (2^61 - 1), which holds when
/// x < 2^125. Our inputs are a[i]*hv + b[i] where a,b < 2^61 and
/// hv < 2^32, so x < 2^93 + 2^61 — well within range.
///
/// This replaces the generic `% (MERSENNE_PRIME as u128)` which compiles
/// to an expensive software division call (__udivti3).
#[inline(always)]
fn mod_mersenne(x: u128) -> u64 {
    let lo = (x & ((1u128 << 61) - 1)) as u64;
    let hi = (x >> 61) as u64;
    let r = lo + hi;
    if r >= MERSENNE_PRIME { r - MERSENNE_PRIME } else { r }
}

// --------------------------------------------------------------------------
// Auto-vectorization-friendly permutation kernel
// --------------------------------------------------------------------------

/// Apply all permutations to a single hash value, min-reduce into `out`.
///
/// Uses fast Mersenne reduction and unchecked indexing to help LLVM
/// auto-vectorize the inner loop.
#[inline(always)]
fn permute_min_into(hv: u64, a: &[u64], b: &[u64], out: &mut [u64]) {
    let n = a.len();
    debug_assert_eq!(n, b.len());
    debug_assert_eq!(n, out.len());
    let h = hv as u128;
    for i in 0..n {
        // SAFETY: bounds are verified by debug_assert above; all three
        // slices have length `n` and `i` is in 0..n.
        unsafe {
            let ai = *a.get_unchecked(i) as u128;
            let bi = *b.get_unchecked(i) as u128;
            let phv = mod_mersenne(ai * h + bi) & MAX_HASH;
            let slot = out.get_unchecked_mut(i);
            if phv < *slot {
                *slot = phv;
            }
        }
    }
}

/// Element-wise min of two u64 slices, writing into `acc`.
#[inline(always)]
fn merge_min_u64(acc: &mut [u64], other: &[u64]) {
    let n = acc.len();
    debug_assert_eq!(n, other.len());
    for i in 0..n {
        unsafe {
            let a = acc.get_unchecked_mut(i);
            let o = *other.get_unchecked(i);
            if o < *a { *a = o; }
        }
    }
}

// --------------------------------------------------------------------------
// HLL single-item helpers
// --------------------------------------------------------------------------

/// Process one item for HLL (32-bit hash). Returns false on rank overflow.
#[inline(always)]
fn hll32_one(data: &[u8], p: u8, max_rank: u8, mask: u64, reg: &mut [i8]) -> bool {
    let hv = xxhash32_impl(data, 0) as u64;
    let reg_index = (hv & mask) as usize;
    let bits = hv >> p;
    let bit_len = if bits == 0 { 0u8 } else { (64 - bits.leading_zeros()) as u8 };
    let rank = (max_rank as i16) - (bit_len as i16) + 1;
    if rank <= 0 { return false; }
    unsafe {
        let slot = reg.get_unchecked_mut(reg_index);
        if (rank as i8) > *slot { *slot = rank as i8; }
    }
    true
}

/// Process one item for HLL++ (64-bit hash). Returns false on rank overflow.
#[inline(always)]
fn hll64_one(data: &[u8], p: u8, max_rank: u8, mask: u64, reg: &mut [i8]) -> bool {
    let hv = xxhash64_impl(data, 0);
    let reg_index = (hv & mask) as usize;
    let bits = hv >> p;
    let bit_len = if bits == 0 { 0u8 } else { (64 - bits.leading_zeros()) as u8 };
    let rank = (max_rank as i16) - (bit_len as i16) + 1;
    if rank <= 0 { return false; }
    unsafe {
        let slot = reg.get_unchecked_mut(reg_index);
        if (rank as i8) > *slot { *slot = rank as i8; }
    }
    true
}

// --------------------------------------------------------------------------
// Standalone hash functions
// --------------------------------------------------------------------------

#[pyfunction]
fn xxhash32(data: &[u8]) -> u32 {
    xxhash32_impl(data, 0)
}

#[pyfunction]
fn xxhash64(data: &[u8]) -> u64 {
    xxhash64_impl(data, 0)
}

// --------------------------------------------------------------------------
// Fused MinHash update (single element)
// --------------------------------------------------------------------------

/// Hash one item, apply all permutations, min-reduce into hashvalues.
#[pyfunction]
fn minhash_update<'py>(
    data: &[u8],
    a: PyReadonlyArray1<'py, u64>,
    b: PyReadonlyArray1<'py, u64>,
    hashvalues: &Bound<'py, PyArray1<u64>>,
) {
    let hv = xxhash32_impl(data, 0) as u64;
    let a_arr = a.as_array();
    let b_arr = b.as_array();
    let mut hv_arr: ArrayViewMut1<u64> = unsafe { hashvalues.as_array_mut() };

    let a_slice = a_arr.as_slice().unwrap();
    let b_slice = b_arr.as_slice().unwrap();
    let hv_slice = hv_arr.as_slice_mut().unwrap();

    permute_min_into(hv, a_slice, b_slice, hv_slice);
}

// --------------------------------------------------------------------------
// Fused MinHash batch update (Rayon-parallel, GIL-released)
// --------------------------------------------------------------------------

/// Hash multiple items, apply all permutations, min-reduce into hashvalues.
///
/// For large batches, uses Rayon to parallelize across items with
/// thread-local accumulators merged by element-wise min. The GIL is
/// released during computation.
#[pyfunction]
fn minhash_update_batch<'py>(
    py: Python<'py>,
    data_list: Vec<Vec<u8>>,
    a: PyReadonlyArray1<'py, u64>,
    b: PyReadonlyArray1<'py, u64>,
    hashvalues: &Bound<'py, PyArray1<u64>>,
) {
    let a_arr = a.as_array();
    let b_arr = b.as_array();
    let a_slice = a_arr.as_slice().unwrap();
    let b_slice = b_arr.as_slice().unwrap();
    let mut hv_arr: ArrayViewMut1<u64> = unsafe { hashvalues.as_array_mut() };
    let hv_slice = hv_arr.as_slice_mut().unwrap();

    if data_list.len() < RAYON_MINHASH_THRESHOLD {
        // Sequential path: operate directly on numpy arrays, zero copies.
        for data in &data_list {
            let h = xxhash32_impl(data, 0) as u64;
            permute_min_into(h, a_slice, b_slice, hv_slice);
        }
    } else {
        // Parallel path: copy to owned Vecs so we can release the GIL.
        let a_vec: Vec<u64> = a_slice.to_vec();
        let b_vec: Vec<u64> = b_slice.to_vec();
        let num_perm = a_vec.len();
        let init_hv: Vec<u64> = hv_slice.to_vec();

        let result = py.allow_threads(|| {
            data_list
                .par_chunks(MINHASH_CHUNK_SIZE)
                .map(|chunk| {
                    let mut local = init_hv.clone();
                    for data in chunk {
                        let h = xxhash32_impl(data, 0) as u64;
                        permute_min_into(h, &a_vec, &b_vec, &mut local);
                    }
                    local
                })
                .reduce(
                    || vec![u64::MAX; num_perm],
                    |mut acc, local| {
                        merge_min_u64(&mut acc, &local);
                        acc
                    },
                )
        });

        hv_slice.copy_from_slice(&result);
    }
}

// --------------------------------------------------------------------------
// Fused HyperLogLog update (single element, 32-bit hash)
// --------------------------------------------------------------------------

#[pyfunction]
fn hll_update<'py>(
    data: &[u8],
    p: u8,
    max_rank: u8,
    reg: &Bound<'py, PyArray1<i8>>,
) -> PyResult<()> {
    let mask = (1u64 << p) - 1;
    let mut reg_arr: ArrayViewMut1<i8> = unsafe { reg.as_array_mut() };
    let reg_slice = reg_arr.as_slice_mut().unwrap();
    if !hll32_one(data, p, max_rank, mask, reg_slice) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            format!("Hash value overflow, maximum size is {} bits", max_rank),
        ));
    }
    Ok(())
}

// --------------------------------------------------------------------------
// Fused HyperLogLog batch update (32-bit hash)
// --------------------------------------------------------------------------
//
// HLL per-item work is very light (~20ns: hash + single register update),
// so Rayon parallelism is counterproductive — the cost of cloning and
// merging large register arrays (up to 64KB for p=16) dominates.
// Sequential with unchecked indexing is fastest.

#[pyfunction]
fn hll_update_batch<'py>(
    data_list: Vec<Vec<u8>>,
    p: u8,
    max_rank: u8,
    reg: &Bound<'py, PyArray1<i8>>,
) -> PyResult<()> {
    let mask = (1u64 << p) - 1;
    let mut reg_arr: ArrayViewMut1<i8> = unsafe { reg.as_array_mut() };
    let reg_slice = reg_arr.as_slice_mut().unwrap();

    for data in &data_list {
        if !hll32_one(data, p, max_rank, mask, reg_slice) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("Hash value overflow, maximum size is {} bits", max_rank),
            ));
        }
    }
    Ok(())
}

// --------------------------------------------------------------------------
// Fused HyperLogLog update (single element, 64-bit hash) for HLL++
// --------------------------------------------------------------------------

#[pyfunction]
fn hll64_update<'py>(
    data: &[u8],
    p: u8,
    max_rank: u8,
    reg: &Bound<'py, PyArray1<i8>>,
) -> PyResult<()> {
    let mask = (1u64 << p) - 1;
    let mut reg_arr: ArrayViewMut1<i8> = unsafe { reg.as_array_mut() };
    let reg_slice = reg_arr.as_slice_mut().unwrap();
    if !hll64_one(data, p, max_rank, mask, reg_slice) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            format!("Hash value overflow, maximum size is {} bits", max_rank),
        ));
    }
    Ok(())
}

// --------------------------------------------------------------------------
// Fused HyperLogLog++ batch update (64-bit hash)
// --------------------------------------------------------------------------

#[pyfunction]
fn hll64_update_batch<'py>(
    data_list: Vec<Vec<u8>>,
    p: u8,
    max_rank: u8,
    reg: &Bound<'py, PyArray1<i8>>,
) -> PyResult<()> {
    let mask = (1u64 << p) - 1;
    let mut reg_arr: ArrayViewMut1<i8> = unsafe { reg.as_array_mut() };
    let reg_slice = reg_arr.as_slice_mut().unwrap();

    for data in &data_list {
        if !hll64_one(data, p, max_rank, mask, reg_slice) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("Hash value overflow, maximum size is {} bits", max_rank),
            ));
        }
    }
    Ok(())
}

// --------------------------------------------------------------------------
// Bulk MinHash creation (Rayon-parallel across MinHashes, GIL-released)
// --------------------------------------------------------------------------

/// Create N MinHashes in parallel from pre-computed hash values.
///
/// Takes a flat array of u32 hash values and an offsets array that
/// delimits which hashes belong to each MinHash. Returns a flat
/// Vec<u64> of length N * num_perm.
///
/// This avoids the expensive Python→Rust conversion of nested byte lists
/// by accepting pre-hashed values (via xxhash32) as numpy arrays.
#[pyfunction]
fn minhash_bulk_from_hashes<'py>(
    py: Python<'py>,
    hashes: PyReadonlyArray1<'py, u32>,
    offsets: PyReadonlyArray1<'py, u64>,
    a: PyReadonlyArray1<'py, u64>,
    b: PyReadonlyArray1<'py, u64>,
) -> Vec<u64> {
    let h_slice = hashes.as_array();
    let h_data = h_slice.as_slice().unwrap();
    let off_slice = offsets.as_array();
    let off_data = off_slice.as_slice().unwrap();
    let a_arr = a.as_array();
    let b_arr = b.as_array();
    let a_data = a_arr.as_slice().unwrap();
    let b_data = b_arr.as_slice().unwrap();
    let num_perm = a_data.len();
    let n = off_data.len() - 1; // number of MinHashes

    // Copy permutation arrays for Rayon (need Send)
    let a_vec: Vec<u64> = a_data.to_vec();
    let b_vec: Vec<u64> = b_data.to_vec();
    // Copy hash data for Rayon (numpy array isn't Send)
    let h_vec: Vec<u32> = h_data.to_vec();
    let off_vec: Vec<u64> = off_data.to_vec();

    py.allow_threads(|| {
        let mut out = vec![MAX_HASH; n * num_perm];
        out.par_chunks_mut(num_perm)
            .enumerate()
            .for_each(|(i, hv_row)| {
                let start = off_vec[i] as usize;
                let end = off_vec[i + 1] as usize;
                for j in start..end {
                    let h = unsafe { *h_vec.get_unchecked(j) } as u64;
                    permute_min_into(h, &a_vec, &b_vec, hv_row);
                }
            });
        out
    })
}

// --------------------------------------------------------------------------
// Module
// --------------------------------------------------------------------------

#[pymodule]
fn _rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(xxhash32, m)?)?;
    m.add_function(wrap_pyfunction!(xxhash64, m)?)?;
    m.add_function(wrap_pyfunction!(minhash_update, m)?)?;
    m.add_function(wrap_pyfunction!(minhash_update_batch, m)?)?;
    m.add_function(wrap_pyfunction!(hll_update, m)?)?;
    m.add_function(wrap_pyfunction!(hll_update_batch, m)?)?;
    m.add_function(wrap_pyfunction!(hll64_update, m)?)?;
    m.add_function(wrap_pyfunction!(hll64_update_batch, m)?)?;
    m.add_function(wrap_pyfunction!(minhash_bulk_from_hashes, m)?)?;
    Ok(())
}
