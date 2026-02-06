use numpy::ndarray::{ArrayView1, ArrayViewMut1};
use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1};
use pyo3::prelude::*;
use xxhash_rust::xxh32::xxh32 as xxhash32_impl;
use xxhash_rust::xxh64::xxh64 as xxhash64_impl;

const MERSENNE_PRIME: u64 = (1u64 << 61) - 1;
const MAX_HASH: u64 = (1u64 << 32) - 1;

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
    let a_arr: ArrayView1<u64> = a.as_array();
    let b_arr: ArrayView1<u64> = b.as_array();

    // SAFETY: we have exclusive access through Python's GIL
    let mut hv_arr: ArrayViewMut1<u64> = unsafe { hashvalues.as_array_mut() };

    for i in 0..a_arr.len() {
        let ai = a_arr[i] as u128;
        let bi = b_arr[i] as u128;
        let h = hv as u128;
        let phv = ((ai * h + bi) % (MERSENNE_PRIME as u128)) as u64 & MAX_HASH;
        if phv < hv_arr[i] {
            hv_arr[i] = phv;
        }
    }
}

// --------------------------------------------------------------------------
// Fused MinHash batch update
// --------------------------------------------------------------------------

/// Hash multiple items, apply all permutations, min-reduce into hashvalues.
#[pyfunction]
fn minhash_update_batch<'py>(
    data_list: Vec<Vec<u8>>,
    a: PyReadonlyArray1<'py, u64>,
    b: PyReadonlyArray1<'py, u64>,
    hashvalues: &Bound<'py, PyArray1<u64>>,
) {
    let a_arr: ArrayView1<u64> = a.as_array();
    let b_arr: ArrayView1<u64> = b.as_array();
    let num_perm = a_arr.len();

    // SAFETY: we have exclusive access through Python's GIL
    let mut hv_arr: ArrayViewMut1<u64> = unsafe { hashvalues.as_array_mut() };

    for data in &data_list {
        let hv = xxhash32_impl(data, 0) as u128;
        for i in 0..num_perm {
            let ai = a_arr[i] as u128;
            let bi = b_arr[i] as u128;
            let phv = ((ai * hv + bi) % (MERSENNE_PRIME as u128)) as u64 & MAX_HASH;
            if phv < hv_arr[i] {
                hv_arr[i] = phv;
            }
        }
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
    let hv = xxhash32_impl(data, 0) as u64;
    let m = 1u64 << p;
    let reg_index = (hv & (m - 1)) as usize;
    let bits = hv >> p;
    let bit_len = if bits == 0 { 0u8 } else { (64 - bits.leading_zeros()) as u8 };
    let rank = max_rank - bit_len + 1;
    if rank == 0 || rank > max_rank {
        return Err(pyo3::exceptions::PyValueError::new_err(
            format!("Hash value overflow, maximum size is {} bits", max_rank),
        ));
    }

    // SAFETY: we have exclusive access through Python's GIL
    let mut reg_arr: ArrayViewMut1<i8> = unsafe { reg.as_array_mut() };
    if (rank as i8) > reg_arr[reg_index] {
        reg_arr[reg_index] = rank as i8;
    }
    Ok(())
}

// --------------------------------------------------------------------------
// Fused HyperLogLog batch update (32-bit hash)
// --------------------------------------------------------------------------

#[pyfunction]
fn hll_update_batch<'py>(
    data_list: Vec<Vec<u8>>,
    p: u8,
    max_rank: u8,
    reg: &Bound<'py, PyArray1<i8>>,
) -> PyResult<()> {
    let m = 1u64 << p;
    let mask = m - 1;

    // SAFETY: we have exclusive access through Python's GIL
    let mut reg_arr: ArrayViewMut1<i8> = unsafe { reg.as_array_mut() };

    for data in &data_list {
        let hv = xxhash32_impl(data, 0) as u64;
        let reg_index = (hv & mask) as usize;
        let bits = hv >> p;
        let bit_len = if bits == 0 { 0u8 } else { (64 - bits.leading_zeros()) as u8 };
        let rank = max_rank - bit_len + 1;
        if rank == 0 || rank > max_rank {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("Hash value overflow, maximum size is {} bits", max_rank),
            ));
        }
        if (rank as i8) > reg_arr[reg_index] {
            reg_arr[reg_index] = rank as i8;
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
    let hv = xxhash64_impl(data, 0);
    let m = 1u64 << p;
    let reg_index = (hv & (m - 1)) as usize;
    let bits = hv >> p;
    let bit_len = if bits == 0 { 0u8 } else { (64 - bits.leading_zeros()) as u8 };
    let rank = max_rank - bit_len + 1;
    if rank == 0 || rank > max_rank {
        return Err(pyo3::exceptions::PyValueError::new_err(
            format!("Hash value overflow, maximum size is {} bits", max_rank),
        ));
    }

    let mut reg_arr: ArrayViewMut1<i8> = unsafe { reg.as_array_mut() };
    if (rank as i8) > reg_arr[reg_index] {
        reg_arr[reg_index] = rank as i8;
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
    let m = 1u64 << p;
    let mask = m - 1;

    let mut reg_arr: ArrayViewMut1<i8> = unsafe { reg.as_array_mut() };

    for data in &data_list {
        let hv = xxhash64_impl(data, 0);
        let reg_index = (hv & mask) as usize;
        let bits = hv >> p;
        let bit_len = if bits == 0 { 0u8 } else { (64 - bits.leading_zeros()) as u8 };
        let rank = max_rank - bit_len + 1;
        if rank == 0 || rank > max_rank {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("Hash value overflow, maximum size is {} bits", max_rank),
            ));
        }
        if (rank as i8) > reg_arr[reg_index] {
            reg_arr[reg_index] = rank as i8;
        }
    }
    Ok(())
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
    Ok(())
}
