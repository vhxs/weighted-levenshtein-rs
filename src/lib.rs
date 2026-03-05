use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

const ALPHABET_SIZE: usize = 128;

// ---------------------------------------------------------------------------
// Core algorithms (pure Rust, no Python types)
// ---------------------------------------------------------------------------

fn c_levenshtein(
    str1: &[u8],
    str2: &[u8],
    insert_costs: &[f64],
    delete_costs: &[f64],
    substitute_costs: &[f64],
) -> f64 {
    let len1 = str1.len();
    let len2 = str2.len();
    let cols = len2 + 1;
    let mut d = vec![0.0f64; (len1 + 1) * cols];

    for i in 1..=len1 {
        let ci = str1[i - 1] as usize;
        d[i * cols] = d[(i - 1) * cols] + delete_costs[ci];
    }
    for j in 1..=len2 {
        let cj = str2[j - 1] as usize;
        d[j] = d[j - 1] + insert_costs[cj];
    }

    for i in 1..=len1 {
        let ci = str1[i - 1] as usize;
        for j in 1..=len2 {
            let cj = str2[j - 1] as usize;
            d[i * cols + j] = if ci == cj {
                d[(i - 1) * cols + (j - 1)]
            } else {
                (d[(i - 1) * cols + j] + delete_costs[ci])
                    .min(d[i * cols + (j - 1)] + insert_costs[cj])
                    .min(d[(i - 1) * cols + (j - 1)] + substitute_costs[ci * ALPHABET_SIZE + cj])
            };
        }
    }

    d[len1 * cols + len2]
}

fn c_optimal_string_alignment(
    str1: &[u8],
    str2: &[u8],
    insert_costs: &[f64],
    delete_costs: &[f64],
    substitute_costs: &[f64],
    transpose_costs: &[f64],
) -> f64 {
    let len1 = str1.len();
    let len2 = str2.len();
    let cols = len2 + 1;
    let mut d = vec![0.0f64; (len1 + 1) * cols];

    for i in 1..=len1 {
        let ci = str1[i - 1] as usize;
        d[i * cols] = d[(i - 1) * cols] + delete_costs[ci];
    }
    for j in 1..=len2 {
        let cj = str2[j - 1] as usize;
        d[j] = d[j - 1] + insert_costs[cj];
    }

    for i in 1..=len1 {
        let ci = str1[i - 1] as usize;
        for j in 1..=len2 {
            let cj = str2[j - 1] as usize;
            d[i * cols + j] = if ci == cj {
                d[(i - 1) * cols + (j - 1)]
            } else {
                (d[(i - 1) * cols + j] + delete_costs[ci])
                    .min(d[i * cols + (j - 1)] + insert_costs[cj])
                    .min(
                        d[(i - 1) * cols + (j - 1)]
                            + substitute_costs[ci * ALPHABET_SIZE + cj],
                    )
            };

            if i > 1 && j > 1 {
                let prev_ci = str1[i - 2] as usize;
                let prev_cj = str2[j - 2] as usize;
                if ci == prev_cj && prev_ci == cj {
                    let trans =
                        d[(i - 2) * cols + (j - 2)] + transpose_costs[prev_ci * ALPHABET_SIZE + ci];
                    if trans < d[i * cols + j] {
                        d[i * cols + j] = trans;
                    }
                }
            }
        }
    }

    d[len1 * cols + len2]
}

fn c_damerau_levenshtein(
    str1: &[u8],
    str2: &[u8],
    insert_costs: &[f64],
    delete_costs: &[f64],
    substitute_costs: &[f64],
    transpose_costs: &[f64],
) -> f64 {
    let len1 = str1.len();
    let len2 = str2.len();

    // (-1)-indexed matrix stored as (i+1, j+1) in a flat array.
    // i ranges from -1..=len1, j from -1..=len2  →  rows = len1+2, cols = len2+2
    let cols = len2 + 2;
    let mut d = vec![0.0f64; (len1 + 2) * cols];

    let idx = |i: isize, j: isize| -> usize { (i + 1) as usize * cols + (j + 1) as usize };

    // da[c] = last 1-indexed position in str1 where character c appeared
    let mut da = [0usize; ALPHABET_SIZE];

    // Sentinel row/column: d[-1][*] = d[*][-1] = f64::MAX
    d[idx(-1, -1)] = f64::MAX;
    for i in 0..=(len1 as isize) {
        d[idx(i, -1)] = f64::MAX;
    }
    for j in 0..=(len2 as isize) {
        d[idx(-1, j)] = f64::MAX;
    }

    // Base cases
    d[idx(0, 0)] = 0.0;
    for i in 1..=len1 {
        let ci = str1[i - 1] as usize;
        d[idx(i as isize, 0)] = d[idx((i - 1) as isize, 0)] + delete_costs[ci];
    }
    for j in 1..=len2 {
        let cj = str2[j - 1] as usize;
        d[idx(0, j as isize)] = d[idx(0, (j - 1) as isize)] + insert_costs[cj];
    }

    for i in 1..=len1 {
        let ci = str1[i - 1] as usize;
        let mut db = 0usize; // last position in str2 where ci appeared

        for j in 1..=len2 {
            let cj = str2[j - 1] as usize;
            let k = da[cj]; // last position in str1 where cj appeared
            let l = db;

            let cost = if ci == cj {
                db = j;
                0.0
            } else {
                substitute_costs[ci * ALPHABET_SIZE + cj]
            };

            // col_delete_range_cost(d, k+1, i-1) = d[i-1][0] - d[k][0]
            let col_del = d[idx((i as isize) - 1, 0)] - d[idx(k as isize, 0)];
            // row_insert_range_cost(d, l+1, j-1) = d[0][j-1] - d[0][l]
            let row_ins = d[idx(0, (j as isize) - 1)] - d[idx(0, l as isize)];

            // str1[k-1] == cj (by definition of da), str1[i-1] == ci
            let trans_cost = d[idx((k as isize) - 1, (l as isize) - 1)]
                + col_del
                + transpose_costs[cj * ALPHABET_SIZE + ci]
                + row_ins;

            d[idx(i as isize, j as isize)] = (d[idx((i - 1) as isize, (j - 1) as isize)] + cost)
                .min(d[idx(i as isize, (j - 1) as isize)] + insert_costs[cj])
                .min(d[idx((i - 1) as isize, j as isize)] + delete_costs[ci])
                .min(trans_cost);
        }

        da[ci] = i;
    }

    d[idx(len1 as isize, len2 as isize)]
}

// ---------------------------------------------------------------------------
// Helpers to extract weight arrays from optional numpy inputs
// ---------------------------------------------------------------------------

fn extract_1d(arr: Option<PyReadonlyArray1<f64>>) -> Vec<f64> {
    match arr {
        Some(a) => a.as_array().iter().copied().collect(),
        None => vec![1.0; ALPHABET_SIZE],
    }
}

fn extract_2d(arr: Option<PyReadonlyArray2<f64>>) -> Vec<f64> {
    match arr {
        Some(a) => a.as_array().iter().copied().collect(),
        None => vec![1.0; ALPHABET_SIZE * ALPHABET_SIZE],
    }
}

// ---------------------------------------------------------------------------
// Python-exposed functions
// ---------------------------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (str1, str2, insert_costs=None, delete_costs=None, substitute_costs=None))]
fn levenshtein(
    str1: &str,
    str2: &str,
    insert_costs: Option<PyReadonlyArray1<f64>>,
    delete_costs: Option<PyReadonlyArray1<f64>>,
    substitute_costs: Option<PyReadonlyArray2<f64>>,
) -> f64 {
    c_levenshtein(
        str1.as_bytes(),
        str2.as_bytes(),
        &extract_1d(insert_costs),
        &extract_1d(delete_costs),
        &extract_2d(substitute_costs),
    )
}

#[pyfunction]
#[pyo3(signature = (str1, str2, insert_costs=None, delete_costs=None, substitute_costs=None, transpose_costs=None))]
fn optimal_string_alignment(
    str1: &str,
    str2: &str,
    insert_costs: Option<PyReadonlyArray1<f64>>,
    delete_costs: Option<PyReadonlyArray1<f64>>,
    substitute_costs: Option<PyReadonlyArray2<f64>>,
    transpose_costs: Option<PyReadonlyArray2<f64>>,
) -> f64 {
    c_optimal_string_alignment(
        str1.as_bytes(),
        str2.as_bytes(),
        &extract_1d(insert_costs),
        &extract_1d(delete_costs),
        &extract_2d(substitute_costs),
        &extract_2d(transpose_costs),
    )
}

#[pyfunction]
#[pyo3(signature = (str1, str2, insert_costs=None, delete_costs=None, substitute_costs=None, transpose_costs=None))]
fn damerau_levenshtein(
    str1: &str,
    str2: &str,
    insert_costs: Option<PyReadonlyArray1<f64>>,
    delete_costs: Option<PyReadonlyArray1<f64>>,
    substitute_costs: Option<PyReadonlyArray2<f64>>,
    transpose_costs: Option<PyReadonlyArray2<f64>>,
) -> f64 {
    c_damerau_levenshtein(
        str1.as_bytes(),
        str2.as_bytes(),
        &extract_1d(insert_costs),
        &extract_1d(delete_costs),
        &extract_2d(substitute_costs),
        &extract_2d(transpose_costs),
    )
}

#[pymodule]
fn _clev(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(levenshtein, m)?)?;
    m.add_function(wrap_pyfunction!(optimal_string_alignment, m)?)?;
    m.add_function(wrap_pyfunction!(damerau_levenshtein, m)?)?;
    // Aliases
    m.add("lev", m.getattr("levenshtein")?)?;
    m.add("osa", m.getattr("optimal_string_alignment")?)?;
    m.add("dam_lev", m.getattr("damerau_levenshtein")?)?;
    Ok(())
}
