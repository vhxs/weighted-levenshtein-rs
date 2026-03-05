use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

const ALPHABET_SIZE: usize = 128;

// ---------------------------------------------------------------------------
// Core algorithms (pure Rust, no Python types)
// ---------------------------------------------------------------------------

// Levenshtein distance (Wagner-Fischer algorithm).
// https://en.wikipedia.org/wiki/Wagner%E2%80%93Fischer_algorithm
//
// dp[i][j] = minimum cost to transform source[..i] into target[..j]
// using weighted insertions, deletions, and substitutions.
fn c_levenshtein(
    source: &[u8],
    target: &[u8],
    insert_costs: &[f64],
    delete_costs: &[f64],
    substitute_costs: &[f64],
) -> f64 {
    let m = source.len();
    let n = target.len();
    let ncols = n + 1;
    let mut dp = vec![0.0f64; (m + 1) * ncols];

    // Base cases: cost of deleting all of source, or inserting all of target
    for i in 1..=m {
        let ch_s = source[i - 1] as usize;
        dp[i * ncols] = dp[(i - 1) * ncols] + delete_costs[ch_s];
    }
    for j in 1..=n {
        let ch_t = target[j - 1] as usize;
        dp[j] = dp[j - 1] + insert_costs[ch_t];
    }

    for i in 1..=m {
        let ch_s = source[i - 1] as usize;
        for j in 1..=n {
            let ch_t = target[j - 1] as usize;
            dp[i * ncols + j] = if ch_s == ch_t {
                dp[(i - 1) * ncols + (j - 1)] // characters match, no cost
            } else {
                (dp[(i - 1) * ncols + j] + delete_costs[ch_s])
                    .min(dp[i * ncols + (j - 1)] + insert_costs[ch_t])
                    .min(dp[(i - 1) * ncols + (j - 1)] + substitute_costs[ch_s * ALPHABET_SIZE + ch_t])
            };
        }
    }

    dp[m * ncols + n]
}

// Optimal String Alignment distance.
// https://en.wikipedia.org/wiki/Damerau%E2%80%93Levenshtein_distance#Optimal_string_alignment_distance
//
// Like Levenshtein but also allows adjacent transpositions. Note: unlike true
// Damerau-Levenshtein, a substring may not be edited more than once.
fn c_optimal_string_alignment(
    source: &[u8],
    target: &[u8],
    insert_costs: &[f64],
    delete_costs: &[f64],
    substitute_costs: &[f64],
    transpose_costs: &[f64],
) -> f64 {
    let m = source.len();
    let n = target.len();
    let ncols = n + 1;
    let mut dp = vec![0.0f64; (m + 1) * ncols];

    // Base cases
    for i in 1..=m {
        let ch_s = source[i - 1] as usize;
        dp[i * ncols] = dp[(i - 1) * ncols] + delete_costs[ch_s];
    }
    for j in 1..=n {
        let ch_t = target[j - 1] as usize;
        dp[j] = dp[j - 1] + insert_costs[ch_t];
    }

    for i in 1..=m {
        let ch_s = source[i - 1] as usize;
        for j in 1..=n {
            let ch_t = target[j - 1] as usize;
            dp[i * ncols + j] = if ch_s == ch_t {
                dp[(i - 1) * ncols + (j - 1)]
            } else {
                (dp[(i - 1) * ncols + j] + delete_costs[ch_s])
                    .min(dp[i * ncols + (j - 1)] + insert_costs[ch_t])
                    .min(dp[(i - 1) * ncols + (j - 1)] + substitute_costs[ch_s * ALPHABET_SIZE + ch_t])
            };

            // Check for adjacent transposition: source[i-2..i] == reverse of target[j-2..j]
            if i > 1 && j > 1 {
                let prev_ch_s = source[i - 2] as usize;
                let prev_ch_t = target[j - 2] as usize;
                if ch_s == prev_ch_t && prev_ch_s == ch_t {
                    let transpose_cost = dp[(i - 2) * ncols + (j - 2)]
                        + transpose_costs[prev_ch_s * ALPHABET_SIZE + ch_s];
                    if transpose_cost < dp[i * ncols + j] {
                        dp[i * ncols + j] = transpose_cost;
                    }
                }
            }
        }
    }

    dp[m * ncols + n]
}

// True Damerau-Levenshtein distance with adjacent transpositions.
// https://en.wikipedia.org/wiki/Damerau%E2%80%93Levenshtein_distance#Distance_with_adjacent_transpositions
//
// Unlike OSA, this allows a substring to be edited more than once, making it
// a true metric. Uses a (-1)-indexed DP matrix with sentinel values.
fn c_damerau_levenshtein(
    source: &[u8],
    target: &[u8],
    insert_costs: &[f64],
    delete_costs: &[f64],
    substitute_costs: &[f64],
    transpose_costs: &[f64],
) -> f64 {
    let m = source.len();
    let n = target.len();

    // The DP matrix is (-1)-indexed: rows span -1..=m, cols span -1..=n.
    // We store it as a flat array with (i+1, j+1) as the 0-based index.
    let ncols = n + 2;
    let mut dp = vec![0.0f64; (m + 2) * ncols];

    // Converts (-1)-based (row, col) to a flat array index
    let cell = |row: isize, col: isize| -> usize { (row + 1) as usize * ncols + (col + 1) as usize };

    // last_row_seen[c] = last 1-indexed row in source where character c appeared
    let mut last_row_seen = [0usize; ALPHABET_SIZE];

    // Sentinel row and column are set to f64::MAX to make transpositions
    // that cross the boundary of the string prohibitively expensive
    dp[cell(-1, -1)] = f64::MAX;
    for i in 0..=(m as isize) {
        dp[cell(i, -1)] = f64::MAX;
    }
    for j in 0..=(n as isize) {
        dp[cell(-1, j)] = f64::MAX;
    }

    // Base cases: cost of transforming to/from empty string
    dp[cell(0, 0)] = 0.0;
    for i in 1..=m {
        let ch_s = source[i - 1] as usize;
        dp[cell(i as isize, 0)] = dp[cell((i - 1) as isize, 0)] + delete_costs[ch_s];
    }
    for j in 1..=n {
        let ch_t = target[j - 1] as usize;
        dp[cell(0, j as isize)] = dp[cell(0, (j - 1) as isize)] + insert_costs[ch_t];
    }

    for i in 1..=m {
        let ch_s = source[i - 1] as usize;
        // last column in target where ch_s was seen (updated as we scan columns)
        let mut last_col_match = 0usize;

        for j in 1..=n {
            let ch_t = target[j - 1] as usize;
            // last row in source where ch_t appeared
            let prev_source_row = last_row_seen[ch_t];
            // last column in target where ch_s appeared (before this iteration)
            let prev_target_col = last_col_match;

            let substitution_cost = if ch_s == ch_t {
                last_col_match = j; // ch_s was just matched at column j
                0.0
            } else {
                substitute_costs[ch_s * ALPHABET_SIZE + ch_t]
            };

            // Cost of deleting source[prev_source_row..i-1] and inserting target[prev_target_col..j-1]
            // to bridge the gap between the transposed characters. Computed as a
            // difference of cumulative costs stored in column 0 and row 0 of dp.
            let delete_range_cost =
                dp[cell((i as isize) - 1, 0)] - dp[cell(prev_source_row as isize, 0)];
            let insert_range_cost =
                dp[cell(0, (j as isize) - 1)] - dp[cell(0, prev_target_col as isize)];

            let transpose_cost = dp[cell((prev_source_row as isize) - 1, (prev_target_col as isize) - 1)]
                + delete_range_cost
                + transpose_costs[ch_t * ALPHABET_SIZE + ch_s]
                + insert_range_cost;

            dp[cell(i as isize, j as isize)] =
                (dp[cell((i - 1) as isize, (j - 1) as isize)] + substitution_cost)
                    .min(dp[cell(i as isize, (j - 1) as isize)] + insert_costs[ch_t])
                    .min(dp[cell((i - 1) as isize, j as isize)] + delete_costs[ch_s])
                    .min(transpose_cost);
        }

        last_row_seen[ch_s] = i;
    }

    dp[cell(m as isize, n as isize)]
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
    // Short aliases
    m.add("lev", m.getattr("levenshtein")?)?;
    m.add("osa", m.getattr("optimal_string_alignment")?)?;
    m.add("dam_lev", m.getattr("damerau_levenshtein")?)?;
    Ok(())
}
