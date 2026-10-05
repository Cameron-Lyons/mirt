//! Bounded probability tables shared by respondents at a common theta grid.

use numpy::ndarray::{Array2, ArrayView2};
use rayon::prelude::*;

use crate::utils::logsumexp;

const TABLE_ENTRIES: usize = 1_000_000;

/// Evaluate an item/category table once per grid point, then gather response
/// contributions directly into the final contiguous person-by-grid buffer.
///
/// The table is stored category-major (`table[(offset_j + category) * count + q]`)
/// so each observed response adds one contiguous grid row to the person's output.
/// Items are still summed in column order at every grid point.
///
/// `fill(q, j, row)` receives a zeroed slice with one entry per category of
/// item `j`, so closures may rely on untouched entries being `0.0`.
pub(crate) fn cached_likelihoods(
    responses: ArrayView2<'_, i32>,
    n_quad: usize,
    categories: &[usize],
    binary: bool,
    fill: impl Fn(usize, usize, &mut [f64]),
) -> Array2<f64> {
    let mut output = Array2::zeros((responses.nrows(), n_quad));
    if n_quad == 0 || responses.nrows() == 0 || responses.ncols() == 0 {
        return output;
    }
    let mut offsets = vec![0; categories.len() + 1];
    for (j, &count) in categories.iter().enumerate() {
        offsets[j + 1] = offsets[j] + count;
    }
    let width = offsets[categories.len()];
    let chunk_size = (TABLE_ENTRIES / width).max(1).min(n_quad);
    let mut scratch = vec![0.0; categories.iter().copied().max().unwrap_or(0)];
    for start in (0..n_quad).step_by(chunk_size) {
        let count = chunk_size.min(n_quad - start);
        let mut table = vec![0.0; count * width];
        for q in 0..count {
            for (j, &n_categories) in categories.iter().enumerate() {
                let row = &mut scratch[..n_categories];
                row.fill(0.0);
                fill(start + q, j, row);
                for (category, &value) in row.iter().enumerate() {
                    table[(offsets[j] + category) * count + q] = value;
                }
            }
        }
        let evaluate = |(i, out): (usize, &mut [f64])| {
            let out = &mut out[start..start + count];
            for (j, &response) in responses.row(i).iter().enumerate() {
                if response >= 0 {
                    let category = if binary {
                        usize::from(response == 1)
                    } else {
                        response as usize
                    };
                    let base = (offsets[j] + category) * count;
                    for (total, &value) in out.iter_mut().zip(&table[base..base + count]) {
                        *total += value;
                    }
                }
            }
        };
        let data = output.as_slice_mut().expect("contiguous output");
        if responses.len().saturating_mul(count) < 32_768 {
            data.chunks_mut(n_quad).enumerate().for_each(evaluate);
        } else {
            data.par_chunks_mut(n_quad).enumerate().for_each(evaluate);
        }
    }
    output
}

/// Add the log prior terms to each row of log-likelihoods, normalize the rows in
/// place to posterior weights, and return each row's log normalizer.
pub(crate) fn normalize_log_posterior_rows(
    posterior: &mut Array2<f64>,
    log_prior: Option<&[f64]>,
    log_weights: &[f64],
) -> Vec<f64> {
    let n_quad = posterior.ncols();
    let mut marginal = vec![f64::NEG_INFINITY; posterior.nrows()];
    if n_quad == 0 {
        return marginal;
    }
    let normalize = |(row, marginal): (&mut [f64], &mut f64)| {
        match log_prior {
            Some(log_prior) => {
                for ((value, prior), weight) in row.iter_mut().zip(log_prior).zip(log_weights) {
                    *value = *value + prior + weight;
                }
            }
            None => {
                for (value, weight) in row.iter_mut().zip(log_weights) {
                    *value += weight;
                }
            }
        }
        *marginal = logsumexp(row);
        for value in row {
            *value = (*value - *marginal).exp();
        }
    };
    let data = posterior.as_slice_mut().expect("contiguous posterior");
    if data.len() < 32_768 {
        data.chunks_mut(n_quad)
            .zip(marginal.iter_mut())
            .for_each(normalize);
    } else {
        data.par_chunks_mut(n_quad)
            .zip(marginal.par_iter_mut())
            .for_each(normalize);
    }
    marginal
}

#[cfg(test)]
mod tests {
    use super::{TABLE_ENTRIES, cached_likelihoods};
    use numpy::ndarray::{Array2, array};

    #[test]
    fn chunks_cover_grid_boundaries_and_missing_rows() {
        let responses = array![[0], [1], [-9]];
        let n_quad = TABLE_ENTRIES / 2 + 3;
        let output = cached_likelihoods(responses.view(), n_quad, &[2], true, |q, _, row| {
            row[0] = q as f64;
            row[1] = -(q as f64);
        });
        for q in [0, n_quad / 2, n_quad - 4, n_quad - 1] {
            assert_eq!(output[[0, q]], q as f64);
            assert_eq!(output[[1, q]], -(q as f64));
            assert_eq!(output[[2, q]], 0.0);
        }
    }

    #[test]
    fn mixed_categories_match_naive_gather_with_zeroed_rows() {
        let categories = [2, 3, 5];
        let responses = array![[0, 2, 4], [1, -1, 0], [-1, 1, 3], [1, 0, -1], [-1, -1, -1]];
        // Leaves category 0 untouched, as the GPCM closure does.
        let value = |q: usize, j: usize, k: usize| {
            if k == 0 {
                0.0
            } else {
                (q as f64 + 0.5) * 0.1 + (j * 7 + k) as f64 * 0.013
            }
        };
        let fill = |q: usize, j: usize, row: &mut [f64]| {
            assert!(row.iter().all(|&v| v == 0.0));
            for (k, entry) in row.iter_mut().enumerate().skip(1) {
                *entry = value(q, j, k);
            }
        };
        let expected = |n_quad: usize| {
            let mut expected = Array2::zeros((responses.nrows(), n_quad));
            for i in 0..responses.nrows() {
                for q in 0..n_quad {
                    let mut total = 0.0;
                    for j in 0..categories.len() {
                        let r = responses[[i, j]];
                        if r >= 0 {
                            total += value(q, j, r as usize);
                        }
                    }
                    expected[[i, q]] = total;
                }
            }
            expected
        };
        for n_quad in [1, 7, 61] {
            let output = cached_likelihoods(responses.view(), n_quad, &categories, false, fill);
            assert_eq!(output, expected(n_quad));
        }

        let binary = array![[0, 1], [5, -2], [1, 0]];
        let output = cached_likelihoods(binary.view(), 4, &[2, 2], true, |q, j, row| {
            row[0] = -(q as f64) - j as f64;
            row[1] = q as f64 * 0.5 + j as f64;
        });
        for q in 0..4 {
            let lo = |j: usize| -(q as f64) - j as f64;
            let hi = |j: usize| q as f64 * 0.5 + j as f64;
            assert_eq!(output[[0, q]], 0.0 + lo(0) + hi(1));
            // Binary coding maps any positive non-one response to category 0.
            assert_eq!(output[[1, q]], 0.0 + lo(0));
            assert_eq!(output[[2, q]], 0.0 + hi(0) + lo(1));
        }
    }
}
