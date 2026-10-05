//! Expected item response counts gathered from contiguous posterior rows.
//!
//! `posterior` is a row-major person-by-grid buffer. Persons are accumulated in
//! row order, so every count is independent of how callers parallelize items.

use numpy::ndarray::ArrayView2;

fn accumulate(target: &mut [f64], row: &[f64], weight: Option<f64>) {
    match weight {
        None => {
            for (total, &value) in target.iter_mut().zip(row) {
                *total += value;
            }
        }
        Some(weight) => {
            for (total, &value) in target.iter_mut().zip(row) {
                *total += value * weight;
            }
        }
    }
}

/// Expected counts `(r_k, n_k)` for binary item `j`: `n_k[q]` sums the posterior
/// mass of respondents who answered the item and `r_k[q]` of those who answered 1.
/// Optional `weights` scale each respondent's row.
pub(crate) fn binary_item_counts(
    responses: ArrayView2<'_, i32>,
    posterior: &[f64],
    n_quad: usize,
    j: usize,
    weights: Option<&[f64]>,
) -> (Vec<f64>, Vec<f64>) {
    debug_assert_eq!(posterior.len(), responses.nrows() * n_quad);
    let mut r_k = vec![0.0; n_quad];
    let mut n_k = vec![0.0; n_quad];
    if n_quad == 0 {
        return (r_k, n_k);
    }
    let rows = posterior.chunks_exact(n_quad).zip(responses.column(j));
    for (i, (row, &response)) in rows.enumerate() {
        if response < 0 {
            continue;
        }
        let weight = weights.map(|weights| weights[i]);
        accumulate(&mut n_k, row, weight);
        if response == 1 {
            accumulate(&mut r_k, row, weight);
        }
    }
    (r_k, n_k)
}

/// Category-major expected counts for item `j`: `counts[c * n_quad + q]` sums the
/// posterior mass at grid point `q` of respondents in category `c`.
pub(crate) fn category_item_counts(
    responses: ArrayView2<'_, i32>,
    posterior: &[f64],
    n_quad: usize,
    j: usize,
    n_categories: usize,
) -> Vec<f64> {
    debug_assert_eq!(posterior.len(), responses.nrows() * n_quad);
    let mut counts = vec![0.0; n_categories * n_quad];
    if n_quad == 0 {
        return counts;
    }
    for (row, &response) in posterior.chunks_exact(n_quad).zip(responses.column(j)) {
        if response >= 0 {
            let start = response as usize * n_quad;
            accumulate(&mut counts[start..start + n_quad], row, None);
        }
    }
    counts
}

#[cfg(test)]
mod tests {
    use super::{binary_item_counts, category_item_counts};
    use numpy::ndarray::{Array2, array};

    fn posterior(n_persons: usize, n_quad: usize) -> Array2<f64> {
        Array2::from_shape_fn((n_persons, n_quad), |(i, q)| {
            ((i * 31 + q * 17) % 23) as f64 / 23.0 + 0.01
        })
    }

    #[test]
    fn binary_counts_match_naive_loop_with_missing_and_weights() {
        let responses = array![[1, 0], [0, -1], [-9, 1], [1, 1], [2, 0]];
        let n_quad = 5;
        let posterior = posterior(responses.nrows(), n_quad);
        let weights = [1.0, 2.5, 0.5, 3.0, 1.5];
        for weighted in [false, true] {
            for j in 0..responses.ncols() {
                let (r_k, n_k) = binary_item_counts(
                    responses.view(),
                    posterior.as_slice().unwrap(),
                    n_quad,
                    j,
                    weighted.then_some(&weights[..]),
                );
                let mut expected_r = vec![0.0; n_quad];
                let mut expected_n = vec![0.0; n_quad];
                for i in 0..responses.nrows() {
                    let response = responses[[i, j]];
                    if response < 0 {
                        continue;
                    }
                    for q in 0..n_quad {
                        let w = if weighted {
                            posterior[[i, q]] * weights[i]
                        } else {
                            posterior[[i, q]]
                        };
                        expected_n[q] += w;
                        if response == 1 {
                            expected_r[q] += w;
                        }
                    }
                }
                assert_eq!(r_k, expected_r);
                assert_eq!(n_k, expected_n);
            }
        }
        let (r_k, n_k) = binary_item_counts(responses.view(), &[], 0, 0, None);
        assert!(r_k.is_empty() && n_k.is_empty());
    }

    #[test]
    fn category_counts_match_naive_loop() {
        let responses = array![[2, 0], [0, -1], [-1, 1], [1, 1], [2, 0], [0, 3]];
        let n_quad = 4;
        let n_categories = 4;
        let posterior = posterior(responses.nrows(), n_quad);
        for j in 0..responses.ncols() {
            let counts = category_item_counts(
                responses.view(),
                posterior.as_slice().unwrap(),
                n_quad,
                j,
                n_categories,
            );
            let mut expected = vec![0.0; n_categories * n_quad];
            for i in 0..responses.nrows() {
                let response = responses[[i, j]];
                if response >= 0 {
                    for q in 0..n_quad {
                        expected[response as usize * n_quad + q] += posterior[[i, q]];
                    }
                }
            }
            assert_eq!(counts, expected);
        }
    }
}
