//! Bounded probability tables shared by respondents at a common theta grid.

use numpy::ndarray::{Array2, ArrayView2};
use rayon::prelude::*;

const TABLE_ENTRIES: usize = 1_000_000;

/// Evaluate an item/category table once per grid point, then gather response
/// contributions directly into the final contiguous person-by-grid buffer.
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
    for start in (0..n_quad).step_by(chunk_size) {
        let count = chunk_size.min(n_quad - start);
        let mut table = vec![0.0; count * width];
        for (q, row) in table.chunks_mut(width).enumerate() {
            for j in 0..categories.len() {
                fill(start + q, j, &mut row[offsets[j]..offsets[j + 1]]);
            }
        }
        let evaluate = |(i, out): (usize, &mut [f64])| {
            let responses = responses.row(i);
            for (q, row) in table.chunks(width).enumerate() {
                let mut total = 0.0;
                for (j, &response) in responses.iter().enumerate() {
                    if response >= 0 {
                        let category = if binary {
                            usize::from(response == 1)
                        } else {
                            response as usize
                        };
                        total += row[offsets[j] + category];
                    }
                }
                out[start + q] = total;
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

#[cfg(test)]
mod tests {
    use super::{TABLE_ENTRIES, cached_likelihoods};
    use numpy::ndarray::array;

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
}
