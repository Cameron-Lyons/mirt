//! Bounded probability tables shared by respondents at a common theta grid.

use numpy::ndarray::{Array2, ArrayView2};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadwriteArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
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

/// Rows of joint mass, scaled by the largest prior mass, below this are
/// normalized again from their own maximum (`_posterior._MIN_SCALED_TOTAL`).
const MIN_SCALED_TOTAL: f64 = 1e-100;

/// Maximum of a row with NumPy's semantics: NaN when any entry is NaN.
fn row_maximum(row: &[f64]) -> f64 {
    let mut maximum = f64::NEG_INFINITY;
    for &value in row {
        if value.is_nan() {
            return f64::NAN;
        }
        maximum = maximum.max(value);
    }
    maximum
}

/// Normalize log-joint values in place from their maximum; return the log
/// total, as `_posterior._normalize_rows` does for one row.
fn normalize_from_maximum(row: &mut [f64]) -> f64 {
    let offset = row_maximum(row);
    let mut total = 0.0;
    for value in row.iter_mut() {
        *value = (*value - offset).exp();
        total += *value;
    }
    for value in row.iter_mut() {
        *value /= total;
    }
    if offset.is_infinite() {
        offset
    } else {
        offset + total.ln()
    }
}

/// Prior of [`normalize_posterior_row`]: shifted by its finite peak, or raw
/// when the peak is not finite.
enum RowPrior<'a> {
    None,
    Shifted { values: &'a [f64], peak: f64 },
    Raw(&'a [f64]),
}

/// One row of `mirt.estimation._posterior.normalize_log_posterior`, with the
/// same operations and therefore the same underflow and non-finite semantics:
/// the likelihood is centered at its maximum, the prior added after shifting
/// it by its largest mass, and rows whose mass underflows renormalized from
/// their own maximum. `scratch` holds a copy of the row of that length.
fn normalize_posterior_row(row: &mut [f64], prior: &RowPrior<'_>, scratch: &mut [f64]) -> f64 {
    let offset = row_maximum(row);
    let shift = if offset.is_finite() { offset } else { 0.0 };
    for value in row.iter_mut() {
        *value -= shift;
    }
    match *prior {
        RowPrior::None => {
            let mut total = 0.0;
            for value in row.iter_mut() {
                *value = value.exp();
                total += *value;
            }
            for value in row.iter_mut() {
                *value /= total;
            }
            shift + total.ln()
        }
        RowPrior::Raw(values) => {
            for (value, &mass) in row.iter_mut().zip(values) {
                *value += mass;
            }
            shift + normalize_from_maximum(row)
        }
        RowPrior::Shifted { values, peak } => {
            for (value, &mass) in row.iter_mut().zip(values) {
                *value += mass;
            }
            scratch.copy_from_slice(row);
            let mut total = 0.0;
            for value in row.iter_mut() {
                *value = value.exp();
                total += *value;
            }
            if total.is_finite() && total >= MIN_SCALED_TOTAL {
                let scale = 1.0 / total;
                for value in row.iter_mut() {
                    *value *= scale;
                }
                shift + (peak + total.ln())
            } else {
                row.copy_from_slice(scratch);
                shift + (peak + normalize_from_maximum(row))
            }
        }
    }
}

/// Normalize an owned person-by-grid log-likelihood buffer to posterior
/// weights in place and return each row's log marginal likelihood.
///
/// `log_joint` must be a writeable C-contiguous float64 matrix and
/// `log_prior` the log prior mass of each grid point, if any. The rows are
/// normalized as `mirt.estimation._posterior.normalize_log_posterior`
/// normalizes them, in parallel with the global GIL released.
#[pyfunction]
#[pyo3(signature = (log_joint, log_prior=None))]
pub fn normalize_log_posterior_inplace<'py>(
    py: Python<'py>,
    mut log_joint: PyReadwriteArray2<'py, f64>,
    log_prior: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let n_points = log_joint.as_array().ncols();
    let prior = log_prior.map(|values| values.as_array().to_vec());
    if prior
        .as_ref()
        .is_some_and(|values| values.len() != n_points)
    {
        return Err(PyValueError::new_err(
            "log_prior must have one entry per grid point",
        ));
    }
    if n_points == 0 {
        return Err(PyValueError::new_err("log_joint must have grid points"));
    }
    let data = log_joint
        .as_slice_mut()
        .map_err(|_| PyValueError::new_err("log_joint must be C-contiguous"))?;
    let marginal = py.detach(|| {
        let peak = prior.as_deref().map_or(0.0, row_maximum);
        let shifted: Option<Vec<f64>> = prior
            .as_deref()
            .filter(|_| peak.is_finite())
            .map(|values| values.iter().map(|mass| mass - peak).collect());
        let row_prior = match (&prior, &shifted) {
            (None, _) => RowPrior::None,
            (Some(_), Some(values)) => RowPrior::Shifted { values, peak },
            (Some(values), None) => RowPrior::Raw(values),
        };
        let mut marginal = vec![0.0; data.len() / n_points];
        let normalize = |scratch: &mut Vec<f64>, (row, marginal): (&mut [f64], &mut f64)| {
            *marginal = normalize_posterior_row(row, &row_prior, scratch);
        };
        if data.len() < 32_768 {
            let mut scratch = vec![0.0; n_points];
            data.chunks_mut(n_points)
                .zip(marginal.iter_mut())
                .for_each(|pair| normalize(&mut scratch, pair));
        } else {
            data.par_chunks_mut(n_points)
                .zip(marginal.par_iter_mut())
                .for_each_init(|| vec![0.0; n_points], normalize);
        }
        marginal
    });
    Ok(marginal.into_pyarray(py))
}

/// Register the posterior normalization kernel with the Python module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(normalize_log_posterior_inplace, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{RowPrior, TABLE_ENTRIES, cached_likelihoods, normalize_posterior_row};
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

    #[test]
    fn posterior_rows_match_the_reference_normalization() {
        let prior = [0.0, -300.0, -900.0, -906.0, f64::NEG_INFINITY];
        let row_prior = RowPrior::Shifted {
            values: &prior,
            peak: 0.0,
        };
        // Peaks where the prior mass is e^-900 times smaller underflow after
        // the shift by the largest prior mass, so the row is renormalized.
        let mut scratch = [0.0; 5];
        let mut row = [-2000.0, -2000.0, 0.0, -1.0, 0.0];
        let marginal = normalize_posterior_row(&mut row, &row_prior, &mut scratch);
        let joint: Vec<f64> = [-2000.0, -2000.0, 0.0, -1.0, 0.0]
            .iter()
            .zip(&prior)
            .map(|(l, p)| l + p)
            .collect();
        let peak = joint.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let total: f64 = joint.iter().map(|value| (value - peak).exp()).sum();
        assert!((marginal - (peak + total.ln())).abs() < 1e-12);
        for (value, joint) in row.iter().zip(&joint) {
            assert!((value - (joint - peak).exp() / total).abs() < 1e-15);
        }
        assert!(row[3] > 0.0);

        let mut empty = [f64::NEG_INFINITY; 2];
        let raw = RowPrior::Raw(&[0.0, 0.0]);
        let marginal = normalize_posterior_row(&mut empty, &raw, &mut scratch[..2]);
        assert_eq!(marginal, f64::NEG_INFINITY);
        assert!(empty.iter().all(|value| value.is_nan()));
        let mut undefined = [f64::NAN, 0.0];
        let marginal = normalize_posterior_row(&mut undefined, &RowPrior::None, &mut scratch[..2]);
        assert!(marginal.is_nan());
    }
}
