//! Lord-Wingersky recursion for observed-score distributions used in equating.

use numpy::ndarray::Array2;
use numpy::{PyArray2, PyReadonlyArray1, ToPyArray};
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::utils::sigmoid;

#[inline]
fn item_logit(theta: f64, discrimination: f64, difficulty: f64) -> f64 {
    let logit = discrimination * (theta - difficulty);
    if !logit.is_finite()
        && theta.is_finite()
        && difficulty.is_finite()
        && discrimination.abs() <= 1.0
    {
        // Finite inputs can have an overflowing difference while the scaled
        // logit is representable. Match the NumPy model's exceptional path,
        // retaining the ordinary centered expression for all other inputs.
        discrimination * theta - discrimination * difficulty
    } else {
        logit
    }
}

/// Lord-Wingersky recursion for observed score distribution.
///
/// Computes P(X = x | theta) for all possible sum scores x and all theta values.
/// Returns matrix of shape (n_theta, n_items + 1).
#[pyfunction]
pub fn lord_wingersky_recursion<'py>(
    py: Python<'py>,
    disc: PyReadonlyArray1<f64>,
    diff: PyReadonlyArray1<f64>,
    theta_grid: PyReadonlyArray1<f64>,
) -> Bound<'py, PyArray2<f64>> {
    let disc = disc.as_array();
    let diff = diff.as_array();
    let theta_grid = theta_grid.as_array();

    let n_items = disc.len();
    let n_theta = theta_grid.len();
    let max_score = n_items;

    let results: Vec<Vec<f64>> = (0..n_theta)
        .into_par_iter()
        .map(|q| {
            let theta = theta_grid[q];

            let probs: Vec<f64> = (0..n_items)
                .map(|j| sigmoid(item_logit(theta, disc[j], diff[j])))
                .collect();

            let mut f_prev = vec![0.0; max_score + 1];
            f_prev[0] = 1.0;

            for (j, &p_j) in probs.iter().enumerate() {
                let mut f_curr = vec![0.0; max_score + 1];
                let q_j = 1.0 - p_j;

                for x in 0..=(j + 1) {
                    if x == 0 {
                        f_curr[x] = f_prev[x] * q_j;
                    } else if x == j + 1 {
                        f_curr[x] = f_prev[x - 1] * p_j;
                    } else {
                        f_curr[x] = f_prev[x] * q_j + f_prev[x - 1] * p_j;
                    }
                }
                f_prev = f_curr;
            }

            f_prev
        })
        .collect();

    let mut arr = Array2::zeros((n_theta, max_score + 1));
    for (q, row) in results.into_iter().enumerate() {
        for (x, val) in row.into_iter().enumerate() {
            arr[[q, x]] = val;
        }
    }

    arr.to_pyarray(py)
}

/// Register equating functions with the Python module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(lord_wingersky_recursion, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::item_logit;

    #[test]
    fn item_logit_preserves_ordinary_centered_expression() {
        for (theta, discrimination, difficulty) in
            [(0.3, 1.7, -0.8), (-4.0, 0.4, 1.3), (2.0, 3.2, 2.0)]
        {
            assert_eq!(
                item_logit(theta, discrimination, difficulty),
                discrimination * (theta - difficulty)
            );
        }
    }

    #[test]
    fn item_logit_recovers_finite_values_after_centering_overflows() {
        assert!((item_logit(1e308, 1e-308, -1e308) - 2.0).abs() < 1e-15);
        assert!((item_logit(-1e308, 5e-309, 1e308) + 1.0).abs() < 1e-15);
        assert_eq!(item_logit(1e308, 0.0, -1e308), 0.0);
        assert_eq!(item_logit(1e308, 2.0, -1e308), f64::INFINITY);
        assert_eq!(item_logit(-1e308, 2.0, 1e308), f64::NEG_INFINITY);
    }
}
