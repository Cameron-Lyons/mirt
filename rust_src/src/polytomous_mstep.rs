//! Batched item-local polytomous optimization with analytic derivatives.

use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::utils::sigmoid;

/// Negative expected log likelihood and its gradient in (a, b_1, ...).
fn objective(
    x: &[f64],
    points: &[f64],
    counts: &[f64],
    grm: bool,
    epsilon: f64,
) -> (f64, Vec<f64>) {
    let k = x.len();
    let a = x[0];
    let mut value = 0.0;
    let mut gradient = vec![0.0; k];
    let mut probabilities = vec![0.0; k];
    let mut cumulative = vec![0.0; k - 1];
    let mut slope_derivatives = vec![0.0; k];
    let mut effective = vec![0.0; k];
    for (q, &theta) in points.iter().enumerate() {
        if grm {
            for t in 0..k - 1 {
                cumulative[t] = sigmoid(a * (theta - x[t + 1]));
            }
            probabilities[0] = 1.0 - cumulative[0];
            for c in 1..k - 1 {
                probabilities[c] = cumulative[c - 1] - cumulative[c];
            }
            probabilities[k - 1] = cumulative[k - 2];
        } else {
            probabilities[0] = 0.0;
            for c in 1..k {
                slope_derivatives[c] = slope_derivatives[c - 1] + theta - x[c];
                probabilities[c] = a * slope_derivatives[c];
            }
            let max = probabilities
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);
            let mut total = 0.0;
            for p in &mut probabilities {
                *p = (*p - max).exp();
                total += *p;
            }
            for p in &mut probabilities {
                *p /= total;
            }
        }
        for c in 0..k {
            let p = probabilities[c];
            let count = counts[q * k + c];
            value -= count * p.clamp(epsilon, 1.0 - epsilon).ln();
            effective[c] = if p > epsilon && p < 1.0 - epsilon {
                count
            } else {
                0.0
            };
        }
        if grm {
            for t in 0..k - 1 {
                let left = if effective[t] == 0.0 {
                    0.0
                } else {
                    effective[t] / probabilities[t]
                };
                let right = if effective[t + 1] == 0.0 {
                    0.0
                } else {
                    effective[t + 1] / probabilities[t + 1]
                };
                let derivative = (left - right) * cumulative[t] * (1.0 - cumulative[t]);
                gradient[0] += derivative * (theta - x[t + 1]);
                gradient[t + 1] -= derivative * a;
            }
        } else {
            let total: f64 = effective.iter().sum();
            let mean: f64 = probabilities
                .iter()
                .zip(&slope_derivatives)
                .map(|(p, d)| p * d)
                .sum();
            for c in 0..k {
                gradient[0] -= effective[c] * (slope_derivatives[c] - mean);
            }
            let mut tail_p = 0.0;
            let mut tail_r = 0.0;
            for c in (1..k).rev() {
                tail_p += probabilities[c];
                tail_r += effective[c];
                gradient[c] += a * (tail_r - total * tail_p);
            }
        }
    }
    (value, gradient)
}

fn bounds(index: usize) -> (f64, f64) {
    if index == 0 { (0.1, 5.0) } else { (-6.0, 6.0) }
}

/// Projected inverse-BFGS with Armijo backtracking; only improving steps commit.
#[allow(clippy::too_many_arguments)]
fn optimize(
    mut x: Vec<f64>,
    free: &[bool],
    points: &[f64],
    counts: &[f64],
    grm: bool,
    epsilon: f64,
    max_iter: usize,
    ftol: f64,
) -> Vec<f64> {
    let k = x.len();
    let (mut value, mut gradient) = objective(&x, points, counts, grm, epsilon);
    let mut inverse = vec![0.0; k * k];
    for j in 0..k {
        inverse[j * k + j] = 1.0;
    }
    let mut previous_active = vec![true; k];
    for _ in 0..max_iter {
        let mut projected = gradient.clone();
        let mut active = free.to_vec();
        for j in 0..k {
            let (lo, hi) = bounds(j);
            if !free[j] || (x[j] <= lo && gradient[j] > 0.0) || (x[j] >= hi && gradient[j] < 0.0) {
                projected[j] = 0.0;
                active[j] = false;
            }
        }
        if active != previous_active {
            inverse.fill(0.0);
            for j in 0..k {
                inverse[j * k + j] = 1.0;
            }
            previous_active.clone_from(&active);
        }
        if projected.iter().all(|g| g.abs() <= 1e-5) {
            break;
        }
        let mut direction: Vec<f64> = (0..k)
            .map(|i| {
                -(0..k)
                    .map(|j| inverse[i * k + j] * projected[j])
                    .sum::<f64>()
            })
            .collect();
        for j in 0..k {
            if !active[j] {
                direction[j] = 0.0;
            }
        }
        let mut accepted = None;
        // A projected quasi-Newton direction can cease to descend at a bound.
        // Retry with the projected gradient before declaring a line-search stop.
        for attempt in 0..2 {
            if attempt == 1 {
                direction = projected.iter().map(|g| -g).collect();
            }
            let mut step = 1.0;
            for _ in 0..40 {
                let candidate: Vec<f64> = (0..k)
                    .map(|j| {
                        if free[j] {
                            let (lo, hi) = bounds(j);
                            (x[j] + step * direction[j]).clamp(lo, hi)
                        } else {
                            x[j]
                        }
                    })
                    .collect();
                let derivative: f64 = (0..k).map(|j| gradient[j] * (candidate[j] - x[j])).sum();
                if derivative < 0.0 {
                    let (next, grad) = objective(&candidate, points, counts, grm, epsilon);
                    if next.is_finite() && next <= value + 1e-4 * derivative {
                        accepted = Some((candidate, next, grad));
                        break;
                    }
                }
                step *= 0.5;
            }
            if accepted.is_some() {
                break;
            }
        }
        let Some((candidate, next, grad)) = accepted else {
            break;
        };
        let delta: Vec<f64> = (0..k).map(|j| candidate[j] - x[j]).collect();
        let change: Vec<f64> = (0..k)
            .map(|j| {
                if active[j] {
                    grad[j] - gradient[j]
                } else {
                    0.0
                }
            })
            .collect();
        let curvature: f64 = delta.iter().zip(&change).map(|(s, y)| s * y).sum();
        if curvature > 1e-12 {
            let hy: Vec<f64> = (0..k)
                .map(|i| (0..k).map(|j| inverse[i * k + j] * change[j]).sum())
                .collect();
            let yhy: f64 = change.iter().zip(&hy).map(|(y, h)| y * h).sum();
            for i in 0..k {
                for j in 0..k {
                    inverse[i * k + j] += (1.0 + yhy / curvature) * delta[i] * delta[j] / curvature
                        - (hy[i] * delta[j] + delta[i] * hy[j]) / curvature;
                }
            }
        }
        let converged = value - next <= ftol * value.abs().max(next.abs()).max(1.0);
        x = candidate;
        value = next;
        gradient = grad;
        if converged {
            break;
        }
    }
    x
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn m_step_polytomous<'py>(
    py: Python<'py>,
    responses: PyReadonlyArray2<i32>,
    posterior: PyReadonlyArray2<f64>,
    points: PyReadonlyArray1<f64>,
    parameters: PyReadonlyArray2<f64>,
    free: PyReadonlyArray2<bool>,
    categories: PyReadonlyArray1<i32>,
    grm: bool,
    max_iter: usize,
    ftol: f64,
    epsilon: f64,
    n_jobs: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let responses = responses.as_array();
    let posterior = posterior.as_array();
    let points = points.as_array().to_vec();
    let parameters = parameters.as_array();
    let free = free.as_array();
    let categories = categories.as_array();
    let items = responses.ncols();
    if posterior.dim() != (responses.nrows(), points.len())
        || parameters.nrows() != items
        || free.dim() != parameters.dim()
        || categories.len() != items
        || categories
            .iter()
            .any(|&k| k < 2 || k as usize > parameters.ncols())
    {
        return Err(PyValueError::new_err(
            "incompatible response, posterior, or parameter shapes",
        ));
    }
    if max_iter == 0
        || n_jobs == 0
        || !ftol.is_finite()
        || ftol <= 0.0
        || !epsilon.is_finite()
        || epsilon <= 0.0
        || epsilon >= 0.5
        || points.iter().any(|v| !v.is_finite())
        || posterior.iter().any(|&v| !v.is_finite() || v < 0.0)
        || parameters.iter().any(|v| !v.is_finite())
    {
        return Err(PyValueError::new_err(
            "invalid optimizer controls or non-finite inputs",
        ));
    }
    for j in 0..items {
        if responses.column(j).iter().any(|&r| r >= categories[j]) {
            return Err(PyValueError::new_err("response category out of range"));
        }
        for c in 0..categories[j] as usize {
            let (lo, hi) = bounds(c);
            if free[[j, c]] && !(lo..=hi).contains(&parameters[[j, c]]) {
                return Err(PyValueError::new_err(
                    "free parameters must lie within optimizer bounds",
                ));
            }
        }
    }
    let fit_item = |j: usize| {
        let k = categories[j] as usize;
        let mut counts = vec![0.0; points.len() * k];
        for i in 0..responses.nrows() {
            let r = responses[[i, j]];
            if r >= 0 {
                for q in 0..points.len() {
                    counts[q * k + r as usize] += posterior[[i, q]];
                }
            }
        }
        let mut params = parameters.row(j).to_vec();
        if counts.iter().any(|&v| v > 0.0) {
            let result = optimize(
                params[..k].to_vec(),
                &free.row(j).to_vec()[..k],
                &points,
                &counts,
                grm,
                epsilon,
                max_iter,
                ftol,
            );
            params[..k].copy_from_slice(&result);
        }
        params
    };
    let results = py.detach(|| -> PyResult<Vec<Vec<f64>>> {
        if n_jobs == 1 || items < 2 {
            Ok((0..items).map(fit_item).collect())
        } else {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(n_jobs.min(items))
                .build()
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            Ok(pool.install(|| (0..items).into_par_iter().map(fit_item).collect()))
        }
    })?;
    let output = Array2::from_shape_vec(parameters.dim(), results.into_iter().flatten().collect())
        .expect("validated shape");
    Ok(output.into_pyarray(py))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(m_step_polytomous, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::objective;
    #[test]
    fn analytic_gradients_match_independent_differences() {
        let points = [-2.0, -0.2, 0.8, 2.0];
        let counts: Vec<f64> = (0..16).map(|i| (i % 5 + 1) as f64).collect();
        let x = [1.3, -1.0, 0.2, 1.5];
        for grm in [false, true] {
            let (_, gradient) = objective(&x, &points, &counts, grm, 1e-10);
            for j in 0..x.len() {
                let mut plus = x;
                let mut minus = x;
                plus[j] += 1e-5;
                minus[j] -= 1e-5;
                let expected = (objective(&plus, &points, &counts, grm, 1e-10).0
                    - objective(&minus, &points, &counts, grm, 1e-10).0)
                    / 2e-5;
                assert!(
                    (expected - gradient[j]).abs() < 1e-6,
                    "{grm} {j}: {expected} != {}",
                    gradient[j]
                );
            }
        }
    }
}
