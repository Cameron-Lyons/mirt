//! One native call for bounded MAP/ML optimization of binary response patterns.

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::utils::{EPSILON, sigmoid};

/// Bounded Brent minimization, using the public scorer's existing tolerances.
/// The parabolic steps also preserve its search behavior for non-concave
/// three/four-parameter likelihoods where a pure golden search can disagree.
fn bounded_minimum(f: impl Fn(f64) -> f64, mut lower: f64, mut upper: f64) -> f64 {
    let golden = (3.0 - 5.0_f64.sqrt()) / 2.0;
    let mut x = lower + golden * (upper - lower);
    let (mut w, mut v) = (x, x);
    let mut fx = f(x);
    let (mut fw, mut fv) = (fx, fx);
    let (mut step, mut previous_step): (f64, f64) = (0.0, 0.0);
    let sign = |value: f64| if value < 0.0 { -1.0 } else { 1.0 };
    for _ in 1..500 {
        let midpoint = (lower + upper) / 2.0;
        let tolerance = 2.2e-16_f64.sqrt() * x.abs() + 1e-5 / 3.0;
        if (x - midpoint).abs() <= 2.0 * tolerance - (upper - lower) / 2.0 {
            break;
        }
        let mut parabolic = false;
        if previous_step.abs() > tolerance {
            let r = (x - w) * (fx - fv);
            let q = (x - v) * (fx - fw);
            let mut numerator = (x - v) * q - (x - w) * r;
            let denominator = 2.0 * (q - r);
            if denominator > 0.0 {
                numerator = -numerator;
            }
            let denominator = denominator.abs();
            let old_step = previous_step;
            previous_step = step;
            if numerator.abs() < (0.5 * denominator * old_step).abs()
                && numerator > denominator * (lower - x)
                && numerator < denominator * (upper - x)
            {
                step = numerator / denominator;
                let candidate = x + step;
                if candidate - lower < 2.0 * tolerance || upper - candidate < 2.0 * tolerance {
                    step = tolerance * sign(midpoint - x);
                }
                parabolic = true;
            }
        }
        if !parabolic {
            previous_step = if x >= midpoint { lower - x } else { upper - x };
            step = golden * previous_step;
        }
        let candidate = x + sign(step) * step.abs().max(tolerance);
        let value = f(candidate);
        if value <= fx {
            if candidate >= x {
                lower = x;
            } else {
                upper = x;
            }
            v = w;
            fv = fw;
            w = x;
            fw = fx;
            x = candidate;
            fx = value;
        } else {
            if candidate < x {
                lower = candidate;
            } else {
                upper = candidate;
            }
            if value <= fw || w == x {
                v = w;
                fv = fw;
                w = candidate;
                fw = value;
            } else if value <= fv || v == x || v == w {
                v = candidate;
                fv = value;
            }
        }
    }
    x
}

type Scores<'py> = (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>);

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn compute_optimized_scores<'py>(
    py: Python<'py>,
    responses: PyReadonlyArray2<i32>,
    parameters: PyReadonlyArray2<f64>,
    lower: f64,
    upper: f64,
    map: bool,
    prior_mean: f64,
    prior_var: f64,
    n_jobs: usize,
) -> PyResult<Scores<'py>> {
    let responses = responses.as_array();
    let parameters = parameters.as_array();
    if parameters.dim() != (responses.ncols(), 4) || parameters.iter().any(|v| !v.is_finite()) {
        return Err(PyValueError::new_err(
            "parameters must be a finite item-by-four matrix",
        ));
    }
    if !lower.is_finite() || !upper.is_finite() || lower >= upper || n_jobs == 0 {
        return Err(PyValueError::new_err(
            "invalid scoring bounds or worker count",
        ));
    }
    if map && (!prior_mean.is_finite() || !prior_var.is_finite() || prior_var <= 0.0) {
        return Err(PyValueError::new_err(
            "prior mean must be finite and variance positive",
        ));
    }
    if responses.iter().any(|&r| r > 1) {
        return Err(PyValueError::new_err(
            "responses must be binary or negative missing codes",
        ));
    }
    for row in parameters.rows() {
        if row[2] < 0.0 || row[3] > 1.0 || row[2] > row[3] {
            return Err(PyValueError::new_err("invalid probability asymptotes"));
        }
    }
    let score = |i: usize| {
        let response = responses.row(i);
        let observed = response.iter().filter(|&&r| r >= 0).count();
        let correct = response.iter().filter(|&&r| r == 1).count();
        if !map {
            if observed == 0 {
                return (0.0, f64::INFINITY);
            }
            if correct == 0 {
                return (lower, f64::INFINITY);
            }
            if correct == observed {
                return (upper, f64::INFINITY);
            }
        }
        let objective = |theta: f64| {
            let mut value = 0.0;
            for (j, &r) in response.iter().enumerate() {
                if r < 0 {
                    continue;
                }
                let p = parameters.row(j);
                let prob = (p[2] + (p[3] - p[2]) * sigmoid(p[0] * (theta - p[1])))
                    .clamp(EPSILON, 1.0 - EPSILON);
                value -= if r == 1 { prob.ln() } else { (1.0 - prob).ln() };
            }
            if map {
                value += 0.5 * (theta - prior_mean).powi(2) / prior_var;
            }
            value
        };
        let theta = bounded_minimum(objective, lower, upper);
        let mut information = 0.0;
        if !map {
            for (j, &r) in response.iter().enumerate() {
                if r < 0 {
                    continue;
                }
                let p = parameters.row(j);
                let logistic = sigmoid(p[0] * (theta - p[1]));
                let prob = p[2] + (p[3] - p[2]) * logistic;
                let denominator = prob * (1.0 - prob);
                if denominator > 0.0 {
                    let derivative = p[0] * (p[3] - p[2]) * logistic * (1.0 - logistic);
                    information += derivative * derivative / denominator;
                }
            }
        }
        if map || information <= 0.0 {
            information = (objective(theta + 1e-5) - 2.0 * objective(theta)
                + objective(theta - 1e-5))
                / 1e-10;
        }
        (
            theta,
            if information > 0.0 {
                1.0 / information.sqrt()
            } else {
                f64::NAN
            },
        )
    };
    let results = py.detach(|| -> PyResult<Vec<(f64, f64)>> {
        if n_jobs == 1 || responses.nrows() < 2 {
            Ok((0..responses.nrows()).map(score).collect())
        } else {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(n_jobs.min(responses.nrows()))
                .build()
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            Ok(pool.install(|| (0..responses.nrows()).into_par_iter().map(score).collect()))
        }
    })?;
    let (theta, se): (Vec<_>, Vec<_>) = results.into_iter().unzip();
    Ok((theta.into_pyarray(py), se.into_pyarray(py)))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_optimized_scores, m)?)?;
    Ok(())
}
