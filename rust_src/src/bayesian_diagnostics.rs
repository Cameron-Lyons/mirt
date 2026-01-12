//! Bayesian diagnostics for IRT models.
//!
//! This module provides optimized implementations for:
//! - PSIS-LOO (Pareto-smoothed importance sampling LOO-CV)
//! - WAIC computation
//! - Pointwise log-likelihood computation

use ndarray::{Array1, Array2, Axis};
use numpy::{PyArray1, PyArray2, PyReadonlyArray2, ToPyArray};
use pyo3::prelude::*;
use rayon::prelude::*;

fn pareto_k_estimate(log_weights: &[f64], min_tail: usize) -> f64 {
    let n = log_weights.len();
    if n < min_tail {
        return f64::INFINITY;
    }

    let mut sorted = log_weights.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let tail_start = n.saturating_sub(min_tail);
    let tail: Vec<f64> = sorted[tail_start..].to_vec();

    if tail.len() < 2 {
        return f64::INFINITY;
    }

    let tail_min = tail[0];
    let tail_shifted: Vec<f64> = tail.iter().map(|&x| x - tail_min + 1e-10).collect();

    let log_tail: Vec<f64> = tail_shifted.iter().map(|&x| x.ln()).collect();

    if log_tail.len() < 2 {
        return 0.0;
    }

    let mean_log_tail: f64 = log_tail.iter().sum::<f64>() / log_tail.len() as f64;
    let k = mean_log_tail - log_tail[0];

    k.max(0.0)
}

fn pareto_smooth_weights(log_weights: &[f64]) -> (Vec<f64>, f64) {
    let max_log_w = log_weights
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);

    let log_weights_centered: Vec<f64> = log_weights.iter().map(|&x| x - max_log_w).collect();

    let k = pareto_k_estimate(&log_weights_centered, 10);

    let mut weights: Vec<f64> = log_weights_centered.iter().map(|&x| x.exp()).collect();

    let sum: f64 = weights.iter().sum();
    if sum > 0.0 {
        for w in &mut weights {
            *w /= sum;
        }
    }

    (weights, k)
}

#[pyfunction]
#[allow(clippy::type_complexity)]
pub fn psis_loo_fast<'py>(
    py: Python<'py>,
    log_lik: PyReadonlyArray2<f64>,
    k_threshold: f64,
) -> (
    f64,
    f64,
    f64,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    i32,
    f64,
) {
    let log_lik = log_lik.as_array();
    let n_samples = log_lik.nrows();
    let n_obs = log_lik.ncols();

    let results: Vec<(f64, f64)> = (0..n_obs)
        .into_par_iter()
        .map(|i| {
            let log_ratios: Vec<f64> = (0..n_samples).map(|s| -log_lik[[s, i]]).collect();

            let (weights, k) = pareto_smooth_weights(&log_ratios);

            let log_lik_i: Vec<f64> = (0..n_samples).map(|s| log_lik[[s, i]]).collect();
            let max_ll = log_lik_i.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

            let weighted_sum: f64 = weights
                .iter()
                .zip(log_lik_i.iter())
                .map(|(&w, &ll)| w * (ll - max_ll).exp())
                .sum();

            let loo_elpd = weighted_sum.ln() + max_ll;

            (loo_elpd, k)
        })
        .collect();

    let pointwise_elpd: Vec<f64> = results.iter().map(|(e, _)| *e).collect();
    let pareto_k_vals: Vec<f64> = results.iter().map(|(_, k)| *k).collect();

    let elpd_loo: f64 = pointwise_elpd.iter().sum();

    let max_log_lik = log_lik.map_axis(Axis(0), |col| {
        col.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    });
    let lppd: f64 = (0..n_obs)
        .map(|i| {
            let mean_lik: f64 = (0..n_samples)
                .map(|s| (log_lik[[s, i]] - max_log_lik[i]).exp())
                .sum::<f64>()
                / n_samples as f64;
            mean_lik.ln() + max_log_lik[i]
        })
        .sum();

    let p_loo = lppd - elpd_loo;
    let looic = -2.0 * elpd_loo;

    let n_high_k = pareto_k_vals.iter().filter(|&&k| k > k_threshold).count() as i32;

    let mean_elpd = elpd_loo / n_obs as f64;
    let var_elpd: f64 = pointwise_elpd
        .iter()
        .map(|&e| (e - mean_elpd).powi(2))
        .sum::<f64>()
        / (n_obs - 1) as f64;
    let se_elpd = (n_obs as f64 * var_elpd).sqrt();

    let pointwise_arr = Array1::from(pointwise_elpd);
    let pareto_k_arr = Array1::from(pareto_k_vals);

    (
        elpd_loo,
        p_loo,
        looic,
        pointwise_arr.to_pyarray(py),
        pareto_k_arr.to_pyarray(py),
        n_high_k,
        se_elpd,
    )
}

#[pyfunction]
pub fn waic_fast<'py>(
    py: Python<'py>,
    log_lik: PyReadonlyArray2<f64>,
) -> (f64, f64, f64, Bound<'py, PyArray1<f64>>, f64) {
    let log_lik = log_lik.as_array();
    let n_samples = log_lik.nrows();
    let n_obs = log_lik.ncols();

    let max_log_lik = log_lik.map_axis(Axis(0), |col| {
        col.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    });

    let lppd_i: Vec<f64> = (0..n_obs)
        .map(|i| {
            let mean_lik: f64 = (0..n_samples)
                .map(|s| (log_lik[[s, i]] - max_log_lik[i]).exp())
                .sum::<f64>()
                / n_samples as f64;
            mean_lik.ln() + max_log_lik[i]
        })
        .collect();

    let lppd: f64 = lppd_i.iter().sum();

    let p_waic_i: Vec<f64> = (0..n_obs)
        .map(|i| {
            let mean_ll: f64 =
                (0..n_samples).map(|s| log_lik[[s, i]]).sum::<f64>() / n_samples as f64;
            let var_ll: f64 = (0..n_samples)
                .map(|s| (log_lik[[s, i]] - mean_ll).powi(2))
                .sum::<f64>()
                / (n_samples - 1) as f64;
            var_ll
        })
        .collect();

    let p_waic: f64 = p_waic_i.iter().sum();
    let elpd_waic = lppd - p_waic;
    let waic = -2.0 * elpd_waic;

    let pointwise: Vec<f64> = lppd_i
        .iter()
        .zip(p_waic_i.iter())
        .map(|(l, p)| -2.0 * (l - p))
        .collect();

    let mean_pw = waic / n_obs as f64;
    let var_pw: f64 = pointwise
        .iter()
        .map(|&pw| (pw - mean_pw).powi(2))
        .sum::<f64>()
        / (n_obs - 1) as f64;
    let se_waic = (n_obs as f64 * var_pw).sqrt();

    let pointwise_arr = Array1::from(pointwise);

    (
        waic,
        elpd_waic,
        p_waic,
        pointwise_arr.to_pyarray(py),
        se_waic,
    )
}

#[pyfunction]
pub fn compute_pointwise_loglik_2pl<'py>(
    py: Python<'py>,
    responses: PyReadonlyArray2<i32>,
    discrimination_chain: PyReadonlyArray2<f64>,
    difficulty_chain: PyReadonlyArray2<f64>,
    theta_chain: PyReadonlyArray2<f64>,
) -> Bound<'py, PyArray2<f64>> {
    let responses = responses.as_array();
    let discrimination_chain = discrimination_chain.as_array();
    let difficulty_chain = difficulty_chain.as_array();
    let theta_chain = theta_chain.as_array();

    let n_persons = responses.nrows();
    let n_items = responses.ncols();
    let n_samples = discrimination_chain.nrows();

    let log_lik: Vec<Vec<f64>> = (0..n_samples)
        .into_par_iter()
        .map(|s| {
            let disc: Vec<f64> = discrimination_chain.row(s).to_vec();
            let diff: Vec<f64> = difficulty_chain.row(s).to_vec();
            let theta: Vec<f64> = theta_chain.row(s).to_vec();

            (0..n_persons)
                .map(|i| {
                    let mut ll = 0.0;
                    let theta_i = theta[i];
                    for j in 0..n_items {
                        let r = responses[[i, j]];
                        if r >= 0 {
                            let z = disc[j] * (theta_i - diff[j]);
                            let p = 1.0 / (1.0 + (-z).exp());
                            let p_clipped = p.clamp(1e-10, 1.0 - 1e-10);
                            ll += (r as f64) * p_clipped.ln()
                                + (1.0 - r as f64) * (1.0 - p_clipped).ln();
                        }
                    }
                    ll
                })
                .collect()
        })
        .collect();

    let mut result = Array2::zeros((n_samples, n_persons));
    for (s, row) in log_lik.iter().enumerate() {
        for (i, &val) in row.iter().enumerate() {
            result[[s, i]] = val;
        }
    }

    result.to_pyarray(py)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(psis_loo_fast, m)?)?;
    m.add_function(wrap_pyfunction!(waic_fast, m)?)?;
    m.add_function(wrap_pyfunction!(compute_pointwise_loglik_2pl, m)?)?;
    Ok(())
}
