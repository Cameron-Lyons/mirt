//! Computerized Adaptive Testing (CAT) functions.

use ndarray::Array1;
use numpy::{PyArray1, PyReadonlyArray1, ToPyArray};
use pyo3::prelude::*;
use rand::prelude::*;
use rand_pcg::Pcg64;
use rayon::prelude::*;

use crate::utils::{
    compute_eap_with_se, fisher_info_2pl_items, log_sigmoid, normalize_log_posterior, sigmoid,
    LOG_2_PI,
};

/// Compute Fisher information for all items at a given theta
#[pyfunction]
pub fn cat_compute_item_info<'py>(
    py: Python<'py>,
    theta: f64,
    discrimination: PyReadonlyArray1<f64>,
    difficulty: PyReadonlyArray1<f64>,
) -> Bound<'py, PyArray1<f64>> {
    let disc = discrimination.as_array().to_vec();
    let diff = difficulty.as_array().to_vec();
    let info = fisher_info_2pl_items(theta, &disc, &diff);
    Array1::from_vec(info).to_pyarray(py)
}

/// Select item with maximum Fisher information from available items
#[pyfunction]
pub fn cat_select_max_info(
    theta: f64,
    discrimination: PyReadonlyArray1<f64>,
    difficulty: PyReadonlyArray1<f64>,
    available_mask: PyReadonlyArray1<bool>,
) -> i32 {
    let disc = discrimination.as_array();
    let diff = difficulty.as_array();
    let available = available_mask.as_array();
    let n_items = disc.len();

    let mut best_item: i32 = -1;
    let mut best_info: f64 = f64::NEG_INFINITY;

    for j in 0..n_items {
        if !available[j] {
            continue;
        }
        let z = disc[j] * (theta - diff[j]);
        let p = sigmoid(z);
        let q = 1.0 - p;
        let info = disc[j] * disc[j] * p * q;

        if info > best_info {
            best_info = info;
            best_item = j as i32;
        }
    }

    best_item
}

/// Incremental EAP update after a single response
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn cat_eap_update<'py>(
    py: Python<'py>,
    administered_items: PyReadonlyArray1<i32>,
    responses: PyReadonlyArray1<i32>,
    discrimination: PyReadonlyArray1<f64>,
    difficulty: PyReadonlyArray1<f64>,
    quad_points: PyReadonlyArray1<f64>,
    quad_weights: PyReadonlyArray1<f64>,
) -> (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>) {
    let items = administered_items.as_array();
    let resp = responses.as_array();
    let disc = discrimination.as_array();
    let diff = difficulty.as_array();
    let nodes = quad_points.as_array();
    let weights = quad_weights.as_array();

    let n_quad = nodes.len();
    let n_administered = items.len();

    let mut log_likes = vec![0.0; n_quad];
    for q in 0..n_quad {
        let theta = nodes[q];
        let mut ll = 0.0;
        for i in 0..n_administered {
            let j = items[i] as usize;
            let r = resp[i];
            if r >= 0 {
                let z = disc[j] * (theta - diff[j]);
                if r == 1 {
                    ll += log_sigmoid(z);
                } else {
                    ll += log_sigmoid(-z);
                }
            }
        }
        log_likes[q] = ll;
    }

    let log_posterior: Vec<f64> = (0..n_quad)
        .map(|q| {
            let log_prior = -0.5 * nodes[q] * nodes[q] - 0.5 * LOG_2_PI;
            log_likes[q] + log_prior + weights[q].ln()
        })
        .collect();

    let posterior = normalize_log_posterior(&log_posterior);
    let nodes_vec: Vec<f64> = nodes.to_vec();
    let (theta_eap, se) = compute_eap_with_se(&posterior, &nodes_vec);

    (
        Array1::from_vec(vec![theta_eap]).to_pyarray(py),
        Array1::from_vec(vec![se]).to_pyarray(py),
    )
}

/// Simulate a single CAT session and return results
#[inline]
#[allow(clippy::too_many_arguments)]
fn cat_simulate_single(
    true_theta: f64,
    discrimination: &[f64],
    difficulty: &[f64],
    quad_points: &[f64],
    quad_weights: &[f64],
    se_threshold: f64,
    max_items: usize,
    min_items: usize,
    rng: &mut Pcg64,
) -> (f64, f64, usize, Vec<i32>, Vec<i32>) {
    let n_items = discrimination.len();
    let n_quad = quad_points.len();

    let mut available: Vec<bool> = vec![true; n_items];
    let mut administered: Vec<i32> = Vec::with_capacity(max_items);
    let mut responses: Vec<i32> = Vec::with_capacity(max_items);

    let mut current_theta = 0.0;
    let mut current_se = f64::INFINITY;

    for step in 0..max_items {
        let mut best_item: i32 = -1;
        let mut best_info: f64 = f64::NEG_INFINITY;

        for j in 0..n_items {
            if !available[j] {
                continue;
            }
            let z = discrimination[j] * (current_theta - difficulty[j]);
            let p = sigmoid(z);
            let q = 1.0 - p;
            let info = discrimination[j] * discrimination[j] * p * q;

            if info > best_info {
                best_info = info;
                best_item = j as i32;
            }
        }

        if best_item < 0 {
            break;
        }

        let item_idx = best_item as usize;
        available[item_idx] = false;

        let z = discrimination[item_idx] * (true_theta - difficulty[item_idx]);
        let p = sigmoid(z);
        let response = if rng.random::<f64>() < p { 1 } else { 0 };

        administered.push(best_item);
        responses.push(response);

        let mut log_likes = vec![0.0; n_quad];
        for q in 0..n_quad {
            let theta = quad_points[q];
            let mut ll = 0.0;
            for (i, &item) in administered.iter().enumerate() {
                let j = item as usize;
                let r = responses[i];
                let z = discrimination[j] * (theta - difficulty[j]);
                if r == 1 {
                    ll += log_sigmoid(z);
                } else {
                    ll += log_sigmoid(-z);
                }
            }
            log_likes[q] = ll;
        }

        let log_posterior: Vec<f64> = (0..n_quad)
            .map(|q| {
                let log_prior = -0.5 * quad_points[q] * quad_points[q];
                log_likes[q] + log_prior + quad_weights[q].ln()
            })
            .collect();

        let posterior = normalize_log_posterior(&log_posterior);
        (current_theta, current_se) = compute_eap_with_se(&posterior, quad_points);

        if step + 1 >= min_items && current_se <= se_threshold {
            break;
        }
    }

    let n_administered = administered.len();
    (
        current_theta,
        current_se,
        n_administered,
        administered,
        responses,
    )
}

/// Run batch CAT simulations in parallel
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::type_complexity)]
pub fn cat_simulate_batch<'py>(
    py: Python<'py>,
    true_thetas: PyReadonlyArray1<f64>,
    discrimination: PyReadonlyArray1<f64>,
    difficulty: PyReadonlyArray1<f64>,
    quad_points: PyReadonlyArray1<f64>,
    quad_weights: PyReadonlyArray1<f64>,
    se_threshold: f64,
    max_items: i32,
    min_items: i32,
    n_replications: i32,
    seed: u64,
) -> (
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<i32>>,
    Bound<'py, PyArray1<f64>>,
) {
    let thetas = true_thetas.as_array().to_vec();
    let disc = discrimination.as_array().to_vec();
    let diff = difficulty.as_array().to_vec();
    let nodes = quad_points.as_array().to_vec();
    let weights = quad_weights.as_array().to_vec();

    let n_thetas = thetas.len();
    let n_total = n_thetas * n_replications as usize;

    let tasks: Vec<(usize, usize)> = (0..n_thetas)
        .flat_map(|t| (0..n_replications as usize).map(move |r| (t, r)))
        .collect();

    let results: Vec<(f64, f64, usize, f64)> = tasks
        .par_iter()
        .map(|(theta_idx, rep)| {
            let task_seed = seed
                .wrapping_add(*theta_idx as u64 * 1000)
                .wrapping_add(*rep as u64);
            let mut rng = Pcg64::seed_from_u64(task_seed);

            let true_theta = thetas[*theta_idx];
            let (est_theta, est_se, n_items, _, _) = cat_simulate_single(
                true_theta,
                &disc,
                &diff,
                &nodes,
                &weights,
                se_threshold,
                max_items as usize,
                min_items as usize,
                &mut rng,
            );

            (est_theta, est_se, n_items, true_theta)
        })
        .collect();

    let mut theta_est = Array1::zeros(n_total);
    let mut se_est = Array1::zeros(n_total);
    let mut n_items = Array1::zeros(n_total);
    let mut true_theta_out = Array1::zeros(n_total);

    for (i, (t, s, n, tt)) in results.into_iter().enumerate() {
        theta_est[i] = t;
        se_est[i] = s;
        n_items[i] = n as i32;
        true_theta_out[i] = tt;
    }

    (
        theta_est.to_pyarray(py),
        se_est.to_pyarray(py),
        n_items.to_pyarray(py),
        true_theta_out.to_pyarray(py),
    )
}

/// Compute conditional MSE at specified theta values
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::type_complexity)]
pub fn cat_conditional_mse<'py>(
    py: Python<'py>,
    eval_thetas: PyReadonlyArray1<f64>,
    discrimination: PyReadonlyArray1<f64>,
    difficulty: PyReadonlyArray1<f64>,
    quad_points: PyReadonlyArray1<f64>,
    quad_weights: PyReadonlyArray1<f64>,
    se_threshold: f64,
    max_items: i32,
    min_items: i32,
    n_replications: i32,
    seed: u64,
) -> (
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
) {
    let thetas = eval_thetas.as_array().to_vec();
    let disc = discrimination.as_array().to_vec();
    let diff = difficulty.as_array().to_vec();
    let nodes = quad_points.as_array().to_vec();
    let weights = quad_weights.as_array().to_vec();

    let n_thetas = thetas.len();
    let n_reps = n_replications as usize;

    let stats: Vec<(f64, f64, f64)> = thetas
        .par_iter()
        .enumerate()
        .map(|(t_idx, &true_theta)| {
            let mut estimates = Vec::with_capacity(n_reps);
            let mut n_items_sum = 0.0;

            for rep in 0..n_reps {
                let task_seed = seed
                    .wrapping_add(t_idx as u64 * 10000)
                    .wrapping_add(rep as u64);
                let mut rng = Pcg64::seed_from_u64(task_seed);

                let (est_theta, _, n_items, _, _) = cat_simulate_single(
                    true_theta,
                    &disc,
                    &diff,
                    &nodes,
                    &weights,
                    se_threshold,
                    max_items as usize,
                    min_items as usize,
                    &mut rng,
                );

                estimates.push(est_theta);
                n_items_sum += n_items as f64;
            }

            let mean_est: f64 = estimates.iter().sum::<f64>() / n_reps as f64;
            let bias = mean_est - true_theta;
            let mse: f64 = estimates
                .iter()
                .map(|&e| (e - true_theta).powi(2))
                .sum::<f64>()
                / n_reps as f64;
            let avg_items = n_items_sum / n_reps as f64;

            (bias, mse, avg_items)
        })
        .collect();

    let mut bias = Array1::zeros(n_thetas);
    let mut mse = Array1::zeros(n_thetas);
    let mut avg_items = Array1::zeros(n_thetas);

    for (i, (b, m, a)) in stats.into_iter().enumerate() {
        bias[i] = b;
        mse[i] = m;
        avg_items[i] = a;
    }

    (
        eval_thetas.as_array().to_owned().to_pyarray(py),
        bias.to_pyarray(py),
        mse.to_pyarray(py),
        avg_items.to_pyarray(py),
    )
}

/// Register CAT functions with the Python module
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(cat_compute_item_info, m)?)?;
    m.add_function(wrap_pyfunction!(cat_select_max_info, m)?)?;
    m.add_function(wrap_pyfunction!(cat_eap_update, m)?)?;
    m.add_function(wrap_pyfunction!(cat_simulate_batch, m)?)?;
    m.add_function(wrap_pyfunction!(cat_conditional_mse, m)?)?;
    Ok(())
}
