//! Many-Facet Rasch Model computations.
//!
//! This module provides optimized implementations for:
//! - Multi-facet log-likelihood computation
//! - Facet parameter updates
//! - Fit statistics (infit/outfit)
//! - Polytomous MFRM likelihood

use ndarray::{Array1, Array2};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, ToPyArray};
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::utils::log_sigmoid;

#[pyfunction]
pub fn mfrm_log_likelihood<'py>(
    py: Python<'py>,
    responses: PyReadonlyArray2<i32>,
    quad_points: PyReadonlyArray1<f64>,
    item_idx: PyReadonlyArray2<i32>,
    facet_idx: PyReadonlyArray2<i32>,
    item_difficulty: PyReadonlyArray1<f64>,
    facet_parameters: PyReadonlyArray1<f64>,
) -> Bound<'py, PyArray2<f64>> {
    let responses = responses.as_array();
    let quad_points = quad_points.as_array();
    let item_idx = item_idx.as_array();
    let facet_idx = facet_idx.as_array();
    let item_difficulty = item_difficulty.as_array();
    let facet_parameters = facet_parameters.as_array();

    let n_persons = responses.nrows();
    let n_obs = responses.ncols();
    let n_quad = quad_points.len();

    let diff_vec: Vec<f64> = item_difficulty.to_vec();
    let facet_vec: Vec<f64> = facet_parameters.to_vec();

    let log_likes: Vec<Vec<f64>> = (0..n_persons)
        .into_par_iter()
        .map(|i| {
            let resp_row: Vec<i32> = responses.row(i).to_vec();
            let item_row: Vec<i32> = item_idx.row(i).to_vec();
            let facet_row: Vec<i32> = facet_idx.row(i).to_vec();

            (0..n_quad)
                .map(|q| {
                    let theta = quad_points[q];

                    let mut ll = 0.0;
                    for j in 0..n_obs {
                        let r = resp_row[j];
                        if r >= 0 {
                            let item_i = item_row[j] as usize;
                            let facet_i = facet_row[j];

                            let mut z = theta - diff_vec[item_i];
                            if facet_i >= 0 {
                                z -= facet_vec[facet_i as usize];
                            }

                            let log_p = log_sigmoid(z);
                            let log_1_minus_p = log_sigmoid(-z);
                            ll += (r as f64) * log_p + (1.0 - r as f64) * log_1_minus_p;
                        }
                    }
                    ll
                })
                .collect()
        })
        .collect();

    let mut result = Array2::zeros((n_persons, n_quad));
    for (i, row) in log_likes.iter().enumerate() {
        for (q, &val) in row.iter().enumerate() {
            result[[i, q]] = val;
        }
    }

    result.to_pyarray(py)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn update_facet_parameters<'py>(
    py: Python<'py>,
    responses: PyReadonlyArray2<i32>,
    theta_eap: PyReadonlyArray1<f64>,
    item_idx: PyReadonlyArray2<i32>,
    facet_idx: PyReadonlyArray2<i32>,
    item_difficulty: PyReadonlyArray1<f64>,
    n_facet_levels: i32,
    current_facet_params: PyReadonlyArray1<f64>,
) -> Bound<'py, PyArray1<f64>> {
    let responses = responses.as_array();
    let theta_eap = theta_eap.as_array();
    let item_idx = item_idx.as_array();
    let facet_idx = facet_idx.as_array();
    let item_difficulty = item_difficulty.as_array();
    let current_facet_params = current_facet_params.as_array();

    let n_persons = responses.nrows();
    let n_obs = responses.ncols();
    let n_levels = n_facet_levels as usize;

    let mut numerator = vec![0.0; n_levels];
    let mut denominator = vec![0.0; n_levels];

    for i in 0..n_persons {
        let theta_i = theta_eap[i];
        for j in 0..n_obs {
            let r = responses[[i, j]];
            let f_idx = facet_idx[[i, j]];

            if r >= 0 && f_idx >= 0 {
                let f_idx = f_idx as usize;
                let item_i = item_idx[[i, j]] as usize;

                let z = theta_i - item_difficulty[item_i] - current_facet_params[f_idx];
                let p = 1.0 / (1.0 + (-z).exp());

                numerator[f_idx] += (r as f64) - p;
                denominator[f_idx] += p * (1.0 - p);
            }
        }
    }

    let mut facet_params = Array1::<f64>::zeros(n_levels);
    for k in 0..n_levels {
        if denominator[k] > 1e-10 {
            facet_params[k] = current_facet_params[k] - numerator[k] / denominator[k];
        } else {
            facet_params[k] = current_facet_params[k];
        }
    }

    let mean: f64 = facet_params.iter().sum::<f64>() / n_levels as f64;
    for param in facet_params.iter_mut() {
        *param -= mean;
    }

    facet_params.to_pyarray(py)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn compute_infit_outfit<'py>(
    py: Python<'py>,
    responses: PyReadonlyArray2<i32>,
    theta: PyReadonlyArray1<f64>,
    item_idx: PyReadonlyArray2<i32>,
    facet_idx: PyReadonlyArray2<i32>,
    item_difficulty: PyReadonlyArray1<f64>,
    facet_parameters: PyReadonlyArray1<f64>,
    entity_idx: PyReadonlyArray2<i32>,
    n_entities: i32,
) -> (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>) {
    let responses = responses.as_array();
    let theta = theta.as_array();
    let item_idx = item_idx.as_array();
    let facet_idx = facet_idx.as_array();
    let item_difficulty = item_difficulty.as_array();
    let facet_parameters = facet_parameters.as_array();
    let entity_idx = entity_idx.as_array();

    let n_persons = responses.nrows();
    let n_obs = responses.ncols();
    let n_ent = n_entities as usize;

    let mut infit_num = vec![0.0; n_ent];
    let mut infit_denom = vec![0.0; n_ent];
    let mut outfit_sum = vec![0.0; n_ent];
    let mut outfit_count = vec![0usize; n_ent];

    for i in 0..n_persons {
        let theta_i = theta[i];
        for j in 0..n_obs {
            let r = responses[[i, j]];
            let f_idx = facet_idx[[i, j]];
            let e_idx = entity_idx[[i, j]];

            if r >= 0 && e_idx >= 0 {
                let e_idx = e_idx as usize;
                let item_i = item_idx[[i, j]] as usize;

                let mut z = theta_i - item_difficulty[item_i];
                if f_idx >= 0 {
                    z -= facet_parameters[f_idx as usize];
                }

                let p = 1.0 / (1.0 + (-z).exp());
                let variance = p * (1.0 - p);
                let residual = (r as f64) - p;
                let std_residual_sq = if variance > 1e-10 {
                    (residual * residual) / variance
                } else {
                    0.0
                };

                infit_num[e_idx] += residual * residual;
                infit_denom[e_idx] += variance;
                outfit_sum[e_idx] += std_residual_sq;
                outfit_count[e_idx] += 1;
            }
        }
    }

    let mut infit = Array1::<f64>::zeros(n_ent);
    let mut outfit = Array1::<f64>::zeros(n_ent);

    for k in 0..n_ent {
        if infit_denom[k] > 1e-10 {
            infit[k] = infit_num[k] / infit_denom[k];
        }
        if outfit_count[k] > 0 {
            outfit[k] = outfit_sum[k] / outfit_count[k] as f64;
        }
    }

    (infit.to_pyarray(py), outfit.to_pyarray(py))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn polytomous_mfrm_likelihood<'py>(
    py: Python<'py>,
    responses: PyReadonlyArray2<i32>,
    quad_points: PyReadonlyArray1<f64>,
    item_idx: PyReadonlyArray2<i32>,
    facet_idx: PyReadonlyArray2<i32>,
    item_difficulty: PyReadonlyArray1<f64>,
    facet_parameters: PyReadonlyArray1<f64>,
    thresholds: PyReadonlyArray1<f64>,
    n_categories: i32,
) -> Bound<'py, PyArray2<f64>> {
    let responses = responses.as_array();
    let quad_points = quad_points.as_array();
    let item_idx = item_idx.as_array();
    let facet_idx = facet_idx.as_array();
    let item_difficulty = item_difficulty.as_array();
    let facet_parameters = facet_parameters.as_array();
    let thresholds = thresholds.as_array();

    let n_persons = responses.nrows();
    let n_obs = responses.ncols();
    let n_quad = quad_points.len();
    let n_cat = n_categories as usize;

    let diff_vec: Vec<f64> = item_difficulty.to_vec();
    let facet_vec: Vec<f64> = facet_parameters.to_vec();
    let thresh_vec: Vec<f64> = thresholds.to_vec();

    let log_likes: Vec<Vec<f64>> = (0..n_persons)
        .into_par_iter()
        .map(|i| {
            let resp_row: Vec<i32> = responses.row(i).to_vec();
            let item_row: Vec<i32> = item_idx.row(i).to_vec();
            let facet_row: Vec<i32> = facet_idx.row(i).to_vec();

            (0..n_quad)
                .map(|q| {
                    let theta = quad_points[q];

                    let mut ll = 0.0;
                    for j in 0..n_obs {
                        let r = resp_row[j];
                        if r >= 0 && (r as usize) < n_cat {
                            let item_i = item_row[j] as usize;
                            let facet_i = facet_row[j];

                            let mut base_measure = theta - diff_vec[item_i];
                            if facet_i >= 0 {
                                base_measure -= facet_vec[facet_i as usize];
                            }

                            let mut exp_terms = vec![1.0; n_cat];
                            let mut cumsum_tau = 0.0;
                            for k in 1..n_cat {
                                cumsum_tau += thresh_vec[k - 1];
                                exp_terms[k] = ((k as f64) * base_measure - cumsum_tau).exp();
                            }

                            let denom: f64 = exp_terms.iter().sum();
                            let prob = exp_terms[r as usize] / denom;
                            ll += prob.ln();
                        }
                    }
                    ll
                })
                .collect()
        })
        .collect();

    let mut result = Array2::zeros((n_persons, n_quad));
    for (i, row) in log_likes.iter().enumerate() {
        for (q, &val) in row.iter().enumerate() {
            result[[i, q]] = val;
        }
    }

    result.to_pyarray(py)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn update_item_difficulty<'py>(
    py: Python<'py>,
    responses: PyReadonlyArray2<i32>,
    theta_eap: PyReadonlyArray1<f64>,
    item_idx: PyReadonlyArray2<i32>,
    facet_idx: PyReadonlyArray2<i32>,
    n_items: i32,
    current_item_diff: PyReadonlyArray1<f64>,
    facet_parameters: PyReadonlyArray1<f64>,
) -> Bound<'py, PyArray1<f64>> {
    let responses = responses.as_array();
    let theta_eap = theta_eap.as_array();
    let item_idx = item_idx.as_array();
    let facet_idx = facet_idx.as_array();
    let current_item_diff = current_item_diff.as_array();
    let facet_parameters = facet_parameters.as_array();

    let n_persons = responses.nrows();
    let n_obs = responses.ncols();
    let n_items = n_items as usize;

    let mut numerator = vec![0.0; n_items];
    let mut denominator = vec![0.0; n_items];

    for i in 0..n_persons {
        let theta_i = theta_eap[i];
        for j in 0..n_obs {
            let r = responses[[i, j]];
            let item_i = item_idx[[i, j]];

            if r >= 0 && item_i >= 0 {
                let item_i = item_i as usize;
                let f_idx = facet_idx[[i, j]];

                let mut z = theta_i - current_item_diff[item_i];
                if f_idx >= 0 {
                    z -= facet_parameters[f_idx as usize];
                }

                let p = 1.0 / (1.0 + (-z).exp());

                numerator[item_i] += p - (r as f64);
                denominator[item_i] += p * (1.0 - p);
            }
        }
    }

    let mut item_diff = Array1::<f64>::zeros(n_items);
    for k in 0..n_items {
        if denominator[k] > 1e-10 {
            item_diff[k] = current_item_diff[k] - numerator[k] / denominator[k];
        } else {
            item_diff[k] = current_item_diff[k];
        }
    }

    let mean: f64 = item_diff.iter().sum::<f64>() / n_items as f64;
    for param in item_diff.iter_mut() {
        *param -= mean;
    }

    item_diff.to_pyarray(py)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(mfrm_log_likelihood, m)?)?;
    m.add_function(wrap_pyfunction!(update_facet_parameters, m)?)?;
    m.add_function(wrap_pyfunction!(compute_infit_outfit, m)?)?;
    m.add_function(wrap_pyfunction!(polytomous_mfrm_likelihood, m)?)?;
    m.add_function(wrap_pyfunction!(update_item_difficulty, m)?)?;
    Ok(())
}
