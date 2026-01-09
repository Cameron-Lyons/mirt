//! SIBTEST (Simultaneous Item Bias Test) functions.

use ndarray::Array1;
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2, ToPyArray};
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::utils::{EPSILON, normal_cdf};

/// Compute SIBTEST beta statistic
#[pyfunction]
pub fn sibtest_compute_beta<'py>(
    py: Python<'py>,
    ref_data: PyReadonlyArray2<i32>,
    focal_data: PyReadonlyArray2<i32>,
    ref_scores: PyReadonlyArray1<i32>,
    focal_scores: PyReadonlyArray1<i32>,
    suspect_items: PyReadonlyArray1<i32>,
) -> (
    f64,
    f64,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
) {
    let ref_data = ref_data.as_array();
    let focal_data = focal_data.as_array();
    let ref_scores = ref_scores.as_array();
    let focal_scores = focal_scores.as_array();
    let suspect_items = suspect_items.as_array();

    let all_scores: Vec<i32> = ref_scores
        .iter()
        .chain(focal_scores.iter())
        .cloned()
        .collect();
    let mut unique_scores: Vec<i32> = all_scores.clone();
    unique_scores.sort();
    unique_scores.dedup();

    let suspect_vec: Vec<usize> = suspect_items.iter().map(|&x| x as usize).collect();

    let results: Vec<(f64, f64)> = unique_scores
        .par_iter()
        .map(|&k| {
            let ref_at_k: Vec<usize> = ref_scores
                .iter()
                .enumerate()
                .filter(|&(_, &s)| s == k)
                .map(|(i, _)| i)
                .collect();

            let focal_at_k: Vec<usize> = focal_scores
                .iter()
                .enumerate()
                .filter(|&(_, &s)| s == k)
                .map(|(i, _)| i)
                .collect();

            let n_ref_k = ref_at_k.len();
            let n_focal_k = focal_at_k.len();

            if n_ref_k == 0 || n_focal_k == 0 {
                return (f64::NAN, 0.0);
            }

            let mean_ref_k: f64 = ref_at_k
                .iter()
                .map(|&i| {
                    suspect_vec
                        .iter()
                        .map(|&j| ref_data[[i, j]] as f64)
                        .sum::<f64>()
                })
                .sum::<f64>()
                / n_ref_k as f64;

            let mean_focal_k: f64 = focal_at_k
                .iter()
                .map(|&i| {
                    suspect_vec
                        .iter()
                        .map(|&j| focal_data[[i, j]] as f64)
                        .sum::<f64>()
                })
                .sum::<f64>()
                / n_focal_k as f64;

            let beta_k = mean_ref_k - mean_focal_k;
            let weight = 2.0 * n_ref_k as f64 * n_focal_k as f64 / (n_ref_k + n_focal_k) as f64;

            (beta_k, weight)
        })
        .collect();

    let valid: Vec<(f64, f64)> = results.into_iter().filter(|(b, _)| !b.is_nan()).collect();

    if valid.is_empty() {
        return (
            f64::NAN,
            f64::NAN,
            Array1::zeros(0).to_pyarray(py),
            Array1::zeros(0).to_pyarray(py),
        );
    }

    let beta_k_arr: Array1<f64> = Array1::from(valid.iter().map(|(b, _)| *b).collect::<Vec<_>>());
    let n_k_arr: Array1<f64> = Array1::from(valid.iter().map(|(_, n)| *n).collect::<Vec<_>>());

    let total_weight: f64 = n_k_arr.sum();
    let beta: f64 = beta_k_arr
        .iter()
        .zip(n_k_arr.iter())
        .map(|(&b, &n)| b * n)
        .sum::<f64>()
        / total_weight;

    let weighted_mean = beta;
    let weighted_var: f64 = beta_k_arr
        .iter()
        .zip(n_k_arr.iter())
        .map(|(&b, &n)| n * (b - weighted_mean).powi(2))
        .sum::<f64>()
        / total_weight;

    let n_total = (ref_scores.len() + focal_scores.len()) as f64;
    let se = (weighted_var / n_total).sqrt();

    (beta, se, beta_k_arr.to_pyarray(py), n_k_arr.to_pyarray(py))
}

/// Compute SIBTEST for all items
#[pyfunction]
#[allow(clippy::type_complexity)]
pub fn sibtest_all_items<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<i32>,
    groups: PyReadonlyArray1<i32>,
    anchor_items: Option<PyReadonlyArray1<i32>>,
) -> (
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
) {
    let data = data.as_array();
    let groups = groups.as_array();

    let n_items = data.ncols();

    let mut unique_groups: Vec<i32> = groups.iter().cloned().collect();
    unique_groups.sort();
    unique_groups.dedup();
    let ref_group = unique_groups[0];
    let focal_group = unique_groups[1];

    let ref_mask: Vec<bool> = groups.iter().map(|&g| g == ref_group).collect();
    let focal_mask: Vec<bool> = groups.iter().map(|&g| g == focal_group).collect();

    let anchor_set: Option<Vec<usize>> =
        anchor_items.map(|a| a.as_array().iter().map(|&x| x as usize).collect());

    let results: Vec<(f64, f64, f64)> = (0..n_items)
        .into_par_iter()
        .map(|item_idx| {
            let matching: Vec<usize> = match &anchor_set {
                Some(anchors) => anchors
                    .iter()
                    .filter(|&&j| j != item_idx)
                    .cloned()
                    .collect(),
                None => (0..n_items).filter(|&j| j != item_idx).collect(),
            };

            if matching.is_empty() {
                return (f64::NAN, f64::NAN, f64::NAN);
            }

            let ref_scores: Vec<i32> = ref_mask
                .iter()
                .enumerate()
                .filter(|&(_, &is_ref)| is_ref)
                .map(|(i, _)| matching.iter().map(|&j| data[[i, j]]).sum())
                .collect();

            let focal_scores: Vec<i32> = focal_mask
                .iter()
                .enumerate()
                .filter(|&(_, &is_focal)| is_focal)
                .map(|(i, _)| matching.iter().map(|&j| data[[i, j]]).sum())
                .collect();

            let all_scores: Vec<i32> = ref_scores
                .iter()
                .chain(focal_scores.iter())
                .cloned()
                .collect();
            let mut unique_scores = all_scores.clone();
            unique_scores.sort();
            unique_scores.dedup();

            let ref_data: Vec<Vec<i32>> = ref_mask
                .iter()
                .enumerate()
                .filter(|&(_, &is_ref)| is_ref)
                .map(|(i, _)| data.row(i).to_vec())
                .collect();

            let focal_data: Vec<Vec<i32>> = focal_mask
                .iter()
                .enumerate()
                .filter(|&(_, &is_focal)| is_focal)
                .map(|(i, _)| data.row(i).to_vec())
                .collect();

            let mut beta_k_vec = Vec::new();
            let mut n_k_vec = Vec::new();

            for &k in &unique_scores {
                let ref_at_k: Vec<usize> = ref_scores
                    .iter()
                    .enumerate()
                    .filter(|&(_, &s)| s == k)
                    .map(|(i, _)| i)
                    .collect();

                let focal_at_k: Vec<usize> = focal_scores
                    .iter()
                    .enumerate()
                    .filter(|&(_, &s)| s == k)
                    .map(|(i, _)| i)
                    .collect();

                let n_ref_k = ref_at_k.len();
                let n_focal_k = focal_at_k.len();

                if n_ref_k > 0 && n_focal_k > 0 {
                    let mean_ref: f64 = ref_at_k
                        .iter()
                        .map(|&i| ref_data[i][item_idx] as f64)
                        .sum::<f64>()
                        / n_ref_k as f64;
                    let mean_focal: f64 = focal_at_k
                        .iter()
                        .map(|&i| focal_data[i][item_idx] as f64)
                        .sum::<f64>()
                        / n_focal_k as f64;

                    beta_k_vec.push(mean_ref - mean_focal);
                    n_k_vec.push(
                        2.0 * n_ref_k as f64 * n_focal_k as f64 / (n_ref_k + n_focal_k) as f64,
                    );
                }
            }

            if beta_k_vec.is_empty() {
                return (f64::NAN, f64::NAN, f64::NAN);
            }

            let total_weight: f64 = n_k_vec.iter().sum();
            let beta: f64 = beta_k_vec
                .iter()
                .zip(n_k_vec.iter())
                .map(|(&b, &n)| b * n)
                .sum::<f64>()
                / total_weight;

            let weighted_var: f64 = beta_k_vec
                .iter()
                .zip(n_k_vec.iter())
                .map(|(&b, &n)| n * (b - beta).powi(2))
                .sum::<f64>()
                / total_weight;

            let n_total = (ref_scores.len() + focal_scores.len()) as f64;
            let se = (weighted_var / n_total).sqrt();

            let z = if se > EPSILON { beta / se } else { f64::NAN };
            let p_value = if z.is_nan() {
                f64::NAN
            } else {
                2.0 * (1.0 - normal_cdf(z.abs()))
            };

            (beta, z, p_value)
        })
        .collect();

    let betas: Array1<f64> = Array1::from(results.iter().map(|(b, _, _)| *b).collect::<Vec<_>>());
    let zs: Array1<f64> = Array1::from(results.iter().map(|(_, z, _)| *z).collect::<Vec<_>>());
    let p_values: Array1<f64> =
        Array1::from(results.iter().map(|(_, _, p)| *p).collect::<Vec<_>>());

    (
        betas.to_pyarray(py),
        zs.to_pyarray(py),
        p_values.to_pyarray(py),
    )
}

/// Register SIBTEST functions with the Python module
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sibtest_compute_beta, m)?)?;
    m.add_function(wrap_pyfunction!(sibtest_all_items, m)?)?;
    Ok(())
}
