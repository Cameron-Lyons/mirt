use ndarray::{Array1, Array2, Array3};
use numpy::{
    PyArray1, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3, ToPyArray,
};
use pyo3::prelude::*;
use rand::prelude::*;
use rand_distr::Normal;
use rand_pcg::Pcg64;
use rayon::prelude::*;

const LOG_2_PI: f64 = 1.8378770664093453;
const EPSILON: f64 = 1e-10;

#[inline]
fn logsumexp(arr: &[f64]) -> f64 {
    if arr.is_empty() {
        return f64::NEG_INFINITY;
    }
    let max_val = arr.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if max_val.is_infinite() {
        return max_val;
    }
    let sum: f64 = arr.iter().map(|x| (x - max_val).exp()).sum();
    max_val + sum.ln()
}

#[inline]
fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let exp_x = x.exp();
        exp_x / (1.0 + exp_x)
    }
}

#[inline]
fn log_sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        -(-x).exp().ln_1p()
    } else {
        x - x.exp().ln_1p()
    }
}

#[inline]
fn clip(x: f64, min: f64, max: f64) -> f64 {
    x.max(min).min(max)
}

#[inline]
fn log_likelihood_2pl_single(
    responses: &[i32],
    theta: f64,
    discrimination: &[f64],
    difficulty: &[f64],
) -> f64 {
    let mut ll = 0.0;
    for (j, &resp) in responses.iter().enumerate() {
        if resp < 0 {
            continue;
        }
        let z = discrimination[j] * (theta - difficulty[j]);
        if resp == 1 {
            ll += log_sigmoid(z);
        } else {
            ll += log_sigmoid(-z);
        }
    }
    ll
}

#[inline]
fn log_likelihood_3pl_single(
    responses: &[i32],
    theta: f64,
    discrimination: &[f64],
    difficulty: &[f64],
    guessing: &[f64],
) -> f64 {
    let mut ll = 0.0;
    for (j, &resp) in responses.iter().enumerate() {
        if resp < 0 {
            continue;
        }
        let p_star = sigmoid(discrimination[j] * (theta - difficulty[j]));
        let p = guessing[j] + (1.0 - guessing[j]) * p_star;
        let p_clipped = clip(p, EPSILON, 1.0 - EPSILON);
        if resp == 1 {
            ll += p_clipped.ln();
        } else {
            ll += (1.0 - p_clipped).ln();
        }
    }
    ll
}

#[pyfunction]
fn compute_log_likelihoods_2pl<'py>(
    py: Python<'py>,
    responses: PyReadonlyArray2<i32>,
    quad_points: PyReadonlyArray1<f64>,
    discrimination: PyReadonlyArray1<f64>,
    difficulty: PyReadonlyArray1<f64>,
) -> Bound<'py, PyArray2<f64>> {
    let responses = responses.as_array();
    let quad_points = quad_points.as_array();
    let discrimination = discrimination.as_array();
    let difficulty = difficulty.as_array();

    let n_persons = responses.nrows();
    let n_quad = quad_points.len();
    let _n_items = responses.ncols();

    let disc_vec: Vec<f64> = discrimination.to_vec();
    let diff_vec: Vec<f64> = difficulty.to_vec();

    let log_likes: Vec<Vec<f64>> = (0..n_persons)
        .into_par_iter()
        .map(|i| {
            let resp_row: Vec<i32> = responses.row(i).to_vec();
            (0..n_quad)
                .map(|q| log_likelihood_2pl_single(&resp_row, quad_points[q], &disc_vec, &diff_vec))
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

/// Compute log-likelihoods for all persons at all quadrature points (3PL)
#[pyfunction]
fn compute_log_likelihoods_3pl<'py>(
    py: Python<'py>,
    responses: PyReadonlyArray2<i32>,
    quad_points: PyReadonlyArray1<f64>,
    discrimination: PyReadonlyArray1<f64>,
    difficulty: PyReadonlyArray1<f64>,
    guessing: PyReadonlyArray1<f64>,
) -> Bound<'py, PyArray2<f64>> {
    let responses = responses.as_array();
    let quad_points = quad_points.as_array();
    let discrimination = discrimination.as_array();
    let difficulty = difficulty.as_array();
    let guessing = guessing.as_array();

    let n_persons = responses.nrows();
    let n_quad = quad_points.len();

    let disc_vec: Vec<f64> = discrimination.to_vec();
    let diff_vec: Vec<f64> = difficulty.to_vec();
    let guess_vec: Vec<f64> = guessing.to_vec();

    let log_likes: Vec<Vec<f64>> = (0..n_persons)
        .into_par_iter()
        .map(|i| {
            let resp_row: Vec<i32> = responses.row(i).to_vec();
            (0..n_quad)
                .map(|q| {
                    log_likelihood_3pl_single(
                        &resp_row,
                        quad_points[q],
                        &disc_vec,
                        &diff_vec,
                        &guess_vec,
                    )
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
fn compute_log_likelihoods_mirt<'py>(
    py: Python<'py>,
    responses: PyReadonlyArray2<i32>,
    quad_points: PyReadonlyArray2<f64>,
    discrimination: PyReadonlyArray2<f64>,
    difficulty: PyReadonlyArray1<f64>,
) -> Bound<'py, PyArray2<f64>> {
    let responses = responses.as_array();
    let quad_points = quad_points.as_array();
    let discrimination = discrimination.as_array();
    let difficulty = difficulty.as_array();

    let n_persons = responses.nrows();
    let n_quad = quad_points.nrows();
    let n_items = responses.ncols();
    let n_factors = quad_points.ncols();

    let disc_sums: Vec<f64> = (0..n_items).map(|j| discrimination.row(j).sum()).collect();

    let log_likes: Vec<Vec<f64>> = (0..n_persons)
        .into_par_iter()
        .map(|i| {
            let resp_row: Vec<i32> = responses.row(i).to_vec();
            (0..n_quad)
                .map(|q| {
                    let theta_q = quad_points.row(q);
                    let mut ll = 0.0;
                    for (j, &resp) in resp_row.iter().enumerate() {
                        if resp < 0 {
                            continue;
                        }

                        let mut z = 0.0;
                        for f in 0..n_factors {
                            z += discrimination[[j, f]] * theta_q[f];
                        }
                        z -= disc_sums[j] * difficulty[j];

                        if resp == 1 {
                            ll += log_sigmoid(z);
                        } else {
                            ll += log_sigmoid(-z);
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

/// Complete E-step computation with posterior weights
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn e_step_complete<'py>(
    py: Python<'py>,
    responses: PyReadonlyArray2<i32>,
    quad_points: PyReadonlyArray1<f64>,
    quad_weights: PyReadonlyArray1<f64>,
    discrimination: PyReadonlyArray1<f64>,
    difficulty: PyReadonlyArray1<f64>,
    prior_mean: f64,
    prior_var: f64,
) -> (Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<f64>>) {
    let responses = responses.as_array();
    let quad_points = quad_points.as_array();
    let quad_weights = quad_weights.as_array();
    let discrimination = discrimination.as_array();
    let difficulty = difficulty.as_array();

    let n_persons = responses.nrows();
    let n_quad = quad_points.len();

    let disc_vec: Vec<f64> = discrimination.to_vec();
    let diff_vec: Vec<f64> = difficulty.to_vec();
    let quad_vec: Vec<f64> = quad_points.to_vec();
    let weight_vec: Vec<f64> = quad_weights.to_vec();

    let log_prior: Vec<f64> = quad_vec
        .iter()
        .map(|&theta| {
            let z = (theta - prior_mean) / prior_var.sqrt();
            -0.5 * (LOG_2_PI + prior_var.ln() + z * z)
        })
        .collect();

    let log_weights: Vec<f64> = weight_vec.iter().map(|w| (w + EPSILON).ln()).collect();

    let results: Vec<(Vec<f64>, f64)> = (0..n_persons)
        .into_par_iter()
        .map(|i| {
            let resp_row: Vec<i32> = responses.row(i).to_vec();

            let log_joint: Vec<f64> = (0..n_quad)
                .map(|q| {
                    let ll =
                        log_likelihood_2pl_single(&resp_row, quad_vec[q], &disc_vec, &diff_vec);
                    ll + log_prior[q] + log_weights[q]
                })
                .collect();

            let log_marginal = logsumexp(&log_joint);

            let posterior: Vec<f64> = log_joint
                .iter()
                .map(|&lj| (lj - log_marginal).exp())
                .collect();

            (posterior, log_marginal.exp())
        })
        .collect();

    let mut posterior_weights = Array2::zeros((n_persons, n_quad));
    let mut marginal_ll = Array1::zeros(n_persons);

    for (i, (post, marg)) in results.iter().enumerate() {
        for (q, &p) in post.iter().enumerate() {
            posterior_weights[[i, q]] = p;
        }
        marginal_ll[i] = *marg;
    }

    (posterior_weights.to_pyarray(py), marginal_ll.to_pyarray(py))
}

/// Compute r_k (expected counts) for dichotomous items
#[pyfunction]
fn compute_expected_counts<'py>(
    py: Python<'py>,
    responses: PyReadonlyArray1<i32>,
    posterior_weights: PyReadonlyArray2<f64>,
) -> (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>) {
    let responses = responses.as_array();
    let posterior_weights = posterior_weights.as_array();

    let n_persons = responses.len();
    let n_quad = posterior_weights.ncols();

    let mut r_k = Array1::zeros(n_quad);
    let mut n_k = Array1::zeros(n_quad);

    for i in 0..n_persons {
        let resp = responses[i];
        if resp < 0 {
            continue;
        }
        for q in 0..n_quad {
            let w = posterior_weights[[i, q]];
            n_k[q] += w;
            if resp == 1 {
                r_k[q] += w;
            }
        }
    }

    (r_k.to_pyarray(py), n_k.to_pyarray(py))
}

/// Compute r_kc (expected counts per category) for polytomous items
#[pyfunction]
fn compute_expected_counts_polytomous<'py>(
    py: Python<'py>,
    responses: PyReadonlyArray1<i32>,
    posterior_weights: PyReadonlyArray2<f64>,
    n_categories: usize,
) -> Bound<'py, PyArray2<f64>> {
    let responses = responses.as_array();
    let posterior_weights = posterior_weights.as_array();

    let n_persons = responses.len();
    let n_quad = posterior_weights.ncols();

    let mut r_kc = Array2::zeros((n_quad, n_categories));

    for i in 0..n_persons {
        let resp = responses[i];
        if resp < 0 || resp as usize >= n_categories {
            continue;
        }
        for q in 0..n_quad {
            r_kc[[q, resp as usize]] += posterior_weights[[i, q]];
        }
    }

    r_kc.to_pyarray(py)
}

#[pyfunction]
fn sibtest_compute_beta<'py>(
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
                .filter(|(_, &s)| s == k)
                .map(|(i, _)| i)
                .collect();

            let focal_at_k: Vec<usize> = focal_scores
                .iter()
                .enumerate()
                .filter(|(_, &s)| s == k)
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

#[pyfunction]
#[allow(clippy::type_complexity)]
fn sibtest_all_items<'py>(
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
                .filter(|(_, &is_ref)| is_ref)
                .map(|(i, _)| matching.iter().map(|&j| data[[i, j]]).sum())
                .collect();

            let focal_scores: Vec<i32> = focal_mask
                .iter()
                .enumerate()
                .filter(|(_, &is_focal)| is_focal)
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
                .filter(|(_, &is_ref)| is_ref)
                .map(|(i, _)| data.row(i).to_vec())
                .collect();

            let focal_data: Vec<Vec<i32>> = focal_mask
                .iter()
                .enumerate()
                .filter(|(_, &is_focal)| is_focal)
                .map(|(i, _)| data.row(i).to_vec())
                .collect();

            let mut beta_k_vec = Vec::new();
            let mut n_k_vec = Vec::new();

            for &k in &unique_scores {
                let ref_at_k: Vec<usize> = ref_scores
                    .iter()
                    .enumerate()
                    .filter(|(_, &s)| s == k)
                    .map(|(i, _)| i)
                    .collect();

                let focal_at_k: Vec<usize> = focal_scores
                    .iter()
                    .enumerate()
                    .filter(|(_, &s)| s == k)
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

/// Standard normal CDF approximation
fn normal_cdf(x: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.2316419 * x.abs());
    let d = 0.3989423 * (-x * x / 2.0).exp();
    let p =
        d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));
    if x > 0.0 {
        1.0 - p
    } else {
        p
    }
}

/// Simulate responses from Graded Response Model
#[pyfunction]
fn simulate_grm<'py>(
    py: Python<'py>,
    theta: PyReadonlyArray2<f64>,
    discrimination: PyReadonlyArray1<f64>,
    thresholds: PyReadonlyArray2<f64>,
    seed: u64,
) -> Bound<'py, PyArray2<i32>> {
    let theta = theta.as_array();
    let discrimination = discrimination.as_array();
    let thresholds = thresholds.as_array();

    let n_persons = theta.nrows();
    let n_items = discrimination.len();
    let n_categories = thresholds.ncols() + 1;

    let disc_vec: Vec<f64> = discrimination.to_vec();
    let thresh_vec: Vec<Vec<f64>> = (0..n_items).map(|i| thresholds.row(i).to_vec()).collect();

    let responses: Vec<Vec<i32>> = (0..n_persons)
        .into_par_iter()
        .map(|i| {
            let mut rng = Pcg64::seed_from_u64(seed + i as u64);
            let theta_i = theta[[i, 0]];

            (0..n_items)
                .map(|j| {
                    let mut cum_probs = vec![1.0; n_categories];
                    for k in 0..(n_categories - 1) {
                        let z = disc_vec[j] * (theta_i - thresh_vec[j][k]);
                        cum_probs[k + 1] = sigmoid(z);
                    }

                    let mut cat_probs = vec![0.0; n_categories];
                    for k in 0..n_categories {
                        let next = if k < n_categories - 1 {
                            cum_probs[k + 1]
                        } else {
                            0.0
                        };
                        cat_probs[k] = (cum_probs[k] - next).max(0.0);
                    }

                    let sum: f64 = cat_probs.iter().sum();
                    if sum > EPSILON {
                        for p in &mut cat_probs {
                            *p /= sum;
                        }
                    }

                    let u: f64 = rng.random();
                    let mut cumsum = 0.0;
                    for (k, &p) in cat_probs.iter().enumerate() {
                        cumsum += p;
                        if u < cumsum {
                            return k as i32;
                        }
                    }
                    (n_categories - 1) as i32
                })
                .collect()
        })
        .collect();

    let mut result = Array2::zeros((n_persons, n_items));
    for (i, row) in responses.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            result[[i, j]] = val;
        }
    }

    result.to_pyarray(py)
}

#[pyfunction]
fn simulate_gpcm<'py>(
    py: Python<'py>,
    theta: PyReadonlyArray2<f64>,
    discrimination: PyReadonlyArray1<f64>,
    thresholds: PyReadonlyArray2<f64>,
    seed: u64,
) -> Bound<'py, PyArray2<i32>> {
    let theta = theta.as_array();
    let discrimination = discrimination.as_array();
    let thresholds = thresholds.as_array();

    let n_persons = theta.nrows();
    let n_items = discrimination.len();
    let n_categories = thresholds.ncols() + 1;

    let disc_vec: Vec<f64> = discrimination.to_vec();
    let thresh_vec: Vec<Vec<f64>> = (0..n_items).map(|i| thresholds.row(i).to_vec()).collect();

    let responses: Vec<Vec<i32>> = (0..n_persons)
        .into_par_iter()
        .map(|i| {
            let mut rng = Pcg64::seed_from_u64(seed + i as u64);
            let theta_i = theta[[i, 0]];

            (0..n_items)
                .map(|j| {
                    let mut numerators = vec![0.0; n_categories];
                    for (k, num) in numerators.iter_mut().enumerate() {
                        let cumsum: f64 = thresh_vec[j][..k]
                            .iter()
                            .map(|&t| disc_vec[j] * (theta_i - t))
                            .sum();
                        *num = cumsum.exp();
                    }

                    let sum: f64 = numerators.iter().sum();
                    let cat_probs: Vec<f64> = numerators.iter().map(|&n| n / sum).collect();

                    let u: f64 = rng.random();
                    let mut cumsum = 0.0;
                    for (k, &p) in cat_probs.iter().enumerate() {
                        cumsum += p;
                        if u < cumsum {
                            return k as i32;
                        }
                    }
                    (n_categories - 1) as i32
                })
                .collect()
        })
        .collect();

    let mut result = Array2::zeros((n_persons, n_items));
    for (i, row) in responses.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            result[[i, j]] = val;
        }
    }

    result.to_pyarray(py)
}

/// Simulate dichotomous responses (2PL/3PL)
#[pyfunction]
fn simulate_dichotomous<'py>(
    py: Python<'py>,
    theta: PyReadonlyArray1<f64>,
    discrimination: PyReadonlyArray1<f64>,
    difficulty: PyReadonlyArray1<f64>,
    guessing: Option<PyReadonlyArray1<f64>>,
    seed: u64,
) -> Bound<'py, PyArray2<i32>> {
    let theta = theta.as_array();
    let discrimination = discrimination.as_array();
    let difficulty = difficulty.as_array();

    let n_persons = theta.len();
    let n_items = discrimination.len();

    let disc_vec: Vec<f64> = discrimination.to_vec();
    let diff_vec: Vec<f64> = difficulty.to_vec();
    let guess_vec: Vec<f64> = guessing
        .map(|g| g.as_array().to_vec())
        .unwrap_or_else(|| vec![0.0; n_items]);

    let responses: Vec<Vec<i32>> = (0..n_persons)
        .into_par_iter()
        .map(|i| {
            let mut rng = Pcg64::seed_from_u64(seed + i as u64);
            let theta_i = theta[i];

            (0..n_items)
                .map(|j| {
                    let z = disc_vec[j] * (theta_i - diff_vec[j]);
                    let p_star = sigmoid(z);
                    let p = guess_vec[j] + (1.0 - guess_vec[j]) * p_star;

                    let u: f64 = rng.random();
                    if u < p {
                        1
                    } else {
                        0
                    }
                })
                .collect()
        })
        .collect();

    let mut result = Array2::zeros((n_persons, n_items));
    for (i, row) in responses.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            result[[i, j]] = val;
        }
    }

    result.to_pyarray(py)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn generate_plausible_values_posterior<'py>(
    py: Python<'py>,
    responses: PyReadonlyArray2<i32>,
    quad_points: PyReadonlyArray1<f64>,
    quad_weights: PyReadonlyArray1<f64>,
    discrimination: PyReadonlyArray1<f64>,
    difficulty: PyReadonlyArray1<f64>,
    n_plausible: usize,
    jitter_sd: f64,
    seed: u64,
) -> Bound<'py, PyArray2<f64>> {
    let responses = responses.as_array();
    let quad_points = quad_points.as_array();
    let quad_weights = quad_weights.as_array();
    let discrimination = discrimination.as_array();
    let difficulty = difficulty.as_array();

    let n_persons = responses.nrows();
    let n_quad = quad_points.len();

    let disc_vec: Vec<f64> = discrimination.to_vec();
    let diff_vec: Vec<f64> = difficulty.to_vec();
    let quad_vec: Vec<f64> = quad_points.to_vec();
    let weight_vec: Vec<f64> = quad_weights.to_vec();

    let log_weights: Vec<f64> = weight_vec.iter().map(|&w| (w + EPSILON).ln()).collect();

    let pvs: Vec<Vec<f64>> = (0..n_persons)
        .into_par_iter()
        .map(|i| {
            let mut rng = Pcg64::seed_from_u64(seed + i as u64);
            let normal = Normal::new(0.0, jitter_sd).unwrap();
            let resp_row: Vec<i32> = responses.row(i).to_vec();

            let log_likes: Vec<f64> = (0..n_quad)
                .map(|q| log_likelihood_2pl_single(&resp_row, quad_vec[q], &disc_vec, &diff_vec))
                .collect();

            let log_posterior: Vec<f64> = log_likes
                .iter()
                .zip(log_weights.iter())
                .map(|(&ll, &lw)| ll + lw)
                .collect();

            let max_lp = log_posterior
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max);
            let posterior: Vec<f64> = log_posterior
                .iter()
                .map(|&lp| (lp - max_lp).exp())
                .collect();
            let sum: f64 = posterior.iter().sum();
            let posterior: Vec<f64> = posterior.iter().map(|&p| p / sum).collect();

            (0..n_plausible)
                .map(|_| {
                    let u: f64 = rng.random();
                    let mut cumsum = 0.0;
                    let mut idx = n_quad - 1;
                    for (q, &p) in posterior.iter().enumerate() {
                        cumsum += p;
                        if u < cumsum {
                            idx = q;
                            break;
                        }
                    }
                    quad_vec[idx] + rng.sample(normal)
                })
                .collect()
        })
        .collect();

    let mut result = Array2::zeros((n_persons, n_plausible));
    for (i, row) in pvs.iter().enumerate() {
        for (p, &val) in row.iter().enumerate() {
            result[[i, p]] = val;
        }
    }

    result.to_pyarray(py)
}

/// Generate plausible values using MCMC
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn generate_plausible_values_mcmc<'py>(
    py: Python<'py>,
    responses: PyReadonlyArray2<i32>,
    discrimination: PyReadonlyArray1<f64>,
    difficulty: PyReadonlyArray1<f64>,
    n_plausible: usize,
    n_iter: usize,
    proposal_sd: f64,
    seed: u64,
) -> Bound<'py, PyArray2<f64>> {
    let responses = responses.as_array();
    let discrimination = discrimination.as_array();
    let difficulty = difficulty.as_array();

    let n_persons = responses.nrows();

    let disc_vec: Vec<f64> = discrimination.to_vec();
    let diff_vec: Vec<f64> = difficulty.to_vec();

    let pvs: Vec<Vec<f64>> = (0..n_persons)
        .into_par_iter()
        .map(|i| {
            let mut rng = Pcg64::seed_from_u64(seed + i as u64);
            let proposal_dist = Normal::new(0.0, proposal_sd).unwrap();
            let resp_row: Vec<i32> = responses.row(i).to_vec();

            let mut theta = 0.0;
            let mut results = Vec::with_capacity(n_plausible);

            for _p in 0..n_plausible {
                for _ in 0..n_iter {
                    let proposal = theta + rng.sample(proposal_dist);

                    let ll_current =
                        log_likelihood_2pl_single(&resp_row, theta, &disc_vec, &diff_vec);
                    let ll_proposal =
                        log_likelihood_2pl_single(&resp_row, proposal, &disc_vec, &diff_vec);

                    let prior_current = -0.5 * theta * theta;
                    let prior_proposal = -0.5 * proposal * proposal;

                    let log_alpha = (ll_proposal + prior_proposal) - (ll_current + prior_current);

                    let u: f64 = rng.random();
                    if u.ln() < log_alpha {
                        theta = proposal;
                    }
                }
                results.push(theta);
            }
            results
        })
        .collect();

    let mut result = Array2::zeros((n_persons, n_plausible));
    for (i, row) in pvs.iter().enumerate() {
        for (p, &val) in row.iter().enumerate() {
            result[[i, p]] = val;
        }
    }

    result.to_pyarray(py)
}

#[pyfunction]
fn compute_observed_margins<'py>(
    py: Python<'py>,
    responses: PyReadonlyArray2<i32>,
) -> (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray2<f64>>) {
    let responses = responses.as_array();
    let n_persons = responses.nrows();
    let n_items = responses.ncols();

    let obs_uni: Array1<f64> = (0..n_items)
        .into_par_iter()
        .map(|j| {
            let valid: Vec<f64> = responses
                .column(j)
                .iter()
                .filter(|&&r| r >= 0)
                .map(|&r| r as f64)
                .collect();
            if valid.is_empty() {
                0.0
            } else {
                valid.iter().sum::<f64>() / valid.len() as f64
            }
        })
        .collect::<Vec<f64>>()
        .into();

    let pairs: Vec<(usize, usize)> = (0..n_items)
        .flat_map(|i| ((i + 1)..n_items).map(move |j| (i, j)))
        .collect();

    let bivariate: Vec<((usize, usize), f64)> = pairs
        .par_iter()
        .map(|&(i, j)| {
            let mut sum = 0.0;
            let mut count = 0;
            for p in 0..n_persons {
                let ri = responses[[p, i]];
                let rj = responses[[p, j]];
                if ri >= 0 && rj >= 0 {
                    sum += (ri * rj) as f64;
                    count += 1;
                }
            }
            let mean = if count > 0 { sum / count as f64 } else { 0.0 };
            ((i, j), mean)
        })
        .collect();

    let mut obs_bi = Array2::zeros((n_items, n_items));
    for ((i, j), val) in bivariate {
        obs_bi[[i, j]] = val;
        obs_bi[[j, i]] = val;
    }

    (obs_uni.to_pyarray(py), obs_bi.to_pyarray(py))
}

#[pyfunction]
fn compute_expected_margins<'py>(
    py: Python<'py>,
    quad_points: PyReadonlyArray1<f64>,
    quad_weights: PyReadonlyArray1<f64>,
    discrimination: PyReadonlyArray1<f64>,
    difficulty: PyReadonlyArray1<f64>,
) -> (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray2<f64>>) {
    let quad_points = quad_points.as_array();
    let quad_weights = quad_weights.as_array();
    let discrimination = discrimination.as_array();
    let difficulty = difficulty.as_array();

    let n_quad = quad_points.len();
    let n_items = discrimination.len();

    let probs: Vec<Vec<f64>> = (0..n_items)
        .map(|j| {
            (0..n_quad)
                .map(|q| sigmoid(discrimination[j] * (quad_points[q] - difficulty[j])))
                .collect()
        })
        .collect();

    let exp_uni: Array1<f64> = (0..n_items)
        .map(|j| {
            probs[j]
                .iter()
                .zip(quad_weights.iter())
                .map(|(&p, &w)| p * w)
                .sum()
        })
        .collect::<Vec<f64>>()
        .into();

    let pairs: Vec<(usize, usize)> = (0..n_items)
        .flat_map(|i| ((i + 1)..n_items).map(move |j| (i, j)))
        .collect();

    let bivariate: Vec<((usize, usize), f64)> = pairs
        .par_iter()
        .map(|&(i, j)| {
            let exp: f64 = (0..n_quad)
                .map(|q| probs[i][q] * probs[j][q] * quad_weights[q])
                .sum();
            ((i, j), exp)
        })
        .collect();

    let mut exp_bi = Array2::zeros((n_items, n_items));
    for ((i, j), val) in bivariate {
        exp_bi[[i, j]] = val;
        exp_bi[[j, i]] = val;
    }

    (exp_uni.to_pyarray(py), exp_bi.to_pyarray(py))
}

#[pyfunction]
fn generate_bootstrap_indices<'py>(
    py: Python<'py>,
    n_persons: usize,
    n_bootstrap: usize,
    seed: u64,
) -> Bound<'py, PyArray2<i64>> {
    let indices: Vec<Vec<i64>> = (0..n_bootstrap)
        .into_par_iter()
        .map(|b| {
            let mut rng = Pcg64::seed_from_u64(seed + b as u64);
            (0..n_persons)
                .map(|_| rng.random_range(0..n_persons as i64))
                .collect()
        })
        .collect();

    let mut result = Array2::zeros((n_bootstrap, n_persons));
    for (b, row) in indices.iter().enumerate() {
        for (i, &val) in row.iter().enumerate() {
            result[[b, i]] = val;
        }
    }

    result.to_pyarray(py)
}

/// Resample responses matrix
#[pyfunction]
fn resample_responses<'py>(
    py: Python<'py>,
    responses: PyReadonlyArray2<i32>,
    indices: PyReadonlyArray1<i64>,
) -> Bound<'py, PyArray2<i32>> {
    let responses = responses.as_array();
    let indices = indices.as_array();

    let n_sample = indices.len();
    let n_items = responses.ncols();

    let mut result = Array2::zeros((n_sample, n_items));
    for (i, &idx) in indices.iter().enumerate() {
        for j in 0..n_items {
            result[[i, j]] = responses[[idx as usize, j]];
        }
    }

    result.to_pyarray(py)
}

#[pyfunction]
fn impute_from_probabilities<'py>(
    py: Python<'py>,
    responses: PyReadonlyArray2<i32>,
    theta: PyReadonlyArray1<f64>,
    discrimination: PyReadonlyArray1<f64>,
    difficulty: PyReadonlyArray1<f64>,
    missing_code: i32,
    seed: u64,
) -> Bound<'py, PyArray2<i32>> {
    let responses = responses.as_array();
    let theta = theta.as_array();
    let discrimination = discrimination.as_array();
    let difficulty = difficulty.as_array();

    let n_persons = responses.nrows();
    let n_items = responses.ncols();

    let disc_vec: Vec<f64> = discrimination.to_vec();
    let diff_vec: Vec<f64> = difficulty.to_vec();

    let imputed: Vec<Vec<i32>> = (0..n_persons)
        .into_par_iter()
        .map(|i| {
            let mut rng = Pcg64::seed_from_u64(seed + i as u64);
            let theta_i = theta[i];

            (0..n_items)
                .map(|j| {
                    let orig = responses[[i, j]];
                    if orig != missing_code {
                        orig
                    } else {
                        let p = sigmoid(disc_vec[j] * (theta_i - diff_vec[j]));
                        let u: f64 = rng.random();
                        if u < p {
                            1
                        } else {
                            0
                        }
                    }
                })
                .collect()
        })
        .collect();

    let mut result = Array2::zeros((n_persons, n_items));
    for (i, row) in imputed.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            result[[i, j]] = val;
        }
    }

    result.to_pyarray(py)
}

/// Multiple imputation in parallel
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn multiple_imputation<'py>(
    py: Python<'py>,
    responses: PyReadonlyArray2<i32>,
    theta_mean: PyReadonlyArray1<f64>,
    theta_se: PyReadonlyArray1<f64>,
    discrimination: PyReadonlyArray1<f64>,
    difficulty: PyReadonlyArray1<f64>,
    missing_code: i32,
    n_imputations: usize,
    seed: u64,
) -> Bound<'py, PyArray3<i32>> {
    let responses = responses.as_array();
    let theta_mean = theta_mean.as_array();
    let theta_se = theta_se.as_array();
    let discrimination = discrimination.as_array();
    let difficulty = difficulty.as_array();

    let n_persons = responses.nrows();
    let n_items = responses.ncols();

    let disc_vec: Vec<f64> = discrimination.to_vec();
    let diff_vec: Vec<f64> = difficulty.to_vec();

    let imputations: Vec<Vec<Vec<i32>>> = (0..n_imputations)
        .into_par_iter()
        .map(|m| {
            let base_seed = seed + (m * n_persons) as u64;

            (0..n_persons)
                .map(|i| {
                    let mut rng = Pcg64::seed_from_u64(base_seed + i as u64);

                    let normal = Normal::new(0.0, 1.0).unwrap();
                    let theta_i = theta_mean[i] + rng.sample(normal) * theta_se[i];

                    (0..n_items)
                        .map(|j| {
                            let orig = responses[[i, j]];
                            if orig != missing_code {
                                orig
                            } else {
                                let p = sigmoid(disc_vec[j] * (theta_i - diff_vec[j]));
                                let u: f64 = rng.random();
                                if u < p {
                                    1
                                } else {
                                    0
                                }
                            }
                        })
                        .collect()
                })
                .collect()
        })
        .collect();

    let mut result = Array3::zeros((n_imputations, n_persons, n_items));
    for (m, imp) in imputations.iter().enumerate() {
        for (i, row) in imp.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                result[[m, i, j]] = val;
            }
        }
    }

    result.to_pyarray(py)
}

#[pyfunction]
fn compute_eap_scores<'py>(
    py: Python<'py>,
    responses: PyReadonlyArray2<i32>,
    quad_points: PyReadonlyArray1<f64>,
    quad_weights: PyReadonlyArray1<f64>,
    discrimination: PyReadonlyArray1<f64>,
    difficulty: PyReadonlyArray1<f64>,
) -> (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>) {
    let responses = responses.as_array();
    let quad_points = quad_points.as_array();
    let quad_weights = quad_weights.as_array();
    let discrimination = discrimination.as_array();
    let difficulty = difficulty.as_array();

    let n_persons = responses.nrows();
    let n_quad = quad_points.len();

    let disc_vec: Vec<f64> = discrimination.to_vec();
    let diff_vec: Vec<f64> = difficulty.to_vec();
    let quad_vec: Vec<f64> = quad_points.to_vec();
    let weight_vec: Vec<f64> = quad_weights.to_vec();

    let log_weights: Vec<f64> = weight_vec.iter().map(|&w| (w + EPSILON).ln()).collect();

    let results: Vec<(f64, f64)> = (0..n_persons)
        .into_par_iter()
        .map(|i| {
            let resp_row: Vec<i32> = responses.row(i).to_vec();

            let log_likes: Vec<f64> = (0..n_quad)
                .map(|q| log_likelihood_2pl_single(&resp_row, quad_vec[q], &disc_vec, &diff_vec))
                .collect();

            let log_posterior: Vec<f64> = log_likes
                .iter()
                .zip(log_weights.iter())
                .map(|(&ll, &lw)| ll + lw)
                .collect();

            let max_lp = log_posterior
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max);
            let posterior: Vec<f64> = log_posterior
                .iter()
                .map(|&lp| (lp - max_lp).exp())
                .collect();
            let sum: f64 = posterior.iter().sum();
            let posterior: Vec<f64> = posterior.iter().map(|&p| p / sum).collect();

            let eap: f64 = posterior
                .iter()
                .zip(quad_vec.iter())
                .map(|(&p, &theta)| p * theta)
                .sum();

            let psd: f64 = posterior
                .iter()
                .zip(quad_vec.iter())
                .map(|(&p, &theta)| p * (theta - eap).powi(2))
                .sum::<f64>()
                .sqrt();

            (eap, psd)
        })
        .collect();

    let theta: Array1<f64> = results.iter().map(|(t, _)| *t).collect::<Vec<_>>().into();
    let se: Array1<f64> = results.iter().map(|(_, s)| *s).collect::<Vec<_>>().into();

    (theta.to_pyarray(py), se.to_pyarray(py))
}

#[pyfunction]
fn em_fit_2pl<'py>(
    py: Python<'py>,
    responses: PyReadonlyArray2<i32>,
    n_quadpts: usize,
    max_iter: usize,
    tol: f64,
) -> (
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    f64,
    usize,
    bool,
) {
    let responses = responses.as_array();
    let n_persons = responses.nrows();
    let n_items = responses.ncols();

    let (quad_points, quad_weights) = gauss_hermite_quadrature(n_quadpts);

    let mut discrimination: Vec<f64> = vec![1.0; n_items];
    let mut difficulty: Vec<f64> = vec![0.0; n_items];

    for j in 0..n_items {
        let mut sum = 0.0;
        let mut count = 0;
        for i in 0..n_persons {
            let r = responses[[i, j]];
            if r >= 0 {
                sum += r as f64;
                count += 1;
            }
        }
        if count > 0 {
            let p = (sum / count as f64).clamp(0.01, 0.99);
            difficulty[j] = -p.ln() / (1.0 - p).ln().abs().max(0.01);
        }
    }

    let mut prev_ll = f64::NEG_INFINITY;
    let mut converged = false;
    let mut iteration = 0;

    for iter in 0..max_iter {
        iteration = iter + 1;

        let (posterior_weights, marginal_ll) = e_step_2pl_internal(
            &responses,
            &quad_points,
            &quad_weights,
            &discrimination,
            &difficulty,
            n_persons,
            n_items,
            n_quadpts,
        );

        let current_ll: f64 = marginal_ll.iter().map(|&x| (x + EPSILON).ln()).sum();

        if (current_ll - prev_ll).abs() < tol {
            converged = true;
            break;
        }
        prev_ll = current_ll;

        m_step_2pl_internal(
            &responses,
            &posterior_weights,
            &quad_points,
            &mut discrimination,
            &mut difficulty,
            n_persons,
            n_items,
            n_quadpts,
        );
    }

    let disc_arr: Array1<f64> = discrimination.into();
    let diff_arr: Array1<f64> = difficulty.into();

    (
        disc_arr.to_pyarray(py),
        diff_arr.to_pyarray(py),
        prev_ll,
        iteration,
        converged,
    )
}

#[allow(clippy::too_many_arguments)]
fn e_step_2pl_internal(
    responses: &ndarray::ArrayView2<i32>,
    quad_points: &[f64],
    quad_weights: &[f64],
    discrimination: &[f64],
    difficulty: &[f64],
    n_persons: usize,
    n_items: usize,
    n_quad: usize,
) -> (Vec<Vec<f64>>, Vec<f64>) {
    let log_weights: Vec<f64> = quad_weights.iter().map(|&w| (w + EPSILON).ln()).collect();

    let results: Vec<(Vec<f64>, f64)> = (0..n_persons)
        .into_par_iter()
        .map(|i| {
            let mut log_joint = vec![0.0; n_quad];

            for q in 0..n_quad {
                let theta = quad_points[q];
                let mut ll = 0.0;

                for j in 0..n_items {
                    let resp = responses[[i, j]];
                    if resp < 0 {
                        continue;
                    }
                    let z = discrimination[j] * (theta - difficulty[j]);
                    if resp == 1 {
                        ll += log_sigmoid(z);
                    } else {
                        ll += log_sigmoid(-z);
                    }
                }

                log_joint[q] = ll + log_weights[q];
            }

            let log_marginal = logsumexp(&log_joint);
            let posterior: Vec<f64> = log_joint
                .iter()
                .map(|&lj| (lj - log_marginal).exp())
                .collect();

            (posterior, log_marginal.exp())
        })
        .collect();

    let posterior_weights: Vec<Vec<f64>> = results.iter().map(|(p, _)| p.clone()).collect();
    let marginal_ll: Vec<f64> = results.iter().map(|(_, m)| *m).collect();

    (posterior_weights, marginal_ll)
}

#[allow(clippy::too_many_arguments)]
fn m_step_2pl_internal(
    responses: &ndarray::ArrayView2<i32>,
    posterior_weights: &[Vec<f64>],
    quad_points: &[f64],
    discrimination: &mut [f64],
    difficulty: &mut [f64],
    n_persons: usize,
    n_items: usize,
    n_quad: usize,
) {
    let new_params: Vec<(f64, f64)> = (0..n_items)
        .into_par_iter()
        .map(|j| {
            let mut r_k = vec![0.0; n_quad];
            let mut n_k = vec![0.0; n_quad];

            for i in 0..n_persons {
                let resp = responses[[i, j]];
                if resp < 0 {
                    continue;
                }
                for q in 0..n_quad {
                    let w = posterior_weights[i][q];
                    n_k[q] += w;
                    if resp == 1 {
                        r_k[q] += w;
                    }
                }
            }

            let mut a = discrimination[j];
            let mut b = difficulty[j];

            for _ in 0..10 {
                let mut grad_a = 0.0;
                let mut grad_b = 0.0;
                let mut hess_aa = 0.0;
                let mut hess_bb = 0.0;
                let mut hess_ab = 0.0;

                for q in 0..n_quad {
                    if n_k[q] < EPSILON {
                        continue;
                    }
                    let theta = quad_points[q];
                    let z = a * (theta - b);
                    let p = sigmoid(z);
                    let p_clipped = p.clamp(EPSILON, 1.0 - EPSILON);

                    let residual = r_k[q] - n_k[q] * p_clipped;

                    grad_a += residual * (theta - b);
                    grad_b += -residual * a;

                    let info = n_k[q] * p_clipped * (1.0 - p_clipped);
                    hess_aa += -info * (theta - b) * (theta - b);
                    hess_bb += -info * a * a;
                    hess_ab += info * a * (theta - b);
                }

                hess_aa -= 0.01;
                hess_bb -= 0.01;

                let det = hess_aa * hess_bb - hess_ab * hess_ab;
                if det.abs() < EPSILON {
                    break;
                }

                let delta_a = (hess_bb * grad_a - hess_ab * grad_b) / det;
                let delta_b = (-hess_ab * grad_a + hess_aa * grad_b) / det;

                a = (a - delta_a * 0.5).clamp(0.1, 5.0);
                b = (b - delta_b * 0.5).clamp(-6.0, 6.0);

                if delta_a.abs() < 1e-4 && delta_b.abs() < 1e-4 {
                    break;
                }
            }

            (a, b)
        })
        .collect();

    for (j, (a, b)) in new_params.into_iter().enumerate() {
        discrimination[j] = a;
        difficulty[j] = b;
    }
}

fn gauss_hermite_quadrature(n: usize) -> (Vec<f64>, Vec<f64>) {
    match n {
        11 => {
            let nodes = vec![
                -3.66847, -2.78329, -2.02594, -1.32656, -0.65681, 0.0, 0.65681, 1.32656, 2.02594,
                2.78329, 3.66847,
            ];
            let weights = vec![
                0.00001, 0.00076, 0.01526, 0.13548, 0.53134, 0.94531, 0.53134, 0.13548, 0.01526,
                0.00076, 0.00001,
            ];

            let sum: f64 = weights.iter().sum();
            let weights: Vec<f64> = weights.iter().map(|&w| w / sum).collect();
            (nodes, weights)
        }
        15 => {
            let nodes = vec![
                -4.49999, -3.66995, -2.96716, -2.32573, -1.71999, -1.13612, -0.56506, 0.0, 0.56506,
                1.13612, 1.71999, 2.32573, 2.96716, 3.66995, 4.49999,
            ];
            let weights = vec![
                1.5e-09, 1.5e-06, 3.9e-04, 0.00494, 0.03204, 0.11094, 0.21181, 0.22418, 0.21181,
                0.11094, 0.03204, 0.00494, 3.9e-04, 1.5e-06, 1.5e-09,
            ];
            let sum: f64 = weights.iter().sum();
            let weights: Vec<f64> = weights.iter().map(|&w| w / sum).collect();
            (nodes, weights)
        }
        21 => {
            let nodes = vec![
                -5.38748, -4.60368, -3.94477, -3.34785, -2.78881, -2.25497, -1.73854, -1.23408,
                -0.73747, -0.24535, 0.24535, 0.73747, 1.23408, 1.73854, 2.25497, 2.78881, 3.34785,
                3.94477, 4.60368, 5.38748, 0.0,
            ];
            let weights = vec![
                2.1e-13, 4.4e-10, 1.1e-07, 7.8e-06, 2.3e-04, 3.5e-03, 3.1e-02, 1.5e-01, 4.3e-01,
                7.2e-01, 7.2e-01, 4.3e-01, 1.5e-01, 3.1e-02, 3.5e-03, 2.3e-04, 7.8e-06, 1.1e-07,
                4.4e-10, 2.1e-13, 1.0,
            ];
            let sum: f64 = weights.iter().sum();
            let weights: Vec<f64> = weights.iter().map(|&w| w / sum).collect();
            (nodes, weights)
        }
        _ => {
            let mut nodes = Vec::with_capacity(n);
            let mut weights = Vec::with_capacity(n);
            let step = 8.0 / (n - 1) as f64;
            for i in 0..n {
                let x = -4.0 + i as f64 * step;
                nodes.push(x);
                weights.push((-x * x / 2.0).exp());
            }
            let sum: f64 = weights.iter().sum();
            let weights: Vec<f64> = weights.iter().map(|&w| w / sum).collect();
            (nodes, weights)
        }
    }
}

#[pyfunction]
#[allow(clippy::type_complexity)]
fn gibbs_sample_2pl<'py>(
    py: Python<'py>,
    responses: PyReadonlyArray2<i32>,
    n_iter: usize,
    burnin: usize,
    thin: usize,
    seed: u64,
) -> (
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray3<f64>>,
    Bound<'py, PyArray1<f64>>,
) {
    let responses = responses.as_array();
    let n_persons = responses.nrows();
    let n_items = responses.ncols();

    let mut discrimination: Vec<f64> = vec![1.0; n_items];
    let mut difficulty: Vec<f64> = vec![0.0; n_items];
    let mut theta: Vec<f64> = vec![0.0; n_persons];

    let n_samples = (n_iter - burnin) / thin;
    let mut disc_chain: Vec<Vec<f64>> = Vec::with_capacity(n_samples);
    let mut diff_chain: Vec<Vec<f64>> = Vec::with_capacity(n_samples);
    let mut theta_chain: Vec<Vec<f64>> = Vec::with_capacity(n_samples);
    let mut ll_chain: Vec<f64> = Vec::with_capacity(n_samples);

    let mut rng = Pcg64::seed_from_u64(seed);
    let proposal_theta = Normal::new(0.0, 0.5).unwrap();
    let proposal_param = Normal::new(0.0, 0.1).unwrap();

    for iter in 0..n_iter {
        theta = sample_theta_mh(
            &responses,
            &theta,
            &discrimination,
            &difficulty,
            n_persons,
            n_items,
            &mut rng,
            &proposal_theta,
        );

        discrimination = sample_discrimination_mh(
            &responses,
            &theta,
            &discrimination,
            &difficulty,
            n_items,
            &mut rng,
            &proposal_param,
        );

        difficulty = sample_difficulty_mh(
            &responses,
            &theta,
            &discrimination,
            &difficulty,
            n_items,
            &mut rng,
            &proposal_param,
        );

        if iter >= burnin && (iter - burnin).is_multiple_of(thin) {
            disc_chain.push(discrimination.clone());
            diff_chain.push(difficulty.clone());
            theta_chain.push(theta.clone());

            let ll = compute_total_ll(
                &responses,
                &theta,
                &discrimination,
                &difficulty,
                n_persons,
                n_items,
            );
            ll_chain.push(ll);
        }
    }

    let disc_arr = Array2::from_shape_vec(
        (n_samples, n_items),
        disc_chain.into_iter().flatten().collect(),
    )
    .unwrap();

    let diff_arr = Array2::from_shape_vec(
        (n_samples, n_items),
        diff_chain.into_iter().flatten().collect(),
    )
    .unwrap();

    let theta_arr = Array3::from_shape_vec(
        (n_samples, n_persons, 1),
        theta_chain.into_iter().flatten().collect(),
    )
    .unwrap();

    let ll_arr: Array1<f64> = ll_chain.into();

    (
        disc_arr.to_pyarray(py),
        diff_arr.to_pyarray(py),
        theta_arr.to_pyarray(py),
        ll_arr.to_pyarray(py),
    )
}

#[allow(clippy::too_many_arguments)]
fn sample_theta_mh(
    responses: &ndarray::ArrayView2<i32>,
    theta: &[f64],
    discrimination: &[f64],
    difficulty: &[f64],
    n_persons: usize,
    n_items: usize,
    rng: &mut Pcg64,
    proposal: &Normal<f64>,
) -> Vec<f64> {
    let seeds: Vec<u64> = (0..n_persons).map(|_| rng.random()).collect();

    let new_theta: Vec<f64> = (0..n_persons)
        .into_par_iter()
        .map(|i| {
            let mut local_rng = Pcg64::seed_from_u64(seeds[i]);

            let current = theta[i];
            let proposed = current + local_rng.sample(proposal);

            let mut ll_current = 0.0;
            let mut ll_proposed = 0.0;

            for j in 0..n_items {
                let resp = responses[[i, j]];
                if resp < 0 {
                    continue;
                }
                let z_curr = discrimination[j] * (current - difficulty[j]);
                let z_prop = discrimination[j] * (proposed - difficulty[j]);

                if resp == 1 {
                    ll_current += log_sigmoid(z_curr);
                    ll_proposed += log_sigmoid(z_prop);
                } else {
                    ll_current += log_sigmoid(-z_curr);
                    ll_proposed += log_sigmoid(-z_prop);
                }
            }

            let prior_current = -0.5 * current * current;
            let prior_proposed = -0.5 * proposed * proposed;

            let log_alpha = (ll_proposed + prior_proposed) - (ll_current + prior_current);

            if local_rng.random::<f64>().ln() < log_alpha {
                proposed
            } else {
                current
            }
        })
        .collect();

    new_theta
}

fn sample_discrimination_mh(
    responses: &ndarray::ArrayView2<i32>,
    theta: &[f64],
    discrimination: &[f64],
    difficulty: &[f64],
    n_items: usize,
    rng: &mut Pcg64,
    proposal: &Normal<f64>,
) -> Vec<f64> {
    let n_persons = theta.len();
    let mut new_disc = discrimination.to_vec();

    for j in 0..n_items {
        let current = discrimination[j];
        let proposed = (current + rng.sample(proposal)).clamp(0.1, 5.0);

        let mut ll_current = 0.0;
        let mut ll_proposed = 0.0;

        for i in 0..n_persons {
            let resp = responses[[i, j]];
            if resp < 0 {
                continue;
            }
            let z_curr = current * (theta[i] - difficulty[j]);
            let z_prop = proposed * (theta[i] - difficulty[j]);

            if resp == 1 {
                ll_current += log_sigmoid(z_curr);
                ll_proposed += log_sigmoid(z_prop);
            } else {
                ll_current += log_sigmoid(-z_curr);
                ll_proposed += log_sigmoid(-z_prop);
            }
        }

        let prior_current = -0.5 * current.ln().powi(2);
        let prior_proposed = -0.5 * proposed.ln().powi(2);

        let log_alpha = (ll_proposed + prior_proposed) - (ll_current + prior_current);

        if rng.random::<f64>().ln() < log_alpha {
            new_disc[j] = proposed;
        }
    }

    new_disc
}

fn sample_difficulty_mh(
    responses: &ndarray::ArrayView2<i32>,
    theta: &[f64],
    discrimination: &[f64],
    difficulty: &[f64],
    n_items: usize,
    rng: &mut Pcg64,
    proposal: &Normal<f64>,
) -> Vec<f64> {
    let n_persons = theta.len();
    let mut new_diff = difficulty.to_vec();

    for j in 0..n_items {
        let current = difficulty[j];
        let proposed = (current + rng.sample(proposal)).clamp(-6.0, 6.0);

        let mut ll_current = 0.0;
        let mut ll_proposed = 0.0;

        for i in 0..n_persons {
            let resp = responses[[i, j]];
            if resp < 0 {
                continue;
            }
            let z_curr = discrimination[j] * (theta[i] - current);
            let z_prop = discrimination[j] * (theta[i] - proposed);

            if resp == 1 {
                ll_current += log_sigmoid(z_curr);
                ll_proposed += log_sigmoid(z_prop);
            } else {
                ll_current += log_sigmoid(-z_curr);
                ll_proposed += log_sigmoid(-z_prop);
            }
        }

        let prior_current = -0.5 * current * current;
        let prior_proposed = -0.5 * proposed * proposed;

        let log_alpha = (ll_proposed + prior_proposed) - (ll_current + prior_current);

        if rng.random::<f64>().ln() < log_alpha {
            new_diff[j] = proposed;
        }
    }

    new_diff
}

fn compute_total_ll(
    responses: &ndarray::ArrayView2<i32>,
    theta: &[f64],
    discrimination: &[f64],
    difficulty: &[f64],
    n_persons: usize,
    n_items: usize,
) -> f64 {
    (0..n_persons)
        .into_par_iter()
        .map(|i| {
            let mut ll = 0.0;
            for j in 0..n_items {
                let resp = responses[[i, j]];
                if resp < 0 {
                    continue;
                }
                let z = discrimination[j] * (theta[i] - difficulty[j]);
                if resp == 1 {
                    ll += log_sigmoid(z);
                } else {
                    ll += log_sigmoid(-z);
                }
            }
            ll
        })
        .sum()
}

#[pyfunction]
fn mhrm_fit_2pl<'py>(
    py: Python<'py>,
    responses: PyReadonlyArray2<i32>,
    n_cycles: usize,
    burnin: usize,
    proposal_sd: f64,
    seed: u64,
) -> (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>, f64) {
    let responses = responses.as_array();
    let n_persons = responses.nrows();
    let n_items = responses.ncols();

    let mut discrimination: Vec<f64> = vec![1.0; n_items];
    let mut difficulty: Vec<f64> = vec![0.0; n_items];
    let mut theta: Vec<f64> = vec![0.0; n_persons];

    let mut rng = Pcg64::seed_from_u64(seed);
    let proposal = Normal::new(0.0, proposal_sd).unwrap();

    for cycle in 0..n_cycles {
        theta = sample_theta_mh(
            &responses,
            &theta,
            &discrimination,
            &difficulty,
            n_persons,
            n_items,
            &mut rng,
            &proposal,
        );

        let gain = 1.0 / (cycle as f64 + 1.0);

        if cycle >= burnin {
            for j in 0..n_items {
                let mut grad_a = 0.0;
                let mut grad_b = 0.0;
                let mut count = 0;

                for i in 0..n_persons {
                    let resp = responses[[i, j]];
                    if resp < 0 {
                        continue;
                    }
                    count += 1;
                    let z = discrimination[j] * (theta[i] - difficulty[j]);
                    let p = sigmoid(z);
                    let residual = resp as f64 - p;

                    grad_a += residual * (theta[i] - difficulty[j]);
                    grad_b += -residual * discrimination[j];
                }

                if count > 0 {
                    grad_a /= count as f64;
                    grad_b /= count as f64;

                    discrimination[j] = (discrimination[j] + gain * grad_a).clamp(0.1, 5.0);
                    difficulty[j] = (difficulty[j] + gain * grad_b).clamp(-6.0, 6.0);
                }
            }
        }
    }

    let ll = compute_total_ll(
        &responses,
        &theta,
        &discrimination,
        &difficulty,
        n_persons,
        n_items,
    );

    let disc_arr: Array1<f64> = discrimination.into();
    let diff_arr: Array1<f64> = difficulty.into();

    (disc_arr.to_pyarray(py), diff_arr.to_pyarray(py), ll)
}

#[pyfunction]
fn bootstrap_fit_2pl<'py>(
    py: Python<'py>,
    responses: PyReadonlyArray2<i32>,
    n_bootstrap: usize,
    n_quadpts: usize,
    max_iter: usize,
    tol: f64,
    seed: u64,
) -> (Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>) {
    let responses = responses.as_array();
    let n_persons = responses.nrows();
    let n_items = responses.ncols();

    let results: Vec<(Vec<f64>, Vec<f64>)> = (0..n_bootstrap)
        .into_par_iter()
        .map(|b| {
            let mut rng = Pcg64::seed_from_u64(seed + b as u64);

            let indices: Vec<usize> = (0..n_persons)
                .map(|_| rng.random_range(0..n_persons))
                .collect();

            let mut boot_responses = Array2::zeros((n_persons, n_items));
            for (new_i, &orig_i) in indices.iter().enumerate() {
                for j in 0..n_items {
                    boot_responses[[new_i, j]] = responses[[orig_i, j]];
                }
            }

            let (quad_points, quad_weights) = gauss_hermite_quadrature(n_quadpts);
            let mut discrimination: Vec<f64> = vec![1.0; n_items];
            let mut difficulty: Vec<f64> = vec![0.0; n_items];

            for j in 0..n_items {
                let mut sum = 0.0;
                let mut count = 0;
                for i in 0..n_persons {
                    let r = boot_responses[[i, j]];
                    if r >= 0 {
                        sum += r as f64;
                        count += 1;
                    }
                }
                if count > 0 {
                    let p = (sum / count as f64).clamp(0.01, 0.99);
                    difficulty[j] = -p.ln() / (1.0 - p).ln().abs().max(0.01);
                }
            }

            let mut prev_ll = f64::NEG_INFINITY;

            for _ in 0..max_iter {
                let (posterior_weights, marginal_ll) = e_step_2pl_internal(
                    &boot_responses.view(),
                    &quad_points,
                    &quad_weights,
                    &discrimination,
                    &difficulty,
                    n_persons,
                    n_items,
                    n_quadpts,
                );

                let current_ll: f64 = marginal_ll.iter().map(|&x| (x + EPSILON).ln()).sum();

                if (current_ll - prev_ll).abs() < tol {
                    break;
                }
                prev_ll = current_ll;

                m_step_2pl_internal(
                    &boot_responses.view(),
                    &posterior_weights,
                    &quad_points,
                    &mut discrimination,
                    &mut difficulty,
                    n_persons,
                    n_items,
                    n_quadpts,
                );
            }

            (discrimination, difficulty)
        })
        .collect();

    let mut disc_samples = Array2::zeros((n_bootstrap, n_items));
    let mut diff_samples = Array2::zeros((n_bootstrap, n_items));

    for (b, (disc, diff)) in results.into_iter().enumerate() {
        for j in 0..n_items {
            disc_samples[[b, j]] = disc[j];
            diff_samples[[b, j]] = diff[j];
        }
    }

    (disc_samples.to_pyarray(py), diff_samples.to_pyarray(py))
}

#[pyfunction]
fn compute_standardized_residuals<'py>(
    py: Python<'py>,
    responses: PyReadonlyArray2<i32>,
    theta: PyReadonlyArray1<f64>,
    discrimination: PyReadonlyArray1<f64>,
    difficulty: PyReadonlyArray1<f64>,
) -> Bound<'py, PyArray2<f64>> {
    let responses = responses.as_array();
    let theta = theta.as_array();
    let discrimination = discrimination.as_array();
    let difficulty = difficulty.as_array();

    let n_persons = responses.nrows();
    let n_items = responses.ncols();

    let residuals: Vec<Vec<f64>> = (0..n_persons)
        .into_par_iter()
        .map(|i| {
            let theta_i = theta[i];
            (0..n_items)
                .map(|j| {
                    let resp = responses[[i, j]];
                    if resp < 0 {
                        return f64::NAN;
                    }
                    let p = sigmoid(discrimination[j] * (theta_i - difficulty[j]));
                    let expected = p;
                    let variance = p * (1.0 - p);
                    (resp as f64 - expected) / (variance + EPSILON).sqrt()
                })
                .collect()
        })
        .collect();

    let mut result = Array2::zeros((n_persons, n_items));
    for (i, row) in residuals.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            result[[i, j]] = val;
        }
    }

    result.to_pyarray(py)
}

/// Compute Q3 (residual correlation) matrix - Yen's local dependence statistic
#[pyfunction]
fn compute_q3_matrix<'py>(
    py: Python<'py>,
    responses: PyReadonlyArray2<i32>,
    theta: PyReadonlyArray1<f64>,
    discrimination: PyReadonlyArray1<f64>,
    difficulty: PyReadonlyArray1<f64>,
) -> Bound<'py, PyArray2<f64>> {
    let responses = responses.as_array();
    let theta = theta.as_array();
    let discrimination = discrimination.as_array();
    let difficulty = difficulty.as_array();

    let n_persons = responses.nrows();
    let n_items = responses.ncols();

    // Compute residuals
    let residuals: Vec<Vec<f64>> = (0..n_persons)
        .into_par_iter()
        .map(|i| {
            let theta_i = theta[i];
            (0..n_items)
                .map(|j| {
                    let resp = responses[[i, j]];
                    if resp < 0 {
                        return f64::NAN;
                    }
                    let p = sigmoid(discrimination[j] * (theta_i - difficulty[j]));
                    let expected = p;
                    let variance = p * (1.0 - p);
                    (resp as f64 - expected) / (variance + EPSILON).sqrt()
                })
                .collect()
        })
        .collect();

    // Compute pairwise correlations in parallel
    let pairs: Vec<(usize, usize)> = (0..n_items)
        .flat_map(|i| ((i + 1)..n_items).map(move |j| (i, j)))
        .collect();

    let correlations: Vec<((usize, usize), f64)> = pairs
        .par_iter()
        .map(|&(i, j)| {
            let mut sum_xy = 0.0;
            let mut sum_x = 0.0;
            let mut sum_y = 0.0;
            let mut sum_x2 = 0.0;
            let mut sum_y2 = 0.0;
            let mut n = 0.0;

            for person_residuals in &residuals {
                let r_i = person_residuals[i];
                let r_j = person_residuals[j];
                if r_i.is_nan() || r_j.is_nan() {
                    continue;
                }
                sum_xy += r_i * r_j;
                sum_x += r_i;
                sum_y += r_j;
                sum_x2 += r_i * r_i;
                sum_y2 += r_j * r_j;
                n += 1.0;
            }

            if n < 3.0 {
                return ((i, j), f64::NAN);
            }

            let mean_x = sum_x / n;
            let mean_y = sum_y / n;
            let var_x = sum_x2 / n - mean_x * mean_x;
            let var_y = sum_y2 / n - mean_y * mean_y;
            let cov_xy = sum_xy / n - mean_x * mean_y;

            let denom = (var_x * var_y).sqrt();
            let corr = if denom > EPSILON { cov_xy / denom } else { 0.0 };

            ((i, j), corr)
        })
        .collect();

    let mut q3_matrix = Array2::zeros((n_items, n_items));
    for ((i, j), corr) in correlations {
        q3_matrix[[i, j]] = corr;
        q3_matrix[[j, i]] = corr;
    }

    q3_matrix.to_pyarray(py)
}

/// Compute LD chi-square statistics for all item pairs
#[pyfunction]
fn compute_ld_chi2_matrix<'py>(
    py: Python<'py>,
    responses: PyReadonlyArray2<i32>,
    theta: PyReadonlyArray1<f64>,
    discrimination: PyReadonlyArray1<f64>,
    difficulty: PyReadonlyArray1<f64>,
) -> Bound<'py, PyArray2<f64>> {
    let responses = responses.as_array();
    let theta = theta.as_array();
    let discrimination = discrimination.as_array();
    let difficulty = difficulty.as_array();

    let n_persons = responses.nrows();
    let n_items = responses.ncols();

    let pairs: Vec<(usize, usize)> = (0..n_items)
        .flat_map(|i| ((i + 1)..n_items).map(move |j| (i, j)))
        .collect();

    let chi2_values: Vec<((usize, usize), f64)> = pairs
        .par_iter()
        .map(|&(i, j)| {
            let mut obs = [[0.0; 2]; 2];
            let mut exp = [[0.0; 2]; 2];

            for p in 0..n_persons {
                let r_i = responses[[p, i]];
                let r_j = responses[[p, j]];
                if r_i < 0 || r_j < 0 {
                    continue;
                }

                let theta_p = theta[p];
                let p_i = sigmoid(discrimination[i] * (theta_p - difficulty[i]));
                let p_j = sigmoid(discrimination[j] * (theta_p - difficulty[j]));

                obs[r_i as usize][r_j as usize] += 1.0;

                exp[0][0] += (1.0 - p_i) * (1.0 - p_j);
                exp[0][1] += (1.0 - p_i) * p_j;
                exp[1][0] += p_i * (1.0 - p_j);
                exp[1][1] += p_i * p_j;
            }

            let mut chi2 = 0.0;
            for a in 0..2 {
                for b in 0..2 {
                    let e = exp[a][b].max(0.5);
                    chi2 += (obs[a][b] - e).powi(2) / e;
                }
            }

            ((i, j), chi2)
        })
        .collect();

    let mut chi2_matrix = Array2::zeros((n_items, n_items));
    for ((i, j), chi2) in chi2_values {
        chi2_matrix[[i, j]] = chi2;
        chi2_matrix[[j, i]] = chi2;
    }

    chi2_matrix.to_pyarray(py)
}

#[pyfunction]
fn mcem_e_step<'py>(
    py: Python<'py>,
    responses: PyReadonlyArray2<i32>,
    theta_samples: PyReadonlyArray3<f64>,
    discrimination: PyReadonlyArray1<f64>,
    difficulty: PyReadonlyArray1<f64>,
) -> (Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<f64>>) {
    let responses = responses.as_array();
    let theta_samples = theta_samples.as_array();
    let discrimination = discrimination.as_array();
    let difficulty = difficulty.as_array();

    let n_persons = responses.nrows();
    let n_samples = theta_samples.shape()[1];

    let disc_vec: Vec<f64> = discrimination.to_vec();
    let diff_vec: Vec<f64> = difficulty.to_vec();

    let results: Vec<(Vec<f64>, f64)> = (0..n_persons)
        .into_par_iter()
        .map(|i| {
            let resp_row: Vec<i32> = responses.row(i).to_vec();

            let log_likes: Vec<f64> = (0..n_samples)
                .map(|s| {
                    let theta_s = theta_samples[[i, s, 0]];
                    log_likelihood_2pl_single(&resp_row, theta_s, &disc_vec, &diff_vec)
                })
                .collect();

            let max_ll = log_likes.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let weights: Vec<f64> = log_likes.iter().map(|&ll| (ll - max_ll).exp()).collect();
            let sum: f64 = weights.iter().sum();
            let normalized: Vec<f64> = weights.iter().map(|&w| w / sum).collect();

            let marginal = (sum / n_samples as f64) * max_ll.exp();

            (normalized, marginal)
        })
        .collect();

    let mut importance_weights = Array2::zeros((n_persons, n_samples));
    let mut marginal_ll = Array1::zeros(n_persons);

    for (i, (weights, marg)) in results.iter().enumerate() {
        for (s, &w) in weights.iter().enumerate() {
            importance_weights[[i, s]] = w;
        }
        marginal_ll[i] = *marg;
    }

    (
        importance_weights.to_pyarray(py),
        marginal_ll.to_pyarray(py),
    )
}

#[pyfunction]
fn generate_qmc_samples<'py>(
    py: Python<'py>,
    n_persons: usize,
    n_samples: usize,
    n_factors: usize,
    _seed: u64,
) -> Bound<'py, PyArray3<f64>> {
    // Generate quasi-random samples transformed to standard normal
    // Using van der Corput sequence for simplicity
    let samples: Vec<Vec<Vec<f64>>> = (0..n_persons)
        .into_par_iter()
        .map(|_i| {
            // Using deterministic quasi-random sequence (Halton)
            // instead of pseudo-random for better coverage
            (0..n_samples)
                .map(|s| {
                    // Use halton-like sequence mixed with pseudo-random
                    (0..n_factors)
                        .map(|f| {
                            let base = (2 + f) as f64;
                            let mut result = 0.0;
                            let mut fraction = 1.0 / base;
                            let mut n = s + 1;
                            while n > 0 {
                                result += fraction * (n as f64 % base);
                                n /= base as usize;
                                fraction /= base;
                            }
                            // Transform uniform to normal via inverse CDF approximation
                            let u = result.clamp(0.001, 0.999);
                            // Approximate inverse normal CDF
                            let t = (-2.0 * (1.0 - u).ln()).sqrt();
                            let c0 = 2.515517;
                            let c1 = 0.802853;
                            let c2 = 0.010328;
                            let d1 = 1.432788;
                            let d2 = 0.189269;
                            let d3 = 0.001308;
                            let z = t
                                - (c0 + c1 * t + c2 * t * t)
                                    / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t);
                            if u < 0.5 {
                                -z
                            } else {
                                z
                            }
                        })
                        .collect()
                })
                .collect()
        })
        .collect();

    let mut result = Array3::zeros((n_persons, n_samples, n_factors));
    for (i, person_samples) in samples.iter().enumerate() {
        for (s, sample) in person_samples.iter().enumerate() {
            for (f, &val) in sample.iter().enumerate() {
                result[[i, s, f]] = val;
            }
        }
    }

    result.to_pyarray(py)
}

// ============================================================================
// WEIGHTED LIKELIHOOD ESTIMATION (WLE) SCORING
// ============================================================================

/// Compute WLE scores for all persons
#[pyfunction]
fn compute_wle_scores<'py>(
    py: Python<'py>,
    responses: PyReadonlyArray2<i32>,
    discrimination: PyReadonlyArray1<f64>,
    difficulty: PyReadonlyArray1<f64>,
    theta_min: f64,
    theta_max: f64,
    tol: f64,
) -> (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>) {
    let responses = responses.as_array();
    let discrimination = discrimination.as_array();
    let difficulty = difficulty.as_array();

    let n_persons = responses.nrows();
    let n_items = responses.ncols();

    let disc_vec: Vec<f64> = discrimination.to_vec();
    let diff_vec: Vec<f64> = difficulty.to_vec();

    let results: Vec<(f64, f64)> = (0..n_persons)
        .into_par_iter()
        .map(|i| {
            let resp_row: Vec<i32> = responses.row(i).to_vec();

            // Check for valid responses
            let valid_count = resp_row.iter().filter(|&&r| r >= 0).count();
            if valid_count == 0 {
                return (0.0, f64::INFINITY);
            }

            // Golden section search for WLE maximum
            let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
            let mut a = theta_min;
            let mut b = theta_max;

            while (b - a) > tol {
                let c = b - (b - a) / phi;
                let d = a + (b - a) / phi;

                let wl_c = wle_criterion(&resp_row, c, &disc_vec, &diff_vec, n_items);
                let wl_d = wle_criterion(&resp_row, d, &disc_vec, &diff_vec, n_items);

                if wl_c > wl_d {
                    b = d;
                } else {
                    a = c;
                }
            }

            let theta_wle = (a + b) / 2.0;

            // Standard error from information
            let info = test_information(theta_wle, &disc_vec, &diff_vec, n_items);
            let se = if info > EPSILON {
                1.0 / info.sqrt()
            } else {
                f64::INFINITY
            };

            (theta_wle, se)
        })
        .collect();

    let theta: Array1<f64> = results.iter().map(|(t, _)| *t).collect::<Vec<_>>().into();
    let se: Array1<f64> = results.iter().map(|(_, s)| *s).collect::<Vec<_>>().into();

    (theta.to_pyarray(py), se.to_pyarray(py))
}

/// WLE criterion function (log-likelihood + 0.5 * log(information))
#[inline]
fn wle_criterion(
    responses: &[i32],
    theta: f64,
    discrimination: &[f64],
    difficulty: &[f64],
    n_items: usize,
) -> f64 {
    let ll = log_likelihood_2pl_single(responses, theta, discrimination, difficulty);
    let info = test_information(theta, discrimination, difficulty, n_items);
    if info > EPSILON {
        ll + 0.5 * info.ln()
    } else {
        ll
    }
}

/// Compute test information at theta
#[inline]
fn test_information(theta: f64, discrimination: &[f64], difficulty: &[f64], n_items: usize) -> f64 {
    let mut info = 0.0;
    for j in 0..n_items {
        let p = sigmoid(discrimination[j] * (theta - difficulty[j]));
        info += discrimination[j] * discrimination[j] * p * (1.0 - p);
    }
    info
}

// ============================================================================
// SURVEY WEIGHTED EM ESTIMATION
// ============================================================================

/// Weighted E-step with survey weights
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn weighted_e_step<'py>(
    py: Python<'py>,
    responses: PyReadonlyArray2<i32>,
    weights: PyReadonlyArray1<f64>,
    quad_points: PyReadonlyArray1<f64>,
    quad_weights: PyReadonlyArray1<f64>,
    discrimination: PyReadonlyArray1<f64>,
    difficulty: PyReadonlyArray1<f64>,
) -> (Bound<'py, PyArray2<f64>>, f64) {
    let responses = responses.as_array();
    let survey_weights = weights.as_array();
    let quad_points = quad_points.as_array();
    let quad_weights = quad_weights.as_array();
    let discrimination = discrimination.as_array();
    let difficulty = difficulty.as_array();

    let n_persons = responses.nrows();
    let n_quad = quad_points.len();

    let disc_vec: Vec<f64> = discrimination.to_vec();
    let diff_vec: Vec<f64> = difficulty.to_vec();
    let quad_vec: Vec<f64> = quad_points.to_vec();
    let weight_vec: Vec<f64> = quad_weights.to_vec();

    let log_weights: Vec<f64> = weight_vec.iter().map(|w| (w + EPSILON).ln()).collect();

    let results: Vec<(Vec<f64>, f64)> = (0..n_persons)
        .into_par_iter()
        .map(|i| {
            let resp_row: Vec<i32> = responses.row(i).to_vec();

            let log_joint: Vec<f64> = (0..n_quad)
                .map(|q| {
                    let ll =
                        log_likelihood_2pl_single(&resp_row, quad_vec[q], &disc_vec, &diff_vec);
                    ll + log_weights[q]
                })
                .collect();

            let log_marginal = logsumexp(&log_joint);

            let posterior: Vec<f64> = log_joint
                .iter()
                .map(|&lj| (lj - log_marginal).exp())
                .collect();

            (posterior, log_marginal)
        })
        .collect();

    let mut posterior_weights = Array2::zeros((n_persons, n_quad));
    let mut weighted_ll = 0.0;

    for (i, (post, log_marg)) in results.iter().enumerate() {
        for (q, &p) in post.iter().enumerate() {
            posterior_weights[[i, q]] = p;
        }
        weighted_ll += survey_weights[i] * log_marg;
    }

    (posterior_weights.to_pyarray(py), weighted_ll)
}

#[pyfunction]
fn weighted_expected_counts<'py>(
    py: Python<'py>,
    responses: PyReadonlyArray1<i32>,
    posterior_weights: PyReadonlyArray2<f64>,
    survey_weights: PyReadonlyArray1<f64>,
) -> (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>) {
    let responses = responses.as_array();
    let posterior_weights = posterior_weights.as_array();
    let survey_weights = survey_weights.as_array();

    let n_persons = responses.len();
    let n_quad = posterior_weights.ncols();

    let mut r_k = Array1::zeros(n_quad);
    let mut n_k = Array1::zeros(n_quad);

    for i in 0..n_persons {
        let resp = responses[i];
        if resp < 0 {
            continue;
        }
        let sw = survey_weights[i];
        for q in 0..n_quad {
            let w = posterior_weights[[i, q]] * sw;
            n_k[q] += w;
            if resp == 1 {
                r_k[q] += w;
            }
        }
    }

    (r_k.to_pyarray(py), n_k.to_pyarray(py))
}

#[pyfunction]
#[allow(clippy::type_complexity)]
fn compute_fit_statistics<'py>(
    py: Python<'py>,
    responses: PyReadonlyArray2<i32>,
    theta: PyReadonlyArray1<f64>,
    discrimination: PyReadonlyArray1<f64>,
    difficulty: PyReadonlyArray1<f64>,
) -> (
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
) {
    let responses = responses.as_array();
    let theta = theta.as_array();
    let discrimination = discrimination.as_array();
    let difficulty = difficulty.as_array();

    let n_persons = responses.nrows();
    let n_items = responses.ncols();

    let z_sq_var: Vec<Vec<(f64, f64)>> = (0..n_persons)
        .into_par_iter()
        .map(|i| {
            let theta_i = theta[i];
            (0..n_items)
                .map(|j| {
                    let resp = responses[[i, j]];
                    if resp < 0 {
                        return (f64::NAN, f64::NAN);
                    }
                    let p = sigmoid(discrimination[j] * (theta_i - difficulty[j]));
                    let var = p * (1.0 - p);
                    let raw_resid = resp as f64 - p;
                    let z_sq = (raw_resid * raw_resid) / (var + EPSILON);
                    (z_sq, var)
                })
                .collect()
        })
        .collect();

    let item_stats: Vec<(f64, f64)> = (0..n_items)
        .into_par_iter()
        .map(|j| {
            let mut sum_z_sq = 0.0;
            let mut sum_z_sq_var = 0.0;
            let mut sum_var = 0.0;
            let mut count = 0.0;

            for person_data in &z_sq_var {
                let (z_sq, var) = person_data[j];
                if z_sq.is_nan() {
                    continue;
                }
                sum_z_sq += z_sq;
                sum_z_sq_var += z_sq * var;
                sum_var += var;
                count += 1.0;
            }

            let outfit = if count > 0.0 {
                sum_z_sq / count
            } else {
                f64::NAN
            };
            let infit = if sum_var > EPSILON {
                sum_z_sq_var / sum_var
            } else {
                f64::NAN
            };

            (outfit, infit)
        })
        .collect();

    let person_stats: Vec<(f64, f64)> = z_sq_var
        .par_iter()
        .map(|person_data| {
            let mut sum_z_sq = 0.0;
            let mut sum_z_sq_var = 0.0;
            let mut sum_var = 0.0;
            let mut count = 0.0;

            for &(z_sq, var) in person_data {
                if z_sq.is_nan() {
                    continue;
                }
                sum_z_sq += z_sq;
                sum_z_sq_var += z_sq * var;
                sum_var += var;
                count += 1.0;
            }

            let outfit = if count > 0.0 {
                sum_z_sq / count
            } else {
                f64::NAN
            };
            let infit = if sum_var > EPSILON {
                sum_z_sq_var / sum_var
            } else {
                f64::NAN
            };

            (outfit, infit)
        })
        .collect();

    let item_outfit: Array1<f64> = item_stats
        .iter()
        .map(|(o, _)| *o)
        .collect::<Vec<_>>()
        .into();
    let item_infit: Array1<f64> = item_stats
        .iter()
        .map(|(_, i)| *i)
        .collect::<Vec<_>>()
        .into();
    let person_outfit: Array1<f64> = person_stats
        .iter()
        .map(|(o, _)| *o)
        .collect::<Vec<_>>()
        .into();
    let person_infit: Array1<f64> = person_stats
        .iter()
        .map(|(_, i)| *i)
        .collect::<Vec<_>>()
        .into();

    (
        item_outfit.to_pyarray(py),
        item_infit.to_pyarray(py),
        person_outfit.to_pyarray(py),
        person_infit.to_pyarray(py),
    )
}

#[pyfunction]
fn compute_partially_compensatory_probs<'py>(
    py: Python<'py>,
    theta: PyReadonlyArray2<f64>,
    discrimination: PyReadonlyArray2<f64>,
    difficulty: PyReadonlyArray2<f64>,
    compensation: PyReadonlyArray2<f64>,
) -> Bound<'py, PyArray2<f64>> {
    let theta = theta.as_array();
    let discrimination = discrimination.as_array();
    let difficulty = difficulty.as_array();
    let compensation = compensation.as_array();

    let n_persons = theta.nrows();
    let n_items = discrimination.nrows();
    let n_factors = theta.ncols();

    let probs: Vec<Vec<f64>> = (0..n_persons)
        .into_par_iter()
        .map(|i| {
            (0..n_items)
                .map(|j| {
                    let mut prob = 1.0;
                    for k in 0..n_factors {
                        let z = discrimination[[j, k]] * (theta[[i, k]] - difficulty[[j, k]]);
                        let p_k = sigmoid(z);
                        prob *= p_k.powf(compensation[[j, k]]);
                    }
                    prob
                })
                .collect()
        })
        .collect();

    let mut result = Array2::zeros((n_persons, n_items));
    for (i, row) in probs.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            result[[i, j]] = val;
        }
    }

    result.to_pyarray(py)
}

/// Compute probabilities for noncompensatory (conjunctive) MIRT model
#[pyfunction]
fn compute_noncompensatory_probs<'py>(
    py: Python<'py>,
    theta: PyReadonlyArray2<f64>,
    discrimination: PyReadonlyArray2<f64>,
    difficulty: PyReadonlyArray2<f64>,
) -> Bound<'py, PyArray2<f64>> {
    let theta = theta.as_array();
    let discrimination = discrimination.as_array();
    let difficulty = difficulty.as_array();

    let n_persons = theta.nrows();
    let n_items = discrimination.nrows();
    let n_factors = theta.ncols();

    let probs: Vec<Vec<f64>> = (0..n_persons)
        .into_par_iter()
        .map(|i| {
            (0..n_items)
                .map(|j| {
                    let mut prob = 1.0;
                    for k in 0..n_factors {
                        let z = discrimination[[j, k]] * (theta[[i, k]] - difficulty[[j, k]]);
                        prob *= sigmoid(z);
                    }
                    prob
                })
                .collect()
        })
        .collect();

    let mut result = Array2::zeros((n_persons, n_items));
    for (i, row) in probs.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            result[[i, j]] = val;
        }
    }

    result.to_pyarray(py)
}

#[pyfunction]
fn compute_disjunctive_probs<'py>(
    py: Python<'py>,
    theta: PyReadonlyArray2<f64>,
    discrimination: PyReadonlyArray2<f64>,
    difficulty: PyReadonlyArray2<f64>,
) -> Bound<'py, PyArray2<f64>> {
    let theta = theta.as_array();
    let discrimination = discrimination.as_array();
    let difficulty = difficulty.as_array();

    let n_persons = theta.nrows();
    let n_items = discrimination.nrows();
    let n_factors = theta.ncols();

    let probs: Vec<Vec<f64>> = (0..n_persons)
        .into_par_iter()
        .map(|i| {
            (0..n_items)
                .map(|j| {
                    let mut prob_fail_all = 1.0;
                    for k in 0..n_factors {
                        let z = discrimination[[j, k]] * (theta[[i, k]] - difficulty[[j, k]]);
                        let p_k = sigmoid(z);
                        prob_fail_all *= 1.0 - p_k;
                    }
                    1.0 - prob_fail_all
                })
                .collect()
        })
        .collect();

    let mut result = Array2::zeros((n_persons, n_items));
    for (i, row) in probs.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            result[[i, j]] = val;
        }
    }

    result.to_pyarray(py)
}

// ============================================================================
// SEQUENTIAL RESPONSE MODEL PROBABILITIES
// ============================================================================

/// Compute probabilities for sequential response model
#[pyfunction]
fn compute_sequential_probs<'py>(
    py: Python<'py>,
    theta: PyReadonlyArray1<f64>,
    discrimination: PyReadonlyArray1<f64>,
    thresholds: PyReadonlyArray2<f64>,
    n_categories: PyReadonlyArray1<i32>,
) -> Bound<'py, PyArray2<f64>> {
    let theta = theta.as_array();
    let discrimination = discrimination.as_array();
    let thresholds = thresholds.as_array();
    let n_categories = n_categories.as_array();

    let n_persons = theta.len();
    let n_items = discrimination.len();
    let max_cats = n_categories.iter().max().copied().unwrap_or(2) as usize;

    let probs: Vec<Vec<f64>> = (0..n_persons)
        .into_par_iter()
        .map(|i| {
            let theta_i = theta[i];
            let mut person_probs = vec![0.0; n_items * max_cats];

            for j in 0..n_items {
                let n_cat = n_categories[j] as usize;
                let a = discrimination[j];

                let mut step_probs = Vec::with_capacity(n_cat - 1);
                for k in 0..(n_cat - 1) {
                    let z = a * (theta_i - thresholds[[j, k]]);
                    step_probs.push(sigmoid(z));
                }

                for k in 0..n_cat {
                    let mut prob: f64 = step_probs.iter().take(k).product();

                    if k < n_cat - 1 && k < step_probs.len() {
                        prob *= 1.0 - step_probs[k];
                    }

                    person_probs[j * max_cats + k] = prob;
                }
            }

            person_probs
        })
        .collect();

    let mut result = Array2::zeros((n_persons, n_items * max_cats));
    for (i, row) in probs.iter().enumerate() {
        for (jk, &val) in row.iter().enumerate() {
            result[[i, jk]] = val;
        }
    }

    result.to_pyarray(py)
}

#[pymodule]
fn mirt_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_log_likelihoods_2pl, m)?)?;
    m.add_function(wrap_pyfunction!(compute_log_likelihoods_3pl, m)?)?;
    m.add_function(wrap_pyfunction!(compute_log_likelihoods_mirt, m)?)?;
    m.add_function(wrap_pyfunction!(e_step_complete, m)?)?;

    m.add_function(wrap_pyfunction!(compute_expected_counts, m)?)?;
    m.add_function(wrap_pyfunction!(compute_expected_counts_polytomous, m)?)?;

    m.add_function(wrap_pyfunction!(sibtest_compute_beta, m)?)?;
    m.add_function(wrap_pyfunction!(sibtest_all_items, m)?)?;

    m.add_function(wrap_pyfunction!(simulate_grm, m)?)?;
    m.add_function(wrap_pyfunction!(simulate_gpcm, m)?)?;
    m.add_function(wrap_pyfunction!(simulate_dichotomous, m)?)?;

    m.add_function(wrap_pyfunction!(generate_plausible_values_posterior, m)?)?;
    m.add_function(wrap_pyfunction!(generate_plausible_values_mcmc, m)?)?;

    m.add_function(wrap_pyfunction!(compute_observed_margins, m)?)?;
    m.add_function(wrap_pyfunction!(compute_expected_margins, m)?)?;

    m.add_function(wrap_pyfunction!(generate_bootstrap_indices, m)?)?;
    m.add_function(wrap_pyfunction!(resample_responses, m)?)?;

    m.add_function(wrap_pyfunction!(impute_from_probabilities, m)?)?;
    m.add_function(wrap_pyfunction!(multiple_imputation, m)?)?;

    m.add_function(wrap_pyfunction!(compute_eap_scores, m)?)?;

    m.add_function(wrap_pyfunction!(em_fit_2pl, m)?)?;
    m.add_function(wrap_pyfunction!(gibbs_sample_2pl, m)?)?;
    m.add_function(wrap_pyfunction!(mhrm_fit_2pl, m)?)?;
    m.add_function(wrap_pyfunction!(bootstrap_fit_2pl, m)?)?;

    // New functions
    m.add_function(wrap_pyfunction!(compute_standardized_residuals, m)?)?;
    m.add_function(wrap_pyfunction!(compute_q3_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(compute_ld_chi2_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(mcem_e_step, m)?)?;
    m.add_function(wrap_pyfunction!(generate_qmc_samples, m)?)?;
    m.add_function(wrap_pyfunction!(compute_wle_scores, m)?)?;
    m.add_function(wrap_pyfunction!(weighted_e_step, m)?)?;
    m.add_function(wrap_pyfunction!(weighted_expected_counts, m)?)?;
    m.add_function(wrap_pyfunction!(compute_fit_statistics, m)?)?;
    m.add_function(wrap_pyfunction!(compute_partially_compensatory_probs, m)?)?;
    m.add_function(wrap_pyfunction!(compute_noncompensatory_probs, m)?)?;
    m.add_function(wrap_pyfunction!(compute_disjunctive_probs, m)?)?;
    m.add_function(wrap_pyfunction!(compute_sequential_probs, m)?)?;

    Ok(())
}
