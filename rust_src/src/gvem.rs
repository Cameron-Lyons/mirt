//! GVEM (Gaussian Variational EM) functions for multidimensional IRT.
//!
//! Provides parallel implementations of E-step and M-step for variational inference.

use ndarray::{Array2, Array3};
use numpy::{
    PyArray1, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3, ToPyArray,
};
use pyo3::prelude::*;
use rayon::prelude::*;

/// Compute Jaakkola-Jordan lambda function: lambda(xi) = tanh(xi/2) / (4*xi)
#[inline]
fn lambda_jj(xi: f64) -> f64 {
    let xi_abs = xi.abs();
    if xi_abs < 1e-6 {
        0.125
    } else {
        (xi_abs / 2.0).tanh() / (4.0 * xi_abs)
    }
}

/// GVEM E-step: update variational parameters (mu, sigma, xi)
/// Parallelizes over persons using Rayon
#[pyfunction]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn gvem_e_step<'py>(
    py: Python<'py>,
    responses: PyReadonlyArray2<i32>,
    loadings: PyReadonlyArray2<f64>,
    intercepts: PyReadonlyArray1<f64>,
    prior_cov_inv: PyReadonlyArray2<f64>,
    mu_in: PyReadonlyArray2<f64>,
    sigma_in: PyReadonlyArray3<f64>,
    xi_in: PyReadonlyArray2<f64>,
    n_inner_iter: usize,
) -> (
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray3<f64>>,
    Bound<'py, PyArray2<f64>>,
) {
    let responses = responses.as_array();
    let loadings = loadings.as_array();
    let intercepts = intercepts.as_array();
    let prior_cov_inv = prior_cov_inv.as_array();
    let mut mu = mu_in.as_array().to_owned();
    let mut sigma = sigma_in.as_array().to_owned();
    let mut xi = xi_in.as_array().to_owned();

    let n_persons = responses.nrows();
    let n_items = responses.ncols();
    let n_factors = loadings.ncols();

    let loadings_vec: Vec<Vec<f64>> = (0..n_items).map(|j| loadings.row(j).to_vec()).collect();
    let intercepts_vec: Vec<f64> = intercepts.to_vec();
    let prior_inv_flat: Vec<f64> = prior_cov_inv.iter().cloned().collect();

    for _ in 0..n_inner_iter {
        let lam: Array2<f64> = xi.mapv(lambda_jj);

        let results: Vec<(usize, Vec<f64>, Vec<f64>)> = (0..n_persons)
            .into_par_iter()
            .filter_map(|i| {
                let valid_count = (0..n_items).filter(|&j| responses[[i, j]] >= 0).count();
                if valid_count == 0 {
                    return None;
                }

                let mut sigma_inv = vec![0.0; n_factors * n_factors];
                sigma_inv.copy_from_slice(&prior_inv_flat);

                for j in 0..n_items {
                    if responses[[i, j]] < 0 {
                        continue;
                    }
                    let lam_ij = lam[[i, j]];
                    let a_j = &loadings_vec[j];
                    for k1 in 0..n_factors {
                        for k2 in 0..n_factors {
                            sigma_inv[k1 * n_factors + k2] += 2.0 * lam_ij * a_j[k1] * a_j[k2];
                        }
                    }
                }

                let sigma_i = invert_matrix(&sigma_inv, n_factors);

                let mut mu_term = vec![0.0; n_factors];
                for j in 0..n_items {
                    if responses[[i, j]] < 0 {
                        continue;
                    }
                    let y_ij = responses[[i, j]] as f64;
                    let lam_ij = lam[[i, j]];
                    let d_j = intercepts_vec[j];
                    let coeff = y_ij - 0.5 - 2.0 * lam_ij * d_j;
                    for k in 0..n_factors {
                        mu_term[k] += coeff * loadings_vec[j][k];
                    }
                }

                let mut mu_i = vec![0.0; n_factors];
                for k1 in 0..n_factors {
                    for k2 in 0..n_factors {
                        mu_i[k1] += sigma_i[k1 * n_factors + k2] * mu_term[k2];
                    }
                }

                Some((i, mu_i, sigma_i))
            })
            .collect();

        for (i, mu_i, sigma_i) in results {
            for k in 0..n_factors {
                mu[[i, k]] = mu_i[k];
            }
            for k1 in 0..n_factors {
                for k2 in 0..n_factors {
                    sigma[[i, k1, k2]] = sigma_i[k1 * n_factors + k2];
                }
            }
        }

        let xi_results: Vec<(usize, usize, f64)> = (0..n_persons)
            .into_par_iter()
            .flat_map(|i| {
                let mu_i: Vec<f64> = (0..n_factors).map(|k| mu[[i, k]]).collect();
                let sigma_i: Vec<Vec<f64>> = (0..n_factors)
                    .map(|k1| (0..n_factors).map(|k2| sigma[[i, k1, k2]]).collect())
                    .collect();

                let mut second_moment = vec![vec![0.0; n_factors]; n_factors];
                for k1 in 0..n_factors {
                    for k2 in 0..n_factors {
                        second_moment[k1][k2] = sigma_i[k1][k2] + mu_i[k1] * mu_i[k2];
                    }
                }

                (0..n_items)
                    .filter_map(|j| {
                        if responses[[i, j]] < 0 {
                            return None;
                        }
                        let a_j = &loadings_vec[j];
                        let d_j = intercepts_vec[j];

                        let mut quad_term = 0.0;
                        for k1 in 0..n_factors {
                            for k2 in 0..n_factors {
                                quad_term += a_j[k1] * second_moment[k1][k2] * a_j[k2];
                            }
                        }

                        let mut linear_term = 0.0;
                        for k in 0..n_factors {
                            linear_term += a_j[k] * mu_i[k];
                        }
                        linear_term *= 2.0 * d_j;

                        let xi_new = (quad_term + linear_term + d_j * d_j).max(1e-10).sqrt();
                        Some((i, j, xi_new))
                    })
                    .collect::<Vec<_>>()
            })
            .collect();

        for (i, j, xi_val) in xi_results {
            xi[[i, j]] = xi_val;
        }
    }

    (mu.to_pyarray(py), sigma.to_pyarray(py), xi.to_pyarray(py))
}

/// GVEM M-step: update item parameters (loadings, intercepts)
/// Parallelizes over items using Rayon
#[pyfunction]
pub fn gvem_m_step<'py>(
    py: Python<'py>,
    responses: PyReadonlyArray2<i32>,
    mu: PyReadonlyArray2<f64>,
    sigma: PyReadonlyArray3<f64>,
    xi: PyReadonlyArray2<f64>,
    loadings_in: PyReadonlyArray2<f64>,
    intercepts_in: PyReadonlyArray1<f64>,
) -> (Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<f64>>) {
    let responses = responses.as_array();
    let mu = mu.as_array();
    let sigma = sigma.as_array();
    let xi = xi.as_array();
    let mut loadings = loadings_in.as_array().to_owned();
    let mut intercepts = intercepts_in.as_array().to_owned();

    let n_persons = responses.nrows();
    let n_items = responses.ncols();
    let n_factors = loadings.ncols();

    let lam: Array2<f64> = xi.mapv(lambda_jj);

    let mut second_moments = Array3::<f64>::zeros((n_persons, n_factors, n_factors));
    for i in 0..n_persons {
        for k1 in 0..n_factors {
            for k2 in 0..n_factors {
                second_moments[[i, k1, k2]] = sigma[[i, k1, k2]] + mu[[i, k1]] * mu[[i, k2]];
            }
        }
    }

    let results: Vec<(usize, Vec<f64>, f64)> = (0..n_items)
        .into_par_iter()
        .map(|j| {
            let d_j = intercepts[j];

            let mut a_mat = vec![0.0; n_factors * n_factors];
            let mut b_vec = vec![0.0; n_factors];

            for i in 0..n_persons {
                if responses[[i, j]] < 0 {
                    continue;
                }
                let lam_ij = lam[[i, j]];
                let y_ij = responses[[i, j]] as f64;

                for k1 in 0..n_factors {
                    for k2 in 0..n_factors {
                        a_mat[k1 * n_factors + k2] += 2.0 * lam_ij * second_moments[[i, k1, k2]];
                    }
                }

                let coeff = y_ij - 0.5 - 2.0 * lam_ij * d_j;
                for k in 0..n_factors {
                    b_vec[k] += coeff * mu[[i, k]];
                }
            }

            for k in 0..n_factors {
                a_mat[k * n_factors + k] += 1e-6;
            }

            let a_new = solve_linear_system(&a_mat, &b_vec, n_factors);

            let mut d_num = 0.0;
            let mut d_den = 0.0;
            for i in 0..n_persons {
                if responses[[i, j]] < 0 {
                    continue;
                }
                let lam_ij = lam[[i, j]];
                let y_ij = responses[[i, j]] as f64;

                let mut linear = 0.0;
                for k in 0..n_factors {
                    linear += a_new[k] * mu[[i, k]];
                }

                d_num += y_ij - 0.5 - 2.0 * lam_ij * linear;
                d_den += 2.0 * lam_ij;
            }

            let d_new = if d_den > 1e-10 {
                (d_num / d_den).clamp(-10.0, 10.0)
            } else {
                0.0
            };

            (j, a_new, d_new)
        })
        .collect();

    for (j, a_new, d_new) in results {
        for k in 0..n_factors {
            loadings[[j, k]] = a_new[k];
        }
        intercepts[j] = d_new;
    }

    (loadings.to_pyarray(py), intercepts.to_pyarray(py))
}

/// Compute ELBO (Evidence Lower Bound)
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn gvem_compute_elbo(
    responses: PyReadonlyArray2<i32>,
    loadings: PyReadonlyArray2<f64>,
    intercepts: PyReadonlyArray1<f64>,
    mu: PyReadonlyArray2<f64>,
    sigma: PyReadonlyArray3<f64>,
    xi: PyReadonlyArray2<f64>,
    prior_mean: PyReadonlyArray1<f64>,
    prior_cov: PyReadonlyArray2<f64>,
) -> f64 {
    let responses = responses.as_array();
    let loadings = loadings.as_array();
    let intercepts = intercepts.as_array();
    let mu = mu.as_array();
    let sigma = sigma.as_array();
    let xi = xi.as_array();
    let prior_mean = prior_mean.as_array();
    let prior_cov = prior_cov.as_array();

    let n_persons = responses.nrows();
    let n_items = responses.ncols();
    let n_factors = loadings.ncols();

    let lam: Array2<f64> = xi.mapv(lambda_jj);

    let prior_cov_inv = invert_matrix(&prior_cov.iter().cloned().collect::<Vec<_>>(), n_factors);
    let log_det_prior = log_determinant(&prior_cov.iter().cloned().collect::<Vec<_>>(), n_factors);

    let person_elbos: f64 = (0..n_persons)
        .into_par_iter()
        .map(|i| {
            let mu_i: Vec<f64> = (0..n_factors).map(|k| mu[[i, k]]).collect();
            let sigma_i: Vec<f64> = (0..n_factors)
                .flat_map(|k1| (0..n_factors).map(move |k2| sigma[[i, k1, k2]]))
                .collect();

            let mut second_moment = vec![0.0; n_factors * n_factors];
            for k1 in 0..n_factors {
                for k2 in 0..n_factors {
                    second_moment[k1 * n_factors + k2] =
                        sigma_i[k1 * n_factors + k2] + mu_i[k1] * mu_i[k2];
                }
            }

            let mut elbo_i = 0.0;

            for j in 0..n_items {
                if responses[[i, j]] < 0 {
                    continue;
                }
                let y_ij = responses[[i, j]] as f64;
                let xi_ij = xi[[i, j]];
                let lam_ij = lam[[i, j]];
                let d_j = intercepts[j];

                let mut eta_mean = d_j;
                for k in 0..n_factors {
                    eta_mean += loadings[[j, k]] * mu_i[k];
                }

                let mut eta_second = d_j * d_j;
                for k1 in 0..n_factors {
                    for k2 in 0..n_factors {
                        eta_second += loadings[[j, k1]]
                            * second_moment[k1 * n_factors + k2]
                            * loadings[[j, k2]];
                    }
                }
                for k in 0..n_factors {
                    eta_second += 2.0 * d_j * loadings[[j, k]] * mu_i[k];
                }

                let log_sigmoid_xi = -(-xi_ij).exp().ln_1p();

                elbo_i += log_sigmoid_xi + (y_ij - 0.5) * eta_mean
                    - 0.5 * xi_ij
                    - lam_ij * (eta_second - xi_ij * xi_ij);
            }

            let mut diff = vec![0.0; n_factors];
            for k in 0..n_factors {
                diff[k] = mu_i[k] - prior_mean[k];
            }

            let mut kl_mean = 0.0;
            for k1 in 0..n_factors {
                for k2 in 0..n_factors {
                    kl_mean += diff[k1] * prior_cov_inv[k1 * n_factors + k2] * diff[k2];
                }
            }
            kl_mean *= 0.5;

            let mut kl_trace = 0.0;
            for k1 in 0..n_factors {
                for k2 in 0..n_factors {
                    kl_trace += prior_cov_inv[k1 * n_factors + k2] * sigma_i[k1 * n_factors + k2];
                }
            }
            kl_trace *= 0.5;

            let log_det_q = log_determinant(&sigma_i, n_factors);
            let kl_logdet = 0.5 * (log_det_prior - log_det_q);

            let kl = kl_mean + kl_trace + kl_logdet - 0.5 * n_factors as f64;

            elbo_i - kl
        })
        .sum();

    person_elbos
}

fn invert_matrix(mat: &[f64], n: usize) -> Vec<f64> {
    let mut result = vec![0.0; n * n];
    let mut work = mat.to_vec();

    for i in 0..n {
        result[i * n + i] = 1.0;
    }

    for i in 0..n {
        let pivot = work[i * n + i];
        if pivot.abs() < 1e-10 {
            continue;
        }

        for j in 0..n {
            work[i * n + j] /= pivot;
            result[i * n + j] /= pivot;
        }

        for k in 0..n {
            if k == i {
                continue;
            }
            let factor = work[k * n + i];
            for j in 0..n {
                work[k * n + j] -= factor * work[i * n + j];
                result[k * n + j] -= factor * result[i * n + j];
            }
        }
    }

    result
}

fn solve_linear_system(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let a_inv = invert_matrix(a, n);
    let mut x = vec![0.0; n];
    for i in 0..n {
        for j in 0..n {
            x[i] += a_inv[i * n + j] * b[j];
        }
    }
    x
}

fn log_determinant(mat: &[f64], n: usize) -> f64 {
    let mut work = mat.to_vec();
    let mut log_det = 0.0;

    for i in 0..n {
        let pivot = work[i * n + i];
        if pivot.abs() < 1e-10 {
            return f64::NEG_INFINITY;
        }
        log_det += pivot.abs().ln();

        for j in (i + 1)..n {
            let factor = work[j * n + i] / pivot;
            for k in i..n {
                work[j * n + k] -= factor * work[i * n + k];
            }
        }
    }

    log_det
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(gvem_e_step, m)?)?;
    m.add_function(wrap_pyfunction!(gvem_m_step, m)?)?;
    m.add_function(wrap_pyfunction!(gvem_compute_elbo, m)?)?;
    Ok(())
}
