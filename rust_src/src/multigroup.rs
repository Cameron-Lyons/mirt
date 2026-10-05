//! Multigroup IRT E-step computations.
//!
//! Each group's person-by-grid log-likelihoods come from the shared
//! `cached_likelihoods` table, the same kernel the single-group likelihoods use,
//! and the rows are then normalized in place into posterior weights.

use numpy::ndarray::{Array2, ArrayView, ArrayView1, ArrayView2, Dimension};
use numpy::{
    Element, IntoPyArray, PyArray1, PyArray2, PyReadonlyArray, PyReadonlyArray1, PyReadonlyArray2,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::counts::binary_item_counts;
use crate::likelihood::{log_2pl_row, log_3pl_row};
use crate::likelihood_cache::{cached_likelihoods, normalize_log_posterior_rows};
use crate::polytomous::{
    adjacent_log_row, category_counts, grm_log_row, log_softmax_floor,
    validate_category_parameters, validate_response_categories,
};
use crate::utils::{compute_log_weights, normalized_log_gaussian_adjustment};

type GroupPosteriors<'py> = (Vec<Bound<'py, PyArray2<f64>>>, Bound<'py, PyArray1<f64>>);

/// Borrow every per-group array as an ndarray view.
fn views<'a, T: Element, D: Dimension>(
    arrays: &'a [PyReadonlyArray<'_, T, D>],
) -> Vec<ArrayView<'a, T, D>> {
    arrays.iter().map(|array| array.as_array()).collect()
}

/// Require one entry per group in every per-group argument list.
fn check_group_counts(n_groups: usize, lengths: &[usize]) -> PyResult<()> {
    if lengths.iter().any(|&length| length != n_groups) {
        return Err(PyValueError::new_err(
            "every per-group argument needs one entry per response matrix",
        ));
    }
    Ok(())
}

/// Require one parameter per item in each group.
fn check_item_vectors(
    responses: &[ArrayView2<'_, i32>],
    vectors: &[ArrayView1<'_, f64>],
) -> PyResult<()> {
    if responses
        .iter()
        .zip(vectors)
        .any(|(responses, vector)| vector.len() != responses.ncols())
    {
        return Err(PyValueError::new_err(
            "item parameter vectors need one entry per item",
        ));
    }
    Ok(())
}

/// Posterior weights and summed marginal log-likelihood for every group.
///
/// `log_likelihoods(g, points)` returns group `g`'s person-by-grid log-likelihood
/// table on the quadrature points.
/// The Gaussian prior of each group is applied as normalized quadrature mass.
fn grouped_e_step<'py>(
    py: Python<'py>,
    quad_points: PyReadonlyArray1<f64>,
    quad_weights: PyReadonlyArray1<f64>,
    prior_means: PyReadonlyArray1<f64>,
    prior_vars: PyReadonlyArray1<f64>,
    n_groups: usize,
    log_likelihoods: impl Fn(usize, &[f64]) -> Array2<f64> + Sync,
) -> PyResult<GroupPosteriors<'py>> {
    let quad_points = quad_points.as_array().to_vec();
    let quad_weights = quad_weights.as_array().to_vec();
    let prior_means = prior_means.as_array();
    let prior_vars = prior_vars.as_array();
    if quad_points.len() != quad_weights.len() {
        return Err(PyValueError::new_err(
            "quad_points and quad_weights must have the same length",
        ));
    }
    check_group_counts(n_groups, &[prior_means.len(), prior_vars.len()])?;
    if prior_means.iter().any(|mean| !mean.is_finite())
        || prior_vars
            .iter()
            .any(|variance| !variance.is_finite() || *variance <= 0.0)
    {
        return Err(PyValueError::new_err(
            "group prior means must be finite and variances finite and positive",
        ));
    }

    let (posteriors, group_lls): (Vec<Array2<f64>>, Vec<f64>) = py.detach(|| {
        let log_weights = compute_log_weights(&quad_weights);
        (0..n_groups)
            .map(|g| {
                let log_prior = normalized_log_gaussian_adjustment(
                    &quad_points,
                    &quad_weights,
                    prior_means[g],
                    prior_vars[g],
                );
                let mut posterior = log_likelihoods(g, &quad_points);
                let marginal =
                    normalize_log_posterior_rows(&mut posterior, Some(&log_prior), &log_weights);
                (posterior, marginal.iter().sum::<f64>())
            })
            .unzip()
    });
    Ok((
        posteriors
            .into_iter()
            .map(|posterior| posterior.into_pyarray(py))
            .collect(),
        group_lls.into_pyarray(py),
    ))
}

/// Compute the multigroup E-step for 2PL models.
///
/// Returns each group's `(n_persons_g, n_quad)` posterior weights and the
/// `(n_groups,)` marginal log-likelihoods.
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (responses_list, quad_points, quad_weights, disc_list, diff_list, prior_means, prior_vars))]
pub fn multigroup_e_step_2pl<'py>(
    py: Python<'py>,
    responses_list: Vec<PyReadonlyArray2<i32>>,
    quad_points: PyReadonlyArray1<f64>,
    quad_weights: PyReadonlyArray1<f64>,
    disc_list: Vec<PyReadonlyArray1<f64>>,
    diff_list: Vec<PyReadonlyArray1<f64>>,
    prior_means: PyReadonlyArray1<f64>,
    prior_vars: PyReadonlyArray1<f64>,
) -> PyResult<GroupPosteriors<'py>> {
    let responses = views(&responses_list);
    let discrimination = views(&disc_list);
    let difficulty = views(&diff_list);
    check_group_counts(responses.len(), &[discrimination.len(), difficulty.len()])?;
    check_item_vectors(&responses, &discrimination)?;
    check_item_vectors(&responses, &difficulty)?;
    grouped_e_step(
        py,
        quad_points,
        quad_weights,
        prior_means,
        prior_vars,
        responses.len(),
        |g, points| {
            let (a, b) = (&discrimination[g], &difficulty[g]);
            let categories = vec![2; a.len()];
            cached_likelihoods(
                responses[g],
                points.len(),
                &categories,
                true,
                |q, j, row| {
                    log_2pl_row(a[j] * (points[q] - b[j]), row);
                },
            )
        },
    )
}

/// Compute the multigroup E-step for 3PL models.
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (responses_list, quad_points, quad_weights, disc_list, diff_list, guess_list, prior_means, prior_vars))]
pub fn multigroup_e_step_3pl<'py>(
    py: Python<'py>,
    responses_list: Vec<PyReadonlyArray2<i32>>,
    quad_points: PyReadonlyArray1<f64>,
    quad_weights: PyReadonlyArray1<f64>,
    disc_list: Vec<PyReadonlyArray1<f64>>,
    diff_list: Vec<PyReadonlyArray1<f64>>,
    guess_list: Vec<PyReadonlyArray1<f64>>,
    prior_means: PyReadonlyArray1<f64>,
    prior_vars: PyReadonlyArray1<f64>,
) -> PyResult<GroupPosteriors<'py>> {
    let responses = views(&responses_list);
    let discrimination = views(&disc_list);
    let difficulty = views(&diff_list);
    let guessing = views(&guess_list);
    check_group_counts(
        responses.len(),
        &[discrimination.len(), difficulty.len(), guessing.len()],
    )?;
    check_item_vectors(&responses, &discrimination)?;
    check_item_vectors(&responses, &difficulty)?;
    check_item_vectors(&responses, &guessing)?;
    grouped_e_step(
        py,
        quad_points,
        quad_weights,
        prior_means,
        prior_vars,
        responses.len(),
        |g, points| {
            let (a, b, c) = (&discrimination[g], &difficulty[g], &guessing[g]);
            let categories = vec![2; a.len()];
            cached_likelihoods(
                responses[g],
                points.len(),
                &categories,
                true,
                |q, j, row| {
                    log_3pl_row(a[j] * (points[q] - b[j]), c[j], row);
                },
            )
        },
    )
}

/// Validated category counts of every group, one per item.
fn group_categories(
    responses: &[ArrayView2<'_, i32>],
    n_categories_list: Vec<PyReadonlyArray1<i32>>,
    parameters: &[&[ArrayView2<'_, f64>]],
    n_parameters: impl Fn(usize) -> usize,
    includes_zero: bool,
) -> PyResult<Vec<Vec<usize>>> {
    let categories: Vec<Vec<usize>> = n_categories_list.into_iter().map(category_counts).collect();
    let mut lengths = vec![categories.len()];
    lengths.extend(parameters.iter().map(|list| list.len()));
    check_group_counts(responses.len(), &lengths)?;
    for (g, (responses, categories)) in responses.iter().zip(&categories).enumerate() {
        for list in parameters {
            validate_category_parameters(
                responses.ncols(),
                categories,
                n_parameters(g),
                list[g],
                includes_zero,
            )?;
        }
        validate_response_categories(*responses, categories)?;
    }
    Ok(categories)
}

/// Compute the multigroup E-step for GRM models.
///
/// `thresh_list` holds `(n_items, max_categories - 1)` threshold matrices.
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (responses_list, quad_points, quad_weights, disc_list, thresh_list, n_categories_list, prior_means, prior_vars))]
pub fn multigroup_e_step_grm<'py>(
    py: Python<'py>,
    responses_list: Vec<PyReadonlyArray2<i32>>,
    quad_points: PyReadonlyArray1<f64>,
    quad_weights: PyReadonlyArray1<f64>,
    disc_list: Vec<PyReadonlyArray1<f64>>,
    thresh_list: Vec<PyReadonlyArray2<f64>>,
    n_categories_list: Vec<PyReadonlyArray1<i32>>,
    prior_means: PyReadonlyArray1<f64>,
    prior_vars: PyReadonlyArray1<f64>,
) -> PyResult<GroupPosteriors<'py>> {
    let responses = views(&responses_list);
    let discrimination = views(&disc_list);
    let thresholds = views(&thresh_list);
    check_group_counts(responses.len(), &[discrimination.len()])?;
    let categories = group_categories(
        &responses,
        n_categories_list,
        &[thresholds.as_slice()],
        |g| discrimination[g].len(),
        false,
    )?;
    let thresholds: Vec<Vec<Vec<f64>>> = thresholds
        .iter()
        .map(|matrix| matrix.rows().into_iter().map(|row| row.to_vec()).collect())
        .collect();
    grouped_e_step(
        py,
        quad_points,
        quad_weights,
        prior_means,
        prior_vars,
        responses.len(),
        |g, points| {
            let (a, b) = (&discrimination[g], &thresholds[g]);
            cached_likelihoods(
                responses[g],
                points.len(),
                &categories[g],
                false,
                |q, j, row| {
                    grm_log_row(points[q], a[j], &b[j], row);
                },
            )
        },
    )
}

/// Compute the multigroup E-step for GPCM and PCM models.
///
/// `steps_list` holds `(n_items, max_categories)` step matrices whose column 0
/// belongs to category 0 and is ignored.
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (responses_list, quad_points, quad_weights, disc_list, steps_list, n_categories_list, prior_means, prior_vars))]
pub fn multigroup_e_step_gpcm<'py>(
    py: Python<'py>,
    responses_list: Vec<PyReadonlyArray2<i32>>,
    quad_points: PyReadonlyArray1<f64>,
    quad_weights: PyReadonlyArray1<f64>,
    disc_list: Vec<PyReadonlyArray1<f64>>,
    steps_list: Vec<PyReadonlyArray2<f64>>,
    n_categories_list: Vec<PyReadonlyArray1<i32>>,
    prior_means: PyReadonlyArray1<f64>,
    prior_vars: PyReadonlyArray1<f64>,
) -> PyResult<GroupPosteriors<'py>> {
    let responses = views(&responses_list);
    let discrimination = views(&disc_list);
    let steps = views(&steps_list);
    check_group_counts(responses.len(), &[discrimination.len()])?;
    let categories = group_categories(
        &responses,
        n_categories_list,
        &[steps.as_slice()],
        |g| discrimination[g].len(),
        true,
    )?;
    grouped_e_step(
        py,
        quad_points,
        quad_weights,
        prior_means,
        prior_vars,
        responses.len(),
        |g, points| {
            let (a, steps) = (&discrimination[g], &steps[g]);
            cached_likelihoods(
                responses[g],
                points.len(),
                &categories[g],
                false,
                |q, j, row| {
                    adjacent_log_row(row, |k| a[j] * (points[q] - steps[[j, k]]));
                },
            )
        },
    )
}

/// Compute the multigroup E-step for nominal response models.
///
/// Category `k` of item `j` has logit `slopes[j, k] * theta + intercepts[j, k]`;
/// both matrices are `(n_items, max_categories)`.
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (responses_list, quad_points, quad_weights, slopes_list, intercepts_list, n_categories_list, prior_means, prior_vars))]
pub fn multigroup_e_step_nrm<'py>(
    py: Python<'py>,
    responses_list: Vec<PyReadonlyArray2<i32>>,
    quad_points: PyReadonlyArray1<f64>,
    quad_weights: PyReadonlyArray1<f64>,
    slopes_list: Vec<PyReadonlyArray2<f64>>,
    intercepts_list: Vec<PyReadonlyArray2<f64>>,
    n_categories_list: Vec<PyReadonlyArray1<i32>>,
    prior_means: PyReadonlyArray1<f64>,
    prior_vars: PyReadonlyArray1<f64>,
) -> PyResult<GroupPosteriors<'py>> {
    let responses = views(&responses_list);
    let slopes = views(&slopes_list);
    let intercepts = views(&intercepts_list);
    let categories = group_categories(
        &responses,
        n_categories_list,
        &[slopes.as_slice(), intercepts.as_slice()],
        |g| responses[g].ncols(),
        true,
    )?;
    grouped_e_step(
        py,
        quad_points,
        quad_weights,
        prior_means,
        prior_vars,
        responses.len(),
        |g, points| {
            let (slopes, intercepts) = (&slopes[g], &intercepts[g]);
            cached_likelihoods(
                responses[g],
                points.len(),
                &categories[g],
                false,
                |q, j, row| {
                    for (k, value) in row.iter_mut().enumerate() {
                        *value = slopes[[j, k]] * points[q] + intercepts[[j, k]];
                    }
                    log_softmax_floor(row);
                },
            )
        },
    )
}

/// Compute expected dichotomous counts for every group (for the M-step).
///
/// Returns per-group `(n_items, n_quad)` matrices of expected correct (`r_k`)
/// and expected observed (`n_k`) responses.
#[allow(clippy::type_complexity)]
#[pyfunction]
#[pyo3(signature = (responses_list, posterior_weights_list))]
pub fn multigroup_expected_counts<'py>(
    py: Python<'py>,
    responses_list: Vec<PyReadonlyArray2<i32>>,
    posterior_weights_list: Vec<PyReadonlyArray2<f64>>,
) -> PyResult<(
    Vec<Bound<'py, PyArray2<f64>>>,
    Vec<Bound<'py, PyArray2<f64>>>,
)> {
    let responses = views(&responses_list);
    let posteriors = views(&posterior_weights_list);
    check_group_counts(responses.len(), &[posteriors.len()])?;
    if responses
        .iter()
        .zip(&posteriors)
        .any(|(responses, posterior)| responses.nrows() != posterior.nrows())
    {
        return Err(PyValueError::new_err(
            "each posterior needs one row per response row",
        ));
    }

    let counts: Vec<(Array2<f64>, Array2<f64>)> = py.detach(|| {
        responses
            .iter()
            .zip(&posteriors)
            .map(|(&responses, posterior)| {
                let posterior = posterior.as_standard_layout();
                let rows = posterior.as_slice().expect("standard layout posterior");
                let (n_items, n_quad) = (responses.ncols(), posterior.ncols());
                let (r_k, n_k): (Vec<Vec<f64>>, Vec<Vec<f64>>) = (0..n_items)
                    .into_par_iter()
                    .map(|j| binary_item_counts(responses, rows, n_quad, j, None))
                    .unzip();
                let matrix = |rows: Vec<Vec<f64>>| {
                    Array2::from_shape_vec((n_items, n_quad), rows.concat())
                        .expect("one count row per item")
                };
                (matrix(r_k), matrix(n_k))
            })
            .collect()
    });
    Ok(counts
        .into_iter()
        .map(|(r_k, n_k)| (r_k.into_pyarray(py), n_k.into_pyarray(py)))
        .unzip())
}

/// Register multigroup functions with the Python module
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(multigroup_e_step_2pl, m)?)?;
    m.add_function(wrap_pyfunction!(multigroup_e_step_3pl, m)?)?;
    m.add_function(wrap_pyfunction!(multigroup_e_step_grm, m)?)?;
    m.add_function(wrap_pyfunction!(multigroup_e_step_gpcm, m)?)?;
    m.add_function(wrap_pyfunction!(multigroup_e_step_nrm, m)?)?;
    m.add_function(wrap_pyfunction!(multigroup_expected_counts, m)?)?;
    Ok(())
}
