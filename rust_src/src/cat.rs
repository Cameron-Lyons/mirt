//! Computerized Adaptive Testing (CAT) functions.

use numpy::ndarray::{Array1, Array2};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand::{prelude::*, rngs::StdRng};
use rayon::prelude::*;

use crate::utils::{
    EPSILON, check_quadrature, compute_eap_with_se, normalize_log_posterior, sigmoid, stream_seed,
};

/// Item parameters for 2PL model
struct ItemBank<'a> {
    discrimination: &'a [f64],
    difficulty: &'a [f64],
}

/// Quadrature parameters for EAP estimation
struct Quadrature<'a> {
    points: &'a [f64],
    weights: &'a [f64],
}

/// CAT stopping criteria
struct StoppingRule {
    se_threshold: f64,
    max_items: usize,
    min_items: usize,
}

/// Result type for batch CAT simulations, including padded administration paths.
type CATBatchResult<'py> = (
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<i32>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray2<i32>>,
    Bound<'py, PyArray2<i32>>,
);
type CATSimulationRecord = (f64, f64, usize, f64, Vec<i32>, Vec<i32>);

/// Result type for conditional MSE: (eval_thetas, bias, mse, avg_items)
type CATMseResult<'py> = (
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
);

/// Validated item bank, grid and stopping controls of one native call.
struct SimulationInputs {
    thetas: Vec<f64>,
    discrimination: Vec<f64>,
    difficulty: Vec<f64>,
    nodes: Vec<f64>,
    weights: Vec<f64>,
    stopping: StoppingRule,
    n_replications: usize,
}

#[allow(clippy::too_many_arguments)]
fn simulation_inputs(
    thetas: PyReadonlyArray1<f64>,
    discrimination: PyReadonlyArray1<f64>,
    difficulty: PyReadonlyArray1<f64>,
    quad_points: PyReadonlyArray1<f64>,
    quad_weights: PyReadonlyArray1<f64>,
    se_threshold: f64,
    max_items: i32,
    min_items: i32,
    n_replications: i32,
) -> PyResult<SimulationInputs> {
    let discrimination = discrimination.as_array().to_vec();
    let difficulty = difficulty.as_array().to_vec();
    if discrimination.len() != difficulty.len() {
        return Err(PyValueError::new_err(
            "discrimination and difficulty must have the same length",
        ));
    }
    let nodes = quad_points.as_array().to_vec();
    let weights = quad_weights.as_array().to_vec();
    check_quadrature(&nodes, &weights)?;
    let count = |value: i32, name: &str| {
        usize::try_from(value)
            .map_err(|_| PyValueError::new_err(format!("{name} must be non-negative")))
    };
    let path_width = count(max_items, "max_items")?.min(discrimination.len());
    if se_threshold.is_nan() || se_threshold < 0.0 {
        return Err(PyValueError::new_err(
            "se_threshold must be a non-negative number",
        ));
    }
    Ok(SimulationInputs {
        thetas: thetas.as_array().to_vec(),
        stopping: StoppingRule {
            se_threshold,
            max_items: path_width,
            min_items: count(min_items, "min_items")?.min(path_width),
        },
        n_replications: count(n_replications, "n_replications")?,
        discrimination,
        difficulty,
        nodes,
        weights,
    })
}

impl SimulationInputs {
    /// Simulate replication `rep` of theta `theta_idx` on its own stream.
    ///
    /// The stream mixes `seed` with both indices, so sessions of one call,
    /// and of neighbouring seeds, are independent; additive seeds such as
    /// `seed + 1000 * theta_idx + rep` replayed each other's streams.
    fn session(
        &self,
        theta_idx: usize,
        rep: usize,
        seed: u64,
    ) -> (f64, f64, usize, Vec<i32>, Vec<i32>) {
        let mut rng = StdRng::seed_from_u64(stream_seed(seed, &[theta_idx as u64, rep as u64]));
        let items = ItemBank {
            discrimination: &self.discrimination,
            difficulty: &self.difficulty,
        };
        let quad = Quadrature {
            points: &self.nodes,
            weights: &self.weights,
        };
        cat_simulate_single(
            self.thetas[theta_idx],
            &items,
            &quad,
            &self.stopping,
            &mut rng,
        )
    }
}

/// Simulate a single CAT session and return results
#[inline]
fn cat_simulate_single(
    true_theta: f64,
    items: &ItemBank,
    quad: &Quadrature,
    stopping: &StoppingRule,
    rng: &mut StdRng,
) -> (f64, f64, usize, Vec<i32>, Vec<i32>) {
    let n_items = items.discrimination.len();

    let mut available: Vec<bool> = vec![true; n_items];
    let mut administered: Vec<i32> = Vec::with_capacity(stopping.max_items);
    let mut responses: Vec<i32> = Vec::with_capacity(stopping.max_items);

    let mut current_theta = 0.0;
    let mut current_se = f64::INFINITY;
    let mut log_posterior: Vec<f64> = quad.weights.iter().map(|weight| weight.ln()).collect();

    for step in 0..stopping.max_items {
        let mut best_item: i32 = -1;
        let mut best_info: f64 = f64::NEG_INFINITY;

        for (j, (&avail, (&disc, &diff))) in available
            .iter()
            .zip(items.discrimination.iter().zip(items.difficulty.iter()))
            .enumerate()
        {
            if !avail {
                continue;
            }
            let z = disc * (current_theta - diff);
            let p = sigmoid(z);
            let info = disc * disc * p * (1.0 - p);

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

        let z = items.discrimination[item_idx] * (true_theta - items.difficulty[item_idx]);
        let p = sigmoid(z);
        let response = if rng.random::<f64>() < p { 1 } else { 0 };

        administered.push(best_item);
        responses.push(response);

        // Only the new response adds evidence. Match the public EAP scorer's
        // bounded Bernoulli probabilities, including saturated response curves.
        for (log_mass, &theta) in log_posterior.iter_mut().zip(quad.points.iter()) {
            let z = items.discrimination[item_idx] * (theta - items.difficulty[item_idx]);
            let probability = sigmoid(z).clamp(EPSILON, 1.0 - EPSILON);
            *log_mass += if response == 1 {
                probability.ln()
            } else {
                (-probability).ln_1p()
            };
        }

        let posterior = normalize_log_posterior(&log_posterior);
        (current_theta, current_se) = compute_eap_with_se(&posterior, quad.points);

        if step + 1 >= stopping.min_items && current_se <= stopping.se_threshold {
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
///
/// Simulates `n_replications` sessions of every theta, in theta-major order,
/// each on its own random stream (see [`SimulationInputs::session`]).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
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
) -> PyResult<CATBatchResult<'py>> {
    let inputs = simulation_inputs(
        true_thetas,
        discrimination,
        difficulty,
        quad_points,
        quad_weights,
        se_threshold,
        max_items,
        min_items,
        n_replications,
    )?;
    let n_reps = inputs.n_replications;
    let n_total = inputs.thetas.len() * n_reps;
    let path_width = inputs.stopping.max_items;

    let results: Vec<CATSimulationRecord> = py.detach(|| {
        (0..n_total)
            .into_par_iter()
            .map(|task| {
                let (theta_idx, rep) = (task / n_reps, task % n_reps);
                let (est_theta, est_se, n_items, administered, responses) =
                    inputs.session(theta_idx, rep, seed);
                (
                    est_theta,
                    est_se,
                    n_items,
                    inputs.thetas[theta_idx],
                    administered,
                    responses,
                )
            })
            .collect()
    });

    let mut theta_est = Array1::zeros(n_total);
    let mut se_est = Array1::zeros(n_total);
    let mut n_items = Array1::zeros(n_total);
    let mut true_theta_out = Array1::zeros(n_total);
    let mut administered_out = Array2::from_elem((n_total, path_width), -1);
    let mut responses_out = Array2::from_elem((n_total, path_width), -1);

    for (i, (t, s, n, tt, administered, responses)) in results.into_iter().enumerate() {
        theta_est[i] = t;
        se_est[i] = s;
        n_items[i] = n as i32;
        true_theta_out[i] = tt;
        for (step, (&item, &response)) in administered.iter().zip(responses.iter()).enumerate() {
            administered_out[[i, step]] = item;
            responses_out[[i, step]] = response;
        }
    }

    Ok((
        theta_est.into_pyarray(py),
        se_est.into_pyarray(py),
        n_items.into_pyarray(py),
        true_theta_out.into_pyarray(py),
        administered_out.into_pyarray(py),
        responses_out.into_pyarray(py),
    ))
}

/// Compute conditional MSE at specified theta values
///
/// Runs `n_replications` sessions of every theta, each on its own random
/// stream as in [`cat_simulate_batch`], and returns each theta's bias, mean
/// squared error and mean test length.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
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
) -> PyResult<CATMseResult<'py>> {
    let inputs = simulation_inputs(
        eval_thetas,
        discrimination,
        difficulty,
        quad_points,
        quad_weights,
        se_threshold,
        max_items,
        min_items,
        n_replications,
    )?;
    let n_reps = inputs.n_replications;
    if n_reps == 0 {
        return Err(PyValueError::new_err("n_replications must be at least 1"));
    }

    let stats: Vec<(f64, f64, f64)> = py.detach(|| {
        (0..inputs.thetas.len())
            .into_par_iter()
            .map(|t_idx| {
                let true_theta = inputs.thetas[t_idx];
                let mut error_sum = 0.0;
                let mut squared_sum = 0.0;
                let mut n_items_sum = 0.0;
                for rep in 0..n_reps {
                    let (est_theta, _, n_items, _, _) = inputs.session(t_idx, rep, seed);
                    error_sum += est_theta - true_theta;
                    squared_sum += (est_theta - true_theta).powi(2);
                    n_items_sum += n_items as f64;
                }
                let n = n_reps as f64;
                (error_sum / n, squared_sum / n, n_items_sum / n)
            })
            .collect()
    });

    let n_thetas = inputs.thetas.len();
    let mut bias = Array1::zeros(n_thetas);
    let mut mse = Array1::zeros(n_thetas);
    let mut avg_items = Array1::zeros(n_thetas);
    for (i, (b, m, a)) in stats.into_iter().enumerate() {
        bias[i] = b;
        mse[i] = m;
        avg_items[i] = a;
    }

    Ok((
        Array1::from(inputs.thetas).into_pyarray(py),
        bias.into_pyarray(py),
        mse.into_pyarray(py),
        avg_items.into_pyarray(py),
    ))
}

/// Register CAT functions with the Python module
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(cat_simulate_batch, m)?)?;
    m.add_function(wrap_pyfunction!(cat_conditional_mse, m)?)?;
    Ok(())
}
