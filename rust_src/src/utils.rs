//! Core utility functions for IRT computations.

use numpy::ndarray::ArrayView2;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand::{Rng, RngExt};

pub(crate) const LOG_2_PI: f64 = 1.8378770664093453;
pub(crate) const EPSILON: f64 = 1e-10;

/// Normal-distribution sampler backed only by the core `rand` crate.
///
/// Marsaglia's polar method produces samples in pairs. Caching the second
/// sample keeps repeated MCMC and imputation draws efficient while avoiding a
/// separate distribution dependency.
#[derive(Clone, Debug)]
pub(crate) struct NormalSampler {
    mean: f64,
    std_dev: f64,
    spare: Option<f64>,
}

impl NormalSampler {
    pub fn new(mean: f64, std_dev: f64) -> Self {
        assert!(mean.is_finite(), "normal mean must be finite");
        assert!(
            std_dev.is_finite() && std_dev >= 0.0,
            "normal standard deviation must be finite and non-negative"
        );
        Self {
            mean,
            std_dev,
            spare: None,
        }
    }

    #[inline]
    pub fn sample<R: Rng + ?Sized>(&mut self, rng: &mut R) -> f64 {
        if self.std_dev == 0.0 {
            return self.mean;
        }

        if let Some(standard_normal) = self.spare.take() {
            return self.mean + self.std_dev * standard_normal;
        }

        loop {
            let u = 2.0 * rng.random::<f64>() - 1.0;
            let v = 2.0 * rng.random::<f64>() - 1.0;
            let radius_squared = u * u + v * v;
            if radius_squared == 0.0 || radius_squared >= 1.0 {
                continue;
            }

            let scale = (-2.0 * radius_squared.ln() / radius_squared).sqrt();
            self.spare = Some(v * scale);
            return self.mean + self.std_dev * u * scale;
        }
    }
}

#[inline]
pub(crate) fn logsumexp(arr: &[f64]) -> f64 {
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
pub(crate) fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let exp_x = x.exp();
        exp_x / (1.0 + exp_x)
    }
}

#[inline]
pub(crate) fn log_sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        -(-x).exp().ln_1p()
    } else {
        x - x.exp().ln_1p()
    }
}

/// Normalize log posterior values and return probabilities
#[inline]
pub(crate) fn normalize_log_posterior(log_posterior: &[f64]) -> Vec<f64> {
    let log_norm = logsumexp(log_posterior);
    log_posterior
        .iter()
        .map(|lp| (lp - log_norm).exp())
        .collect()
}

/// Compute EAP estimate and standard error from posterior and quadrature points
#[inline]
pub(crate) fn compute_eap_with_se(posterior: &[f64], quad_points: &[f64]) -> (f64, f64) {
    let mut theta_eap = 0.0;
    for (p, &theta) in posterior.iter().zip(quad_points.iter()) {
        theta_eap += p * theta;
    }

    let mut variance = 0.0;
    for (p, &theta) in posterior.iter().zip(quad_points.iter()) {
        let diff = theta - theta_eap;
        variance += p * diff * diff;
    }

    (theta_eap, variance.sqrt())
}

/// Compute log weights without inflating small Gauss-Hermite masses.
#[inline]
pub(crate) fn compute_log_weights(weights: &[f64]) -> Vec<f64> {
    weights
        .iter()
        .map(|&w| w.max(f64::MIN_POSITIVE).ln())
        .collect()
}

/// Density-ratio adjustment for normalized Gaussian masses on standard-normal GH nodes.
pub(crate) fn normalized_log_gaussian_adjustment(
    quad_points: &[f64],
    quad_weights: &[f64],
    prior_mean: f64,
    prior_var: f64,
) -> Vec<f64> {
    assert_eq!(quad_points.len(), quad_weights.len());
    assert!(prior_mean.is_finite());
    assert!(prior_var.is_finite() && prior_var > 0.0);

    let adjustment: Vec<f64> = quad_points
        .iter()
        .map(|&theta| {
            let centered = theta - prior_mean;
            let log_target = -0.5 * (LOG_2_PI + prior_var.ln() + centered * centered / prior_var);
            let log_reference = -0.5 * (LOG_2_PI + theta * theta);
            log_target - log_reference
        })
        .collect();
    let log_weights = compute_log_weights(quad_weights);
    let log_total = logsumexp(
        &log_weights
            .iter()
            .zip(adjustment.iter())
            .map(|(log_weight, correction)| log_weight + correction)
            .collect::<Vec<_>>(),
    );

    adjustment
        .into_iter()
        .map(|correction| correction - log_total)
        .collect()
}

/// Floor a probability at `EPSILON`, keeping NaN: `f64::max` would turn the
/// NaN of a non-finite parameter into a valid-looking `EPSILON`.
#[inline]
pub(crate) fn floor_probability(p: f64) -> f64 {
    if p.is_nan() { p } else { p.max(EPSILON) }
}

#[inline]
pub(crate) fn grm_category_probability(
    theta: f64,
    discrimination: f64,
    thresholds: &[f64],
    category: usize,
    n_categories: usize,
) -> f64 {
    if category == 0 {
        let z = discrimination * (theta - thresholds[0]);
        let p_above = sigmoid(z);
        floor_probability(1.0 - p_above)
    } else if category == n_categories - 1 {
        let z = discrimination * (theta - thresholds[category - 1]);
        let p_above = sigmoid(z);
        floor_probability(p_above)
    } else {
        let z_upper = discrimination * (theta - thresholds[category - 1]);
        let z_lower = discrimination * (theta - thresholds[category]);
        let p_upper = sigmoid(z_upper);
        let p_lower = sigmoid(z_lower);
        floor_probability(p_upper - p_lower)
    }
}

/// SplitMix64 finalizer (Steele, Lea and Flood, 2014): a bijective 64-bit mix.
#[inline]
pub(crate) fn splitmix64(value: u64) -> u64 {
    let mut z = value.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Seed of the random stream that `keys` (such as a task's indices) select
/// under `base`. Mixing every key keeps the streams of neighbouring bases and
/// indices unrelated, unlike `base + index`, under which seed `s + 1` replays
/// the streams of seed `s` shifted by one task.
pub(crate) fn stream_seed(base: u64, keys: &[u64]) -> u64 {
    // XOR-ing two hashes would be symmetric in `base` and a key.
    keys.iter()
        .fold(splitmix64(base), |state, &key| splitmix64(state ^ key))
}

/// Reject observed binary responses other than 0 and 1; negative codes are
/// missing.
pub(crate) fn check_binary_responses(responses: ArrayView2<'_, i32>) -> PyResult<()> {
    if responses.iter().any(|&response| response > 1) {
        return Err(PyValueError::new_err(
            "observed responses must contain only 0 or 1",
        ));
    }
    Ok(())
}

/// Check a quadrature rule: matching non-empty lengths, finite points and
/// non-negative finite weights with a positive total.
pub(crate) fn check_quadrature(points: &[f64], weights: &[f64]) -> PyResult<()> {
    if points.is_empty() {
        return Err(PyValueError::new_err(
            "quad_points must be a non-empty one-dimensional array",
        ));
    }
    if weights.len() != points.len() {
        return Err(PyValueError::new_err(format!(
            "quad_weights must have shape ({},), got ({},)",
            points.len(),
            weights.len()
        )));
    }
    if !points.iter().all(|value| value.is_finite()) {
        return Err(PyValueError::new_err("quad_points must be finite"));
    }
    if !weights
        .iter()
        .all(|&value| value.is_finite() && value >= 0.0)
        || weights.iter().sum::<f64>() <= 0.0
    {
        return Err(PyValueError::new_err(
            "quad_weights must be finite and non-negative with a positive sum",
        ));
    }
    Ok(())
}

/// Check that a `(lower, upper)` box is ordered; `f64::clamp` panics on NaN
/// or inverted bounds.
pub(crate) fn check_bounds(name: &str, (lower, upper): (f64, f64)) -> PyResult<()> {
    if lower.is_nan() || upper.is_nan() || lower > upper {
        return Err(PyValueError::new_err(format!(
            "{name} must satisfy lower <= upper"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{RngExt, SeedableRng, rngs::StdRng};

    #[test]
    fn sigmoid_at_zero_is_half() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn sigmoid_saturates_at_extremes() {
        assert!((sigmoid(50.0) - 1.0).abs() < 1e-12);
        assert!(sigmoid(-50.0).abs() < 1e-12);
    }

    #[test]
    fn logsumexp_of_equal_values() {
        let expected = 1.0 + 3.0_f64.ln();
        assert!((logsumexp(&[1.0, 1.0, 1.0]) - expected).abs() < 1e-12);
    }

    #[test]
    fn normalized_gaussian_adjustment_preserves_reference_mass() {
        let points = [-1.0, 0.0, 1.0];
        let weights = [0.25, 0.5, 0.25];
        let adjustment = normalized_log_gaussian_adjustment(&points, &weights, 0.0, 1.0);
        assert!(adjustment.iter().all(|value| value.abs() < 1e-15));
        let total: f64 = compute_log_weights(&weights)
            .iter()
            .zip(adjustment.iter())
            .map(|(log_weight, correction)| (log_weight + correction).exp())
            .sum();
        assert!((total - 1.0).abs() < 1e-15);
    }

    #[test]
    fn shifted_gaussian_adjustment_is_normalized() {
        let points = [-2.0, -1.0, 0.0, 1.0, 2.0];
        let weights = [0.05, 0.2, 0.5, 0.2, 0.05];
        let adjustment = normalized_log_gaussian_adjustment(&points, &weights, 0.7, 0.8);
        let total: f64 = compute_log_weights(&weights)
            .iter()
            .zip(adjustment.iter())
            .map(|(log_weight, correction)| (log_weight + correction).exp())
            .sum();
        assert!((total - 1.0).abs() < 1e-14);
    }

    #[test]
    fn probability_floor_keeps_nan() {
        assert!(floor_probability(f64::NAN).is_nan());
        assert_eq!(floor_probability(0.0), EPSILON);
        assert_eq!(floor_probability(0.25), 0.25);
        assert!(grm_category_probability(0.0, f64::NAN, &[0.0], 1, 2).is_nan());
        assert!(grm_category_probability(0.0, 1.0, &[f64::NAN, 1.0], 1, 3).is_nan());
    }

    #[test]
    fn stream_seeds_differ_across_neighbouring_bases_and_keys() {
        let mut seeds: Vec<u64> = (0..50u64)
            .flat_map(|base| (0..50u64).map(move |key| stream_seed(base, &[key])))
            .collect();
        seeds.sort_unstable();
        seeds.dedup();
        assert_eq!(seeds.len(), 2500);
        // Replicate b of seed s + 1 is not replicate b + 1 of seed s.
        assert_ne!(stream_seed(11, &[0]), stream_seed(10, &[1]));
        assert_ne!(stream_seed(7, &[1, 0]), stream_seed(7, &[0, 1]));
        assert_eq!(stream_seed(7, &[3, 4]), stream_seed(7, &[3, 4]));
    }

    #[test]
    fn normal_sampler_is_reproducible() {
        let mut first_rng = StdRng::seed_from_u64(42);
        let mut second_rng = StdRng::seed_from_u64(42);
        let mut first = NormalSampler::new(1.5, 0.75);
        let mut second = NormalSampler::new(1.5, 0.75);

        let first_samples: Vec<f64> = (0..16).map(|_| first.sample(&mut first_rng)).collect();
        let second_samples: Vec<f64> = (0..16).map(|_| second.sample(&mut second_rng)).collect();

        assert_eq!(first_samples, second_samples);
    }

    #[test]
    fn normal_sampler_has_expected_moments() {
        let mut rng = StdRng::seed_from_u64(7);
        let mut normal = NormalSampler::new(2.0, 3.0);
        let samples: Vec<f64> = (0..100_000).map(|_| normal.sample(&mut rng)).collect();
        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        let variance = samples
            .iter()
            .map(|sample| (sample - mean).powi(2))
            .sum::<f64>()
            / samples.len() as f64;

        assert!((mean - 2.0).abs() < 0.04);
        assert!((variance - 9.0).abs() < 0.12);
    }

    #[test]
    fn zero_deviation_normal_does_not_advance_rng() {
        let mut sampled_rng = StdRng::seed_from_u64(99);
        let mut untouched_rng = StdRng::seed_from_u64(99);
        let mut normal = NormalSampler::new(-3.0, 0.0);

        assert_eq!(normal.sample(&mut sampled_rng), -3.0);
        assert_eq!(sampled_rng.random::<u64>(), untouched_rng.random::<u64>());
    }
}
