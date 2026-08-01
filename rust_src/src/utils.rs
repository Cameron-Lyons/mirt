//! Core utility functions for IRT computations.

use ndarray::ArrayView1;
use rand::{Rng, RngExt};

pub const LOG_2_PI: f64 = 1.8378770664093453;
pub const EPSILON: f64 = 1e-10;

/// Normal-distribution sampler backed only by the core `rand` crate.
///
/// Marsaglia's polar method produces samples in pairs. Caching the second
/// sample keeps repeated MCMC and imputation draws efficient while avoiding a
/// separate distribution dependency.
#[derive(Clone, Debug)]
pub struct NormalSampler {
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
pub fn logsumexp(arr: &[f64]) -> f64 {
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
pub fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let exp_x = x.exp();
        exp_x / (1.0 + exp_x)
    }
}

#[inline]
pub fn log_sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        -(-x).exp().ln_1p()
    } else {
        x - x.exp().ln_1p()
    }
}

#[inline]
pub fn clip(x: f64, min: f64, max: f64) -> f64 {
    x.max(min).min(max)
}

#[inline]
pub fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + libm::erf(x / std::f64::consts::SQRT_2))
}

#[inline]
pub fn log_likelihood_2pl_single(
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
pub fn log_likelihood_2pl_view(
    responses: ArrayView1<i32>,
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
pub fn log_likelihood_3pl_single(
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

#[inline]
pub fn log_likelihood_3pl_view(
    responses: ArrayView1<i32>,
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

/// Normalize log posterior values and return probabilities
#[inline]
pub fn normalize_log_posterior(log_posterior: &[f64]) -> Vec<f64> {
    let log_norm = logsumexp(log_posterior);
    log_posterior
        .iter()
        .map(|lp| (lp - log_norm).exp())
        .collect()
}

/// Compute EAP estimate and standard error from posterior and quadrature points
#[inline]
pub fn compute_eap_with_se(posterior: &[f64], quad_points: &[f64]) -> (f64, f64) {
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
pub fn compute_log_weights(weights: &[f64]) -> Vec<f64> {
    weights
        .iter()
        .map(|&w| w.max(f64::MIN_POSITIVE).ln())
        .collect()
}

/// Density-ratio adjustment for normalized Gaussian masses on standard-normal GH nodes.
pub fn normalized_log_gaussian_adjustment(
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

/// Compute Fisher information for 2PL at a single theta
#[inline]
pub fn fisher_info_2pl(theta: f64, discrimination: &[f64], difficulty: &[f64]) -> f64 {
    let mut info = 0.0;
    for (a, b) in discrimination.iter().zip(difficulty.iter()) {
        let p = sigmoid(a * (theta - b));
        info += a * a * p * (1.0 - p);
    }
    info
}

/// Compute Fisher information for each item at a single theta
#[inline]
pub fn fisher_info_2pl_items(theta: f64, discrimination: &[f64], difficulty: &[f64]) -> Vec<f64> {
    discrimination
        .iter()
        .zip(difficulty.iter())
        .map(|(a, b)| {
            let p = sigmoid(a * (theta - b));
            a * a * p * (1.0 - p)
        })
        .collect()
}

#[inline]
pub fn grm_category_probability(
    theta: f64,
    discrimination: f64,
    thresholds: &[f64],
    category: usize,
    n_categories: usize,
) -> f64 {
    if category == 0 {
        let z = discrimination * (theta - thresholds[0]);
        let p_above = sigmoid(z);
        (1.0 - p_above).max(EPSILON)
    } else if category == n_categories - 1 {
        let z = discrimination * (theta - thresholds[category - 1]);
        let p_above = sigmoid(z);
        p_above.max(EPSILON)
    } else {
        let z_upper = discrimination * (theta - thresholds[category - 1]);
        let z_lower = discrimination * (theta - thresholds[category]);
        let p_upper = sigmoid(z_upper);
        let p_lower = sigmoid(z_lower);
        (p_upper - p_lower).max(EPSILON)
    }
}

/// Gauss-Hermite quadrature nodes and weights
pub fn gauss_hermite_quadrature(n: usize) -> (Vec<f64>, Vec<f64>) {
    assert!(n > 0, "quadrature requires at least one point");
    if n == 1 {
        return (vec![0.0], vec![1.0]);
    }

    let n_symmetric = n.div_ceil(2);
    let mut nodes = vec![0.0; n];
    let mut weights = vec![0.0; n];
    let mut positive_roots = Vec::with_capacity(n_symmetric);
    let mut root = 0.0;

    for index in 0..n_symmetric {
        root = match index {
            0 => {
                let order = 2.0 * n as f64 + 1.0;
                order.sqrt() - 1.85575 * order.powf(-1.0 / 6.0)
            }
            1 => root - 1.14 * (n as f64).powf(0.426) / root,
            2 => 1.86 * root - 0.86 * positive_roots[0],
            3 => 1.91 * root - 0.91 * positive_roots[1],
            _ => 2.0 * root - positive_roots[index - 2],
        };

        let mut derivative = 0.0;
        for _ in 0..50 {
            let mut polynomial = std::f64::consts::PI.powf(-0.25);
            let mut previous = 0.0;
            for degree in 1..=n {
                let older = previous;
                previous = polynomial;
                polynomial = root * (2.0 / degree as f64).sqrt() * previous
                    - ((degree - 1) as f64 / degree as f64).sqrt() * older;
            }
            derivative = (2.0 * n as f64).sqrt() * previous;
            let old_root = root;
            root -= polynomial / derivative;
            if (root - old_root).abs() < 1e-14 {
                break;
            }
        }

        positive_roots.push(root);
        let symmetric = n - 1 - index;
        nodes[index] = -std::f64::consts::SQRT_2 * root;
        nodes[symmetric] = std::f64::consts::SQRT_2 * root;
        let weight = 2.0 / (derivative * derivative * std::f64::consts::PI.sqrt());
        weights[index] = weight;
        weights[symmetric] = weight;
    }

    let total: f64 = weights.iter().sum();
    for weight in &mut weights {
        *weight /= total;
    }
    (nodes, weights)
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
    fn gauss_hermite_quadrature_has_standard_normal_moments() {
        for order in [9, 15, 21, 31] {
            let (nodes, weights) = gauss_hermite_quadrature(order);
            let mean: f64 = nodes
                .iter()
                .zip(weights.iter())
                .map(|(node, weight)| node * weight)
                .sum();
            let variance: f64 = nodes
                .iter()
                .zip(weights.iter())
                .map(|(node, weight)| node * node * weight)
                .sum();
            assert!((weights.iter().sum::<f64>() - 1.0).abs() < 1e-14);
            assert!(mean.abs() < 1e-14);
            assert!((variance - 1.0).abs() < 1e-13);
        }
    }

    #[test]
    fn clip_bounds_values() {
        assert!((clip(0.5, 0.0, 1.0) - 0.5).abs() < 1e-12);
        assert!((clip(-1.0, 0.0, 1.0) - 0.0).abs() < 1e-12);
        assert!((clip(2.0, 0.0, 1.0) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn log_likelihood_2pl_all_correct_is_finite_negative() {
        let responses = [1, 1, 1];
        let discrimination = [1.0, 1.0, 1.0];
        let difficulty = [0.0, 0.0, 0.0];
        let ll = log_likelihood_2pl_single(&responses, 0.0, &discrimination, &difficulty);
        assert!(ll.is_finite());
        assert!(ll < 0.0);
        assert!((ll - 3.0 * 0.5_f64.ln()).abs() < 1e-10);
    }

    #[test]
    fn fisher_info_2pl_at_difficulty() {
        let info = fisher_info_2pl(0.0, &[1.0, 1.0], &[0.0, 0.0]);
        assert!((info - 0.5).abs() < 1e-12);
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
