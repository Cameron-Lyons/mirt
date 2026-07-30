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

/// Compute log weights from weights with EPSILON protection
#[inline]
pub fn compute_log_weights(weights: &[f64]) -> Vec<f64> {
    weights.iter().map(|&w| (w + EPSILON).ln()).collect()
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
    match n {
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
        // Three items at P=0.5 => 3 * ln(0.5)
        assert!((ll - 3.0 * 0.5_f64.ln()).abs() < 1e-10);
    }

    #[test]
    fn fisher_info_2pl_at_difficulty() {
        // At theta = b with a = 1: I = a^2 * p * (1-p) = 0.25 per item
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
