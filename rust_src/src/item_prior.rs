//! Item priors of the native M-steps, as `mirt.estimation.priors` defines them.

/// Prior of one item coordinate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum CoordinatePrior {
    None,
    /// `NormalPrior(mu, sigma)`.
    Normal {
        mu: f64,
        sigma: f64,
    },
    /// `LogNormalPrior(mu, sigma)` on the log of a positive coordinate.
    LogNormal {
        mu: f64,
        sigma: f64,
    },
    /// `BetaPrior(alpha, beta)` on `[0, 1]`.
    Beta {
        alpha: f64,
        beta: f64,
    },
}

/// `coefficient / value`, zero for a zero coefficient, as `priors._ratio`.
#[inline]
fn ratio(coefficient: f64, value: f64) -> f64 {
    if coefficient == 0.0 {
        0.0
    } else {
        coefficient / value
    }
}

/// `coefficient * ln(value)`, zero for a zero coefficient, as `scipy.special.xlogy`.
#[inline]
fn coefficient_log(coefficient: f64, value: f64) -> f64 {
    if coefficient == 0.0 {
        0.0
    } else {
        coefficient * value.ln()
    }
}

impl CoordinatePrior {
    /// Decode one `(code, first, second)` row of a prior table: code 0 is no
    /// prior, 1 normal, 2 log-normal and 3 beta with their two parameters.
    /// Returns `None` for an unknown code or invalid parameters.
    pub(crate) fn decode(code: f64, first: f64, second: f64) -> Option<Self> {
        let positive = |value: f64| value.is_finite() && value > 0.0;
        match code {
            0.0 => Some(Self::None),
            1.0 if first.is_finite() && positive(second) => Some(Self::Normal {
                mu: first,
                sigma: second,
            }),
            2.0 if first.is_finite() && positive(second) => Some(Self::LogNormal {
                mu: first,
                sigma: second,
            }),
            3.0 if positive(first) && positive(second) => Some(Self::Beta {
                alpha: first,
                beta: second,
            }),
            _ => None,
        }
    }

    /// Log density up to its normalizing constant; `-inf` outside the support.
    pub(crate) fn log_density(&self, x: f64) -> f64 {
        match *self {
            Self::None => 0.0,
            Self::Normal { mu, sigma } => {
                let standardized = (x - mu) / sigma;
                -0.5 * standardized * standardized
            }
            Self::LogNormal { mu, sigma } => {
                if x > 0.0 {
                    let standardized = (x.ln() - mu) / sigma;
                    -x.ln() - 0.5 * standardized * standardized
                } else {
                    f64::NEG_INFINITY
                }
            }
            Self::Beta { alpha, beta } => {
                if (0.0..=1.0).contains(&x) {
                    coefficient_log(alpha - 1.0, x) + coefficient_log(beta - 1.0, 1.0 - x)
                } else {
                    f64::NEG_INFINITY
                }
            }
        }
    }

    /// First and second derivatives of the log density, as the priors'
    /// `grad_log_pdf` and `hess_log_pdf` give them.
    pub(crate) fn derivatives(&self, x: f64) -> (f64, f64) {
        match *self {
            Self::None => (0.0, 0.0),
            Self::Normal { mu, sigma } => {
                let variance = sigma * sigma;
                ((mu - x) / variance, -1.0 / variance)
            }
            Self::LogNormal { mu, sigma } => {
                let variance = sigma * sigma;
                let standardized = (x.ln() - mu) / variance;
                (
                    -(1.0 + standardized) / x,
                    (1.0 + standardized - 1.0 / variance) / (x * x),
                )
            }
            Self::Beta { alpha, beta } => (
                ratio(alpha - 1.0, x) - ratio(beta - 1.0, 1.0 - x),
                -ratio(alpha - 1.0, x * x) - ratio(beta - 1.0, (1.0 - x) * (1.0 - x)),
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::CoordinatePrior;

    #[test]
    fn derivatives_match_differences_of_the_log_density() {
        let priors = [
            CoordinatePrior::Normal {
                mu: 0.3,
                sigma: 1.7,
            },
            CoordinatePrior::LogNormal {
                mu: 0.0,
                sigma: 0.5,
            },
            CoordinatePrior::Beta {
                alpha: 5.0,
                beta: 17.0,
            },
        ];
        let step = 1e-5;
        for prior in priors {
            for x in [0.1, 0.25, 0.6] {
                let (gradient, second) = prior.derivatives(x);
                let up = prior.log_density(x + step);
                let down = prior.log_density(x - step);
                let centre = prior.log_density(x);
                assert!((gradient - (up - down) / (2.0 * step)).abs() < 1e-5);
                let difference = (up - 2.0 * centre + down) / (step * step);
                assert!((second - difference).abs() < 1e-3 * difference.abs().max(1.0));
            }
        }
    }

    #[test]
    fn decoding_rejects_unknown_codes_and_parameters() {
        assert_eq!(
            CoordinatePrior::decode(0.0, f64::NAN, f64::NAN),
            Some(CoordinatePrior::None)
        );
        assert!(CoordinatePrior::decode(1.0, 0.0, 0.0).is_none());
        assert!(CoordinatePrior::decode(3.0, -1.0, 2.0).is_none());
        assert!(CoordinatePrior::decode(4.0, 1.0, 1.0).is_none());
        let support = CoordinatePrior::decode(2.0, 0.0, 0.5).unwrap();
        assert_eq!(support.log_density(0.0), f64::NEG_INFINITY);
        let beta = CoordinatePrior::decode(3.0, 1.0, 3.0).unwrap();
        assert_eq!(beta.log_density(0.0), 0.0);
        assert_eq!(beta.log_density(1.5), f64::NEG_INFINITY);
    }
}
