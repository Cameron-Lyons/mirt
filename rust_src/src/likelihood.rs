//! Binary item log-probability rows shared by the likelihood kernels.

use crate::utils::{EPSILON, log_sigmoid, sigmoid};

/// Fill a binary row with the 2PL log probabilities `[log(1 - p), log(p)]` of logit `z`.
#[inline]
pub(crate) fn log_2pl_row(z: f64, row: &mut [f64]) {
    row[0] = log_sigmoid(-z);
    row[1] = log_sigmoid(z);
}

/// Fill a binary row with clipped 3PL log probabilities for logit `z`.
#[inline]
pub(crate) fn log_3pl_row(z: f64, guessing: f64, row: &mut [f64]) {
    let p = (guessing + (1.0 - guessing) * sigmoid(z)).clamp(EPSILON, 1.0 - EPSILON);
    row[0] = (1.0 - p).ln();
    row[1] = p.ln();
}
