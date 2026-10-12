//! Box-constrained Newton ascent shared by the item M-steps.

use crate::utils::EPSILON;

/// Iteration limits, ridge and box of a bounded item M-step.
#[derive(Clone, Copy, Debug)]
pub(crate) struct BoxNewtonControls<const N: usize> {
    pub max_iter: usize,
    /// Largest coordinate change at which the ascent stops.
    pub tol: f64,
    /// Ridge added to the information diagonal; it only stabilizes the steps.
    pub ridge: [f64; N],
    pub lower: [f64; N],
    pub upper: [f64; N],
}

/// Largest distance from a bound at which [`projected_newton`] treats a
/// coordinate as on that bound.
const ACTIVE_BOUND_WIDTH: f64 = 1e-3;

/// Step halvings tried before an iteration gives up.
const MAX_HALVINGS: usize = 12;

/// Solve `information[F, F] step[F] = score[F]` for the free coordinates `F` by
/// Gaussian elimination with partial pivoting; fixed coordinates get a zero step.
fn solve_free_coordinates<const N: usize>(
    information: &[[f64; N]; N],
    score: &[f64; N],
    free: &[bool; N],
) -> Option<[f64; N]> {
    let index: Vec<usize> = (0..N).filter(|&i| free[i]).collect();
    let m = index.len();
    let mut matrix = [[0.0; N]; N];
    let mut rhs = [0.0; N];
    for (row, &i) in index.iter().enumerate() {
        for (col, &k) in index.iter().enumerate() {
            matrix[row][col] = information[i][k];
        }
        rhs[row] = score[i];
    }
    for col in 0..m {
        let pivot =
            (col..m).max_by(|&x, &y| matrix[x][col].abs().total_cmp(&matrix[y][col].abs()))?;
        matrix.swap(col, pivot);
        rhs.swap(col, pivot);
        if !matrix[col][col].is_finite() || matrix[col][col].abs() < EPSILON {
            return None;
        }
        let pivot_row = matrix[col];
        let pivot_rhs = rhs[col];
        for row in col + 1..m {
            let factor = matrix[row][col] / pivot_row[col];
            for (value, &pivot_value) in matrix[row][col..].iter_mut().zip(&pivot_row[col..]) {
                *value -= factor * pivot_value;
            }
            rhs[row] -= factor * pivot_rhs;
        }
    }
    let mut solution = [0.0; N];
    for row in (0..m).rev() {
        let tail: f64 = (row + 1..m).map(|k| matrix[row][k] * solution[k]).sum();
        solution[row] = (rhs[row] - tail) / matrix[row][row];
    }
    let mut step = [0.0; N];
    for (row, &i) in index.iter().enumerate() {
        step[i] = solution[row];
    }
    step.iter().all(|value| value.is_finite()).then_some(step)
}

/// Bounded Newton (Fisher-scoring) ascent of `objective` over the box.
///
/// This is a projected Newton method with an epsilon-active set (Bertsekas,
/// 1982). A coordinate within `min(ACTIVE_BOUND_WIDTH, |x - P(x + score)|)` of
/// a bound is active when its score points out of the box, and steps onto that
/// bound. Free coordinates on such a band whose reduced Newton step points out
/// of the box are held fixed. The remaining coordinates solve the reduced
/// system, so a coordinate held by its bound leaves the others to the
/// lower-dimensional problem instead of an infeasible joint step that is then
/// clipped. Without the band, iterates that stop just inside a bound jam
/// there. Steps are halved until the objective does not drop, so the ascent
/// ends at a stationary point of the box-constrained problem.
///
/// `score_and_information(x)` returns the gradient of `objective` at `x` and a
/// positive-definite curvature, such as the ridge-stabilized Fisher
/// information.
pub(crate) fn projected_newton<const N: usize>(
    start: [f64; N],
    controls: &BoxNewtonControls<N>,
    objective: impl Fn([f64; N]) -> f64,
    score_and_information: impl Fn([f64; N]) -> ([f64; N], [[f64; N]; N]),
) -> [f64; N] {
    let BoxNewtonControls { lower, upper, .. } = *controls;
    let mut x = start;
    for _ in 0..controls.max_iter {
        let (score, information) = score_and_information(x);
        let projected_score = (0..N)
            .map(|i| (x[i] - (x[i] + score[i]).clamp(lower[i], upper[i])).abs())
            .fold(0.0, f64::max);
        let width = ACTIVE_BOUND_WIDTH.min(projected_score);
        let at_lower = |i: usize, direction: f64| x[i] <= lower[i] + width && direction <= 0.0;
        let at_upper = |i: usize, direction: f64| x[i] >= upper[i] - width && direction >= 0.0;
        let blocked = |i: usize, direction: f64| at_lower(i, direction) || at_upper(i, direction);
        let active: [bool; N] = std::array::from_fn(|i| blocked(i, score[i]));
        let mut free = active.map(|is_active| !is_active);
        let step = loop {
            let Some(step) = solve_free_coordinates(&information, &score, &free) else {
                break None;
            };
            let outward: Vec<usize> = (0..N)
                .filter(|&i| free[i] && step[i] != 0.0 && blocked(i, step[i]))
                .collect();
            if outward.is_empty() {
                break Some(step);
            }
            for i in outward {
                free[i] = false;
            }
        };
        let Some(mut step) = step else {
            break;
        };
        for i in (0..N).filter(|&i| active[i]) {
            let bound = if at_lower(i, score[i]) {
                lower[i]
            } else {
                upper[i]
            };
            step[i] = bound - x[i];
        }

        let current = objective(x);
        let mut scale = 1.0;
        let mut accepted = None;
        for _ in 0..MAX_HALVINGS {
            let candidate: [f64; N] =
                std::array::from_fn(|i| (x[i] + scale * step[i]).clamp(lower[i], upper[i]));
            let value = objective(candidate);
            if value.is_finite() && value + EPSILON >= current {
                accepted = Some(candidate);
                break;
            }
            scale *= 0.5;
        }
        let Some(next) = accepted else {
            break;
        };
        let max_change = (0..N).map(|i| (next[i] - x[i]).abs()).fold(0.0, f64::max);
        x = next;
        if max_change < controls.tol {
            break;
        }
    }
    x
}

#[cfg(test)]
mod tests {
    use super::{BoxNewtonControls, projected_newton, solve_free_coordinates};

    #[test]
    fn free_solve_matches_the_reduced_system_and_zeroes_fixed_steps() {
        let information = [[4.0, 1.0, 0.5], [1.0, 3.0, 0.2], [0.5, 0.2, 2.0]];
        let score = [1.0, -2.0, 0.5];
        let step = solve_free_coordinates(&information, &score, &[true, false, true]).unwrap();
        // [[4, 0.5], [0.5, 2]] [s0, s2] = [1, 0.5]
        let det = 4.0 * 2.0 - 0.5 * 0.5;
        assert!((step[0] - (2.0 * 1.0 - 0.5 * 0.5) / det).abs() < 1e-14);
        assert_eq!(step[1], 0.0);
        assert!((step[2] - (4.0 * 0.5 - 0.5 * 1.0) / det).abs() < 1e-14);
        assert!(solve_free_coordinates(&[[0.0; 2]; 2], &[1.0, 1.0], &[true, true]).is_none());
    }

    #[test]
    fn concave_quadratic_ends_at_its_box_optimum() {
        // Maximize -(x - 3)^2 - (y + 1)^2 - x y on [0, 2] x [-5, 5]: the
        // unconstrained optimum x = 14/3 lies outside, so the box optimum has
        // x = 2 and y solving -2 (y + 1) - 2 = 0.
        let controls = BoxNewtonControls {
            max_iter: 50,
            tol: 1e-14,
            ridge: [0.0; 2],
            lower: [0.0, -5.0],
            upper: [2.0, 5.0],
        };
        let objective = |[x, y]: [f64; 2]| -(x - 3.0).powi(2) - (y + 1.0).powi(2) - x * y;
        let derivatives = |[x, y]: [f64; 2]| {
            (
                [-2.0 * (x - 3.0) - y, -2.0 * (y + 1.0) - x],
                [[2.0, 1.0], [1.0, 2.0]],
            )
        };
        let [x, y] = projected_newton([0.5, 4.0], &controls, objective, derivatives);
        assert_eq!(x, 2.0);
        assert!((y + 2.0).abs() < 1e-12, "y = {y}");
    }
}
