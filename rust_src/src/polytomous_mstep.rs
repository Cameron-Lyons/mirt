//! Batched item-local polytomous optimization with analytic derivatives.

use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::counts::category_item_counts;
use crate::utils::sigmoid;

/// Worker pool whose lifetime is controlled by the Python fit context.
#[pyclass]
pub struct EMThreadPool {
    pool: rayon::ThreadPool,
}

#[pymethods]
impl EMThreadPool {
    #[new]
    fn new(n_jobs: usize) -> PyResult<Self> {
        if n_jobs == 0 {
            return Err(PyValueError::new_err("n_jobs must be positive"));
        }
        Ok(Self {
            pool: rayon::ThreadPoolBuilder::new()
                .num_threads(n_jobs)
                .build()
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
        })
    }
}

/// Negative expected log likelihood and its gradient in (a, b_1, ...).
///
/// `counts` is category-major: `counts[c * points.len() + q]`.
fn objective(
    x: &[f64],
    points: &[f64],
    counts: &[f64],
    grm: bool,
    epsilon: f64,
) -> (f64, Vec<f64>) {
    let k = x.len();
    let n_points = points.len();
    let a = x[0];
    let mut value = 0.0;
    let mut gradient = vec![0.0; k];
    let mut probabilities = vec![0.0; k];
    let mut cumulative = vec![0.0; k - 1];
    let mut slope_derivatives = vec![0.0; k];
    let mut effective = vec![0.0; k];
    for (q, &theta) in points.iter().enumerate() {
        if grm {
            for t in 0..k - 1 {
                cumulative[t] = sigmoid(a * (theta - x[t + 1]));
            }
            probabilities[0] = 1.0 - cumulative[0];
            for c in 1..k - 1 {
                probabilities[c] = cumulative[c - 1] - cumulative[c];
            }
            probabilities[k - 1] = cumulative[k - 2];
        } else {
            probabilities[0] = 0.0;
            for c in 1..k {
                slope_derivatives[c] = slope_derivatives[c - 1] + theta - x[c];
                probabilities[c] = a * slope_derivatives[c];
            }
            let max = probabilities
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);
            let mut total = 0.0;
            for p in &mut probabilities {
                *p = (*p - max).exp();
                total += *p;
            }
            for p in &mut probabilities {
                *p /= total;
            }
        }
        for c in 0..k {
            let p = probabilities[c];
            let count = counts[c * n_points + q];
            value -= count * p.clamp(epsilon, 1.0 - epsilon).ln();
            effective[c] = if p > epsilon && p < 1.0 - epsilon {
                count
            } else {
                0.0
            };
        }
        if grm {
            for t in 0..k - 1 {
                let left = if effective[t] == 0.0 {
                    0.0
                } else {
                    effective[t] / probabilities[t]
                };
                let right = if effective[t + 1] == 0.0 {
                    0.0
                } else {
                    effective[t + 1] / probabilities[t + 1]
                };
                let derivative = (left - right) * cumulative[t] * (1.0 - cumulative[t]);
                gradient[0] += derivative * (theta - x[t + 1]);
                gradient[t + 1] -= derivative * a;
            }
        } else {
            let total: f64 = effective.iter().sum();
            let mean: f64 = probabilities
                .iter()
                .zip(&slope_derivatives)
                .map(|(p, d)| p * d)
                .sum();
            for c in 0..k {
                gradient[0] -= effective[c] * (slope_derivatives[c] - mean);
            }
            let mut tail_p = 0.0;
            let mut tail_r = 0.0;
            for c in (1..k).rev() {
                tail_p += probabilities[c];
                tail_r += effective[c];
                gradient[c] += a * (tail_r - total * tail_p);
            }
        }
    }
    (value, gradient)
}

fn bounds(index: usize) -> (f64, f64) {
    if index == 0 { (0.1, 5.0) } else { (-6.0, 6.0) }
}

/// Smallest gap between adjacent graded thresholds, as in the generic
/// optimizer's linear constraint; it keeps every category probability positive.
const MIN_GAP: f64 = 1e-6;
/// Adjacent thresholds closer than this form a tied run whose ordering
/// constraint may bind.
const TIE_GAP: f64 = 2.0 * MIN_GAP;

/// Least-squares non-decreasing fit by pooling adjacent violators.
///
/// Returns each pooled block as `(offset, length, sum)`.
fn pool_adjacent_violators(values: &[f64]) -> Vec<(usize, usize, f64)> {
    let mut blocks: Vec<(usize, usize, f64)> = Vec::with_capacity(values.len());
    for (index, &value) in values.iter().enumerate() {
        let mut block = (index, 1, value);
        while let Some(&(offset, length, sum)) = blocks.last() {
            if sum / length as f64 <= block.2 / block.1 as f64 {
                break;
            }
            blocks.pop();
            block = (offset, length + block.1, sum + block.2);
        }
        blocks.push(block);
    }
    blocks
}

/// Maximal run `start..=end` of adjacent free graded thresholds.
///
/// `lower` bounds the first and `upper` the last threshold, combining the box
/// with the gap to fixed neighbours; with the minimum gaps these imply the box
/// for every member.
struct Segment {
    start: usize,
    end: usize,
    lower: f64,
    upper: f64,
}

/// Feasible set of one item: coordinate boxes and, for graded items, adjacent
/// thresholds at least `MIN_GAP` apart.
struct Region {
    free: Vec<bool>,
    segments: Vec<Segment>,
    /// Whether a coordinate belongs to a segment rather than to a plain box.
    ordered: Vec<bool>,
}

impl Region {
    /// Returns `None` when fixed thresholds are disordered or leave no room
    /// for their free neighbours. Parameters must be finite.
    fn new(x: &[f64], free: Vec<bool>, grm: bool) -> Option<Self> {
        let k = x.len();
        let mut segments = Vec::new();
        let mut ordered = vec![false; k];
        if grm {
            if (1..k.saturating_sub(1)).any(|t| !free[t] && !free[t + 1] && x[t + 1] < x[t]) {
                return None;
            }
            let mut start = 1;
            while start < k {
                if !free[start] {
                    start += 1;
                    continue;
                }
                let mut end = start;
                while end + 1 < k && free[end + 1] {
                    end += 1;
                }
                let (low, high) = bounds(start);
                let lower = if start > 1 {
                    low.max(x[start - 1] + MIN_GAP)
                } else {
                    low
                };
                let upper = if end + 1 < k {
                    high.min(x[end + 1] - MIN_GAP)
                } else {
                    high
                };
                if lower > upper - (end - start) as f64 * MIN_GAP {
                    return None;
                }
                ordered[start..=end].fill(true);
                segments.push(Segment {
                    start,
                    end,
                    lower,
                    upper,
                });
                start = end + 1;
            }
        }
        Some(Self {
            free,
            segments,
            ordered,
        })
    }

    /// Euclidean projection of free coordinates onto the region, in place.
    ///
    /// A segment whose clamped values are already ordered keeps them, so
    /// unconstrained steps match a plain box projection bit for bit.
    fn project(&self, y: &mut [f64]) {
        for (j, value) in y.iter_mut().enumerate() {
            if self.free[j] && !self.ordered[j] {
                let (low, high) = bounds(j);
                *value = value.clamp(low, high);
            }
        }
        for segment in &self.segments {
            let values = &mut y[segment.start..=segment.end];
            let (low, high) = bounds(segment.start);
            let clamped = |i: usize| values[i].clamp(low, high);
            let last = values.len() - 1;
            if clamped(0) >= segment.lower
                && clamped(last) <= segment.upper
                && (0..last).all(|i| clamped(i + 1) - clamped(i) >= MIN_GAP)
            {
                for value in values.iter_mut() {
                    *value = value.clamp(low, high);
                }
                continue;
            }
            // Thresholds minus their minimum offsets must be non-decreasing
            // and lie in one interval; the projection there clips the
            // isotonic fit.
            let shifted: Vec<f64> = values
                .iter()
                .enumerate()
                .map(|(i, value)| value - i as f64 * MIN_GAP)
                .collect();
            let ceiling = segment.upper - last as f64 * MIN_GAP;
            for (offset, length, sum) in pool_adjacent_violators(&shifted) {
                let level = (sum / length as f64).clamp(segment.lower, ceiling);
                for (i, value) in values.iter_mut().enumerate().skip(offset).take(length) {
                    // Count down from a clipped ceiling so the last threshold
                    // lands exactly on its bound.
                    *value = if level == ceiling {
                        segment.upper - (last - i) as f64 * MIN_GAP
                    } else {
                        level + i as f64 * MIN_GAP
                    };
                }
            }
        }
    }

    /// Moving group of each coordinate at `x`, named by its first member, or
    /// `None` when the coordinate stays put.
    ///
    /// Coordinates move alone unless tied thresholds whose ordering binds
    /// must move together; the groups follow the projection of the negative
    /// gradient onto the region's tangent cone.
    fn groups(&self, x: &[f64], gradient: &[f64]) -> Vec<Option<usize>> {
        let mut groups: Vec<Option<usize>> = (0..x.len())
            .map(|j| {
                let (low, high) = bounds(j);
                let blocked =
                    (x[j] <= low && gradient[j] > 0.0) || (x[j] >= high && gradient[j] < 0.0);
                (self.free[j] && !self.ordered[j] && !blocked).then_some(j)
            })
            .collect();
        for segment in &self.segments {
            let mut first = segment.start;
            while first <= segment.end {
                let mut last = first;
                while last < segment.end && x[last + 1] - x[last] < TIE_GAP {
                    last += 1;
                }
                let at_lower = first == segment.start && x[first] <= segment.lower;
                let at_upper = last == segment.end && x[last] >= segment.upper;
                let blocked =
                    |velocity: f64| (at_lower && velocity < 0.0) || (at_upper && velocity > 0.0);
                if first == last {
                    groups[first] = (!blocked(-gradient[first])).then_some(first);
                } else {
                    let velocity: Vec<f64> = gradient[first..=last].iter().map(|g| -g).collect();
                    for (offset, length, sum) in pool_adjacent_violators(&velocity) {
                        let group = (!blocked(sum / length as f64)).then_some(first + offset);
                        groups[first + offset..first + offset + length].fill(group);
                    }
                }
                first = last + 1;
            }
        }
        groups
    }
}

/// Projected inverse-BFGS with Armijo backtracking; only improving steps commit.
///
/// Steps are projected onto the item's region. Quasi-Newton updates act on
/// the moving groups, so tied graded thresholds move as one coordinate.
#[allow(clippy::too_many_arguments)]
fn optimize(
    mut x: Vec<f64>,
    region: &Region,
    points: &[f64],
    counts: &[f64],
    grm: bool,
    epsilon: f64,
    max_iter: usize,
    ftol: f64,
) -> Vec<f64> {
    let k = x.len();
    region.project(&mut x);
    let (mut value, mut gradient) = objective(&x, points, counts, grm, epsilon);
    let mut inverse = vec![0.0; k * k];
    for j in 0..k {
        inverse[j * k + j] = 1.0;
    }
    let mut previous_groups: Vec<Option<usize>> = (0..k).map(Some).collect();
    for _ in 0..max_iter {
        let groups = region.groups(&x, &gradient);
        if groups != previous_groups {
            inverse.fill(0.0);
            for j in 0..k {
                inverse[j * k + j] = 1.0;
            }
            previous_groups.clone_from(&groups);
        }
        // Group sums are the gradient in group coordinates; their means are
        // the projected gradient.
        let mut reduced = vec![0.0; k];
        let mut sizes = vec![0.0; k];
        for j in 0..k {
            if let Some(group) = groups[j] {
                reduced[group] += gradient[j];
                sizes[group] += 1.0;
            }
        }
        let projected: Vec<f64> = groups
            .iter()
            .map(|group| group.map_or(0.0, |group| reduced[group] / sizes[group]))
            .collect();
        if projected.iter().all(|g| g.abs() <= 1e-5) {
            break;
        }
        let mut direction: Vec<f64> = (0..k)
            .map(|i| {
                if groups[i] == Some(i) {
                    -(0..k).map(|j| inverse[i * k + j] * reduced[j]).sum::<f64>()
                } else {
                    0.0
                }
            })
            .collect();
        for j in 0..k {
            if let Some(group) = groups[j] {
                direction[j] = direction[group];
            }
        }
        let mut accepted = None;
        // A projected quasi-Newton direction can cease to descend at a bound.
        // Retry with the projected gradient before declaring a line-search stop.
        for attempt in 0..2 {
            if attempt == 1 {
                direction = projected.iter().map(|g| -g).collect();
            }
            let mut step = 1.0;
            for _ in 0..40 {
                let mut candidate: Vec<f64> = (0..k)
                    .map(|j| {
                        if region.free[j] {
                            x[j] + step * direction[j]
                        } else {
                            x[j]
                        }
                    })
                    .collect();
                region.project(&mut candidate);
                let derivative: f64 = (0..k).map(|j| gradient[j] * (candidate[j] - x[j])).sum();
                if derivative < 0.0 {
                    let (next, grad) = objective(&candidate, points, counts, grm, epsilon);
                    if next.is_finite() && next <= value + 1e-4 * derivative {
                        accepted = Some((candidate, next, grad));
                        break;
                    }
                }
                step *= 0.5;
            }
            if accepted.is_some() {
                break;
            }
        }
        let Some((candidate, next, grad)) = accepted else {
            break;
        };
        let delta: Vec<f64> = (0..k)
            .map(|j| {
                if groups[j] == Some(j) {
                    candidate[j] - x[j]
                } else {
                    0.0
                }
            })
            .collect();
        let mut change = vec![0.0; k];
        for j in 0..k {
            if let Some(group) = groups[j] {
                change[group] += grad[j] - gradient[j];
            }
        }
        let curvature: f64 = delta.iter().zip(&change).map(|(s, y)| s * y).sum();
        if curvature > 1e-12 {
            let hy: Vec<f64> = (0..k)
                .map(|i| (0..k).map(|j| inverse[i * k + j] * change[j]).sum())
                .collect();
            let yhy: f64 = change.iter().zip(&hy).map(|(y, h)| y * h).sum();
            for i in 0..k {
                for j in 0..k {
                    inverse[i * k + j] += (1.0 + yhy / curvature) * delta[i] * delta[j] / curvature
                        - (hy[i] * delta[j] + delta[i] * hy[j]) / curvature;
                }
            }
        }
        let converged = value - next <= ftol * value.abs().max(next.abs()).max(1.0);
        x = candidate;
        value = next;
        gradient = grad;
        if converged {
            break;
        }
    }
    x
}

#[pyfunction]
#[pyo3(signature = (responses, posterior, points, parameters, free, categories, grm, max_iter, ftol, epsilon, n_jobs, pool=None))]
#[allow(clippy::too_many_arguments)]
pub fn m_step_polytomous<'py>(
    py: Python<'py>,
    responses: PyReadonlyArray2<i32>,
    posterior: PyReadonlyArray2<f64>,
    points: PyReadonlyArray1<f64>,
    parameters: PyReadonlyArray2<f64>,
    free: PyReadonlyArray2<bool>,
    categories: PyReadonlyArray1<i32>,
    grm: bool,
    max_iter: usize,
    ftol: f64,
    epsilon: f64,
    n_jobs: usize,
    pool: Option<PyRef<'_, EMThreadPool>>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let responses = responses.as_array();
    let posterior = posterior.as_array();
    let points = points.as_array().to_vec();
    let parameters = parameters.as_array();
    let free = free.as_array();
    let categories = categories.as_array();
    let items = responses.ncols();
    if posterior.dim() != (responses.nrows(), points.len())
        || parameters.nrows() != items
        || free.dim() != parameters.dim()
        || categories.len() != items
        || categories
            .iter()
            .any(|&k| k < 2 || k as usize > parameters.ncols())
    {
        return Err(PyValueError::new_err(
            "incompatible response, posterior, or parameter shapes",
        ));
    }
    if max_iter == 0
        || n_jobs == 0
        || !ftol.is_finite()
        || ftol <= 0.0
        || !epsilon.is_finite()
        || epsilon <= 0.0
        || epsilon >= 0.5
        || points.iter().any(|v| !v.is_finite())
        || posterior.iter().any(|&v| !v.is_finite() || v < 0.0)
        || parameters.iter().any(|v| !v.is_finite())
    {
        return Err(PyValueError::new_err(
            "invalid optimizer controls or non-finite inputs",
        ));
    }
    let mut regions = Vec::with_capacity(items);
    for j in 0..items {
        let k = categories[j] as usize;
        if responses.column(j).iter().any(|&r| r >= categories[j]) {
            return Err(PyValueError::new_err("response category out of range"));
        }
        for c in 0..k {
            let (lo, hi) = bounds(c);
            if free[[j, c]] && !(lo..=hi).contains(&parameters[[j, c]]) {
                return Err(PyValueError::new_err(
                    "free parameters must lie within optimizer bounds",
                ));
            }
        }
        let values = parameters.row(j).to_vec();
        let item_free = free.row(j).iter().take(k).copied().collect();
        regions.push(Region::new(&values[..k], item_free, grm).ok_or_else(|| {
            PyValueError::new_err(
                "fixed graded thresholds must be ordered and leave room for free ones",
            )
        })?);
    }
    let posterior = posterior.as_standard_layout();
    let posterior_rows = posterior.as_slice().expect("standard layout posterior");
    let fit_item = |j: usize| {
        let k = categories[j] as usize;
        let counts = category_item_counts(responses, posterior_rows, points.len(), j, k);
        let mut params = parameters.row(j).to_vec();
        if counts.iter().any(|&v| v > 0.0) {
            let result = optimize(
                params[..k].to_vec(),
                &regions[j],
                &points,
                &counts,
                grm,
                epsilon,
                max_iter,
                ftol,
            );
            params[..k].copy_from_slice(&result);
        }
        params
    };
    let shared_pool = pool.as_ref().map(|value| &value.pool);
    if shared_pool.is_some_and(|pool| pool.current_num_threads() != n_jobs.min(items)) {
        return Err(PyValueError::new_err(
            "pool size must match n_jobs and item count",
        ));
    }
    let results = py.detach(|| -> PyResult<Vec<Vec<f64>>> {
        if n_jobs == 1 || items < 2 {
            Ok((0..items).map(fit_item).collect())
        } else if let Some(pool) = shared_pool {
            Ok(pool.install(|| (0..items).into_par_iter().map(fit_item).collect()))
        } else {
            let pool = EMThreadPool::new(n_jobs.min(items))?;
            Ok(pool
                .pool
                .install(|| (0..items).into_par_iter().map(fit_item).collect()))
        }
    })?;
    let output = Array2::from_shape_vec(parameters.dim(), results.into_iter().flatten().collect())
        .expect("validated shape");
    Ok(output.into_pyarray(py))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<EMThreadPool>()?;
    m.add_function(wrap_pyfunction!(m_step_polytomous, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{MIN_GAP, Region, objective, optimize};
    use crate::utils::sigmoid;

    /// Deterministic uniform draws in `[low, high)`.
    fn uniforms(seed: u64, n: usize, low: f64, high: f64) -> Vec<f64> {
        let mut state = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (0..n)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                low + (high - low) * (state >> 11) as f64 / (1u64 << 53) as f64
            })
            .collect()
    }

    fn assert_feasible(region: &Region, x: &[f64]) {
        for segment in &region.segments {
            assert!(x[segment.start] >= segment.lower, "{x:?}");
            assert!(x[segment.end] <= segment.upper, "{x:?}");
            for t in segment.start..segment.end {
                assert!(x[t + 1] - x[t] >= MIN_GAP * (1.0 - 1e-6), "{x:?}");
            }
        }
    }

    /// Category-major expected counts of a graded item on `points`.
    fn graded_counts(points: &[f64], a: f64, thresholds: &[f64], total: f64) -> Vec<f64> {
        let k = thresholds.len() + 1;
        let n = points.len();
        let mut counts = vec![0.0; k * n];
        for (q, &theta) in points.iter().enumerate() {
            let weight = total * (-0.5 * theta * theta).exp();
            let upper: Vec<f64> = thresholds
                .iter()
                .map(|&b| sigmoid(a * (theta - b)))
                .collect();
            for c in 0..k {
                let above = if c == 0 { 1.0 } else { upper[c - 1] };
                let below = if c == k - 1 { 0.0 } else { upper[c] };
                counts[c * n + q] = weight * (above - below);
            }
        }
        counts
    }

    #[test]
    fn ordered_projection_matches_the_box_projection() {
        let region = Region::new(&[1.0, -1.0, 0.0, 1.0], vec![true; 4], true).unwrap();
        let mut y = vec![7.0, -8.0, 0.5, 6.5];
        region.project(&mut y);
        assert_eq!(y, vec![5.0, -6.0, 0.5, 6.0]);
        let free = vec![true, false, true];
        let gpcm = Region::new(&[1.0, 2.0, -2.0], free, false).unwrap();
        let mut steps = vec![0.0, 2.0, 3.0];
        gpcm.project(&mut steps);
        assert_eq!(steps, vec![0.1, 2.0, 3.0]);
    }

    #[test]
    fn clipped_thresholds_land_exactly_on_their_bound() {
        // (-1e-6 - 2e-6) + 2e-6 rounds below -1e-6.
        let x = [1.0, -3.0, -2.0, -1.0, 0.0];
        let free = vec![true, true, true, true, false];
        let region = Region::new(&x, free, true).unwrap();
        let mut y = vec![1.0, 1.0, 2.0, 3.0, 0.0];
        region.project(&mut y);
        assert_eq!(y[3], -MIN_GAP);
        assert_eq!(y[3], region.segments[0].upper);
        assert_feasible(&region, &y);
    }

    #[test]
    fn projection_is_the_nearest_ordered_point() {
        let x = [1.0, -1.0, 0.0, 0.5, 1.0, 2.0, 3.0];
        let mut free = vec![true; x.len()];
        free[4] = false;
        let region = Region::new(&x, free, true).unwrap();
        for seed in 0..200 {
            let mut y = uniforms(seed, x.len(), -7.0, 7.0);
            y[0] = x[0];
            y[4] = x[4];
            let mut projected = y.clone();
            region.project(&mut projected);
            assert_feasible(&region, &projected);
            assert_eq!(projected[4], x[4]);
            // Variational inequality against random feasible points.
            for draw in 0..20 {
                let mut z = uniforms(1000 * seed + draw, x.len(), -7.0, 7.0);
                z[0] = x[0];
                z[4] = x[4];
                region.project(&mut z);
                let inner: f64 = (1..x.len())
                    .map(|j| (y[j] - projected[j]) * (z[j] - projected[j]))
                    .sum();
                assert!(inner <= 1e-9, "{seed} {draw}: {inner}");
            }
        }
    }

    #[test]
    fn region_rejects_fixed_thresholds_without_room() {
        let disordered = [1.0, 0.5, -0.5, 1.0];
        assert!(Region::new(&disordered, vec![true, false, false, true], true).is_none());
        assert!(Region::new(&disordered, vec![true, false, false, true], false).is_some());
        let crowded = [1.0, 0.0, 0.0, 1e-6];
        assert!(Region::new(&crowded, vec![true, false, true, false], true).is_none());
        let roomy = [1.0, 0.0, 0.0, 3e-6];
        assert!(Region::new(&roomy, vec![true, false, true, false], true).is_some());
    }

    #[test]
    fn graded_optimizer_ties_thresholds_of_an_empty_category() {
        let points: Vec<f64> = (0..21).map(|q| -4.0 + 0.4 * q as f64).collect();
        let mut counts = graded_counts(&points, 1.4, &[-1.4, -0.9, -0.8], 300.0);
        counts[2 * points.len()..3 * points.len()].fill(0.0);
        let free = vec![true; 4];
        for start in [
            [1.0, -1.0, 0.0, 1.0],
            [1.0, -2.0, -1.9, 2.0],
            [2.0, 1.0, -1.0, 0.5],
        ] {
            let region = Region::new(&start, free.clone(), true).unwrap();
            let x = optimize(
                start.to_vec(),
                &region,
                &points,
                &counts,
                true,
                1e-10,
                500,
                1e-12,
            );
            assert_feasible(&region, &x);
            assert!(x[3] - x[2] < 1e-5, "{x:?}");
            let value = objective(&x, &points, &counts, true, 1e-10).0;
            // No feasible point nearby does better.
            for draw in 0..50 {
                let noise = uniforms(draw, 4, -0.05, 0.05);
                let mut z: Vec<f64> = x.iter().zip(&noise).map(|(v, e)| v + e).collect();
                region.project(&mut z);
                let other = objective(&z, &points, &counts, true, 1e-10).0;
                assert!(value <= other + 1e-6, "{x:?} {value} > {z:?} {other}");
            }
        }
    }

    #[test]
    fn graded_optimizer_stacks_thresholds_of_empty_top_categories_at_the_bound() {
        let points: Vec<f64> = (0..21).map(|q| -4.0 + 0.4 * q as f64).collect();
        // The box, then a fixed last threshold, bounds the stacked ones.
        for (start, free) in [
            (vec![1.0, -0.5, 0.5, 1.5], vec![true; 4]),
            (
                vec![1.0, -0.5, 0.0, 0.4, 0.7],
                vec![true, true, true, true, false],
            ),
        ] {
            let mut counts = graded_counts(&points, 1.3, &start[1..], 400.0);
            counts[2 * points.len()..4 * points.len()].fill(0.0);
            let region = Region::new(&start, free, true).unwrap();
            let x = optimize(
                start.clone(),
                &region,
                &points,
                &counts,
                true,
                1e-10,
                500,
                1e-12,
            );
            let top = region.segments[0].end;
            assert_eq!(x[top], region.segments[0].upper, "{x:?}");
            assert!((x[top] - x[top - 1] - MIN_GAP).abs() < 1e-12, "{x:?}");
            let (_, gradient) = objective(&x, &points, &counts, true, 1e-10);
            // Only the bound and the ordering hold the stacked thresholds.
            assert!(gradient[top - 1] + gradient[top] < 0.0, "{gradient:?}");
            assert!(
                gradient[0].abs() < 1e-3 && gradient[1].abs() < 1e-3,
                "{gradient:?}"
            );
        }
    }

    #[test]
    fn graded_optimizer_keeps_fixed_thresholds_and_their_order() {
        let points: Vec<f64> = (0..21).map(|q| -4.0 + 0.4 * q as f64).collect();
        let mut counts = graded_counts(&points, 1.2, &[-1.0, -0.2, 0.6, 1.4], 500.0);
        // Category 2 lies between the fixed second and the free third
        // threshold, which starts out of order with it.
        counts[2 * points.len()..3 * points.len()].fill(0.0);
        let start = [1.0, -1.5, 0.3, -0.1, 2.0];
        let free = vec![true, true, false, true, true];
        let region = Region::new(&start, free, true).unwrap();
        let x = optimize(
            start.to_vec(),
            &region,
            &points,
            &counts,
            true,
            1e-10,
            500,
            1e-12,
        );
        assert_eq!(x[2], 0.3);
        assert_feasible(&region, &x);
        assert!(x[3] > 0.3 && x[3] < 0.3 + 1e-5, "{x:?}");
        assert!(x[1] < 0.3, "{x:?}");
    }

    #[test]
    fn analytic_gradients_match_independent_differences() {
        let points = [-2.0, -0.2, 0.8, 2.0];
        let counts: Vec<f64> = (0..16).map(|i| (i % 5 + 1) as f64).collect();
        let x = [1.3, -1.0, 0.2, 1.5];
        for grm in [false, true] {
            let (_, gradient) = objective(&x, &points, &counts, grm, 1e-10);
            for j in 0..x.len() {
                let mut plus = x;
                let mut minus = x;
                plus[j] += 1e-5;
                minus[j] -= 1e-5;
                let expected = (objective(&plus, &points, &counts, grm, 1e-10).0
                    - objective(&minus, &points, &counts, grm, 1e-10).0)
                    / 2e-5;
                assert!(
                    (expected - gradient[j]).abs() < 1e-6,
                    "{grm} {j}: {expected} != {}",
                    gradient[j]
                );
            }
        }
    }
}
