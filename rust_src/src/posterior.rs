//! Exact independent row searches for posterior summaries and joint draws.

use numpy::ndarray::{Array2, ArrayView1, ArrayView2};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn search_rows(
    cumulative: ArrayView2<'_, f64>,
    targets: ArrayView2<'_, f64>,
    right: bool,
) -> Array2<isize> {
    let mut indices = Array2::zeros(targets.dim());
    for row in 0..targets.nrows() {
        let cumulative_row = cumulative.row(row);
        let mut previous_target = f64::NAN;
        let mut previous_index = 0;
        for column in 0..targets.ncols() {
            let target = targets[[row, column]];
            let target_is_nan = target.is_nan();
            // Interval targets are ascending. Resume at the preceding insertion
            // point; unsorted quantiles and random draws reset the lower bound.
            let mut low = if target >= previous_target || target_is_nan {
                previous_index
            } else {
                0
            };
            let mut high = cumulative.ncols();
            while low < high {
                let middle = low + (high - low) / 2;
                let value = cumulative_row[middle];
                // NumPy orders NaNs after all finite values and infinities.
                let advance = if right {
                    value <= target || target_is_nan
                } else {
                    value < target || (target_is_nan && !value.is_nan())
                };
                if advance {
                    low = middle + 1;
                } else {
                    high = middle;
                }
            }
            indices[[row, column]] = low as isize;
            previous_target = target;
            previous_index = low;
        }
    }
    indices
}

/// Search ascending CDF rows without changing their floating-point coordinates.
/// Strided CDFs and broadcast target views are supported without copying.
#[pyfunction]
pub fn row_searchsorted<'py>(
    py: Python<'py>,
    cumulative: PyReadonlyArray2<'py, f64>,
    targets: PyReadonlyArray2<'py, f64>,
    right: bool,
) -> PyResult<Bound<'py, PyArray2<isize>>> {
    let cumulative = cumulative.as_array();
    let targets = targets.as_array();
    if cumulative.nrows() != targets.nrows() {
        return Err(PyValueError::new_err(
            "targets must have one row per cumulative row",
        ));
    }
    let indices = py.detach(|| search_rows(cumulative, targets, right));
    Ok(indices.into_pyarray(py))
}

fn interval_bounds(
    coordinates: ArrayView1<'_, f64>,
    cumulative: ArrayView2<'_, f64>,
    level: f64,
) -> Array2<f64> {
    let n_points = coordinates.len();
    let width_tolerance =
        16.0 * f64::EPSILON * (coordinates[n_points - 1] - coordinates[0]).max(1.0);
    let mass_tolerance = 16.0 * f64::EPSILON;
    let mut bounds = Array2::zeros((cumulative.nrows(), 2));
    // Reuse one grid-sized buffer instead of allocating person-by-grid matrices.
    let mut ends = vec![0; n_points];
    for row in 0..cumulative.nrows() {
        let cdf = cumulative.row(row);
        let mut end = 1;
        let mut valid_starts = 0;
        let mut minimum_width = f64::INFINITY;
        for start in 0..n_points {
            let target = cdf[start] + level;
            if target > 1.0 {
                break;
            }
            end = end.max(start + 1);
            while end <= n_points && cdf[end] < target {
                end += 1;
            }
            if end > n_points {
                break;
            }
            ends[start] = end;
            valid_starts += 1;
            minimum_width = minimum_width.min(coordinates[end - 1] - coordinates[start]);
        }
        // Separate passes preserve the global width and mass tie tolerances.
        let mut greatest_mass = f64::NEG_INFINITY;
        for start in 0..valid_starts {
            if coordinates[ends[start] - 1] - coordinates[start] <= minimum_width + width_tolerance
            {
                greatest_mass = greatest_mass.max(cdf[ends[start]] - cdf[start]);
            }
        }
        for start in 0..valid_starts {
            if coordinates[ends[start] - 1] - coordinates[start] <= minimum_width + width_tolerance
                && cdf[ends[start]] - cdf[start] >= greatest_mass - mass_tolerance
            {
                bounds[[row, 0]] = coordinates[start];
                bounds[[row, 1]] = coordinates[ends[start] - 1];
                break;
            }
        }
    }
    bounds
}

/// Select intervals from normalized CDF prefixes and increasing coordinates.
/// CDFs include a leading zero and trailing one, computed identically by both backends.
#[pyfunction]
pub fn shortest_mass_intervals<'py>(
    py: Python<'py>,
    coordinates: PyReadonlyArray1<'py, f64>,
    cumulative: PyReadonlyArray2<'py, f64>,
    level: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let coordinates = coordinates.as_array();
    let cumulative = cumulative.as_array();
    if coordinates.is_empty() || cumulative.ncols() != coordinates.len() + 1 {
        return Err(PyValueError::new_err(
            "cumulative must have one more column than the non-empty coordinates",
        ));
    }
    if !level.is_finite() || level <= 0.0 || level >= 1.0 {
        return Err(PyValueError::new_err(
            "level must be finite and strictly between zero and one",
        ));
    }
    let bounds = py.detach(|| interval_bounds(coordinates, cumulative, level));
    Ok(bounds.into_pyarray(py))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(row_searchsorted, m)?)?;
    m.add_function(wrap_pyfunction!(shortest_mass_intervals, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{interval_bounds, search_rows};
    use numpy::ndarray::array;

    #[test]
    fn distinguishes_ties_and_adjacent_targets() {
        let cumulative = array![[0.0, 0.5, 0.5, 1.0]];
        let targets = array![[0.5, 0.5_f64.next_up(), 1.0, 2.0]];
        assert_eq!(
            search_rows(cumulative.view(), targets.view(), false),
            array![[1, 3, 3, 4]]
        );
        assert_eq!(
            search_rows(cumulative.view(), targets.view(), true),
            array![[3, 3, 4, 4]]
        );
    }

    #[test]
    fn interval_boundaries_and_ties() {
        let points = array![0.0, 1.0, 3.0];
        let cdf = array![[0.0, 0.5, 0.75, 1.0]];
        assert_eq!(
            interval_bounds(points.view(), cdf.view(), 0.5_f64.next_up()),
            array![[0.0, 1.0]]
        );
        assert_eq!(
            interval_bounds(points.view(), cdf.view(), 1e-20),
            array![[0.0, 0.0]]
        );
    }
}
