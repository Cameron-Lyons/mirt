//! Group integer response rows without sorting or copying row keys.

use std::collections::{HashMap, hash_map::Entry};
use std::hash::Hash;

use numpy::{Element, IntoPyArray, PyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;

type PatternIndices = (Vec<isize>, Vec<isize>, Vec<isize>);
type PyPatternIndices<'py> = (
    Bound<'py, PyArray1<isize>>,
    Bound<'py, PyArray1<isize>>,
    Bound<'py, PyArray1<isize>>,
);

fn group_rows<T: Eq + Hash>(responses: &[T], n_persons: usize, n_items: usize) -> PatternIndices {
    let mut first = Vec::new();
    let mut inverse = Vec::with_capacity(n_persons);
    let mut counts = Vec::new();
    // Borrow complete rows: equality still compares every value on hash collisions.
    // Grow with the number of patterns instead of reserving for every respondent.
    let mut patterns: HashMap<&[T], usize> = HashMap::new();
    for person in 0..n_persons {
        let row = &responses[person * n_items..(person + 1) * n_items];
        let index = match patterns.entry(row) {
            Entry::Occupied(entry) => *entry.get(),
            Entry::Vacant(entry) => {
                let index = first.len();
                entry.insert(index);
                first.push(person as isize);
                counts.push(0);
                index
            }
        };
        counts[index] += 1;
        inverse.push(index as isize);
    }
    (first, inverse, counts)
}

fn group_array<'py, T: Element + Eq + Hash + Sync>(
    py: Python<'py>,
    responses: PyReadonlyArray2<'py, T>,
) -> PyResult<PyPatternIndices<'py>> {
    if !responses.is_c_contiguous() {
        return Err(PyValueError::new_err("responses must be C-contiguous"));
    }
    let (n_persons, n_items) = (responses.shape()[0], responses.shape()[1]);
    // Check alignment before creating a typed slice. An ndarray view assumes
    // aligned pointers even when NumPy exposes an unaligned contiguous buffer.
    let rows = responses
        .as_slice()
        .map_err(|_| PyValueError::new_err("responses must be C-contiguous and aligned"))?;
    // NumPy shape metadata can change while detached even for read-only data.
    // Retain owned dimensions rather than borrowing that metadata across it.
    let (first, inverse, counts) = py.detach(|| group_rows(rows, n_persons, n_items));
    Ok((
        first.into_pyarray(py),
        inverse.into_pyarray(py),
        counts.into_pyarray(py),
    ))
}

/// Return first row indices, inverse indices, and counts in first-appearance order.
/// Input must be an aligned, C-contiguous signed integer matrix with normalized missing values.
#[pyfunction]
pub fn response_pattern_indices<'py>(
    py: Python<'py>,
    responses: &Bound<'py, PyAny>,
) -> PyResult<PyPatternIndices<'py>> {
    if let Ok(values) = responses.extract::<PyReadonlyArray2<'py, i8>>() {
        group_array(py, values)
    } else if let Ok(values) = responses.extract::<PyReadonlyArray2<'py, i16>>() {
        group_array(py, values)
    } else if let Ok(values) = responses.extract::<PyReadonlyArray2<'py, i32>>() {
        group_array(py, values)
    } else if let Ok(values) = responses.extract::<PyReadonlyArray2<'py, i64>>() {
        group_array(py, values)
    } else {
        Err(PyTypeError::new_err(
            "responses must be a two-dimensional signed integer array",
        ))
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(response_pattern_indices, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::group_rows;
    use std::hash::{Hash, Hasher};

    #[test]
    fn preserves_order_and_full_width_values() {
        let rows = [i64::MAX, -7, 0, i64::MIN, i64::MAX, -7, 0, i64::MIN, 1, 2];
        assert_eq!(
            group_rows(&rows, 5, 2),
            (vec![0, 1, 4], vec![0, 1, 0, 1, 2], vec![2, 2, 1])
        );
    }

    #[test]
    fn supports_empty_dimensions() {
        assert_eq!(group_rows::<i8>(&[], 0, 3), (vec![], vec![], vec![]));
        assert_eq!(
            group_rows::<i8>(&[], 3, 0),
            (vec![0], vec![0, 0, 0], vec![3])
        );
    }

    #[test]
    fn compares_complete_rows_on_hash_collision() {
        #[derive(PartialEq, Eq)]
        struct CollidingValue(i8);

        impl Hash for CollidingValue {
            fn hash<H: Hasher>(&self, state: &mut H) {
                0.hash(state);
            }
        }

        let rows = [1, 2, 1, 3, 1, 2].map(CollidingValue);
        assert_eq!(
            group_rows(&rows, 3, 2),
            (vec![0, 1], vec![0, 1, 0], vec![2, 1])
        );
    }
}
