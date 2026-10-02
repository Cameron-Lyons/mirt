//! Pooled score-stratum SIBTEST effects and within-stratum sampling errors.

use std::collections::BTreeMap;

use numpy::ndarray::Array1;
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2, ToPyArray};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::utils::{EPSILON, normal_sf};

#[derive(Default)]
struct ScoreMoments {
    count: usize,
    sum: f64,
    squares: f64,
}

impl ScoreMoments {
    fn add(&mut self, value: f64) {
        self.count += 1;
        self.sum += value;
        self.squares += value * value;
    }

    fn mean(&self) -> f64 {
        self.sum / self.count as f64
    }

    fn mean_variance(&self) -> f64 {
        let centered = (self.squares - self.sum * self.mean()).max(0.0);
        centered / (self.count * (self.count - 1)) as f64
    }
}

fn conditional_effects(
    reference: impl Iterator<Item = (i32, f64)>,
    focal: impl Iterator<Item = (i32, f64)>,
) -> (f64, f64, Vec<f64>, Vec<f64>) {
    let mut strata: BTreeMap<i32, (ScoreMoments, ScoreMoments)> = BTreeMap::new();
    for (score, suspect) in reference {
        strata.entry(score).or_default().0.add(suspect);
    }
    for (score, suspect) in focal {
        strata.entry(score).or_default().1.add(suspect);
    }
    let mut differences = Vec::new();
    let mut counts = Vec::new();
    let mut sampling_variances = Vec::new();
    for (reference, focal) in strata.into_values() {
        if reference.count < 2 || focal.count < 2 {
            continue;
        }
        differences.push(reference.mean() - focal.mean());
        counts.push((reference.count + focal.count) as f64);
        sampling_variances.push(reference.mean_variance() + focal.mean_variance());
    }
    if differences.is_empty() {
        return (f64::NAN, f64::NAN, differences, counts);
    }
    let total: f64 = counts.iter().sum();
    let beta = differences
        .iter()
        .zip(&counts)
        .map(|(difference, count)| difference * (count / total))
        .sum();
    let variance: f64 = sampling_variances
        .iter()
        .zip(&counts)
        .map(|(variance, count)| variance * (count / total).powi(2))
        .sum();
    (beta, variance.sqrt(), differences, counts)
}

/// Compute uncorrected SIBTEST beta, SE, conditional differences, pooled counts.
#[pyfunction]
#[allow(clippy::type_complexity)]
pub fn sibtest_compute_beta<'py>(
    py: Python<'py>,
    ref_data: PyReadonlyArray2<i32>,
    focal_data: PyReadonlyArray2<i32>,
    ref_scores: PyReadonlyArray1<i32>,
    focal_scores: PyReadonlyArray1<i32>,
    suspect_items: PyReadonlyArray1<i32>,
) -> PyResult<(
    f64,
    f64,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    let ref_data = ref_data.as_array();
    let focal_data = focal_data.as_array();
    let ref_scores = ref_scores.as_array();
    let focal_scores = focal_scores.as_array();
    let suspect_items = suspect_items.as_array();
    if ref_data.nrows() != ref_scores.len()
        || focal_data.nrows() != focal_scores.len()
        || ref_data.ncols() != focal_data.ncols()
        || ref_data.nrows() < 2
        || focal_data.nrows() < 2
        || suspect_items.is_empty()
        || suspect_items
            .iter()
            .any(|&item| item < 0 || item as usize >= ref_data.ncols())
        || ref_scores
            .iter()
            .chain(focal_scores.iter())
            .any(|&score| score < 0)
    {
        return Err(PyValueError::new_err(
            "invalid SIBTEST group shapes, scores, or suspect indices",
        ));
    }
    if ref_data
        .iter()
        .chain(focal_data.iter())
        .any(|&value| value != 0 && value != 1)
    {
        return Err(PyValueError::new_err(
            "SIBTEST responses must be complete and binary",
        ));
    }
    let suspect: Vec<usize> = suspect_items.iter().map(|&item| item as usize).collect();
    let (beta, se, differences, counts) = conditional_effects(
        ref_scores.iter().enumerate().map(|(person, &score)| {
            (
                score,
                suspect
                    .iter()
                    .map(|&item| ref_data[[person, item]] as f64)
                    .sum(),
            )
        }),
        focal_scores.iter().enumerate().map(|(person, &score)| {
            (
                score,
                suspect
                    .iter()
                    .map(|&item| focal_data[[person, item]] as f64)
                    .sum(),
            )
        }),
    );
    Ok((
        beta,
        se,
        Array1::from(differences).to_pyarray(py),
        Array1::from(counts).to_pyarray(py),
    ))
}

/// Run uncorrected SIBTEST across items with shared matching totals.
#[pyfunction]
#[allow(clippy::type_complexity)]
pub fn sibtest_all_items<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<i32>,
    groups: PyReadonlyArray1<i32>,
    anchor_items: Option<PyReadonlyArray1<i32>>,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    let data = data.as_array();
    let groups = groups.as_array();
    let n_items = data.ncols();
    if data.nrows() != groups.len() || data.nrows() == 0 || n_items < 2 {
        return Err(PyValueError::new_err(
            "invalid SIBTEST response or group shape",
        ));
    }
    if data.iter().any(|&value| value != 0 && value != 1) {
        return Err(PyValueError::new_err(
            "SIBTEST responses must be complete and binary",
        ));
    }
    let mut unique_groups: Vec<i32> = groups.iter().copied().collect();
    unique_groups.sort_unstable();
    unique_groups.dedup();
    if unique_groups.len() != 2 {
        return Err(PyValueError::new_err("SIBTEST requires exactly two groups"));
    }
    let reference: Vec<usize> = groups
        .iter()
        .enumerate()
        .filter_map(|(person, &group)| (group == unique_groups[0]).then_some(person))
        .collect();
    let focal: Vec<usize> = groups
        .iter()
        .enumerate()
        .filter_map(|(person, &group)| (group == unique_groups[1]).then_some(person))
        .collect();
    if reference.len() < 2 || focal.len() < 2 {
        return Err(PyValueError::new_err(
            "each SIBTEST group requires at least two persons",
        ));
    }
    let anchors: Vec<usize> = match anchor_items {
        Some(items) => {
            let items = items.as_array();
            if items.is_empty()
                || items
                    .iter()
                    .any(|&item| item < 0 || item as usize >= n_items)
            {
                return Err(PyValueError::new_err("invalid SIBTEST anchor indices"));
            }
            items.iter().map(|&item| item as usize).collect()
        }
        None => (0..n_items).collect(),
    };
    let mut anchor_membership = vec![false; n_items];
    for &item in &anchors {
        if anchor_membership[item] {
            return Err(PyValueError::new_err(
                "SIBTEST anchors must not contain duplicates",
            ));
        }
        anchor_membership[item] = true;
    }
    let matching_totals: Vec<i32> = data
        .rows()
        .into_iter()
        .map(|row| anchors.iter().map(|&item| row[item]).sum())
        .collect();
    let results: Vec<(f64, f64, f64)> = (0..n_items)
        .into_par_iter()
        .map(|item| {
            if anchors.len() == 1 && anchor_membership[item] {
                return (f64::NAN, f64::NAN, f64::NAN);
            }
            let conditional = |&person: &usize| {
                let response = data[[person, item]];
                let score =
                    matching_totals[person] - if anchor_membership[item] { response } else { 0 };
                (score, response as f64)
            };
            let (beta, se, _, _) = conditional_effects(
                reference.iter().map(conditional),
                focal.iter().map(conditional),
            );
            let z = if se > EPSILON { beta / se } else { f64::NAN };
            let p = if z.is_nan() {
                f64::NAN
            } else {
                2.0 * normal_sf(z.abs())
            };
            (beta, z, p)
        })
        .collect();
    Ok((
        Array1::from_iter(results.iter().map(|&(beta, _, _)| beta)).to_pyarray(py),
        Array1::from_iter(results.iter().map(|&(_, z, _)| z)).to_pyarray(py),
        Array1::from_iter(results.iter().map(|&(_, _, p)| p)).to_pyarray(py),
    ))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sibtest_compute_beta, m)?)?;
    m.add_function(wrap_pyfunction!(sibtest_all_items, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::conditional_effects;

    #[test]
    fn constant_conditional_effect_has_positive_sampling_error() {
        let reference: Vec<_> = (0..3)
            .flat_map(|score| {
                (0..100).map(move |person| (score, if person < 70 { 1.0 } else { 0.0 }))
            })
            .collect();
        let focal: Vec<_> = (0..3)
            .flat_map(|score| {
                (0..100).map(move |person| (score, if person < 50 { 1.0 } else { 0.0 }))
            })
            .collect();
        let (beta, se, _, counts) = conditional_effects(reference.into_iter(), focal.into_iter());
        assert!((beta - 0.2).abs() < 1e-12);
        let expected_variance: f64 = (0.7 * 0.3 + 0.5 * 0.5) / (99.0 * 3.0);
        assert!((se - expected_variance.sqrt()).abs() < 1e-12);
        assert_eq!(counts, vec![200.0; 3]);
    }

    #[test]
    fn empty_common_sample_is_unestimable() {
        let (beta, se, differences, counts) = conditional_effects(
            [(0, 0.0), (0, 1.0)].into_iter(),
            [(1, 0.0), (1, 1.0)].into_iter(),
        );
        assert!(beta.is_nan() && se.is_nan());
        assert!(differences.is_empty() && counts.is_empty());
    }
}
