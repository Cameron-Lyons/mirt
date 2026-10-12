"""Bifactor item models with one general and one specific factor per item."""

import numpy as np
from numpy.typing import NDArray

from mirt._logistic import (
    _affine_logits,
    _affine_probability,
    _information,
    _item_information,
    _test_information,
)
from mirt._model_defaults import record_model_base as _record_model_base
from mirt._model_defaults import register_builtin_model as _register_builtin_model
from mirt.models.base import (
    BaseItemModel,
    DichotomousItemModel,
    PolytomousItemModel,
    validate_category_index,
)
from mirt.models.multidimensional import standardized_loadings
from mirt.models.polytomous import (
    _category_count_chunks,
    _graded_information,
    _graded_probabilities,
    _require_finite_parameters,
)


@_record_model_base
class _BifactorStructure(BaseItemModel):
    """Specific-factor structure and loadings shared by the bifactor families.

    Item ``j`` loads on the general factor (theta column 0) with
    ``general_loadings[j]`` and on its specific factor with
    ``specific_loadings[j]``. External specific-factor labels may be any
    non-negative integers; a contiguous internal index places each specific
    factor in theta column ``1 + index``.
    """

    def _set_specific_factors(
        self,
        n_items: int,
        specific_factors: NDArray[np.int_] | list[int],
    ) -> int:
        """Validate and store the specific-factor labels; return ``n_factors``.

        Raises
        ------
        ValueError
            If the labels do not give one non-negative integer per item.
        """
        factor_values = np.asarray(specific_factors)

        if factor_values.ndim != 1 or factor_values.shape[0] != n_items:
            raise ValueError(
                f"Length of specific_factors ({factor_values.size}) "
                f"must match n_items ({n_items})"
            )

        if not np.issubdtype(factor_values.dtype, np.integer):
            if not np.issubdtype(factor_values.dtype, np.floating) or not np.all(
                np.isfinite(factor_values) & (factor_values == np.floor(factor_values))
            ):
                raise ValueError("specific_factors must contain integer labels")

        factor_values = factor_values.astype(np.int_, copy=False)
        if np.any(factor_values < 0):
            raise ValueError("specific_factors must be non-negative integers")

        factor_labels, factor_indices = np.unique(factor_values, return_inverse=True)
        self._specific_factors = factor_values.copy()
        self._specific_factor_labels = factor_labels
        self._specific_factor_indices = factor_indices
        self._n_specific_factors = len(factor_labels)
        return 1 + self._n_specific_factors

    @property
    def general_loadings(self) -> NDArray[np.float64]:
        return self._parameters["general_loadings"]

    @property
    def specific_loadings(self) -> NDArray[np.float64]:
        return self._parameters["specific_loadings"]

    @property
    def intercepts(self) -> NDArray[np.float64]:
        return self._parameters["intercepts"]

    @property
    def specific_factors(self) -> NDArray[np.int_]:
        return self._specific_factors.copy()

    @property
    def n_specific_factors(self) -> int:
        return self._n_specific_factors

    @property
    def specific_factor_labels(self) -> NDArray[np.int_]:
        """Return the sorted external labels for the specific factors."""
        return self._specific_factor_labels.copy()

    def get_factor_structure(self) -> dict[int, list[int]]:
        """Map each external specific-factor label to its item indices."""
        return {
            int(label): np.flatnonzero(self._specific_factors == label).tolist()
            for label in self._specific_factor_labels
        }

    def get_factor_loadings(self, standardized: bool = True) -> NDArray[np.float64]:
        """Return the sparse item-by-factor loadings.

        Parameters
        ----------
        standardized : bool, default=True
            Convert the slopes to standardized loadings as R mirt's
            ``summary()`` does; otherwise return :meth:`get_loading_matrix`.

        Returns
        -------
        ndarray of shape (n_items, 1 + n_specific_factors)
            General loadings in the first column and each item's specific
            loading in the column of its specific factor.

        Notes
        -----
        The factors are uncorrelated with unit variance. Slopes ``a`` are
        divided by ``D = 1.702``, and the standardized loading on a factor is
        ``(a / D) / sqrt(1 + (a_g / D)**2 + (a_s / D)**2)``; see
        :func:`~mirt.models.multidimensional.standardized_loadings`.
        """
        loadings = self.get_loading_matrix()
        if not standardized:
            return loadings
        return standardized_loadings(loadings)[0]

    def communalities(self) -> NDArray[np.float64]:
        """Return the share of each item's latent response variance explained.

        These are R mirt's ``h2`` of the standardized loadings; their
        complements are the uniquenesses the omega coefficients use.
        """
        return standardized_loadings(self.get_loading_matrix())[1]

    def _standardized_components(
        self,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Return standardized general and specific loadings and uniquenesses."""
        loadings, communalities = standardized_loadings(self.get_loading_matrix())
        specific = loadings[np.arange(self.n_items), 1 + self._specific_factor_indices]
        return loadings[:, 0], specific, 1.0 - communalities

    def omega_hierarchical(self) -> float:
        """Estimate general-factor reliability for the total score.

        McDonald's omega hierarchical from the standardized loadings of
        :meth:`get_factor_loadings`: the squared sum of general loadings over
        the variance of the sum of standardized latent responses, which adds
        every specific factor's squared loading sum and the uniquenesses
        ``1 - h2``.
        """
        general, specific, unique = self._score_variance_components()
        return float(general / (general + specific + unique))

    def omega_total(self) -> float:
        """Estimate reliability due to all general and specific factors.

        McDonald's omega total from the standardized loadings of
        :meth:`get_factor_loadings`.
        """
        general, specific, unique = self._score_variance_components()
        common = general + specific
        return float(common / (common + unique))

    def _score_variance_components(
        self,
        items: NDArray[np.int_] | None = None,
    ) -> tuple[float, float, float]:
        """Return general, specific and unique variance of a standardized sum.

        The general and specific components are squared loading sums, the
        unique component the sum of the uniquenesses ``1 - h2``.
        """
        if items is None:
            items = np.arange(self.n_items)

        general_loadings, specific_loadings, uniquenesses = (
            self._standardized_components()
        )
        general_variance = float(np.sum(general_loadings[items]) ** 2)
        specific_variance = 0.0
        for label in self._specific_factor_labels:
            factor_items = items[self._specific_factors[items] == label]
            if factor_items.size:
                specific_variance += float(np.sum(specific_loadings[factor_items]) ** 2)
        unique_variance = float(np.sum(uniquenesses[items]))
        return general_variance, specific_variance, unique_variance

    def omega_subscale(self, specific_factor: int) -> float:
        """Estimate total reliability for one externally labeled subscale.

        Omega total of the subscale's items from their standardized general
        and specific loadings, or NaN for a label without items.
        """
        items = np.where(self._specific_factors == specific_factor)[0]

        if len(items) == 0:
            return np.nan

        general, specific, unique = self._score_variance_components(items)
        common = general + specific
        return float(common / (common + unique))

    def explained_common_variance(self) -> dict[str, float]:
        """Return general and group-specific shares of common variance.

        Shares of the summed squared standardized loadings of
        :meth:`get_factor_loadings`; NaN when no item loads on any factor.
        """
        a_g, a_s, _ = self._standardized_components()

        sum_a_g_sq = np.sum(a_g**2)
        sum_a_s_sq = np.sum(a_s**2)
        total_common = sum_a_g_sq + sum_a_s_sq

        result = {"general": sum_a_g_sq / total_common if total_common > 0 else np.nan}

        for label in self._specific_factor_labels:
            items = np.where(self._specific_factors == label)[0]
            sf_variance = np.sum(a_s[items] ** 2)
            result[f"specific_{int(label)}"] = (
                sf_variance / total_common if total_common > 0 else np.nan
            )

        return result

    def get_loading_matrix(self) -> NDArray[np.float64]:
        """Return the sparse item-by-factor loading matrix."""
        loadings = np.zeros((self.n_items, 1 + self._n_specific_factors))

        loadings[:, 0] = self._parameters["general_loadings"]
        loadings[
            np.arange(self.n_items),
            1 + self._specific_factor_indices,
        ] = self._parameters["specific_loadings"]

        return loadings

    def get_item_parameters(
        self,
        item_idx: int,
    ) -> dict[str, float | NDArray[np.float64]]:
        """Return item parameters plus its full multidimensional slope vector."""
        parameters = super().get_item_parameters(item_idx)
        parameters["slopes"] = self._item_slope(item_idx)
        return parameters

    def _item_slope(self, item_idx: int) -> NDArray[np.float64]:
        """Return one sparse multidimensional slope vector."""
        slopes = np.zeros(self.n_factors, dtype=np.float64)
        slopes[0] = self._parameters["general_loadings"][item_idx]
        slopes[1 + self._specific_factor_indices[item_idx]] = self._parameters[
            "specific_loadings"
        ][item_idx]
        return slopes


@_register_builtin_model
class BifactorModel(_BifactorStructure, DichotomousItemModel):
    """Bifactor model with one general and one specific loading per item.

    Specific-factor labels may be any non-negative integers. Labels are kept
    for reporting while an internal contiguous index determines theta columns.
    """

    model_name = "Bifactor"
    supports_multidimensional = True

    def __init__(
        self,
        n_items: int,
        specific_factors: NDArray[np.int_] | list[int],
        item_names: list[str] | None = None,
    ) -> None:
        n_factors = self._set_specific_factors(n_items, specific_factors)
        super().__init__(n_items, n_factors, item_names)

    def _initialize_parameters(self) -> None:
        self._parameters["general_loadings"] = np.ones(self.n_items) * 0.7

        self._parameters["specific_loadings"] = np.ones(self.n_items) * 0.5

        self._parameters["intercepts"] = np.zeros(self.n_items)

    def _curve_parameters(
        self, item_idx: int | None
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.intp]]:
        if item_idx is None:
            slopes = np.column_stack((self.general_loadings, self.specific_loadings))
            return slopes, self.intercepts, self._specific_factor_indices
        item_idx = self._validate_item_index(item_idx)
        return (
            np.array(
                [self.general_loadings[item_idx], self.specific_loadings[item_idx]]
            ),
            self.intercepts[item_idx],
            np.asarray(self._specific_factor_indices[item_idx]),
        )

    def probability(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        theta = self._ensure_theta_2d(theta)
        slopes, intercepts, columns = self._curve_parameters(item_idx)
        return _affine_probability(theta, slopes, intercepts, specific_indices=columns)

    def _logits(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        slopes, intercepts, columns = self._curve_parameters(item_idx)
        return _affine_logits(theta, slopes, intercepts, specific_indices=columns)

    def probability_pairs(
        self,
        theta: NDArray[np.float64],
        item_indices: NDArray[np.int_],
    ) -> NDArray[np.float64]:
        """Evaluate aligned respondent-item pairs in bounded vectorized batches."""
        theta, indices = self._prepare_probability_pairs(theta, item_indices)
        return _affine_probability(
            theta,
            (self.general_loadings, self.specific_loadings),
            self.intercepts,
            item_indices=indices,
            specific_indices=self._specific_factor_indices,
        )

    def information(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        theta = self._ensure_theta_2d(theta)
        slopes, intercepts, columns = self._curve_parameters(item_idx)
        return _information(
            theta,
            slopes,
            lambda points: _affine_logits(
                points, slopes, intercepts, specific_indices=columns
            ),
        )

    def item_information_matrix(
        self,
        theta: NDArray[np.float64],
        item_idx: int,
    ) -> NDArray[np.float64]:
        """Return item Fisher matrices across multidimensional theta points."""
        item_idx = self._validate_item_index(item_idx)
        theta = self._ensure_theta_2d(theta)
        return _item_information(
            self._logits(theta, item_idx), self._item_slope(item_idx)
        )

    def test_information_matrix(
        self,
        theta: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return summed Fisher matrices across all items and theta points."""
        theta = self._ensure_theta_2d(theta)
        return _test_information(theta, self.get_loading_matrix(), self._logits)

    def copy(self) -> "BifactorModel":
        """Return an independent copy that preserves the bifactor structure."""
        new_model = self.__class__(
            n_items=self.n_items,
            specific_factors=self._specific_factors.copy(),
            item_names=self.item_names.copy(),
        )
        new_model._parameters = {
            name: values.copy() for name, values in self._parameters.items()
        }
        new_model._is_fitted = self._is_fitted
        self._copy_parameter_restrictions_to(new_model)
        return new_model


@_register_builtin_model
class GradedBifactorModel(_BifactorStructure, PolytomousItemModel):
    """Bifactor graded response model, R mirt's ``bfactor(itemtype="graded")``.

    Item ``j`` with ``C_j`` ordered categories loads on the general factor
    and on one specific factor ``s(j)``. Its cumulative curves are

    .. math::

       P(Y_j \\ge c \\mid \\theta) = \\operatorname{logistic}(a_{gj} \\theta_0
       + a_{sj} \\theta_{s(j)} + d_{jc}), \\quad c = 1, \\ldots, C_j - 1,

    with intercepts ``intercepts[j, c - 1] = d_jc`` that decrease in ``c``,
    in R's slope-intercept form (Gibbons et al., 2007). A two-category item
    is the item of :class:`BifactorModel`, so one model holds dichotomous
    and graded items. Intercept columns beyond an item's ``C_j - 1`` are
    padding that never enters a curve.

    Parameters
    ----------
    n_items : int
        Number of items.
    specific_factors : array_like of int
        Specific-factor label of each item; any non-negative integers.
    n_categories : int or list of int
        Number of response categories, shared or one per item (at least 2).
    item_names : list of str, optional
        Item names.

    Notes
    -----
    :meth:`set_parameters` requires finite loadings and active intercepts
    but not their order, which the estimators impose on the free ones. The
    factors are independent standard normal; :func:`mirt.bfactor` and
    :class:`~mirt.estimation.bifactor_em.BifactorEMEstimator` fit the model
    by dimension reduction.

    References
    ----------
    Gibbons, R. D., Bock, R. D., Hedeker, D., Weiss, D. J., Segawa, E.,
    Bhaumik, D. K., Kupfer, D. J., Frank, E., Grochocinski, V. J., & Stover,
    A. (2007). Full-information item bifactor analysis of graded response
    data. *Applied Psychological Measurement*, 31(1), 4-19.
    """

    model_name = "GradedBifactor"
    supports_multidimensional = True
    # R mirt leaves slopes and intercepts unbounded; these boxes match the
    # logistic families'.
    _parameter_boxes = {
        "general_loadings": (-20.0, 20.0),
        "specific_loadings": (-20.0, 20.0),
        "intercepts": (-30.0, 30.0),
    }

    def __init__(
        self,
        n_items: int,
        specific_factors: NDArray[np.int_] | list[int],
        n_categories: int | list[int],
        item_names: list[str] | None = None,
    ) -> None:
        n_factors = self._set_specific_factors(n_items, specific_factors)
        super().__init__(n_items, n_categories, n_factors, item_names)

    def _initialize_parameters(self) -> None:
        self._parameters["general_loadings"] = np.full(self.n_items, 0.7)
        self._parameters["specific_loadings"] = np.full(self.n_items, 0.5)
        intercepts = np.zeros((self.n_items, self.max_categories - 1))
        for item, n_categories in enumerate(self._n_categories):
            if n_categories > 2:
                intercepts[item, : n_categories - 1] = np.linspace(
                    2.0, -2.0, n_categories - 1
                )
        self._parameters["intercepts"] = intercepts

    @property
    def free_parameter_masks(self) -> dict[str, NDArray[np.bool_]]:
        masks = super().free_parameter_masks
        masks["intercepts"] = self._category_columns(self.intercepts.shape[1], 1)
        return self._apply_free_parameter_restrictions(masks)

    def _canonical_parameter_values(
        self,
        name: str,
        values: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        canonical = super()._canonical_parameter_values(name, values)
        if name == "intercepts":
            canonical[~self._category_columns(canonical.shape[1], 1)] = 0.0
        return canonical

    def set_parameters(self, **params: NDArray[np.float64]) -> "GradedBifactorModel":
        """Set parameters atomically after validating the complete state.

        Raises
        ------
        MirtValidationError
            If a loading or active intercept is not finite. The model is left
            unchanged.
        """
        candidate = {**self._parameters, **self._coerce_parameter_updates(params)}
        self._validate_parameter_state(candidate)
        self._parameters = candidate
        return self

    def _validate_parameter_state(
        self,
        parameters: dict[str, NDArray[np.float64]],
    ) -> None:
        """Require finite loadings and active intercepts."""
        _require_finite_parameters(self, parameters, "intercepts")

    def _projections(self, theta: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return every item's ``a_g theta_0 + a_s theta_s``, ``(n_theta, n_items)``."""
        return (
            theta[:, :1] * self.general_loadings
            + theta[:, 1 + self._specific_factor_indices] * self.specific_loadings
        )

    def category_probability(
        self,
        theta: NDArray[np.float64],
        item_idx: int,
        category: int,
    ) -> NDArray[np.float64]:
        item_idx = self._validate_item_index(item_idx)
        category = validate_category_index(category, self._n_categories[item_idx])
        return self._category_probabilities(theta, item_idx)[:, category]

    def _category_probabilities(
        self,
        theta: NDArray[np.float64],
        item_idx: int,
    ) -> NDArray[np.float64]:
        """Compute one item's category probabilities in one vectorized pass."""
        theta = self._ensure_theta_2d(theta)
        n_thresholds = self._n_categories[item_idx] - 1
        projection = (
            theta[:, 0] * self.general_loadings[item_idx]
            + theta[:, 1 + self._specific_factor_indices[item_idx]]
            * self.specific_loadings[item_idx]
        )
        intercepts = self.intercepts[item_idx, :n_thresholds]
        return _graded_probabilities(projection[:, None] + intercepts)

    def probability(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        """Return category probabilities, padded with zeros for all items.

        One item gives shape ``(n_theta, n_categories[item_idx])`` and all
        items ``(n_theta, n_items, max_categories)``.
        """
        theta = self._ensure_theta_2d(theta)
        if item_idx is not None:
            return self._category_probabilities(
                theta, self._validate_item_index(item_idx)
            )
        n_theta = theta.shape[0]
        probabilities = np.zeros((n_theta, self.n_items, self.max_categories))
        projections = self._projections(theta)
        for n_categories, items in _category_count_chunks(self._n_categories, n_theta):
            logits = (
                projections[:, items, None]
                + self.intercepts[items, : n_categories - 1][None, :, :]
            )
            probabilities[:, items, :n_categories] = _graded_probabilities(logits)
        return probabilities

    def _item_information(
        self,
        theta: NDArray[np.float64],
        item_idx: int,
    ) -> NDArray[np.float64]:
        probabilities = self._category_probabilities(theta, item_idx)
        slope = np.hypot(
            self.general_loadings[item_idx], self.specific_loadings[item_idx]
        )
        return _graded_information(probabilities, slope)

    def _information_by_item(self, theta: NDArray[np.float64]) -> NDArray[np.float64]:
        theta = self._ensure_theta_2d(theta)
        slopes = np.hypot(self.general_loadings, self.specific_loadings)
        return _graded_information(self.probability(theta), slopes)

    def item_information_matrix(
        self,
        theta: NDArray[np.float64],
        item_idx: int,
    ) -> NDArray[np.float64]:
        """Return exact Fisher matrices of shape ``(n_theta, n_factors, n_factors)``.

        Every cumulative logit has the item's sparse slope vector as its
        ability gradient, so each matrix is the unit-slope graded information
        times the outer product of that vector. The scalar ``information``
        returns the trace of this matrix.
        """
        item_idx = self._validate_item_index(item_idx)
        theta = self._ensure_theta_2d(theta)
        slope = self._item_slope(item_idx)
        information = _graded_information(
            self._category_probabilities(theta, item_idx), 1.0
        )
        return information[:, None, None] * np.outer(slope, slope)

    def test_information_matrix(
        self,
        theta: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Sum item Fisher matrices across conditionally independent items."""
        theta = self._ensure_theta_2d(theta)
        information = np.zeros((theta.shape[0], self.n_factors, self.n_factors))
        for item_idx in range(self.n_items):
            information += self.item_information_matrix(theta, item_idx)
        return information

    def copy(self) -> "GradedBifactorModel":
        """Return an independent copy that preserves the bifactor structure."""
        new_model = self.__class__(
            n_items=self.n_items,
            specific_factors=self._specific_factors.copy(),
            n_categories=self._n_categories.copy(),
            item_names=self.item_names.copy(),
        )
        new_model._parameters = {
            name: values.copy() for name, values in self._parameters.items()
        }
        new_model._is_fitted = self._is_fitted
        self._copy_parameter_restrictions_to(new_model)
        return new_model
