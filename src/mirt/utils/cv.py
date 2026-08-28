"""Cross-validation framework for IRT models.

This module provides flexible cross-validation tools:
- Data splitting strategies (K-Fold, Stratified, Leave-One-Out)
- Scoring metrics for model evaluation
- Main cross_validate function
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from mirt.results.fit_result import FitResult


ModelType = Literal["1PL", "2PL", "3PL", "4PL", "GRM", "GPCM", "PCM", "NRM"]


def _validate_split_responses(
    responses: NDArray[np.int_],
    *,
    minimum_persons: int = 1,
) -> NDArray[Any]:
    """Return a response matrix suitable for standalone splitter use."""
    values = np.asarray(responses)
    if values.ndim != 2 or values.shape[0] < minimum_persons or values.shape[1] == 0:
        raise ValueError(
            "responses must be a two-dimensional matrix with at least "
            f"{minimum_persons} persons and 1 item"
        )
    return values


def _validate_shuffle(value: bool) -> None:
    """Validate a splitter shuffle flag."""
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError("shuffle must be a boolean")


@runtime_checkable
class Splitter(Protocol):
    """Protocol for cross-validation splitters."""

    @property
    def n_splits(self) -> int:
        """Number of folds/splits."""
        ...

    def split(
        self,
        responses: NDArray[np.int_],
    ) -> Iterator[tuple[NDArray[np.intp], NDArray[np.intp]]]:
        """Yield (train_indices, test_indices) tuples."""
        ...


@dataclass
class KFold:
    """K-Fold cross-validation splitter.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds.
    shuffle : bool, default=True
        Whether to shuffle before splitting.
    random_state : int | None, default=None
        Random seed for reproducibility.

    Examples
    --------
    >>> splitter = KFold(n_splits=5, shuffle=True, random_state=42)
    >>> for train_idx, test_idx in splitter.split(responses):
    ...     train_data = responses[train_idx]
    ...     test_data = responses[test_idx]
    """

    n_splits: int = 5
    shuffle: bool = True
    random_state: int | None = None

    def __post_init__(self) -> None:
        _validate_split_count(self.n_splits)
        _validate_shuffle(self.shuffle)

    def split(
        self,
        responses: NDArray[np.int_],
    ) -> Iterator[tuple[NDArray[np.intp], NDArray[np.intp]]]:
        """Split data into k folds.

        Parameters
        ----------
        responses : NDArray
            Response matrix (n_persons, n_items).

        Yields
        ------
        train_idx, test_idx : tuple[NDArray, NDArray]
            Indices for training and testing sets.
        """
        response_values = _validate_split_responses(responses)
        n_persons = response_values.shape[0]
        if self.n_splits > n_persons:
            raise ValueError(
                f"n_splits={self.n_splits} cannot exceed n_persons={n_persons}"
            )
        indices = np.arange(n_persons)

        if self.shuffle:
            rng = np.random.default_rng(self.random_state)
            rng.shuffle(indices)

        fold_sizes = np.full(self.n_splits, n_persons // self.n_splits)
        fold_sizes[: n_persons % self.n_splits] += 1

        current = 0
        for fold_size in fold_sizes:
            test_idx = indices[current : current + fold_size]
            train_idx = np.concatenate(
                [indices[:current], indices[current + fold_size :]]
            )
            yield train_idx, test_idx
            current += fold_size


@dataclass
class StratifiedKFold:
    """Stratified K-Fold based on sum scores.

    Ensures each fold has similar score distribution by stratifying
    on binned sum scores.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds.
    n_bins : int, default=5
        Number of bins for stratification.
    random_state : int | None, default=None
        Random seed.
    shuffle : bool, default=True
        Whether to shuffle persons within each score stratum.

    Examples
    --------
    >>> splitter = StratifiedKFold(n_splits=5, n_bins=5, random_state=42)
    >>> for train_idx, test_idx in splitter.split(responses):
    ...     # Each fold has similar score distribution
    ...     pass
    """

    n_splits: int = 5
    n_bins: int = 5
    random_state: int | None = None
    shuffle: bool = True

    def __post_init__(self) -> None:
        _validate_split_count(self.n_splits)
        if isinstance(self.n_bins, bool) or not isinstance(
            self.n_bins, (int, np.integer)
        ):
            raise ValueError("n_bins must be an integer")
        if self.n_bins < 1:
            raise ValueError("n_bins must be at least 1")
        _validate_shuffle(self.shuffle)

    def split(
        self,
        responses: NDArray[np.int_],
    ) -> Iterator[tuple[NDArray[np.intp], NDArray[np.intp]]]:
        """Split data with stratification on sum scores.

        Parameters
        ----------
        responses : NDArray
            Response matrix (n_persons, n_items).

        Yields
        ------
        train_idx, test_idx : tuple[NDArray, NDArray]
            Indices for training and testing sets.
        """
        response_values = _validate_split_responses(responses)
        n_persons = response_values.shape[0]
        if self.n_splits > n_persons:
            raise ValueError(
                f"n_splits={self.n_splits} cannot exceed n_persons={n_persons}"
            )

        numeric_scores = np.issubdtype(
            response_values.dtype, np.number
        ) or np.issubdtype(response_values.dtype, np.bool_)
        if not numeric_scores or np.issubdtype(
            response_values.dtype, np.complexfloating
        ):
            raise ValueError("stratified responses must contain numeric scores")
        if np.any(np.isinf(response_values)):
            raise ValueError("stratified responses must not contain infinite values")
        observed = np.isfinite(response_values) & (response_values >= 0)
        sum_scores = np.sum(np.where(observed, response_values, 0.0), axis=1)

        bins = np.percentile(sum_scores, np.linspace(0, 100, self.n_bins + 1))
        bins = np.unique(bins)
        if len(bins) < 2:
            bins = np.array([sum_scores.min(), sum_scores.max() + 1])

        strata = np.digitize(sum_scores, bins[:-1]) - 1
        strata = np.clip(strata, 0, len(bins) - 2)

        rng = np.random.default_rng(self.random_state) if self.shuffle else None

        fold_assignments = np.zeros(n_persons, dtype=np.intp)

        next_fold = 0
        for stratum in range(strata.max() + 1):
            stratum_indices = np.flatnonzero(strata == stratum)
            if rng is not None:
                rng.shuffle(stratum_indices)
            fold_assignments[stratum_indices] = (
                next_fold + np.arange(stratum_indices.size, dtype=np.intp)
            ) % self.n_splits
            next_fold = (next_fold + len(stratum_indices)) % self.n_splits

        for fold in range(self.n_splits):
            test_idx = np.flatnonzero(fold_assignments == fold)
            train_idx = np.flatnonzero(fold_assignments != fold)
            yield train_idx, test_idx


@dataclass
class GroupKFold:
    """Cross-validation splitter that keeps each group in one fold.

    This splitter prevents leakage when rows are clustered, repeated, or
    otherwise share a subject, site, classroom, test form, or similar group.
    Groups are assigned largest-first to the currently lightest fold so fold
    sizes remain as balanced as the group sizes permit.

    Parameters
    ----------
    groups : NDArray
        One-dimensional group label for every response-matrix row.
    n_splits : int, default=5
        Number of folds. Cannot exceed the number of unique groups.
    shuffle : bool, default=False
        Randomize equal-sized group ordering and fold labels while retaining
        largest-first load balancing.
    random_state : int | None, default=None
        Random seed used when ``shuffle=True``.

    Examples
    --------
    >>> groups = np.repeat(np.arange(20), 5)
    >>> splitter = GroupKFold(groups, n_splits=5)
    >>> for train_idx, test_idx in splitter.split(responses):
    ...     assert not set(groups[train_idx]) & set(groups[test_idx])
    """

    groups: NDArray[Any]
    n_splits: int = 5
    shuffle: bool = False
    random_state: int | None = None

    def __post_init__(self) -> None:
        _validate_split_count(self.n_splits)
        _validate_shuffle(self.shuffle)

    def split(
        self,
        responses: NDArray[np.int_],
    ) -> Iterator[tuple[NDArray[np.intp], NDArray[np.intp]]]:
        """Yield complementary train and test indices without group leakage."""
        response_values = _validate_split_responses(responses)
        labels = np.asarray(self.groups)
        if labels.ndim != 1:
            raise ValueError("groups must be one-dimensional")
        if labels.shape[0] != response_values.shape[0]:
            raise ValueError(
                "groups and responses must contain the same number of rows"
            )
        if labels.dtype.kind in "fc" and not np.all(np.isfinite(labels)):
            raise ValueError("groups must not contain missing or non-finite labels")
        if labels.dtype.kind == "O":
            missing_object_label = any(
                label is None
                or (
                    isinstance(
                        label,
                        (float, complex, np.floating, np.complexfloating),
                    )
                    and not np.isfinite(label)
                )
                for label in labels
            )
            if missing_object_label:
                raise ValueError("groups must not contain missing labels")

        try:
            unique_groups, group_indices, group_sizes = np.unique(
                labels, return_inverse=True, return_counts=True
            )
        except TypeError as exc:
            raise ValueError("group labels must be mutually comparable") from exc
        n_groups = unique_groups.size
        if self.n_splits > n_groups:
            raise ValueError(
                f"n_splits={self.n_splits} cannot exceed n_groups={n_groups}"
            )

        if self.shuffle:
            rng = np.random.default_rng(self.random_state)
            tie_breakers = rng.random(n_groups)
            fold_order = rng.permutation(self.n_splits)
            group_order = np.lexsort((tie_breakers, -group_sizes))
        else:
            fold_order = np.arange(self.n_splits, dtype=np.intp)
            group_order = np.argsort(-group_sizes, kind="stable")

        group_folds = np.empty(n_groups, dtype=np.intp)
        if np.all(group_sizes == group_sizes[0]):
            group_folds[group_order] = fold_order[
                np.arange(n_groups, dtype=np.intp) % self.n_splits
            ]
        else:
            fold_sizes = np.zeros(self.n_splits, dtype=np.intp)
            for group_idx in group_order:
                fold = fold_order[np.argmin(fold_sizes[fold_order])]
                group_folds[group_idx] = fold
                fold_sizes[fold] += group_sizes[group_idx]

        fold_assignments = group_folds[group_indices]
        for fold in range(self.n_splits):
            test_idx = np.flatnonzero(fold_assignments == fold)
            train_idx = np.flatnonzero(fold_assignments != fold)
            yield train_idx, test_idx


@dataclass
class LeaveOneOut:
    """Leave-One-Out cross-validation splitter.

    Each observation is used once as the test set while all
    remaining observations form the training set.

    Examples
    --------
    >>> splitter = LeaveOneOut()
    >>> for train_idx, test_idx in splitter.split(responses):
    ...     # test_idx contains exactly one index
    ...     pass
    """

    _n_splits: int = field(default=0, init=False, repr=False)

    @property
    def n_splits(self) -> int:
        """Number of splits (equals number of samples)."""
        return self._n_splits

    def split(
        self,
        responses: NDArray[np.int_],
    ) -> Iterator[tuple[NDArray[np.intp], NDArray[np.intp]]]:
        """Split data with leave-one-out.

        Parameters
        ----------
        responses : NDArray
            Response matrix (n_persons, n_items).

        Yields
        ------
        train_idx, test_idx : tuple[NDArray, NDArray]
            Indices for training and testing sets.
        """
        response_values = _validate_split_responses(responses, minimum_persons=2)
        n_persons = response_values.shape[0]
        self._n_splits = n_persons
        indices = np.arange(n_persons)

        for i in range(n_persons):
            test_idx = np.array([i])
            train_idx = np.concatenate([indices[:i], indices[i + 1 :]])
            yield train_idx, test_idx


class Scorer(ABC):
    """Abstract base class for cross-validation scorers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the scorer for results dictionary."""
        ...

    @abstractmethod
    def __call__(
        self,
        result: FitResult,
        train_responses: NDArray[np.int_],
        test_responses: NDArray[np.int_],
        test_indices: NDArray[np.intp] | None = None,
    ) -> float:
        """Compute score on test data.

        Parameters
        ----------
        result : FitResult
            Fitted model result from training data.
        train_responses : NDArray
            Training response matrix.
        test_responses : NDArray
            Test response matrix.
        test_indices : NDArray, optional
            Original indices of test observations.

        Returns
        -------
        float
            Score value (higher is generally better).
        """
        ...


@dataclass
class LogLikelihoodScorer(Scorer):
    """Scorer based on log-likelihood on held-out data.

    Computes the log-likelihood of the test data given the model
    fitted on training data.
    """

    @property
    def name(self) -> str:
        return "log_likelihood"

    def __call__(
        self,
        result: FitResult,
        train_responses: NDArray[np.int_],
        test_responses: NDArray[np.int_],
        test_indices: NDArray[np.intp] | None = None,
    ) -> float:
        """Compute log-likelihood on test data."""
        from mirt.scoring import fscores

        _ = train_responses
        scores = fscores(result.model, test_responses, method="EAP")
        theta = scores.theta
        if theta.ndim == 1:
            theta = theta.reshape(-1, 1)

        ll = result.model.log_likelihood(test_responses, theta)
        return float(np.sum(ll))


@dataclass
class AbilityRMSEScorer(Scorer):
    """Scorer based on ability estimation RMSE.

    Requires true theta values to be provided. Useful for
    simulation studies.

    Parameters
    ----------
    true_theta : NDArray
        True ability values for all persons.
    """

    true_theta: NDArray[np.float64]

    @property
    def name(self) -> str:
        return "ability_rmse"

    def __call__(
        self,
        result: FitResult,
        train_responses: NDArray[np.int_],
        test_responses: NDArray[np.int_],
        test_indices: NDArray[np.intp] | None = None,
    ) -> float:
        """Compute RMSE between estimated and true abilities."""
        from mirt.scoring import fscores

        _ = train_responses
        scores = fscores(result.model, test_responses, method="EAP")
        estimated = scores.theta.ravel()

        if test_indices is not None:
            true = self.true_theta[test_indices]
        else:
            true = self.true_theta[: len(estimated)]

        return -float(np.sqrt(np.mean((estimated - true) ** 2)))


@dataclass
class AICScorer(Scorer):
    """Scorer based on AIC (Akaike Information Criterion).

    Returns negative AIC since lower AIC is better but
    cross-validation expects higher scores to be better.
    """

    @property
    def name(self) -> str:
        return "aic"

    def __call__(
        self,
        result: FitResult,
        train_responses: NDArray[np.int_],
        test_responses: NDArray[np.int_],
        test_indices: NDArray[np.intp] | None = None,
    ) -> float:
        """Return negative AIC (higher is better)."""
        _ = train_responses, test_responses, test_indices
        return -result.aic


@dataclass
class BICScorer(Scorer):
    """Scorer based on BIC (Bayesian Information Criterion).

    Returns negative BIC since lower BIC is better but
    cross-validation expects higher scores to be better.
    """

    @property
    def name(self) -> str:
        return "bic"

    def __call__(
        self,
        result: FitResult,
        train_responses: NDArray[np.int_],
        test_responses: NDArray[np.int_],
        test_indices: NDArray[np.intp] | None = None,
    ) -> float:
        """Return negative BIC (higher is better)."""
        _ = train_responses, test_responses, test_indices
        return -result.bic


def _validate_split_count(n_splits: int) -> None:
    """Validate a requested number of cross-validation folds."""
    if isinstance(n_splits, bool) or not isinstance(n_splits, (int, np.integer)):
        raise ValueError("n_splits must be an integer")
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")


@dataclass(frozen=True)
class _CVFoldTask:
    """Serializable inputs for fitting one cross-validation fold."""

    fold_idx: int
    train_responses: NDArray[np.int_]
    model_type: ModelType
    n_categories: int | None
    n_factors: int
    n_quadpts: int
    max_iter: int
    tol: float


def _fit_cv_fold(task: _CVFoldTask) -> tuple[int, FitResult]:
    """Fit one fold in a process-safe module-level worker."""
    from mirt import fit_mirt

    result = fit_mirt(
        task.train_responses,
        model=task.model_type,
        n_categories=task.n_categories,
        n_factors=task.n_factors,
        n_quadpts=task.n_quadpts,
        max_iter=task.max_iter,
        tol=task.tol,
        verbose=False,
    )
    return task.fold_idx, result


def _validated_splits(
    splitter: Splitter,
    responses: NDArray[np.int_],
) -> list[tuple[NDArray[np.intp], NDArray[np.intp]]]:
    """Materialize and validate indices supplied by a splitter."""
    n_persons = responses.shape[0]
    splits: list[tuple[NDArray[np.intp], NDArray[np.intp]]] = []

    for fold_idx, (train_indices, test_indices) in enumerate(splitter.split(responses)):
        train_values = np.asarray(train_indices)
        test_values = np.asarray(test_indices)
        if train_values.ndim != 1 or test_values.ndim != 1:
            raise ValueError(f"fold {fold_idx} indices must be one-dimensional")
        if train_values.size == 0 or test_values.size == 0:
            raise ValueError(f"fold {fold_idx} must have non-empty train and test sets")
        if not np.issubdtype(train_values.dtype, np.integer) or not np.issubdtype(
            test_values.dtype, np.integer
        ):
            raise ValueError(f"fold {fold_idx} indices must be integers")
        train_idx = train_values.astype(np.intp, copy=False)
        test_idx = test_values.astype(np.intp, copy=False)
        if (
            np.any(train_idx < 0)
            or np.any(train_idx >= n_persons)
            or np.any(test_idx < 0)
            or np.any(test_idx >= n_persons)
        ):
            raise ValueError(f"fold {fold_idx} contains out-of-bounds indices")
        if np.unique(train_idx).size != train_idx.size:
            raise ValueError(f"fold {fold_idx} contains duplicate training indices")
        if np.unique(test_idx).size != test_idx.size:
            raise ValueError(f"fold {fold_idx} contains duplicate test indices")
        if np.intersect1d(train_idx, test_idx, assume_unique=True).size:
            raise ValueError(f"fold {fold_idx} train and test sets overlap")
        splits.append((train_idx, test_idx))

    if not splits:
        raise ValueError("splitter produced no folds")
    return splits


@dataclass
class CVResult:
    """Result of cross-validation.

    Attributes
    ----------
    scores : dict[str, list[float]]
        Scores per fold for each scorer.
    mean_scores : dict[str, float]
        Mean score across folds.
    std_scores : dict[str, float]
        Standard deviation across folds.
    n_folds : int
        Number of folds.
    fold_results : list[FitResult] | None
        Fitted results for each fold (if return_models=True).
    """

    scores: dict[str, list[float]]
    mean_scores: dict[str, float]
    std_scores: dict[str, float]
    n_folds: int
    fold_results: list[FitResult] | None = None

    def summary(self) -> str:
        """Generate a text summary of cross-validation results.

        Returns
        -------
        str
            Formatted summary string.
        """
        lines = ["Cross-Validation Results", "=" * 50]
        lines.append(f"Number of folds: {self.n_folds}")
        lines.append("-" * 50)
        lines.append(f"{'Metric':<20} {'Mean':>12} {'Std':>12}")
        lines.append("-" * 50)
        for metric in self.mean_scores:
            mean = self.mean_scores[metric]
            std = self.std_scores[metric]
            lines.append(f"{metric:<20} {mean:>12.4f} {std:>12.4f}")
        return "\n".join(lines)

    def to_dataframe(self) -> Any:
        """Convert results to a DataFrame.

        Returns
        -------
        DataFrame
            Results as pandas or polars DataFrame.
        """
        from mirt.utils.dataframe import create_dataframe

        data = {
            "metric": list(self.mean_scores.keys()),
            "mean": list(self.mean_scores.values()),
            "std": list(self.std_scores.values()),
        }
        return create_dataframe(data)


def cross_validate(
    model_type: ModelType,
    responses: NDArray[np.int_],
    splitter: Splitter | None = None,
    scorers: list[Scorer] | None = None,
    n_categories: int | None = None,
    n_factors: int = 1,
    n_quadpts: int = 21,
    max_iter: int = 500,
    tol: float = 1e-4,
    verbose: bool = False,
    return_models: bool = False,
    n_jobs: int = 1,
) -> CVResult:
    """Perform cross-validation for an IRT model.

    Parameters
    ----------
    model_type : str
        Type of IRT model to fit ('1PL', '2PL', '3PL', '4PL',
        'GRM', 'GPCM', 'PCM', 'NRM').
    responses : NDArray
        Response matrix (n_persons, n_items).
    splitter : Splitter, optional
        Data splitting strategy. Default is KFold(n_splits=5).
    scorers : list[Scorer], optional
        Scoring functions. Default is [LogLikelihoodScorer()].
    n_categories : int, optional
        Number of categories for polytomous models.
    n_factors : int, default=1
        Number of latent factors.
    n_quadpts : int, default=21
        Quadrature points for EM.
    max_iter : int, default=500
        Maximum EM iterations.
    tol : float, default=1e-4
        Convergence tolerance.
    verbose : bool, default=False
        Print progress.
    return_models : bool, default=False
        Whether to return fitted models for each fold.
    n_jobs : int, default=1
        Number of process workers for fold fitting. Use -1 for all CPUs.
        Process startup has overhead, so ``n_jobs=1`` is preferable for
        very small or fast fits.

    Returns
    -------
    CVResult
        Cross-validation results with scores per fold.

    Examples
    --------
    >>> from mirt import load_dataset
    >>> from mirt.utils.cv import cross_validate, KFold, LogLikelihoodScorer
    >>> data = load_dataset("LSAT7")
    >>> cv_result = cross_validate(
    ...     model_type="2PL",
    ...     responses=data["data"],
    ...     splitter=KFold(n_splits=5, random_state=42),
    ...     scorers=[LogLikelihoodScorer()],
    ... )
    >>> print(cv_result.summary())
    """
    import os
    from concurrent.futures import ProcessPoolExecutor

    responses = np.asarray(responses)
    if responses.ndim != 2 or responses.shape[0] < 2 or responses.shape[1] == 0:
        raise ValueError(
            "responses must be a two-dimensional matrix with at least "
            "2 persons and 1 item"
        )
    if isinstance(n_jobs, bool) or not isinstance(n_jobs, (int, np.integer)):
        raise ValueError("n_jobs must be an integer")
    if n_jobs == 0 or n_jobs < -1:
        raise ValueError("n_jobs must be -1 or a positive integer")
    n_jobs = int(n_jobs)

    if splitter is None:
        splitter = KFold(n_splits=5)

    if scorers is None:
        scorers = [LogLikelihoodScorer()]
    if not scorers:
        raise ValueError("scorers must contain at least one scorer")
    scorer_names = [scorer.name for scorer in scorers]
    if any(not isinstance(name, str) or not name for name in scorer_names):
        raise ValueError("each scorer must have a non-empty string name")
    if len(set(scorer_names)) != len(scorer_names):
        raise ValueError("scorer names must be unique")

    scores: dict[str, list[float]] = {name: [] for name in scorer_names}
    fold_results: list[FitResult] = []

    if n_jobs == -1:
        n_jobs = os.cpu_count() or 1

    splits = _validated_splits(splitter, responses)
    n_folds = len(splits)
    tasks = [
        _CVFoldTask(
            fold_idx=fold_idx,
            train_responses=responses[train_idx],
            model_type=model_type,
            n_categories=n_categories,
            n_factors=n_factors,
            n_quadpts=n_quadpts,
            max_iter=max_iter,
            tol=tol,
        )
        for fold_idx, (train_idx, _) in enumerate(splits)
    ]

    if n_jobs > 1 and n_folds > 1:
        with ProcessPoolExecutor(max_workers=min(n_jobs, n_folds)) as executor:
            results_list = list(executor.map(_fit_cv_fold, tasks))

        results_list.sort(key=lambda x: x[0])

        for fold_idx, result in results_list:
            if verbose:
                print(f"Fold {fold_idx + 1}/{n_folds} completed")

            train_idx, test_idx = splits[fold_idx]
            train_data = responses[train_idx]
            test_data = responses[test_idx]

            if return_models:
                fold_results.append(result)

            for scorer in scorers:
                score = scorer(result, train_data, test_data, test_idx)
                scores[scorer.name].append(score)
    else:
        for task, (train_idx, test_idx) in zip(tasks, splits):
            if verbose:
                print(f"Fold {task.fold_idx + 1}/{n_folds}")

            train_data = responses[train_idx]
            test_data = responses[test_idx]

            _, result = _fit_cv_fold(task)

            if return_models:
                fold_results.append(result)

            for scorer in scorers:
                score = scorer(result, train_data, test_data, test_idx)
                scores[scorer.name].append(score)

    mean_scores = {k: float(np.mean(v)) for k, v in scores.items()}
    std_scores = {
        k: float(np.std(v, ddof=1)) if len(v) > 1 else 0.0 for k, v in scores.items()
    }

    return CVResult(
        scores=scores,
        mean_scores=mean_scores,
        std_scores=std_scores,
        n_folds=n_folds,
        fold_results=fold_results if return_models else None,
    )


__all__ = [
    "Splitter",
    "KFold",
    "StratifiedKFold",
    "LeaveOneOut",
    "Scorer",
    "LogLikelihoodScorer",
    "AbilityRMSEScorer",
    "AICScorer",
    "BICScorer",
    "CVResult",
    "cross_validate",
]
