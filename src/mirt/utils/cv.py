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
        n_persons = responses.shape[0]
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
        n_persons = responses.shape[0]

        sum_scores = np.sum(np.maximum(responses, 0), axis=1)

        bins = np.percentile(sum_scores, np.linspace(0, 100, self.n_bins + 1))
        bins = np.unique(bins)
        if len(bins) < 2:
            bins = np.array([sum_scores.min(), sum_scores.max() + 1])

        strata = np.digitize(sum_scores, bins[:-1]) - 1
        strata = np.clip(strata, 0, len(bins) - 2)

        rng = np.random.default_rng(self.random_state)

        fold_assignments = np.zeros(n_persons, dtype=int)

        for stratum in range(strata.max() + 1):
            stratum_indices = np.where(strata == stratum)[0]
            rng.shuffle(stratum_indices)
            for i, idx in enumerate(stratum_indices):
                fold_assignments[idx] = i % self.n_splits

        for fold in range(self.n_splits):
            test_idx = np.where(fold_assignments == fold)[0]
            train_idx = np.where(fold_assignments != fold)[0]
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
        n_persons = responses.shape[0]
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
    model_type: Literal["1PL", "2PL", "3PL", "4PL", "GRM", "GPCM", "PCM", "NRM"],
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
        Number of parallel jobs for fold fitting. Use -1 for all CPUs.

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

    from mirt import fit_mirt

    responses = np.asarray(responses)

    if splitter is None:
        splitter = KFold(n_splits=5)

    if scorers is None:
        scorers = [LogLikelihoodScorer()]

    scores: dict[str, list[float]] = {s.name: [] for s in scorers}
    fold_results: list[FitResult] = []

    if n_jobs == -1:
        n_jobs = os.cpu_count() or 1

    splits = list(splitter.split(responses))

    def fit_fold(
        fold_data: tuple[int, tuple[NDArray[np.intp], NDArray[np.intp]]],
    ) -> tuple[int, FitResult, NDArray[np.intp]]:
        fold_idx, (train_idx, test_idx) = fold_data
        train_data = responses[train_idx]
        result = fit_mirt(
            train_data,
            model=model_type,
            n_categories=n_categories,
            n_factors=n_factors,
            n_quadpts=n_quadpts,
            max_iter=max_iter,
            tol=tol,
            verbose=False,
        )
        return fold_idx, result, test_idx

    if n_jobs > 1 and len(splits) > 1:
        with ProcessPoolExecutor(max_workers=min(n_jobs, len(splits))) as executor:
            fold_data_list = list(enumerate(splits))
            results_list = list(executor.map(fit_fold, fold_data_list))

        results_list.sort(key=lambda x: x[0])

        for fold_idx, result, test_idx in results_list:
            if verbose:
                print(f"Fold {fold_idx + 1}/{splitter.n_splits} completed")

            train_idx = splits[fold_idx][0]
            train_data = responses[train_idx]
            test_data = responses[test_idx]

            if return_models:
                fold_results.append(result)

            for scorer in scorers:
                score = scorer(result, train_data, test_data, test_idx)
                scores[scorer.name].append(score)
    else:
        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            if verbose:
                print(f"Fold {fold_idx + 1}/{splitter.n_splits}")

            train_data = responses[train_idx]
            test_data = responses[test_idx]

            result = fit_mirt(
                train_data,
                model=model_type,
                n_categories=n_categories,
                n_factors=n_factors,
                n_quadpts=n_quadpts,
                max_iter=max_iter,
                tol=tol,
                verbose=False,
            )

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
        n_folds=splitter.n_splits,
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
