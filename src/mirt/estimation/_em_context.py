"""Prepared data and resources owned by a single EM fit."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import AbstractContextManager, ExitStack
from types import TracebackType
from typing import Any

import numpy as np
from numpy.typing import NDArray

_MAX_COUNT_ENTRIES = 1_000_000
_MAX_DIRECT_COUNT_ENTRIES = 131_072


class EMFitContext(AbstractContextManager["EMFitContext"]):
    def __init__(
        self,
        responses: NDArray[np.int_],
        *,
        compress: bool = False,
        native: bool = False,
    ) -> None:
        from mirt.estimation._patterns import compress_responses

        self.n_observations = responses.shape[0]
        self.frequencies: NDArray[np.float64] | None = None
        if compress:
            responses, self.frequencies = compress_responses(responses)
        # Validation has already normalized missing codes. Preserve full-width
        # values when they cannot be represented by the native response dtype.
        if native and responses.min() >= -(2**31) and responses.max() < 2**31:
            responses = np.ascontiguousarray(responses, dtype=np.int32)
        self.responses = responses
        self._observed: NDArray[np.bool_] | None = None
        self._components: tuple[NDArray[np.float64], NDArray[np.float64]] | None = None
        self._resources = ExitStack()
        self._executor: ThreadPoolExecutor | None = None
        self._native_pool: Any = None

    @property
    def observed(self) -> NDArray[np.bool_]:
        if self._observed is None:
            self._observed = self.responses >= 0
        return self._observed

    def expected_counts(
        self,
        posterior: NDArray[np.float64],
        person_weights: NDArray[np.float64] | None = None,
        *,
        cache_components: bool = True,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Accumulate all item counts with bounded floating-point scratch space."""
        n_persons, n_items = self.responses.shape
        if (
            person_weights is None
            and cache_components
            and 2 * self.responses.size <= _MAX_COUNT_ENTRIES
        ):
            correct, observed = self.response_components(0, n_persons)
            return correct.T @ posterior, observed.T @ posterior

        correct_counts = np.zeros((n_items, posterior.shape[1]))
        observed_counts = np.zeros_like(correct_counts)
        max_entries = _MAX_COUNT_ENTRIES
        if person_weights is not None or not cache_components:
            max_entries = min(max_entries, _MAX_DIRECT_COUNT_ENTRIES)
        chunk_size = max(1, max_entries // (2 * n_items))
        for start in range(0, n_persons, chunk_size):
            stop = min(start + chunk_size, n_persons)
            if person_weights is None and cache_components:
                correct, observed = self.response_components(start, stop)
            else:
                # Reduce through narrow, owned response blocks instead of
                # copying the person-by-quadrature posterior. Cached components
                # and caller-owned weights stay unchanged.
                data = self.responses[start:stop]
                observed = (data >= 0).astype(np.float64)
                if person_weights is not None:
                    observed *= person_weights[start:stop, None]
                correct = data * observed
            weights = posterior[start:stop]
            correct_counts += correct.T @ weights
            observed_counts += observed.T @ weights
            del correct, observed
        return correct_counts, observed_counts

    def expected_category_counts(
        self,
        item_idx: int,
        n_categories: int,
        posterior: NDArray[np.float64],
        person_weights: NDArray[np.float64] | NDArray[np.bool_] | None = None,
    ) -> NDArray[np.float64]:
        """Accumulate an item's category counts without posterior row copies."""
        counts = np.zeros((posterior.shape[1], n_categories))
        chunk_size = max(1, _MAX_COUNT_ENTRIES // n_categories)
        categories = np.arange(n_categories)
        n_persons = self.responses.shape[0]
        for start in range(0, n_persons, chunk_size):
            stop = min(start + chunk_size, n_persons)
            indicators = (
                self.responses[start:stop, item_idx, None] == categories
            ).astype(np.float64)
            if person_weights is not None:
                indicators *= person_weights[start:stop, None]
            counts += posterior[start:stop].T @ indicators
        return counts

    def response_components(
        self, start: int, stop: int
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return prepared response values and observed indicators for a row block.

        Small matrices are cached for repeated E/M-steps. Large matrices prepare
        only the requested rows, including their missing-response mask.
        """
        if 2 * self.responses.size <= _MAX_COUNT_ENTRIES:
            if self._components is None:
                self._components = (
                    np.where(self.observed, self.responses, 0).astype(np.float64),
                    self.observed.astype(np.float64),
                )
            return self._components[0][start:stop], self._components[1][start:stop]
        data = self.responses[start:stop]
        observed = data >= 0
        return np.where(observed, data, 0).astype(np.float64), observed.astype(
            np.float64
        )

    def executor(self, n_jobs: int) -> ThreadPoolExecutor:
        if self._executor is None:
            self._executor = self._resources.enter_context(
                ThreadPoolExecutor(max_workers=min(n_jobs, self.responses.shape[1]))
            )
        return self._executor

    def native_pool(self, n_jobs: int) -> Any:
        if n_jobs == 1 or self.responses.shape[1] < 2:
            return None
        if self._native_pool is None:
            from mirt.backends.rust._helpers import mirt_rs

            self._native_pool = mirt_rs.EMThreadPool(
                min(n_jobs, self.responses.shape[1])
            )
        return self._native_pool

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        try:
            self._resources.close()
        finally:
            self._executor = None
            self._native_pool = None
