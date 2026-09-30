"""Response preparation shared by one IRTree fit's E/M and uncertainty steps."""

import numpy as np
from numpy.typing import NDArray

from mirt.estimation import _em_context
from mirt.estimation._em_context import EMFitContext


class IRTreeFitContext(EMFitContext):
    def __init__(
        self,
        pseudo_responses: NDArray[np.int_],
        valid_mask: NDArray[np.bool_],
    ) -> None:
        self.pseudo_responses = pseudo_responses
        self.valid_mask = valid_mask
        super().__init__(pseudo_responses.reshape(pseudo_responses.shape[0], -1))
        self._observed = valid_mask.reshape(self.responses.shape)

    def response_components(
        self, start: int, stop: int
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Only visited binary nodes contribute, including for private callers."""
        cached = 2 * self.responses.size <= _em_context._MAX_COUNT_ENTRIES
        if cached:
            if self._components is None:
                self._components = (
                    ((self.responses == 1) & self.observed).astype(np.float64),
                    self.observed.astype(np.float64),
                )
            return self._components[0][start:stop], self._components[1][start:stop]
        return self._uncached_response_components(start, stop)

    def _uncached_response_components(
        self,
        start: int,
        stop: int,
        person_weights: NDArray[np.float64] | None = None,
        *,
        first: int = 0,
        last: int | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        data = self.responses[start:stop, first:last]
        observed = self.observed[start:stop, first:last].astype(np.float64)
        if person_weights is not None:
            observed *= person_weights[start:stop, None]
        return (data == 1) * observed, observed

    def expected_totals(self, posterior: NDArray[np.float64]) -> NDArray[np.float64]:
        """Prepare uncertainty counts without computing unused correct counts."""
        n_persons, n_nodes = self.responses.shape
        if 2 * self.responses.size <= _em_context._MAX_COUNT_ENTRIES:
            observed = (
                self.observed.astype(np.float64)
                if self._components is None
                else self._components[1]
            )
            return observed.T @ posterior
        total = np.zeros((n_nodes, posterior.shape[1]))
        chunk_size = max(1, _em_context._MAX_COUNT_ENTRIES // n_nodes)
        for start in range(0, n_persons, chunk_size):
            stop = min(start + chunk_size, n_persons)
            observed = self.observed[start:stop].astype(np.float64)
            total += observed.T @ posterior[start:stop]
        return total

    def node_components(
        self, start: int, stop: int, first: int, last: int
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Prepare only the selected columns when a wide matrix cannot be cached."""
        if 2 * self.responses.size <= _em_context._MAX_COUNT_ENTRIES:
            correct, observed = self.response_components(start, stop)
            return correct[:, first:last], observed[:, first:last]
        return self._uncached_response_components(start, stop, first=first, last=last)
