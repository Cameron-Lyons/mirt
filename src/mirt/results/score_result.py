"""Result container for person scoring."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    import pandas as pd


@dataclass
class ScoreResult:
    """Container for person ability (theta) estimates.

    This class holds the results of scoring respondents, including
    theta estimates and their standard errors.

    Parameters
    ----------
    theta : ndarray of shape (n_persons,) or (n_persons, n_factors)
        Estimated ability values.
    standard_error : ndarray
        Standard errors of theta estimates.
    method : str
        Scoring method used (EAP, MAP, ML, WLE).
    person_ids : list, optional
        Identifiers for each person.

    Examples
    --------
    >>> scores = mirt.fscores(result, responses, method='EAP')
    >>> print(scores.theta)
    >>> print(scores.to_dataframe())
    """

    theta: NDArray[np.float64]
    standard_error: NDArray[np.float64]
    method: str
    person_ids: Optional[list] = None

    @property
    def n_persons(self) -> int:
        """Number of persons scored."""
        return self.theta.shape[0]

    @property
    def n_factors(self) -> int:
        """Number of factors."""
        if self.theta.ndim == 1:
            return 1
        return self.theta.shape[1]

    def to_dataframe(self) -> "pd.DataFrame":
        """Convert results to a pandas DataFrame.

        Returns
        -------
        pandas.DataFrame
            DataFrame with theta and SE columns.
        """
        import pandas as pd

        data = {}

        if self.n_factors == 1:
            theta_1d = self.theta.ravel()
            se_1d = self.standard_error.ravel()
            data["theta"] = theta_1d
            data["se"] = se_1d
        else:
            for j in range(self.n_factors):
                data[f"theta_{j+1}"] = self.theta[:, j]
                data[f"se_{j+1}"] = self.standard_error[:, j]

        df = pd.DataFrame(data)

        if self.person_ids is not None:
            df.index = self.person_ids
            df.index.name = "person"

        return df

    def to_array(self, include_se: bool = False) -> NDArray[np.float64]:
        """Return theta values as array, optionally with SEs.

        Parameters
        ----------
        include_se : bool, default=False
            Whether to include standard errors.

        Returns
        -------
        ndarray
            Theta values, or theta and SE concatenated.
        """
        if not include_se:
            return self.theta.copy()

        if self.n_factors == 1:
            return np.column_stack([self.theta.ravel(), self.standard_error.ravel()])

        return np.column_stack([self.theta, self.standard_error])

    def __repr__(self) -> str:
        return (
            f"ScoreResult(n_persons={self.n_persons}, "
            f"n_factors={self.n_factors}, "
            f"method='{self.method}')"
        )
