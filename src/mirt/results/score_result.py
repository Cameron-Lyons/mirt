from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class ScoreResult:
    theta: NDArray[np.float64]
    standard_error: NDArray[np.float64]
    method: str
    person_ids: list | None = None

    @property
    def n_persons(self) -> int:
        return self.theta.shape[0]

    @property
    def n_factors(self) -> int:
        if self.theta.ndim == 1:
            return 1
        return self.theta.shape[1]

    def to_dataframe(self) -> Any:
        from mirt.utils.dataframe import create_dataframe

        data: dict[str, Any] = {}

        if self.n_factors == 1:
            theta_1d = self.theta.ravel()
            se_1d = self.standard_error.ravel()
            data["theta"] = theta_1d
            data["se"] = se_1d
        else:
            for j in range(self.n_factors):
                data[f"theta_{j + 1}"] = self.theta[:, j]
                data[f"se_{j + 1}"] = self.standard_error[:, j]

        return create_dataframe(
            data,
            index=self.person_ids,
            index_name="person" if self.person_ids else None,
        )

    def to_array(self, include_se: bool = False) -> NDArray[np.float64]:
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
