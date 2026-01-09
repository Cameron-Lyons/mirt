"""CAT result classes for tracking adaptive testing state and outcomes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    pass


@dataclass
class CATState:
    """Current state during CAT administration.

    This class tracks the evolving state of a CAT session, including the
    current ability estimate, items administered, and whether the test
    has reached a stopping condition.

    Attributes
    ----------
    theta : float
        Current ability estimate.
    standard_error : float
        Standard error of the current ability estimate.
    items_administered : list[int]
        Indices of items that have been administered.
    responses : list[int]
        Responses to administered items (0/1 for dichotomous, 0..k for polytomous).
    n_items : int
        Number of items administered so far.
    is_complete : bool
        Whether the CAT session has reached a stopping condition.
    next_item : int | None
        Index of the next item to administer, or None if complete.
    """

    theta: float
    standard_error: float
    items_administered: list[int] = field(default_factory=list)
    responses: list[int] = field(default_factory=list)
    n_items: int = 0
    is_complete: bool = False
    next_item: int | None = None

    def __repr__(self) -> str:
        return (
            f"CATState(theta={self.theta:.3f}, "
            f"se={self.standard_error:.3f}, "
            f"n_items={self.n_items}, "
            f"complete={self.is_complete})"
        )


@dataclass
class CATResult:
    """Final result of a completed CAT session.

    This class contains the complete record of a CAT administration,
    including the final ability estimate, all items administered,
    responses given, and the history of ability estimates.

    Attributes
    ----------
    theta : float
        Final ability estimate.
    standard_error : float
        Standard error of the final ability estimate.
    items_administered : list[int]
        Indices of all items administered in order.
    responses : NDArray[np.int_]
        Array of responses to administered items.
    n_items_administered : int
        Total number of items administered.
    stopping_reason : str
        Description of why the test stopped (e.g., "SE threshold reached").
    theta_history : list[float]
        History of ability estimates after each item.
    se_history : list[float]
        History of standard errors after each item.
    item_info_history : list[float]
        History of item information values for selected items.
    """

    theta: float
    standard_error: float
    items_administered: list[int]
    responses: NDArray[np.int_]
    n_items_administered: int
    stopping_reason: str
    theta_history: list[float] = field(default_factory=list)
    se_history: list[float] = field(default_factory=list)
    item_info_history: list[float] = field(default_factory=list)

    def summary(self) -> str:
        """Return a formatted summary of the CAT result.

        Returns
        -------
        str
            Multi-line summary string.
        """
        lines = [
            "CAT Result Summary",
            "=" * 40,
            f"Final theta estimate:  {self.theta:.4f}",
            f"Standard error:        {self.standard_error:.4f}",
            f"Items administered:    {self.n_items_administered}",
            f"Stopping reason:       {self.stopping_reason}",
            "",
            "Response pattern:",
            f"  Correct: {np.sum(self.responses == 1)} / {self.n_items_administered}",
            f"  Items:   {self.items_administered}",
        ]
        return "\n".join(lines)

    def to_dataframe(self) -> Any:
        """Convert CAT history to a DataFrame.

        Returns
        -------
        DataFrame
            DataFrame with columns: item, response, theta, se, info.
        """
        from mirt.utils.dataframe import create_dataframe

        n = len(self.items_administered)
        data: dict[str, Any] = {
            "step": list(range(1, n + 1)),
            "item": self.items_administered,
            "response": list(self.responses),
            "theta": self.theta_history[:n] if self.theta_history else [np.nan] * n,
            "se": self.se_history[:n] if self.se_history else [np.nan] * n,
        }

        if self.item_info_history:
            data["info"] = self.item_info_history[:n]

        return create_dataframe(data)

    def to_array(self) -> NDArray[np.float64]:
        """Convert result to numpy array.

        Returns
        -------
        NDArray[np.float64]
            Array with shape (n_items, 4) containing [item, response, theta, se].
        """
        n = len(self.items_administered)
        arr = np.zeros((n, 4), dtype=np.float64)
        arr[:, 0] = self.items_administered
        arr[:, 1] = self.responses
        if self.theta_history:
            arr[:, 2] = self.theta_history[:n]
        if self.se_history:
            arr[:, 3] = self.se_history[:n]
        return arr

    def plot_convergence(self) -> Any:
        """Plot theta and SE convergence over items.

        Returns
        -------
        matplotlib.figure.Figure
            Figure with two subplots showing theta and SE history.

        Raises
        ------
        ImportError
            If matplotlib is not installed.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as e:
            raise ImportError(
                "matplotlib is required for plotting. "
                "Install with: pip install matplotlib"
            ) from e

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

        steps = list(range(1, len(self.theta_history) + 1))

        ax1.plot(steps, self.theta_history, "b-o", markersize=4)
        ax1.axhline(y=self.theta, color="r", linestyle="--", alpha=0.7)
        ax1.set_ylabel("Theta Estimate")
        ax1.set_title("CAT Convergence")
        ax1.grid(True, alpha=0.3)

        ax2.plot(steps, self.se_history, "g-o", markersize=4)
        ax2.axhline(y=self.standard_error, color="r", linestyle="--", alpha=0.7)
        ax2.set_xlabel("Items Administered")
        ax2.set_ylabel("Standard Error")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def __repr__(self) -> str:
        return (
            f"CATResult(theta={self.theta:.3f}, "
            f"se={self.standard_error:.3f}, "
            f"n_items={self.n_items_administered}, "
            f"reason='{self.stopping_reason}')"
        )
