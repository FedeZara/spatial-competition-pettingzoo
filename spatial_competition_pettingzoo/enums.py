"""Enums used across the spatial competition environment."""

from enum import Enum


class TransportationCostNorm(Enum):
    """
    Transportation cost norm based on Lp norms (Minkowski distance).

    The transportation cost norm determines how distance between buyers and sellers
    is calculated, affecting buyer utility and market dynamics.

    Norms:
        L1: Manhattan distance (sum of absolute differences)
        L2: Euclidean distance (square root of sum of squared differences)
        L3: Minkowski distance with p=3
        L4: Minkowski distance with p=4
        L_INF: Chebyshev distance (maximum absolute difference)
    """

    L1 = 1
    L2 = 2
    L3 = 3
    L4 = 4
    L_INF = float("inf")

    @property
    def order(self) -> float:
        """Return the norm order for numpy.linalg.norm()."""
        return float(self.value)


class InformationLevel(Enum):
    """
    Information modes for sellers - determines what type of information they can observe.

    These modes create different levels of information asymmetry in the spatial
    competition game, affecting strategic behavior and equilibrium outcomes.

    Modes:
        PRIVATE: Sellers only see their own position, price, and quality.

        LIMITED: Sellers also see buyer valuation of their product.

        COMPLETE: Sellers also see all other sellers' prices and qualities.
    """

    PRIVATE = "private"
    LIMITED = "limited"
    COMPLETE = "complete"
