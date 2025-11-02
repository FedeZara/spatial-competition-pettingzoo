"""Enums used across the spatial competition environment."""

from enum import Enum


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
