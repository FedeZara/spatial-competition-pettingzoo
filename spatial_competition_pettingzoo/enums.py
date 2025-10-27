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


class ViewScope(Enum):
    """
    View scope modes controlling the spatial range of observation.

    These modes determine how much of the map sellers can see, independent
    of what type of information they can observe.

    Modes:
        LIMITED: Sellers have restricted spatial vision (vision_radius parameter)
                and only see information within that range. Creates local
                information and spatial competition effects.

        COMPLETE: Sellers can see the entire space. All available information
                 (based on InformationLevel) is visible across the full spatial area.
    """

    LIMITED = "limited"
    COMPLETE = "complete"
