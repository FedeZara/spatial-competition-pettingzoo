from enum import Enum


class Topology(Enum):
    """
    Topology modes controlling the spatial structure of the environment.

    These modes determine how the edges of the spatial environment are treated,
    affecting movement and distance calculations.

    Modes:
        RECTANGLE: The environment has hard boundaries. Agents cannot move
                   beyond the edges, and distance is calculated using standard
                   Euclidean metrics.

        TORUS: The environment wraps around at the edges, creating a continuous
               loop. Agents moving off one edge reappear on the opposite side,
               and distance calculations account for this wrap-around effect.
    """

    RECTANGLE = "rectangle"
    TORUS = "torus"
