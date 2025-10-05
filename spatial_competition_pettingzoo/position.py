from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.random import Generator

from spatial_competition_pettingzoo.topology import Topology


class Position:
    def __init__(
        self,
        space_resolution: float,
        topology: Topology,
        tensor_coordinates: np.ndarray[Any, np.dtype[np.int32]] | None = None,
        space_coordinates: np.ndarray[Any, np.dtype[np.float32]] | None = None,
    ) -> None:
        if tensor_coordinates is None and space_coordinates is None:
            msg = "Either tensor_coordinates or space_coordinates must be provided."
            raise ValueError(msg)
        if tensor_coordinates is not None and space_coordinates is not None:
            msg = "Only one of tensor_coordinates or space_coordinates must be provided."
            raise ValueError(msg)

        if tensor_coordinates is not None:
            self._tensor_coordinates = tensor_coordinates
        else:
            assert space_coordinates is not None
            self._tensor_coordinates = (space_coordinates / space_resolution).astype(np.int32)

        self._space_resolution = space_resolution
        self._dimensions = self._tensor_coordinates.shape[0]
        self._topology = topology

    @classmethod
    def uniform(
        cls,
        rng: Generator,
        dimensions: int,
        space_resolution: float,
        topology: Topology,
    ) -> Position:
        return cls(
            tensor_coordinates=rng.integers(0, int(1 / space_resolution), size=(dimensions,), dtype=np.int32),
            space_resolution=space_resolution,
            topology=topology,
        )

    @property
    def tensor_coordinates(self) -> np.ndarray[Any, np.dtype[np.int32]]:
        return self._tensor_coordinates

    @cached_property
    def space_coordinates(self) -> np.ndarray[Any, np.dtype[np.float32]]:
        return self._tensor_coordinates * self._space_resolution

    def clip_norm(self, max_norm: float) -> Position:
        """Return a new Position with its norm clipped to max_norm in space coordinates."""
        norm = np.linalg.norm(self.space_coordinates)
        if norm > max_norm:
            new_space_coordinates = (self.space_coordinates / norm) * max_norm
            new_tensor_coordinates = (new_space_coordinates / self._space_resolution).astype(np.int32)
        else:
            new_tensor_coordinates = self._tensor_coordinates.copy()

        return Position(
            tensor_coordinates=new_tensor_coordinates,
            space_resolution=self._space_resolution,
            topology=self._topology,
        )

    def space_norm(self) -> float:
        """Return the norm of the position in space coordinates."""
        return float(np.linalg.norm(self.space_coordinates))

    def __add__(self, other: Position) -> Position:
        if other._dimensions != self._dimensions:
            error = f"Position has mismatched shape. Expected ({self._dimensions},), got {other._dimensions}"
            raise ValueError(error)

        if other._topology != self._topology or other._space_resolution != self._space_resolution:
            error = "Positions must have the same topology and space resolution."
            raise ValueError(error)

        new_coordinates = self.space_coordinates + other.space_coordinates

        match self._topology:
            case Topology.RECTANGLE:
                new_coordinates = np.clip(new_coordinates, 0, 1 - self._space_resolution)
            case Topology.TORUS:
                new_coordinates = new_coordinates % 1.0

        new_tensor_coordinates = (new_coordinates / self._space_resolution).astype(np.int32)
        return Position(
            tensor_coordinates=new_tensor_coordinates,
            space_resolution=self._space_resolution,
            topology=self._topology,
        )
