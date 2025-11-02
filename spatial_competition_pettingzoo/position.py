from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.random import Generator

from spatial_competition_pettingzoo.topology import Topology


class Position:
    SPACE_COORDINATES_TOLERANCE = 1e-6

    def __init__(
        self,
        space_resolution: int,
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
            self._tensor_coordinates = (space_coordinates * space_resolution).round().astype(np.int32)

            if any(
                abs(self._tensor_coordinates - (space_coordinates * space_resolution))
                > self.SPACE_COORDINATES_TOLERANCE
            ):
                msg = f"Space coordinates must be multiples of 1/space_resolution ({1.0 / space_resolution})"
                raise ValueError(msg)

        # Validate that tensor_coordinates are in the valid range [0, space_resolution)
        if np.any(self._tensor_coordinates < 0) or np.any(self._tensor_coordinates >= space_resolution):
            msg = f"Tensor coordinates must be in the range [0, {space_resolution}). Got: {self._tensor_coordinates}"
            raise ValueError(msg)

        self._space_resolution = space_resolution
        self._dimensions = self._tensor_coordinates.shape[0]
        self._topology = topology

    @classmethod
    def uniform(
        cls,
        rng: Generator,
        dimensions: int,
        space_resolution: int,
        topology: Topology,
    ) -> Position:
        return cls(
            tensor_coordinates=rng.integers(0, space_resolution, size=(dimensions,), dtype=np.int32),
            space_resolution=space_resolution,
            topology=topology,
        )

    @property
    def tensor_coordinates(self) -> np.ndarray[Any, np.dtype[np.int32]]:
        return self._tensor_coordinates

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def topology(self) -> Topology:
        return self._topology

    @property
    def space_resolution(self) -> int:
        return self._space_resolution

    @cached_property
    def space_coordinates(self) -> np.ndarray[Any, np.dtype[np.float32]]:
        return self._tensor_coordinates.astype(np.float32) / self._space_resolution

    def clip_norm(self, max_norm: float) -> Position:
        """Return a new Position with its norm clipped to max_norm in space coordinates."""
        max_norm_tensor = max_norm * self._space_resolution

        norm = np.linalg.norm(self.tensor_coordinates)
        if norm > max_norm_tensor:
            new_tensor_coordinates = ((self.tensor_coordinates / norm) * max_norm_tensor).astype(np.int32)
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

        new_coordinates = self.tensor_coordinates + other.tensor_coordinates

        match self._topology:
            case Topology.RECTANGLE:
                new_coordinates = np.clip(new_coordinates, 0, self._space_resolution - 1)
            case Topology.TORUS:
                new_coordinates = new_coordinates % self._space_resolution

        return Position(
            tensor_coordinates=new_coordinates,
            space_resolution=self._space_resolution,
            topology=self._topology,
        )

    def __sub__(self, other: Position) -> Position:
        if other._dimensions != self._dimensions:
            error = f"Position has mismatched shape. Expected ({self._dimensions},), got {other._dimensions}"
            raise ValueError(error)

        if other._topology != self._topology or other._space_resolution != self._space_resolution:
            error = "Positions must have the same topology and space resolution."
            raise ValueError(error)

        new_coordinates = self.tensor_coordinates - other.tensor_coordinates

        match self._topology:
            case Topology.RECTANGLE:
                new_coordinates = np.clip(new_coordinates, 0, self._space_resolution - 1)
            case Topology.TORUS:
                new_coordinates = (new_coordinates + self._space_resolution) % self._space_resolution

        return Position(
            tensor_coordinates=new_coordinates,
            space_resolution=self._space_resolution,
            topology=self._topology,
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Position):
            return False

        if (
            self.dimensions != other.dimensions
            or self.topology != other.topology
            or self.space_resolution != other.space_resolution
        ):
            return False
        return np.array_equal(self.tensor_coordinates, other.tensor_coordinates)

    def __hash__(self) -> int:
        return hash((tuple(self.tensor_coordinates), self.space_resolution, self.topology))

    def distance(self, other: Position) -> float:
        assert self.dimensions == other.dimensions
        assert self.topology == other.topology
        assert self.space_resolution == other.space_resolution

        match self._topology:
            case Topology.RECTANGLE:
                return float(np.linalg.norm(self.space_coordinates - other.space_coordinates))
            case Topology.TORUS:
                return float(
                    np.linalg.norm(
                        np.minimum(
                            np.abs(self.space_coordinates - other.space_coordinates),
                            1 - np.abs(self.space_coordinates - other.space_coordinates),
                        )
                    )
                )
