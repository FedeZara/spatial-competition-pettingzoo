from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from spatial_competition_pettingzoo.position import Position
from spatial_competition_pettingzoo.topology import Topology

if TYPE_CHECKING:
    from spatial_competition_pettingzoo.buyer import Buyer
    from spatial_competition_pettingzoo.distributions import (
        MultivariateDistributionProtocol,
    )
    from spatial_competition_pettingzoo.seller import Seller


class CompetitionSpace:
    def __init__(
        self,
        dimensions: int,
        topology: Topology,
        space_resolution: int,
        base: Position | None = None,
        extent: Position | None = None,
    ) -> None:
        self._dimensions = dimensions
        self._topology = topology
        self._space_resolution = space_resolution
        self._base = (
            base
            if base is not None
            else Position(
                space_resolution=space_resolution,
                topology=topology,
                tensor_coordinates=np.array([0] * self._dimensions),
            )
        )
        self._extent = (
            extent
            if extent is not None
            else Position(
                space_resolution=space_resolution,
                topology=topology,
                tensor_coordinates=np.array([self._space_resolution - 1] * self._dimensions),
            )
        )

        self._sellers_dict: dict[str, Seller] = {}
        self._buyers: list[Buyer] = []

    @property
    def is_full_space(self) -> bool:
        return bool(
            np.array_equal(self._base.tensor_coordinates, np.array([0] * self._dimensions))
            and np.array_equal(
                self._extent.tensor_coordinates, np.array([self._space_resolution - 1] * self._dimensions)
            )
        )

    @property
    def base(self) -> Position:
        return self._base

    @property
    def extent(self) -> Position:
        return self._extent

    @property
    def relative_extent(self) -> Position:
        return self.relative_position(self._extent)

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def topology(self) -> Topology:
        return self._topology

    @property
    def space_resolution(self) -> int:
        return self._space_resolution

    @property
    def sellers_dict(self) -> dict[str, Seller]:
        return self._sellers_dict

    @property
    def sellers(self) -> list[Seller]:
        return list(self._sellers_dict.values())

    @property
    def buyers(self) -> list[Buyer]:
        return self._buyers

    def add_seller(self, seller: Seller) -> None:
        self._sellers_dict[seller.agent_id] = seller

    def add_buyer(self, buyer: Buyer) -> None:
        self._buyers.append(buyer)

    def remove_buyer(self, buyer: Buyer) -> None:
        self._buyers.remove(buyer)

    def is_position_free_of_sellers(self, position: Position, exclude_agent_id: str | None = None) -> bool:
        """Check if a position is free of sellers.

        Args:
            position: The position to check.
            exclude_agent_id: Optional agent ID to exclude from the check (e.g., the moving seller).

        Returns:
            True if the position is free of sellers, False otherwise.
        """
        for seller in self._sellers_dict.values():
            if exclude_agent_id is not None and seller.agent_id == exclude_agent_id:
                continue
            if seller.position == position:
                return False
        return True

    def is_position_free_of_buyers(self, position: Position) -> bool:
        """Check if a position is free of buyers."""
        for buyer in self._buyers:
            if buyer.position == position:
                return False
        return True

    def _is_in_interval(self, start: int, end: int, coordinate: int) -> bool:
        match self._topology:
            case Topology.RECTANGLE:
                return start <= coordinate <= end
            case Topology.TORUS:
                if start <= end:
                    return start <= coordinate <= end
                return coordinate <= end or coordinate >= start

    def is_in_subspace(self, position: Position) -> bool:
        assert position.dimensions == self._dimensions
        assert position.topology == self._topology
        assert position.space_resolution == self._space_resolution

        return all(
            self._is_in_interval(
                self._base.tensor_coordinates[i], self._extent.tensor_coordinates[i], position.tensor_coordinates[i]
            )
            for i in range(self._dimensions)
        )

    def relative_position(self, position: Position) -> Position:
        return position - self._base

    def is_position_free(self, position: Position) -> bool:
        """
        Check if a position is free in the competition space.

        Args:
            position: The position to check.

        Returns:
            True if the position is free (no seller or buyer), False otherwise.

        """
        return self.is_position_free_of_sellers(position, exclude_agent_id=None) and self.is_position_free_of_buyers(
            position
        )

    @property
    def num_cells(self) -> int:
        """Return the total number of cells in the competition space."""
        return int(self._space_resolution**self._dimensions)

    def _build_seller_positions_set(self) -> set[Position]:
        """Build a set of all seller positions for efficient lookup."""
        return {seller.position for seller in self._sellers_dict.values()}

    def _build_buyer_positions_set(self) -> set[Position]:
        """Build a set of all buyer positions for efficient lookup."""
        return {buyer.position for buyer in self._buyers}

    def _sample_position(
        self,
        distribution: MultivariateDistributionProtocol,
        rng: np.random.Generator,
        occupied_positions: set[Position],
    ) -> Position:
        """
        Sample a position that is not in the occupied positions set.

        Args:
            distribution: The distribution to sample from. It should be a multivariate distribution.
            rng: The random number generator to use.
            occupied_positions: Set of positions that are considered occupied.

        Returns: A position not in the occupied set.

        """
        position = None
        while position is None:
            sample = distribution.rvs(random_state=rng)

            # Convert to tensor coordinates
            tensor_coordinates = (sample * self._space_resolution).round().astype(np.int32)
            tensor_coordinates = np.clip(tensor_coordinates, 0, self._space_resolution - 1)
            position = Position(
                space_resolution=self._space_resolution,
                topology=self._topology,
                tensor_coordinates=tensor_coordinates,
            )

            if position in occupied_positions:
                position = None

        return position

    def sample_position_for_seller(
        self, distribution: MultivariateDistributionProtocol, rng: np.random.Generator
    ) -> Position:
        """
        Sample a free position for a seller from the competition space.

        Sellers cannot be placed on positions occupied by other sellers.

        Args:
            distribution: The distribution to sample from. It should be a multivariate distribution.
            rng: The random number generator to use.

        Returns: A position free of other sellers.

        """
        assert len(self._sellers_dict) < self.num_cells
        occupied = self._build_seller_positions_set()
        return self._sample_position(distribution, rng, occupied)

    def sample_position_for_buyer(
        self, distribution: MultivariateDistributionProtocol, rng: np.random.Generator
    ) -> Position:
        """
        Sample a position for a buyer from the competition space.

        Buyers can be placed on seller positions but not on other buyer positions.

        Args:
            distribution: The distribution to sample from. It should be a multivariate distribution.
            rng: The random number generator to use.

        Returns: A position free for a buyer (may overlap with sellers).

        """
        assert len(self._buyers) < self.num_cells
        occupied = self._build_buyer_positions_set()
        return self._sample_position(distribution, rng, occupied)

    def subspace(self, base: Position, extent: Position) -> CompetitionSpace:
        """Return a subspace of the competition space."""
        assert base.dimensions == self._dimensions
        assert extent.dimensions == self._dimensions
        assert base.topology == self._topology
        assert extent.topology == self._topology
        assert base.space_resolution == self._space_resolution
        assert extent.space_resolution == self._space_resolution
        assert self.is_in_subspace(base)
        assert self.is_in_subspace(extent)

        subspace = CompetitionSpace(
            dimensions=self._dimensions,
            topology=self._topology,
            space_resolution=self._space_resolution,
            base=base,
            extent=extent,
        )

        for seller in self.sellers:
            if subspace.is_in_subspace(seller.position):
                subspace.add_seller(seller)

        for buyer in self._buyers:
            if subspace.is_in_subspace(buyer.position):
                subspace.add_buyer(buyer)

        return subspace
