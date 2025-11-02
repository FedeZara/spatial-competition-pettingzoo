from __future__ import annotations

from pettingzoo.utils.env import np

from spatial_competition_pettingzoo.buyer import Buyer
from spatial_competition_pettingzoo.position import Position
from spatial_competition_pettingzoo.seller import Seller
from spatial_competition_pettingzoo.topology import Topology


class CompetitionSpace:
    def __init__(
        self,
        dimensions: int,
        topology: Topology,
        space_resolution: int,
        sellers: dict[str, Seller],
        buyers: list[Buyer],
        base: Position | None = None,
        extent: Position | None = None,
    ) -> None:
        self._dimensions = dimensions
        self._topology = topology
        self._space_resolution = space_resolution
        self._sellers = sellers
        self._buyers = buyers
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
    def sellers(self) -> dict[str, Seller]:
        return self._sellers

    @property
    def buyers(self) -> list[Buyer]:
        return self._buyers

    def add_buyers(self, buyers: list[Buyer]) -> None:
        self._buyers.extend(buyers)

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
            sellers={},
            buyers=[],
            base=base,
            extent=extent,
        )

        for agent, seller in self._sellers.items():
            if subspace.is_in_subspace(seller.position):
                new_seller = Seller(
                    idx=seller.idx,
                    position=seller.position,
                    price=seller.price,
                    quality=seller.quality,
                )
                subspace._sellers[agent] = new_seller

        for buyer in self._buyers:
            if subspace.is_in_subspace(buyer.position):
                new_buyer = Buyer(
                    position=buyer.position,
                    value=buyer.value,
                    quality_taste=buyer.quality_taste,
                    distance_factor=buyer.distance_factor,
                )
                subspace._buyers.append(new_buyer)

        return subspace
