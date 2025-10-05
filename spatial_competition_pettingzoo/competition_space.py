from __future__ import annotations

from pettingzoo.utils.env import np

from spatial_competition_pettingzoo import topology
from spatial_competition_pettingzoo.buyer import Buyer
from spatial_competition_pettingzoo.position import Position
from spatial_competition_pettingzoo.seller import Seller
from spatial_competition_pettingzoo.topology import Topology


class CompetitionSpace:
    def __init__(
        self,
        dimensions: int,
        topology: Topology,
        space_resolution: float,
        sellers: dict[str, Seller],
        buyers: list[Buyer],
    ) -> None:
        self.dimensions = dimensions
        self.topology = topology
        self.space_resolution = space_resolution
        self.sellers = sellers
        self.buyers = buyers

    def add_buyers(self, buyers: list[Buyer]) -> None:
        self.buyers.extend(buyers)

    def subspace(self, subintervals: list[tuple[float, float]]) -> CompetitionSpace:
        """Return a subspace of the competition space."""
        assert len(subintervals) == self.dimensions

        for interval in subintervals:
            assert interval[0] >= 0.0
            assert interval[1] <= 1.0
            assert topology == Topology.TORUS or interval[0] < interval[1]

        displacement = Position(
            tensor_coordinates=np.array([interval[0] for interval in subintervals]),
            space_resolution=self.space_resolution,
            topology=self.topology,
        )

        sellers: dict[str, Seller] = {}
        for agent, seller in self.sellers.items():
            if self._is_in_subspace(seller.position, subintervals):
                new_seller = Seller(
                    idx=seller.idx,
                    position=seller.position + displacement,
                    price=seller.price,
                    quality=seller.quality,
                )
                sellers[agent] = new_seller

        buyers: list[Buyer] = []
        for buyer in self.buyers:
            if all(
                subintervals[i][0] <= buyer.position.space_coordinates[i] < subintervals[i][1]
                for i in range(self.dimensions)
            ):
                new_buyer = Buyer(
                    position=buyer.position + displacement,
                    value=buyer.value,
                    quality_taste=buyer.quality_taste,
                    distance_factor=buyer.distance_factor,
                )
                buyers.append(new_buyer)

        return CompetitionSpace(
            dimensions=self.dimensions,
            topology=self.topology,
            space_resolution=self.space_resolution,
            sellers=self.sellers,
            buyers=self.buyers,
        )

    def _is_in_subspace(self, position: Position, subintervals: list[tuple[float, float]]) -> bool:
        return all(
            subintervals[i][0] <= position.space_coordinates[i] < subintervals[i][1] for i in range(self.dimensions)
        )
