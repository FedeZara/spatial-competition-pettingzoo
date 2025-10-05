from spatial_competition_pettingzoo.competition_space import CompetitionSpace
from spatial_competition_pettingzoo.environment import InformationLevel, ViewScope
from spatial_competition_pettingzoo.position import Position
from spatial_competition_pettingzoo.seller import Seller
from spatial_competition_pettingzoo.topology import Topology


class Competition:
    def __init__(
        self,
        dimensions: int,
        topology: Topology,
        space_resolution: float,
        sellers: dict[str, Seller],
        max_price: float,
        max_quality: float,
        max_step_size: float,
        information_level: InformationLevel,
        view_scope: ViewScope,
        vision_radius: float,
    ) -> None:
        self.dimensions = dimensions
        self.topology = topology
        self.space = CompetitionSpace(
            dimensions,
            topology,
            space_resolution,
            sellers,
            [],
        )

        self.information_level = information_level
        self.view_scope = view_scope
        self.vision_radius = vision_radius

        self.max_price = max_price
        self.max_quality = max_quality
        self.max_step_size = max_step_size

        self._spawn_new_buyers()

    def agent_step(self, agent: str, movement: Position, price: float, quality: float) -> None:
        """Step the agent."""
        assert 0.0 <= price <= self.max_price
        assert 0.0 <= quality <= self.max_quality
        assert movement.space_norm() <= self.max_step_size

        seller = self.space.sellers[agent]
        seller.move(movement)
        seller.set_price(price)
        seller.set_quality(quality)

    def env_step(self) -> None:
        """Step the environment."""
        self._spawn_new_buyers()

    def compute_rewards(self) -> dict[str, float]:
        """Compute rewards for all agents."""
        return dict.fromkeys(self.space.sellers.keys(), 0.0)

    def _spawn_new_buyers(self) -> None:
        """Spawn buyers in the environment."""
        self.space.add_buyers([])
