from __future__ import annotations

import random
from typing import TYPE_CHECKING

from spatial_competition_pettingzoo.buyer import Buyer
from spatial_competition_pettingzoo.competition_space import CompetitionSpace
from spatial_competition_pettingzoo.observation import Observation
from spatial_competition_pettingzoo.position import Position
from spatial_competition_pettingzoo.seller import Seller
from spatial_competition_pettingzoo.utils import sample_and_clip_univariate_distribution

if TYPE_CHECKING:
    import numpy as np

    from spatial_competition_pettingzoo.distributions import (
        DistributionProtocol,
        MultivariateDistributionProtocol,
    )
    from spatial_competition_pettingzoo.enums import InformationLevel
    from spatial_competition_pettingzoo.topology import Topology
    from spatial_competition_pettingzoo.view_scope import ViewScope


class Competition:
    def __init__(
        self,
        dimensions: int,
        topology: Topology,
        space_resolution: int,
        information_level: InformationLevel,
        view_scope: ViewScope,
        agent_ids: list[str],
        max_price: float,
        max_quality: float,
        max_step_size: float,
        production_cost_factor: float,
        movement_cost: float,
        seller_position_distr: MultivariateDistributionProtocol,
        seller_price_distr: DistributionProtocol,
        seller_quality_distr: DistributionProtocol,
        new_buyers_per_step: int,
        buyer_position_distr: MultivariateDistributionProtocol,
        buyer_valuation_distr: DistributionProtocol,
        buyer_quality_taste_distr: DistributionProtocol,
        buyer_distance_factor_distr: DistributionProtocol,
        rng: np.random.Generator,
    ) -> None:
        self.dimensions = dimensions
        self.topology = topology
        self.space = CompetitionSpace(
            dimensions,
            topology,
            space_resolution,
        )

        self.max_price = max_price
        self.max_quality = max_quality
        self.information_level = information_level
        self.view_scope = view_scope

        self.production_cost_factor = production_cost_factor
        self.movement_cost = movement_cost
        self.max_step_size = max_step_size

        # Buyer generation parameters
        self.new_buyers_per_step = new_buyers_per_step
        self.buyer_position_distr = buyer_position_distr
        self.buyer_valuation_distr = buyer_valuation_distr
        self.buyer_quality_taste_distr = buyer_quality_taste_distr
        self.buyer_distance_factor_distr = buyer_distance_factor_distr

        # Initialize random number generator
        self.rng = rng

        self._spawn_sellers(agent_ids, seller_position_distr, seller_price_distr, seller_quality_distr)
        self._spawn_new_buyers()

    def agent_step(
        self,
        agent_id: str,
        movement: Position | None = None,
        price: float | None = None,
        quality: float | None = None,
    ) -> None:
        """Step the agent. None values mean keep current value."""
        assert movement is None or movement.space_norm() <= self.max_step_size + Position.SPACE_COORDINATES_TOLERANCE
        assert price is None or 0.0 <= price <= self.max_price
        assert quality is None or 0.0 <= quality <= self.max_quality

        seller = self.space.sellers_dict[agent_id]

        if movement is not None:
            seller.move(movement)
        if price is not None:
            seller.set_price(price)
        if quality is not None:
            seller.set_quality(quality)

    def env_step(self) -> None:
        """Step the environment."""
        self._process_sales()
        self._spawn_new_buyers()

    def compute_agent_reward(self, agent_id: str) -> float:
        """Compute rewards for all agents."""
        agent = self.space.sellers_dict[agent_id]
        return agent.step_reward(self.production_cost_factor, self.movement_cost)

    def get_agent_observation(self, agent_id: str) -> Observation:
        """Get observation for the specified agent."""
        return Observation.build_from_competition_space(
            space=self.space,
            information_level=self.information_level,
            view_scope=self.view_scope,
            agent_id=agent_id,
        )

    def _process_sales(self) -> None:
        """Process sales."""
        for seller in self.space.sellers:
            seller.reset_running_sales()
        buyers = self.space.buyers.copy()
        random.shuffle(buyers)
        for buyer in buyers:
            if buyer.choose_seller_and_buy(self.space.sellers):
                self.space.remove_buyer(buyer)

    def _spawn_sellers(
        self,
        agent_ids: list[str],
        seller_position_distr: MultivariateDistributionProtocol,
        seller_price_distr: DistributionProtocol,
        seller_quality_distr: DistributionProtocol,
    ) -> None:
        """Spawn sellers."""
        for agent_id in agent_ids:
            if self.space.num_free_cells == 0:
                error_msg = "No free cells available to spawn sellers."
                raise ValueError(error_msg)

            position = self.space.sample_free_position(seller_position_distr, self.rng)
            price = sample_and_clip_univariate_distribution(
                "price", seller_price_distr, self.rng, min_value=0.0, max_value=self.max_price
            )
            quality = sample_and_clip_univariate_distribution(
                "quality", seller_quality_distr, self.rng, min_value=0.0, max_value=self.max_quality
            )
            self.space.add_seller(Seller(agent_id=agent_id, position=position, price=price, quality=quality))

    def _spawn_new_buyers(self) -> None:
        """Spawn new buyers."""
        for _ in range(self.new_buyers_per_step):
            if self.space.num_free_cells == 0:
                break

            position = self.space.sample_free_position(self.buyer_position_distr, self.rng)
            value = sample_and_clip_univariate_distribution("value", self.buyer_valuation_distr, self.rng)
            quality_taste = sample_and_clip_univariate_distribution(
                "quality_taste", self.buyer_quality_taste_distr, self.rng
            )
            distance_factor = sample_and_clip_univariate_distribution(
                "distance_factor", self.buyer_distance_factor_distr, self.rng
            )

            self.space.add_buyer(
                Buyer(
                    position=position,
                    value=value,
                    quality_taste=quality_taste,
                    distance_factor=distance_factor,
                )
            )
