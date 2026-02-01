from __future__ import annotations

import random
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from spatial_competition_pettingzoo.action import Action
from spatial_competition_pettingzoo.buyer import Buyer
from spatial_competition_pettingzoo.competition_space import CompetitionSpace
from spatial_competition_pettingzoo.enums import TransportationCostNorm
from spatial_competition_pettingzoo.observation import Observation
from spatial_competition_pettingzoo.position import Position
from spatial_competition_pettingzoo.seller import Seller
from spatial_competition_pettingzoo.topology import Topology
from spatial_competition_pettingzoo.utils import sample_and_clip_univariate_distribution

if TYPE_CHECKING:
    from spatial_competition_pettingzoo.distributions import (
        DistributionProtocol,
        MultivariateDistributionProtocol,
    )
    from spatial_competition_pettingzoo.enums import InformationLevel
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
        include_quality: bool,
        max_quality: float,
        max_step_size: float,
        production_cost_factor: float,
        movement_cost: float,
        transportation_cost_norm: TransportationCostNorm,
        seller_position_distr: MultivariateDistributionProtocol,
        seller_price_distr: DistributionProtocol,
        seller_quality_distr: DistributionProtocol | None,
        new_buyers_per_step: int,
        max_buyers: int,
        buyer_position_distr: MultivariateDistributionProtocol,
        include_buyer_valuation: bool,
        buyer_valuation_distr: DistributionProtocol | None,
        buyer_quality_taste_distr: DistributionProtocol | None,
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
        self.include_quality = include_quality
        self.include_buyer_valuation = include_buyer_valuation
        self.information_level = information_level
        self.view_scope = view_scope

        self.production_cost_factor = production_cost_factor
        self.movement_cost = movement_cost
        self.transportation_cost_norm = transportation_cost_norm
        self.max_step_size = max_step_size

        # Buyer generation parameters
        self.new_buyers_per_step = new_buyers_per_step
        self.max_buyers = max_buyers
        self.buyer_position_distr = buyer_position_distr
        self.buyer_valuation_distr = buyer_valuation_distr
        self.buyer_quality_taste_distr = buyer_quality_taste_distr
        self.buyer_distance_factor_distr = buyer_distance_factor_distr

        # Initialize random number generator
        self.rng = rng

        self._spawn_sellers(agent_ids, seller_position_distr, seller_price_distr, seller_quality_distr)
        self._new_cycle = True

        # Optional render callback for visualization during step phases
        self.render_callback: Callable[[], None] | None = None

    def _agent_step(
        self,
        agent_id: str,
        occupied_positions: set[Position],
        movement: Position | None = None,
        price: float | None = None,
        quality: float | None = None,
    ) -> None:
        """Step the agent. None values mean keep current value.

        If movement results in a collision with another seller, small variations
        of the movement are tried until a free position is found.

        Updates occupied_positions with the seller's final position.

        Args:
            agent_id: The agent ID to step.
            occupied_positions: Set of positions occupied by already-processed sellers.
                Will be updated with this seller's final position.
            movement: The movement to apply.
            price: The new price to set.
            quality: The new quality to set.
        """
        assert movement is None or movement.space_norm() <= self.max_step_size + Position.SPACE_COORDINATES_TOLERANCE
        assert price is None or 0.0 <= price <= self.max_price
        assert quality is None or 0.0 <= quality <= self.max_quality

        seller = self.space.sellers_dict[agent_id]

        if movement is not None and movement.space_norm() > 0:
            seller.move(movement)
            self._fix_collisions(seller, occupied_positions)
        if price is not None:
            seller.set_price(price)
        if self.include_quality and quality is not None:
            seller.set_quality(quality)

        occupied_positions.add(seller.position)

    def _fix_collisions(
        self,
        seller: Seller,
        occupied_positions: set[Position],
        max_attempts: int = 50,
    ) -> None:
        """Fix collisions by trying different offsets.

        If the seller's current position is occupied, tries random offsets
        until a free position is found, then updates the seller's position.
        """
        original_position = seller.position
        attempt = 0

        while seller.position in occupied_positions:
            # Generate a random offset, expanding search radius on later attempts
            # Start with ±1, expand to ±2 after 10 attempts, ±3 after 20, etc.
            radius = 1 + attempt // 10
            offset = self.rng.integers(-radius, radius + 1, size=(self.dimensions,), dtype=np.int32)
            seller.position = original_position + Position(
                space_resolution=self.space.space_resolution,
                topology=self.topology,
                tensor_coordinates=offset,
            )
            attempt += 1

            if attempt >= max_attempts:
                raise ValueError(
                    "Max attempts reached while fixing collisions, reduce the number of sellers or increase the space_resolution."
                )

    def step(
        self,
        actions: dict[str, Action],
    ) -> None:
        """
        Step the competition:
        - Remove buyers who made a purchase in the previous step.
        - Spawn new buyers.
        - Step the agents by applying their actions (movement, price, quality) and resolving collisions.
        - Process sales.

        Args:
            actions: Dictionary mapping agent IDs to their actions.
        """
        self._remove_buyers_who_purchased()
        if self.render_callback:
            self.render_callback()

        self._spawn_new_buyers()
        if self.render_callback:
            self.render_callback()

        # Separate agents into stationary (no movement) and moving
        stationary_agents = []
        moving_agents = []
        for agent_id, action in actions.items():
            if action.movement is None or action.movement.space_norm() == 0:
                stationary_agents.append(agent_id)
            else:
                moving_agents.append(agent_id)

        # Track occupied positions of already-processed sellers (for collision detection)
        occupied_positions: set[Position] = set()

        # Process stationary agents first (they establish positions without collisions)
        for agent_id in stationary_agents:
            action = actions[agent_id]
            self._agent_step(agent_id, occupied_positions, action.movement, action.price, action.quality)

        # Then process moving agents in random order for fair collision resolution
        self.rng.shuffle(moving_agents)
        for agent_id in moving_agents:
            action = actions[agent_id]
            self._agent_step(agent_id, occupied_positions, action.movement, action.price, action.quality)

        if self.render_callback:
            self.render_callback()

        self._process_sales()

        if self.render_callback:
            self.render_callback()

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

    def _remove_buyers_who_purchased(self) -> None:
        """Remove buyers who made a purchase in the previous step."""
        buyers_to_remove = [buyer for buyer in self.space.buyers if buyer.has_purchased]
        for buyer in buyers_to_remove:
            self.space.remove_buyer(buyer)

    def _process_sales(self) -> None:
        """Process sales - buyers choose sellers and make purchases."""
        for seller in self.space.sellers:
            seller.reset_running_sales()
        buyers = self.space.buyers.copy()
        random.shuffle(buyers)
        for buyer in buyers:
            buyer.choose_seller_and_buy(self.space.sellers)

    def _spawn_sellers(
        self,
        agent_ids: list[str],
        seller_position_distr: MultivariateDistributionProtocol,
        seller_price_distr: DistributionProtocol,
        seller_quality_distr: DistributionProtocol | None,
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
            quality = None
            if self.include_quality:
                assert seller_quality_distr is not None
                quality = sample_and_clip_univariate_distribution(
                    "quality", seller_quality_distr, self.rng, min_value=0.0, max_value=self.max_quality
                )
            self.space.add_seller(Seller(agent_id=agent_id, position=position, price=price, quality=quality))

    def _spawn_new_buyers(self) -> None:
        """Spawn new buyers."""
        for _ in range(self.new_buyers_per_step):
            if self.space.num_free_cells == 0:
                break

            if len(self.space.buyers) >= self.max_buyers:
                break

            position = self.space.sample_free_position(self.buyer_position_distr, self.rng)
            distance_factor = sample_and_clip_univariate_distribution(
                "distance_factor", self.buyer_distance_factor_distr, self.rng
            )

            value = None
            if self.buyer_valuation_distr is not None:
                value = sample_and_clip_univariate_distribution("value", self.buyer_valuation_distr, self.rng)

            quality_taste = None
            if self.buyer_quality_taste_distr is not None:
                quality_taste = sample_and_clip_univariate_distribution(
                    "quality_taste", self.buyer_quality_taste_distr, self.rng
                )

            self.space.add_buyer(
                Buyer(
                    position=position,
                    value=value,
                    quality_taste=quality_taste,
                    distance_factor=distance_factor,
                    transportation_cost_norm=self.transportation_cost_norm,
                )
            )
