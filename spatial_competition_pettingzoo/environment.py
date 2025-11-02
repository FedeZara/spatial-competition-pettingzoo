"""
Spatial Competition PettingZoo Environment.

A multi-agent environment where sellers compete in a spatial market with horizontal
(location) and vertical (quality) differentiation. Buyers make purchasing decisions
based on utility maximization.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Literal

import numpy as np
from gymnasium import spaces
from pettingzoo import AECEnv
from pettingzoo.utils import AgentSelector, wrappers

from spatial_competition_pettingzoo.competition import Competition
from spatial_competition_pettingzoo.distributions import (
    ConstantUnivariateDistribution,
    MultivariateUniformDistribution,
)
from spatial_competition_pettingzoo.enums import InformationLevel
from spatial_competition_pettingzoo.observation import Observation
from spatial_competition_pettingzoo.position import Position
from spatial_competition_pettingzoo.topology import Topology
from spatial_competition_pettingzoo.view_scope import (
    CompleteViewScope,
    LimitedViewScope,
    ViewScope,
)

if TYPE_CHECKING:
    from spatial_competition_pettingzoo.distributions import (
        DistributionProtocol,
        MultivariateDistributionProtocol,
    )


def env(
    *,
    dimensions: int = 2,
    topology: Topology = Topology.RECTANGLE,
    space_resolution: int = 100,
    information_level: InformationLevel = InformationLevel.COMPLETE,
    view_scope: Literal["limited", "complete"] = "complete",
    vision_radius: int = 10,
    num_sellers: int = 3,
    max_price: float = 10.0,
    max_quality: float = 5.0,
    max_step_size: float = 0.1,
    production_cost_factor: float = 0.5,
    movement_cost: float = 0.1,
    seller_position_distr: MultivariateDistributionProtocol | None = None,
    seller_price_distr: DistributionProtocol | None = None,
    seller_quality_distr: DistributionProtocol | None = None,
    new_buyers_per_step: int = 50,
    buyer_position_distr: MultivariateDistributionProtocol | None = None,
    buyer_valuation_distr: DistributionProtocol | None = None,
    buyer_quality_taste_distr: DistributionProtocol | None = None,
    buyer_distance_factor_distr: DistributionProtocol | None = None,
    max_steps: int = 100,
    render_mode: str | None = None,
) -> wrappers.OrderEnforcingWrapper:
    """Create a new spatial competition environment."""
    return wrappers.OrderEnforcingWrapper(
        wrappers.AssertOutOfBoundsWrapper(
            raw_env(
                dimensions=dimensions,
                topology=topology,
                space_resolution=space_resolution,
                information_level=information_level,
                view_scope=view_scope,
                vision_radius=vision_radius,
                num_sellers=num_sellers,
                max_price=max_price,
                max_quality=max_quality,
                max_step_size=max_step_size,
                production_cost_factor=production_cost_factor,
                movement_cost=movement_cost,
                seller_position_distr=seller_position_distr,
                seller_price_distr=seller_price_distr,
                seller_quality_distr=seller_quality_distr,
                new_buyers_per_step=new_buyers_per_step,
                buyer_position_distr=buyer_position_distr,
                buyer_valuation_distr=buyer_valuation_distr,
                buyer_quality_taste_distr=buyer_quality_taste_distr,
                buyer_distance_factor_distr=buyer_distance_factor_distr,
                max_steps=max_steps,
                render_mode=render_mode,
            )
        )
    )


def raw_env(
    *,
    dimensions: int = 2,
    topology: Topology = Topology.RECTANGLE,
    space_resolution: int = 100,
    information_level: InformationLevel = InformationLevel.COMPLETE,
    view_scope: Literal["limited", "complete"] = "complete",
    vision_radius: int = 10,
    num_sellers: int = 3,
    max_price: float = 10.0,
    max_quality: float = 5.0,
    max_step_size: float = 0.1,
    production_cost_factor: float = 0.5,
    movement_cost: float = 0.1,
    seller_position_distr: MultivariateDistributionProtocol | None = None,
    seller_price_distr: DistributionProtocol | None = None,
    seller_quality_distr: DistributionProtocol | None = None,
    new_buyers_per_step: int = 50,
    buyer_position_distr: MultivariateDistributionProtocol | None = None,
    buyer_valuation_distr: DistributionProtocol | None = None,
    buyer_quality_taste_distr: DistributionProtocol | None = None,
    buyer_distance_factor_distr: DistributionProtocol | None = None,
    max_steps: int = 100,
    render_mode: str | None = None,
) -> SpatialCompetitionEnv:
    """Create a raw spatial competition environment."""
    return SpatialCompetitionEnv(
        dimensions=dimensions,
        topology=topology,
        space_resolution=space_resolution,
        information_level=information_level,
        view_scope=view_scope,
        vision_radius=vision_radius,
        num_sellers=num_sellers,
        max_price=max_price,
        max_quality=max_quality,
        max_step_size=max_step_size,
        production_cost_factor=production_cost_factor,
        movement_cost=movement_cost,
        seller_position_distr=seller_position_distr,
        seller_price_distr=seller_price_distr,
        seller_quality_distr=seller_quality_distr,
        new_buyers_per_step=new_buyers_per_step,
        buyer_position_distr=buyer_position_distr,
        buyer_valuation_distr=buyer_valuation_distr,
        buyer_quality_taste_distr=buyer_quality_taste_distr,
        buyer_distance_factor_distr=buyer_distance_factor_distr,
        max_steps=max_steps,
        render_mode=render_mode,
    )


class SpatialCompetitionEnv(AECEnv):
    """
    Spatial Competition Environment using PettingZoo AEC API.

    Sellers compete in a spatial market where buyers make purchasing decisions
    based on price, quality, distance, and personal preferences.
    """

    metadata: ClassVar[dict[str, Any]] = {
        "render_modes": ["human", "rgb_array"],
        "name": "spatial_competition_v0",
        "is_parallelizable": True,
    }

    def __init__(
        self,
        *,
        dimensions: int = 2,
        topology: Topology = Topology.RECTANGLE,
        space_resolution: int = 100,
        information_level: InformationLevel = InformationLevel.COMPLETE,
        view_scope: Literal["limited", "complete"] = "complete",
        vision_radius: int = 10,
        num_sellers: int = 3,
        max_price: float = 10.0,
        max_quality: float = 5.0,
        max_step_size: float = 0.1,
        production_cost_factor: float = 0.5,
        movement_cost: float = 0.1,
        seller_position_distr: MultivariateDistributionProtocol | None = None,
        seller_price_distr: DistributionProtocol | None = None,
        seller_quality_distr: DistributionProtocol | None = None,
        new_buyers_per_step: int = 50,
        buyer_position_distr: MultivariateDistributionProtocol | None = None,
        buyer_valuation_distr: DistributionProtocol | None = None,
        buyer_quality_taste_distr: DistributionProtocol | None = None,
        buyer_distance_factor_distr: DistributionProtocol | None = None,
        max_steps: int = 10000,
        render_mode: str | None = None,
    ) -> None:
        """
        Initialize the spatial competition environment.

        Args:
            dimensions: Number of spatial dimensions (N)
            topology: Map topology ("rectangle" or "torus")
            space_resolution: Grid resolution for discretized continuous space
            information_level: Information mode controlling what sellers can observe
            view_scope: View scope controlling spatial observation range
            vision_radius: Radius of vision for limited view modes in number of grid cells
            num_sellers: Number of competing sellers
            max_price: Maximum price sellers can set (P)
            max_quality: Maximum quality level (Q)
            max_step_size: Maximum movement distance per step
            production_cost_factor: Production cost parameter (gamma, C(q) = gamma*q^2)
            movement_cost: Cost of moving per unit distance (m)
            seller_position_distr: Multivariate distribution for seller positions.
                Defaults to multivariate uniform distribution in the unit hypercube.
            seller_price_distr: Distribution for seller prices.
                Defaults to constant max_price/2.
            seller_quality_distr: Distribution for seller quality.
                Defaults to constant max_quality/2.
            new_buyers_per_step: Number of new buyers to spawn each environment step
            buyer_position_distr: Multivariate distribution for buyer positions.
                Defaults to multivariate uniform distribution in the unit hypercube.
            buyer_valuation_distr: Distribution for buyer valuations.
                Defaults to constant +inf.
            buyer_quality_taste_distr: Distribution for buyer quality taste (k_b).
                Defaults to constant 1.
            buyer_distance_factor_distr: Distribution for buyer distance factor.
                Defaults to constant 1.
            max_steps: Maximum episode length
            render_mode: Rendering mode

        """
        super().__init__()

        # Space configuration
        self.dimensions = dimensions
        self.topology = topology

        # Market parameters
        self.num_sellers = num_sellers
        self.max_price = max_price
        self.max_quality = max_quality
        self.production_cost_factor = production_cost_factor
        self.movement_cost = movement_cost
        self.new_buyers_per_step = new_buyers_per_step

        # Seller distribution parameters
        self.seller_position_distr = seller_position_distr or MultivariateUniformDistribution(
            dim=dimensions, loc=0.0, scale=1 - 1 / space_resolution
        )
        self.seller_price_distr = seller_price_distr or ConstantUnivariateDistribution(max_price / 2)
        self.seller_quality_distr = seller_quality_distr or ConstantUnivariateDistribution(max_quality / 2)

        # Buyer distribution parameters
        self.buyer_position_distr = buyer_position_distr or MultivariateUniformDistribution(
            dim=dimensions, loc=0.0, scale=1 - 1 / space_resolution
        )
        self.buyer_valuation_distr = buyer_valuation_distr or ConstantUnivariateDistribution(np.inf)
        self.buyer_quality_taste_distr = buyer_quality_taste_distr or ConstantUnivariateDistribution(1)
        self.buyer_distance_factor_distr = buyer_distance_factor_distr or ConstantUnivariateDistribution(1)
        self.max_step_size = max_step_size
        self.space_resolution = space_resolution

        # Environment parameters
        self.max_steps = max_steps
        self.render_mode = render_mode

        # Information level parameters
        self.information_level = information_level
        match view_scope:
            case "limited":
                match topology:
                    case Topology.RECTANGLE:
                        if vision_radius >= space_resolution:
                            error_msg = f"Vision radius ({vision_radius}) must be less than space resolution ({space_resolution}) for rectangle topology"
                            raise ValueError(error_msg)
                    case Topology.TORUS:
                        if vision_radius >= space_resolution / 2:
                            error_msg = f"Vision radius ({vision_radius}) must be less than half of space resolution ({space_resolution}) for torus topology"
                            raise ValueError(error_msg)
                self.view_scope: ViewScope = LimitedViewScope(vision_radius)
            case "complete":
                self.view_scope = CompleteViewScope()

        # Initialize agents
        self.possible_agents = [f"seller_{i}" for i in range(num_sellers)]

        # Define action and observation spaces
        self._setup_action_space()
        self._setup_observation_space()

        self.rng = np.random.default_rng()

        self.competition: Competition | None = None
        self.reset()

    def _setup_action_space(self) -> None:
        """
        Set up action space.

        Action space contains:
        - movement: N-dimensional vector with norm <= max_step_size
        - price: scalar in [0, max_price]
        - quality: scalar in [0, max_quality]
        """
        self._action_spaces = {
            agent: spaces.Dict(
                {
                    "movement": spaces.Box(
                        low=-self.max_step_size, high=self.max_step_size, shape=(self.dimensions,), dtype=np.float32
                    ),
                    "price": spaces.Box(low=0.0, high=self.max_price, dtype=np.float32),
                    "quality": spaces.Box(low=0.0, high=self.max_quality, dtype=np.float32),
                }
            )
            for agent in self.possible_agents
        }

    def _setup_observation_space(self) -> None:
        """
        Set up observation space.

        It contains the following components:
        - Own state: [position (N dims), price, quality]
        - A layer containing an N dimensional matrix with information about presence of other sellers or buyers,
         restricted by view scope.
        - If partial view mode, a layer containing an N dimensional matrix with information about sellers.
        - If complete view mode, a layer containing an N dimensional matrix with information about buyers.
        """
        observation_space = Observation.create_observation_space(
            information_level=self.information_level,
            view_scope=self.view_scope,
            dimensions=self.dimensions,
            space_resolution=self.space_resolution,
            max_price=self.max_price,
            max_quality=self.max_quality,
        )

        self._observation_spaces = dict.fromkeys(self.possible_agents, observation_space)

    def observation_space(self, agent: str) -> spaces.Space:
        """Get observation space for agent."""
        return self._observation_spaces[agent]

    def action_space(self, agent: str) -> spaces.Space:
        """Get action space for agent."""
        return self._action_spaces[agent]

    def reset(self, seed: int | None = None) -> None:
        """Reset the environment to initial state."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.agents = self.possible_agents[:]
        self.num_steps = 0

        self.competition = Competition(
            dimensions=self.dimensions,
            topology=self.topology,
            space_resolution=self.space_resolution,
            information_level=self.information_level,
            view_scope=self.view_scope,
            agent_ids=self.agents,
            max_price=self.max_price,
            max_quality=self.max_quality,
            max_step_size=self.max_step_size,
            production_cost_factor=self.production_cost_factor,
            movement_cost=self.movement_cost,
            seller_position_distr=self.seller_position_distr,
            seller_price_distr=self.seller_price_distr,
            seller_quality_distr=self.seller_quality_distr,
            new_buyers_per_step=self.new_buyers_per_step,
            buyer_position_distr=self.buyer_position_distr,
            buyer_valuation_distr=self.buyer_valuation_distr,
            buyer_quality_taste_distr=self.buyer_quality_taste_distr,
            buyer_distance_factor_distr=self.buyer_distance_factor_distr,
            rng=self.rng,
        )

        # Initialize rewards, terminations, truncations, infos
        self.rewards = dict.fromkeys(self.agents, 0.0)
        self.terminations = dict.fromkeys(self.agents, False)
        self.truncations = dict.fromkeys(self.agents, False)

        # Initialize observations
        self.observations: dict[str, dict[str, Any]] = {
            agent: self.competition.get_agent_observation(agent).get_observation() for agent in self.agents
        }

        # Initialize agent selector for turn-based play
        self._agent_selector = AgentSelector(self.rng.permutation(self.agents))
        self.agent_selection = self._agent_selector.reset()

    def observe(self, agent: str) -> dict[str, Any]:
        """Get observation for the specified agent."""
        return self.observations[agent]

    def step(self, action: np.ndarray) -> None:
        """Execute one step of the environment."""
        assert self.competition is not None

        if self.terminations[self.agent_selection] or self.truncations[self.agent_selection]:
            self.agents.remove(self.agent_selection)
            self._next_agent()
            return

        agent = self.agent_selection

        # Parse action: [movement, price, quality]
        movement = Position(
            tensor_coordinates=np.array(action["movement"], dtype=np.float32),
            space_resolution=self.space_resolution,
            topology=self.topology,
        )
        new_price = float(action["price"])
        new_quality = float(action["quality"])

        # Clip movement, clip price and quality to valid ranges
        movement = movement.clip_norm(self.max_step_size)
        new_price = np.clip(new_price, 0.0, self.max_price)
        new_quality = np.clip(new_quality, 0.0, self.max_quality)

        self.competition.agent_step(agent, movement, new_price, new_quality)

        # Update observations
        self.observations[agent] = self.competition.get_agent_observation(agent).get_observation()

        # Check termination conditions
        self.num_steps += 1
        if self.num_steps >= self.max_steps:
            self.truncations = dict.fromkeys(self.agents, True)

        # Process market interactions if all agents have acted
        if self._agent_selector.is_last():
            self.competition.env_step()
            self.rewards = {agent: self.competition.compute_agent_reward(agent) for agent in self.agents}
            self._agent_selector.reinit(self.rng.permutation(self.agents))

        self._next_agent()

    def _next_agent(self) -> None:
        """Advance to the next agent in the turn order."""
        self.agent_selection = next(self._agent_selector)

    def render(self) -> None:
        """Render the environment (placeholder implementation)."""

    def close(self) -> None:
        """Close the environment."""
