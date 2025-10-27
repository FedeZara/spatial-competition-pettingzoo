"""
Spatial Competition PettingZoo Environment.

A multi-agent environment where sellers compete in a spatial market with horizontal
(location) and vertical (quality) differentiation. Buyers make purchasing decisions
based on utility maximization.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from gymnasium import spaces
from pettingzoo import AECEnv
from pettingzoo.utils import AgentSelector, wrappers
from scipy.stats import uniform

from spatial_competition_pettingzoo.competition import Competition
from spatial_competition_pettingzoo.enums import InformationLevel, ViewScope
from spatial_competition_pettingzoo.observation import Observation
from spatial_competition_pettingzoo.position import Position
from spatial_competition_pettingzoo.seller import Seller
from spatial_competition_pettingzoo.topology import Topology

if TYPE_CHECKING:
    from scipy.stats._distn_infrastructure import rv_continuous_frozen


def env(
    *,
    dimensions: int = 2,
    topology: Topology = Topology.RECTANGLE,
    num_sellers: int = 3,
    buyer_rate: float = 0.1,
    max_price: float = 10.0,
    max_quality: float = 5.0,
    max_valuation: float = 10.0,
    production_cost_factor: float = 0.5,
    movement_cost: float = 0.1,
    quality_taste_distr: rv_continuous_frozen | None = None,
    max_step_size: float = 0.1,
    space_resolution: int = 100,
    max_steps: int = 100,
    information_level: InformationLevel = InformationLevel.COMPLETE,
    view_scope: ViewScope = ViewScope.COMPLETE,
    vision_radius: int = 10,
    render_mode: str | None = None,
) -> wrappers.OrderEnforcingWrapper:
    """Create a new spatial competition environment."""
    return wrappers.OrderEnforcingWrapper(
        wrappers.AssertOutOfBoundsWrapper(
            raw_env(
                dimensions=dimensions,
                topology=topology,
                num_sellers=num_sellers,
                buyer_rate=buyer_rate,
                max_price=max_price,
                max_quality=max_quality,
                max_valuation=max_valuation,
                production_cost_factor=production_cost_factor,
                movement_cost=movement_cost,
                quality_taste_distr=quality_taste_distr,
                max_step_size=max_step_size,
                space_resolution=space_resolution,
                max_steps=max_steps,
                information_level=information_level,
                view_scope=view_scope,
                vision_radius=vision_radius,
                render_mode=render_mode,
            )
        )
    )


def raw_env(
    *,
    dimensions: int = 2,
    topology: Topology = Topology.RECTANGLE,
    num_sellers: int = 3,
    buyer_rate: float = 0.1,
    max_price: float = 10.0,
    max_quality: float = 5.0,
    max_valuation: float = 10.0,
    production_cost_factor: float = 0.5,
    movement_cost: float = 0.1,
    quality_taste_distr: rv_continuous_frozen | None = None,
    max_step_size: float = 0.1,
    space_resolution: float = 0.01,
    max_steps: int = 100,
    information_level: InformationLevel = InformationLevel.COMPLETE,
    view_scope: ViewScope = ViewScope.COMPLETE,
    vision_radius: int = 10,
    render_mode: str | None = None,
) -> SpatialCompetitionEnv:
    """Create a raw spatial competition environment."""
    return SpatialCompetitionEnv(
        dimensions=dimensions,
        topology=topology,
        num_sellers=num_sellers,
        buyer_rate=buyer_rate,
        max_price=max_price,
        max_quality=max_quality,
        max_valuation=max_valuation,
        production_cost_factor=production_cost_factor,
        movement_cost=movement_cost,
        quality_taste_distr=quality_taste_distr,
        max_step_size=max_step_size,
        space_resolution=space_resolution,
        max_steps=max_steps,
        information_level=information_level,
        view_scope=view_scope,
        vision_radius=vision_radius,
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
        num_sellers: int = 3,
        buyer_rate: float = 0.1,
        max_price: float = 10.0,
        max_quality: float = 5.0,
        max_valuation: float = 10.0,
        production_cost_factor: float = 0.5,
        movement_cost: float = 0.1,
        quality_taste_distr: rv_continuous_frozen | None = None,
        max_step_size: float = 0.1,
        space_resolution: int = 100,
        max_steps: int = 10000,
        information_level: InformationLevel = InformationLevel.COMPLETE,
        view_scope: ViewScope = ViewScope.COMPLETE,
        vision_radius: int = 10,
        render_mode: str | None = None,
    ) -> None:
        """
        Initialize the spatial competition environment.

        Args:
            dimensions: Number of spatial dimensions (N)
            topology: Map topology ("rectangle" or "torus")
            num_sellers: Number of competing sellers
            buyer_rate: Fraction of space that spawns buyers each timestep (beta)
            max_price: Maximum price sellers can set (P)
            max_quality: Maximum quality level (Q)
            max_valuation: Maximum buyer valuation (V)
            production_cost_factor: Production cost parameter (gamma, C(q) = gamma*q^2)
            movement_cost: Cost of moving per unit distance (m)
            quality_taste_distr: Distribution for buyer quality taste (k_b)
            max_step_size: Maximum movement distance per step
            space_resolution: Grid resolution for discretized continuous space
            max_steps: Maximum episode length
            information_level: Information mode controlling what sellers can observe
            view_scope: View scope controlling spatial observation range
            vision_radius: Radius of vision for limited view modes in number of grid cells
            render_mode: Rendering mode

        """
        super().__init__()

        # Space configuration
        self.dimensions = dimensions
        self.topology = topology

        # Market parameters
        self.num_sellers = num_sellers
        self.buyer_rate = buyer_rate
        self.max_price = max_price
        self.max_quality = max_quality
        self.max_valuation = max_valuation
        self.production_cost_factor = production_cost_factor
        self.movement_cost = movement_cost
        self.quality_taste_distr = quality_taste_distr or uniform(0, 1)
        self.max_step_size = max_step_size
        self.space_resolution = space_resolution

        # Environment parameters
        self.max_steps = max_steps
        self.render_mode = render_mode

        # Information level parameters
        self.information_level = information_level
        self.view_scope = view_scope

        # TODO: Check its less than max vision radius for the topology
        self.vision_radius = space_resolution if view_scope == ViewScope.COMPLETE else vision_radius

        # Validate parameters
        if max_valuation > max_price:
            msg = f"max_valuation ({max_valuation}) must be <= max_price ({max_price})"
            raise ValueError(msg)

        # Initialize agents
        self.possible_agents = [f"seller_{i}" for i in range(num_sellers)]
        self.agent_name_mapping = dict(zip(self.possible_agents, list(range(len(self.possible_agents))), strict=True))

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
            dimensions=self.dimensions,
            vision_radius=self.vision_radius,
            space_resolution=self.space_resolution,
            max_price=self.max_price,
            max_quality=self.max_quality,
            max_valuation=self.max_valuation,
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

        # Initialize seller states: position, price, quality
        sellers: dict[str, Seller] = {}
        for agent_id in self.agents:
            sellers[agent_id] = Seller(
                idx=self.agent_name_mapping[agent_id],
                position=Position.uniform(self.rng, self.dimensions, self.space_resolution, self.topology),
                price=self.rng.uniform(0, self.max_price),
                quality=self.rng.uniform(0, self.max_quality),
            )

        self.competition = Competition(
            dimensions=self.dimensions,
            topology=self.topology,
            sellers=sellers,
            space_resolution=self.space_resolution,
            information_level=self.information_level,
            view_scope=self.view_scope,
            vision_radius=self.vision_radius,
            max_price=self.max_price,
            max_quality=self.max_quality,
            max_step_size=self.max_step_size,
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
        self.observations[agent] = self._get_agent_observation(agent)

        # Check termination conditions
        self.num_steps += 1
        if self.num_steps >= self.max_steps:
            self.truncations = dict.fromkeys(self.agents, True)

        # Process market interactions if all agents have acted
        if self._agent_selector.is_last():
            self.competition.env_step()
            self.rewards = self.competition.compute_rewards()
            self._agent_selector.reinit(self.rng.permutation(self.agents))

        self._next_agent()

    def _next_agent(self) -> None:
        """Advance to the next agent in the turn order."""
        self.agent_selection = next(self._agent_selector)

    def render(self) -> None:
        """Render the environment (placeholder implementation)."""

    def close(self) -> None:
        """Close the environment."""
