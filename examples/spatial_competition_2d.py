"""
2D Spatial Competition with Torus Topology and Gaussian Distributions.

This example demonstrates a 2D spatial competition model with:
- 4 sellers on a torus topology (wrap-around edges)
- Quality differentiation
- Gaussian buyer position distribution (off-center)
- Gaussian quality taste and distance factor for buyers
- Real-time Pygame rendering

Usage:
    python examples/spatial_competition_2d.py

Controls:
    - Close the window or press Ctrl+C to stop the simulation
"""

# ruff: noqa: T201

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
from scipy.stats import norm

from spatial_competition_pettingzoo.distributions import (
    ConstantUnivariateDistribution,
    MultivariateNormalDistribution,
)
from spatial_competition_pettingzoo.environment import env
from spatial_competition_pettingzoo.topology import Topology

if TYPE_CHECKING:
    from spatial_competition_pettingzoo.distributions import DistributionProtocol


class RandomPolicy2D:
    """Random policy for 2D movement with random price and quality adjustments."""

    def __init__(
        self,
        max_step_size: float,
        max_price: float,
        max_quality: float,
        seed: int | None = None,
    ) -> None:
        self.max_step_size = max_step_size
        self.max_price = max_price
        self.max_quality = max_quality
        self.rng = np.random.default_rng(seed)
        self.current_price = max_price / 2
        self.current_quality = max_quality / 2

    def compute_action(self, observation: dict[str, Any], reward: float) -> dict[str, Any]:
        """Compute random action including 2D movement, price, and quality."""
        del observation, reward  # Unused - pure random policy

        # Random 2D movement in range [-max_step_size, +max_step_size] for each dimension
        movement = self.rng.uniform(-self.max_step_size, self.max_step_size, size=2)

        # Random price adjustment
        price_change = self.rng.uniform(-0.5, 0.5)
        self.current_price = float(np.clip(self.current_price + price_change, 0.5, self.max_price - 0.5))

        # Random quality adjustment
        quality_change = self.rng.uniform(-0.3, 0.3)
        self.current_quality = float(np.clip(self.current_quality + quality_change, 0.0, self.max_quality))

        return {
            "movement": np.array(movement, dtype=np.float32),
            "price": np.array(self.current_price, dtype=np.float32),
            "quality": np.array(self.current_quality, dtype=np.float32),
        }


def run_2d_competition(
    num_cycles: int = 200,
    seed: int = 42,
) -> None:
    """
    Run the 2D spatial competition simulation with real-time Pygame rendering.

    Args:
        num_cycles: Number of environment cycles to run
        seed: Random seed for reproducibility

    """
    print("=" * 60)
    print("2D Spatial Competition - Torus Topology")
    print("=" * 60)
    print("""
Setup:
  - 4 sellers compete on a 2D torus (wrap-around) market
  - Buyers are distributed with a 2D Gaussian (off-center)
  - Buyer quality taste and distance factor are Gaussian
  - Sellers move randomly and adjust prices/quality
  - Buyers choose based on: value - distance*cost + quality*taste - price

Watch the sellers move around the torus!
Close the window to stop.
    """)

    # Environment parameters
    max_price = 10.0
    max_quality = 5.0
    max_step_size = 0.10
    new_buyers_per_step = 200

    # Buyer position: 2D Gaussian centered at (0.7, 0.3) - not the center
    buyer_position_mean = np.array([0.7, 0.3])
    buyer_position_cov = np.array([[0.01, 0.0], [0.0, 0.01]])  # Moderate spread

    # Buyer quality taste: Gaussian with mean 1.5, std 0.5
    buyer_quality_taste_mean = 1.5
    buyer_quality_taste_std = 0.5

    # Buyer distance factor: Gaussian with mean 2.0, std 0.5
    buyer_distance_factor_mean = 2.0
    buyer_distance_factor_std = 0.5

    # Seller initial quality: Gaussian with mean 2.5, std 1.0
    seller_quality_mean = 2.5
    seller_quality_std = 1.0

    # Create environment with rendering enabled
    environment = env(
        dimensions=2,  # 2D market
        topology=Topology.TORUS,  # Wrap-around edges
        space_resolution=100,
        num_sellers=10,  # 10 competing sellers
        max_price=max_price,
        max_quality=max_quality,
        include_quality=True,  # Enable quality differentiation
        max_step_size=max_step_size,
        production_cost_factor=0.1,  # Cost of quality
        movement_cost=0.0,
        seller_price_distr=ConstantUnivariateDistribution(max_price / 2),
        seller_quality_distr=cast("DistributionProtocol", norm(loc=seller_quality_mean, scale=seller_quality_std)),
        new_buyers_per_step=new_buyers_per_step,
        buyer_position_distr=MultivariateNormalDistribution(buyer_position_mean, buyer_position_cov),
        buyer_quality_taste_distr=cast(
            "DistributionProtocol", norm(loc=buyer_quality_taste_mean, scale=buyer_quality_taste_std)
        ),
        buyer_distance_factor_distr=cast(
            "DistributionProtocol", norm(loc=buyer_distance_factor_mean, scale=buyer_distance_factor_std)
        ),
        max_env_steps=num_cycles,
        render_mode="human",
        step_delay=0.1,
    )

    # Reset environment
    environment.reset(seed=seed)

    # Create policies for each seller
    policies = {
        agent: RandomPolicy2D(
            max_step_size=max_step_size,
            max_price=max_price,
            max_quality=max_quality,
            seed=seed + i,
        )
        for i, agent in enumerate(environment.possible_agents)
    }

    print(f"Starting simulation with {num_cycles} cycles...")
    print("Initial positions:")
    for agent in environment.possible_agents:
        obs = environment.observe(agent)
        pos = obs["own_position"]
        print(f"  {agent}: position = ({pos[0]:.3f}, {pos[1]:.3f})")
    print("-" * 60)
    print("Controls: Space=Pause, Click=Select entity, Slider=Speed")
    print("-" * 60)

    # Track cycle progress
    cycle_count = 0
    agents_acted_this_cycle = 0
    running = True

    try:
        # Main simulation loop
        for agent in environment.agent_iter():
            if not running:
                break

            # Get current state
            observation, reward, termination, truncation, _ = environment.last()

            # Get action (None if episode is done)
            action = None if termination or truncation else policies[agent].compute_action(observation, reward)

            environment.step(action)

            # Track cycles
            agents_acted_this_cycle += 1
            if agents_acted_this_cycle >= len(environment.possible_agents):
                agents_acted_this_cycle = 0
                cycle_count += 1

                # Print progress every 25 cycles
                if cycle_count % 25 == 0:
                    print(f"Cycle {cycle_count:3d}:")
                    for ag in environment.possible_agents:
                        obs = environment.observe(ag)
                        pos = obs["own_position"]
                        print(f"    {ag}: ({pos[0]:.3f}, {pos[1]:.3f})")

    except KeyboardInterrupt:
        print("\nSimulation interrupted by user.")
        running = False

    # Final results
    print("-" * 60)
    print("Final Results:")
    for agent in environment.possible_agents:
        obs = environment.observe(agent)
        pos = obs["own_position"]
        print(f"  {agent}: position = ({pos[0]:.3f}, {pos[1]:.3f})")

    print("=" * 60)

    # Cleanup
    environment.close()


if __name__ == "__main__":
    run_2d_competition(num_cycles=200, seed=42)
