"""
3D Spatial Competition with Torus Topology.

This example demonstrates a 3D spatial competition model with:
- 8 sellers on a torus topology (wrap-around in all 3 dimensions)
- Quality differentiation
- 3D Gaussian buyer position distribution
- Real-time stats display (3D cannot be directly visualized)

Usage:
    python examples/spatial_competition_3d.py

Controls:
    - Space = Pause/Resume
    - Slider = Adjust speed (far right = MAX)
    - Click leaderboard = Select seller for details
    - Close window to stop
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


class RandomPolicy3D:
    """Random policy for 3D movement with random price and quality adjustments."""

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
        """Compute random action including 3D movement, price, and quality."""
        del observation, reward  # Unused - pure random policy

        # Random 3D movement in range [-max_step_size, +max_step_size] for each dimension
        movement = self.rng.uniform(-self.max_step_size, self.max_step_size, size=3)

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


def run_3d_competition(
    num_cycles: int = 300,
    seed: int = 42,
) -> None:
    """
    Run the 3D spatial competition simulation.

    Args:
        num_cycles: Number of environment cycles to run
        seed: Random seed for reproducibility

    """
    print("=" * 60)
    print("3D Spatial Competition - Torus Topology")
    print("=" * 60)
    print("""
Setup:
  - 8 sellers compete in a 3D torus (wrap-around) market
  - Buyers are distributed with a 3D Gaussian (off-center)
  - Buyer quality taste and distance factor are Gaussian
  - Buyer valuation is Gaussian - some buyers may NOT buy if price is too high!
  - Sellers move randomly in 3D and adjust prices/quality
  - Buyers choose based on: value - distance*cost + quality*taste - price
  - Buyers only buy if utility > 0

Note: 3D space cannot be directly visualized, but you can:
  - See the leaderboard ranking sellers by reward
  - Click on sellers to view their 3D positions
  - Watch stats update in real-time
    """)

    # Environment parameters
    max_price = 10.0
    max_quality = 5.0
    max_step_size = 0.08
    new_buyers_per_step = 150
    num_sellers = 8

    # Buyer position: 3D Gaussian centered at (0.7, 0.3, 0.5) - not the center
    buyer_position_mean = np.array([0.7, 0.3, 0.5])
    buyer_position_cov = np.eye(3) * 0.02  # Moderate spread in all dimensions

    # Buyer quality taste: Gaussian with mean 1.5, std 0.5
    buyer_quality_taste_mean = 1.5
    buyer_quality_taste_std = 0.5

    # Buyer distance factor: Gaussian with mean 2.5, std 0.7
    buyer_distance_factor_mean = 2.5
    buyer_distance_factor_std = 0.7

    # Buyer valuation: Gaussian - buyers with low valuation may not buy!
    # Mean of 12 with std 4 means some buyers will have values below typical prices
    buyer_valuation_mean = 12.0
    buyer_valuation_std = 4.0

    # Seller initial quality: Gaussian with mean 2.5, std 1.0
    seller_quality_mean = 2.5
    seller_quality_std = 1.0

    # Create environment with rendering enabled
    environment = env(
        dimensions=3,  # 3D market!
        topology=Topology.TORUS,  # Wrap-around edges in all dimensions
        space_resolution=20,  # Grid resolution per dimension
        num_sellers=num_sellers,
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
        include_buyer_valuation=True,  # Enable finite buyer valuations
        buyer_valuation_distr=cast("DistributionProtocol", norm(loc=buyer_valuation_mean, scale=buyer_valuation_std)),
        max_env_steps=num_cycles,
        render_mode="human",
        step_delay=0.1,
    )

    # Reset environment
    environment.reset(seed=seed)

    # Create policies for each seller
    policies = {
        agent: RandomPolicy3D(
            max_step_size=max_step_size,
            max_price=max_price,
            max_quality=max_quality,
            seed=seed + i,
        )
        for i, agent in enumerate(environment.possible_agents)
    }

    print(f"Starting simulation with {num_cycles} cycles...")
    print("Initial 3D positions:")
    for agent in environment.possible_agents:
        obs = environment.observe(agent)
        pos = obs["own_position"]
        print(f"  {agent}: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
    print("-" * 60)
    print("Controls: Space=Pause, Click leaderboard=Select, Slider=Speed")
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

                # Print progress every 50 cycles
                if cycle_count % 50 == 0:
                    print(f"Cycle {cycle_count:3d}:")
                    for ag in environment.possible_agents[:3]:  # Just show first 3
                        obs = environment.observe(ag)
                        pos = obs["own_position"]
                        print(f"    {ag}: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
                    print("    ...")

    except KeyboardInterrupt:
        print("\nSimulation interrupted by user.")
        running = False

    # Final results
    print("-" * 60)
    print("Final 3D Positions:")
    for agent in environment.possible_agents:
        obs = environment.observe(agent)
        pos = obs["own_position"]
        print(f"  {agent}: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")

    print("=" * 60)

    # Cleanup
    environment.close()


if __name__ == "__main__":
    run_3d_competition(num_cycles=300, seed=42)
