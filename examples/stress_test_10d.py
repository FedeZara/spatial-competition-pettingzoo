"""
10D Stress Test - Massive Spatial Competition.

This example is a stress test with extreme parameters:
- 10 dimensions
- 100 suppliers (sellers)
- 1000 buyers spawning per step
- Quality and distance differentiation
- Torus topology

This tests the performance of the simulation and rendering.

Usage:
    python examples/stress_test_10d.py

Controls:
    - Space = Pause/Resume
    - Slider = Adjust speed (far right = MAX)
    - Click = Select entity for details
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


class RandomPolicyND:
    """Random policy for N-dimensional movement with random price and quality adjustments."""

    def __init__(
        self,
        dimensions: int,
        max_step_size: float,
        max_price: float,
        max_quality: float,
        seed: int | None = None,
    ) -> None:
        self.dimensions = dimensions
        self.max_step_size = max_step_size
        self.max_price = max_price
        self.max_quality = max_quality
        self.rng = np.random.default_rng(seed)
        self.current_price = max_price / 2
        self.current_quality = max_quality / 2

    def compute_action(self, observation: dict[str, Any], reward: float) -> dict[str, Any]:
        """Compute random action including N-D movement, price, and quality."""
        del observation, reward  # Unused - pure random policy

        # Random N-D movement
        movement = self.rng.uniform(-self.max_step_size, self.max_step_size, size=self.dimensions)

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


def run_stress_test(
    num_cycles: int = 500,
    seed: int = 42,
) -> None:
    """
    Run the 10D stress test simulation.

    Args:
        num_cycles: Number of environment cycles to run
        seed: Random seed for reproducibility

    """
    dimensions = 10
    num_sellers = 100
    new_buyers_per_step = 1000

    print("=" * 70)
    print("🔥 STRESS TEST: 10D Spatial Competition 🔥")
    print("=" * 70)
    print(f"""
Configuration:
  - Dimensions: {dimensions}
  - Sellers: {num_sellers}
  - Buyers per step: {new_buyers_per_step}
  - Topology: Torus (wrap-around in all dimensions)
  - Quality: Enabled

This is a LOT of computation per step!
The 10D space cannot be visualized, but you can see:
  - Leaderboard showing top 10 sellers by reward
  - Click on leaderboard to see seller details
  - Speed slider (use MAX for fastest simulation)
    """)

    # Environment parameters
    max_price = 10.0
    max_quality = 5.0
    max_step_size = 0.05  # Smaller steps in high dimensions

    # Buyer position: 10D Gaussian centered off-center
    buyer_position_mean = np.array([0.3, 0.7, 0.4, 0.6, 0.5, 0.5, 0.2, 0.8, 0.4, 0.6])
    # Small covariance for clustering
    buyer_position_cov = np.eye(dimensions) * 0.02

    # Buyer quality taste: Gaussian
    buyer_quality_taste_mean = 1.5
    buyer_quality_taste_std = 0.8

    # Buyer distance factor: Gaussian (higher in more dimensions = more sensitive)
    buyer_distance_factor_mean = 3.0
    buyer_distance_factor_std = 1.0

    # Seller initial quality: Gaussian
    seller_quality_mean = 2.5
    seller_quality_std = 1.5

    print("Creating environment...")

    # Create environment with rendering enabled
    environment = env(
        dimensions=dimensions,
        topology=Topology.TORUS,
        space_resolution=5,  # Lower resolution for performance
        num_sellers=num_sellers,
        max_price=max_price,
        max_quality=max_quality,
        include_quality=True,
        max_step_size=max_step_size,
        production_cost_factor=0.15,
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
    )

    print("Resetting environment...")
    observations, _ = environment.reset(seed=seed)

    # Create policies for each seller
    print(f"Creating {num_sellers} policies...")
    policies = {
        agent: RandomPolicyND(
            dimensions=dimensions,
            max_step_size=max_step_size,
            max_price=max_price,
            max_quality=max_quality,
            seed=seed + i,
        )
        for i, agent in enumerate(environment.possible_agents)
    }

    print(f"Starting stress test with {num_cycles} cycles...")
    print("-" * 70)
    print("TIP: Drag the speed slider all the way right for MAX speed!")
    print("-" * 70)

    # Track rewards for policy updates
    rewards: dict[str, float] = dict.fromkeys(environment.possible_agents, 0.0)
    running = True

    try:
        # Main simulation loop - parallel stepping
        for cycle_count in range(1, num_cycles + 1):
            if not running:
                break

            # Collect actions from all agents
            actions = {}
            for agent in environment.agents:
                actions[agent] = policies[agent].compute_action(observations[agent], rewards[agent])

            # Step all agents simultaneously
            observations, rewards, terminations, truncations, _ = environment.step(actions)

            # Check if all agents are done
            if all(terminations.values()) or all(truncations.values()):
                break

            # Print progress every 50 cycles
            if cycle_count % 50 == 0:
                print(f"Cycle {cycle_count}/{num_cycles} completed")

    except KeyboardInterrupt:
        print("\nSimulation interrupted by user.")
        running = False

    # Final results
    print("=" * 70)
    print(f"Stress test complete! Ran {cycle_count} cycles.")
    print(f"Total actions: {cycle_count * num_sellers}")
    print(f"Total buyers processed: ~{cycle_count * new_buyers_per_step}")
    print("=" * 70)

    # Cleanup
    environment.close()


if __name__ == "__main__":
    run_stress_test(num_cycles=500, seed=42)
