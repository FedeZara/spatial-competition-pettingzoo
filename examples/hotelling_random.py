"""
Hotelling's Linear City Model with Real-Time Pygame Rendering.

This example demonstrates a spatial competition model with live visualization
using Pygame. Two sellers move randomly on a 1D linear market while buyers
choose sellers based on price, distance, and quality.

Usage:
    python examples/hotelling_rendered.py

Controls:
    - Close the window or press Ctrl+C to stop the simulation
"""

# ruff: noqa: T201

from __future__ import annotations

from typing import Any

import numpy as np

from spatial_competition_pettingzoo.distributions import (
    ConstantUnivariateDistribution,
    MultivariateUniformDistribution,
)
from spatial_competition_pettingzoo.environment import env
from spatial_competition_pettingzoo.topology import Topology


class RandomPolicy:
    """Random policy that moves sellers in random directions with random price and quality."""

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
        """Compute random action including movement, price, and quality."""
        del observation, reward  # Unused - pure random policy

        # Random movement in range [-max_step_size, +max_step_size]
        movement = self.rng.uniform(-self.max_step_size, self.max_step_size)

        # Random price adjustment
        price_change = self.rng.uniform(-0.5, 0.5)
        self.current_price = float(np.clip(self.current_price + price_change, 0.5, self.max_price - 0.5))

        # Random quality adjustment
        quality_change = self.rng.uniform(-0.3, 0.3)
        self.current_quality = float(np.clip(self.current_quality + quality_change, 0.0, self.max_quality))

        return {
            "movement": np.array([movement], dtype=np.float32),
            "price": np.array(self.current_price, dtype=np.float32),
            "quality": np.array(self.current_quality, dtype=np.float32),
        }


def run_hotelling_with_rendering(
    num_cycles: int = 200,
    seed: int = 42,
) -> None:
    """
    Run the Hotelling model simulation with real-time Pygame rendering.

    Args:
        num_cycles: Number of environment cycles to run
        seed: Random seed for reproducibility

    """
    print("=" * 60)
    print("Hotelling's Linear City Model - Live Visualization")
    print("=" * 60)
    print("""
Setup:
  - Two sellers compete on a 1D linear market (beach)
  - Buyers are uniformly distributed
  - Sellers move randomly and adjust prices/quality randomly
  - Buyers choose based on: value - distance*cost + quality*taste - price

Watch the sellers move around the market!
Close the window to stop.
    """)

    # Environment parameters
    max_price = 10.0
    max_quality = 5.0
    max_step_size = 0.02
    transport_cost = 2.0
    quality_taste = 0.0  # How much buyers value quality
    new_buyers_per_step = 50

    # Create environment with rendering enabled
    environment = env(
        dimensions=1,  # 1D linear market
        topology=Topology.RECTANGLE,
        space_resolution=100,
        num_sellers=2,
        max_price=max_price,
        max_quality=max_quality,
        include_quality=True,  # Enable quality differentiation
        max_step_size=max_step_size,
        production_cost_factor=0.1,  # Cost of quality: gamma * q^2
        movement_cost=0.0,
        seller_position_distr=MultivariateUniformDistribution(dim=1, loc=0.0, scale=1.0),
        seller_price_distr=ConstantUnivariateDistribution(max_price / 2),
        seller_quality_distr=ConstantUnivariateDistribution(max_quality / 2),
        new_buyers_per_step=new_buyers_per_step,
        buyer_position_distr=MultivariateUniformDistribution(dim=1, loc=0.0, scale=1.0),
        buyer_quality_taste_distr=ConstantUnivariateDistribution(quality_taste),
        buyer_distance_factor_distr=ConstantUnivariateDistribution(transport_cost),
        max_env_steps=num_cycles,
        render_mode="human",
    )

    # Reset environment
    observations, _ = environment.reset(seed=seed)

    # Create policies for each seller
    policies = {
        agent: RandomPolicy(
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
        print(f"  {agent}: position = {observations[agent]['own_position'][0]:.3f}")
    print("-" * 60)

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

            # Print progress every 25 cycles
            if cycle_count % 25 == 0:
                positions = [observations[ag]["own_position"][0] for ag in environment.possible_agents]
                print(f"Cycle {cycle_count:3d}: Positions = ({positions[0]:.3f}, {positions[1]:.3f})")

    except KeyboardInterrupt:
        print("\nSimulation interrupted by user.")
        running = False

    # Final results
    print("-" * 60)
    print("Final Results:")
    for agent in environment.possible_agents:
        pos = observations[agent]["own_position"][0]
        print(f"  {agent}: position = {pos:.3f}")

    if len(environment.possible_agents) == 2:
        positions = [observations[ag]["own_position"][0] for ag in environment.possible_agents]
        distance = abs(positions[1] - positions[0])
        midpoint = sum(positions) / 2
        print(f"  Distance between sellers: {distance:.3f}")
        print(f"  Midpoint: {midpoint:.3f} (center = 0.500)")

    print("=" * 60)

    # Cleanup
    environment.close()


if __name__ == "__main__":
    run_hotelling_with_rendering(num_cycles=200, seed=42)
