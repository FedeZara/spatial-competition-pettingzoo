"""
Hotelling's Linear City Model Simulation.

This example demonstrates the classic Hotelling spatial competition model (1929) using
the PettingZoo AEC environment interface. In this model:

- Two ice cream vendors (sellers) compete on a linear beach (1D space)
- Customers (buyers) are uniformly distributed along the beach
- Customers choose the vendor that minimizes their total cost (price + travel cost)
- The classic result: both vendors converge to the center (principle of minimum differentiation)

Reference: Hotelling, H. (1929). "Stability in Competition." The Economic Journal.
"""

# ruff: noqa: T201

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from spatial_competition_pettingzoo.distributions import (
    ConstantUnivariateDistribution,
    MultivariateUniformDistribution,
)
from spatial_competition_pettingzoo.environment import env
from spatial_competition_pettingzoo.topology import Topology


class HotellingPolicy:
    """
    Simple adaptive policy for Hotelling competition.

    Each seller moves toward the center while adjusting prices based on
    relative sales performance.
    """

    def __init__(
        self,
        agent_id: str,
        max_step_size: float,
        max_price: float,
        price_adjustment: float = 0.1,
        seed: int | None = None,
    ) -> None:
        self.agent_id = agent_id
        self.max_step_size = max_step_size
        self.max_price = max_price
        self.price_adjustment = price_adjustment
        self.rng = np.random.default_rng(seed)

        # Track state for adaptive behavior
        self.last_reward = 0.0
        self.current_price = max_price / 2

    def compute_action(self, observation: dict[str, Any], reward: float) -> dict[str, Any]:
        """
        Compute action based on observation and previous reward.

        Strategy:
        - Movement: Move toward the market center (0.5)
        - Price: Increase if reward improved, decrease otherwise
        """
        # Extract own position from observation
        own_position = observation["own_position"]
        current_pos = float(own_position[0])  # First element is position (1D)

        # Movement strategy: Move toward center
        center = 0.5
        toward_center = center - current_pos

        # Calculate movement with some randomness
        movement_magnitude = min(abs(toward_center), self.max_step_size) * np.sign(toward_center)
        movement_magnitude += self.rng.uniform(-self.max_step_size * 0.1, self.max_step_size * 0.1)

        movement = np.array([movement_magnitude], dtype=np.float32)

        # Price strategy: Adjust based on reward changes
        if reward > self.last_reward:
            # Reward improved, try increasing price slightly
            self.current_price = min(self.current_price + self.price_adjustment * 0.5, self.max_price)
        elif reward < self.last_reward:
            # Reward decreased, lower price to attract more customers
            self.current_price = max(self.current_price - self.price_adjustment, 0.1)

        self.last_reward = reward

        return {
            "movement": movement,
            "price": np.array(self.current_price, dtype=np.float32),
        }


def run_hotelling_simulation(
    num_env_cycles: int = 100,
    seed: int = 42,
    initial_price: float = 5.0,
    max_price: float = 10.0,
    transport_cost: float = 2.0,
    step_size: float = 0.02,
    new_buyers_per_step: int = 100,
    verbose: bool = True,
) -> dict:
    """
    Run a Hotelling model simulation using the PettingZoo environment.

    Args:
        num_env_cycles: Number of environment cycles (each cycle = all agents act once)
        seed: Random seed for reproducibility
        initial_price: Starting price for sellers
        max_price: Maximum allowable price
        transport_cost: Cost per unit distance for buyers (distance_factor)
        step_size: Maximum movement per step for sellers
        new_buyers_per_step: Number of customers arriving each period
        verbose: Whether to print progress updates

    Returns:
        Dictionary containing simulation history and final results

    """
    # Create the PettingZoo environment configured for classic Hotelling model
    hotelling_env = env(
        dimensions=1,  # Linear market (beach)
        topology=Topology.RECTANGLE,  # Linear market with endpoints
        space_resolution=100,
        num_sellers=2,
        max_price=max_price,
        max_quality=0.0,  # No quality differentiation in classic Hotelling
        max_step_size=step_size,
        production_cost_factor=0.0,  # Zero production cost for simplicity
        movement_cost=0.0,  # No movement cost for simplicity
        seller_position_distr=MultivariateUniformDistribution(dim=1, loc=0.0, scale=1.0),
        seller_price_distr=ConstantUnivariateDistribution(initial_price),
        seller_quality_distr=ConstantUnivariateDistribution(0.0),
        new_buyers_per_step=new_buyers_per_step,
        buyer_position_distr=MultivariateUniformDistribution(dim=1, loc=0.0, scale=1.0),
        buyer_valuation_distr=ConstantUnivariateDistribution(np.inf),  # High valuation
        buyer_quality_taste_distr=ConstantUnivariateDistribution(0.0),  # No quality preference
        buyer_distance_factor_distr=ConstantUnivariateDistribution(transport_cost),
        max_env_steps=num_env_cycles,  # Number of full environment cycles
    )

    # Reset the environment
    hotelling_env.reset(seed=seed)

    # Create policies for each agent
    policies = {
        agent: HotellingPolicy(
            agent_id=agent,
            max_step_size=step_size,
            max_price=max_price,
            seed=seed + i,
        )
        for i, agent in enumerate(hotelling_env.possible_agents)
    }

    # History tracking
    history: dict[str, dict[str, list]] = {
        "positions": {agent: [] for agent in hotelling_env.possible_agents},
        "prices": {agent: [] for agent in hotelling_env.possible_agents},
        "rewards": {agent: [] for agent in hotelling_env.possible_agents},
    }

    if verbose:
        print("=" * 60)
        print("Hotelling's Linear City Model Simulation (PettingZoo)")
        print("=" * 60)
        print(f"Number of environment cycles: {num_env_cycles}")
        print(f"Transport cost: {transport_cost}")
        print(f"Buyers per step: {new_buyers_per_step}")
        print("-" * 60)

    cycle_count = 0
    agent_count_in_cycle = 0

    # Run the AEC environment loop
    for agent in hotelling_env.agent_iter():
        # Get observation, reward, termination, truncation, info for current agent
        observation, reward, termination, truncation, _ = hotelling_env.last()

        # Record state at the start of each cycle
        if agent_count_in_cycle == 0:
            for ag in hotelling_env.possible_agents:
                obs = hotelling_env.observe(ag)
                history["positions"][ag].append(float(obs["own_position"][0]))  # Position
                history["prices"][ag].append(float(obs["own_price"]))  # Price

        # Check if episode is done
        if termination or truncation:
            action = None
        else:
            # Get action from policy
            policy = policies[agent]
            action = policy.compute_action(observation, reward)

        # Record reward
        history["rewards"][agent].append(reward)

        # Take action
        hotelling_env.step(action)

        # Track cycles
        agent_count_in_cycle += 1
        if agent_count_in_cycle >= len(hotelling_env.possible_agents):
            agent_count_in_cycle = 0
            cycle_count += 1

            # Periodic progress update
            if verbose and cycle_count % 25 == 0:
                positions = [history["positions"][ag][-1] for ag in hotelling_env.possible_agents]
                print(f"Cycle {cycle_count:3d}: Positions = ({positions[0]:.3f}, {positions[1]:.3f})")

    hotelling_env.close()

    # Calculate final results
    final_positions = tuple(history["positions"][ag][-1] for ag in hotelling_env.possible_agents)
    total_rewards = tuple(sum(history["rewards"][ag]) for ag in hotelling_env.possible_agents)

    if verbose:
        print("-" * 60)
        print("Final Results:")
        for i, agent in enumerate(hotelling_env.possible_agents):
            pos = final_positions[i]
            total_reward = total_rewards[i]
            print(f"  {agent}: Position = {pos:.3f}, Total Reward = {total_reward:.1f}")
        print(f"  Final distance between sellers: {abs(final_positions[1] - final_positions[0]):.3f}")
        print(f"  Midpoint: {sum(final_positions) / 2:.3f} (center = 0.500)")
        print("=" * 60)

    return {
        "history": history,
        "final_positions": final_positions,
        "total_rewards": total_rewards,
        "possible_agents": hotelling_env.possible_agents,
    }


def plot_results(results: dict, save_path: str | None = None) -> None:
    """
    Plot the simulation results.

    Args:
        results: Dictionary from run_hotelling_simulation
        save_path: Optional path to save the figure

    """
    history = results["history"]
    agents = results["possible_agents"]
    steps = range(len(history["positions"][agents[0]]))

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Hotelling's Linear City Model - Simulation Results", fontsize=14, fontweight="bold")

    # Colors
    colors = ["#E63946", "#457B9D"]

    # 1. Position over time
    ax1 = axes[0, 0]
    for i, agent in enumerate(agents):
        ax1.plot(steps, history["positions"][agent], label=agent, color=colors[i], linewidth=2)
    ax1.axhline(y=0.5, color="gray", linestyle="--", alpha=0.7, label="Center")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Position")
    ax1.set_title("Seller Positions Over Time")
    ax1.legend()
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)

    # 2. Price over time
    ax2 = axes[0, 1]
    for i, agent in enumerate(agents):
        ax2.plot(steps, history["prices"][agent], label=agent, color=colors[i], linewidth=2)
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Price")
    ax2.set_title("Seller Prices Over Time")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Rewards per step
    ax3 = axes[1, 0]
    # Rewards are recorded per agent turn, so we need to handle differently
    for i, agent in enumerate(agents):
        rewards = history["rewards"][agent]
        ax3.plot(range(len(rewards)), rewards, label=agent, color=colors[i], alpha=0.7, linewidth=1.5)
    ax3.set_xlabel("Agent Turn")
    ax3.set_ylabel("Reward")
    ax3.set_title("Rewards Per Turn")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Cumulative rewards
    ax4 = axes[1, 1]
    for i, agent in enumerate(agents):
        cum_rewards = np.cumsum(history["rewards"][agent])
        ax4.plot(range(len(cum_rewards)), cum_rewards, label=agent, color=colors[i], linewidth=2)
    ax4.set_xlabel("Agent Turn")
    ax4.set_ylabel("Cumulative Reward")
    ax4.set_title("Cumulative Rewards Over Time")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    else:
        plt.show()


def visualize_market(results: dict, step: int = -1) -> None:
    """
    Visualize the linear market at a specific step.

    Args:
        results: Dictionary from run_hotelling_simulation
        step: Step to visualize (-1 for final step)

    """
    history = results["history"]
    agents = results["possible_agents"]

    if step == -1:
        step = len(history["positions"][agents[0]]) - 1

    positions = [history["positions"][agent][step] for agent in agents]

    _, ax = plt.subplots(figsize=(12, 3))

    # Draw the linear market (beach)
    ax.axhline(y=0, color="sandybrown", linewidth=20, alpha=0.5, label="Beach")
    ax.axhline(y=0, color="tan", linewidth=2)

    # Mark the endpoints
    ax.plot([0, 1], [0, 0], "k|", markersize=30, markeredgewidth=3)

    # Colors
    colors = ["#E63946", "#457B9D"]

    # Plot sellers as ice cream vendors
    for i, (agent, pos) in enumerate(zip(agents, positions, strict=True)):
        ax.plot(pos, 0, "o", markersize=25, color=colors[i], label=agent, zorder=5)
        y_offset = 0.15 if i == 0 else -0.15
        va = "bottom" if i == 0 else "top"
        ax.annotate(f"{agent}\n({pos:.2f})", (pos, y_offset), ha="center", fontsize=10, fontweight="bold", va=va)

    # Market division line (customers go to nearest seller)
    if len(positions) == 2 and positions[0] != positions[1]:
        midpoint = sum(positions) / 2
        ax.axvline(x=midpoint, color="gray", linestyle="--", alpha=0.7, label=f"Market divide ({midpoint:.2f})")

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.4, 0.4)
    ax.set_xlabel("Position on Linear Market", fontsize=12)
    ax.set_title(f"Hotelling's Linear City - Step {step + 1}", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right")
    ax.set_yticks([])

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Run the simulation
    print("\n" + "=" * 60)
    print("HOTELLING'S SPATIAL COMPETITION MODEL")
    print("=" * 60)
    print("""
This simulation demonstrates the classic Hotelling model (1929)
using the PettingZoo AEC (Agent Environment Cycle) framework.

Setup:
- Two ice cream vendors compete on a linear beach (0 to 1)
- Customers are uniformly distributed along the beach
- Each customer buys from the vendor minimizing: price + transport cost

Classic Result (Principle of Minimum Differentiation):
- Both vendors tend to locate at the center of the market
- This maximizes their potential customer base
- Even though spreading out would reduce customer travel costs
    """)

    # Scenario 1: Classic Hotelling with default settings
    print("\n" + "-" * 60)
    print("Scenario 1: Standard Hotelling Model")
    print("-" * 60)
    results_1 = run_hotelling_simulation(
        num_env_cycles=100,
        seed=42,
        transport_cost=2.0,
    )

    # Scenario 2: High transport cost (makes location more important)
    print("\n" + "-" * 60)
    print("Scenario 2: High transport cost (t=5.0)")
    print("-" * 60)
    results_2 = run_hotelling_simulation(
        num_env_cycles=100,
        seed=123,
        transport_cost=5.0,
    )

    # Scenario 3: Many buyers (more stable demand)
    print("\n" + "-" * 60)
    print("Scenario 3: High buyer density (200 buyers/step)")
    print("-" * 60)
    results_3 = run_hotelling_simulation(
        num_env_cycles=100,
        seed=456,
        new_buyers_per_step=200,
        transport_cost=2.0,
    )

    # Plot results for the first scenario
    try:
        print("\n" + "-" * 60)
        print("Generating plots for Scenario 1...")
        print("-" * 60)
        plot_results(results_1)
        visualize_market(results_1)
    except Exception as e:  # noqa: BLE001
        print(f"Could not display plots (matplotlib backend issue): {e}")
        print("Run in an environment with GUI support to see visualizations.")
