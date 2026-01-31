"""
Hotelling's Linear City Model with Q-Learning Agents.

This example demonstrates the classic Hotelling spatial competition model (1929)
where agents learn optimal strategies using tabular Q-learning.

State Space (discretized):
- Position: 10 bins along the linear market
- Own price: 5 bins from 0 to max_price
- Total: 50 states per agent

Action Space (discretized):
- Movement: 5 options (-max_step, -max_step/2, 0, +max_step/2, +max_step)
- Price change: 5 options (large decrease, small decrease, no change, small increase, large increase)
- Total: 25 actions

Reference: Hotelling, H. (1929). "Stability in Competition." The Economic Journal.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.random import Generator
from tqdm import tqdm

from spatial_competition_pettingzoo.distributions import (
    ConstantUnivariateDistribution,
    MultivariateUniformDistribution,
)
from spatial_competition_pettingzoo.environment import env
from spatial_competition_pettingzoo.topology import Topology


class QLearningAgent:
    """
    Tabular Q-learning agent for Hotelling competition.

    Uses discretized state (position + price) and action (movement + price change) spaces.
    """

    # State discretization
    N_POSITION_BINS = 10
    N_PRICE_BINS = 5

    # Action discretization
    N_MOVEMENT_OPTIONS = 5  # -max, -max/2, 0, +max/2, +max
    N_PRICE_OPTIONS = 5  # large dec, small dec, no change, small inc, large inc

    def __init__(
        self,
        agent_id: str,
        max_step_size: float,
        max_price: float,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        seed: int | None = None,
    ) -> None:
        self.agent_id = agent_id
        self.max_step_size = max_step_size
        self.max_price = max_price

        # Q-learning parameters
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Random number generator
        self.rng: Generator = np.random.default_rng(seed)

        # State and action dimensions
        self.n_states = self.N_POSITION_BINS * self.N_PRICE_BINS
        self.n_actions = self.N_MOVEMENT_OPTIONS * self.N_PRICE_OPTIONS

        # Initialize Q-table with small random values
        self.q_table = self.rng.uniform(0, 0.01, size=(self.n_states, self.n_actions))

        # Track current price (needed for price change actions)
        self.current_price = max_price / 2

        # Previous state-action for learning updates
        self.prev_state: int | None = None
        self.prev_action: int | None = None

        # Define movement and price change options
        self._movement_options = np.array(
            [
                -max_step_size,
                -max_step_size / 2,
                0.0,
                max_step_size / 2,
                max_step_size,
            ]
        )

        # Price changes as fractions of max_price
        self._price_change_options = (
            np.array(
                [
                    -0.2,  # Large decrease (20% of max)
                    -0.05,  # Small decrease (5% of max)
                    0.0,  # No change
                    0.05,  # Small increase (5% of max)
                    0.2,  # Large increase (20% of max)
                ]
            )
            * max_price
        )

    def discretize_state(self, observation: dict[str, Any]) -> int:
        """Convert continuous observation to discrete state index."""
        # Extract position and price from observation
        position = float(observation["own_position"][0])
        price = float(observation["own_price"])

        # Discretize position into bins [0, N_POSITION_BINS)
        position_bin = int(np.clip(position * self.N_POSITION_BINS, 0, self.N_POSITION_BINS - 1))

        # Discretize price into bins [0, N_PRICE_BINS)
        price_bin = int(np.clip(price / self.max_price * self.N_PRICE_BINS, 0, self.N_PRICE_BINS - 1))

        # Combine into single state index
        state = position_bin * self.N_PRICE_BINS + price_bin
        return state

    def action_to_continuous(self, action_idx: int) -> dict[str, Any]:
        """Convert discrete action index to continuous action dict."""
        # Decode action index
        movement_idx = action_idx // self.N_PRICE_OPTIONS
        price_change_idx = action_idx % self.N_PRICE_OPTIONS

        # Get movement value
        movement = self._movement_options[movement_idx]

        # Update current price with change
        price_change = self._price_change_options[price_change_idx]
        self.current_price = float(np.clip(self.current_price + price_change, 0.1, self.max_price))

        return {
            "movement": np.array([movement], dtype=np.float32),
            "price": np.array(self.current_price, dtype=np.float32),
        }

    def select_action(self, state: int, training: bool = True) -> int:
        """Select action using epsilon-greedy policy."""
        if training and self.rng.random() < self.epsilon:
            # Exploration: random action
            return int(self.rng.integers(0, self.n_actions))
        else:
            # Exploitation: best action from Q-table
            return int(np.argmax(self.q_table[state]))

    def update(self, reward: float, next_state: int, done: bool = False) -> None:
        """Update Q-table using the Q-learning update rule."""
        if self.prev_state is not None and self.prev_action is not None:
            # Q-learning update
            if done:
                target = reward
            else:
                target = reward + self.discount_factor * np.max(self.q_table[next_state])

            # Update Q-value
            self.q_table[self.prev_state, self.prev_action] += self.learning_rate * (
                target - self.q_table[self.prev_state, self.prev_action]
            )

    def decay_epsilon(self) -> None:
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def compute_action(self, observation: dict[str, Any], reward: float, training: bool = True) -> dict[str, Any]:
        """
        Compute action based on observation and update Q-table.

        Args:
            observation: Current observation from environment
            reward: Reward from previous action
            training: Whether to use exploration and update Q-table

        Returns:
            Action dictionary for the environment
        """
        # Discretize current state
        state = self.discretize_state(observation)

        # Update Q-table with previous transition
        if training:
            self.update(reward, state)

        # Select action
        action_idx = self.select_action(state, training)

        # Store for next update
        self.prev_state = state
        self.prev_action = action_idx

        # Convert to continuous action
        return self.action_to_continuous(action_idx)

    def reset_episode(self) -> None:
        """Reset agent state for new episode."""
        self.prev_state = None
        self.prev_action = None
        self.current_price = self.max_price / 2

    def get_policy_summary(self) -> dict[str, Any]:
        """Get summary of learned policy."""
        # For each state, get the best action
        best_actions = np.argmax(self.q_table, axis=1)

        # Decode actions
        movements = best_actions // self.N_PRICE_OPTIONS
        price_changes = best_actions % self.N_PRICE_OPTIONS

        return {
            "q_table": self.q_table.copy(),
            "best_actions": best_actions,
            "movement_distribution": np.bincount(movements, minlength=self.N_MOVEMENT_OPTIONS),
            "price_change_distribution": np.bincount(price_changes, minlength=self.N_PRICE_OPTIONS),
        }


def run_qlearning_simulation(
    num_episodes: int = 500,
    steps_per_episode: int = 50,
    seed: int = 42,
    max_price: float = 10.0,
    transport_cost: float = 2.0,
    step_size: float = 0.05,
    new_buyers_per_step: int = 100,
    learning_rate: float = 0.1,
    discount_factor: float = 0.95,
    epsilon_start: float = 1.0,
    epsilon_min: float = 0.01,
    epsilon_decay: float = 0.995,
    verbose: bool = True,
) -> dict:
    """
    Train Q-learning agents in the Hotelling environment.

    Args:
        num_episodes: Number of training episodes
        steps_per_episode: Number of environment cycles per episode
        seed: Random seed for reproducibility
        max_price: Maximum allowable price
        transport_cost: Cost per unit distance for buyers
        step_size: Maximum movement per step for sellers
        new_buyers_per_step: Number of customers arriving each period
        learning_rate: Q-learning learning rate (alpha)
        discount_factor: Q-learning discount factor (gamma)
        epsilon_start: Initial exploration rate
        epsilon_min: Minimum exploration rate
        epsilon_decay: Epsilon decay rate per episode
        verbose: Whether to print progress updates

    Returns:
        Dictionary containing training history and trained agents
    """
    if verbose:
        print("=" * 60)
        print("Q-Learning Hotelling Model Training")
        print("=" * 60)
        print(f"Episodes: {num_episodes}")
        print(f"Steps per episode: {steps_per_episode}")
        print(f"Learning rate: {learning_rate}")
        print(f"Discount factor: {discount_factor}")
        print(f"Epsilon: {epsilon_start} -> {epsilon_min} (decay={epsilon_decay})")
        print("-" * 60)

    # Create environment for agent initialization
    sample_env = env(
        dimensions=1,
        topology=Topology.RECTANGLE,
        space_resolution=100,
        num_sellers=2,
        max_price=max_price,
        max_quality=0.0,
        max_step_size=step_size,
        production_cost_factor=0.0,
        movement_cost=0.0,
        seller_position_distr=MultivariateUniformDistribution(dim=1, loc=0.0, scale=1.0),
        seller_price_distr=ConstantUnivariateDistribution(max_price / 2),
        seller_quality_distr=ConstantUnivariateDistribution(0.0),
        new_buyers_per_step=new_buyers_per_step,
        buyer_position_distr=MultivariateUniformDistribution(dim=1, loc=0.0, scale=1.0),
        buyer_quality_taste_distr=ConstantUnivariateDistribution(0.0),
        buyer_distance_factor_distr=ConstantUnivariateDistribution(transport_cost),
        max_env_steps=steps_per_episode,
    )
    sample_env.reset(seed=seed)

    # Create Q-learning agents
    agents = {
        agent_id: QLearningAgent(
            agent_id=agent_id,
            max_step_size=step_size,
            max_price=max_price,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            epsilon=epsilon_start,
            epsilon_min=epsilon_min,
            epsilon_decay=epsilon_decay,
            seed=seed + i,
        )
        for i, agent_id in enumerate(sample_env.possible_agents)
    }
    sample_env.close()

    # Training history
    history: dict[str, Any] = {
        "episode_rewards": {agent_id: [] for agent_id in agents},
        "episode_positions": {agent_id: [] for agent_id in agents},
        "epsilon": [],
    }

    # Training loop with progress bar
    pbar = tqdm(range(num_episodes), desc="Training", disable=not verbose)
    for episode in pbar:
        # Create fresh environment for each episode
        hotelling_env = env(
            dimensions=1,
            topology=Topology.RECTANGLE,
            space_resolution=100,
            num_sellers=2,
            max_price=max_price,
            max_quality=0.0,
            max_step_size=step_size,
            production_cost_factor=0.0,
            movement_cost=0.0,
            seller_position_distr=MultivariateUniformDistribution(dim=1, loc=0.0, scale=1.0),
            seller_price_distr=ConstantUnivariateDistribution(max_price / 2),
            seller_quality_distr=ConstantUnivariateDistribution(0.0),
            new_buyers_per_step=new_buyers_per_step,
            buyer_position_distr=MultivariateUniformDistribution(dim=1, loc=0.0, scale=1.0),
            buyer_quality_taste_distr=ConstantUnivariateDistribution(0.0),
            buyer_distance_factor_distr=ConstantUnivariateDistribution(transport_cost),
            max_env_steps=steps_per_episode,
        )
        hotelling_env.reset(seed=seed + episode)

        # Reset agents for new episode
        for agent in agents.values():
            agent.reset_episode()

        # Episode tracking
        episode_rewards = {agent_id: 0.0 for agent_id in agents}
        final_positions = {agent_id: 0.5 for agent_id in agents}

        # Run episode
        for agent_id in hotelling_env.agent_iter():
            observation, reward, termination, truncation, _ = hotelling_env.last()

            if termination or truncation:
                action = None
                # Final update for terminated agent
                if agents[agent_id].prev_state is not None:
                    state = agents[agent_id].discretize_state(observation)
                    agents[agent_id].update(reward, state, done=True)
            else:
                action = agents[agent_id].compute_action(observation, reward, training=True)
                final_positions[agent_id] = float(observation["own_position"][0])

            episode_rewards[agent_id] += reward
            hotelling_env.step(action)

        hotelling_env.close()

        # Record episode metrics
        for agent_id in agents:
            history["episode_rewards"][agent_id].append(episode_rewards[agent_id])
            history["episode_positions"][agent_id].append(final_positions[agent_id])

        # Decay epsilon for all agents
        for agent in agents.values():
            agent.decay_epsilon()

        history["epsilon"].append(list(agents.values())[0].epsilon)

        # Update progress bar with metrics
        if episode >= 10:
            avg_rewards = {agent_id: np.mean(history["episode_rewards"][agent_id][-50:]) for agent_id in agents}
            pbar.set_postfix(
                {
                    "ε": f"{history['epsilon'][-1]:.3f}",
                    "R0": f"{avg_rewards['seller_0']:.1f}",
                    "R1": f"{avg_rewards['seller_1']:.1f}",
                }
            )

    pbar.close()

    if verbose:
        print("-" * 60)
        print("Training Complete!")
        print("-" * 60)

    return {
        "agents": agents,
        "history": history,
        "config": {
            "num_episodes": num_episodes,
            "steps_per_episode": steps_per_episode,
            "max_price": max_price,
            "transport_cost": transport_cost,
            "step_size": step_size,
        },
    }


def evaluate_agents(
    agents: dict[str, QLearningAgent],
    num_episodes: int = 10,
    steps_per_episode: int = 100,
    seed: int = 9999,
    max_price: float = 10.0,
    transport_cost: float = 2.0,
    step_size: float = 0.05,
    new_buyers_per_step: int = 100,
    verbose: bool = True,
) -> dict:
    """
    Evaluate trained Q-learning agents without exploration.

    Args:
        agents: Dictionary of trained QLearningAgent instances
        num_episodes: Number of evaluation episodes
        steps_per_episode: Steps per evaluation episode
        seed: Random seed
        max_price: Maximum price
        transport_cost: Transport cost for buyers
        step_size: Maximum movement step
        new_buyers_per_step: Buyers per step
        verbose: Whether to print results

    Returns:
        Dictionary with evaluation results
    """
    if verbose:
        print("=" * 60)
        print("Evaluating Trained Agents (Greedy Policy)")
        print("=" * 60)

    eval_results: dict[str, Any] = {
        "episode_rewards": {agent_id: [] for agent_id in agents},
        "final_positions": {agent_id: [] for agent_id in agents},
        "final_prices": {agent_id: [] for agent_id in agents},
    }

    for episode in tqdm(range(num_episodes), desc="Evaluating", disable=not verbose):
        # Create environment
        hotelling_env = env(
            dimensions=1,
            topology=Topology.RECTANGLE,
            space_resolution=100,
            num_sellers=2,
            max_price=max_price,
            max_quality=0.0,
            max_step_size=step_size,
            production_cost_factor=0.0,
            movement_cost=0.0,
            seller_position_distr=MultivariateUniformDistribution(dim=1, loc=0.0, scale=1.0),
            seller_price_distr=ConstantUnivariateDistribution(max_price / 2),
            seller_quality_distr=ConstantUnivariateDistribution(0.0),
            new_buyers_per_step=new_buyers_per_step,
            buyer_position_distr=MultivariateUniformDistribution(dim=1, loc=0.0, scale=1.0),
            buyer_quality_taste_distr=ConstantUnivariateDistribution(0.0),
            buyer_distance_factor_distr=ConstantUnivariateDistribution(transport_cost),
            max_env_steps=steps_per_episode,
        )
        hotelling_env.reset(seed=seed + episode)

        # Reset agents
        for agent in agents.values():
            agent.reset_episode()

        episode_rewards: dict[str, float] = dict.fromkeys(agents, 0.0)
        final_obs: dict[str, dict[str, Any]] = {agent_id: {} for agent_id in agents}

        for agent_id in hotelling_env.agent_iter():
            observation, reward, termination, truncation, _ = hotelling_env.last()

            if termination or truncation:
                action = None
            else:
                # Use greedy policy (training=False disables exploration)
                action = agents[agent_id].compute_action(observation, reward, training=False)
                final_obs[agent_id] = observation

            episode_rewards[agent_id] += reward
            hotelling_env.step(action)

        hotelling_env.close()

        # Record results
        for agent_id in agents:
            eval_results["episode_rewards"][agent_id].append(episode_rewards[agent_id])
            if final_obs[agent_id] is not None:
                eval_results["final_positions"][agent_id].append(float(final_obs[agent_id]["own_position"][0]))
                eval_results["final_prices"][agent_id].append(float(final_obs[agent_id]["own_price"]))

    if verbose:
        print(f"Evaluation over {num_episodes} episodes:")
        for agent_id in agents:
            avg_reward = np.mean(eval_results["episode_rewards"][agent_id])
            avg_pos = np.mean(eval_results["final_positions"][agent_id])
            avg_price = np.mean(eval_results["final_prices"][agent_id])
            print(
                f"  {agent_id}: Avg Reward = {avg_reward:.1f}, "
                f"Avg Final Pos = {avg_pos:.3f}, Avg Final Price = {avg_price:.2f}"
            )
        print("=" * 60)

    return eval_results


def plot_learning_curves(results: dict, save_path: str | None = None) -> None:
    """
    Plot training learning curves.

    Args:
        results: Dictionary from run_qlearning_simulation
        save_path: Optional path to save the figure
    """
    history = results["history"]
    config = results["config"]
    agents = list(results["agents"].keys())

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Q-Learning Hotelling Model - Training Progress", fontsize=14, fontweight="bold")

    colors = ["#E63946", "#457B9D"]
    episodes = range(1, config["num_episodes"] + 1)

    # 1. Episode rewards
    ax1 = axes[0, 0]
    for i, agent_id in enumerate(agents):
        rewards = history["episode_rewards"][agent_id]
        ax1.plot(episodes, rewards, label=agent_id, color=colors[i], alpha=0.3, linewidth=0.5)
        # Smoothed version
        window = min(50, len(rewards) // 10 + 1)
        smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
        ax1.plot(
            range(window, len(rewards) + 1),
            smoothed,
            color=colors[i],
            linewidth=2,
            label=f"{agent_id} (smoothed)",
        )
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Total Reward")
    ax1.set_title("Episode Rewards")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Final positions over training
    ax2 = axes[0, 1]
    for i, agent_id in enumerate(agents):
        positions = history["episode_positions"][agent_id]
        ax2.plot(episodes, positions, label=agent_id, color=colors[i], alpha=0.5, linewidth=1)
    ax2.axhline(y=0.5, color="gray", linestyle="--", alpha=0.7, label="Center")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Final Position")
    ax2.set_title("Final Positions Over Training")
    ax2.set_ylim(0, 1)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Epsilon decay
    ax3 = axes[1, 0]
    ax3.plot(episodes, history["epsilon"], color="#2A9D8F", linewidth=2)
    ax3.set_xlabel("Episode")
    ax3.set_ylabel("Epsilon")
    ax3.set_title("Exploration Rate Decay")
    ax3.grid(True, alpha=0.3)

    # 4. Cumulative rewards
    ax4 = axes[1, 1]
    for i, agent_id in enumerate(agents):
        cum_rewards = np.cumsum(history["episode_rewards"][agent_id])
        ax4.plot(episodes, cum_rewards, label=agent_id, color=colors[i], linewidth=2)
    ax4.set_xlabel("Episode")
    ax4.set_ylabel("Cumulative Reward")
    ax4.set_title("Cumulative Rewards Over Training")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    else:
        plt.show()


def plot_q_table_heatmap(agent: QLearningAgent, save_path: str | None = None) -> None:
    """
    Visualize the Q-table as a heatmap showing best action values.

    Args:
        agent: Trained QLearningAgent
        save_path: Optional path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Q-Table Analysis: {agent.agent_id}", fontsize=14, fontweight="bold")

    # Reshape Q-table max values into position x price grid
    max_q_values = np.max(agent.q_table, axis=1).reshape(agent.N_POSITION_BINS, agent.N_PRICE_BINS)

    # 1. Max Q-values heatmap
    ax1 = axes[0]
    im1 = ax1.imshow(max_q_values, aspect="auto", cmap="viridis", origin="lower")
    ax1.set_xlabel("Price Bin")
    ax1.set_ylabel("Position Bin")
    ax1.set_title("Max Q-Values by State")
    ax1.set_xticks(range(agent.N_PRICE_BINS))
    ax1.set_yticks(range(agent.N_POSITION_BINS))
    ax1.set_xticklabels([f"{i * agent.max_price / agent.N_PRICE_BINS:.1f}" for i in range(agent.N_PRICE_BINS)])
    ax1.set_yticklabels([f"{i / agent.N_POSITION_BINS:.1f}" for i in range(agent.N_POSITION_BINS)])
    plt.colorbar(im1, ax=ax1, label="Max Q-Value")

    # 2. Best action heatmap (showing movement tendency)
    best_actions = np.argmax(agent.q_table, axis=1)
    movement_actions = best_actions // agent.N_PRICE_OPTIONS
    movement_grid = movement_actions.reshape(agent.N_POSITION_BINS, agent.N_PRICE_BINS)

    ax2 = axes[1]
    im2 = ax2.imshow(movement_grid, aspect="auto", cmap="RdYlBu", origin="lower", vmin=0, vmax=4)
    ax2.set_xlabel("Price Bin")
    ax2.set_ylabel("Position Bin")
    ax2.set_title("Best Movement Action by State")
    ax2.set_xticks(range(agent.N_PRICE_BINS))
    ax2.set_yticks(range(agent.N_POSITION_BINS))
    ax2.set_xticklabels([f"{i * agent.max_price / agent.N_PRICE_BINS:.1f}" for i in range(agent.N_PRICE_BINS)])
    ax2.set_yticklabels([f"{i / agent.N_POSITION_BINS:.1f}" for i in range(agent.N_POSITION_BINS)])
    cbar = plt.colorbar(im2, ax=ax2, label="Movement Action")
    cbar.set_ticks([0, 1, 2, 3, 4])
    cbar.set_ticklabels(["Left Max", "Left Half", "Stay", "Right Half", "Right Max"])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    else:
        plt.show()


def plot_policy_summary(agents: dict[str, QLearningAgent], save_path: str | None = None) -> None:
    """
    Summarize the learned policies of all agents.

    Args:
        agents: Dictionary of trained agents
        save_path: Optional path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Learned Policy Summary", fontsize=14, fontweight="bold")

    colors = ["#E63946", "#457B9D"]
    agent_ids = list(agents.keys())

    # Movement action distribution
    ax1 = axes[0]
    x = np.arange(5)
    width = 0.35
    movement_labels = ["Left\nMax", "Left\nHalf", "Stay", "Right\nHalf", "Right\nMax"]

    for i, agent_id in enumerate(agent_ids):
        summary = agents[agent_id].get_policy_summary()
        movement_dist = summary["movement_distribution"] / summary["movement_distribution"].sum()
        offset = width / 2 if i == 0 else -width / 2
        ax1.bar(x + offset, movement_dist, width, label=agent_id, color=colors[i])

    ax1.set_xlabel("Movement Action")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Movement Action Distribution (Best Actions)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(movement_labels)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")

    # Price change distribution
    ax2 = axes[1]
    price_labels = ["Large\nDec", "Small\nDec", "No\nChange", "Small\nInc", "Large\nInc"]

    for i, agent_id in enumerate(agent_ids):
        summary = agents[agent_id].get_policy_summary()
        price_dist = summary["price_change_distribution"] / summary["price_change_distribution"].sum()
        offset = width / 2 if i == 0 else -width / 2
        ax2.bar(x + offset, price_dist, width, label=agent_id, color=colors[i])

    ax2.set_xlabel("Price Change Action")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Price Change Action Distribution (Best Actions)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(price_labels)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("HOTELLING MODEL WITH Q-LEARNING AGENTS")
    print("=" * 60)
    print("""
This simulation trains Q-learning agents to compete in the
classic Hotelling spatial competition model.

The agents learn:
- Where to position themselves on the linear market
- What prices to set to maximize profits

State Space: Position (10 bins) x Price (5 bins) = 50 states
Action Space: Movement (5 options) x Price change (5 options) = 25 actions
    """)

    # Training phase
    print("\n" + "-" * 60)
    print("Phase 1: Training")
    print("-" * 60)
    training_results = run_qlearning_simulation(
        num_episodes=500,
        steps_per_episode=50,
        seed=42,
        max_price=10.0,
        transport_cost=2.0,
        step_size=0.05,
        new_buyers_per_step=100,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon_start=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        verbose=True,
    )

    # Evaluation phase
    print("\n" + "-" * 60)
    print("Phase 2: Evaluation")
    print("-" * 60)
    eval_results = evaluate_agents(
        agents=training_results["agents"],
        num_episodes=20,
        steps_per_episode=100,
        seed=9999,
        max_price=10.0,
        transport_cost=2.0,
        step_size=0.05,
        new_buyers_per_step=100,
        verbose=True,
    )

    # Visualization
    try:
        print("\n" + "-" * 60)
        print("Phase 3: Visualization")
        print("-" * 60)
        plot_learning_curves(training_results)
        plot_policy_summary(training_results["agents"])

        # Show Q-table for one agent
        first_agent = list(training_results["agents"].values())[0]
        plot_q_table_heatmap(first_agent)

    except Exception as e:  # noqa: BLE001
        print(f"Could not display plots (matplotlib backend issue): {e}")
        print("Run in an environment with GUI support to see visualizations.")

    print("\n" + "=" * 60)
    print("Simulation Complete!")
    print("=" * 60)
