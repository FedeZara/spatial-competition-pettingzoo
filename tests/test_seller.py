"""Tests for the Seller class."""

import numpy as np
import pytest

from spatial_competition_pettingzoo.position import Position
from spatial_competition_pettingzoo.seller import Seller
from spatial_competition_pettingzoo.topology import Topology


class TestSeller:
    """Test class for Seller functionality."""

    @pytest.fixture
    def sample_position(self) -> Position:
        """Create a sample position for testing."""
        return Position(
            space_resolution=10,
            topology=Topology.RECTANGLE,
            tensor_coordinates=np.array([3, 4], dtype=np.int32),
        )

    @pytest.fixture
    def sample_seller(self, sample_position: Position) -> Seller:
        """Create a sample seller for testing."""
        return Seller(agent_id="seller_1", position=sample_position, price=5.0, quality=0.8)

    @pytest.fixture
    def movement_position(self) -> Position:
        """Create a movement position for testing."""
        # Movement size is 0.22360679774997896
        return Position(
            space_resolution=10,
            topology=Topology.RECTANGLE,
            tensor_coordinates=np.array([2, 1], dtype=np.int32),
        )

    def test_move_updates_position(self, sample_seller: Seller, movement_position: Position) -> None:
        """Test that move method updates the seller's position correctly."""
        original_position = sample_seller.position
        original_coordinates = original_position.tensor_coordinates.copy()

        sample_seller.move(movement_position)

        # Position should be updated by adding the movement
        expected_coordinates = original_coordinates + movement_position.tensor_coordinates
        # Clipped to valid bounds for RECTANGLE topology
        expected_coordinates = np.clip(expected_coordinates, 0, 9)

        assert np.array_equal(sample_seller.position.tensor_coordinates, expected_coordinates)

    def test_sell_increments_counters(self, sample_seller: Seller) -> None:
        """Test that sell method increments both running and total sales."""
        initial_running = sample_seller.running_sales
        initial_total = sample_seller.total_sales

        sample_seller.sell()

        assert sample_seller.running_sales == initial_running + 1
        assert sample_seller.total_sales == initial_total + 1

    def test_sell_multiple_times(self, sample_seller: Seller) -> None:
        """Test sell method called multiple times."""
        initial_running = sample_seller.running_sales
        initial_total = sample_seller.total_sales

        # Call sell 3 times
        for _ in range(3):
            sample_seller.sell()

        assert sample_seller.running_sales == initial_running + 3
        assert sample_seller.total_sales == initial_total + 3

    def test_sell_after_reset_running_sales(self, sample_seller: Seller) -> None:
        """Test sell method behavior after resetting running sales."""
        # Make some sales
        sample_seller.sell()
        sample_seller.sell()

        initial_total = sample_seller.total_sales

        # Reset running sales
        sample_seller.reset_running_sales()

        # Make another sale
        sample_seller.sell()

        assert sample_seller.running_sales == 1
        assert sample_seller.total_sales == initial_total + 1

    def test_step_reward_basic_calculation(self, sample_seller: Seller, movement_position: Position) -> None:
        """Test basic step reward calculation."""
        # Set up known values
        sample_seller.sell()
        sample_seller.sell()
        sample_seller.move(movement_position)

        production_cost_factor = 10.0
        movement_cost = 2.0

        reward = sample_seller.step_reward(production_cost_factor, movement_cost)

        # Expected calculation:
        # revenue = 2 * 5.0 = 10.0
        # production_cost = 10.0 * 0.8^2 = 10.0 * 0.64 = 6.4
        # movement_cost = 2.0 * 0.22360679774997896 = 0.4472135954999579
        # reward = 10.0 - 6.4 - 0.4472135954999579 = 3.152786404500042
        expected_reward = 3.152786404500042
        assert abs(reward - expected_reward) < 1e-6

    def test_step_reward_no_sales(self, sample_seller: Seller, movement_position: Position) -> None:
        """Test step reward with no sales."""
        # No sales made
        sample_seller.reset_running_sales()
        sample_seller.move(movement_position)

        production_cost_factor = 10.0
        movement_cost = 2.0

        reward = sample_seller.step_reward(production_cost_factor, movement_cost)

        # Expected calculation:
        # revenue = 0 * 5.0 = 0.0
        # production_cost = 10.0 * 0.8^2 = 10.0 * 0.64 = 6.4
        # movement_cost = 2.0 * 0.22360679774997896 = 0.4472135954999579
        # reward = 0.0 - 6.4 - 0.4472135954999579 = -6.847213595499958
        expected_reward = -6.847213595499958
        assert abs(reward - expected_reward) < 1e-6

    def test_step_reward_no_movement(self, sample_seller: Seller) -> None:
        """Test step reward with no movement."""
        sample_seller.sell()
        sample_seller.move(
            Position(
                space_resolution=10,
                topology=Topology.RECTANGLE,
                tensor_coordinates=np.array([0, 0], dtype=np.int32),
            )
        )

        production_cost_factor = 5.0
        movement_cost = 3.0

        reward = sample_seller.step_reward(production_cost_factor, movement_cost)

        # Expected calculation:
        # revenue = 1 * 5.0 = 5.0
        # production_cost = 5.0 * 0.8^2 = 5.0 * 0.64 = 3.2
        # movement_cost = 3.0 * 0.0 = 0.0
        # reward = 5.0 - 3.2 - 0.0 = 1.8
        expected_reward = 1.8
        assert abs(reward - expected_reward) < 1e-6
