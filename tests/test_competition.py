"""Unit tests for the Competition class."""

from __future__ import annotations

from typing import Any
from unittest.mock import Mock, call, patch

import numpy as np
import pytest

from spatial_competition_pettingzoo.action import Action
from spatial_competition_pettingzoo.competition import Competition
from spatial_competition_pettingzoo.enums import InformationLevel, TransportationCostNorm
from spatial_competition_pettingzoo.position import Position
from spatial_competition_pettingzoo.topology import Topology


class TestCompetition:
    """Test cases for the Competition class."""

    @pytest.fixture
    def mock_dependencies(self) -> dict[str, Any]:
        """Create mock dependencies for Competition."""
        # Mock distributions
        mock_seller_position_distr = Mock()
        mock_seller_price_distr = Mock()
        mock_seller_quality_distr = Mock()
        mock_buyer_position_distr = Mock()
        mock_buyer_valuation_distr = Mock()
        mock_buyer_quality_taste_distr = Mock()
        mock_buyer_distance_factor_distr = Mock()

        # Mock view scope
        mock_view_scope = Mock()

        # Mock RNG
        mock_rng = Mock(spec=np.random.Generator)

        return {
            "seller_position_distr": mock_seller_position_distr,
            "seller_price_distr": mock_seller_price_distr,
            "seller_quality_distr": mock_seller_quality_distr,
            "buyer_position_distr": mock_buyer_position_distr,
            "buyer_valuation_distr": mock_buyer_valuation_distr,
            "buyer_quality_taste_distr": mock_buyer_quality_taste_distr,
            "buyer_distance_factor_distr": mock_buyer_distance_factor_distr,
            "view_scope": mock_view_scope,
            "rng": mock_rng,
        }

    @pytest.fixture
    def competition_params(self, mock_dependencies: dict[str, Any]) -> dict[str, Any]:
        """Default parameters for Competition initialization."""
        return {
            "dimensions": 2,
            "topology": Topology.RECTANGLE,
            "space_resolution": 10,
            "information_level": InformationLevel.COMPLETE,
            "view_scope": mock_dependencies["view_scope"],
            "agent_ids": ["seller_0", "seller_1", "seller_2"],
            "max_price": 10.0,
            "max_quality": 5.0,
            "max_step_size": 1.0,
            "production_cost_factor": 0.5,
            "transportation_cost_norm": TransportationCostNorm.L2,
            "movement_cost": 0.1,
            "include_quality": True,
            "include_buyer_valuation": True,
            "seller_position_distr": mock_dependencies["seller_position_distr"],
            "seller_price_distr": mock_dependencies["seller_price_distr"],
            "seller_quality_distr": mock_dependencies["seller_quality_distr"],
            "new_buyers_per_step": 5,
            "max_buyers": 10,
            "buyer_position_distr": mock_dependencies["buyer_position_distr"],
            "buyer_valuation_distr": mock_dependencies["buyer_valuation_distr"],
            "buyer_quality_taste_distr": mock_dependencies["buyer_quality_taste_distr"],
            "buyer_distance_factor_distr": mock_dependencies["buyer_distance_factor_distr"],
            "rng": mock_dependencies["rng"],
        }

    @patch("spatial_competition_pettingzoo.competition.CompetitionSpace")
    @patch("spatial_competition_pettingzoo.competition.Seller")
    @patch("spatial_competition_pettingzoo.competition.Buyer")
    @patch("spatial_competition_pettingzoo.competition.sample_and_clip_univariate_distribution")
    def test_init(
        self,
        mock_sample_clip: Mock,
        mock_buyer: Mock,
        mock_seller: Mock,
        mock_space_class: Mock,
        competition_params: dict[str, Any],
    ) -> None:
        """Test Competition initialization."""
        # Setup mocks
        mock_space = Mock()
        mock_space_class.return_value = mock_space
        mock_space.sample_free_position.return_value = Mock()
        mock_space.num_free_cells = 10

        # Values for seller spawning (3 sellers: price, quality each)
        seller_values = [5.0, 2.5, 4.0, 3.0, 6.0, 1.5]
        mock_sample_clip.side_effect = seller_values

        mock_seller_instance = Mock()
        mock_seller.return_value = mock_seller_instance

        mock_buyer_instance = Mock()
        mock_buyer.return_value = mock_buyer_instance

        # Create competition
        competition = Competition(**competition_params)

        # Verify space initialization
        mock_space_class.assert_called_once_with(2, Topology.RECTANGLE, 10)

        # Verify attributes are set correctly
        assert competition.dimensions == 2
        assert competition.topology == Topology.RECTANGLE
        assert competition.max_price == 10.0
        assert competition.max_quality == 5.0
        assert competition.information_level == InformationLevel.COMPLETE
        assert competition.view_scope == competition_params["view_scope"]
        assert competition.production_cost_factor == 0.5
        assert competition.movement_cost == 0.1
        assert competition.max_step_size == 1.0
        assert competition.new_buyers_per_step == 5

        # Verify sellers were spawned (3 sellers)
        assert mock_seller.call_count == 3
        assert mock_space.add_seller.call_count == 3

        # Verify render_callback is initialized to None
        assert competition.render_callback is None

    @patch("spatial_competition_pettingzoo.competition.CompetitionSpace")
    @patch("spatial_competition_pettingzoo.competition.sample_and_clip_univariate_distribution")
    def test_agent_step(
        self, mock_sample_clip: Mock, mock_space_class: Mock, competition_params: dict[str, Any]
    ) -> None:
        """Test _agent_step method."""
        # Setup mocks
        mock_space = Mock()
        mock_space_class.return_value = mock_space
        mock_space.sample_free_position.return_value = Mock()
        mock_space.num_free_cells = 10
        mock_space.space_resolution = 10

        mock_seller = Mock()
        mock_seller_position = Mock()
        mock_seller.position = mock_seller_position
        mock_space.sellers_dict = {"seller_0": mock_seller}

        mock_position = Mock()
        mock_position.space_norm.return_value = 0.5  # Within max_step_size

        mock_sample_clip.return_value = 1.0

        # Create competition
        competition = Competition(**competition_params)

        # Test agent step with occupied_positions set
        occupied_positions: set[Position] = set()
        competition._agent_step("seller_0", occupied_positions, mock_position, 5.0, 2.5)

        # Verify seller methods were called
        mock_seller.move.assert_called_once_with(mock_position)
        mock_seller.set_price.assert_called_once_with(5.0)
        mock_seller.set_quality.assert_called_once_with(2.5)

        # Verify position was added to occupied set
        assert mock_seller_position in occupied_positions

    @patch("spatial_competition_pettingzoo.competition.CompetitionSpace")
    @patch("spatial_competition_pettingzoo.competition.sample_and_clip_univariate_distribution")
    def test_agent_step_no_movement(
        self, mock_sample_clip: Mock, mock_space_class: Mock, competition_params: dict[str, Any]
    ) -> None:
        """Test _agent_step with no movement (None or zero movement)."""
        # Setup mocks
        mock_space = Mock()
        mock_space_class.return_value = mock_space
        mock_space.sample_free_position.return_value = Mock()
        mock_space.num_free_cells = 10

        mock_seller = Mock()
        mock_seller_position = Mock()
        mock_seller.position = mock_seller_position
        mock_space.sellers_dict = {"seller_0": mock_seller}

        mock_sample_clip.return_value = 1.0

        # Create competition
        competition = Competition(**competition_params)

        # Test with None movement
        occupied_positions: set[Position] = set()
        competition._agent_step("seller_0", occupied_positions, None, 5.0, 2.5)

        # Verify move was NOT called
        mock_seller.move.assert_not_called()
        mock_seller.set_price.assert_called_once_with(5.0)

        # Test with zero movement
        mock_seller.reset_mock()
        mock_zero_movement = Mock()
        mock_zero_movement.space_norm.return_value = 0.0
        competition._agent_step("seller_0", occupied_positions, mock_zero_movement, 6.0, 3.0)

        # Verify move was NOT called for zero movement
        mock_seller.move.assert_not_called()

    @patch("spatial_competition_pettingzoo.competition.CompetitionSpace")
    @patch("spatial_competition_pettingzoo.competition.sample_and_clip_univariate_distribution")
    def test_agent_step_invalid_price(
        self, mock_sample_clip: Mock, mock_space_class: Mock, competition_params: dict[str, Any]
    ) -> None:
        """Test _agent_step with invalid price raises assertion error."""
        # Setup mocks
        mock_space = Mock()
        mock_space_class.return_value = mock_space
        mock_space.sample_free_position.return_value = Mock()
        mock_space.num_free_cells = 10
        mock_sample_clip.return_value = 1.0

        mock_position = Mock()
        mock_position.space_norm.return_value = 0.5

        # Create competition
        competition = Competition(**competition_params)

        # Test with price > max_price
        with pytest.raises(AssertionError):
            competition._agent_step("seller_0", set(), mock_position, 15.0, 2.5)

        # Test with negative price
        with pytest.raises(AssertionError):
            competition._agent_step("seller_0", set(), mock_position, -1.0, 2.5)

    @patch("spatial_competition_pettingzoo.competition.CompetitionSpace")
    @patch("spatial_competition_pettingzoo.competition.sample_and_clip_univariate_distribution")
    def test_agent_step_invalid_quality(
        self, mock_sample_clip: Mock, mock_space_class: Mock, competition_params: dict[str, Any]
    ) -> None:
        """Test _agent_step with invalid quality raises assertion error."""
        # Setup mocks
        mock_space = Mock()
        mock_space_class.return_value = mock_space
        mock_space.sample_free_position.return_value = Mock()
        mock_space.num_free_cells = 10
        mock_sample_clip.return_value = 1.0

        mock_position = Mock()
        mock_position.space_norm.return_value = 0.5

        # Create competition
        competition = Competition(**competition_params)

        # Test with quality > max_quality
        with pytest.raises(AssertionError):
            competition._agent_step("seller_0", set(), mock_position, 5.0, 10.0)

        # Test with negative quality
        with pytest.raises(AssertionError):
            competition._agent_step("seller_0", set(), mock_position, 5.0, -1.0)

    @patch("spatial_competition_pettingzoo.competition.CompetitionSpace")
    @patch("spatial_competition_pettingzoo.competition.sample_and_clip_univariate_distribution")
    def test_agent_step_invalid_movement(
        self, mock_sample_clip: Mock, mock_space_class: Mock, competition_params: dict[str, Any]
    ) -> None:
        """Test _agent_step with invalid movement raises assertion error."""
        # Setup mocks
        mock_space = Mock()
        mock_space_class.return_value = mock_space
        mock_space.sample_free_position.return_value = Mock()
        mock_space.num_free_cells = 10
        mock_sample_clip.return_value = 1.0

        mock_position = Mock()
        mock_position.space_norm.return_value = 2.0  # Greater than max_step_size

        # Create competition
        competition = Competition(**competition_params)

        # Test with movement > max_step_size
        with pytest.raises(AssertionError):
            competition._agent_step("seller_0", set(), mock_position, 5.0, 2.5)

    @patch("spatial_competition_pettingzoo.competition.CompetitionSpace")
    @patch("spatial_competition_pettingzoo.competition.sample_and_clip_univariate_distribution")
    @patch("spatial_competition_pettingzoo.competition.random.shuffle")
    def test_step(
        self,
        mock_shuffle: Mock,
        mock_sample_clip: Mock,
        mock_space_class: Mock,
        competition_params: dict[str, Any],
    ) -> None:
        """Test step method processes all phases correctly."""
        # Setup mocks
        mock_space = Mock()
        mock_space_class.return_value = mock_space
        mock_space.sample_free_position.return_value = Mock()
        mock_space.num_free_cells = 10
        mock_space.space_resolution = 10

        # Setup sellers
        mock_seller1 = Mock()
        mock_seller2 = Mock()
        mock_seller1.position = Mock()
        mock_seller2.position = Mock()
        mock_space.sellers_dict = {"seller_0": mock_seller1, "seller_1": mock_seller2}
        mock_space.sellers = [mock_seller1, mock_seller2]

        # Setup buyers
        mock_buyer1 = Mock()
        mock_buyer2 = Mock()
        mock_buyer1.has_purchased = True
        mock_buyer2.has_purchased = False
        mock_space.buyers = [mock_buyer1, mock_buyer2]

        mock_sample_clip.return_value = 1.0

        # Create competition
        competition = Competition(**competition_params)

        # Reset mocks after initialization
        mock_space.remove_buyer.reset_mock()
        mock_space.add_buyer.reset_mock()

        # Create actions with Action objects
        mock_movement1 = Mock()
        mock_movement1.space_norm.return_value = 0.0  # Stationary
        mock_movement2 = Mock()
        mock_movement2.space_norm.return_value = 0.5  # Moving

        actions = {
            "seller_0": Action(movement=mock_movement1, price=5.0, quality=2.5),
            "seller_1": Action(movement=mock_movement2, price=6.0, quality=3.0),
        }

        # Execute step
        competition.step(actions)

        # Verify buyers who purchased were removed
        mock_space.remove_buyer.assert_called_once_with(mock_buyer1)

        # Verify seller updates
        mock_seller1.set_price.assert_called_with(5.0)
        mock_seller2.set_price.assert_called_with(6.0)

        # Verify sales were processed
        mock_seller1.reset_running_sales.assert_called()
        mock_seller2.reset_running_sales.assert_called()

    @patch("spatial_competition_pettingzoo.competition.CompetitionSpace")
    @patch("spatial_competition_pettingzoo.competition.sample_and_clip_univariate_distribution")
    def test_step_with_render_callback(
        self, mock_sample_clip: Mock, mock_space_class: Mock, competition_params: dict[str, Any]
    ) -> None:
        """Test step method calls render callback at correct phases."""
        # Setup mocks
        mock_space = Mock()
        mock_space_class.return_value = mock_space
        mock_space.sample_free_position.return_value = Mock()
        mock_space.num_free_cells = 10
        mock_space.space_resolution = 10

        mock_seller = Mock()
        mock_seller.position = Mock()
        mock_space.sellers_dict = {"seller_0": mock_seller}
        mock_space.sellers = [mock_seller]
        mock_space.buyers = []

        mock_sample_clip.return_value = 1.0

        # Create competition
        competition = Competition(**competition_params)

        # Set up render callback
        render_callback = Mock()
        competition.render_callback = render_callback

        # Create action
        mock_movement = Mock()
        mock_movement.space_norm.return_value = 0.0
        actions = {"seller_0": Action(movement=mock_movement, price=5.0, quality=2.5)}

        # Execute step
        competition.step(actions)

        # Verify render callback was called 4 times:
        # 1. After removing buyers who purchased
        # 2. After spawning new buyers
        # 3. After all agents stepped
        # 4. After processing sales
        assert render_callback.call_count == 4

    @patch("spatial_competition_pettingzoo.competition.CompetitionSpace")
    @patch("spatial_competition_pettingzoo.competition.sample_and_clip_univariate_distribution")
    def test_step_stationary_agents_first(
        self, mock_sample_clip: Mock, mock_space_class: Mock, competition_params: dict[str, Any]
    ) -> None:
        """Test step processes stationary agents before moving agents."""
        # Setup mocks
        mock_space = Mock()
        mock_space_class.return_value = mock_space
        mock_space.sample_free_position.return_value = Mock()
        mock_space.num_free_cells = 10
        mock_space.space_resolution = 10

        # Track processing order
        processing_order = []

        mock_seller1 = Mock()
        mock_seller2 = Mock()
        mock_seller1.position = Mock()
        mock_seller2.position = Mock()

        def track_seller1(*args: Any, **kwargs: Any) -> None:
            processing_order.append("seller_0")

        def track_seller2(*args: Any, **kwargs: Any) -> None:
            processing_order.append("seller_1")

        mock_seller1.set_price.side_effect = track_seller1
        mock_seller2.set_price.side_effect = track_seller2

        mock_space.sellers_dict = {"seller_0": mock_seller1, "seller_1": mock_seller2}
        mock_space.sellers = [mock_seller1, mock_seller2]
        mock_space.buyers = []

        mock_sample_clip.return_value = 1.0

        # Create competition
        competition = Competition(**competition_params)

        # seller_0 is stationary, seller_1 is moving
        mock_movement_stationary = Mock()
        mock_movement_stationary.space_norm.return_value = 0.0
        mock_movement_moving = Mock()
        mock_movement_moving.space_norm.return_value = 0.5

        actions = {
            "seller_0": Action(movement=mock_movement_moving, price=5.0, quality=2.5),
            "seller_1": Action(movement=mock_movement_stationary, price=6.0, quality=3.0),
        }

        # Execute step
        competition.step(actions)

        # Verify stationary agent (seller_0) was processed first
        assert processing_order[0] == "seller_1"

    @patch("spatial_competition_pettingzoo.competition.CompetitionSpace")
    @patch("spatial_competition_pettingzoo.competition.sample_and_clip_univariate_distribution")
    def test_compute_agent_reward(
        self, mock_sample_clip: Mock, mock_space_class: Mock, competition_params: dict[str, Any]
    ) -> None:
        """Test compute_agent_reward method."""
        # Setup mocks
        mock_space = Mock()
        mock_space_class.return_value = mock_space
        mock_space.sample_free_position.return_value = Mock()
        mock_space.num_free_cells = 10

        mock_seller = Mock()
        mock_seller.step_reward.return_value = 42.0
        mock_space.sellers_dict = {"seller_0": mock_seller}

        mock_sample_clip.return_value = 1.0

        # Create competition
        competition = Competition(**competition_params)

        # Test reward computation
        reward = competition.compute_agent_reward("seller_0")

        # Verify reward calculation
        mock_seller.step_reward.assert_called_once_with(0.5, 0.1)  # production_cost_factor, movement_cost
        assert reward == 42.0

    @patch("spatial_competition_pettingzoo.competition.CompetitionSpace")
    @patch("spatial_competition_pettingzoo.competition.Observation")
    @patch("spatial_competition_pettingzoo.competition.sample_and_clip_univariate_distribution")
    def test_get_agent_observation(
        self,
        mock_sample_clip: Mock,
        mock_observation_class: Mock,
        mock_space_class: Mock,
        competition_params: dict[str, Any],
    ) -> None:
        """Test get_agent_observation method."""
        # Setup mocks
        mock_space = Mock()
        mock_space_class.return_value = mock_space
        mock_space.sample_free_position.return_value = Mock()
        mock_space.num_free_cells = 10

        mock_observation = Mock()
        mock_observation_class.build_from_competition_space.return_value = mock_observation

        mock_sample_clip.return_value = 1.0

        # Create competition
        competition = Competition(**competition_params)

        # Test observation retrieval
        observation = competition.get_agent_observation("seller_0")

        # Verify observation building
        mock_observation_class.build_from_competition_space.assert_called_once_with(
            space=mock_space,
            information_level=InformationLevel.COMPLETE,
            view_scope=competition_params["view_scope"],
            agent_id="seller_0",
        )
        assert observation == mock_observation

    @patch("spatial_competition_pettingzoo.competition.CompetitionSpace")
    @patch("spatial_competition_pettingzoo.competition.Seller")
    @patch("spatial_competition_pettingzoo.competition.sample_and_clip_univariate_distribution")
    def test_spawn_sellers(
        self,
        mock_sample_clip: Mock,
        mock_seller_class: Mock,
        mock_space_class: Mock,
        competition_params: dict[str, Any],
    ) -> None:
        """Test _spawn_sellers method."""
        # Setup mocks
        mock_space = Mock()
        mock_space_class.return_value = mock_space
        mock_position = Mock()
        mock_space.sample_free_position.return_value = mock_position
        mock_space.num_free_cells = 10

        mock_seller = Mock()
        mock_seller_class.return_value = mock_seller

        # Mock sample_and_clip_univariate_distribution to return different values for price and quality
        seller_values = [5.0, 2.5, 4.0, 3.0, 6.0, 1.5]  # price, quality pairs for 3 sellers
        mock_sample_clip.side_effect = seller_values

        # Create competition (this will call _spawn_sellers internally)
        competition = Competition(**competition_params)

        # Verify competition was created successfully
        assert competition is not None

        # Verify sellers were created correctly
        assert mock_seller_class.call_count == 3
        expected_calls = [
            call(agent_id="seller_0", position=mock_position, price=5.0, quality=2.5),
            call(agent_id="seller_1", position=mock_position, price=4.0, quality=3.0),
            call(agent_id="seller_2", position=mock_position, price=6.0, quality=1.5),
        ]
        mock_seller_class.assert_has_calls(expected_calls)

        # Verify sellers were added to space
        assert mock_space.add_seller.call_count == 3

    @patch("spatial_competition_pettingzoo.competition.CompetitionSpace")
    @patch("spatial_competition_pettingzoo.competition.Seller")
    @patch("spatial_competition_pettingzoo.competition.sample_and_clip_univariate_distribution")
    def test_spawn_sellers_no_free_cells(
        self,
        mock_sample_clip: Mock,
        mock_seller_class: Mock,
        mock_space_class: Mock,
        competition_params: dict[str, Any],
    ) -> None:
        """Test _spawn_sellers when no free cells are available."""
        # Setup mocks
        mock_space = Mock()
        mock_space_class.return_value = mock_space
        mock_space.sample_free_position.return_value = Mock()
        mock_space.num_free_cells = 0

        mock_sample_clip.return_value = 1.0

        # Test spawning sellers when no free cells are available
        with pytest.raises(ValueError, match="No free cells available to spawn sellers\\."):
            Competition(**competition_params)

        # Verify no sellers were added
        assert mock_seller_class.call_count == 0
        assert mock_space.add_seller.call_count == 0

    @patch("spatial_competition_pettingzoo.competition.CompetitionSpace")
    @patch("spatial_competition_pettingzoo.competition.Buyer")
    @patch("spatial_competition_pettingzoo.competition.sample_and_clip_univariate_distribution")
    def test_spawn_new_buyers(
        self,
        mock_sample_clip: Mock,
        mock_buyer_class: Mock,
        mock_space_class: Mock,
        competition_params: dict[str, Any],
    ) -> None:
        """Test _spawn_new_buyers method."""
        # Setup mocks
        mock_space = Mock()
        mock_space_class.return_value = mock_space
        mock_position = Mock()
        mock_space.sample_free_position.return_value = mock_position
        mock_space.num_free_cells = 10
        mock_space.sellers = []
        mock_space.buyers = []

        mock_buyer = Mock()
        mock_buyer_class.return_value = mock_buyer

        mock_sample_clip.return_value = 1.0

        # Create competition
        competition = Competition(**competition_params)

        # Reset mocks
        mock_buyer_class.reset_mock()
        mock_space.add_buyer.reset_mock()

        # Test spawning new buyers
        competition._spawn_new_buyers()

        # Verify new buyers were created (5 = new_buyers_per_step)
        assert mock_buyer_class.call_count == 5
        assert mock_space.add_buyer.call_count == 5

    @patch("spatial_competition_pettingzoo.competition.CompetitionSpace")
    @patch("spatial_competition_pettingzoo.competition.sample_and_clip_univariate_distribution")
    def test_spawn_new_buyers_no_free_cells(
        self, mock_sample_clip: Mock, mock_space_class: Mock, competition_params: dict[str, Any]
    ) -> None:
        """Test _spawn_new_buyers when no free cells are available."""
        # Setup mocks
        mock_space = Mock()
        mock_space_class.return_value = mock_space
        mock_space.sample_free_position.return_value = Mock()
        mock_space.num_free_cells = 10  # Initially allow seller spawning

        mock_sample_clip.return_value = 1.0

        # Create competition
        competition = Competition(**competition_params)

        # Now set no free cells for buyer spawning test
        mock_space.num_free_cells = 0
        mock_space.add_buyer.reset_mock()

        # Test spawning new buyers when no space available
        competition._spawn_new_buyers()

        # Verify no new buyers were added
        mock_space.add_buyer.assert_not_called()

    @patch("spatial_competition_pettingzoo.competition.CompetitionSpace")
    @patch("spatial_competition_pettingzoo.competition.sample_and_clip_univariate_distribution")
    def test_spawn_new_buyers_max_buyers_reached(
        self, mock_sample_clip: Mock, mock_space_class: Mock, competition_params: dict[str, Any]
    ) -> None:
        """Test _spawn_new_buyers stops when max_buyers is reached."""
        # Setup mocks
        mock_space = Mock()
        mock_space_class.return_value = mock_space
        mock_space.sample_free_position.return_value = Mock()
        mock_space.num_free_cells = 100

        mock_sample_clip.return_value = 1.0

        # Create competition
        competition = Competition(**competition_params)

        # Set buyers at max capacity (max_buyers = 10)
        mock_space.buyers = [Mock() for _ in range(10)]
        mock_space.add_buyer.reset_mock()

        # Test spawning new buyers
        competition._spawn_new_buyers()

        # Verify no new buyers were added
        mock_space.add_buyer.assert_not_called()

    @patch("spatial_competition_pettingzoo.competition.CompetitionSpace")
    @patch("spatial_competition_pettingzoo.competition.sample_and_clip_univariate_distribution")
    @patch("spatial_competition_pettingzoo.competition.random.shuffle")
    def test_process_sales(
        self,
        mock_shuffle: Mock,
        mock_sample_clip: Mock,
        mock_space_class: Mock,
        competition_params: dict[str, Any],
    ) -> None:
        """Test _process_sales method."""
        # Setup mocks
        mock_space = Mock()
        mock_space_class.return_value = mock_space
        mock_space.sample_free_position.return_value = Mock()
        mock_space.num_free_cells = 10

        mock_seller1 = Mock()
        mock_seller2 = Mock()
        mock_space.sellers = [mock_seller1, mock_seller2]

        mock_buyer1 = Mock()
        mock_buyer2 = Mock()
        mock_buyer3 = Mock()
        mock_buyer1.choose_seller_and_buy.return_value = True
        mock_buyer2.choose_seller_and_buy.return_value = False
        mock_buyer3.choose_seller_and_buy.return_value = True

        mock_space.buyers = [mock_buyer1, mock_buyer2, mock_buyer3]

        mock_sample_clip.return_value = 1.0

        # Create competition
        competition = Competition(**competition_params)

        # Test process sales
        competition._process_sales()

        # Verify all sellers reset their sales
        mock_seller1.reset_running_sales.assert_called_once()
        mock_seller2.reset_running_sales.assert_called_once()

        # Verify shuffle was called
        mock_shuffle.assert_called()

        # Verify all buyers were given chance to buy
        mock_buyer1.choose_seller_and_buy.assert_called_once_with([mock_seller1, mock_seller2])
        mock_buyer2.choose_seller_and_buy.assert_called_once_with([mock_seller1, mock_seller2])
        mock_buyer3.choose_seller_and_buy.assert_called_once_with([mock_seller1, mock_seller2])

    @patch("spatial_competition_pettingzoo.competition.CompetitionSpace")
    @patch("spatial_competition_pettingzoo.competition.sample_and_clip_univariate_distribution")
    def test_remove_buyers_who_purchased(
        self, mock_sample_clip: Mock, mock_space_class: Mock, competition_params: dict[str, Any]
    ) -> None:
        """Test _remove_buyers_who_purchased method."""
        # Setup mocks
        mock_space = Mock()
        mock_space_class.return_value = mock_space
        mock_space.sample_free_position.return_value = Mock()
        mock_space.num_free_cells = 10

        mock_buyer1 = Mock()
        mock_buyer2 = Mock()
        mock_buyer3 = Mock()
        mock_buyer1.has_purchased = True
        mock_buyer2.has_purchased = False
        mock_buyer3.has_purchased = True
        mock_space.buyers = [mock_buyer1, mock_buyer2, mock_buyer3]

        mock_sample_clip.return_value = 1.0

        # Create competition
        competition = Competition(**competition_params)

        # Test removing buyers
        competition._remove_buyers_who_purchased()

        # Verify only buyers who purchased were removed
        mock_space.remove_buyer.assert_has_calls([call(mock_buyer1), call(mock_buyer3)], any_order=True)
        assert mock_space.remove_buyer.call_count == 2

    @patch("spatial_competition_pettingzoo.competition.CompetitionSpace")
    @patch("spatial_competition_pettingzoo.competition.sample_and_clip_univariate_distribution")
    @patch("spatial_competition_pettingzoo.competition.random.shuffle")
    def test_integration_full_step(
        self,
        mock_shuffle: Mock,
        mock_sample_clip: Mock,
        mock_space_class: Mock,
        competition_params: dict[str, Any],
    ) -> None:
        """Test a full step: all phases from step to reward computation."""
        # Setup mocks
        mock_space = Mock()
        mock_space_class.return_value = mock_space
        mock_space.sample_free_position.return_value = Mock()
        mock_space.num_free_cells = 10
        mock_space.space_resolution = 10

        mock_seller = Mock()
        mock_seller.step_reward.return_value = 15.5
        mock_seller.position = Mock()
        mock_space.sellers_dict = {"seller_0": mock_seller}
        mock_space.sellers = [mock_seller]
        mock_space.buyers = []

        mock_sample_clip.return_value = 1.0

        # Create competition
        competition = Competition(**competition_params)

        # Create action
        mock_movement = Mock()
        mock_movement.space_norm.return_value = 0.8
        actions = {"seller_0": Action(movement=mock_movement, price=7.5, quality=3.2)}

        # Execute full step
        competition.step(actions)
        reward = competition.compute_agent_reward("seller_0")

        # Verify all operations work together
        mock_seller.move.assert_called_once_with(mock_movement)
        mock_seller.set_price.assert_called_once_with(7.5)
        mock_seller.set_quality.assert_called_once_with(3.2)
        mock_seller.reset_running_sales.assert_called()
        assert reward == 15.5


class TestCollisionResolution:
    """Test cases for collision resolution in the Competition class."""

    @pytest.fixture
    def mock_dependencies(self) -> dict[str, Any]:
        """Create mock dependencies for Competition."""
        mock_seller_position_distr = Mock()
        mock_seller_price_distr = Mock()
        mock_seller_quality_distr = Mock()
        mock_buyer_position_distr = Mock()
        mock_buyer_valuation_distr = Mock()
        mock_buyer_quality_taste_distr = Mock()
        mock_buyer_distance_factor_distr = Mock()
        mock_view_scope = Mock()
        mock_rng = Mock(spec=np.random.Generator)

        return {
            "seller_position_distr": mock_seller_position_distr,
            "seller_price_distr": mock_seller_price_distr,
            "seller_quality_distr": mock_seller_quality_distr,
            "buyer_position_distr": mock_buyer_position_distr,
            "buyer_valuation_distr": mock_buyer_valuation_distr,
            "buyer_quality_taste_distr": mock_buyer_quality_taste_distr,
            "buyer_distance_factor_distr": mock_buyer_distance_factor_distr,
            "view_scope": mock_view_scope,
            "rng": mock_rng,
        }

    @pytest.fixture
    def competition_params(self, mock_dependencies: dict[str, Any]) -> dict[str, Any]:
        """Default parameters for Competition initialization."""
        return {
            "dimensions": 2,
            "topology": Topology.RECTANGLE,
            "space_resolution": 10,
            "information_level": InformationLevel.COMPLETE,
            "view_scope": mock_dependencies["view_scope"],
            "agent_ids": ["seller_0", "seller_1"],
            "max_price": 10.0,
            "max_quality": 5.0,
            "max_step_size": 1.0,
            "production_cost_factor": 0.5,
            "transportation_cost_norm": TransportationCostNorm.L2,
            "movement_cost": 0.1,
            "include_quality": True,
            "include_buyer_valuation": True,
            "seller_position_distr": mock_dependencies["seller_position_distr"],
            "seller_price_distr": mock_dependencies["seller_price_distr"],
            "seller_quality_distr": mock_dependencies["seller_quality_distr"],
            "new_buyers_per_step": 0,
            "max_buyers": 10,
            "buyer_position_distr": mock_dependencies["buyer_position_distr"],
            "buyer_valuation_distr": mock_dependencies["buyer_valuation_distr"],
            "buyer_quality_taste_distr": mock_dependencies["buyer_quality_taste_distr"],
            "buyer_distance_factor_distr": mock_dependencies["buyer_distance_factor_distr"],
            "rng": mock_dependencies["rng"],
        }

    @patch("spatial_competition_pettingzoo.competition.CompetitionSpace")
    @patch("spatial_competition_pettingzoo.competition.sample_and_clip_univariate_distribution")
    def test_fix_collisions_no_collision(
        self, mock_sample_clip: Mock, mock_space_class: Mock, competition_params: dict[str, Any]
    ) -> None:
        """Test _fix_collisions does nothing when position is not occupied."""
        # Setup mocks
        mock_space = Mock()
        mock_space_class.return_value = mock_space
        mock_space.sample_free_position.return_value = Mock()
        mock_space.num_free_cells = 10
        mock_space.space_resolution = 10

        mock_sample_clip.return_value = 1.0

        # Create competition
        competition = Competition(**competition_params)

        # Create a seller with a position not in occupied_positions
        mock_seller = Mock()
        seller_position = Position(
            tensor_coordinates=np.array([5, 5], dtype=np.int32),
            space_resolution=10,
            topology=Topology.RECTANGLE,
        )
        mock_seller.position = seller_position

        # Occupied positions do not include seller's position
        occupied_positions: set[Position] = {
            Position(
                tensor_coordinates=np.array([3, 3], dtype=np.int32),
                space_resolution=10,
                topology=Topology.RECTANGLE,
            )
        }

        # Call _fix_collisions - should not raise and not modify anything
        competition._fix_collisions(mock_seller, occupied_positions)

        # Verify RNG was not called (no collision resolution needed)
        competition_params["rng"].integers.assert_not_called()

    @patch("spatial_competition_pettingzoo.competition.CompetitionSpace")
    @patch("spatial_competition_pettingzoo.competition.sample_and_clip_univariate_distribution")
    def test_fix_collisions_finds_free_position(
        self, mock_sample_clip: Mock, mock_space_class: Mock, competition_params: dict[str, Any]
    ) -> None:
        """Test _fix_collisions finds a free position when collision occurs."""
        # Setup mocks
        mock_space = Mock()
        mock_space_class.return_value = mock_space
        mock_space.sample_free_position.return_value = Mock()
        mock_space.num_free_cells = 10
        mock_space.space_resolution = 10

        mock_sample_clip.return_value = 1.0

        # Create competition
        competition = Competition(**competition_params)

        # Create a seller with a position that IS in occupied_positions
        seller_position = Position(
            tensor_coordinates=np.array([5, 5], dtype=np.int32),
            space_resolution=10,
            topology=Topology.RECTANGLE,
        )
        mock_seller = Mock()
        mock_seller.position = seller_position

        # Occupied positions include seller's current position (collision!)
        occupied_positions: set[Position] = {seller_position}

        # Mock RNG to return an offset that leads to a free position (offset [1, 0] -> position [6, 5])
        competition_params["rng"].integers.return_value = np.array([1, 0], dtype=np.int32)

        # Call _fix_collisions - should find a new position and call seller.move
        competition._fix_collisions(mock_seller, occupied_positions)

        # Verify RNG was called to generate offset
        competition_params["rng"].integers.assert_called()

        # Verify seller.move was called with the offset movement
        assert mock_seller.position == Position(
            tensor_coordinates=np.array([6, 5], dtype=np.int32),
            space_resolution=10,
            topology=Topology.RECTANGLE,
        )

    @patch("spatial_competition_pettingzoo.competition.CompetitionSpace")
    @patch("spatial_competition_pettingzoo.competition.sample_and_clip_univariate_distribution")
    def test_fix_collisions_max_attempts_reached(
        self, mock_sample_clip: Mock, mock_space_class: Mock, competition_params: dict[str, Any]
    ) -> None:
        """Test _fix_collisions raises ValueError when max attempts reached."""
        # Setup mocks
        mock_space = Mock()
        mock_space_class.return_value = mock_space
        mock_space.sample_free_position.return_value = Mock()
        mock_space.num_free_cells = 10
        mock_space.space_resolution = 10

        mock_sample_clip.return_value = 1.0

        # Create competition
        competition = Competition(**competition_params)

        # Create a seller with a real Position (needed for arithmetic operations)
        seller_position = Position(
            tensor_coordinates=np.array([5, 5], dtype=np.int32),
            space_resolution=10,
            topology=Topology.RECTANGLE,
        )
        mock_seller = Mock()
        mock_seller.position = seller_position

        # Create a situation where every position is "occupied"
        # by making __contains__ always return True
        class AlwaysOccupied:
            def __contains__(self, item: Any) -> bool:
                return True

        always_occupied = AlwaysOccupied()

        # Mock RNG to return consistent offsets
        competition_params["rng"].integers.return_value = np.array([1, 1], dtype=np.int32)

        # Call _fix_collisions with max_attempts=3 - should raise ValueError
        with pytest.raises(ValueError, match="Max attempts reached"):
            competition._fix_collisions(mock_seller, always_occupied, max_attempts=3)  # type: ignore[arg-type]

    @patch("spatial_competition_pettingzoo.competition.CompetitionSpace")
    @patch("spatial_competition_pettingzoo.competition.sample_and_clip_univariate_distribution")
    def test_step_collision_resolution_stationary_first(
        self, mock_sample_clip: Mock, mock_space_class: Mock, competition_params: dict[str, Any]
    ) -> None:
        """Test that stationary agents claim positions before moving agents."""
        # Setup mocks
        mock_space = Mock()
        mock_space_class.return_value = mock_space
        mock_space.num_free_cells = 10
        mock_space.space_resolution = 10

        # Create real Position objects for sellers
        position_0 = Position(
            tensor_coordinates=np.array([5, 5], dtype=np.int32),
            space_resolution=10,
            topology=Topology.RECTANGLE,
        )
        position_1 = Position(
            tensor_coordinates=np.array([3, 3], dtype=np.int32),
            space_resolution=10,
            topology=Topology.RECTANGLE,
        )

        mock_space.sample_free_position.side_effect = [position_0, position_1]

        mock_seller_0 = Mock()
        mock_seller_0.position = position_0
        mock_seller_1 = Mock()
        mock_seller_1.position = position_1

        mock_space.sellers_dict = {"seller_0": mock_seller_0, "seller_1": mock_seller_1}
        mock_space.sellers = [mock_seller_0, mock_seller_1]
        mock_space.buyers = []

        mock_sample_clip.return_value = 1.0

        # Create competition
        competition = Competition(**competition_params)

        # seller_0 stays in place (stationary), seller_1 tries to move
        mock_movement_stationary = Mock()
        mock_movement_stationary.space_norm.return_value = 0.0
        mock_movement_moving = Mock()
        mock_movement_moving.space_norm.return_value = 0.5

        actions = {
            "seller_0": Action(movement=mock_movement_stationary, price=5.0, quality=2.5),
            "seller_1": Action(movement=mock_movement_moving, price=6.0, quality=3.0),
        }

        # Execute step
        competition.step(actions)

        # Verify stationary seller_0 did NOT call move (no movement)
        mock_seller_0.move.assert_not_called()

        # Verify moving seller_1 DID call move
        mock_seller_1.move.assert_called_once_with(mock_movement_moving)

    @patch("spatial_competition_pettingzoo.competition.CompetitionSpace")
    @patch("spatial_competition_pettingzoo.competition.sample_and_clip_univariate_distribution")
    def test_step_moving_agents_shuffled(
        self, mock_sample_clip: Mock, mock_space_class: Mock, competition_params: dict[str, Any]
    ) -> None:
        """Test that moving agents are processed in random order for fairness."""
        # Setup mocks
        mock_space = Mock()
        mock_space_class.return_value = mock_space
        mock_space.num_free_cells = 10
        mock_space.space_resolution = 10

        position_0 = Position(
            tensor_coordinates=np.array([2, 2], dtype=np.int32),
            space_resolution=10,
            topology=Topology.RECTANGLE,
        )
        position_1 = Position(
            tensor_coordinates=np.array([8, 8], dtype=np.int32),
            space_resolution=10,
            topology=Topology.RECTANGLE,
        )

        mock_space.sample_free_position.side_effect = [position_0, position_1]

        mock_seller_0 = Mock()
        mock_seller_0.position = position_0
        mock_seller_1 = Mock()
        mock_seller_1.position = position_1

        mock_space.sellers_dict = {"seller_0": mock_seller_0, "seller_1": mock_seller_1}
        mock_space.sellers = [mock_seller_0, mock_seller_1]
        mock_space.buyers = []

        mock_sample_clip.return_value = 1.0

        # Create competition
        competition = Competition(**competition_params)

        # Both sellers are moving
        mock_movement = Mock()
        mock_movement.space_norm.return_value = 0.5

        actions = {
            "seller_0": Action(movement=mock_movement, price=5.0, quality=2.5),
            "seller_1": Action(movement=mock_movement, price=6.0, quality=3.0),
        }

        # Execute step
        competition.step(actions)

        # Verify rng.shuffle was called with moving agents list
        competition_params["rng"].shuffle.assert_called()
