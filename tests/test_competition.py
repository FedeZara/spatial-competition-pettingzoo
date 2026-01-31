"""Unit tests for the Competition class."""

from __future__ import annotations

from typing import Any
from unittest.mock import Mock, call, patch

import numpy as np
import pytest

from spatial_competition_pettingzoo.competition import Competition
from spatial_competition_pettingzoo.enums import InformationLevel
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
            "movement_cost": 0.1,
            "include_quality": True,
            "include_buyer_valuation": True,
            "seller_position_distr": mock_dependencies["seller_position_distr"],
            "seller_price_distr": mock_dependencies["seller_price_distr"],
            "seller_quality_distr": mock_dependencies["seller_quality_distr"],
            "new_buyers_per_step": 5,
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
        # Values for buyer spawning (5 buyers: value, quality_taste, distance_factor each)
        buyer_values = [100.0, 1.0, 1.5, 200.0, 0.8, 1.2, 150.0, 1.2, 0.9, 300.0, 0.5, 2.0, 250.0, 1.5, 1.0]
        mock_sample_clip.side_effect = seller_values + buyer_values

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

    @patch("spatial_competition_pettingzoo.competition.CompetitionSpace")
    @patch("spatial_competition_pettingzoo.competition.sample_and_clip_univariate_distribution")
    def test_agent_step(
        self, mock_sample_clip: Mock, mock_space_class: Mock, competition_params: dict[str, Any]
    ) -> None:
        """Test agent_step method."""
        # Setup mocks
        mock_space = Mock()
        mock_space_class.return_value = mock_space
        mock_space.sample_free_position.return_value = Mock()
        mock_space.num_free_cells = 10

        mock_seller = Mock()
        mock_space.sellers_dict = {"seller_0": mock_seller}

        mock_position = Mock()
        mock_position.space_norm.return_value = 0.5  # Within max_step_size

        # Provide enough values for initialization (3 sellers * 2 + 5 buyers * 3 = 21 values)
        mock_sample_clip.return_value = 1.0

        # Create competition
        competition = Competition(**competition_params)

        # Test agent step
        competition.agent_step("seller_0", mock_position, 5.0, 2.5)

        # Verify seller methods were called
        mock_seller.move.assert_called_once_with(mock_position)
        mock_seller.set_price.assert_called_once_with(5.0)
        mock_seller.set_quality.assert_called_once_with(2.5)

    @patch("spatial_competition_pettingzoo.competition.CompetitionSpace")
    @patch("spatial_competition_pettingzoo.competition.sample_and_clip_univariate_distribution")
    def test_agent_step_invalid_price(
        self, mock_sample_clip: Mock, mock_space_class: Mock, competition_params: dict[str, Any]
    ) -> None:
        """Test agent_step with invalid price raises assertion error."""
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
            competition.agent_step("seller_0", mock_position, 15.0, 2.5)

        # Test with negative price
        with pytest.raises(AssertionError):
            competition.agent_step("seller_0", mock_position, -1.0, 2.5)

    @patch("spatial_competition_pettingzoo.competition.CompetitionSpace")
    @patch("spatial_competition_pettingzoo.competition.sample_and_clip_univariate_distribution")
    def test_agent_step_invalid_quality(
        self, mock_sample_clip: Mock, mock_space_class: Mock, competition_params: dict[str, Any]
    ) -> None:
        """Test agent_step with invalid quality raises assertion error."""
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
            competition.agent_step("seller_0", mock_position, 5.0, 10.0)

        # Test with negative quality
        with pytest.raises(AssertionError):
            competition.agent_step("seller_0", mock_position, 5.0, -1.0)

    @patch("spatial_competition_pettingzoo.competition.CompetitionSpace")
    @patch("spatial_competition_pettingzoo.competition.sample_and_clip_univariate_distribution")
    def test_agent_step_invalid_movement(
        self, mock_sample_clip: Mock, mock_space_class: Mock, competition_params: dict[str, Any]
    ) -> None:
        """Test agent_step with invalid movement raises assertion error."""
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
            competition.agent_step("seller_0", mock_position, 5.0, 2.5)

    @patch("spatial_competition_pettingzoo.competition.CompetitionSpace")
    @patch("spatial_competition_pettingzoo.competition.sample_and_clip_univariate_distribution")
    def test_start_cycle(
        self, mock_sample_clip: Mock, mock_space_class: Mock, competition_params: dict[str, Any]
    ) -> None:
        """Test start_cycle method."""
        # Setup mocks
        mock_space = Mock()
        mock_space_class.return_value = mock_space
        mock_space.sample_free_position.return_value = Mock()
        mock_space.num_free_cells = 10
        mock_sample_clip.return_value = 1.0

        mock_buyer1 = Mock()
        mock_buyer2 = Mock()
        mock_buyer3 = Mock()
        mock_space.buyers = [mock_buyer1, mock_buyer2, mock_buyer3]
        mock_buyer1.has_purchased = True
        mock_buyer2.has_purchased = False
        mock_buyer3.has_purchased = True

        # Create competition
        competition = Competition(**competition_params)

        # Test start_cycle
        competition.start_cycle()

        # Verify buyers were spawned
        assert mock_space.add_buyer.call_count == 5
        mock_space.remove_buyer.assert_has_calls([call(mock_buyer1), call(mock_buyer3)])

    @patch("spatial_competition_pettingzoo.competition.CompetitionSpace")
    @patch("spatial_competition_pettingzoo.competition.sample_and_clip_univariate_distribution")
    @patch("spatial_competition_pettingzoo.competition.random.shuffle")
    def test_end_cycle(
        self,
        mock_shuffle: Mock,
        mock_sample_clip: Mock,
        mock_space_class: Mock,
        competition_params: dict[str, Any],
    ) -> None:
        """Test env_step method."""
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
        mock_buyer1.choose_seller_and_buy.return_value = True
        mock_buyer2.choose_seller_and_buy.return_value = False

        # Create a mock list with copy method
        mock_buyers_list = Mock()
        mock_buyers_list.copy.return_value = [mock_buyer1, mock_buyer2]
        mock_space.buyers = mock_buyers_list

        mock_sample_clip.return_value = 1.0

        # Create competition
        competition = Competition(**competition_params)

        # Test env step
        competition.end_cycle()

        # Verify sellers reset their sales
        mock_seller1.reset_running_sales.assert_called_once()
        mock_seller2.reset_running_sales.assert_called_once()

        # Verify buyers were shuffled and processed
        mock_shuffle.assert_called()
        mock_buyers_list.copy.assert_called_once()
        mock_buyer1.choose_seller_and_buy.assert_called_once_with([mock_seller1, mock_seller2])
        mock_buyer2.choose_seller_and_buy.assert_called_once_with([mock_seller1, mock_seller2])

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
        # Need values for: 3 sellers (6 values) + 5 buyers (15 values) = 21 total values
        seller_values = [5.0, 2.5, 4.0, 3.0, 6.0, 1.5]  # price, quality pairs for 3 sellers
        buyer_values = [100.0, 1.0, 1.5, 200.0, 0.8, 1.2, 150.0, 1.2, 0.9, 300.0, 0.5, 2.0, 250.0, 1.5, 1.0]
        mock_sample_clip.side_effect = seller_values + buyer_values

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
            # Create competition
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

        # Setup sellers list for process_sales in env_step
        mock_space.sellers = []
        mock_space.buyers = []

        mock_buyer = Mock()
        mock_buyer_class.return_value = mock_buyer

        # Mock sample values for buyer attributes (value, quality_taste, distance_factor per buyer)
        mock_sample_clip.side_effect = [
            # Initial seller spawning (3 sellers, 2 values each)
            5.0,
            2.5,
            4.0,
            3.0,
            6.0,
            1.5,
            # Initial buyer spawning (5 buyers, 3 values each)
            100.0,
            1.0,
            1.5,
            200.0,
            0.8,
            1.2,
            150.0,
            1.2,
            0.9,
            300.0,
            0.5,
            2.0,
            250.0,
            1.5,
            1.0,
            # Second buyer spawning call (5 more buyers, 3 values each)
            180.0,
            0.9,
            1.3,
            220.0,
            1.1,
            0.8,
            175.0,
            1.3,
            1.1,
            280.0,
            0.7,
            1.8,
            200.0,
            1.4,
            1.2,
        ]

        # Create competition (this will spawn initial buyers)
        competition = Competition(**competition_params)

        # Reset call count to test only the new spawning
        mock_buyer_class.reset_mock()
        mock_space.add_buyer.reset_mock()

        # Test spawning new buyers explicitly
        competition.start_cycle()

        # Verify new buyers were created
        assert mock_buyer_class.call_count == 5
        assert mock_space.add_buyer.call_count == 5

    @patch("spatial_competition_pettingzoo.competition.CompetitionSpace")
    @patch("spatial_competition_pettingzoo.competition.sample_and_clip_univariate_distribution")
    def test_spawn_new_buyers_no_free_cells(
        self, mock_sample_clip: Mock, mock_space_class: Mock, competition_params: dict[str, Any]
    ) -> None:
        """Test _spawn_new_buyers when no free cells are available."""
        # Setup mocks - start with free cells for seller spawning
        mock_space = Mock()
        mock_space_class.return_value = mock_space
        mock_space.sample_free_position.return_value = Mock()
        mock_space.num_free_cells = 10  # Initially allow seller spawning

        mock_sample_clip.return_value = 1.0

        # Create competition (this will spawn sellers successfully)
        competition = Competition(**competition_params)

        # Now set no free cells for buyer spawning test
        mock_space.num_free_cells = 0

        # Reset to count only new buyer spawning
        mock_space.add_buyer.reset_mock()

        # Test spawning new buyers when no space available
        competition._spawn_new_buyers()  # noqa: SLF001 (private method)

        # Verify no new buyers were added
        mock_space.add_buyer.assert_not_called()

    @patch("spatial_competition_pettingzoo.competition.CompetitionSpace")
    @patch("spatial_competition_pettingzoo.competition.sample_and_clip_univariate_distribution")
    def test_process_sales(
        self, mock_sample_clip: Mock, mock_space_class: Mock, competition_params: dict[str, Any]
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

        # Set up buyer purchase decisions
        mock_buyer1.choose_seller_and_buy.return_value = True  # Buys
        mock_buyer2.choose_seller_and_buy.return_value = False  # Doesn't buy
        mock_buyer3.choose_seller_and_buy.return_value = True  # Buys

        # Create a mock list with copy method
        mock_buyers_list = Mock()
        mock_buyers_list.copy.return_value = [mock_buyer1, mock_buyer2, mock_buyer3]
        mock_space.buyers = mock_buyers_list

        mock_sample_clip.return_value = 1.0

        # Create competition
        competition = Competition(**competition_params)

        # Test process sales
        competition._process_sales()  # noqa: SLF001 (private method)

        # Verify all sellers reset their sales
        mock_seller1.reset_running_sales.assert_called_once()
        mock_seller2.reset_running_sales.assert_called_once()

        # Verify buyers list was copied and all buyers were given chance to buy
        mock_buyers_list.copy.assert_called_once()
        mock_buyer1.choose_seller_and_buy.assert_called_once_with([mock_seller1, mock_seller2])
        mock_buyer2.choose_seller_and_buy.assert_called_once_with([mock_seller1, mock_seller2])
        mock_buyer3.choose_seller_and_buy.assert_called_once_with([mock_seller1, mock_seller2])

    @patch("spatial_competition_pettingzoo.competition.CompetitionSpace")
    @patch("spatial_competition_pettingzoo.competition.sample_and_clip_univariate_distribution")
    def test_integration_full_step_cycle(
        self, mock_sample_clip: Mock, mock_space_class: Mock, competition_params: dict[str, Any]
    ) -> None:
        """Test a full step cycle: agent_step -> env_step -> compute_reward -> get_observation."""
        # Setup mocks
        mock_space = Mock()
        mock_space_class.return_value = mock_space
        mock_space.sample_free_position.return_value = Mock()
        mock_space.num_free_cells = 10

        mock_seller = Mock()
        mock_seller.step_reward.return_value = 15.5
        mock_space.sellers_dict = {"seller_0": mock_seller}
        mock_space.sellers = [mock_seller]

        # Setup buyers list that can be shuffled
        mock_buyers_list = Mock()
        mock_buyers_list.copy.return_value = []  # Return empty list for shuffle
        mock_space.buyers = mock_buyers_list

        mock_position = Mock()
        mock_position.space_norm.return_value = 0.8

        mock_sample_clip.return_value = 1.0

        # Create competition
        competition = Competition(**competition_params)

        # Execute full cycle
        competition.agent_step("seller_0", mock_position, 7.5, 3.2)
        competition.end_cycle()
        reward = competition.compute_agent_reward("seller_0")

        # Verify all operations work together
        mock_seller.move.assert_called_once_with(mock_position)
        mock_seller.set_price.assert_called_once_with(7.5)
        mock_seller.set_quality.assert_called_once_with(3.2)
        mock_seller.reset_running_sales.assert_called()
        assert reward == 15.5
