from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from spatial_competition_pettingzoo.buyer import Buyer
from spatial_competition_pettingzoo.position import Position
from spatial_competition_pettingzoo.seller import Seller
from spatial_competition_pettingzoo.topology import Topology


class TestBuyer:
    @pytest.fixture
    def topology(self) -> Topology:
        return Topology.RECTANGLE

    @pytest.fixture
    def space_resolution(self) -> int:
        return 10

    @pytest.fixture
    def buyer_position(self, space_resolution: int, topology: Topology) -> Position:
        return Position(
            tensor_coordinates=np.array([2, 3], dtype=np.int32),
            space_resolution=space_resolution,
            topology=topology,
        )

    @pytest.fixture
    def buyer(self, buyer_position: Position) -> Buyer:
        return Buyer(
            position=buyer_position,
            value=10.0,
            quality_taste=2.0,
            distance_factor=1.5,
        )

    @pytest.fixture
    def seller_position(self, space_resolution: int, topology: Topology) -> Position:
        return Position(
            tensor_coordinates=np.array([5, 7], dtype=np.int32),
            space_resolution=space_resolution,
            topology=topology,
        )

    @pytest.fixture
    def seller(self, seller_position: Position) -> Seller:
        return Seller(
            agent_id="seller_0",
            position=seller_position,
            price=3.0,
            quality=2.5,
        )

    @pytest.mark.parametrize(
        ("value", "distance_factor", "distance", "quality_taste", "quality", "expected_value"),
        [
            (10.0, 1.0, 2.0, 2.0, 0.8, 9.6),
            (10.0, 0.7, 2.0, 2.0, 0.8, 10.2),
            (10.0, 1.0, 0.0, 2.0, 0.8, 11.6),
        ],
    )
    def test_value_for_seller(
        self,
        value: float,
        distance_factor: float,
        distance: float,
        quality_taste: float,
        quality: float,
        expected_value: float,
    ) -> None:
        """Test the basic value calculation formula."""
        # Formula is value - distance_factor * distance + quality_taste * quality
        buyer = Buyer(
            position=Position(
                tensor_coordinates=np.array([0, 0], dtype=np.int32),
                space_resolution=10,
                topology=Topology.RECTANGLE,
            ),
            value=value,
            distance_factor=distance_factor,
            quality_taste=quality_taste,
        )
        seller = MagicMock()
        seller.quality = quality
        with (
            patch("spatial_competition_pettingzoo.buyer.Position.distance", return_value=distance) as mock_distance,
        ):
            result = buyer.value_for_seller(seller)
            assert abs(result - expected_value) < 1e-6
            mock_distance.assert_called_once_with(seller.position, ord=2)

    @pytest.mark.parametrize(
        ("value_for_seller", "price", "expected_reward"),
        [
            (9.6, 3.0, 6.6),
            (10.2, 3.0, 7.2),
            (11.6, 3.0, 8.6),
        ],
    )
    def test_reward_for_seller(self, value_for_seller: float, price: float, expected_reward: float) -> None:
        """Test the basic reward calculation formula."""
        # Formula is value_for_seller - price
        buyer = Buyer(
            position=Position(
                tensor_coordinates=np.array([0, 0], dtype=np.int32),
                space_resolution=10,
                topology=Topology.RECTANGLE,
            ),
            value=10.0,
            quality_taste=2.0,
            distance_factor=0.1,
        )
        seller = MagicMock()
        seller.price = price
        with patch.object(buyer, "value_for_seller", return_value=value_for_seller) as mock_value_for_seller:
            result = buyer.reward_for_seller(seller)
            assert abs(result - expected_reward) < 1e-6
            mock_value_for_seller.assert_called_once_with(seller)

    def test_choose_seller_and_buy_with_positive_reward(self, buyer_position: Position) -> None:
        """Test that buyer chooses to buy when there's a positive reward."""
        buyer = Buyer(
            position=buyer_position,
            value=10.0,
            quality_taste=2.0,
            distance_factor=0.1,
        )

        seller = Seller(
            agent_id="seller_0",
            position=buyer_position,
            price=2.0,
            quality=3.0,
        )

        # Mock the sell method to track if it was called
        with patch.object(seller, "sell") as mock_sell:
            result = buyer.choose_seller_and_buy([seller])

            assert result is True
            mock_sell.assert_called_once()

    def test_choose_seller_and_buy_with_negative_reward(self, buyer_position: Position) -> None:
        """Test that buyer doesn't buy when all rewards are negative."""
        buyer = Buyer(
            position=buyer_position,
            value=1.0,
            quality_taste=0.1,
            distance_factor=5.0,
        )

        seller = Seller(
            agent_id="seller_0",
            position=buyer_position,
            price=10.0,
            quality=0.5,
        )

        # Mock the sell method to track if it was called
        with patch.object(seller, "sell") as mock_sell:
            result = buyer.choose_seller_and_buy([seller])

            assert result is False
            mock_sell.assert_not_called()

    def test_choose_seller_and_buy_selects_best_seller(self, buyer_position: Position) -> None:
        """Test that buyer selects the seller with the highest reward."""
        buyer = Buyer(
            position=buyer_position,
            value=10.0,
            quality_taste=2.0,
            distance_factor=0.1,
        )

        # Create sellers with different rewards
        good_seller = Seller(
            agent_id="seller_0",
            position=buyer_position,
            price=2.0,
            quality=3.0,
        )

        bad_seller = Seller(
            agent_id="seller_1",
            position=buyer_position,
            price=8.0,
            quality=1.0,
        )

        # Mock the sell methods
        with patch.object(good_seller, "sell") as mock_good_sell, patch.object(bad_seller, "sell") as mock_bad_sell:
            result = buyer.choose_seller_and_buy([good_seller, bad_seller])

            assert result is True
            mock_good_sell.assert_called_once()
            mock_bad_sell.assert_not_called()

    def test_choose_seller_and_buy_with_empty_list(self, buyer: Buyer) -> None:
        """Test behavior when given an empty list of sellers."""
        result = buyer.choose_seller_and_buy([])
        assert result is False

    def test_choose_seller_and_buy_doesnt_modify_original_list(self, buyer_position: Position) -> None:
        """Test that the original sellers list is not modified."""
        buyer = Buyer(
            position=buyer_position,
            value=10.0,
            quality_taste=2.0,
            distance_factor=0.1,
        )

        seller1 = Seller(agent_id="seller_0", position=buyer_position, price=2.0, quality=3.0)
        seller2 = Seller(agent_id="seller_1", position=buyer_position, price=3.0, quality=2.0)

        original_sellers = [seller1, seller2]
        original_order = original_sellers.copy()

        with patch.object(seller1, "sell"), patch.object(seller2, "sell"):
            buyer.choose_seller_and_buy(original_sellers)

            # Check that original list order is preserved
            assert original_sellers == original_order

    @patch("random.shuffle")
    def test_choose_seller_and_buy_shuffles_sellers(self, mock_shuffle: Mock, buyer_position: Position) -> None:
        """Test that sellers list is shuffled before selection."""
        buyer = Buyer(
            position=buyer_position,
            value=10.0,
            quality_taste=2.0,
            distance_factor=0.1,
        )

        seller = Seller(
            agent_id="seller_0",
            position=buyer_position,
            price=2.0,
            quality=3.0,
        )
        with patch.object(seller, "sell"):
            buyer.choose_seller_and_buy([seller])

            # Check that shuffle was called
            mock_shuffle.assert_called_once()

    def test_choose_seller_and_buy_with_multiple_sellers_same_reward(self, buyer_position: Position) -> None:
        """Test behavior when multiple sellers have the same reward."""
        buyer = Buyer(
            position=buyer_position,
            value=10.0,
            quality_taste=2.0,
            distance_factor=0.1,
        )

        # Create sellers with identical rewards
        seller1 = Seller(agent_id="seller_0", position=buyer_position, price=5.0, quality=2.5)
        seller2 = Seller(agent_id="seller_1", position=buyer_position, price=5.0, quality=2.5)

        with patch.object(seller1, "sell") as mock_sell1, patch.object(seller2, "sell") as mock_sell2:
            result = buyer.choose_seller_and_buy([seller1, seller2])

            assert result is True
            # One of them should be called, but we can't predict which due to shuffling
            assert mock_sell1.called or mock_sell2.called
            # But not both
            assert not (mock_sell1.called and mock_sell2.called)

    def test_choose_seller_and_buy_with_zero_reward_threshold(self, buyer_position: Position) -> None:
        """Test that buyer doesn't buy when reward is exactly zero."""
        buyer = Buyer(
            position=buyer_position,
            value=5.0,
            quality_taste=1.0,
            distance_factor=1.0,
        )

        # Create seller where reward is exactly 0
        seller = Seller(
            agent_id="seller_0",
            position=buyer_position,
            price=6.0,  # value (5.0) + quality_taste (1.0) * quality (1.0) = 6.0
            quality=1.0,
        )

        with patch.object(seller, "sell") as mock_sell:
            result = buyer.choose_seller_and_buy([seller])

            assert result is False
            mock_sell.assert_not_called()
