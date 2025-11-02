"""Tests for the Observation class."""

from unittest.mock import patch

import numpy as np
import pytest
from gymnasium import spaces

from spatial_competition_pettingzoo.buyer import Buyer
from spatial_competition_pettingzoo.competition_space import CompetitionSpace
from spatial_competition_pettingzoo.enums import InformationLevel
from spatial_competition_pettingzoo.observation import Observation
from spatial_competition_pettingzoo.position import Position
from spatial_competition_pettingzoo.seller import Seller
from spatial_competition_pettingzoo.topology import Topology
from spatial_competition_pettingzoo.view_scope import LimitedViewScope


class TestObservation:
    """Test class for Observation functionality."""

    @pytest.fixture
    def sample_position(self) -> Position:
        """Create a sample position for testing."""
        return Position(
            space_resolution=10,
            topology=Topology.TORUS,
            tensor_coordinates=np.array([5, 5]),
        )

    @pytest.fixture
    def sample_arrays(self) -> dict[str, np.ndarray]:
        """Create sample arrays for observation components."""
        return {
            "local_view": np.array([[0, 1], [2, 3]], dtype=np.int8),
            "buyers": np.array([[10.5, 15.2], [8.7, 12.3]], dtype=np.float32),
            "sellers_price": np.array([[5.0, 7.5], [6.2, 9.1]], dtype=np.float32),
            "sellers_quality": np.array([[0.8, 0.9], [0.7, 0.85]], dtype=np.float32),
        }

    @pytest.fixture
    def sample_competition_space(self) -> CompetitionSpace:
        """Create a sample competition space for testing."""
        # Create positions first
        seller1_position = Position(
            space_resolution=10,
            topology=Topology.TORUS,
            tensor_coordinates=np.array([1, 2]),
        )

        seller2_position = Position(
            space_resolution=10,
            topology=Topology.TORUS,
            tensor_coordinates=np.array([5, 6]),
        )

        # Create sellers
        seller1 = Seller(idx=1, position=seller1_position, price=5.0, quality=0.8)
        seller2 = Seller(idx=2, position=seller2_position, price=7.5, quality=0.9)

        # Create positions for buyers
        buyer1_position = Position(
            space_resolution=10,
            topology=Topology.TORUS,
            tensor_coordinates=np.array([2, 3]),
        )

        buyer2_position = Position(
            space_resolution=10,
            topology=Topology.TORUS,
            tensor_coordinates=np.array([9, 3]),
        )

        # Create buyers
        buyer1 = Buyer(position=buyer1_position, value=15.0, quality_taste=0.7, distance_factor=0.3)
        buyer2 = Buyer(position=buyer2_position, value=12.0, quality_taste=0.8, distance_factor=0.2)

        return CompetitionSpace(
            dimensions=2,
            topology=Topology.TORUS,
            space_resolution=10,
            sellers={"seller_1": seller1, "seller_2": seller2},
            buyers=[buyer1, buyer2],
        )

    def test_get_observation(self, sample_position: Position, sample_arrays: dict[str, np.ndarray]) -> None:
        """Test get_observation method returns correct dictionary."""
        observation = Observation(
            own_price=5.0,
            own_quality=0.8,
            own_position=sample_position,
            local_view=sample_arrays["local_view"],
            buyers=sample_arrays["buyers"],
            sellers_price=sample_arrays["sellers_price"],
            sellers_quality=sample_arrays["sellers_quality"],
        )

        obs_dict = observation.get_observation()

        assert obs_dict["own_price"] == 5.0
        assert obs_dict["own_quality"] == 0.8
        assert obs_dict["own_position"] == sample_position
        assert np.array_equal(obs_dict["local_view"], sample_arrays["local_view"])
        assert np.array_equal(obs_dict["buyers"], sample_arrays["buyers"])
        assert np.array_equal(obs_dict["sellers_price"], sample_arrays["sellers_price"])
        assert np.array_equal(obs_dict["sellers_quality"], sample_arrays["sellers_quality"])

    def test_create_buyers_space(self) -> None:
        """Test create_buyers_space static method."""
        buyers_space = Observation.create_buyers_space(
            view_scope=LimitedViewScope(5),
            space_resolution=10,
            dimensions=2,
            max_valuation=20.0,
            max_quality=1.0,
        )

        assert isinstance(buyers_space, spaces.Box)
        assert np.all(buyers_space.low == Observation.NO_BUYER_PLACEHOLDER)
        assert np.all(buyers_space.high == 21.0)  # max_valuation + max_quality
        assert buyers_space.shape == (11, 11)  # (2 * 5 + 1) for each dimension
        assert buyers_space.dtype == np.float32

    def test_create_sellers_spaces(self) -> None:
        """Test create_sellers_spaces static method."""
        price_space, quality_space = Observation.create_sellers_spaces(
            view_scope=LimitedViewScope(3),
            space_resolution=10,
            dimensions=2,
            max_price=10.0,
            max_quality=1.0,
        )

        # Test price space
        assert isinstance(price_space, spaces.Box)
        assert np.all(price_space.low == Observation.NO_SELLER_PRICE_PLACEHOLDER)
        assert np.all(price_space.high == 10.0)
        assert price_space.shape == (7, 7)  # (2 * 3 + 1) for each dimension
        assert price_space.dtype == np.float32

        # Test quality space
        assert isinstance(quality_space, spaces.Box)
        assert np.all(quality_space.low == Observation.NO_SELLER_QUALITY_PLACEHOLDER)
        assert np.all(quality_space.high == 1.0)
        assert quality_space.shape == (7, 7)  # (2 * 3 + 1) for each dimension
        assert quality_space.dtype == np.float32

    def test_create_observation_space_private_level(self) -> None:
        """Test create_observation_space with PRIVATE information level."""
        obs_space = Observation.create_observation_space(
            information_level=InformationLevel.PRIVATE,
            view_scope=LimitedViewScope(4),
            dimensions=2,
            space_resolution=10,
            max_price=10.0,
            max_quality=1.0,
            max_valuation=20.0,
        )

        assert isinstance(obs_space, spaces.Dict)
        assert "own_position" in obs_space.spaces
        assert "own_price" in obs_space.spaces
        assert "own_quality" in obs_space.spaces
        assert "local_view" in obs_space.spaces
        assert "buyers" not in obs_space.spaces
        assert "sellers_price" not in obs_space.spaces
        assert "sellers_quality" not in obs_space.spaces

        # Check individual spaces
        assert isinstance(obs_space.spaces["own_position"], spaces.Box)
        assert isinstance(obs_space.spaces["own_price"], spaces.Box)
        assert isinstance(obs_space.spaces["own_quality"], spaces.Box)
        assert isinstance(obs_space.spaces["local_view"], spaces.Box)
        assert obs_space.spaces["own_position"].shape == (2,)
        assert obs_space.spaces["own_price"].high.item() == 10.0
        assert obs_space.spaces["own_quality"].high.item() == 1.0
        assert obs_space.spaces["local_view"].shape == (9, 9)  # (2 * 4 + 1) for each dimension

    def test_create_observation_space_limited_level(self) -> None:
        """Test create_observation_space with LIMITED information level."""
        obs_space = Observation.create_observation_space(
            information_level=InformationLevel.LIMITED,
            view_scope=LimitedViewScope(4),
            dimensions=2,
            space_resolution=10,
            max_price=10.0,
            max_quality=1.0,
            max_valuation=20.0,
        )

        assert isinstance(obs_space, spaces.Dict)
        assert "own_position" in obs_space.spaces
        assert "own_price" in obs_space.spaces
        assert "own_quality" in obs_space.spaces
        assert "local_view" in obs_space.spaces
        assert "buyers" in obs_space.spaces
        assert "sellers_price" not in obs_space.spaces
        assert "sellers_quality" not in obs_space.spaces

    def test_create_observation_space_complete_level(self) -> None:
        """Test create_observation_space with COMPLETE information level."""
        obs_space = Observation.create_observation_space(
            information_level=InformationLevel.COMPLETE,
            view_scope=LimitedViewScope(4),
            dimensions=2,
            space_resolution=10,
            max_price=10.0,
            max_quality=1.0,
            max_valuation=20.0,
        )

        assert isinstance(obs_space, spaces.Dict)
        assert "own_position" in obs_space.spaces
        assert "own_price" in obs_space.spaces
        assert "own_quality" in obs_space.spaces
        assert "local_view" in obs_space.spaces
        assert "buyers" in obs_space.spaces
        assert "sellers_price" in obs_space.spaces
        assert "sellers_quality" in obs_space.spaces

    def test_local_view_observation(self, sample_competition_space: CompetitionSpace) -> None:
        """Test local view observation."""
        observation = Observation.build_from_competition_space(
            sample_competition_space, InformationLevel.COMPLETE, LimitedViewScope(2), "seller_1"
        )

        # 0 = empty, 1 = self, 2 = other seller, 3 = buyer
        expected = np.array(
            [
                [0, 0, 0, 3, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 3, 0],
                [0, 0, 0, 0, 0],
            ]
        )

        local_view = observation.get_observation()["local_view"]
        assert np.array_equal(local_view, expected)

    def test_buyers_observation(self, sample_competition_space: CompetitionSpace) -> None:
        """Test buyers observation."""
        with patch("spatial_competition_pettingzoo.buyer.Buyer.value_for_seller", return_value=15.0):
            observation = Observation.build_from_competition_space(
                sample_competition_space, InformationLevel.COMPLETE, LimitedViewScope(2), "seller_1"
            )

        # 0 = empty, 1 = self, 2 = other seller, 3 = buyer
        expected = np.full(shape=(5, 5), fill_value=Observation.NO_BUYER_PLACEHOLDER, dtype=np.float32)
        expected[0, 3] = 15.0
        expected[3, 3] = 15.0

        buyers_obs = observation.get_observation()["buyers"]
        assert np.array_equal(buyers_obs, expected)

    def test_sellers_price_observation(self, sample_competition_space: CompetitionSpace) -> None:
        """Test sellers price observation."""
        observation = Observation.build_from_competition_space(
            sample_competition_space, InformationLevel.COMPLETE, LimitedViewScope(4), "seller_1"
        )

        expected = np.full(shape=(9, 9), fill_value=Observation.NO_SELLER_PRICE_PLACEHOLDER, dtype=np.float32)
        expected[8, 8] = 7.5

        sellers_price_obs = observation.get_observation()["sellers_price"]
        assert np.array_equal(sellers_price_obs, expected)

    def test_sellers_quality_observation(self, sample_competition_space: CompetitionSpace) -> None:
        """Test sellers quality observation."""
        observation = Observation.build_from_competition_space(
            sample_competition_space, InformationLevel.COMPLETE, LimitedViewScope(4), "seller_2"
        )

        expected = np.full(shape=(9, 9), fill_value=Observation.NO_SELLER_QUALITY_PLACEHOLDER, dtype=np.float32)
        expected[0, 0] = 0.8

        sellers_quality_obs = observation.get_observation()["sellers_quality"]
        assert np.array_equal(sellers_quality_obs, expected)

    def test_complete_observation_dict_keys_match_space_keys(self, sample_position: Position) -> None:
        """Test that observation dictionary keys match the observation space keys for COMPLETE information level."""
        local_view = np.array([[0, 1]], dtype=np.int8)
        buyers = np.array([[10.5]], dtype=np.float32)
        sellers_price = np.array([[5.0]], dtype=np.float32)
        sellers_quality = np.array([[0.8]], dtype=np.float32)

        observation = Observation(
            own_price=5.0,
            own_quality=0.8,
            own_position=sample_position,
            local_view=local_view,
            buyers=buyers,
            sellers_price=sellers_price,
            sellers_quality=sellers_quality,
        )

        obs_dict = observation.get_observation()
        obs_space = Observation.create_observation_space(
            information_level=InformationLevel.COMPLETE,
            dimensions=2,
            space_resolution=10,
            view_scope=LimitedViewScope(1),
            max_price=10.0,
            max_quality=1.0,
            max_valuation=20.0,
        )

        # All keys in observation should be in the observation space
        for key in obs_dict:
            assert key in obs_space.spaces

    def test_limited_observation_dict_keys_match_space_keys(self, sample_position: Position) -> None:
        """Test that observation dictionary keys match the observation space keys for LIMITED information level."""
        local_view = np.array([[0, 1]], dtype=np.int8)
        buyers = np.array([[10.5]], dtype=np.float32)

        observation = Observation(
            own_price=5.0,
            own_quality=0.8,
            own_position=sample_position,
            local_view=local_view,
            buyers=buyers,
            sellers_price=None,
            sellers_quality=None,
        )

        obs_dict = observation.get_observation()
        obs_space = Observation.create_observation_space(
            information_level=InformationLevel.LIMITED,
            dimensions=2,
            space_resolution=10,
            view_scope=LimitedViewScope(1),
            max_price=10.0,
            max_quality=1.0,
            max_valuation=20.0,
        )

        # All keys in observation should be in the observation space
        for key in obs_dict:
            assert key in obs_space.spaces

    def test_private_observation_dict_keys_match_space_keys(self, sample_position: Position) -> None:
        """Test that observation dictionary keys match the observation space keys for PRIVATE information level."""
        local_view = np.array([[0, 1]], dtype=np.int8)
        buyers = None
        sellers_price = None
        sellers_quality = None

        observation = Observation(
            own_price=5.0,
            own_quality=0.8,
            own_position=sample_position,
            local_view=local_view,
            buyers=buyers,
            sellers_price=sellers_price,
            sellers_quality=sellers_quality,
        )

        obs_dict = observation.get_observation()
        obs_space = Observation.create_observation_space(
            information_level=InformationLevel.PRIVATE,
            dimensions=2,
            space_resolution=10,
            view_scope=LimitedViewScope(1),
            max_price=10.0,
            max_quality=1.0,
            max_valuation=20.0,
        )

        # All keys in observation should be in the observation space
        for key in obs_dict:
            assert key in obs_space.spaces
