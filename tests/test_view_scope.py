"""Tests for the ViewScope classes."""

import numpy as np
import pytest

from spatial_competition_pettingzoo.buyer import Buyer
from spatial_competition_pettingzoo.competition_space import CompetitionSpace
from spatial_competition_pettingzoo.position import Position
from spatial_competition_pettingzoo.seller import Seller
from spatial_competition_pettingzoo.topology import Topology
from spatial_competition_pettingzoo.view_scope import (
    CompleteViewScope,
    LimitedViewScope,
)


class TestLimitedViewScope:
    """Test class for LimitedViewScope functionality."""

    def test_properties(self) -> None:
        """Test LimitedViewScope properties."""
        view_scope = LimitedViewScope(vision_radius=3)

        assert view_scope.is_limited is True
        assert view_scope.is_complete is False
        assert view_scope.vision_radius == 3

    def test_seller_view_shape(self) -> None:
        """Test seller_view_shape calculation for 3D space."""
        vision_radius = 1
        view_scope = LimitedViewScope(vision_radius)

        space_resolution = 10
        dimensions = 3

        shape = view_scope.seller_view_shape(space_resolution, dimensions)
        expected_shape = (3, 3, 3)  # (2 * 1 + 1, 2 * 1 + 1, 2 * 1 + 1)

        assert shape == expected_shape

    @pytest.fixture
    def sample_competition_space(self) -> CompetitionSpace:
        """Create a sample competition space for testing."""
        # Create some sellers
        seller1_position = Position(
            space_resolution=10,
            topology=Topology.RECTANGLE,
            tensor_coordinates=np.array([2, 3], dtype=np.int32),
        )
        seller2_position = Position(
            space_resolution=10,
            topology=Topology.RECTANGLE,
            tensor_coordinates=np.array([7, 8], dtype=np.int32),
        )

        seller1 = Seller(idx=1, position=seller1_position, price=5.0, quality=0.8)
        seller2 = Seller(idx=2, position=seller2_position, price=7.5, quality=0.9)
        sellers = {"seller_1": seller1, "seller_2": seller2}

        # Create some buyers
        buyer1_position = Position(
            space_resolution=10,
            topology=Topology.RECTANGLE,
            tensor_coordinates=np.array([1, 2], dtype=np.int32),
        )
        buyer2_position = Position(
            space_resolution=10,
            topology=Topology.RECTANGLE,
            tensor_coordinates=np.array([8, 9], dtype=np.int32),
        )

        buyer1 = Buyer(position=buyer1_position, value=15.0, quality_taste=0.7, distance_factor=0.3)
        buyer2 = Buyer(position=buyer2_position, value=12.0, quality_taste=0.8, distance_factor=0.2)
        buyers = [buyer1, buyer2]

        return CompetitionSpace(
            dimensions=2,
            topology=Topology.RECTANGLE,
            space_resolution=10,
            sellers=sellers,
            buyers=buyers,
        )

    def test_build_seller_subspace(self, sample_competition_space: CompetitionSpace) -> None:
        """Test build_seller_subspace with rectangle topology."""
        vision_radius = 2
        view_scope = LimitedViewScope(vision_radius)

        # Test with seller at (5, 5)
        seller_position = Position(
            space_resolution=10,
            topology=Topology.RECTANGLE,
            tensor_coordinates=np.array([5, 5], dtype=np.int32),
        )

        subspace = view_scope.build_seller_subspace(sample_competition_space, seller_position)

        # Check that seller position is within the subspace
        assert subspace.is_in_subspace(seller_position)

        # Check subspace dimensions
        assert subspace.dimensions == 2
        assert subspace.topology == Topology.RECTANGLE
        assert subspace.space_resolution == 10

    def test_vision_radius_too_large_rectangle_raises_error(self) -> None:
        """Test that vision radius >= space_resolution raises error for rectangle topology."""
        # Vision radius equal to space resolution should fail
        vision_radius = 10
        view_scope = LimitedViewScope(vision_radius)

        seller_position = Position(
            space_resolution=10,
            topology=Topology.RECTANGLE,
            tensor_coordinates=np.array([5, 5], dtype=np.int32),
        )
        seller = Seller(idx=1, position=seller_position, price=5.0, quality=0.8)
        sellers = {"seller_1": seller}

        buyer_position = Position(
            space_resolution=10,
            topology=Topology.RECTANGLE,
            tensor_coordinates=np.array([1, 1], dtype=np.int32),
        )
        buyer = Buyer(position=buyer_position, value=15.0, quality_taste=0.7, distance_factor=0.3)
        buyers = [buyer]

        space = CompetitionSpace(
            dimensions=2,
            topology=Topology.RECTANGLE,
            space_resolution=10,
            sellers=sellers,
            buyers=buyers,
        )

        with pytest.raises(
            AssertionError, match="Vision radius must be less than space resolution for rectangle topology"
        ):
            view_scope.build_seller_subspace(space, seller_position)

    def test_vision_radius_too_large_torus_raises_error(self) -> None:
        """Test that vision radius >= space_resolution/2 raises error for torus topology."""
        # Vision radius equal to half space resolution should fail
        vision_radius = 5  # space_resolution/2 = 10/2 = 5
        view_scope = LimitedViewScope(vision_radius)

        seller_position = Position(
            space_resolution=10,
            topology=Topology.TORUS,
            tensor_coordinates=np.array([5, 5], dtype=np.int32),
        )
        seller = Seller(idx=1, position=seller_position, price=5.0, quality=0.8)
        sellers = {"seller_1": seller}

        buyer_position = Position(
            space_resolution=10,
            topology=Topology.TORUS,
            tensor_coordinates=np.array([1, 1], dtype=np.int32),
        )
        buyer = Buyer(position=buyer_position, value=15.0, quality_taste=0.7, distance_factor=0.3)
        buyers = [buyer]

        space = CompetitionSpace(
            dimensions=2,
            topology=Topology.TORUS,
            space_resolution=10,
            sellers=sellers,
            buyers=buyers,
        )

        with pytest.raises(
            AssertionError, match="Vision radius must be less than half of space resolution for torus topology"
        ):
            view_scope.build_seller_subspace(space, seller_position)


class TestCompleteViewScope:
    """Test class for CompleteViewScope functionality."""

    def test_properties(self) -> None:
        """Test CompleteViewScope properties."""
        view_scope = CompleteViewScope()

        assert view_scope.is_limited is False
        assert view_scope.is_complete is True

    def test_seller_view_shape(self) -> None:
        """Test seller_view_shape calculation for 3D space."""
        view_scope = CompleteViewScope()

        space_resolution = 15
        dimensions = 3

        shape = view_scope.seller_view_shape(space_resolution, dimensions)
        expected_shape = (15, 15, 15)

        assert shape == expected_shape

    @pytest.fixture
    def sample_competition_space(self) -> CompetitionSpace:
        """Create a sample competition space for testing."""
        seller_position = Position(
            space_resolution=10,
            topology=Topology.RECTANGLE,
            tensor_coordinates=np.array([3, 4], dtype=np.int32),
        )
        seller = Seller(idx=1, position=seller_position, price=5.0, quality=0.8)
        sellers = {"seller_1": seller}

        buyer_position = Position(
            space_resolution=10,
            topology=Topology.RECTANGLE,
            tensor_coordinates=np.array([7, 8], dtype=np.int32),
        )
        buyer = Buyer(position=buyer_position, value=15.0, quality_taste=0.7, distance_factor=0.3)
        buyers = [buyer]

        return CompetitionSpace(
            dimensions=2,
            topology=Topology.RECTANGLE,
            space_resolution=10,
            sellers=sellers,
            buyers=buyers,
        )

    def test_build_seller_subspace(self, sample_competition_space: CompetitionSpace) -> None:
        """Test that build_seller_subspace returns the entire space."""
        view_scope = CompleteViewScope()

        seller_position = Position(
            space_resolution=10,
            topology=Topology.RECTANGLE,
            tensor_coordinates=np.array([5, 5], dtype=np.int32),
        )

        subspace = view_scope.build_seller_subspace(sample_competition_space, seller_position)

        # The returned subspace should be the same as the original space
        assert subspace is sample_competition_space
        assert subspace.dimensions == sample_competition_space.dimensions
        assert subspace.topology == sample_competition_space.topology
        assert subspace.space_resolution == sample_competition_space.space_resolution

    def test_build_seller_subspace_different_seller_positions(self, sample_competition_space: CompetitionSpace) -> None:
        """Test that build_seller_subspace returns the same space regardless of seller position."""
        view_scope = CompleteViewScope()

        positions = [
            np.array([0, 0], dtype=np.int32),
            np.array([5, 5], dtype=np.int32),
            np.array([9, 9], dtype=np.int32),
            np.array([2, 8], dtype=np.int32),
        ]

        for pos_coords in positions:
            seller_position = Position(
                space_resolution=10,
                topology=Topology.RECTANGLE,
                tensor_coordinates=pos_coords,
            )

            subspace = view_scope.build_seller_subspace(sample_competition_space, seller_position)
            assert subspace is sample_competition_space
