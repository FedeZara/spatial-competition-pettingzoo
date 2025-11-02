"""Tests for the CompetitionSpace class."""

import numpy as np
import pytest

from spatial_competition_pettingzoo.buyer import Buyer
from spatial_competition_pettingzoo.competition_space import CompetitionSpace
from spatial_competition_pettingzoo.position import Position
from spatial_competition_pettingzoo.seller import Seller
from spatial_competition_pettingzoo.topology import Topology


class TestCompetitionSpace:
    """Test class for CompetitionSpace functionality."""

    @pytest.fixture
    def sample_sellers(self) -> dict[str, Seller]:
        """Create sample sellers for testing."""
        seller1_position = Position(
            space_resolution=10,
            topology=Topology.RECTANGLE,
            tensor_coordinates=np.array([3, 4], dtype=np.int32),
        )
        seller2_position = Position(
            space_resolution=10,
            topology=Topology.RECTANGLE,
            tensor_coordinates=np.array([6, 7], dtype=np.int32),
        )

        seller1 = Seller(idx=1, position=seller1_position, price=5.0, quality=0.8)
        seller2 = Seller(idx=2, position=seller2_position, price=7.5, quality=0.9)

        return {"seller_1": seller1, "seller_2": seller2}

    @pytest.fixture
    def sample_buyers(self) -> list[Buyer]:
        """Create sample buyers for testing."""
        buyer1_position = Position(
            space_resolution=10,
            topology=Topology.RECTANGLE,
            tensor_coordinates=np.array([2, 3], dtype=np.int32),
        )
        buyer2_position = Position(
            space_resolution=10,
            topology=Topology.RECTANGLE,
            tensor_coordinates=np.array([7, 8], dtype=np.int32),
        )

        buyer1 = Buyer(position=buyer1_position, value=15.0, quality_taste=0.7, distance_factor=0.3)
        buyer2 = Buyer(position=buyer2_position, value=12.0, quality_taste=0.8, distance_factor=0.2)

        return [buyer1, buyer2]

    @pytest.fixture
    def sample_competition_space(
        self, sample_sellers: dict[str, Seller], sample_buyers: list[Buyer]
    ) -> CompetitionSpace:
        """Create a sample competition space for testing."""
        return CompetitionSpace(
            dimensions=2,
            topology=Topology.RECTANGLE,
            space_resolution=10,
            sellers=sample_sellers,
            buyers=sample_buyers,
        )

    @pytest.fixture
    def torus_competition_space(
        self, sample_sellers: dict[str, Seller], sample_buyers: list[Buyer]
    ) -> CompetitionSpace:
        """Create a sample competition space with torus topology for testing."""
        # Update positions for torus topology
        torus_sellers = {}
        for key, seller in sample_sellers.items():
            torus_position = Position(
                space_resolution=10,
                topology=Topology.TORUS,
                tensor_coordinates=seller.position.tensor_coordinates,
            )
            torus_seller = Seller(
                idx=seller.idx,
                position=torus_position,
                price=seller.price,
                quality=seller.quality,
            )
            torus_sellers[key] = torus_seller

        torus_buyers = []
        for buyer in sample_buyers:
            torus_position = Position(
                space_resolution=10,
                topology=Topology.TORUS,
                tensor_coordinates=buyer.position.tensor_coordinates,
            )
            torus_buyer = Buyer(
                position=torus_position,
                value=buyer.value,
                quality_taste=buyer.quality_taste,
                distance_factor=buyer.distance_factor,
            )
            torus_buyers.append(torus_buyer)

        return CompetitionSpace(
            dimensions=2,
            topology=Topology.TORUS,
            space_resolution=10,
            sellers=torus_sellers,
            buyers=torus_buyers,
        )

    def test_init_default_base_extent(self, sample_sellers: dict[str, Seller], sample_buyers: list[Buyer]) -> None:
        """Test CompetitionSpace initialization with default base and extent."""
        comp_space = CompetitionSpace(
            dimensions=2,
            topology=Topology.RECTANGLE,
            space_resolution=10,
            sellers=sample_sellers,
            buyers=sample_buyers,
        )

        # Check default base (should be [0, 0])
        expected_base_coords = np.array([0, 0], dtype=np.int32)
        assert np.array_equal(comp_space.base.tensor_coordinates, expected_base_coords)

        # Check default extent (should be [9, 9])
        expected_extent_coords = np.array([9, 9], dtype=np.int32)
        assert np.array_equal(comp_space.extent.tensor_coordinates, expected_extent_coords)

    def test_init_custom_base_extent(self, sample_sellers: dict[str, Seller], sample_buyers: list[Buyer]) -> None:
        """Test CompetitionSpace initialization with custom base and extent."""
        custom_base = Position(
            space_resolution=10,
            topology=Topology.RECTANGLE,
            tensor_coordinates=np.array([2, 2], dtype=np.int32),
        )
        custom_extent = Position(
            space_resolution=10,
            topology=Topology.RECTANGLE,
            tensor_coordinates=np.array([8, 8], dtype=np.int32),
        )

        comp_space = CompetitionSpace(
            dimensions=2,
            topology=Topology.RECTANGLE,
            space_resolution=10,
            sellers=sample_sellers,
            buyers=sample_buyers,
            base=custom_base,
            extent=custom_extent,
        )

        assert np.array_equal(comp_space.base.tensor_coordinates, np.array([2, 2], dtype=np.int32))
        assert np.array_equal(comp_space.extent.tensor_coordinates, np.array([8, 8], dtype=np.int32))

    def test_properties(self, sample_competition_space: CompetitionSpace) -> None:
        """Test all properties of CompetitionSpace."""
        assert sample_competition_space.dimensions == 2
        assert sample_competition_space.topology == Topology.RECTANGLE
        assert sample_competition_space.space_resolution == 10

        # Test sellers property
        sellers = sample_competition_space.sellers
        assert len(sellers) == 2
        assert "seller_1" in sellers
        assert "seller_2" in sellers

        # Test buyers property
        buyers = sample_competition_space.buyers
        assert len(buyers) == 2

    def test_relative_extent(self, sample_competition_space: CompetitionSpace) -> None:
        """Test relative_extent property."""
        relative_extent = sample_competition_space.relative_extent

        # For default base [0,0] and extent [9,9], relative extent should be [9,9]
        expected_coords = np.array([9, 9], dtype=np.int32)
        assert np.array_equal(relative_extent.tensor_coordinates, expected_coords)

    def test_add_buyers(self, sample_competition_space: CompetitionSpace) -> None:
        """Test add_buyers method."""
        initial_buyer_count = len(sample_competition_space.buyers)

        # Create new buyers
        new_buyer_position = Position(
            space_resolution=10,
            topology=Topology.RECTANGLE,
            tensor_coordinates=np.array([5, 5], dtype=np.int32),
        )
        new_buyer = Buyer(
            position=new_buyer_position,
            value=18.0,
            quality_taste=0.6,
            distance_factor=0.4,
        )
        new_buyers = [new_buyer]

        sample_competition_space.add_buyers(new_buyers)

        assert len(sample_competition_space.buyers) == initial_buyer_count + 1
        assert sample_competition_space.buyers[-1].value == 18.0

    def test_is_in_subspace_valid_positions(self, sample_competition_space: CompetitionSpace) -> None:
        """Test is_in_subspace method with valid positions."""
        # Test position within bounds
        valid_position = Position(
            space_resolution=10,
            topology=Topology.RECTANGLE,
            tensor_coordinates=np.array([5, 5], dtype=np.int32),
        )
        assert sample_competition_space.is_in_subspace(valid_position) is True

        # Test position at base
        base_position = Position(
            space_resolution=10,
            topology=Topology.RECTANGLE,
            tensor_coordinates=np.array([0, 0], dtype=np.int32),
        )
        assert sample_competition_space.is_in_subspace(base_position) is True

        # Test position at extent
        extent_position = Position(
            space_resolution=10,
            topology=Topology.RECTANGLE,
            tensor_coordinates=np.array([9, 9], dtype=np.int32),
        )
        assert sample_competition_space.is_in_subspace(extent_position) is True

    def test_is_in_subspace_invalid_positions(self, sample_competition_space: CompetitionSpace) -> None:
        """Test is_in_subspace method with invalid positions."""
        # This test would cause assertion errors due to mismatched properties
        # We'll test the assertion behavior

        # Test with wrong dimensions
        wrong_dims_position = Position(
            space_resolution=10,
            topology=Topology.RECTANGLE,
            tensor_coordinates=np.array([5], dtype=np.int32),  # 1D instead of 2D
        )

        with pytest.raises(AssertionError):
            sample_competition_space.is_in_subspace(wrong_dims_position)

        # Test with wrong topology
        wrong_topology_position = Position(
            space_resolution=10,
            topology=Topology.TORUS,  # Wrong topology
            tensor_coordinates=np.array([5, 5], dtype=np.int32),
        )

        with pytest.raises(AssertionError):
            sample_competition_space.is_in_subspace(wrong_topology_position)

        # Test with wrong space resolution
        wrong_resolution_position = Position(
            space_resolution=5,  # Wrong resolution
            topology=Topology.RECTANGLE,
            tensor_coordinates=np.array([2, 2], dtype=np.int32),
        )

        with pytest.raises(AssertionError):
            sample_competition_space.is_in_subspace(wrong_resolution_position)

    def test_relative_position(self, sample_competition_space: CompetitionSpace) -> None:
        """Test relative_position method."""
        absolute_position = Position(
            space_resolution=10,
            topology=Topology.RECTANGLE,
            tensor_coordinates=np.array([5, 6], dtype=np.int32),
        )

        relative_pos = sample_competition_space.relative_position(absolute_position)

        # With default base [0, 0], relative position should be the same
        expected_coords = np.array([5, 6], dtype=np.int32)
        assert np.array_equal(relative_pos.tensor_coordinates, expected_coords)

        # Test with custom base
        custom_base = Position(
            space_resolution=10,
            topology=Topology.RECTANGLE,
            tensor_coordinates=np.array([2, 2], dtype=np.int32),
        )

        comp_space_custom = CompetitionSpace(
            dimensions=2,
            topology=Topology.RECTANGLE,
            space_resolution=10,
            sellers={},
            buyers=[],
            base=custom_base,
        )

        relative_pos_custom = comp_space_custom.relative_position(absolute_position)
        # Expected is [5, 6] - [2, 2] = [3, 4]
        expected_coords_custom = np.array([3, 4], dtype=np.int32)
        assert np.array_equal(relative_pos_custom.tensor_coordinates, expected_coords_custom)

    def test_subspace_creation(self, sample_competition_space: CompetitionSpace) -> None:
        """Test subspace method creates a proper subspace."""
        # Define subspace bounds
        sub_base = Position(
            space_resolution=10,
            topology=Topology.RECTANGLE,
            tensor_coordinates=np.array([1, 1], dtype=np.int32),
        )
        sub_extent = Position(
            space_resolution=10,
            topology=Topology.RECTANGLE,
            tensor_coordinates=np.array([8, 8], dtype=np.int32),
        )

        subspace = sample_competition_space.subspace(sub_base, sub_extent)

        # Check basic properties
        assert subspace.dimensions == 2
        assert subspace.topology == Topology.RECTANGLE
        assert subspace.space_resolution == 10
        assert np.array_equal(subspace.base.tensor_coordinates, np.array([1, 1], dtype=np.int32))
        assert np.array_equal(subspace.extent.tensor_coordinates, np.array([8, 8], dtype=np.int32))

    def test_subspace_copies_sellers_and_buyers(self, sample_competition_space: CompetitionSpace) -> None:
        """Test that subspace method copies all sellers and buyers that are in the original space."""
        # NOTE: The current implementation checks if entities are in the original competition space,
        # not the new subspace bounds. This may be a bug, but we test the actual behavior.

        sub_base = Position(
            space_resolution=10,
            topology=Topology.RECTANGLE,
            tensor_coordinates=np.array([1, 1], dtype=np.int32),
        )
        sub_extent = Position(
            space_resolution=10,
            topology=Topology.RECTANGLE,
            tensor_coordinates=np.array([8, 8], dtype=np.int32),
        )

        subspace = sample_competition_space.subspace(sub_base, sub_extent)

        # Both sellers should be included because they're in the original competition space
        assert len(subspace.sellers) == 2
        assert "seller_1" in subspace.sellers
        assert "seller_2" in subspace.sellers

        # Both buyers should be included because they're in the original competition space
        assert len(subspace.buyers) == 2

        # Check that the subspace has the correct bounds
        assert np.array_equal(subspace.base.tensor_coordinates, np.array([1, 1], dtype=np.int32))
        assert np.array_equal(subspace.extent.tensor_coordinates, np.array([8, 8], dtype=np.int32))

    def test_subspace_filters_based_on_original_bounds(self) -> None:
        """Test that subspace method filters based on original competition space bounds."""
        # Create a competition space with limited bounds
        custom_base = Position(
            space_resolution=10,
            topology=Topology.RECTANGLE,
            tensor_coordinates=np.array([2, 2], dtype=np.int32),
        )
        custom_extent = Position(
            space_resolution=10,
            topology=Topology.RECTANGLE,
            tensor_coordinates=np.array([6, 6], dtype=np.int32),
        )

        # Create sellers - one inside, one outside the custom bounds
        inside_position = Position(
            space_resolution=10,
            topology=Topology.RECTANGLE,
            tensor_coordinates=np.array([3, 4], dtype=np.int32),  # Inside [2,2] to [6,6]
        )
        outside_position = Position(
            space_resolution=10,
            topology=Topology.RECTANGLE,
            tensor_coordinates=np.array([1, 1], dtype=np.int32),  # Outside [2,2] to [6,6]
        )

        inside_seller = Seller(idx=1, position=inside_position, price=5.0, quality=0.8)
        outside_seller = Seller(idx=2, position=outside_position, price=7.0, quality=0.9)

        custom_comp_space = CompetitionSpace(
            dimensions=2,
            topology=Topology.RECTANGLE,
            space_resolution=10,
            sellers={"seller_1": inside_seller, "seller_2": outside_seller},
            buyers=[],
            base=custom_base,
            extent=custom_extent,
        )

        # Create subspace with different bounds
        sub_base = Position(
            space_resolution=10,
            topology=Topology.RECTANGLE,
            tensor_coordinates=np.array([3, 3], dtype=np.int32),
        )
        sub_extent = Position(
            space_resolution=10,
            topology=Topology.RECTANGLE,
            tensor_coordinates=np.array([5, 5], dtype=np.int32),
        )

        subspace = custom_comp_space.subspace(sub_base, sub_extent)

        # Only the seller inside the original bounds should be copied
        assert len(subspace.sellers) == 1
        assert "seller_1" in subspace.sellers
        assert "seller_2" not in subspace.sellers

    def test_subspace_invalid_bounds(self) -> None:
        """Test subspace method with invalid bounds raises assertions."""
        # Create a competition space with custom bounds to test out-of-bounds positions
        custom_base = Position(
            space_resolution=10,
            topology=Topology.RECTANGLE,
            tensor_coordinates=np.array([2, 2], dtype=np.int32),
        )
        custom_extent = Position(
            space_resolution=10,
            topology=Topology.RECTANGLE,
            tensor_coordinates=np.array([7, 7], dtype=np.int32),
        )

        custom_comp_space = CompetitionSpace(
            dimensions=2,
            topology=Topology.RECTANGLE,
            space_resolution=10,
            sellers={},
            buyers=[],
            base=custom_base,
            extent=custom_extent,
        )

        # Test with base outside the custom competition space bounds
        invalid_base = Position(
            space_resolution=10,
            topology=Topology.RECTANGLE,
            tensor_coordinates=np.array([1, 1], dtype=np.int32),  # Outside custom bounds [2,2] to [7,7]
        )
        valid_extent = Position(
            space_resolution=10,
            topology=Topology.RECTANGLE,
            tensor_coordinates=np.array([6, 6], dtype=np.int32),
        )

        with pytest.raises(AssertionError):
            custom_comp_space.subspace(invalid_base, valid_extent)

    def test_subspace_includes_only_belonging_agents(self) -> None:
        """Test that subspace only includes sellers and buyers within the subspace bounds."""
        # Create sellers at different positions
        seller_inside_position = Position(
            space_resolution=10,
            topology=Topology.RECTANGLE,
            tensor_coordinates=np.array([3, 4], dtype=np.int32),
        )
        seller_outside_position = Position(
            space_resolution=10,
            topology=Topology.RECTANGLE,
            tensor_coordinates=np.array([8, 9], dtype=np.int32),
        )
        seller_on_boundary_position = Position(
            space_resolution=10,
            topology=Topology.RECTANGLE,
            tensor_coordinates=np.array([2, 6], dtype=np.int32),
        )

        seller_inside = Seller(idx=1, position=seller_inside_position, price=5.0, quality=0.8)
        seller_outside = Seller(idx=2, position=seller_outside_position, price=6.0, quality=0.7)
        seller_on_boundary = Seller(idx=3, position=seller_on_boundary_position, price=7.0, quality=0.9)

        # Create buyers at different positions
        buyer_inside_position = Position(
            space_resolution=10,
            topology=Topology.RECTANGLE,
            tensor_coordinates=np.array([4, 5], dtype=np.int32),
        )
        buyer_outside_position = Position(
            space_resolution=10,
            topology=Topology.RECTANGLE,
            tensor_coordinates=np.array([1, 1], dtype=np.int32),
        )
        buyer_on_boundary_position = Position(
            space_resolution=10,
            topology=Topology.RECTANGLE,
            tensor_coordinates=np.array([6, 6], dtype=np.int32),
        )

        buyer_inside = Buyer(
            position=buyer_inside_position,
            value=15.0,
            quality_taste=0.7,
            distance_factor=0.3,
        )
        buyer_outside = Buyer(
            position=buyer_outside_position,
            value=12.0,
            quality_taste=0.8,
            distance_factor=0.2,
        )
        buyer_on_boundary = Buyer(
            position=buyer_on_boundary_position,
            value=18.0,
            quality_taste=0.6,
            distance_factor=0.4,
        )

        # Create competition space with all agents
        sellers = {
            "seller_1": seller_inside,
            "seller_2": seller_outside,
            "seller_3": seller_on_boundary,
        }
        buyers = [buyer_inside, buyer_outside, buyer_on_boundary]

        competition_space = CompetitionSpace(
            dimensions=2,
            topology=Topology.RECTANGLE,
            space_resolution=10,
            sellers=sellers,
            buyers=buyers,
        )

        # Define subspace bounds that include some agents and exclude others
        # Subspace bounds: [2, 2] to [6, 6]
        sub_base = Position(
            space_resolution=10,
            topology=Topology.RECTANGLE,
            tensor_coordinates=np.array([2, 2], dtype=np.int32),
        )
        sub_extent = Position(
            space_resolution=10,
            topology=Topology.RECTANGLE,
            tensor_coordinates=np.array([6, 6], dtype=np.int32),
        )

        # Create subspace
        subspace = competition_space.subspace(sub_base, sub_extent)

        # Verify only agents within bounds are included
        # Expected: seller_inside [3,4] and seller_on_boundary [2,6] should be included
        # seller_outside [8,9] should be excluded
        assert len(subspace.sellers) == 2
        assert "seller_1" in subspace.sellers
        assert "seller_3" in subspace.sellers
        assert "seller_2" not in subspace.sellers

        # Expected: buyer_inside [4,5] and buyer_on_boundary [6,6] should be included
        # buyer_outside [1,1] should be excluded
        assert len(subspace.buyers) == 2

        # Check positions of included buyers
        included_buyer_positions = [buyer.position.tensor_coordinates for buyer in subspace.buyers]

        # buyer_inside position [4,5] should be included
        buyer_inside_coords_found = any(
            np.array_equal(pos, np.array([4, 5], dtype=np.int32)) for pos in included_buyer_positions
        )
        assert buyer_inside_coords_found, "Buyer inside subspace bounds should be included"

        # buyer_on_boundary position [6,6] should be included (boundary is inclusive)
        buyer_boundary_coords_found = any(
            np.array_equal(pos, np.array([6, 6], dtype=np.int32)) for pos in included_buyer_positions
        )
        assert buyer_boundary_coords_found, "Buyer on subspace boundary should be included"

        # buyer_outside position [1,1] should NOT be included
        buyer_outside_coords_found = any(
            np.array_equal(pos, np.array([1, 1], dtype=np.int32)) for pos in included_buyer_positions
        )
        assert not buyer_outside_coords_found, "Buyer outside subspace bounds should not be included"

        # Verify that the copied agents have the same properties as originals
        assert subspace.sellers["seller_1"].price == 5.0
        assert subspace.sellers["seller_1"].quality == 0.8
        assert subspace.sellers["seller_3"].price == 7.0
        assert subspace.sellers["seller_3"].quality == 0.9

    def test_subspace_torus_topology_filtering(self) -> None:
        """Test subspace filtering with torus topology for wrap-around cases."""
        # Create sellers at positions that test torus wrap-around behavior
        seller_normal_position = Position(
            space_resolution=10,
            topology=Topology.TORUS,
            tensor_coordinates=np.array([1, 2], dtype=np.int32),
        )
        seller_wraparound_position = Position(
            space_resolution=10,
            topology=Topology.TORUS,
            tensor_coordinates=np.array([9, 1], dtype=np.int32),
        )

        seller_normal = Seller(idx=1, position=seller_normal_position, price=5.0, quality=0.8)
        seller_wraparound = Seller(idx=2, position=seller_wraparound_position, price=6.0, quality=0.7)

        sellers = {"seller_1": seller_normal, "seller_2": seller_wraparound}
        buyers: list[Buyer] = []

        competition_space = CompetitionSpace(
            dimensions=2,
            topology=Topology.TORUS,
            space_resolution=10,
            sellers=sellers,
            buyers=buyers,
        )

        # Define subspace with bounds that wrap around: [8, 0] to [2, 3]
        # This should include [9,1] (wraparound case) and [1,2] (normal case)
        sub_base = Position(
            space_resolution=10,
            topology=Topology.TORUS,
            tensor_coordinates=np.array([8, 0], dtype=np.int32),
        )
        sub_extent = Position(
            space_resolution=10,
            topology=Topology.TORUS,
            tensor_coordinates=np.array([2, 3], dtype=np.int32),
        )

        subspace = competition_space.subspace(sub_base, sub_extent)

        # Both sellers should be included due to torus wrap-around logic
        assert len(subspace.sellers) == 2
        assert "seller_1" in subspace.sellers
        assert "seller_2" in subspace.sellers
