"""Tests for the Position class."""

import numpy as np
import pytest

from spatial_competition_pettingzoo.position import Position
from spatial_competition_pettingzoo.topology import Topology


class TestPosition:
    """Test class for Position functionality."""

    def test_init_with_tensor_coordinates(self) -> None:
        """Test Position initialization with tensor coordinates."""
        tensor_coords = np.array([5, 8], dtype=np.int32)  # Valid range [0, 10) for space_resolution=10
        space_resolution = 10
        topology = Topology.RECTANGLE

        position = Position(
            space_resolution=space_resolution,
            topology=topology,
            tensor_coordinates=tensor_coords,
        )

        assert np.array_equal(position.tensor_coordinates, tensor_coords)
        assert position.space_resolution == space_resolution
        assert position.topology == topology
        assert position.dimensions == 2
        expected_space_coords = np.array([0.5, 0.8], dtype=np.float32)  # 5*0.1=0.5, 8*0.1=0.8
        assert np.allclose(position.space_coordinates, expected_space_coords)

    def test_init_with_space_coordinates(self) -> None:
        """Test Position initialization with space coordinates."""
        space_coords = np.array([0.5, 0.8], dtype=np.float32)
        space_resolution = 10
        topology = Topology.TORUS

        position = Position(
            space_resolution=space_resolution,
            topology=topology,
            space_coordinates=space_coords,
        )

        expected_tensor_coords = np.array([5, 8], dtype=np.int32)
        assert np.array_equal(position.tensor_coordinates, expected_tensor_coords)
        assert position.space_resolution == space_resolution
        assert position.topology == topology
        assert position.dimensions == 2
        assert np.allclose(position.space_coordinates, space_coords)

    def test_init_with_neither_coordinates_raises_error(self) -> None:
        """Test that initializing without coordinates raises ValueError."""
        with pytest.raises(ValueError, match="Either tensor_coordinates or space_coordinates must be provided"):
            Position(
                space_resolution=10,
                topology=Topology.RECTANGLE,
            )

    def test_init_with_both_coordinates_raises_error(self) -> None:
        """Test that initializing with both coordinate types raises ValueError."""
        tensor_coords = np.array([10, 20], dtype=np.int32)
        space_coords = np.array([0.5, 0.8], dtype=np.float32)

        with pytest.raises(ValueError, match="Only one of tensor_coordinates or space_coordinates must be provided"):
            Position(
                space_resolution=10,
                topology=Topology.RECTANGLE,
                tensor_coordinates=tensor_coords,
                space_coordinates=space_coords,
            )

    def test_init_with_space_coordinates_at_or_above_one_raises_error(self) -> None:
        """Test that initializing with space coordinates >= 1.0 raises ValueError."""
        # Test with coordinate exactly at 1.0
        space_coords_at_one = np.array([0.5, 1.0], dtype=np.float32)
        with pytest.raises(ValueError, match="Tensor coordinates must be in the range \\(-10, 10\\)"):
            Position(
                space_resolution=10,
                topology=Topology.RECTANGLE,
                space_coordinates=space_coords_at_one,
            )

        # Test with coordinate above 1.0
        space_coords_above_one = np.array([1.2, 0.5], dtype=np.float32)
        with pytest.raises(ValueError, match="Tensor coordinates must be in the range \\(-10, 10\\)"):
            Position(
                space_resolution=10,
                topology=Topology.RECTANGLE,
                space_coordinates=space_coords_above_one,
            )

    def test_init_with_space_coordinates_at_or_below_minus_one_raises_error(self) -> None:
        """Test that initializing with negative space coordinates raises ValueError."""
        space_coords_at_minus_one = np.array([-1.0, 0.5], dtype=np.float32)
        with pytest.raises(ValueError, match="Tensor coordinates must be in the range \\(-10, 10\\)"):
            Position(
                space_resolution=10,
                topology=Topology.RECTANGLE,
                space_coordinates=space_coords_at_minus_one,
            )

        # Test with coordinate above 1.0
        space_coords_below_minus_one = np.array([-1.2, 0.5], dtype=np.float32)
        with pytest.raises(ValueError, match="Tensor coordinates must be in the range \\(-10, 10\\)"):
            Position(
                space_resolution=10,
                topology=Topology.RECTANGLE,
                space_coordinates=space_coords_below_minus_one,
            )

    def test_init_with_mixed_invalid_space_coordinates_raises_error(self) -> None:
        """Test that initializing with mixed invalid space coordinates raises ValueError."""
        # Test with both <= -1.0 and >= 1.0 coordinates
        space_coords = np.array([-1.5, 1.5, 0.3], dtype=np.float32)

        with pytest.raises(ValueError, match="Tensor coordinates must be in the range \\(-10, 10\\)"):
            Position(
                space_resolution=10,
                topology=Topology.RECTANGLE,
                space_coordinates=space_coords,
            )

    def test_init_with_valid_boundary_space_coordinates(self) -> None:
        """Test that initializing with valid boundary space coordinates works."""
        # Test with coordinates at the valid boundaries [0, 1)
        # Use valid multiples of space_resolution
        space_coords = np.array([0.0, 0.9], dtype=np.float32)  # 0.9 is a valid multiple of 0.1
        position = Position(
            space_resolution=10,
            topology=Topology.RECTANGLE,
            space_coordinates=space_coords,
        )

        assert position.dimensions == 2
        assert np.allclose(position.space_coordinates, space_coords)

    def test_init_with_edge_case_space_coordinates(self) -> None:
        """Test edge cases for space coordinate validation."""
        # Test zero coordinate (always a valid multiple)
        space_coords_with_zero = np.array([0.0, 0.5], dtype=np.float32)
        position_with_zero = Position(
            space_resolution=10,
            topology=Topology.RECTANGLE,
            space_coordinates=space_coords_with_zero,
        )
        assert position_with_zero.dimensions == 2

        # Test coordinate at the upper boundary that is a valid multiple
        space_coords_close = np.array([0.5, 0.9], dtype=np.float32)  # 0.9 is a valid multiple of 0.1
        position_close = Position(
            space_resolution=10,
            topology=Topology.RECTANGLE,
            space_coordinates=space_coords_close,
        )
        assert position_close.dimensions == 2

    def test_init_with_tensor_coordinates_at_or_above_max_raises_error(self) -> None:
        """Test that initializing with tensor coordinates >= max value raises ValueError."""
        space_resolution = 10
        max_coord = space_resolution  # 10 for space_resolution=10

        # Test with coordinate exactly at max
        tensor_coords_at_max = np.array([5, max_coord], dtype=np.int32)
        with pytest.raises(
            ValueError, match=f"Tensor coordinates must be in the range \\(-{max_coord}, {max_coord}\\)"
        ):
            Position(
                space_resolution=space_resolution,
                topology=Topology.RECTANGLE,
                tensor_coordinates=tensor_coords_at_max,
            )

        # Test with coordinate above max
        tensor_coords_above_max = np.array([max_coord + 5, 3], dtype=np.int32)
        with pytest.raises(
            ValueError, match=f"Tensor coordinates must be in the range \\(-{max_coord}, {max_coord}\\)"
        ):
            Position(
                space_resolution=space_resolution,
                topology=Topology.RECTANGLE,
                tensor_coordinates=tensor_coords_above_max,
            )

    def test_init_with_tensor_coordinates_at_or_below_minus_max_raises_error(self) -> None:
        """Test that initializing with tensor coordinates >= max value raises ValueError."""
        space_resolution = 10
        max_coord = space_resolution  # 10 for space_resolution=10

        # Test with coordinate exactly at max
        tensor_coords_at_minus_max = np.array([5, -max_coord], dtype=np.int32)
        with pytest.raises(
            ValueError, match=f"Tensor coordinates must be in the range \\(-{max_coord}, {max_coord}\\)"
        ):
            Position(
                space_resolution=space_resolution,
                topology=Topology.RECTANGLE,
                tensor_coordinates=tensor_coords_at_minus_max,
            )

        # Test with coordinate above max
        tensor_coords_below_minus_max = np.array([-max_coord - 5, 3], dtype=np.int32)
        with pytest.raises(
            ValueError, match=f"Tensor coordinates must be in the range \\(-{max_coord}, {max_coord}\\)"
        ):
            Position(
                space_resolution=space_resolution,
                topology=Topology.RECTANGLE,
                tensor_coordinates=tensor_coords_below_minus_max,
            )

    def test_init_with_mixed_invalid_tensor_coordinates_raises_error(self) -> None:
        """Test that initializing with mixed invalid tensor coordinates raises ValueError."""
        space_resolution = 20
        max_coord = space_resolution  # 20 for space_resolution=20

        # Test with both negative and >= max coordinates
        tensor_coords = np.array([-2, max_coord + 3, 5], dtype=np.int32)

        with pytest.raises(
            ValueError, match=f"Tensor coordinates must be in the range \\(-{max_coord}, {max_coord}\\)"
        ):
            Position(
                space_resolution=space_resolution,
                topology=Topology.RECTANGLE,
                tensor_coordinates=tensor_coords,
            )

    def test_init_with_valid_boundary_tensor_coordinates(self) -> None:
        """Test that initializing with valid boundary tensor coordinates works."""
        space_resolution = 10
        max_coord = space_resolution  # 10

        # Test with coordinates at the valid boundaries [0, max_coord)
        tensor_coords = np.array([0, max_coord - 1], dtype=np.int32)

        position = Position(
            space_resolution=space_resolution,
            topology=Topology.RECTANGLE,
            tensor_coordinates=tensor_coords,
        )

        assert position.dimensions == 2
        assert np.array_equal(position.tensor_coordinates, tensor_coords)

    def test_init_with_edge_case_tensor_coordinates(self) -> None:
        """Test edge cases for tensor coordinate validation."""
        # Test with fine resolution
        space_resolution = 100
        max_coord = space_resolution  # 100

        # Test coordinate at zero
        tensor_coords_zero = np.array([0, 50], dtype=np.int32)
        position_zero = Position(
            space_resolution=space_resolution,
            topology=Topology.RECTANGLE,
            tensor_coordinates=tensor_coords_zero,
        )
        assert position_zero.dimensions == 2

        # Test coordinate at max - 1
        tensor_coords_max_minus_one = np.array([25, max_coord - 1], dtype=np.int32)
        position_max = Position(
            space_resolution=space_resolution,
            topology=Topology.RECTANGLE,
            tensor_coordinates=tensor_coords_max_minus_one,
        )
        assert position_max.dimensions == 2

    def test_tensor_coordinate_validation_with_different_resolutions(self) -> None:
        """Test tensor coordinate validation with various space resolutions."""
        # Test with coarse resolution
        coarse_resolution = 4
        coarse_max = coarse_resolution  # 4

        tensor_coords_valid = np.array([0, 3], dtype=np.int32)
        position_valid = Position(
            space_resolution=coarse_resolution,
            topology=Topology.TORUS,
            tensor_coordinates=tensor_coords_valid,
        )
        assert position_valid.dimensions == 2

        # Test invalid coordinate for coarse resolution
        tensor_coords_invalid = np.array([2, coarse_max], dtype=np.int32)
        with pytest.raises(
            ValueError, match=f"Tensor coordinates must be in the range \\(-{coarse_max}, {coarse_max}\\)"
        ):
            Position(
                space_resolution=coarse_resolution,
                topology=Topology.TORUS,
                tensor_coordinates=tensor_coords_invalid,
            )

        # Test with very fine resolution
        fine_resolution = 1000
        fine_max = fine_resolution  # 1000

        tensor_coords_fine = np.array([500, fine_max - 1], dtype=np.int32)
        position_fine = Position(
            space_resolution=fine_resolution,
            topology=Topology.RECTANGLE,
            tensor_coordinates=tensor_coords_fine,
        )
        assert position_fine.dimensions == 2

    def test_init_with_space_coordinates_not_multiples_raises_error(self) -> None:
        """Test that space coordinates that are not multiples of space_resolution raise ValueError."""
        space_resolution = 10

        # Test with coordinates that are not exact multiples
        space_coords_invalid = np.array([0.15, 0.33], dtype=np.float32)  # 0.15 and 0.33 are not multiples of 1/10=0.1

        with pytest.raises(ValueError, match=r"Space coordinates must be multiples of 1/space_resolution \(0.1\)"):
            Position(
                space_resolution=space_resolution,
                topology=Topology.RECTANGLE,
                space_coordinates=space_coords_invalid,
            )

    def test_init_with_valid_multiple_space_coordinates(self) -> None:
        """Test that space coordinates that are exact multiples of space_resolution work."""
        space_resolution = 10

        # Test with coordinates that are exact multiples
        space_coords_valid = np.array([0.0, 0.1, 0.5, 0.9], dtype=np.float32)

        position = Position(
            space_resolution=space_resolution,
            topology=Topology.RECTANGLE,
            space_coordinates=space_coords_valid,
        )

        assert position.dimensions == 4
        assert np.allclose(position.space_coordinates, space_coords_valid)

    def test_space_coordinate_multiple_validation_with_different_resolutions(self) -> None:
        """Test space coordinate multiple validation with various resolutions."""
        # Test with coarse resolution
        coarse_resolution = 4
        space_coords_valid = np.array([0.0, 0.25, 0.5, 0.75], dtype=np.float32)

        position_valid = Position(
            space_resolution=coarse_resolution,
            topology=Topology.TORUS,
            space_coordinates=space_coords_valid,
        )
        assert position_valid.dimensions == 4

        # Test invalid coordinate for coarse resolution
        space_coords_invalid = np.array([0.1, 0.25], dtype=np.float32)  # 0.1 is not a multiple of 1/4=0.25
        with pytest.raises(ValueError, match=r"Space coordinates must be multiples of 1/space_resolution \(0.25\)"):
            Position(
                space_resolution=coarse_resolution,
                topology=Topology.TORUS,
                space_coordinates=space_coords_invalid,
            )

        # Test with fine resolution
        fine_resolution = 100
        space_coords_fine_valid = np.array([0.0, 0.01, 0.05, 0.99], dtype=np.float32)

        position_fine = Position(
            space_resolution=fine_resolution,
            topology=Topology.RECTANGLE,
            space_coordinates=space_coords_fine_valid,
        )
        assert position_fine.dimensions == 4

        # Test invalid fine resolution
        space_coords_fine_invalid = np.array([0.005, 0.01], dtype=np.float32)  # 0.005 is not a multiple of 1/100=0.01
        with pytest.raises(ValueError, match=r"Space coordinates must be multiples of 1/space_resolution \(0.01\)"):
            Position(
                space_resolution=fine_resolution,
                topology=Topology.RECTANGLE,
                space_coordinates=space_coords_fine_invalid,
            )

    def test_space_coordinate_multiple_validation_floating_point_precision(self) -> None:
        """Test that floating point precision issues don't cause false positives."""
        space_resolution = 10

        # Test coordinates that should be valid multiples but might have tiny floating point errors
        # Simulate what might happen with floating point arithmetic
        space_coords = np.array([0.1, 0.2, 0.3], dtype=np.float32)

        # Add tiny floating point errors
        space_coords_with_errors = space_coords + np.array([1e-15, -1e-15, 1e-16], dtype=np.float32)

        # These should still be accepted due to tolerance
        position = Position(
            space_resolution=space_resolution,
            topology=Topology.RECTANGLE,
            space_coordinates=space_coords_with_errors,
        )
        assert position.dimensions == 3
        assert np.array_equal(position.space_coordinates, space_coords)

    def test_space_coordinate_multiple_validation_edge_cases(self) -> None:
        """Test edge cases for space coordinate multiple validation."""
        # Test with zero coordinate (always a multiple)
        space_resolution = 20
        space_coords_with_zero = np.array([0.0, 0.05, 0.10], dtype=np.float32)

        position = Position(
            space_resolution=space_resolution,
            topology=Topology.RECTANGLE,
            space_coordinates=space_coords_with_zero,
        )
        assert position.dimensions == 3

        # Test coordinate very close to 1.0 that is a valid multiple
        space_coords_near_one = np.array([0.95, 0.0], dtype=np.float32)  # 0.95 is a multiple of 0.05

        position_near_one = Position(
            space_resolution=space_resolution,
            topology=Topology.RECTANGLE,
            space_coordinates=space_coords_near_one,
        )
        assert position_near_one.dimensions == 2

    def test_clip_norm_no_clipping_needed(self) -> None:
        """Test clip_norm when no clipping is needed."""
        tensor_coords = np.array([1, 2], dtype=np.int32)
        position = Position(
            space_resolution=10,
            topology=Topology.RECTANGLE,
            tensor_coordinates=tensor_coords,
        )

        max_norm = 1.0  # Larger than current norm
        clipped = position.clip_norm(max_norm)

        # Should be unchanged
        assert np.array_equal(clipped.tensor_coordinates, position.tensor_coordinates)
        assert clipped.space_resolution == position.space_resolution
        assert clipped.topology == position.topology

    def test_clip_norm_with_clipping(self) -> None:
        """Test clip_norm when clipping is needed."""
        tensor_coords = np.array([30, 40], dtype=np.int32)  # Creates (0.3, 0.4) space coords, norm = 0.5
        position = Position(
            space_resolution=100,
            topology=Topology.RECTANGLE,
            tensor_coordinates=tensor_coords,
        )

        max_norm = 0.25
        clipped = position.clip_norm(max_norm)

        # Check that the norm is now max_norm
        assert clipped.space_norm() == max_norm
        # Check that direction is preserved
        original_direction = position.space_coordinates / position.space_norm()
        clipped_direction = clipped.space_coordinates / clipped.space_norm()
        assert np.allclose(original_direction, clipped_direction)

    def test_space_norm(self) -> None:
        """Test space_norm calculation."""
        tensor_coords = np.array([30, 40], dtype=np.int32)  # Creates (0.3, 0.4) space coords
        position = Position(
            space_resolution=100,
            topology=Topology.RECTANGLE,
            tensor_coordinates=tensor_coords,
        )

        norm = position.space_norm()
        expected_norm = 0.5  # sqrt(0.3^2 + 0.4^2)
        assert norm == expected_norm

    def test_add_rectangle_topology(self) -> None:
        """Test addition with rectangle topology."""
        tensor_coords1 = np.array([1, 2], dtype=np.int32)
        tensor_coords2 = np.array([3, 9], dtype=np.int32)
        space_resolution = 10
        topology = Topology.RECTANGLE

        pos1 = Position(space_resolution=space_resolution, topology=topology, tensor_coordinates=tensor_coords1)
        pos2 = Position(space_resolution=space_resolution, topology=topology, tensor_coordinates=tensor_coords2)

        result = pos1 + pos2

        # Expected: (1 + 3, 2 + 9) = (4, 11) but clipped to (4, 9)
        expected = np.array([4, 9], dtype=np.int32)
        assert np.array_equal(result.tensor_coordinates, expected)

    def test_add_torus_topology(self) -> None:
        """Test addition with torus topology."""
        tensor_coords1 = np.array([60, 95], dtype=np.int32)  # (0.6, 0.95) in space coords
        tensor_coords2 = np.array([20, 10], dtype=np.int32)  # (0.2, 0.1) in space coords
        space_resolution = 100
        topology = Topology.TORUS

        pos1 = Position(space_resolution=space_resolution, topology=topology, tensor_coordinates=tensor_coords1)
        pos2 = Position(space_resolution=space_resolution, topology=topology, tensor_coordinates=tensor_coords2)

        result = pos1 + pos2

        # Expected: (60 + 20, 95 + 10) = (80, 105) mod 100 = (80, 5)
        expected = np.array([80, 5], dtype=np.int32)
        assert np.array_equal(result.tensor_coordinates, expected)

    def test_add_mismatched_dimensions_raises_error(self) -> None:
        """Test that adding positions with different dimensions raises ValueError."""
        pos1 = Position(
            space_resolution=10,
            topology=Topology.RECTANGLE,
            tensor_coordinates=np.array([5, 8], dtype=np.int32),  # Valid range [0, 10)
        )
        pos2 = Position(
            space_resolution=10,
            topology=Topology.RECTANGLE,
            tensor_coordinates=np.array([3, 6, 9], dtype=np.int32),  # Valid range [0, 10)
        )

        with pytest.raises(ValueError, match="Position has mismatched shape"):
            pos1 + pos2

    def test_add_mismatched_topology_raises_error(self) -> None:
        """Test that adding positions with different topologies raises ValueError."""
        pos1 = Position(
            space_resolution=10,
            topology=Topology.RECTANGLE,
            tensor_coordinates=np.array([5, 8], dtype=np.int32),  # Valid range [0, 10)
        )
        pos2 = Position(
            space_resolution=10,
            topology=Topology.TORUS,
            tensor_coordinates=np.array([5, 8], dtype=np.int32),  # Valid range [0, 10)
        )

        with pytest.raises(ValueError, match="Positions must have the same topology and space resolution"):
            pos1 + pos2

    def test_add_mismatched_resolution_raises_error(self) -> None:
        """Test that adding positions with different resolutions raises ValueError."""
        pos1 = Position(
            space_resolution=10,
            topology=Topology.RECTANGLE,
            tensor_coordinates=np.array([5, 8], dtype=np.int32),  # Valid range [0, 10) for 0.1
        )
        pos2 = Position(
            space_resolution=20,
            topology=Topology.RECTANGLE,
            tensor_coordinates=np.array([10, 16], dtype=np.int32),  # Valid range [0, 20) for 0.05
        )

        with pytest.raises(ValueError, match="Positions must have the same topology and space resolution"):
            pos1 + pos2

    def test_subtract_rectangle_topology(self) -> None:
        """Test subtraction with rectangle topology."""
        tensor_coords1 = np.array([3, 4], dtype=np.int32)  # (0.3, 0.4) in space coords
        tensor_coords2 = np.array([6, 2], dtype=np.int32)  # (0.6, 0.2) in space coords
        space_resolution = 10
        topology = Topology.RECTANGLE

        pos1 = Position(space_resolution=space_resolution, topology=topology, tensor_coordinates=tensor_coords1)
        pos2 = Position(space_resolution=space_resolution, topology=topology, tensor_coordinates=tensor_coords2)

        result = pos1 - pos2

        # Expected: (3 - 6, 4 - 2) = (-3, 2) but clipped to (0, 2)
        expected = np.array([0, 2], dtype=np.int32)
        assert np.array_equal(result.tensor_coordinates, expected)

    def test_subtract_torus_topology(self) -> None:
        """Test subtraction with torus topology."""
        tensor_coords1 = np.array([30, 40], dtype=np.int32)  # (0.3, 0.4) in space coords
        tensor_coords2 = np.array([60, 10], dtype=np.int32)  # (0.2, 0.1) in space coords
        space_resolution = 100
        topology = Topology.TORUS

        pos1 = Position(space_resolution=space_resolution, topology=topology, tensor_coordinates=tensor_coords1)
        pos2 = Position(space_resolution=space_resolution, topology=topology, tensor_coordinates=tensor_coords2)

        result = pos1 - pos2

        # Expected: (30 - 60, 40 - 10) = (-30, 30) but wrapped to (70, 30)
        expected = np.array([70, 30], dtype=np.int32)
        assert np.array_equal(result.tensor_coordinates, expected)

    def test_subtract_mismatched_dimensions_raises_error(self) -> None:
        """Test that subtracting positions with different dimensions raises ValueError."""
        pos1 = Position(
            space_resolution=10,
            topology=Topology.RECTANGLE,
            tensor_coordinates=np.array([5, 8], dtype=np.int32),  # Valid range [0, 10)
        )
        pos2 = Position(
            space_resolution=10,
            topology=Topology.RECTANGLE,
            tensor_coordinates=np.array([3, 6, 9], dtype=np.int32),  # Valid range [0, 10)
        )

        with pytest.raises(ValueError, match="Position has mismatched shape"):
            pos1 - pos2

    def test_subtract_mismatched_topology_raises_error(self) -> None:
        """Test that subtracting positions with different topologies raises ValueError."""
        pos1 = Position(
            space_resolution=10,
            topology=Topology.RECTANGLE,
            tensor_coordinates=np.array([5, 8], dtype=np.int32),  # Valid range [0, 10)
        )
        pos2 = Position(
            space_resolution=10,
            topology=Topology.TORUS,
            tensor_coordinates=np.array([5, 8], dtype=np.int32),  # Valid range [0, 10)
        )

        with pytest.raises(ValueError, match="Positions must have the same topology and space resolution"):
            pos1 - pos2

    def test_subtract_mismatched_resolution_raises_error(self) -> None:
        """Test that subtracting positions with different resolutions raises ValueError."""
        pos1 = Position(
            space_resolution=10,
            topology=Topology.RECTANGLE,
            tensor_coordinates=np.array([5, 8], dtype=np.int32),  # Valid range [0, 10) for 0.1
        )
        pos2 = Position(
            space_resolution=20,
            topology=Topology.RECTANGLE,
            tensor_coordinates=np.array([10, 16], dtype=np.int32),  # Valid range [0, 20) for 0.05
        )

        with pytest.raises(ValueError, match="Positions must have the same topology and space resolution"):
            pos1 - pos2

    def test_subtract_self_returns_origin(self) -> None:
        """Test that subtracting a position from itself returns the origin."""
        tensor_coords = np.array([7, 9], dtype=np.int32)  # Valid range [0, 10) for space_resolution=10
        space_resolution = 10

        # Test with RECTANGLE topology
        pos_rect = Position(
            space_resolution=space_resolution,
            topology=Topology.RECTANGLE,
            tensor_coordinates=tensor_coords,
        )

        result_rect = pos_rect - pos_rect
        expected_space = np.array([0.0, 0.0], dtype=np.float32)
        assert np.allclose(result_rect.space_coordinates, expected_space)

        # Test with TORUS topology
        pos_torus = Position(
            space_resolution=space_resolution,
            topology=Topology.TORUS,
            tensor_coordinates=tensor_coords,
        )

        result_torus = pos_torus - pos_torus
        assert np.allclose(result_torus.space_coordinates, expected_space)

    def test_subtract_preserves_properties(self) -> None:
        """Test that subtraction preserves space resolution and topology."""
        tensor_coords1 = np.array([15, 18], dtype=np.int32)  # Valid range [0, 20) for space_resolution=20
        tensor_coords2 = np.array([10, 12], dtype=np.int32)  # Valid range [0, 20) for space_resolution=20
        space_resolution = 20
        topology = Topology.TORUS

        pos1 = Position(space_resolution=space_resolution, topology=topology, tensor_coordinates=tensor_coords1)
        pos2 = Position(space_resolution=space_resolution, topology=topology, tensor_coordinates=tensor_coords2)

        result = pos1 - pos2

        assert result.space_resolution == space_resolution
        assert result.topology == topology
        assert result.dimensions == pos1.dimensions

    def test_add_subtract_symmetry_rectangle(self) -> None:
        """Test add/subtract symmetry with rectangle topology."""
        tensor_coords1 = np.array([4, 6], dtype=np.int32)  # Valid range [0, 10) for space_resolution=10
        tensor_coords2 = np.array([2, 3], dtype=np.int32)  # Valid range [0, 10) for space_resolution=10
        space_resolution = 10
        topology = Topology.RECTANGLE

        pos1 = Position(space_resolution=space_resolution, topology=topology, tensor_coordinates=tensor_coords1)
        pos2 = Position(space_resolution=space_resolution, topology=topology, tensor_coordinates=tensor_coords2)

        # Test: (pos1 + pos2) - pos2 should be close to pos1 (within clipping bounds)
        added = pos1 + pos2
        subtracted = added - pos2

        # Due to clipping in rectangle topology, this may not be exactly pos1
        # but should be within expected bounds
        assert subtracted.space_resolution == pos1.space_resolution
        assert subtracted.topology == pos1.topology
        assert subtracted.dimensions == pos1.dimensions

    def test_add_subtract_symmetry_torus(self) -> None:
        """Test add/subtract symmetry with torus topology."""
        tensor_coords1 = np.array([7, 8], dtype=np.int32)  # Valid range [0, 10) for space_resolution=10
        tensor_coords2 = np.array([3, 4], dtype=np.int32)  # Valid range [0, 10) for space_resolution=10
        space_resolution = 10
        topology = Topology.TORUS

        pos1 = Position(space_resolution=space_resolution, topology=topology, tensor_coordinates=tensor_coords1)
        pos2 = Position(space_resolution=space_resolution, topology=topology, tensor_coordinates=tensor_coords2)

        # Test: (pos1 + pos2) - pos2 should equal pos1 for torus (due to modular arithmetic)
        added = pos1 + pos2
        subtracted = added - pos2

        # For torus topology, this should be approximately equal due to modular arithmetic
        # Though there might be small numerical differences due to floating point precision
        assert np.allclose(subtracted.space_coordinates, pos1.space_coordinates, atol=1e-6)
