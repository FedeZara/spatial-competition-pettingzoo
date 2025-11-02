from abc import ABC, abstractmethod

import numpy as np

from spatial_competition_pettingzoo.competition_space import CompetitionSpace
from spatial_competition_pettingzoo.position import Position
from spatial_competition_pettingzoo.topology import Topology


class ViewScope(ABC):
    """
    View scope modes controlling the spatial range of observation.

    These modes determine how much of the map sellers can see, independent
    of what type of information they can observe.

    Modes:
        LimitedViewScope: Sellers have restricted spatial vision (vision_radius parameter)
                and only see information within that range. Creates local
                information and spatial competition effects.

        CompleteViewScope: Sellers can see the entire space. All available information
                 (based on InformationLevel) is visible across the full spatial area.
    """

    @property
    @abstractmethod
    def is_limited(self) -> bool:
        pass

    @property
    @abstractmethod
    def is_complete(self) -> bool:
        pass

    @abstractmethod
    def seller_view_shape(self, space_resolution: int, dimensions: int) -> tuple[int, ...]:
        pass

    @abstractmethod
    def build_seller_subspace(self, space: CompetitionSpace, seller_position: Position) -> CompetitionSpace:
        pass


class LimitedViewScope(ViewScope):
    def __init__(self, vision_radius: int) -> None:
        self.vision_radius = vision_radius

    @property
    def is_limited(self) -> bool:
        return True

    @property
    def is_complete(self) -> bool:
        return False

    def seller_view_shape(self, _: int, dimensions: int) -> tuple[int, ...]:
        return (2 * self.vision_radius + 1,) * dimensions

    def build_seller_subspace(self, space: CompetitionSpace, seller_position: Position) -> CompetitionSpace:
        match space.topology:
            case Topology.RECTANGLE:
                assert self.vision_radius < space.space_resolution, (
                    "Vision radius must be less than space resolution for rectangle topology"
                )
            case Topology.TORUS:
                assert self.vision_radius < space.space_resolution / 2, (
                    "Vision radius must be less than half of space resolution for torus topology"
                )

        vision_radius_position = Position(
            space_resolution=space.space_resolution,
            topology=space.topology,
            tensor_coordinates=np.array([self.vision_radius] * space.dimensions),
        )

        base = seller_position - vision_radius_position
        extent = seller_position + vision_radius_position

        seller_subspace = space.subspace(base, extent)

        assert seller_subspace.is_in_subspace(seller_position)

        return seller_subspace


class CompleteViewScope(ViewScope):
    @property
    def is_limited(self) -> bool:
        return False

    @property
    def is_complete(self) -> bool:
        return True

    def seller_view_shape(self, space_resolution: int, dimensions: int) -> tuple[int, ...]:
        return (space_resolution,) * dimensions

    def build_seller_subspace(self, space: CompetitionSpace, _: Position) -> CompetitionSpace:
        return space
