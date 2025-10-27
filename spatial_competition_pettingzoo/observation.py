from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from gymnasium import spaces

from spatial_competition_pettingzoo.enums import InformationLevel
from spatial_competition_pettingzoo.position import Position

if TYPE_CHECKING:
    from spatial_competition_pettingzoo.competition_space import CompetitionSpace


class Observation:
    # Constants for placeholder values
    NO_BUYER_PLACEHOLDER = -999
    NO_SELLER_PRICE_PLACEHOLDER = -999
    NO_SELLER_QUALITY_PLACEHOLDER = -999

    def __init__(
        self,
        own_price: float,
        own_quality: float,
        own_position: Position,
        local_view: np.ndarray,
        buyers: np.ndarray,
        sellers_price: np.ndarray,
        sellers_quality: np.ndarray,
    ) -> None:
        self._own_price = own_price
        self._own_quality = own_quality
        self._own_position = own_position
        self._local_view = local_view
        self._buyers = buyers
        self._sellers_price = sellers_price
        self._sellers_quality = sellers_quality

    @classmethod
    def build_from_competition_space(
        cls,
        space: CompetitionSpace,
        information_level: InformationLevel,
        vision_radius: int,
        agent_id: str,
    ) -> Observation:
        assert space.is_full_space, "Space must be the full space to build an observation"

        seller = space.sellers[agent_id]

        vision_radius_position = Position(
            space_resolution=space.space_resolution,
            topology=space.topology,
            tensor_coordinates=np.array([vision_radius] * space.dimensions),
        )

        base = seller.position - vision_radius_position
        extent = seller.position + vision_radius_position

        seller_subspace = space.subspace(base, extent)

        assert seller_subspace.is_in_subspace(seller.position)

        return cls(
            own_price=seller.price,
            own_quality=seller.quality,
            own_position=seller.position,
            local_view=cls._build_local_view_observation(agent_id, seller_subspace),
            buyers=cls._build_buyers_observation(agent_id, seller_subspace)
            if information_level in (InformationLevel.LIMITED, InformationLevel.COMPLETE)
            else None,
            sellers_price=cls._build_sellers_price_observation(agent_id, seller_subspace)
            if information_level == InformationLevel.COMPLETE
            else None,
            sellers_quality=cls._build_sellers_quality_observation(agent_id, seller_subspace)
            if information_level == InformationLevel.COMPLETE
            else None,
        )

    def get_observation(self) -> dict[str, Any]:
        return {
            "own_position": self._own_position,
            "own_price": self._own_price,
            "own_quality": self._own_quality,
            "local_view": self._local_view,
            **({"buyers": self._buyers} if self._buyers is not None else {}),
            **({"sellers_price": self._sellers_price} if self._sellers_price is not None else {}),
            **({"sellers_quality": self._sellers_quality} if self._sellers_quality is not None else {}),
        }

    @staticmethod
    def create_buyers_space(
        vision_radius: int,
        dimensions: int,
        max_valuation: float,
        max_quality: float,
    ) -> spaces.Box:
        """Create observation space for buyers information."""
        local_view_shape = (2 * vision_radius + 1,) * dimensions

        return spaces.Box(
            low=Observation.NO_BUYER_PLACEHOLDER,
            high=max_valuation + max_quality,
            shape=local_view_shape,
            dtype=np.float32,
        )

    @staticmethod
    def create_sellers_spaces(
        vision_radius: int,
        dimensions: int,
        max_price: float,
        max_quality: float,
    ) -> tuple[spaces.Box, spaces.Box]:
        """Create observation spaces for sellers price and quality information."""
        local_view_shape = (2 * vision_radius + 1,) * dimensions

        sellers_price_space = spaces.Box(
            low=Observation.NO_SELLER_PRICE_PLACEHOLDER,
            high=max_price,
            shape=local_view_shape,
            dtype=np.float32,
        )

        sellers_quality_space = spaces.Box(
            low=Observation.NO_SELLER_QUALITY_PLACEHOLDER,
            high=max_quality,
            shape=local_view_shape,
            dtype=np.float32,
        )

        return sellers_price_space, sellers_quality_space

    @staticmethod
    def create_observation_space(
        information_level: InformationLevel,
        dimensions: int,
        vision_radius: int,
        max_price: float,
        max_quality: float,
        max_valuation: float,
    ) -> spaces.Dict:
        """Create complete observation space based on information level."""
        # Start with basic spaces
        local_view_shape = (2 * vision_radius + 1,) * dimensions

        space_dict = {
            "own_position": spaces.Box(low=0, high=1, shape=(dimensions,), dtype=np.float32),
            "own_price": spaces.Box(low=0.0, high=max_price, dtype=np.float32),
            "own_quality": spaces.Box(low=0.0, high=max_quality, dtype=np.float32),
            # 0 = empty, 1 = self, 2 = other seller, 3 = buyer
            "local_view": spaces.Box(low=0, high=3, shape=local_view_shape, dtype=np.int8),
        }

        # Add buyers space for LIMITED and COMPLETE information levels
        if information_level in (InformationLevel.LIMITED, InformationLevel.COMPLETE):
            space_dict["buyers"] = Observation.create_buyers_space(
                vision_radius=vision_radius,
                dimensions=dimensions,
                max_valuation=max_valuation,
                max_quality=max_quality,
            )

        # Add sellers spaces for COMPLETE information level
        if information_level == InformationLevel.COMPLETE:
            sellers_price_space, sellers_quality_space = Observation.create_sellers_spaces(
                vision_radius=vision_radius,
                dimensions=dimensions,
                max_price=max_price,
                max_quality=max_quality,
            )
            space_dict["sellers_price"] = sellers_price_space
            space_dict["sellers_quality"] = sellers_quality_space

        return spaces.Dict(space_dict)

    @classmethod
    def _build_local_view_observation(cls, agent_id: str, seller_subspace: CompetitionSpace) -> np.ndarray:
        # n dimensional array of zeros, where each dimension is the size of the subspace
        local_view = np.zeros(seller_subspace.relative_extent.tensor_coordinates + 1, dtype=np.int8)

        for seller in seller_subspace.sellers.values():
            index = tuple(seller_subspace.relative_position(seller.position).tensor_coordinates)
            if seller.agent_id != agent_id:
                local_view[index] = 2
            else:
                local_view[index] = 1
        for buyer in seller_subspace.buyers:
            index = tuple(seller_subspace.relative_position(buyer.position).tensor_coordinates)
            local_view[index] = 3

        return local_view

    @classmethod
    def _build_buyers_observation(cls, agent_id: str, space: CompetitionSpace) -> np.ndarray:
        buyers = np.full(space.relative_extent.tensor_coordinates + 1, cls.NO_BUYER_PLACEHOLDER, dtype=np.float32)
        seller = space.sellers[agent_id]

        for buyer in space.buyers:
            index = tuple(space.relative_position(buyer.position).tensor_coordinates)
            buyers[index] = buyer.value_for_seller(seller)

        return buyers

    @classmethod
    def _build_sellers_price_observation(cls, agent_id: str, space: CompetitionSpace) -> np.ndarray:
        sellers_price = np.full(
            space.relative_extent.tensor_coordinates + 1, cls.NO_SELLER_PRICE_PLACEHOLDER, dtype=np.float32
        )

        for seller in space.sellers.values():
            if seller.agent_id == agent_id:
                continue
            index = tuple(space.relative_position(seller.position).tensor_coordinates)
            sellers_price[index] = seller.price

        return sellers_price

    @classmethod
    def _build_sellers_quality_observation(cls, agent_id: str, space: CompetitionSpace) -> np.ndarray:
        sellers_quality = np.full(
            space.relative_extent.tensor_coordinates + 1, cls.NO_SELLER_QUALITY_PLACEHOLDER, dtype=np.float32
        )

        for seller in space.sellers.values():
            if seller.agent_id == agent_id:
                continue
            index = tuple(space.relative_position(seller.position).tensor_coordinates)
            sellers_quality[index] = seller.quality

        return sellers_quality
