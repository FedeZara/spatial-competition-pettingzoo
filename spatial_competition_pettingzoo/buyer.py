from spatial_competition_pettingzoo.position import Position
from spatial_competition_pettingzoo.seller import Seller


class Buyer:
    def __init__(
        self,
        position: Position,
        value: float,
        quality_taste: float,
        distance_factor: float,
    ) -> None:
        self.position = position
        self.value = value
        self.quality_taste = quality_taste
        self.distance_factor = distance_factor

    def value_for_seller(self, seller: Seller) -> float:
        return (
            self.value
            - self.distance_factor * self.position.distance(seller.position)
            + self.quality_taste * seller.quality
        )

    def reward_for_seller(self, seller: Seller) -> float:
        return self.value_for_seller(seller) - seller.price
