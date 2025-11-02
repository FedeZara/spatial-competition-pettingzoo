import random

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
        """
        Calculate the value for a seller.

        Formula is value - distance_factor * distance + quality_taste * quality
        """
        return (
            self.value
            - self.distance_factor * self.position.distance(seller.position)
            + self.quality_taste * seller.quality
        )

    def reward_for_seller(self, seller: Seller) -> float:
        """
        Calculate the reward for a seller.

        Formula is value - price - distance_factor * distance + quality_taste * quality
        """
        return self.value_for_seller(seller) - seller.price

    def choose_seller_and_buy(self, sellers: list[Seller]) -> bool:
        """
        Choose a seller and buy from them if the reward is positive.

        The buyer chooses the seller with the highest reward and buys from them if the reward is positive.
        The buyer then sells to the chosen seller.

        Returns True if the buyer bought from a seller, False otherwise.
        """
        if len(sellers) == 0:
            return False

        sellers = sellers.copy()
        random.shuffle(sellers)

        best_seller = max(sellers, key=self.reward_for_seller)
        if self.reward_for_seller(best_seller) > 0:
            best_seller.sell()
            return True

        return False
