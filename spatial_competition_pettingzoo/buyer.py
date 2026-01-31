import random

from spatial_competition_pettingzoo.position import Position
from spatial_competition_pettingzoo.seller import Seller


class Buyer:
    def __init__(
        self,
        position: Position,
        value: float | None,
        quality_taste: float | None,
        distance_factor: float,
    ) -> None:
        self.position = position
        self.value = value
        self.quality_taste = quality_taste
        self.distance_factor = distance_factor
        self.purchased_from_id: str | None = None

    @property
    def has_purchased(self) -> bool:
        """Check if this buyer has already made a purchase."""
        return self.purchased_from_id is not None

    def value_for_seller(self, seller: Seller) -> float:
        """
        Calculate the value for a seller.

        Formula is value - distance_factor * distance + quality_taste * quality
        """
        value = self.value if self.value is not None else 0

        distance_term = self.distance_factor * self.position.distance(seller.position)

        quality_term = 0.0
        if self.quality_taste is not None:
            assert seller.quality is not None
            quality_term = self.quality_taste * seller.quality

        return value - distance_term + quality_term

    def reward_for_seller(self, seller: Seller) -> float:
        """
        Calculate the reward for a seller.

        Formula is value - price - distance_factor * distance + quality_taste * quality
        """
        return self.value_for_seller(seller) - seller.price

    def get_preferred_seller(self, sellers: list[Seller]) -> Seller | None:
        """
        Get the seller this buyer would choose (highest positive reward).

        Returns the seller with the highest reward if that reward is positive,
        otherwise returns None.
        """
        if len(sellers) == 0:
            return None

        best_seller = max(sellers, key=self.reward_for_seller)
        # If the buyer has no value, they will always buy from the best seller
        if self.value is None or self.reward_for_seller(best_seller) > 0:
            return best_seller
        return None

    def choose_seller_and_buy(self, sellers: list[Seller]) -> bool:
        """
        Choose a seller and buy from them if the reward is positive.

        The buyer chooses the seller with the highest reward and buys from them if the reward is positive.
        Sets purchased_from to track which seller they bought from.

        Returns True if the buyer bought from a seller, False otherwise.
        """
        if self.has_purchased:
            return False  # Already bought

        # Shuffle to break ties randomly
        sellers = sellers.copy()
        random.shuffle(sellers)

        best_seller = self.get_preferred_seller(sellers)
        if best_seller is not None:
            best_seller.sell()
            self.purchased_from_id = best_seller.agent_id
            return True

        return False
