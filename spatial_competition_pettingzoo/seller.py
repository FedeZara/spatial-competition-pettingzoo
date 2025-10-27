from spatial_competition_pettingzoo.position import Position


class Seller:
    def __init__(
        self,
        idx: int,
        position: Position,
        price: float,
        quality: float,
    ) -> None:
        self.idx = idx
        self.position = position
        self.price = price
        self.quality = quality
        self._last_movement_size = 0.0

    @property
    def agent_id(self) -> str:
        return f"seller_{self.idx}"

    def set_price(self, new_price: float) -> None:
        self.price = new_price

    def set_quality(self, new_quality: float) -> None:
        self.quality = new_quality

    def move(self, movement: Position) -> None:
        self.position = self.position + movement
        self._last_movement_size = movement.space_norm()
