from spatial_competition_pettingzoo.position import Position


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
