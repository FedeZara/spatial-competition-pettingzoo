from dataclasses import dataclass

from spatial_competition_pettingzoo.position import Position


@dataclass
class Action:
    movement: Position
    price: float
    quality: float | None
