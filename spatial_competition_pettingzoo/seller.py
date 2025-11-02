from spatial_competition_pettingzoo.position import Position


class Seller:
    def __init__(
        self,
        agent_id: str,
        position: Position,
        price: float,
        quality: float,
    ) -> None:
        self._agent_id = agent_id
        self.position = position
        self.price = price
        self.quality = quality
        self._last_movement_size = 0.0
        self._running_sales = 0
        self._total_sales = 0

    @property
    def agent_id(self) -> str:
        return self._agent_id

    @property
    def running_sales(self) -> int:
        return self._running_sales

    @property
    def total_sales(self) -> int:
        return self._total_sales

    def set_price(self, new_price: float) -> None:
        self.price = new_price

    def set_quality(self, new_quality: float) -> None:
        self.quality = new_quality

    def move(self, movement: Position) -> None:
        self.position = self.position + movement
        self._last_movement_size = movement.space_norm()

    def reset_running_sales(self) -> None:
        self._running_sales = 0

    def sell(self) -> None:
        self._running_sales += 1
        self._total_sales += 1

    def step_reward(self, production_cost_factor: float, movement_cost: float) -> float:
        production_cost = production_cost_factor * self.quality**2
        movement_cost = movement_cost * self._last_movement_size
        revenue = self._running_sales * self.price
        return revenue - production_cost - movement_cost
