from enum import Enum


class Direction(Enum):
    LEFT = 0
    UP = 1
    RIGHT = 2
    DOWN = 3

    @property
    def opposite(self) -> "Direction":
        return Direction((self.value + 2) % 4)

    @property
    def cw(self) -> "Direction":
        return Direction((self.value + 1) % 4)

    @property
    def ccw(self) -> "Direction":
        return Direction((self.value - 1) % 4)
