from enum import Enum

from snake_ai_pytorch.models.point import Point


class Direction(Enum):
    LEFT = 0
    UP = 1
    RIGHT = 2
    DOWN = 3

    @property
    def opposite(self) -> "Direction":
        """Returns the opposite direction."""
        return Direction((self.value + 2) % 4)

    @property
    def cw(self) -> "Direction":
        """Returns the clockwise direction."""
        return Direction((self.value + 1) % 4)

    @property
    def ccw(self) -> "Direction":
        """Returns the counter-clockwise direction."""
        return Direction((self.value - 1) % 4)

    @property
    def vector(self) -> Point:
        """Returns the unit vector for the direction."""
        return _VECTOR_MAP[self]


# This map provides a fast, O(1) lookup for a direction's vector.
# It's defined at the module level after the Direction enum is created
# to ensure the enum members are available to be used as keys.
_VECTOR_MAP = {
    Direction.LEFT: Point(-1, 0),
    Direction.RIGHT: Point(1, 0),
    Direction.UP: Point(0, -1),
    Direction.DOWN: Point(0, 1),
}
