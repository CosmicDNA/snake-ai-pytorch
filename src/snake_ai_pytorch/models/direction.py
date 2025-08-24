from enum import Enum


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

    @property
    def opposite(self) -> "Direction":
        return Direction._opposit_direction_map[self]


# The map is defined outside the class and attached to it after definition.
# This is necessary because the enum members (LEFT, RIGHT, etc.) are not
# available within the class body at definition time.
Direction._opposit_direction_map = {
    Direction.LEFT: Direction.RIGHT,
    Direction.UP: Direction.DOWN,
    Direction.RIGHT: Direction.LEFT,
    Direction.DOWN: Direction.UP,
}
