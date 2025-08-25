from collections import namedtuple

import numpy as np


class Point(namedtuple("Point", "x, y")):
    def to_array(self):
        return np.array(self)

    def __sub__(self, other: "Point") -> "Point":
        """Subtract one from another."""
        return Point(self.x - other.x, self.y - other.y)

    def __add__(self, other: "Point") -> "Point":
        """Add one to another."""
        return Point(self.x + other.x, self.y + other.y)

    def distance_to(self, other: "Point"):
        return np.linalg.norm((self - other).to_array())
