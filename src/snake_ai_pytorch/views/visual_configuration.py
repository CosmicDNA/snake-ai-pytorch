from pathlib import Path


class GameColors:
    WHITE = (255, 255, 255)
    RED = (200, 0, 0)
    BLUE1 = (0, 0, 255)
    BLUE2 = (0, 100, 255)
    BLACK = (0, 0, 0)


BASE_DIR = Path(__file__).resolve().parent.parent


class FontConfig:
    # Construct a path to the font file relative to this script's location
    path = BASE_DIR / "assets" / "arial.ttf"
    size = 25


BLOCK_SIZE = 20
SPEED = 75
