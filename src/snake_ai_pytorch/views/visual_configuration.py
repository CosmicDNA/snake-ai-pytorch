class GameColors:
    WHITE = (255, 255, 255)
    RED = (200, 0, 0)
    BLUE1 = (0, 0, 255)
    BLUE2 = (0, 100, 255)
    BLACK = (0, 0, 0)


class FontConfig:
    # Path to the font file relative to the package root.
    path = "assets/arial.ttf"
    size = 25


class SoundConfig:
    # Path to the sound file relative to the package root.
    eat_path = "assets/eat.wav"
    collide_path = "assets/collide.wav"
    itself_path = "assets/itself.wav"


BLOCK_SIZE = 20
SPEED = 75
BORDER_SIZE = 4
