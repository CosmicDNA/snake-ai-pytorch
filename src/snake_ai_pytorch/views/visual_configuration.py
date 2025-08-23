from pathlib import Path

import pygame


class GameColors:
    WHITE = (255, 255, 255)
    RED = (200, 0, 0)
    BLUE1 = (0, 0, 255)
    BLUE2 = (0, 100, 255)
    BLACK = (0, 0, 0)


# Construct a path to the font file relative to this script's location
BASE_DIR = Path(__file__).resolve().parent
font = pygame.font.Font(BASE_DIR / "assets" / "arial.ttf", 25)

BLOCK_SIZE = 20
SPEED = 75
