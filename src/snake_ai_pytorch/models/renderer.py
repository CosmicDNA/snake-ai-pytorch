import importlib.resources
import logging
import os
import sys
from typing import TYPE_CHECKING

import pygame

from snake_ai_pytorch.views.visual_configuration import BLOCK_SIZE, BORDER_SIZE, SPEED, FontConfig, GameColors, SoundConfig

if TYPE_CHECKING:
    from snake_ai_pytorch.models.snake_game import SnakeGame


class Renderer:
    def __init__(self, game: "SnakeGame"):
        # On Linux-based systems like WSL2, force SDL to use the 'pulse' audio driver,
        # which is often forwarded correctly. This must be set before pygame is initialized.
        if sys.platform == "linux":
            os.environ["SDL_AUDIODRIVER"] = "pulse"

        pygame.init()

        self.game = game

        # Dynamically get the root package name to load resources.
        root_package = __package__.split(".")[0]

        try:
            font_path_traversable = importlib.resources.files(root_package).joinpath(FontConfig.path)
            with importlib.resources.as_file(font_path_traversable) as font_path:
                self.font = pygame.font.Font(font_path, FontConfig.size)
        except (FileNotFoundError, pygame.error) as e:
            logging.warning(f"Could not load custom font '{FontConfig.path}': {e}. Falling back to default font.")
            self.font = pygame.font.Font(None, FontConfig.size)

        pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=256)
        try:
            sound_path_traversable = importlib.resources.files(root_package).joinpath(SoundConfig.eat_path)
            with importlib.resources.as_file(sound_path_traversable) as sound_path:
                self.eat_sound = pygame.mixer.Sound(sound_path)
        except (FileNotFoundError, pygame.error) as e:
            logging.warning(f"Could not initialise eat sound: {e}")
        try:
            collide_path_traversable = importlib.resources.files(root_package).joinpath(SoundConfig.collide_path)
            with importlib.resources.as_file(collide_path_traversable) as collide_path:
                self.collide_sound = pygame.mixer.Sound(collide_path)
        except (FileNotFoundError, pygame.error) as e:
            logging.warning(f"Could not initialise collide sound: {e}")
        try:
            itself_path_traversable = importlib.resources.files(root_package).joinpath(SoundConfig.itself_path)
            with importlib.resources.as_file(itself_path_traversable) as itself_path:
                self.itself_sound = pygame.mixer.Sound(itself_path)
        except (FileNotFoundError, pygame.error) as e:
            logging.warning(f"Could not initialise collide sound: {e}")
        # init display
        self.display = pygame.display.set_mode((self.game.w, self.game.h))
        pygame.display.set_caption("Snake")
        self.clock = pygame.time.Clock()
        self.INNER_BLOCK_SIZE = BLOCK_SIZE - 2 * BORDER_SIZE

    def play_eat_sound(self):
        if self.eat_sound:
            self.eat_sound.play()

    def play_collide_sound(self):
        if self.collide_sound:
            self.collide_sound.play()

    def play_itself_sound(self):
        if self.itself_sound:
            self.itself_sound.play()

    def render(self, render_fps=SPEED):
        self.display.fill(GameColors.BLACK)

        for pt in self.game.snake:
            pygame.draw.rect(self.display, GameColors.BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(
                self.display, GameColors.BLUE2, pygame.Rect(pt.x + BORDER_SIZE, pt.y + BORDER_SIZE, self.INNER_BLOCK_SIZE, self.INNER_BLOCK_SIZE)
            )

        pygame.draw.rect(self.display, GameColors.RED, pygame.Rect(self.game.food.x, self.game.food.y, BLOCK_SIZE, BLOCK_SIZE))

        text = self.font.render("Score: " + str(self.game.score), True, GameColors.WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()
        self.clock.tick(render_fps)
