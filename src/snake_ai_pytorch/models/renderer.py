import logging
import os
from typing import TYPE_CHECKING

import pygame

from snake_ai_pytorch.views.visual_configuration import BLOCK_SIZE, BORDER_SIZE, SPEED, FontConfig, GameColors, SoundConfig

if TYPE_CHECKING:
    from snake_ai_pytorch.models.snake_game import SnakeGame


class Renderer:
    def __init__(self, game: "SnakeGame"):
        # Force SDL to use the pulse audio driver, which is forwarded by WSL2.
        # This must be set before pygame is initialized.
        os.environ["SDL_AUDIODRIVER"] = "pulse"
        pygame.init()
        self.game = game
        self.font = pygame.font.Font(FontConfig.path, FontConfig.size)
        self.eat_sound = None
        try:
            # Initialize with a smaller buffer for lower latency.
            self.eat_sound = pygame.mixer.Sound(SoundConfig.eat_path)
        except pygame.error as e:
            logging.warning(f"Could not initialise sound: {e}")
        # init display
        self.display = pygame.display.set_mode((self.game.w, self.game.h))
        pygame.display.set_caption("Snake")
        self.clock = pygame.time.Clock()
        self.INNER_BLOCK_SIZE = BLOCK_SIZE - 2 * BORDER_SIZE

    def play_eat_sound(self):
        if self.eat_sound:
            self.eat_sound.play()

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
