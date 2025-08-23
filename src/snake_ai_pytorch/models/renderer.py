from typing import TYPE_CHECKING

import pygame

from snake_ai_pytorch.views.visual_configuration import BLOCK_SIZE, SPEED, FontConfig, GameColors

if TYPE_CHECKING:
    from snake_ai_pytorch.models.snake_game import SnakeGame


class Renderer:
    def __init__(self, game: "SnakeGame"):
        pygame.init()
        self.game = game
        self.font = pygame.font.Font(FontConfig.path, FontConfig.size)
        # init display
        self.display = pygame.display.set_mode((self.game.w, self.game.h))
        pygame.display.set_caption("Snake")
        self.clock = pygame.time.Clock()

    def render(self, render_fps=SPEED):
        self.display.fill(GameColors.BLACK)

        for pt in self.game.snake:
            pygame.draw.rect(self.display, GameColors.BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, GameColors.BLUE2, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))

        pygame.draw.rect(self.display, GameColors.RED, pygame.Rect(self.game.food.x, self.game.food.y, BLOCK_SIZE, BLOCK_SIZE))

        text = self.font.render("Score: " + str(self.game.score), True, GameColors.WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()
        self.clock.tick(render_fps)
