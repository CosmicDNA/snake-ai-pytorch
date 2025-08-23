import random

import pygame

from snake_ai_pytorch.models import Direction, Point
from snake_ai_pytorch.views import BLOCK_SIZE, SPEED, FontConfig, GameColors


class Renderer:
    def __init__(self, game):
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

class SnakeGame:
    renderer: Renderer = None
    def __init__(self, w=640, h=480, render_mode="human"):
        self.w = w
        self.h = h
        if render_mode == "human":
            self.renderer = Renderer(self)
        self.reset()

    def reset(self):
        # init game state
        self.direction = Direction.RIGHT

        self.head = Point(self.w / 2, self.h / 2)
        self.snake = [self.head, Point(self.head.x - BLOCK_SIZE, self.head.y), Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)]

        self.score = 0
        self.food = None
        self._place_food()

    def _place_food(self):
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def play_step(self, direction):
        # 2. move
        self._move(direction)  # update the head
        self.snake.insert(0, self.head)

        # 3. check if game over
        game_over = False
        if self.is_collision():
            game_over = True
        else:
            # 4. place new food or just move
            if self.head == self.food:
                self.score += 1
                self._place_food()
            else:
                self.snake.pop()

        # 6. return game over and score
        return game_over, self.score

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # hits itself
        return pt in self.snake[1:]

    def _move(self, direction):
        x = self.head.x
        y = self.head.y
        if direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)

    def render(self, render_fps=SPEED):
        if self.renderer is None:
            return

        # This is the standard way to handle rendering and events in a Pygame-based gym env
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        self.renderer.render(render_fps)
