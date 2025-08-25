import numpy as np
import pygame

from snake_ai_pytorch.models.direction import Direction
from snake_ai_pytorch.models.point import Point
from snake_ai_pytorch.models.renderer import Renderer
from snake_ai_pytorch.views.visual_configuration import BLOCK_SIZE, SPEED


class SnakeGame:
    renderer: Renderer = None

    direction_map = {
        Direction.RIGHT: Point(BLOCK_SIZE, 0),
        Direction.LEFT: Point(-BLOCK_SIZE, 0),
        Direction.DOWN: Point(0, BLOCK_SIZE),
        Direction.UP: Point(0, -BLOCK_SIZE),
    }

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
        grid_dims = np.array([self.w, self.h]) // BLOCK_SIZE

        while True:
            # Generate random x, y block coordinates. `high` is exclusive.
            block_coords = np.random.randint(0, grid_dims, size=2)
            new_food = Point(*(block_coords * BLOCK_SIZE))

            if new_food not in self.snake:
                self.food = new_food
                break

    def _eat_food(self):
        self.score += 1
        if self.renderer:
            self.renderer.play_eat_sound()
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
                self._eat_food()
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
        self.head += self.direction_map[direction]

    def render(self, render_fps=SPEED):
        if self.renderer is None:
            return

        # This is the standard way to handle rendering and events in a Pygame-based gym env
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        self.renderer.render(render_fps)
