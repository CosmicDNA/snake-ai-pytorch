import random

import pygame

from snake_ai_pytorch.models.collision import Collision
from snake_ai_pytorch.models.direction import Direction
from snake_ai_pytorch.models.point import Point
from snake_ai_pytorch.models.renderer import Renderer
from snake_ai_pytorch.views.visual_configuration import BLOCK_SIZE, SPEED


class SnakeGame:
    renderer: Renderer = None

    def __init__(self, w=640, h=480, render_mode="human"):
        self.w = w
        self.h = h
        # Game dimensions in grid units
        self.grid_w = self.w // BLOCK_SIZE
        self.grid_h = self.h // BLOCK_SIZE

        if render_mode == "human":
            self.renderer = Renderer(self)
        self.reset()

    def reset(self):
        """Reset the game to its initial state."""
        # Note: The renderer's reset is called at the end of this method
        # to ensure it draws the fully initialized game state.
        # init game state
        self.direction = Direction.RIGHT

        # The game now operates on a grid, not pixels.
        self.head = Point(self.grid_w / 2, self.grid_h / 2)
        self.snake = [self.head, Point(self.head.x - 1, self.head.y), Point(self.head.x - 2, self.head.y)]

        self.score = 0
        self.food = None
        self._place_food()

        if self.renderer:
            # Perform a full redraw for the new game state.
            self.renderer.reset()

    def _place_food(self):
        # Create a set of all possible grid coordinates
        all_points = {Point(x, y) for x in range(self.grid_w) for y in range(self.grid_h)}

        # Create a set of coordinates occupied by the snake
        snake_points = set(self.snake)

        # Find the coordinates that are not occupied by the snake
        available_points = list(all_points - snake_points)

        if available_points:
            self.food = random.choice(available_points)  # nosec B311
        else:
            # No available space left; the snake has won by filling the board.
            # The game will end on the next move as the snake has nowhere to go.
            self.food = None

    def _eat_food(self):
        self.score += 1
        if self.renderer:
            self.renderer.play_eat_sound()
        self._place_food()

    def _collide(self, collision: Collision):
        if self.renderer:
            if collision == Collision.BOUNDARY:
                self.renderer.play_collide_sound()
            elif collision == Collision.ITSELF:
                self.renderer.play_itself_sound()

    def play_step(self, direction: Direction):
        # 2. move
        self._move(direction)  # update the head
        self.snake.insert(0, self.head)

        # 3. check if game over
        game_over = False
        collision = self.get_collision()
        if collision:
            game_over = True
            self._collide(collision)
        else:
            # 4. place new food or just move
            if self.head == self.food:
                self._eat_food()
            else:
                self.snake.pop()

        # 6. return game over and score
        return game_over, self.score

    def get_collision(self, pt=None) -> Collision | None:
        if pt is None:
            pt = self.head
        # hits boundary
        # The check is now against grid dimensions.
        if pt.x >= self.grid_w or pt.x < 0 or pt.y >= self.grid_h or pt.y < 0:
            return Collision.BOUNDARY
        # hits itself
        return Collision.ITSELF if pt in self.snake[1:] else None

    def is_collision(self, pt=None):
        # The collision check now correctly uses the provided point.
        return self.get_collision(pt) is not None

    def _move(self, direction: Direction):
        self.head += direction.vector

    def render(self, render_fps=SPEED):
        if self.renderer is None:
            return

        # This is the standard way to handle rendering and events in a Pygame-based gym env
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        self.renderer.render(render_fps)
