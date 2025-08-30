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

        # Store previous state for optimized drawing
        self.prev_snake = []
        self.prev_food = None
        self.prev_score = -1
        self.prev_score_rect = None

    def reset(self):
        """Clears the screen and performs a full redraw of the current game state."""
        self.display.fill(GameColors.BLACK)

        # Draw snake
        for pt in self.game.snake:
            self._draw_snake_block(pt)

        # Draw food
        if self.game.food:
            self._draw_food_block(self.game.food)

        # Draw the initial score and store its rect
        score_text = self.font.render("Score: " + str(self.game.score), True, GameColors.WHITE)
        self.prev_score_rect = self.display.blit(score_text, (0, 0))

        pygame.display.flip()

        # Update previous state trackers
        self.prev_snake = self.game.snake[:]
        self.prev_food = self.game.food
        self.prev_score = self.game.score

    def _draw_snake_block(self, pt):
        """Draws a single block of the snake and returns its rect."""
        rect = pygame.Rect(pt.x * BLOCK_SIZE, pt.y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)
        pygame.draw.rect(self.display, GameColors.BLUE1, rect)
        pygame.draw.rect(
            self.display,
            GameColors.BLUE2,
            pygame.Rect(pt.x * BLOCK_SIZE + BORDER_SIZE, pt.y * BLOCK_SIZE + BORDER_SIZE, self.INNER_BLOCK_SIZE, self.INNER_BLOCK_SIZE),
        )
        return rect

    def _draw_food_block(self, pt):
        """Draws the food block and returns its rect."""
        rect = pygame.Rect(pt.x * BLOCK_SIZE, pt.y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)
        pygame.draw.rect(self.display, GameColors.RED, rect)
        return rect

    def _erase_block(self, pt):
        """Erases a block by drawing a black rectangle over it and returns its rect."""
        rect = pygame.Rect(pt.x * BLOCK_SIZE, pt.y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)
        pygame.draw.rect(self.display, GameColors.BLACK, rect)
        return rect

    def play_eat_sound(self):
        if hasattr(self, "eat_sound") and self.eat_sound:
            self.eat_sound.play()

    def play_collide_sound(self):
        if hasattr(self, "collide_sound") and self.collide_sound:
            self.collide_sound.play()

    def play_itself_sound(self):
        if hasattr(self, "itself_sound") and self.itself_sound:
            self.itself_sound.play()

    def render(self, render_fps=SPEED):
        """Optimized render function that only updates changed parts of the screen."""
        updated_rects = []
        score_needs_redraw = False

        # Find and erase the snake tail that was removed
        to_erase = set(self.prev_snake) - set(self.game.snake)
        for pt in to_erase:
            erased_rect = self._erase_block(pt)
            updated_rects.append(erased_rect)
            # If the erased tail was under the score, we need to redraw the score.
            if self.prev_score_rect and self.prev_score_rect.colliderect(erased_rect):
                score_needs_redraw = True

        # Draw the new snake head
        to_draw = set(self.game.snake) - set(self.prev_snake)
        for pt in to_draw:
            updated_rects.append(self._draw_snake_block(pt))

        # Handle food changes
        if self.game.food != self.prev_food:
            # Erase the old food's position only if the snake's head hasn't moved there.
            if self.prev_food and self.prev_food not in self.game.snake:
                updated_rects.append(self._erase_block(self.prev_food))

            if self.game.food:
                updated_rects.append(self._draw_food_block(self.game.food))

        # Handle score changes
        if self.game.score != self.prev_score or score_needs_redraw:
            # To correctly render the score as an overlay, we must restore the background
            # behind the old score before drawing the new one.

            # Create the new score surface to get its dimensions
            new_score_surface = self.font.render("Score: " + str(self.game.score), True, GameColors.WHITE)
            new_score_rect = new_score_surface.get_rect(topleft=(0, 0))

            # The area to update is the union of the old and new score rectangles.
            update_area = new_score_rect
            if self.prev_score_rect:
                update_area = self.prev_score_rect.union(new_score_rect)

            # Restore the background by filling the update area, then redrawing any snake parts on top.
            self.display.fill(GameColors.BLACK, update_area)
            for pt in self.game.snake:
                snake_part_rect = pygame.Rect(pt.x * BLOCK_SIZE, pt.y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)
                if update_area.colliderect(snake_part_rect):
                    self._draw_snake_block(pt)

            # Also redraw the food if it was in the update area
            if self.game.food:
                food_rect = pygame.Rect(self.game.food.x * BLOCK_SIZE, self.game.food.y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)
                if update_area.colliderect(food_rect):
                    self._draw_food_block(self.game.food)

            # Now, blit the new score text onto the correctly restored background.
            self.display.blit(new_score_surface, new_score_rect)
            self.prev_score_rect = new_score_rect
            updated_rects.append(update_area)

        # Update only the changed parts of the screen
        if updated_rects:
            pygame.display.update(updated_rects)

        # Update state for the next frame
        self.prev_snake = self.game.snake[:]
        self.prev_food = self.game.food
        self.prev_score = self.game.score

        self.clock.tick(render_fps)
