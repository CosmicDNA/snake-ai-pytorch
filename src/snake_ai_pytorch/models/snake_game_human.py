import logging

import pygame

from snake_ai_pytorch.models.direction import Direction
from snake_ai_pytorch.models.snake_game import SnakeGame


class SnakeGameHuman:
    """An environment to run the human-playable version of the Snake game.

    This class is responsible for the game loop, handling user input,
    and rendering the game state. It separates the game logic from the
    application's main loop.
    """

    def __init__(self, w=640, h=480):
        self.game = SnakeGame(w, h, render_mode="human")
        self.speed = 15  # Set a comfortable speed for human play

    def run(self):
        """Starts and manages the main game loop."""
        running = True
        while running:
            # 1. Handle user input
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    # Prevent the snake from reversing on itself
                    if event.key == pygame.K_LEFT and self.game.direction != Direction.RIGHT:
                        self.game.direction = Direction.LEFT
                    elif event.key == pygame.K_RIGHT and self.game.direction != Direction.LEFT:
                        self.game.direction = Direction.RIGHT
                    elif event.key == pygame.K_UP and self.game.direction != Direction.DOWN:
                        self.game.direction = Direction.UP
                    elif event.key == pygame.K_DOWN and self.game.direction != Direction.UP:
                        self.game.direction = Direction.DOWN

            # 2. Advance the game state
            game_over, score = self.game.play_step(self.game.direction)

            # 3. Render the game
            self.game.render(self.speed)

            # 4. Check for game over
            if game_over:
                logging.info("Final Score", score)
                # A short delay to see the final score before the window closes
                pygame.time.wait(1500)
                running = False

        pygame.quit()


if __name__ == "__main__":
    env = SnakeGameHuman()
    env.run()
