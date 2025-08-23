import numpy as np

from snake_ai_pytorch.snake_game_human import Direction, SnakeGame


class SnakeGameAI(SnakeGame):
    """A subclass of SnakeGame that is designed for AI agents to interact with.

    It provides methods to get the current state of the game and to perform actions.
    """

    def __init__(self, w=640, h=480, render_mode="human"):
        super().__init__(w, h, render_mode)

    def reset(self):
        """Reset the game state to the initial conditions."""
        super().reset()
        self.frame_iteration = 0

    def play_step(self, action):
        self.frame_iteration += 1

        # 1. determine new direction from action
        self._determine_direction(action)

        # 2. move
        self._move(self.direction)  # update the head
        self.snake.insert(0, self.head)

        # 3. check if game over
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # 4. place new food or just move
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()

        return reward, game_over, self.score

    def _determine_direction(self, action):
        """Determines the new direction based on the AI's action.

        Action is a 3-element list: [straight, right_turn, left_turn]
        """
        # Define clockwise direction sequence
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [0, 1, 0]):  # right turn
            next_idx = (idx + 1) % 4
            self.direction = clock_wise[next_idx]
        elif np.array_equal(action, [0, 0, 1]):  # left turn
            next_idx = (idx - 1) % 4
            self.direction = clock_wise[next_idx]
