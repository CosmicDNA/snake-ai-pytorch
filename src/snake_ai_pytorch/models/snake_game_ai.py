import numpy as np

from snake_ai_pytorch.models.snake_game import SnakeGame


class SnakeGameAI(SnakeGame):
    """A subclass of SnakeGame that is designed for AI agents to interact with.

    It provides methods to get the current state of the game and to perform actions.
    """

    REWARD_FOOD = 10
    REWARD_GAMEOVER = -10
    REWARD_MOVE = -0.01
    REWARD_CHANGE_DIRECTION = -0.02
    REWARD_DIRECTION_KEPT = 0

    def __init__(self, w=640, h=480, render_mode="human"):
        super().__init__(w, h, render_mode)

    def reset(self):
        """Reset the game state to the initial conditions."""
        super().reset()
        self.frame_iteration = 0

    def play_step(self, action):
        self.frame_iteration += 1

        # 1. determine new direction from action
        reward = self._determine_direction(action)

        # 2. move
        self._move(self.direction)  # update the head
        self.snake.insert(0, self.head)

        # 3. check if game over
        game_over = False
        collision = self.get_collision()
        if collision or self.frame_iteration > 100 * len(self.snake):
            if collision:
                self._collide(collision)
            game_over = True
            reward += self.REWARD_GAMEOVER
            return reward, game_over, self.score

        # 4. place new food or just move
        if self.head == self.food:
            reward += self.REWARD_FOOD
            self._eat_food()
        else:
            reward += self.REWARD_MOVE
            self.snake.pop()

        return reward, game_over, self.score

    def _determine_direction(self, action):
        """Determines the new direction based on the AI's action.

        Action is a 3-element list: [straight, right_turn, left_turn]
        """
        if np.array_equal(action, [0, 1, 0]):  # right turn
            self.direction = self.direction.cw
            return self.REWARD_CHANGE_DIRECTION
        elif np.array_equal(action, [0, 0, 1]):  # left turn
            self.direction = self.direction.ccw
            return self.REWARD_CHANGE_DIRECTION
        return self.REWARD_DIRECTION_KEPT
