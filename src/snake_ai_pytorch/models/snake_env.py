import gymnasium as gym
import numpy as np
from gymnasium import spaces

from snake_ai_pytorch.models.direction import Direction
from snake_ai_pytorch.models.point import Point
from snake_ai_pytorch.models.snake_game_ai import SnakeGameAI
from snake_ai_pytorch.views.visual_configuration import BLOCK_SIZE


class SnakeEnv(gym.Env):
    """A custom Gymnasium environment for the Snake game."""

    metadata = {"render_modes": ["human"], "render_fps": 400}

    def __init__(self, w=640, h=480, render_mode="human"):
        super().__init__()
        self.game = SnakeGameAI(w, h, render_mode=render_mode)
        self.render_mode = render_mode

        # Action space: 0: straight, 1: right turn, 2: left turn
        self.action_space = spaces.Discrete(3)

        # Observation space: 11 boolean values
        self.observation_space = spaces.Box(low=0, high=1, shape=(11,), dtype=np.int32)

    def _get_obs(self):
        """Generates the observation from the current game state.

        This logic was previously in Agent.get_state().
        """
        head = self.game.snake[0]
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)

        dir_l = self.game.direction == Direction.LEFT
        dir_r = self.game.direction == Direction.RIGHT
        dir_u = self.game.direction == Direction.UP
        dir_d = self.game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and self.game.is_collision(point_r))
            or (dir_l and self.game.is_collision(point_l))
            or (dir_u and self.game.is_collision(point_u))
            or (dir_d and self.game.is_collision(point_d)),
            # Danger right
            (dir_u and self.game.is_collision(point_r))
            or (dir_d and self.game.is_collision(point_l))
            or (dir_l and self.game.is_collision(point_u))
            or (dir_r and self.game.is_collision(point_d)),
            # Danger left
            (dir_d and self.game.is_collision(point_r))
            or (dir_u and self.game.is_collision(point_l))
            or (dir_r and self.game.is_collision(point_u))
            or (dir_l and self.game.is_collision(point_d)),
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            # Food location
            self.game.food.x < self.game.head.x,  # food left
            self.game.food.x > self.game.head.x,  # food right
            self.game.food.y < self.game.head.y,  # food up
            self.game.food.y > self.game.head.y,  # food down
        ]
        return np.array(state, dtype=np.int32)

    def _get_info(self):
        return {"score": self.game.score}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game.reset()
        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action):
        action_array = [0, 0, 0]
        action_array[action] = 1

        reward, terminated, score = self.game.play_step(action_array)
        observation = self._get_obs()
        info = self._get_info()
        return observation, reward, terminated, False, info  # truncated is always False

    def render(self):
        self.game.render(self.metadata["render_fps"])
