import atexit
import logging
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure

from snake_ai_pytorch.models.snake_env import SnakeEnv
from snake_ai_pytorch.views import Plotting

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class CustomCallback(BaseCallback):
    """A custom callback that integrates with Plotting, saves the best model, and logs progress."""

    def __init__(self, agent: "SB3Agent", verbose=0):
        super().__init__(verbose)
        self.agent = agent

    def _on_step(self) -> bool:
        # Check if any environments are done (since we have one, we check for any)
        if np.any(self.locals["dones"]):
            score = self.locals["infos"][0]["score"]
            is_new_record = self.agent._on_episode_end(score)

            if is_new_record:
                logging.info(f"New record: {self.agent.record}! Saving model to {self.agent.model_path}")
                self.model.save(self.agent.model_path)
                # Also save the stats, so the record and plot data are persisted
                self.agent._save_stats()
        return True


class SB3Agent:
    MODEL_FILENAME = "sb3_dqn_snake.zip"
    STATS_FILENAME = "sb3_dqn_snake_stats.pth"

    def __init__(self, render_mode="human"):
        self.render_mode = render_mode
        self.model_path = Path("model") / self.MODEL_FILENAME
        self.stats_path = Path("model") / self.STATS_FILENAME
        self.model_path.parent.mkdir(parents=True, exist_ok=True)

        # The environment will not render if render_mode is not 'human'
        self.env = SnakeEnv(render_mode=self.render_mode)

        # Plotting is now initialized lazily in train() or play()
        self.plotting = None
        self._loaded_plot_scores = []
        self._loaded_plot_mean_scores = []
        self.model = None  # Model is now lazy-loaded

        # Load existing stats if available
        self.n_games = 0
        self.record = 0
        self.total_score = 0
        self._load_stats()

    def _setup_model(self, for_training: bool):
        """Initializes the model, loading from a file or creating a new one."""
        if self.model is not None:
            return

        # Setup SB3 logger to output to console, csv, and tensorboard
        log_path = Path("logs/sb3_logs/")
        sb3_logger = configure(str(log_path), ["stdout", "csv", "tensorboard"])

        model_to_load = None
        if for_training:
            # For training, prioritize the final model to resume, then the best model
            final_model_path = self.model_path.with_suffix(".final.zip")
            if final_model_path.exists():
                logging.info(f"Found a final model state. Resuming training from {final_model_path}")
                model_to_load = final_model_path
            elif self.model_path.exists():
                logging.info(f"Found a best model. Continuing training from {self.model_path}...")
                model_to_load = self.model_path
        else:  # For playing, only use the 'best' model
            if self.model_path.exists():
                logging.info(f"Loading best model for playing from {self.model_path}...")
                model_to_load = self.model_path

        # Define the model
        if model_to_load:
            # When loading, SB3 automatically handles the device. Re-assigning the env and logger is good practice.
            self.model = DQN.load(model_to_load, env=self.env)
        elif for_training:
            logging.info("Creating a new model...")
            policy_kwargs = {"net_arch": [256]}  # Corresponds to hidden_size=256
            self.model = DQN(
                "MlpPolicy",
                self.env,
                learning_rate=0.001,
                buffer_size=100_000,
                learning_starts=1000,
                batch_size=1000,
                gamma=0.9,
                train_freq=(1, "step"),
                gradient_steps=1,
                target_update_interval=1000,
                exploration_fraction=0.2,
                exploration_final_eps=0.02,
                policy_kwargs=policy_kwargs,
                verbose=1,
            )
        else:  # Not training and no model exists
            logging.error(f"No model found at {self.model_path}. Please train the agent first.")
            return

        self.model.set_logger(sb3_logger)

    def _load_stats(self):
        if self.stats_path.exists():
            logging.info(f"Loading stats from {self.stats_path}...")
            stats = torch.load(self.stats_path, weights_only=False)  # nosec B614
            self.n_games = stats.get("n_games", 0)
            self.record = stats.get("record", 0)
            self.total_score = stats.get("total_score", 0)
            # Store plot data to be loaded when plotting is initialized
            self._loaded_plot_scores = stats.get("plot_scores", [])
            self._loaded_plot_mean_scores = stats.get("plot_mean_scores", [])
            logging.info(f"Resuming from game {self.n_games} with record {self.record}")

    def _start_plotting(self, train: bool = False):
        """Initializes the plotting process if it hasn't been already."""
        if self.render_mode != "human" or self.plotting is not None:
            return

        self.plotting = Plotting(train=train)
        if self._loaded_plot_scores:
            self.plotting.load_data(self._loaded_plot_scores, self._loaded_plot_mean_scores)

        self.plotting.start()
        atexit.register(self.plotting.stop)

    def _on_episode_end(self, score: int) -> bool:
        """Handles the logic at the end of an episode: updates stats, plots, and returns if a new record was set.

        Args:
            score: The score of the completed episode.

        Returns:
            A boolean indicating if a new record was achieved.

        """
        self.n_games += 1
        self.total_score += score

        is_new_record = False
        if score > self.record:
            self.record = score
            is_new_record = True

        logging.info(f"Game: {self.n_games}, Score: {score}, Record: {self.record}")

        if self.plotting:
            mean_score = self.total_score / self.n_games
            self.plotting.plot(score, mean_score)

        return is_new_record

    def _save_stats(self):
        """Saves plotting data and other statistics."""
        if not self.plotting:
            return
        stats = {
            "n_games": self.n_games,
            "record": self.record,
            "total_score": self.total_score,
            "plot_scores": self.plotting.scores,
            "plot_mean_scores": self.plotting.mean_scores,
        }
        torch.save(stats, self.stats_path)
        logging.info(f"Stats saved to {self.stats_path}")

    def train(self, total_timesteps=200_000):
        """Trains the agent using the SB3 learn method."""
        self._setup_model(for_training=True)
        self._start_plotting(train=True)
        custom_callback = CustomCallback(agent=self)

        try:
            self.model.learn(
                total_timesteps=total_timesteps,
                callback=custom_callback,
                reset_num_timesteps=False,  # Continue training from where we left off
            )
            logging.info("Training finished.")
        except KeyboardInterrupt:
            logging.info("Training interrupted by user.")
        finally:
            # Save the final model and stats
            final_model_path = self.model_path.with_suffix(".final.zip")
            self.model.save(final_model_path)
            logging.info(f"Final model saved to {final_model_path}")
            self._save_stats()

    def play(self):
        """Lets the trained agent play the game and plots the scores."""
        self._setup_model(for_training=False)
        self._start_plotting()

        if self.model is None:
            return

        if self.plotting and self.plotting.scores:
            self.plotting.add_training_marker(len(self.plotting.scores) - 1)

        obs, _ = self.env.reset()

        while True:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = self.env.step(action)
            if terminated or truncated:
                score = info["score"]
                # The agent's internal state will be updated
                self._on_episode_end(score)
                obs, _ = self.env.reset()
