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

    def __init__(self, plotting: Plotting | None, stats_path: Path, model_save_path: Path, verbose=0):
        super().__init__(verbose)
        self.plotting = plotting
        self.stats_path = stats_path
        self.model_save_path = model_save_path
        # These will be initialized from the agent before training starts
        self.total_score = 0
        self.n_games = 0
        self.record = 0

    def _on_step(self) -> bool:
        # Check if any environments are done (since we have one, we check for any)
        if np.any(self.locals["dones"]):
            self.n_games += 1
            score = self.locals["infos"][0]["score"]
            self.total_score += score

            if score > self.record:
                self.record = score
                logging.info(f"New record: {self.record}! Saving model to {self.model_save_path}")
                self.model.save(self.model_save_path)
                self._save_stats()

            if self.plotting:
                mean_score = self.total_score / self.n_games
                self.plotting.plot(score, mean_score)

            logging.info(f"Game: {self.n_games}, Score: {score}, Record: {self.record}")
        return True

    def _save_stats(self):
        """Saves plotting data and other statistics."""
        if not self.plotting:
            return
        stats = {
            "n_games": len(self.plotting.scores),
            "record": self.record,
            "total_score": sum(self.plotting.scores),
            "plot_scores": self.plotting.scores,
            "plot_mean_scores": self.plotting.mean_scores,
        }
        torch.save(stats, self.stats_path)
        logging.info(f"Stats saved to {self.stats_path}")


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

        # Setup plotting only if rendering is enabled
        self.plotting = None
        if self.render_mode == "human":
            self.plotting = Plotting(train=True)
            self.plotting.start()
            atexit.register(self.plotting.stop)

        # Load existing stats if available
        self.n_games = 0
        self.record = 0
        self.total_score = 0
        self._load_stats()

        # Setup SB3 logger to output to console, csv, and tensorboard
        log_path = Path("logs/sb3_logs/")
        sb3_logger = configure(str(log_path), ["stdout", "csv", "tensorboard"])

        # Define the model
        if self.model_path.exists():
            logging.info(f"Loading existing model from {self.model_path}...")
            self.model = DQN.load(self.model_path, env=self.env)
            self.model.set_logger(sb3_logger)
        else:
            logging.info("Creating a new model...")
            policy_kwargs = {
                "net_arch": [256],  # Corresponds to hidden_size=256
            }
            self.model = DQN(
                "MlpPolicy",
                self.env,
                learning_rate=0.001,  # Corresponds to LR
                buffer_size=100_000,  # Corresponds to MAX_MEMORY
                learning_starts=1000,  # Steps to collect before training starts
                batch_size=1000,  # Corresponds to BATCH_SIZE
                gamma=0.9,  # Discount factor
                train_freq=(1, "step"),  # Train the agent every step
                gradient_steps=1,
                target_update_interval=1000,  # Update the target network every 1000 steps
                exploration_fraction=0.2,  # Epsilon decay over 20% of total training
                exploration_final_eps=0.02,  # Final value of epsilon
                policy_kwargs=policy_kwargs,
                verbose=1,
            )
            self.model.set_logger(sb3_logger)

    def _load_stats(self):
        if self.stats_path.exists():
            logging.info(f"Loading stats from {self.stats_path}...")
            stats = torch.load(self.stats_path, weights_only=True)
            self.n_games = stats.get("n_games", 0)
            self.record = stats.get("record", 0)
            self.total_score = stats.get("total_score", 0)
            if self.plotting:
                plot_scores = stats.get("plot_scores", [])
                plot_mean_scores = stats.get("plot_mean_scores", [])
                self.plotting.load_data(plot_scores, plot_mean_scores)
            logging.info(f"Resuming from game {self.n_games} with record {self.record}")

    def _save_stats_on_exit(self, callback: CustomCallback):
        """Saves the final stats upon exiting training."""
        if not self.plotting:
            return
        stats = {
            "n_games": len(callback.plotting.scores),
            "record": callback.record,
            "total_score": sum(callback.plotting.scores),
            "plot_scores": callback.plotting.scores,
            "plot_mean_scores": callback.plotting.mean_scores,
        }
        torch.save(stats, self.stats_path)
        logging.info(f"Final stats saved to {self.stats_path}")

    def train(self, total_timesteps=200_000):
        """Trains the agent using the SB3 learn method."""
        custom_callback = CustomCallback(
            plotting=self.plotting,
            stats_path=self.stats_path,
            model_save_path=self.model_path,
        )
        # Pass current stats to the callback
        custom_callback.n_games = self.n_games
        custom_callback.total_score = self.total_score
        custom_callback.record = self.record

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
            self._save_stats_on_exit(custom_callback)

    def play(self):
        """Lets the trained agent play the game and plots the scores."""
        if not self.model_path.exists():
            logging.error(f"No model found at {self.model_path}. Please train the agent first.")
            return

        model = DQN.load(self.model_path, env=self.env)

        if self.plotting and self.plotting.scores:
            self.plotting.add_training_marker(len(self.plotting.scores) - 1)

        obs, _ = self.env.reset()
        play_n_games = self.n_games
        play_total_score = self.total_score

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = self.env.step(action)
            if terminated or truncated:
                score = info["score"]
                play_n_games += 1
                play_total_score += score
                mean_score = play_total_score / play_n_games

                logging.info(f"Game: {play_n_games}, Score: {score}")

                if self.plotting:
                    self.plotting.plot(score, mean_score)

                obs, _ = self.env.reset()
