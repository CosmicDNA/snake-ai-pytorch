import atexit
import logging
import random
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor

from snake_ai_pytorch.models.dueling_qnet import DuelingQNet
from snake_ai_pytorch.models.snake_env import SnakeEnv
from snake_ai_pytorch.views import Plotting

# Configure logging to show info-level messages
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class Agent:
    MAX_MEMORY = 100_000
    BATCH_SIZE = 1000
    LR = 0.001

    def __init__(self, render_mode="human"):
        self.n_games = 0
        # Epsilon is now calculated dynamically based on the number of games
        # self.epsilon = 0 # randomness
        self.gamma = 0.9  # discount rate
        self.memory = deque(maxlen=self.MAX_MEMORY)  # popleft()
        self.render_mode = render_mode

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DuelingQNet(input_size=11, hidden_size=256, output_size=3).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.LR)
        self.criterion = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))  # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        mini_sample = random.sample(self.memory, self.BATCH_SIZE) if len(self.memory) > self.BATCH_SIZE else self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample, strict=False)
        self._train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self._train_step([state], [action], [reward], [next_state], [done])

    def _train_step(self, states, actions, rewards, next_states, dones):
        # 1. Convert to tensors and move to the correct device
        states = torch.tensor(np.array(states), dtype=torch.float).to(self.device)
        # Actions are now integers, convert to long tensor for indexing
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float).to(self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float).to(self.device)
        dones = torch.tensor(np.array(dones), dtype=torch.bool).to(self.device)

        # 2. Get predicted Q-values for current state
        pred: Tensor = self.model(states)

        # 3. Get Q-values for next state and calculate max
        target = pred.clone()
        next_pred: Tensor = self.model(next_states)
        next_q_values = next_pred.detach()
        max_next_q = next_q_values.max(dim=1)[0]

        # 4. Calculate target Q-value (Bellman equation)
        # For terminal states (done=True), the future reward is 0
        Q_new = rewards + (self.gamma * max_next_q * (~dones))

        # 5. Create target tensor by cloning predictions and updating with new Q-values
        target[torch.arange(len(dones)), actions] = Q_new

        # 6. Calculate loss
        loss = self.criterion(target, pred)

        # 7. Manually perform the optimization
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        epsilon = 80 - self.n_games
        if random.randint(0, 200) < epsilon:  # nosec
            move = random.randint(0, 2)  # nosec
        else:
            # Add a batch dimension for the model, which expects a 2D tensor
            state0 = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self.device)
            prediction: Tensor = self.model(state0)
            move = torch.argmax(prediction).item()

        return move

    def load_existing_model(self):
        # Create model directory if it doesn't exist
        model_folder_path = Path("model")
        model_folder_path.mkdir(parents=True, exist_ok=True)
        model_path = model_folder_path / "model.pth"

        # Load existing model if it exists
        if model_path.exists():
            logging.info("Loading existing state from model.pth...")
            # The saved checkpoint is a dictionary, not just weights.
            # `weights_only=False` (the default) is needed.
            return torch.load(model_path, weights_only=True), model_path
        return None, model_path

    def train(self):
        self.play(train=True)

    def play(self, train=False):
        total_score = 0
        record = 0
        # Only create a plot if we are in a visual render mode
        plotting = None
        if self.render_mode == "human":
            plotting = Plotting(train)
            plotting.start()
            # Register a cleanup function to stop the plotting process on exit
            atexit.register(plotting.stop)
        # Use the new Gymnasium environment
        env = SnakeEnv(render_mode=self.render_mode)

        checkpoint, model_path = self.load_existing_model()

        # Load existing model if it exists
        if checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.n_games = checkpoint["n_games"]
            record = checkpoint["record"]
            total_score = checkpoint["total_score"]
            # To ensure the agent's exploration continues from where it left off
            logging.info(f"Resuming from game {self.n_games} with record {record}")
            if plotting:
                plot_scores = checkpoint.get("plot_scores", [])
                plot_mean_scores = checkpoint.get("plot_mean_scores", [])
                plotting.load_data(plot_scores, plot_mean_scores)
                # If we are in 'play' mode (not training) and there's historical data, add the marker.
                if not train and len(plot_scores) > 0:
                    plotting.add_training_marker(len(plot_scores) - 1)

        # Initial state from the environment
        state_old, info = env.reset()

        while True:
            # get move (action is now an integer: 0, 1, or 2)
            action = self.get_action(state_old)

            # perform move and get new state from environment
            state_new, reward, terminated, truncated, info = env.step(action)
            score = info["score"]
            env.render()

            if train:
                # train short memory
                self.train_short_memory(state_old, action, reward, state_new, terminated)

            # remember
            self.remember(state_old, action, reward, state_new, terminated)

            # The current step is done, update the state for the next iteration
            state_old = state_new

            if terminated:
                # train long memory, plot result
                self.n_games += 1
                if train:
                    self.train_long_memory()

                if score > record:
                    record = score
                    if train:
                        # Save model checkpoint
                        checkpoint = {
                            "n_games": self.n_games,
                            "record": record,
                            "total_score": total_score,
                            "plot_scores": plotting.scores if plotting else [],
                            "plot_mean_scores": plotting.mean_scores if plotting else [],
                            "model_state_dict": self.model.state_dict(),
                            "optimizer_state_dict": self.optimizer.state_dict(),
                        }
                        torch.save(checkpoint, model_path)

                logging.info(f"Game: {self.n_games}, Score: {score}, Record: {record}")

                total_score += score
                mean_score = total_score / self.n_games
                if plotting:
                    plotting.plot(score, mean_score)

                # Reset the environment and get the new initial state
                state_old, info = env.reset()
