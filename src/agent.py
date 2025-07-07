import random
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from helper import plot
from model import Linear_QNet
from snake_env import SnakeEnv

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:

    def __init__(self):
        self.n_games = 0
        # Epsilon is now calculated dynamically based on the number of games
        # self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Linear_QNet(input_size=11, hidden_size=256, output_size=3).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)
        self.criterion = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
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
        pred = self.model(states)

        # 3. Get Q-values for next state and calculate max
        target = pred.clone()
        next_q_values = self.model(next_states).detach()
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
        epsilon = 80 - self.n_games
        if random.randint(0, 200) < epsilon:
            move = random.randint(0, 2)
        else:
            state0 = torch.tensor(state, dtype=torch.float).to(self.device)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()

        return move


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    # Use the new Gymnasium environment
    env = SnakeEnv()

    # Create model directory if it doesn't exist
    model_folder_path = Path('model')
    model_folder_path.mkdir(parents=True, exist_ok=True)
    model_path = model_folder_path / 'model.pth'

    # Load existing model if it exists
    if model_path.exists():
        print("Loading existing state from model.pth...")
        checkpoint = torch.load(model_path)
        agent.model.load_state_dict(checkpoint['model_state_dict'])
        agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        agent.n_games = checkpoint['n_games']
        record = checkpoint['record']
        total_score = checkpoint['total_score']
        plot_scores = checkpoint['plot_scores']
        plot_mean_scores = checkpoint['plot_mean_scores']
        # To ensure the agent's exploration continues from where it left off
        print(f"Resuming from game {agent.n_games} with record {record}")

    # Initial state from the environment
    state_old, info = env.reset()

    while True:
        # get move (action is now an integer: 0, 1, or 2)
        action = agent.get_action(state_old)

        # perform move and get new state from environment
        state_new, reward, terminated, truncated, info = env.step(action)
        score = info['score']
        env.render()

        # train short memory
        agent.train_short_memory(state_old, action, reward, state_new, terminated)

        # remember
        agent.remember(state_old, action, reward, state_new, terminated)

        # The current step is done, update the state for the next iteration
        state_old = state_new

        if terminated:
            # train long memory, plot result
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                # Save model checkpoint
                checkpoint = {
                    'n_games': agent.n_games, 'record': record, 'total_score': total_score,
                    'plot_scores': plot_scores, 'plot_mean_scores': plot_mean_scores,
                    'model_state_dict': agent.model.state_dict(),
                    'optimizer_state_dict': agent.optimizer.state_dict(),
                }
                torch.save(checkpoint, model_path)

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

            # Reset the environment and get the new initial state
            state_old, info = env.reset()


if __name__ == '__main__':
    train()