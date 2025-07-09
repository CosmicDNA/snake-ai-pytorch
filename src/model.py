import torch.nn as nn
import torch.nn.functional as F
import torch

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

class Dueling_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        # Shared feature learning layer
        self.linear1 = nn.Linear(input_size, hidden_size)

        # Value stream
        self.value_stream = nn.Linear(hidden_size, 1)

        # Advantage stream
        self.advantage_stream = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Pass input through the shared layer
        x = F.relu(self.linear1(x))

        # Calculate value and advantage
        values = self.value_stream(x)
        advantages = self.advantage_stream(x)

        # Combine value and advantage streams to get Q-values
        # Q(s, a) = V(s) + (A(s, a) - mean(A(s, a)))
        q_values = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_values
