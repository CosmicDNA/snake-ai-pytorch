import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

class QNetLightning(pl.LightningModule):
    def __init__(self, input_size, hidden_size, output_size, lr, gamma):
        super().__init__()
        self.save_hyperparameters()
        self.model = Linear_QNet(input_size, hidden_size, output_size)
        self.criterion = nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        state, action, reward, next_state, done = batch

        # 1. Get predicted Q-values for current state
        pred = self(state)

        # 2. Get Q-values for next state and calculate max
        # .detach() is used to prevent gradients from flowing into the target network
        next_q_values = self(next_state).detach()
        max_next_q = next_q_values.max(dim=1)[0]

        # 3. Calculate target Q-value (Bellman equation)
        # For terminal states (done=True), the future reward is 0
        Q_new = reward + (self.hparams.gamma * max_next_q * (~done))

        # 4. Create target tensor by cloning predictions and updating with new Q-values
        target = pred.clone()
        action_idxs = torch.argmax(action, dim=1)
        target[torch.arange(len(done)), action_idxs] = Q_new

        # 5. Calculate loss
        loss = self.criterion(target, pred)
        return loss
