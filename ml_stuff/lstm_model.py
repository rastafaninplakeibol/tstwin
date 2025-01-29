import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class LSTMDynamicsModel(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64, device='cpu'):
        super(LSTMDynamicsModel, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.device = torch.device(device)

        self.lstm = nn.LSTM(state_size + action_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, state_size)

        self.to(self.device)
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.criterion = nn.L1Loss()
        #self.criterion = nn.MSELoss()

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        return x

    def train_batch(self, states, actions, next_states):
        a_one_hot = np.zeros((actions.shape[0], self.action_size))
        for i, act in enumerate(actions):
            a_one_hot[i, act] = 1.0

        input_data = np.concatenate((states, a_one_hot), axis=1)
        input_tensor = torch.FloatTensor(input_data).unsqueeze(1).to(self.device)
        target_tensor = torch.FloatTensor(next_states).to(self.device)

        self.optimizer.zero_grad()
        output = self.forward(input_tensor)
        loss = self.criterion(output, target_tensor)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    @torch.no_grad()
    def predict_next_state(self, state, action):
        a_one_hot = np.zeros((1, self.action_size))
        a_one_hot[0, action] = 1.0
        input_data = np.concatenate((state[0].reshape(1, -1), a_one_hot), axis=1)
        input_tensor = torch.FloatTensor(input_data).unsqueeze(1).to(self.device)
        pred = self.forward(input_tensor)
        return pred
