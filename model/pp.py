import torch
import torch.nn as nn


class NextStateActionPredictor(nn.Module):
    def __init__(self, latent_size, input_size, num_actions):
        super(NextStateActionPredictor, self).__init__()
        self.fc1 = nn.Linear(latent_size + input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc_state = nn.Linear(64, input_size)
        self.fc_action = nn.Linear(64, num_actions)

    def forward(self, latent, current_state):
        combined = torch.cat((latent, current_state), dim=1)
        x = torch.relu(self.fc1(combined))
        x = torch.relu(self.fc2(x))
        next_state = self.fc_state(x)
        next_action_logits = self.fc_action(x)
        return next_state, next_action_logits