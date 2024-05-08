import torch.nn as nn
class CompositeLoss(nn.Module):
    """Loss function combining MSE for next state and cross-entropy for action."""
    def __init__(self, state_weight=1.0, action_weight=1.0):
        super(CompositeLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.state_weight = state_weight
        self.action_weight = action_weight

    def forward(self, predicted_state, true_state, predicted_action_logits, true_action):
        state_loss = self.mse_loss(predicted_state, true_state) * self.state_weight
        action_loss = self.ce_loss(predicted_action_logits, true_action) * self.action_weight
        return state_loss + action_loss

