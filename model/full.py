from ln import *
from pp import *
class CombinedTrajectoryNetwork(nn.Module):
    def __init__(self, num_lstm_layers=1, hidden_size=128, latent_size=64, resnet_layers=[2, 2, 2, 2], num_actions=4):
        super(CombinedTrajectoryNetwork, self).__init__()
        self.latent_space_network = LatentSpaceNetwork(num_lstm_layers=num_lstm_layers, hidden_size=hidden_size,
                                                       latent_size=latent_size, resnet_layers=resnet_layers)

        self.next_state_action_predictor = NextStateActionPredictor(latent_size, input_size=128, num_actions=num_actions)

    def forward(self, trajectory, current_state):
        latent = self.latent_space_network(trajectory)
        next_state, next_action_logits = self.next_state_action_predictor(latent, current_state)

        return next_state, next_action_logits

