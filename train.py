import json
import torch.optim as optim
from dataset.storage import AgentTrajectoryStorage, collect_trajectories, TrajectoryDataset
from agents.d_agent import DeterministicAgent
from agents.r_agent import RandomAgent
from agents.v_agent import ValueIterationAgent
from utils.visualize import visualize_latent_log
from utils.log import log_latent_space
from model.full import CombinedTrajectoryNetwork
from utils.loss import CompositeLoss
from torch.utils.data import DataLoader
def train_with_logging(model, latent_model, dataloader, criterion, optimizer, num_epochs=100, device='cuda', log_path="latent_space_log.json"):
    model = model.to(device)
    latent_model = latent_model.to(device)
    criterion = criterion.to(device)
    model.train()
    latent_model.eval()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in dataloader:
            positions, actions, rewards, agent_types, goal_colors = batch
            positions = positions.to(device)
            actions = actions.to(device)
            rewards = rewards.to(device)
            agent_types = agent_types.to(device)
            goal_colors = goal_colors.to(device)

            optimizer.zero_grad()
            latent = latent_model(positions)
            predicted_state, predicted_action_logits = model(positions, positions)
            loss = criterion(predicted_state, rewards, predicted_action_logits, actions)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            log_latent_space(latent, agent_types, goal_colors, file_path=log_path)

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

storage = AgentTrajectoryStorage()
collect_trajectories(RandomAgent, 'random', storage, num_episodes=20000, grid_size=10)
collect_trajectories(DeterministicAgent, 'deterministic', storage, num_episodes=20000, grid_size=10)
collect_trajectories(ValueIterationAgent, 'value_iteration', storage, num_episodes=20000, grid_size=10)
# change exp
agent_types = ['random', 'deterministic', 'value_iteration']
dataset = TrajectoryDataset(storage, agent_types)

batch_size = 4
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = CombinedTrajectoryNetwork(num_lstm_layers=1, hidden_size=128, latent_size=64,
                                    resnet_layers=[2, 2, 2, 2], num_actions=4)

criterion = CompositeLoss(state_weight=1.0, action_weight=1.0)

optimizer = optim.Adam(model.parameters(), lr=0.01)
train_with_logging(model, dataloader, criterion, optimizer, num_epochs=100)
visualize_latent_log("latent_space_log.json", "latent_space_visualization.png")