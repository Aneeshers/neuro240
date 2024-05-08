import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import jax.numpy as jnp
import random
from env.grid import GridEnvironment
class AgentTrajectoryStorage:
    def __init__(self):
        self.trajectories = {
            'value_iteration': [],
            'random': [],
            'deterministic': []
        }

    def log_trajectory(self, agent_type, trajectory):
        self.trajectories[agent_type].append(trajectory)

    def get_trajectories(self, agent_type):
        return self.trajectories[agent_type]

    def clear_trajectories(self):
        for agent_type in self.trajectories:
            self.trajectories[agent_type] = []

def collect_trajectories(agent_class, agent_type, storage, num_episodes=5, grid_size=10):
    for _ in range(num_episodes):
        goal_color = random.choice([GridEnvironment.RED, GridEnvironment.BLUE, GridEnvironment.GREEN, GridEnvironment.YELLOW])
        new_env = GridEnvironment(grid_size=grid_size, seed=random.randint(0, 100000), goal_color=goal_color)
        agent = agent_class(new_env)
        if agent_type == 'deterministic':
            agent.set_goal_direction(goal_color)
        if agent_type == 'value_iteration':
            agent.set_goal_color(goal_color)
            agent.value_iteration(1000)
        trajectory = []
        count = 0
        while True:
            count +=1 
            action, reward = agent.act()
            position = list(new_env.agent_position)
            trajectory.append({
                'position': position,
                'action': action,
                'reward': reward,
                'agent_type': agent_type,
                'goal_color': goal_color
            })
            if reward == new_env.GOAL_REWARD or reward == new_env.WALL_PENALTY or count > 41:
                break
        storage.log_trajectory(agent_type, trajectory)


class TrajectoryDataset(Dataset):
    def __init__(self, storage, agent_types):
        self.data = []
        for agent_type in agent_types:
            trajectories = storage.get_trajectories(agent_type)
            for trajectory in trajectories:
                for step in trajectory:
                    self.data.append(step)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        step = self.data[idx]
        position = torch.tensor([int(p) for p in step['position']], dtype=torch.float32)
        action = torch.tensor(step['action'], dtype=torch.int64)
        reward = torch.tensor(step['reward'], dtype=torch.float32)
        agent_type = torch.tensor(ord(step['agent_type'][0]), dtype=torch.int64)
        goal_color = torch.tensor(step['goal_color'], dtype=torch.int64)
        return position, action, reward, agent_type, goal_color
