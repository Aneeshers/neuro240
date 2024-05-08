from storage import *
from agents.v_agent import ValueIterationAgent
from agents.r_agent import RandomAgent
from agents.d_agent import DeterministicAgent
storage = AgentTrajectoryStorage()
collect_trajectories(ValueIterationAgent, 'value_iteration', storage, num_episodes=5, grid_size=10)
collect_trajectories(RandomAgent, 'random', storage, num_episodes=5, grid_size=10)
collect_trajectories(DeterministicAgent, 'deterministic', storage, num_episodes=5, grid_size=10)
storage.display_trajectories('value_iteration')
storage.display_trajectories('random')
storage.display_trajectories('deterministic')