import random
class RandomAgent:
    def __init__(self, environment):
        self.environment = environment
        self.actions = [0, 1, 2, 3]
        self.environment.set_agent_position([0, 0]) #lft corner again here

    def act(self):
        """Randomly choose an action and execute it."""
        action_idx = random.choice(self.actions)
        self.environment.move_agent(action_idx)
        reward = self.environment.calculate_reward()
        return action_idx, reward