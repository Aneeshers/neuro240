import random
class DeterministicAgent:
    def __init__(self, environment, action_sequence=None):
        self.environment = environment
        # sample action sequences
        sequences = []
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    for l in range(4):
                        sequences.append([i,j,k,l])
        if action_sequence is None:
            # choose one from the sample sequeces at random
            self.action_sequence = random.choice(sequences)
        else:
            self.action_sequence = action_sequence
        self.action_map = {0: 'up', 1: 'down', 2: 'left', 3: 'right'}
        self.action_index = 0  
        self.environment.set_agent_position([0, 0]) #lft corner
    def act(self):
        # deterministic loop
        if self.action_index >= len(self.action_sequence):
            self.action_index = 0

        action_idx = self.action_sequence[self.action_index]
        self.environment.move_agent(action_idx)
        reward = self.environment.calculate_reward()
        self.action_index += 1
        return action_idx, reward