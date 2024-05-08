# tried to optimize for jax, but not enough time, but the jax arrays are there to use iin
# future dev
import random
import jax.numpy as jnp
class ValueIterationAgent:
    def __init__(self, environment):
        self.environment = environment
        self.goal_colors = [environment.RED, environment.BLUE, environment.GREEN, environment.YELLOW]
        self.goal_color = random.choice(self.goal_colors)
        self.environment.set_goal_color(self.goal_color)
        self.environment.set_agent_position([0, 0]) # lft corner

        self.grid_size = environment.grid_size
        self.value_table = jnp.zeros((self.grid_size, self.grid_size))
        self.policy = jnp.zeros((self.grid_size, self.grid_size), dtype=jnp.int32) 
        self.gamma = 0.9  
        self.epsilon = 0.1
        self.actions = ['up', 'down', 'left', 'right']
        self.action_map = {0: 'up', 1: 'down', 2: 'left', 3: 'right'}
        self.rewards = jnp.zeros_like(self.value_table)

        self.initialize_rewards()

    def initialize_rewards(self):
        all_rewards = jnp.zeros((self.grid_size, self.grid_size))
        for horizontal in range(self.grid_size):
            for vertical in range(self.grid_size):
                candidate_spot = [horizontal, vertical]
                starting_spot = self.environment.agent_position
                self.environment.set_agent_position(candidate_spot)
                points_scored = self.environment.calculate_reward()
                all_rewards = all_rewards.at[horizontal, vertical].set(points_scored)
                self.environment.set_agent_position(starting_spot)
        self.rewards = all_rewards

    def value_iteration(self, iterations=100):
        for _ in range(iterations):
            updated_values = jnp.copy(self.value_table)
            revised_strategy = jnp.copy(self.policy)

            for row in range(self.grid_size):
                for column in range(self.grid_size):
                    evaluations = []
                    for move in range(4):
                        future_pos = self.get_next_position([row, column], move)
                        row_new, col_new = future_pos
                        achieved_score = self.rewards[row_new, col_new]
                        future_value = achieved_score + self.gamma * self.value_table[row_new, col_new]
                        evaluations.append(future_value)

                    optimal_value = max(evaluations)
                    updated_values = updated_values.at[row, column].set(optimal_value)
                    revised_strategy = revised_strategy.at[row, column].set(jnp.argmax(jnp.array(evaluations)))
            self.value_table = updated_values
            self.policy = revised_strategy


    def get_next_position(self, current_position, direction):
        x, y = current_position
        if direction == 0 and x > 0:
            x -= 1
        
        elif direction == 1 and x < self.grid_size - 1:
            x += 1
        elif direction == 2 and y > 0:
            y -= 1
        elif direction == 3 and y < self.grid_size - 1:
            y += 1
        return [x, y]

    def act(self):
        x, y = self.environment.agent_position
        action_idx = int(self.policy[x, y])
        self.environment.move_agent(action_idx)
        reward = self.environment.calculate_reward()
        return action_idx, reward