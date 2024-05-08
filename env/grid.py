import numpy as np
import random
class GridEnvironment:
    WHITE = 0
    BLACK = 1
    RED = 2
    BLUE = 3
    GREEN = 4
    YELLOW = 5

    MOVE_PENALTY = -1
    WALL_PENALTY = -5
    GOAL_REWARD = 10

    def __init__(self, grid_size=10, seed=None):
        self.grid_size = grid_size
        self.grid = None
        self.agent_position = [0, 0]
        self.goal_color = self.RED
        if seed is not None:
            np.random.seed(seed)

        self.reset_grid()

    def reset_grid(self):
        self.grid = np.random.choice([self.WHITE, self.BLACK, self.RED, self.BLUE, self.GREEN, self.YELLOW], 
                                     (self.grid_size, self.grid_size))

    def set_agent_position(self, position):
        if 0 <= position[0] < self.grid_size and 0 <= position[1] < self.grid_size:
            self.agent_position = position
    def set_goal_color(self, color):
        if color in [self.WHITE, self.BLACK, self.RED, self.BLUE, self.GREEN, self.YELLOW]:
            self.goal_color = color

    def move_agent(self, direction):
        x, y = self.agent_position
        if direction == 'up' and x > 0:
            x -= 1
        elif direction == 'down' and x < self.grid_size - 1:
            x += 1
        elif direction == 'left' and y > 0:
            y -= 1
        elif direction == 'right' and y < self.grid_size - 1:
            y += 1
        self.agent_position = [x, y]

    def calculate_reward(self):
        x, y = self.agent_position
        cell_value = self.grid[x, y]

        if cell_value == self.BLACK:
            return self.WALL_PENALTY
        elif cell_value == self.goal_color:
            return self.GOAL_REWARD
        else:
            return self.MOVE_PENALTY

    def display_grid(self):
        # a little buggy
        symbols = {
            self.WHITE: 'W', 
            self.BLACK: 'B', 
            self.RED: 'R', 
            self.BLUE: 'B', 
            self.GREEN: 'G', 
            self.YELLOW: 'Y'
        }
        grid_display = [[symbols.get(cell, ' ') for cell in row] for row in self.grid]
        x, y = self.agent_position
        grid_display[x][y] = 'A' 
        for row in grid_display:
            print(" ".join(row))
