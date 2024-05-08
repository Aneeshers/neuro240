import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

goal_color_map = {2: 'red', 3: 'blue', 4: 'green', 5: 'yellow'}
deterministic_color_map = {0: 'orange', 1: 'purple', 2: 'brown', 3: 'cyan'}
agent_type_color_map = {'r': 'pink', 'd': 'cyan', 'v': None}

def visualize_latent_log(file_path="latent_space_log.json"):
    latent_data = []
    agent_types = []
    goal_colors = []
    deterministic_directions = []

    with open(file_path, "r") as f:
        for line in f:
            entry = json.loads(line)
            latent_data.append(entry['latent'])
            agent_type = entry['agent_type']
            agent_types.append(agent_type)
            if agent_type == 'v':
                goal_colors.append(goal_color_map.get(entry['goal_color'], 'black'))
                deterministic_directions.append(None) 
            elif agent_type == 'd':
                deterministic_directions.append(entry['goal_color'])
                goal_colors.append(None)
            else:
                goal_colors.append(agent_type_color_map[agent_type])
                deterministic_directions.append(None)
    latent_data_np = np.array(latent_data)
    tsne = TSNE(n_components=2, random_state=42)
    tsne_result = tsne.fit_transform(latent_data_np)

    plt.figure(figsize=(10, 8))
    for i, (x, y) in enumerate(tsne_result):
        agent_type = agent_types[i]
        color = 'black'

        if agent_type == 'v':
            color = goal_colors[i]
        elif agent_type == 'd':
            direction = deterministic_directions[i]
            color = deterministic_color_map.get(direction, 'cyan')
        else:
            color = goal_colors[i]

        plt.scatter(x, y, color=color, label=agent_type if i == 0 else "", alpha=0.7)

    plt.grid(True)
    plt.show()