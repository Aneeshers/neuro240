import json

def log_latent_space(latent, agent_types, goal_colors, file_path="latent_space_log.json"):
    log_data = []
    for i in range(latent.size(0)):
        log_entry = {
            "latent": latent[i].tolist(),
            "agent_type": chr(agent_types[i].item()),
            "goal_color": int(goal_colors[i].item())
        }
        log_data.append(log_entry)

    with open(file_path, "a") as f:
        for entry in log_data:
            json.dump(entry, f)
            f.write("\n")