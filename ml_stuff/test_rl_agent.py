import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from config import config
from environment import RunnerEnv
from lstm_model import LSTMDynamicsModel

def load_models(env, ppo_path="ppo_model.zip", lstm_path="lstm_model.pth"):
    model = PPO.load(ppo_path, env=env)
    lstm_model = LSTMDynamicsModel(state_size=env.observation_space.shape[0], action_size=env.action_space.n, hidden_size=64, device='cuda')
    lstm_model.load_state_dict(torch.load(lstm_path))
    print("Models loaded.")
    return model, lstm_model

def plot_game(env, ax):
    ax.clear()
    grid = np.zeros((env.grid_h, env.grid_w), dtype=int)
    grid[env.runner_y, env.runner_x] = 1
    grid[env.target_y, env.target_x] = 2
    
    print(env.runner_x, env.runner_y, env.target_x, env.target_y)

    cmap = plt.cm.get_cmap('viridis', 3)  # 3 discrete colors
    ax.imshow(grid, cmap=cmap, interpolation='nearest')
    ax.set_xticks(np.arange(-.5, env.grid_w, 1), minor=True)
    ax.set_yticks(np.arange(-.5, env.grid_h, 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=2)
    ax.tick_params(which='minor', size=0)
    ax.set_xticks([])
    ax.set_yticks([])

def main():
    env = RunnerEnv(config)
    ppo_path = "ppo_model.zip"
    lstm_path = "lstm_model.pth"

    if os.path.exists(ppo_path) and os.path.exists(lstm_path):
        model, lstm_model = load_models(env, ppo_path, lstm_path)
    else:
        print("Models not found. Please train the models first.")
        return

    obs, _ = env.reset()
    done = False
    fig, ax = plt.subplots(figsize=(12,12))
    plt.ion()
    plt.show()


    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, truncated, _ = env.step(action)
        print(f"Action: {env.action_list[action]}")
        plot_game(env, ax)
        plt.pause(0.5)  # Adjust the pause duration as needed

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()