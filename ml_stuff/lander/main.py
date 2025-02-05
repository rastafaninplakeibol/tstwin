from logging import config
import os
import random
import signal
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.monitor import Monitor

from environment import LanderEnv
from lander.lander_callback import LanderCallback
import gymnasium as gym

def seed_everything(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_models(rl_model, ppo_path="ppo_model.zip"):
    rl_model.save(ppo_path)
    print("Models saved.")

def load_model(env, ppo_path="lander_model.zip"):
    rl_model = PPO.load(ppo_path, env=env, device='cpu')
    print("Models loaded.")
    return rl_model

def make_env(config, rank, seed=42):
    def _init():
        env = LanderEnv(config)
        #env = Monitor(env)
        #env.seed(seed + rank)
        return env
    return _init

def main():
    seed_everything(42)
    num_envs = 10 # Number of parallel environments
    #print(gym.envs.registry.keys())
    #exit(0)

    config = {
        "width": 1600,
        "height": 1000,
        "gravity": 900,
        "thrust_force": 15000,
        "tilt_torque": 8000
    }

    env = LanderEnv(config)
    
    #state_size = env.observation_space.shape[0]
    #action_size = env.action_space.n

    ppo_path = "ppo_model.zip"

    if os.path.exists(ppo_path):
        model= load_model(env, ppo_path)
    else:
        model = PPO("MlpPolicy", env, verbose=1, n_steps=2048, batch_size=64, n_epochs=20, device='cpu')

    def signal_handler(sig, frame):
        save_models(model, ppo_path)
        exit(0)

    signal.signal(signal.SIGINT, signal_handler)


    # Train PPO model with LSTM callback
    model.learn(total_timesteps=5000000, callback=LanderCallback(), log_interval=20, reset_num_timesteps=False)
    save_models(model, ppo_path)
    
if __name__ == "__main__":
    main()