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
from lander_callback import LanderCallback
import gymnasium as gym

def save_models(rl_model, ppo_path="ppo_model.zip"):
    rl_model.save(ppo_path)
    print("Models saved.")

def load_model(env, ppo_path="ppo_model.zip"):
    rl_model = PPO.load(ppo_path, env=env, device='cpu')
    print("Models loaded.")
    return rl_model

def make_env(config, rank, seed=42):
    def _init():
        env = LanderEnv()
        #env = Monitor(env)
        #env.seed(seed + rank)
        return env
    return _init

def main():
    #seed_everything(42)
    #print(gym.envs.registry.keys())
    #exit(0)

    num_envs = 20 # Number of parallel environments
    config = {
        "width": 1600,
        "height": 1000,
        "gravity": 900,
        "thrust_force": 15000,
        "tilt_torque": 8000
    }
    
    if num_envs == 1:
        env = LanderEnv()
    else: 
        env_fns = [make_env(config, i) for i in range(num_envs)]
        env = SubprocVecEnv(env_fns)
        env = VecMonitor(env)

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
    model.learn(total_timesteps=50000000, callback=LanderCallback(), log_interval=20, reset_num_timesteps=False)
    save_models(model, ppo_path)
    
if __name__ == "__main__":
    main()
