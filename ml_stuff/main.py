import os
import random
import signal
import numpy as np
import torch
from stable_baselines3 import PPO
from config import config
from environment import RunnerEnv
from lstm_model import LSTMDynamicsModel
from replay_buffer import LSTMReplayBuffer
from lstm_callback import LSTMTrainerCallback

def seed_everything(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_models(model, lstm_model, ppo_path="ppo_model.zip", lstm_path="lstm_model.pth"):
    model.save(ppo_path)
    torch.save(lstm_model.state_dict(), lstm_path)
    print("Models saved.")

def load_models(env, ppo_path="ppo_model.zip", lstm_path="lstm_model.pth"):
    rl_model = PPO.load(ppo_path, env=env, device='cpu')
    lstm_model = LSTMDynamicsModel(state_size=env.observation_space.shape[0], action_size=env.action_space.n, hidden_size=64, device='cuda')
    lstm_model.load_state_dict(torch.load(lstm_path))
    print("Models loaded.")
    return rl_model, lstm_model

def main():
    seed_everything(42)
    env = RunnerEnv(config)
    device = "cpu" #"cuda" if torch.cuda.is_available() else "cpu"
    
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    lstm_buffer = LSTMReplayBuffer(max_size=10000)

    ppo_path = "ppo_model.zip"
    lstm_path = "lstm_model.pth"

    #with open("lstm_training_loss.log", "w") as f:
    #    print("Reset training loss log.")
    #with open("lstm_error.log", "w") as f:
    #    print("Reset error log.")

    if os.path.exists(ppo_path) and os.path.exists(lstm_path):
        model, lstm_model = load_models(env, ppo_path, lstm_path)
    else:
        lstm_model = LSTMDynamicsModel(state_size=state_size, action_size=action_size, hidden_size=64, device='cuda')
        model = PPO("MlpPolicy", env, verbose=1, n_steps=2048, batch_size=64, n_epochs=10, device='cpu')

    lstm_callback = LSTMTrainerCallback(rl_model=model, lstm_model=lstm_model, lstm_buffer=lstm_buffer, train_freq=5, batch_size=64, verbose=1)

    def signal_handler(sig, frame):
        save_models(model, lstm_model, ppo_path, lstm_path)
        exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    # Train PPO model with LSTM callback
    model.learn(total_timesteps=5000000, callback=lstm_callback, log_interval=20, reset_num_timesteps=False)
    save_models(model, lstm_model, ppo_path, lstm_path)

if __name__ == "__main__":
    main()