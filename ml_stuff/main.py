from stable_baselines3 import PPO
import torch
from config import config
from environment import RunnerEnv
from lstm_model import LSTMDynamicsModel
from replay_buffer import LSTMReplayBuffer
from lstm_callback import LSTMTrainerCallback

def main():
    env = RunnerEnv(config)
    device = "cpu" #"cuda" if torch.cuda.is_available() else "cpu"
    
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    lstm_model = LSTMDynamicsModel(state_size=state_size, action_size=action_size, hidden_size=64, device=device)
    lstm_buffer = LSTMReplayBuffer(max_size=10000)

    # Create PPO model
    model = PPO("MlpPolicy", env, verbose=1, n_steps=2048, batch_size=64, n_epochs=10)
    # Create callback
    lstm_callback = LSTMTrainerCallback(rl_model=model, lstm_model=lstm_model, lstm_buffer=lstm_buffer, train_freq=5, batch_size=64, verbose=1)

    # Train PPO model with LSTM callback
    model.learn(total_timesteps=50000, callback=lstm_callback)

    # Test the LSTM prediction
    test_obs = env.reset()
    test_action = env.action_space.sample()
    pred_ns = lstm_model.predict_next_state(test_obs, test_action)
    print("Predicted next state:", pred_ns)

    env.state = test_obs.copy()
    real_ns, _, _, _ = env.step(test_action)
    print("Real next state:", real_ns)


if __name__ == "__main__":
    main()
