import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import torch
from config import config

from environment import RunnerEnv
from test_rl_agent import test_rl_agent

class LSTMTrainerCallback(BaseCallback):
    def __init__(self, rl_model: PPO, lstm_model, lstm_buffer, train_freq=5, batch_size=64, verbose=0):
        super(LSTMTrainerCallback, self).__init__(verbose)
        self.rl_model = rl_model
        self.lstm_model = lstm_model
        self.lstm_buffer = lstm_buffer
        self.train_freq = train_freq
        self.batch_size = batch_size
        self.step_count = 0

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self):
        #test_rl_agent(self.rl_model)
        return True


        rollout_buffer = self.rl_model.rollout_buffer
        obs = rollout_buffer.observations
        acts = rollout_buffer.actions

        # Compute next_observations manually
        next_obs = obs[1:]

        # Flatten assuming single environment
        obs = obs[:-1, 0, :]
        acts = acts[:, 0]
        next_obs = next_obs[:, 0, :]

        for s, a, ns in zip(obs, acts, next_obs):
            self.lstm_buffer.store(s, int(a), ns)

        self.step_count += 1
        if self.lstm_buffer.size() > self.batch_size and self.step_count % self.train_freq == 0:
            s, a, ns = self.lstm_buffer.sample_batch(self.batch_size)
            loss = self.lstm_model.train_batch(s, a, ns)
            if self.verbose > 0:
                env = RunnerEnv(config)
                test_obs, _ = env.create_random_state()
                test_action = env.action_space.sample()
                pred_ns = self.lstm_model.predict_next_state(test_obs, test_action)
                print("Predicted next state:", pred_ns)
                env.state = tuple(list(test_obs))
                real_ns, _, _, _, _ = env.step(test_action)
                print("Real next state:", real_ns)

                distance = np.linalg.norm(pred_ns.cpu().numpy() - real_ns)
                print("Distance between predicted and real next state:", distance)
                print(f"LSTM training step. Loss: {loss:.4f}")


                with open("lstm_training_loss.log", "a") as f:
                    print(f"{loss:.4f}", file=f)

                with open("lstm_error.log", "a") as f:
                    print(f"{distance:.4f}", file=f)
        return True
