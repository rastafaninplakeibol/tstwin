from stable_baselines3.common.callbacks import BaseCallback
import torch

class LSTMTrainerCallback(BaseCallback):
    def __init__(self, rl_model, lstm_model, lstm_buffer, train_freq=5, batch_size=64, verbose=0):
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
                print(f"LSTM training step. Loss: {loss:.4f}")
        return True
