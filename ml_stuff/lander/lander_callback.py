import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import torch

from environment import test_rl_agent

class LanderCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(LanderCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self):
        #test_rl_agent(self.rl_model)
        return True