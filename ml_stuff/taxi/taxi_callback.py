import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import torch

class TaxiCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(TaxiCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self):
        #test_rl_agent(self.rl_model)
        return True