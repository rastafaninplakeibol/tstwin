import random
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class TaxiEnv(gym.Env):
    #metadata = {'render.modes': ['human']}

    def __init__(self, config):
        super().__init__()
        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        
        return self._get_obs(), {}

    def _get_obs(self):
        return np.array([], dtype=np.float32)

    def step(self, action):

        reward = 0
        terminated = False
        truncated = False

        #i want to simulate



        return self._get_obs(), reward, terminated, truncated, {}
