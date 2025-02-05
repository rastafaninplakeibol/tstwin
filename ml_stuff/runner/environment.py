import random
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class RunnerEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.grid_w = config["grid_width"]
        self.grid_h = config["grid_height"]
        
        # State: [runner_x, runner_y, target_x, target_y, energy, steps_remaining]
        # steps_remaining might be useful if you want a time constraint
        self.state_size = 6
        self.action_list = config["actions"]
        self.action_size = len(self.action_list)

        # Observation space: normalized positions and energy
        # runner_x, runner_y, target_x, target_y normalized by grid size
        # energy normalized by max_energy
        # steps_remaining normalized by max_steps
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.state_size,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.action_size)

        self.max_steps = config["max_steps"]
        self.max_energy = config["max_energy"]
        self.closest_distance = 999999999
        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        # Random spawn for runner
        self.runner_x = np.random.randint(self.grid_w)
        self.runner_y = np.random.randint(self.grid_h)
        # Random target far away
        self.target_x = np.random.randint(self.grid_w)
        self.target_y = np.random.randint(self.grid_h)
        # Ensure target is at least 50 units away from the runner
        while np.linalg.norm([self.target_x - self.runner_x, self.target_y - self.runner_y]) < 50:
            self.target_x = np.random.randint(self.grid_w)
            self.target_y = np.random.randint(self.grid_h)

        self.energy = self.config["initial_energy"]
        self.closest_distance = 999999999

        return self._get_obs(), {}
    
    def create_random_state(self):
        self.steps = random.randint(0, self.max_steps)
        # Random spawn for runner
        self.runner_x = np.random.randint(self.grid_w)
        self.runner_y = np.random.randint(self.grid_h)
        # Random target far away
        self.target_x = np.random.randint(self.grid_w)
        self.target_y = np.random.randint(self.grid_h)
        # Ensure target is not the same cell
        while self.target_x == self.runner_x and self.target_y == self.runner_y:
            self.target_x = np.random.randint(self.grid_w)
            self.target_y = np.random.randint(self.grid_h)

        self.energy = random.randint(0, self.max_energy)
        self.closest_distance = np.linalg.norm([self.runner_x - self.target_x, self.runner_y - self.target_y])
        return self._get_obs(), {}


    def _get_obs(self):
        # Normalize positions: x and y in [-1,1] by (pos/(grid_size-1)*2 -1)
        runner_x_norm = 2.0*(self.runner_x / (self.grid_w - 1)) - 1.0
        runner_y_norm = 2.0*(self.runner_y / (self.grid_h - 1)) - 1.0
        target_x_norm = 2.0*(self.target_x / (self.grid_w - 1)) - 1.0
        target_y_norm = 2.0*(self.target_y / (self.grid_h - 1)) - 1.0
        energy_norm = 2.0*(self.energy / self.max_energy) - 1.0
        steps_remaining_norm = 2.0*((self.max_steps - self.steps) / self.max_steps) - 1.0

        return np.array([runner_x_norm, runner_y_norm,
                         target_x_norm, target_y_norm,
                         energy_norm, steps_remaining_norm], dtype=np.float32)

    def step(self, action):
        prev_dist_to_target = np.linalg.norm([self.runner_x - self.target_x, self.runner_y - self.target_y])

        #if prev_dist_to_target < 20:
        #    if self.runner_x < self.target_x:
        #        self.target_x = min(self.grid_w - 2, self.target_x + 1)
        #    elif self.runner_x > self.target_x:
        #        self.target_x = max(1, self.target_x - 1)
        #
        #    if self.runner_y < self.target_y:
        #        self.target_y = min(self.grid_h - 2, self.target_y + 1)
        #    elif self.runner_y > self.target_y:
        #        self.target_y = max(1, self.target_y - 1)
        #else:
        #    mv_x = np.random.randint(-1, 2)
        #    mv_y = np.random.randint(-1, 2)
        #    self.target_x = np.clip(self.target_x + mv_x, 0, self.grid_w - 1)
        #    self.target_y = np.clip(self.target_y + mv_y, 0, self.grid_h - 1)

        move_type, direction = self.action_list[action]
        reward = 0
        done = False
        truncated = False
        dist = 1

        # Move the runner
        if direction == "up":
            self.runner_y = max(0, self.runner_y - dist)
        elif direction == "down":
            self.runner_y = min(self.grid_h - 1, self.runner_y + dist)
        elif direction == "left":
            self.runner_x = max(0, self.runner_x - dist)
        elif direction == "right":
            self.runner_x = min(self.grid_w - 1, self.runner_x + dist)

        self.steps += 1

        if self.runner_x == self.target_x and self.runner_y == self.target_y:
            reward = self.config["reward_reach_target"] + (0.1 * (self.max_steps - self.steps))
            done = True  
            return self._get_obs(), reward, done, truncated, {}

        reward = -1

        if self.steps >= self.max_steps:
            truncated = True
            reward = -self.config["reward_distance_factor"] * prev_dist_to_target
        return self._get_obs(), reward, done, truncated, {}

    def render(self, mode='human'):
        grid = np.zeros((self.grid_h, self.grid_w), dtype=str)
        grid[:] = '.'
        grid[self.runner_y, self.runner_x] = 'R'
        grid[self.target_y, self.target_x] = 'T'
        
        print("\n".join(" ".join(row) for row in grid))
        print(f"Energy: {self.energy}, Steps: {self.steps}/{self.max_steps}")
