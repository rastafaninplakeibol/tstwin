import random
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import pymunk
import pymunk.pygame_util


class LanderEnv(gym.Env):
    #metadata = {'render.modes': ['human']}

    def __init__(self, config):
        super().__init__()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        
        return self._get_obs(), {}

    def _get_obs(self):
        return np.array([], dtype=np.float32)

    def step(self, action):

        reward = 0
        terminated = False
        truncated = False

        return self._get_obs(), reward, terminated, truncated, {}




class LanderEnv:
    def __init__(self, width=800, height=600,
                 gravity=900,           # gravity strength (pixels/s^2)
                 thrust_force=15000,    # force applied when rocket is fired
                 tilt_torque=5000       # torque applied for tilting the lander
                 ):
        self.width = width
        self.height = height
        self.gravity = gravity
        self.thrust_force = thrust_force
        self.tilt_torque = tilt_torque

        # Initialize Pygame and set up display
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Lander Simulation")
        self.clock = pygame.time.Clock()

        # Initialize Pymunk space (physics simulation)
        self.space = pymunk.Space()
        # In Pymunk, positive y goes upward; so to simulate gravity downward,
        # set gravity to (0, -gravity)
        self.space.gravity = (0, -self.gravity)
        self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)

        self.lander_body = None  # will be created in reset()
        self.lander_shape = None

        self.reset()

    def reset(self):
        # Clear the physics space
        for body in self.space.bodies[:]:
            self.space.remove(body)
        for shape in self.space.shapes[:]:
            self.space.remove(shape)

        # Create static floor (the terrain)
        floor_body = pymunk.Body(body_type=pymunk.Body.STATIC)
        floor_y = 50  # Floor y-position from the bottom
        floor_shape = pymunk.Segment(floor_body, (0, floor_y), (self.width, floor_y), 5)
        floor_shape.friction = 1.0
        self.space.add(floor_body, floor_shape)

        # Create the lander as a dynamic body (a black box)
        mass = 1
        lander_width = 40
        lander_height = 20
        moment = pymunk.moment_for_box(mass, (lander_width, lander_height))
        self.lander_body = pymunk.Body(mass, moment)
        # Start near the top center
        self.lander_body.position = (self.width / 2, self.height - 100)
        self.lander_shape = pymunk.Poly.create_box(self.lander_body, (lander_width, lander_height))
        self.lander_shape.friction = 0.5
        # Color is set during rendering (we'll draw it as black)
        self.space.add(self.lander_body, self.lander_shape)

        return self.get_observation()

    def get_observation(self):
        # Return the state of the lander: position, velocity, angle, and angular velocity.
        pos = np.array(self.lander_body.position)
        vel = np.array(self.lander_body.velocity)
        angle = self.lander_body.angle
        angular_vel = self.lander_body.angular_velocity
        return {
            "position": pos,
            "velocity": vel,
            "angle": angle,
            "angular_velocity": angular_vel
        }

    def step(self, action):
        """
        Action is a dictionary with:
         - "thrust": Boolean indicating if the rocket vent is fired.
         - "tilt": -1 for counter-clockwise torque, 1 for clockwise, 0 for no tilt.
        """
        # Apply thrust force if action["thrust"] is True.
        # The thrust is applied relative to the lander's orientation.
        if action.get("thrust", False):
            angle = self.lander_body.angle
            # Compute force vector: thrust is applied in the direction opposite to the lander's "down"
            # (i.e., it pushes the lander upward relative to its orientation).
            force = (-self.thrust_force * np.sin(angle),
                      self.thrust_force * np.cos(angle))
            self.lander_body.apply_force_at_local_point(force, (0, 0))

        # Apply torque for tilting the lander
        tilt = action.get("tilt", 0)
        if tilt != 0:
            self.lander_body.apply_torque(tilt * self.tilt_torque)

        # Step the physics simulation
        dt = 1 / 60.0
        self.space.step(dt)

        # Get observation, reward, done (set blank for now)
        obs = self.get_observation()
        reward = 0.0  # Placeholder: define your reward later
        done = False  # Placeholder: define termination condition later
        info = {}     # Additional info if needed

        return obs, reward, done, info

    def render(self):
        # Clear screen (white background)
        self.screen.fill((255, 255, 255))
        # Draw all objects in the physics space
        self.space.debug_draw(self.draw_options)
        # Optionally, draw the lander as a filled black rectangle.
        # Get the vertices of the lander's box:
        vertices = self.lander_shape.get_vertices()
        vertices = [(int(self.lander_body.position.x + v.x), 
                     int(self.height - (self.lander_body.position.y + v.y))) for v in vertices]
        pygame.draw.polygon(self.screen, (0, 0, 0), vertices)
        pygame.display.flip()
        self.clock.tick(60)

    def close(self):
        pygame.quit()


if __name__ == "__main__":
    # This main loop demonstrates manual control via keyboard.
    # Later, you can remove this loop and integrate your RL agent.
    env = LanderEnv()
    obs = env.reset()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Example manual controls:
        keys = pygame.key.get_pressed()
        action = {
            "thrust": keys[pygame.K_UP],         # Press UP to fire the rocket vent.
            "tilt": -1 if keys[pygame.K_LEFT] else (1 if keys[pygame.K_RIGHT] else 0)
        }

        obs, reward, done, info = env.step(action)
        env.render()

        # (Optional) Reset if a termination condition is met:
        # if done: obs = env.reset()

    env.close()
