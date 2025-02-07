import random
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import pymunk
import pymunk.pygame_util
from stable_baselines3 import PPO
import torch

def seed_everything(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

config = {
    "reward_crashed": -200,
    "reward_landed": 200,
    "reward_per_step": -0.01,
    "reward_per_distance_landing": -0.1,
    "reward_wrong_tilt": -1,
    "reward_per_velocity": -0.1,
    "reward_getting_closer": 0.1
}




class LanderEnv(gym.Env):
    #do not render
    render_mode = None

    def __init__(self, width=1600, height=1000,
                 gravity=900,           # gravity strength (pixels/s^2)
                 thrust_force=15000,    # force applied when rocket is fired
                 tilt_torque=8000       # torque applied for tilting the lander
                 ):
        super().__init__()
        self.width = width
        self.height = height
        self.gravity = gravity
        self.max_thrust_force = thrust_force
        self.max_tilt_torque = tilt_torque

        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Lander Simulation")
        self.clock = pygame.time.Clock()

        self.space = pymunk.Space(threaded=True)
        self.space.threads = 2
        self.space.gravity = (0, -self.gravity)
        self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)

        self.lander_body = None
        self.lander_shape = None

        self.observation_space = spaces.Box(
            low=np.array([0, 0, -np.inf, -np.inf, -2*np.pi, -np.inf]),
            high=np.array([self.width, self.height, np.inf, np.inf, 2*np.pi, np.inf]),
            dtype=np.float64
        )

        self.action_space = spaces.Box(
            low=np.array([0.0, -self.max_tilt_torque]),
            high=np.array([self.max_thrust_force, self.max_tilt_torque]),
            dtype=np.float64
        )


        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        for body in self.space.bodies[:]:
            self.space.remove(body)
        for shape in self.space.shapes[:]:
            self.space.remove(shape)

        floor_body = pymunk.Body(body_type=pymunk.Body.STATIC)
        floor_y = 60
        floor_shape = pymunk.Segment(floor_body, (0, floor_y), (self.width, floor_y), 25)
        floor_shape.friction = 1.0
        self.floor = floor_shape
        self.space.add(floor_body, floor_shape)

        ceiling_body = pymunk.Body(body_type=pymunk.Body.STATIC)
        ceiling_shape = pymunk.Segment(ceiling_body, (0, self.height - 50), (self.width, self.height - 50), 25)
        ceiling_shape.friction = 1.0
        self.ceiling = ceiling_shape
        self.space.add(ceiling_body, ceiling_shape)

        #left wall
        left_wall_body = pymunk.Body(body_type=pymunk.Body.STATIC)
        left_wall_shape = pymunk.Segment(left_wall_body, (0, 0), (0, self.height), 25)
        left_wall_shape.friction = 1.0
        self.left_wall = left_wall_shape
        self.space.add(left_wall_body, left_wall_shape)

        #right wall
        right_wall_body = pymunk.Body(body_type=pymunk.Body.STATIC)
        right_wall_shape = pymunk.Segment(right_wall_body, (self.width, 0), (self.width, self.height), 25)  
        right_wall_shape.friction = 1.0
        self.right_wall = right_wall_shape
        self.space.add(right_wall_body, right_wall_shape)

        mass = 3
        lander_width = 60
        lander_height = 30
        moment = pymunk.moment_for_box(mass, (lander_width, lander_height))
        self.lander_body = pymunk.Body(mass, moment)
        self.lander_body.position = (self.width / 2, self.height - 400)
        self.lander_body.angle = random.uniform(-np.pi / 4, np.pi / 4)

        # Define the lander shape vertices *relative to the body's center*
        lander_vertices = [
            (-lander_width / 2, -lander_height / 2),  # Bottom-left
            (lander_width / 2, -lander_height / 2),   # Bottom-right
            (lander_width / 2, lander_height / 2),    # Top-right
            (-lander_width / 2, lander_height / 2)   # Top-left
        ]
        self.lander_shape = pymunk.Poly(self.lander_body, lander_vertices)
        self.lander_shape.friction = 0.5
        self.space.add(self.lander_body, self.lander_shape)

        #generate random position for a flag on the floor
        self.flag_position = (random.randint(200, self.width - 200), self.height - floor_y - 50)
        
        self.previous_velocity = 0
        self.closest_distance_to_target = 50000
        self.previous_distance = 50000

        return self.get_observation(), {}

    def get_observation(self):
        # Return the state of the lander: position, velocity, angle, and angular velocity.
        pos = np.array([self.lander_body.position.x, self.lander_body.position.y])
        vel = np.array([self.lander_body.velocity.x, self.lander_body.velocity.y])
        angle = self.lander_body.angle
        angular_vel = self.lander_body.angular_velocity
        #return {
        #    "position": pos,
        #    "velocity": vel,
        #    "angle": angle % (2 * np.pi),
        #    "angular_velocity": angular_vel
        #}
        return np.array([pos[0], pos[1], vel[0], vel[1], angle, angular_vel], dtype=np.float64)

    
    def are_colliding(shape1, shape2):
        return len(shape1.shapes_collide(shape2).points) > 0

    def step(self, action):
        """
        Action is a dictionary with:
        - "thrust": Boolean indicating if the rocket vent is fired.
        - "tilt": -1 for counter-clockwise torque, 1 for clockwise, 0 for no tilt.
        """
        #print("Action: ", action)
        thrust, tilt = action[0], action[1]

        #map thrust into the range [0, max_thrust_force]

        
        thrust = np.clip(thrust, 0, 1) * self.max_thrust_force        
        tilt = np.tanh(tilt) * self.max_tilt_torque

        action = [thrust, tilt]
        #print("Action: ", action)
        
        if thrust > 0:
            local_thrust = (0, thrust)    # Upward in local coords
            local_anchor = (0, 0) # Bottom center in local coords
            self.lander_body.apply_force_at_local_point(local_thrust, local_anchor)

        # Apply torque for tilting the lander
        if tilt != 0:
            self.lander_body.torque = -tilt

        # Step the physics simulation
        dt = 1 / 60.0
        self.space.step(dt)

        # Get observation, reward, terminated
        reward = 0.0
        terminated = False
        truncated = False
        info = {}

        #print("Velocity: ", self.previous_velocity)
        if LanderEnv.are_colliding(self.lander_shape, self.left_wall) or LanderEnv.are_colliding(self.lander_shape, self.right_wall) or LanderEnv.are_colliding(self.lander_shape, self.ceiling):
            truncated = True
            #print("Crashed!")
        elif LanderEnv.are_colliding(self.lander_shape, self.floor):
            #check if it is horizontal to the floor
            if abs(self.lander_body.angle) < np.pi / 18:
                if abs(self.previous_velocity) < 200:
                    #print("Landed!")
                    terminated = True
                else:
                    #print("Crashed!")
                    truncated = True
            else:
                #print("Crashed!")
                truncated = True

            
        distance_to_target = np.linalg.norm(np.array(self.lander_body.position) - np.array(self.flag_position))
        
        self.previous_velocity = self.lander_body.velocity.y
    
        if truncated:
            reward += config["reward_crashed"]
        elif terminated:
            reward += np.tanh(config["reward_per_distance_landing"] * distance_to_target)
            reward += config["reward_landed"]

        angle = abs(self.lander_body.angle % (2 * np.pi))
        
        reward -= np.tanh(angle) * config["reward_wrong_tilt"]
        
        if self.closest_distance_to_target > distance_to_target:
            self.closest_distance_to_target = distance_to_target
            reward += config["reward_getting_closer"] * 10

        if self.previous_distance < distance_to_target:
            reward -= config["reward_getting_closer"] * (self.previous_distance - distance_to_target)

        velocity = self.lander_body.velocity.y
        if velocity > 200 or velocity < -100:
            reward += config["reward_per_velocity"] * np.tanh(velocity)

        #reward += config["reward_per_step"]

        self.last_action = action
        self.previous_distance = distance_to_target

        obs = self.get_observation()
        return obs, reward, terminated, truncated, info


    def render(self):
        self.screen.fill((255, 255, 255))
        #self.space.debug_draw(self.draw_options)

        angle = self.lander_body.angle

        # Get world coordinates of the lander's vertices
        vertices = [self.lander_body.local_to_world(v) for v in self.lander_shape.get_vertices()]

        # Convert to Pygame coordinates (origin at top-left, y-axis inverted)
        pygame_vertices = [((v.x), (self.height - v.y)) for v in vertices]

        # Separate vertices for top and bottom halves
        bottom_left = pygame_vertices[0]
        bottom_right = pygame_vertices[1]
        top_right = pygame_vertices[2]
        top_left = pygame_vertices[3]

        # Calculate the midpoint of the left and right edges
        mid_left = ((top_left[0] + bottom_left[0]) / 2, (top_left[1] + bottom_left[1]) / 2)
        mid_right = ((top_right[0] + bottom_right[0]) / 2, (top_right[1] + bottom_right[1]) / 2)

        # Define the top and bottom vertices
        top_vertices = [top_left, top_right, mid_right, mid_left]
        bottom_vertices = [mid_left, mid_right, bottom_right, bottom_left]

        # Draw the top half of the lander (black)
        pygame.draw.polygon(self.screen, (0, 0, 0), top_vertices)

        # Draw the bottom half of the lander (red)
        pygame.draw.polygon(self.screen, (255, 0, 0), bottom_vertices)

        thrust, tilt = self.last_action[0], self.last_action[1]

        if thrust > 0:
            thrust_magnitude = thrust / 1000
            mid_bottom = ((bottom_left[0] + bottom_right[0]) / 2, (bottom_left[1] + bottom_right[1]) / 2 + 10)

            # Thrust vector is now calculated correctly from the bottom center
            thrust_direction = pymunk.Vec2d(0, thrust_magnitude).rotated(angle + np.pi)
            thrust_end = (int(mid_bottom[0] + thrust_direction.x), int(mid_bottom[1] - thrust_direction.y))
            pygame.draw.line(self.screen, (255, 255, 0), mid_bottom, thrust_end, 8)

        #render walls floor and ceiling
        pygame.draw.line(self.screen, (0, 0, 0), (0, 25), (self.width, 25), 100)
        pygame.draw.line(self.screen, (0, 0, 0), (0, self.height - 40), (self.width, self.height - 40), 100)
        pygame.draw.line(self.screen, (0, 0, 0), (0, 0), (0, self.height), 50)
        pygame.draw.line(self.screen, (0, 0, 0), (self.width, 0), (self.width, self.height), 50)
        #render flag as a green line on the floor
        pygame.draw.line(self.screen, (0, 255, 0), (self.flag_position[0], self.flag_position[1]), (self.flag_position[0], self.flag_position[1] + 20), 10)

        pygame.display.update()
        self.clock.tick(60)

    def close(self):
        pygame.quit()



def load_model(env, ppo_path="ppo_model.zip"):
    rl_model = PPO.load(ppo_path, env=env, device='cpu')
    print("Models loaded.")
    return rl_model

def test_rl_agent():
    env = LanderEnv()
    rl_agent = load_model(env=env)
    obs, _ = env.reset()
    running = True

    total_reward = 0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        action, _ = rl_agent.predict(obs)
        obs, reward, terminated, truncated, _ = env.step(action)

        print("Reward:", reward)
        env.render()
        total_reward += reward

        if terminated or truncated:
            print("Total reward:", total_reward)
            obs = env.reset()
            total_reward = 0
            running = False
    env.close()


if __name__ == "__main__":
    while True:
        test_rl_agent()
        #sleep(5)
