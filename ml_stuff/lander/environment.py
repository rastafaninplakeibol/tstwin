import random
from time import sleep
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import pymunk
import pymunk.pygame_util

'''
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
'''

config = {
    "reward_crashed": -1000,
    "reward_landed": 1000,
    "reward_per_step": -1,
    "reward_per_distance": -2,
}




class LanderEnv(gym.Env):
    def __init__(self, width=1600, height=1000,
                 gravity=900,           # gravity strength (pixels/s^2)
                 thrust_force=15000,    # force applied when rocket is fired
                 tilt_torque=8000       # torque applied for tilting the lander
                 ):
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

        self.reset()

    def reset(self):
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
        self.lander_body.position = (self.width / 2, self.height - 100)

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
    
    def are_colliding(shape1, shape2):
        return len(shape1.shapes_collide(shape2).points) > 0

    def step(self, action):
        """
        Action is a dictionary with:
        - "thrust": Boolean indicating if the rocket vent is fired.
        - "tilt": -1 for counter-clockwise torque, 1 for clockwise, 0 for no tilt.
        """
        # Apply thrust force if action["thrust"] is True.
        thrust = action.get("thrust", 0.0)
        
        if thrust > 0:
            local_thrust = (0, thrust)    # Upward in local coords
            local_anchor = (0, 0) # Bottom center in local coords
            self.lander_body.apply_force_at_local_point(local_thrust, local_anchor)

        # Apply torque for tilting the lander
        tilt = action.get("tilt", 0.0)
        if tilt != 0:
            self.lander_body.torque = -tilt

        # Step the physics simulation
        dt = 1 / 60.0
        self.space.step(dt)

        # Get observation, reward, done
        obs = self.get_observation()
        reward = 0.0
        done = False
        truncated = False
        info = {}

        #print("Right wall: ", self.lander_shape.shapes_collide(self.right_wall).points)
        #print("Left wall: ", self.lander_shape.shapes_collide(self.left_wall))
        #print("Ceiling: ", self.lander_shape.shapes_collide(self.ceiling))
        #print("Floor: ", self.lander_shape.shapes_collide(self.floor))


        print("Velocity: ", self.previous_velocity)
        if LanderEnv.are_colliding(self.lander_shape, self.left_wall) or LanderEnv.are_colliding(self.lander_shape, self.right_wall) or LanderEnv.are_colliding(self.lander_shape, self.ceiling):
            truncated = True
            print("Crashed!")
        elif LanderEnv.are_colliding(self.lander_shape, self.floor):

            #check if it is horizontal to the floor
            if abs(self.lander_body.angle) < 0.1:
                if abs(self.previous_velocity) < 200:
                    print("Landed!")
                    done = True
                else:
                    print("Crashed!")
                    truncated = True
            else:
                print("Crashed!")
                truncated = True

            

        self.previous_velocity = self.lander_body.velocity.y

        if truncated:
            reward = config["reward_crashed"]
        elif done:
            reward = config["reward_landed"]
            reward += config["reward_per_distance"] * abs((self.flag_position[0] - self.lander_body.position.x) // 10)

        reward += config["reward_per_step"]

        self.last_action = action
        return obs, reward, done, truncated, info


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


        thrust = self.last_action.get("thrust", 0.0)
        if self.last_action and thrust > 0:
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


def run_game():
    env = LanderEnv()
    obs = env.reset()
    running = True

    total_reward = 0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        action = {
            "thrust": 15000.0 if keys[pygame.K_UP] else 0.0,
            "tilt": -8000.0 if keys[pygame.K_LEFT] else (8000.0 if keys[pygame.K_RIGHT] else 0.0)
        }

        obs, reward, done, truncated, info = env.step(action)
        print("Reward:", reward)
        env.render()
        total_reward += reward

        if done or truncated:
            print("Total reward:", total_reward)
            obs = env.reset()
            total_reward = 0
            running = False
    env.close()


if __name__ == "__main__":
    while True:
        run_game()
        sleep(5)