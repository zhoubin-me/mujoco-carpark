
import gym
import numpy as np
import pybullet as p
import pybullet_data
from gym import spaces
import time

class CarParkingEnv(gym.Env):
    def __init__(self):
        super(CarParkingEnv, self).__init__()

        # Initialize PyBullet
        self.physics_client = p.connect(p.GUI)
        p.setGravity(0, 0, -9.8)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Load plane and car
        self.plane_id = p.loadURDF("plane.urdf")
        self.car_id = p.loadURDF("racecar/racecar.urdf", [0, 0, 0.2])

        # Define action and observation spaces
        self.action_space = spaces.Box(low=np.array([-1, -1]), high=np.array([1, 1]), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)

        # Parking spot coordinates
        self.parking_spot = np.array([5, 5])

        # Simulation parameters
        self.max_steps = 500
        self.current_step = 0

    def reset(self):
        # Reset environment and car position
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        p.loadURDF("plane.urdf")
        self.car_id = p.loadURDF("racecar/racecar.urdf", [0, 0, 0.2])

        # Reset step count
        self.current_step = 0
        obs = self._get_observation().astype(np.float32)
        # Return initial observation: tuple obs, info
        return obs, {}

    def _get_observation(self):
        pos, ori = p.getBasePositionAndOrientation(self.car_id)
        linear_vel, angular_vel = p.getBaseVelocity(self.car_id)
        distance_to_goal = np.linalg.norm(np.array(pos[:2]) - self.parking_spot)

        obs = np.array([
            pos[0], pos[1], pos[2],
            ori[0], ori[1], ori[2],
            linear_vel[0], linear_vel[1],
            distance_to_goal
        ])
        return obs

    def step(self, action):
        throttle, steering = action
        p.setJointMotorControl2(self.car_id, 0, p.VELOCITY_CONTROL, targetVelocity=throttle * 10)
        p.setJointMotorControl2(self.car_id, 2, p.VELOCITY_CONTROL, targetVelocity=steering * 2)
        p.stepSimulation()

        # Get observation
        obs = self._get_observation().astype(np.float32)  # Ensure float32 type

        # Reward calculation
        reward = self._calculate_reward(obs)
        done = bool(self._is_done(obs))  # Ensure boolean type

        self.current_step += 1
        return obs, reward, done, {}

    def _calculate_reward(self, obs):
        pos = obs[:3]
        distance_to_goal = np.linalg.norm(np.array(pos[:2]) - self.parking_spot)

        # Reward for getting closer to the spot
        reward = -distance_to_goal

        # Penalty for collisions
        contacts = p.getContactPoints(bodyA=self.car_id)
        if len(contacts) > 0:
            reward -= 50

        # Success reward for parking correctly
        if distance_to_goal < 0.5:
            reward += 100

        # Time penalty to encourage efficient parking
        reward -= 0.1

        return reward

    def _is_done(self, obs):
        distance_to_goal = obs[8]

        # Check if parked successfully
        if distance_to_goal < 0.5:
            return True

        # Check if maximum steps reached
        if self.current_step >= self.max_steps:
            return True

        return False

    def render(self, mode='human'):
        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=[0, 0, 0],
            distance=10,
            yaw=30,
            pitch=-30,
            roll=0,
            upAxisIndex=2
        )
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60, aspect=1.0, nearVal=0.1, farVal=100.0
        )
        (_, _, px, _, _) = p.getCameraImage(
            width=640, height=480, viewMatrix=view_matrix, projectionMatrix=proj_matrix
        )
        return np.array(px)

    def close(self):
        p.disconnect()

# Register the environment
gym.envs.registration.register(
    id='CarParking-v0',
    entry_point='car_parking_env:CarParkingEnv',
)
