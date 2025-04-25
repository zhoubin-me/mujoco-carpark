import gymnasium as gym
import numpy as np
import mujoco
import mujoco.viewer
from gymnasium import spaces
import random
import time

def quaternion_to_yaw(q):
    w, x, y, z = q
    theta = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
    return theta


class CarParkingEnvPPO(gym.Env):
    def __init__(self, **kwargs):
        super(CarParkingEnvPPO, self).__init__()

        xml_path = "mujoco_xmls/my_car_ppo.xml"
        if 'phase' in kwargs:
            if kwargs['phase'] == 2:
                xml_path = "mujoco_xmls/my_car.xml"

        # Load the MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # Set simulation timestep
        self.model.opt.timestep = 0.02

        # Define action and observation spaces
        self.action_space = spaces.Box(
            low=np.array([-1, -1]),  # [steering, throttle]
            high=np.array([1, 1]),
            dtype=np.float32,
        )

        self.desired_goal = None

        # Observation space: Dict with achieved_goal, desired_goal, and observation
        self.observation_space = spaces.Dict(
            {
                "achieved_goal": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(6,), dtype=np.float64
                ),
                "desired_goal": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(6,), dtype=np.float64
                ),
                "observation": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(6,), dtype=np.float64
                ),
                "range_sensors": spaces.Box(
                    low=0, high=30, shape=(17,), dtype=np.float64
                ),
            }
        )

        # Configuration
        self.config = {
            "reward_weights": [1, 1, 0.0, 0.0, 0.1, 0.1],
            "reward_scale": 0.05,
            "success_distance_threshold": 0.1,
            "collision_reward": -15,
            "success_reward": 15,
            "max_steps": 250,
            "action_scale": 1.0,
        }

        # Initialize viewer
        self.viewer = None
        self.current_step = 0
        self.repeat_action = kwargs["repeat_action"] if "repeat_action" in kwargs else 4


    def reset_goal(self):
        # Get parking spot position from MuJoCo model
        goal_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "goal"
        )


        # Random a parking spot

        if random.random() < 0.5:
            self.parking_spot = np.array([5.0, np.random.uniform(-4, 4), 0.001])
        else:
            left = np.array([np.random.uniform(1, 4), -5, 0.001])
            right = np.array([np.random.uniform(1, 4), 5, 0.001])
            self.parking_spot = random.choice([left, right])


        if self.parking_spot[0] == 5.0:
            self.parking_quat = np.array([1.0, 0.0, 0.0, 0.0])
        else:
            self.parking_quat = np.array([0.70710678, 0.0, 0.0, 0.70710678])


        self.model.site_pos[goal_site_id][:] = self.parking_spot
        self.model.site_quat[goal_site_id][:] = self.parking_quat

        # Calculate heading angle from quaternion
        theta = np.rad2deg(quaternion_to_yaw(self.parking_quat))

        self.desired_goal = np.array(
            [
                self.parking_spot[0],
                self.parking_spot[1],  # target x, y
                0,
                0,  # target velocity (vx, vy)
                np.cos(theta),
                np.sin(theta),  # target heading (cos_h, sin_h) - facing positive x
            ],
            dtype=np.float64,
        )


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset the simulation
        mujoco.mj_resetData(self.model, self.data)

        # Reset car position to origin
        self.data.qpos[:3] = [0, 0, 0.1]  # x, y, z
        self.data.qpos[3:7] = [1, 0, 0, 0]  # quaternion (w, x, y, z)

        # Reset velocities
        self.data.qvel[:] = 0

        # Reset step count
        self.current_step = 0

        self.reset_goal()
        # Get initial observation
        obs = self._get_observation()

        # Initialize info dictionary
        info = {
            "speed": 0.0,
            "crashed": False,
            "action": np.array([0.0, 0.0], dtype=np.float32),
            "is_success": False,
        }

        return obs, info

    def _get_observation(self):
        # Get car position and orientation
        pos = self.data.qpos[:2]  # x, y
        quat = self.data.qpos[3:7]  # quaternion (w, x, y, z)

        # Convert quaternion to heading angle
        theta = np.rad2deg(quaternion_to_yaw(quat))
        cos_h = np.cos(theta)
        sin_h = np.sin(theta)

        # Get linear velocity (only x and y components)
        linear_vel = self.data.qvel[:2]  # vx, vy

        # Construct observation components
        achieved_goal = np.array(
            [
                pos[0],
                pos[1],  # x, y
                linear_vel[0],
                linear_vel[1],  # vx, vy
                cos_h,
                sin_h,  # heading components
            ],
            dtype=np.float64,
        )

        observation = np.array(
            [
                pos[0],
                pos[1],  # x, y
                linear_vel[0],
                linear_vel[1],  # vx, vy
                cos_h,
                sin_h,  # heading components
            ],
            dtype=np.float64,
        )

        range_sensors = 3 - np.copy(self.data.sensordata[9:]) / 10.0

        return {
            "achieved_goal": achieved_goal,
            "desired_goal": self.desired_goal,
            "observation": observation,
            "range_sensors": range_sensors,
        }

    def compute_reward(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        info: dict | None = None,
        p: float = 0.5,
    ) -> float:
        """
        Proximity to the goal is rewarded using a weighted p-norm
        """

        return (
            -np.power(
                np.dot(np.abs(achieved_goal - desired_goal), np.array(self.config["reward_weights"])),
                p,
            )
            * self.config["reward_scale"]
        )


    def step(self, action, action_repeat=4):
        # Apply actions to the car
        steering, throttle = action

        # Set steering control (position control)
        self.data.ctrl[0] = steering * self.config["action_scale"]
        # Set throttle control (velocity control)
        self.data.ctrl[1] = throttle * self.config["action_scale"]

        # Step the simulation
        for _ in range(action_repeat):
            mujoco.mj_step(self.model, self.data)

        # Get observation
        obs = self._get_observation()

        # Calculate speed (magnitude of linear velocity)
        speed = np.linalg.norm(self.data.qvel[:3])

        # Check for collisions
        crashed = False
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1_name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1
            )
            geom2_name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2
            )
            if geom1_name is not None and geom2_name is not None:
                if (
                    "buddy" in geom1_name
                    and ("wall" in geom2_name or "obstacle" in geom2_name)
                    or "buddy" in geom2_name
                    and ("wall" in geom1_name or "obstacle" in geom1_name)
                ):
                    crashed = True

        # Calculate reward
        reward = self.compute_reward(obs["achieved_goal"], obs["desired_goal"])
        if crashed:
            reward += self.config["collision_reward"]

        # Check if successfully parked
        is_success = self._is_success(obs["achieved_goal"], obs["desired_goal"])
        if is_success:
            reward += self.config["success_reward"]

        # Create info dictionary
        info = {
            "speed": float(speed),
            "crashed": crashed,
            "action": np.array(action, dtype=np.float32),
            "is_success": is_success,
        }

        # Check termination conditions
        done = crashed or is_success
        truncated = self.current_step >= self.config["max_steps"]

        # Increment step count
        self.current_step += 1

        return obs, reward, done, truncated, info

    def _is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> bool:
        return np.linalg.norm(achieved_goal[:2] - desired_goal[:2]) < self.config["success_distance_threshold"]


    def render(self):
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.sync()
        time.sleep(0.01)
        return None

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


# Register the environment
gym.register(
    id="CarParkingMujocoPPO-v0",
    entry_point="car_parking_mujoco_ppo:CarParkingEnvPPO",
)
