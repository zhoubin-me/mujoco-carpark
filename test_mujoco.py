import gymnasium as gym
import car_parking_mujoco # Registers the custom environment
import numpy as np
import time

env = gym.make("CarParkingMujoco-v0")
for _ in range(10):
    obs, info = env.reset()
    done = False
    ep_return = 0
    while not done:
        action = np.array([0.5, 0.5])
        obs, reward, done, truncated, info = env.step(action)
        print(reward)
        ep_return += reward
        env.render()
        if done or truncated:
            print(f'Episode return: {ep_return}')
            break
env.close()
