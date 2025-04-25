import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

import car_parking_mujoco_ppo


# ==================================
#        Main script
# ==================================

if __name__ == "__main__":
    train = True
    if train:
        n_cpu = 8
        env = make_vec_env("CarParkingMujocoPPO-v0", n_envs=n_cpu, vec_env_cls=SubprocVecEnv)
        model = PPO(
            "MultiInputPolicy",
            env,
            policy_kwargs=dict(net_arch=[dict(pi=[512, 512], vf=[512, 512])]),
            n_steps=1024,
            batch_size=1024,
            n_epochs=8,
            ent_coef=0.01,
            learning_rate=5e-4,
            gamma=0.95,
            verbose=2,
            tensorboard_log="runs/",
        )
        # Train the agent
        checkpoint_callback = CheckpointCallback(
            save_freq=100000,
            save_path='checkpoints/',
            name_prefix='car_parking_mujoco_ppo'
        )
        model.learn(total_timesteps=int(8e6), callback=checkpoint_callback)
        # Save the agent
        model.save("checkpoints/car_parking_mujoco_ppo_final")
    else:
        model = PPO.load("checkpoints/car_parking_mujoco_ppo_final")
        env = gym.make("CarParkingMujocoPPO-v0")
        for _ in range(5):
            obs, info = env.reset()
            done = truncated = False
            while not (done or truncated):
                action, _ = model.predict(obs)
                obs, reward, done, truncated, info = env.step(action)
                env.render()