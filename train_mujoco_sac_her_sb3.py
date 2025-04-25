import gymnasium as gym
from stable_baselines3 import SAC, HerReplayBuffer
from stable_baselines3.common.callbacks import CheckpointCallback
import car_parking_mujoco
import time

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--train", action="store_true")
parser.add_argument("--model-path", type=str, default=None)
parser.add_argument("--render", action="store_true")
args = parser.parse_args()

# ==================================
#        Main script
# ==================================

if __name__ == "__main__":

    if args.train:
        env = gym.make('CarParkingMujoco-v0')
        her_kwargs = dict(n_sampled_goal=4, goal_selection_strategy='future')
        model = SAC('MultiInputPolicy', env, replay_buffer_class=HerReplayBuffer,
                    replay_buffer_kwargs=her_kwargs, verbose=1,
                    tensorboard_log="runs",
                    buffer_size=int(1e6),
                    learning_starts=500,
                    learning_rate=1e-3,
                    gamma=0.95, batch_size=1024, tau=0.05,
                    policy_kwargs=dict(net_arch=[256, 256]))

        # Train the agent
        checkpoint_callback = CheckpointCallback(
            save_freq=10000,
            save_path='checkpoints/',
            name_prefix='car_parking_mujoco_sac'
        )
        model.learn(total_timesteps=int(5e5), callback=checkpoint_callback)
        # Save the agent
        model.save("checkpoints/car_parking_mujoco_sac_final")

    else:
        env = gym.make('CarParkingMujoco-v0', repeat_action=1)
        if args.model_path is None:
            raise ValueError("model_path is required for testing")
        model = SAC.load(args.model_path, env)
        success_count = 0
        test_count = 100
        for episode in range(test_count):
            obs, info = env.reset()
            done = truncated = False
            ep_reward = 0
            while not (done or truncated):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
                success_count += info['is_success']
                ep_reward += reward
                if args.render:
                    env.render()
                if done or truncated:
                    success_info = "Success" if info['is_success'] else "Failed !!!!!!!!!"
                    print(f"Episode {episode} finished, ep_reward: {ep_reward}, {success_info}")
                    break
        print(f"Success rate: {success_count / test_count}")
        env.close()
