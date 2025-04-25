import gymnasium as gym
from stable_baselines3 import SAC, HerReplayBuffer
from stable_baselines3.common.callbacks import CheckpointCallback
import car_parking_mujoco
import time

# ==================================
#        Main script
# ==================================

if __name__ == "__main__":
    train = True
    if train:
        env = gym.make('CarParkingMujoco-v0')
        model = SAC('MultiInputPolicy', env, verbose=2,
                    tensorboard_log="runs")

        # Train the agent
        model.learn(total_timesteps=int(5e5))
        # Save the agent


    # Test the agent
    if not train:
        env = gym.make('CarParkingMujoco-v0', repeat_action=1)
        model = SAC.load("model_best_sac", env)
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
                # env.render()
                # time.sleep(0.01)
                if done or truncated:
                    success_info = "Success" if info['is_success'] else "Failed !!!!!!!!!"
                    print(f"Episode {episode} finished, ep_reward: {ep_reward}, {success_info}")
                    break
        print(f"Success rate: {success_count / test_count}")
        env.close()
