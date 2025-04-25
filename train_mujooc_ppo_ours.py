import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import gymnasium as gym
from gymnasium.core import ActType, ObsType, WrapperObsType
from gymnasium.wrappers.utils import RunningMeanStd

import numpy as np
from collections import namedtuple
from datetime import datetime
from tqdm import tqdm
import car_parking_mujoco_ppo
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--train", action="store_true")
parser.add_argument("--next-phase", action="store_true")
parser.add_argument("--model-path", type=str, default=None)
parser.add_argument("--render", action="store_true")
args = parser.parse_args()


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_ENVS = 8
SAMPLE_STEPS = 1024
TOTAL_STEPS = 4_000_000 if args.next_phase else 10_000_000
MINI_BATCH_SIZE = 1024
EPOCHES = 8
GAMMA = 0.95
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
ENV_ID = "CarParkingMujocoPPO-v0"


def orthogonal_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class Agent(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=512):
        super(Agent, self).__init__()

        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )

        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

        self.log_std = nn.Parameter(torch.zeros(action_dim))
        self.apply(orthogonal_init)

    def forward(self, x, action=None, test=False):
        feature = self.shared(x)
        mean = self.actor_head(feature)
        std = torch.exp(self.log_std).expand_as(mean)
        dist = torch.distributions.Normal(mean, std)
        entropy = dist.entropy().sum(-1)
        if action is None:
            if test:
                action = mean
            else:
                action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1)
        value = self.critic_head(feature)
        return action, log_prob, entropy, value


class NormalizeObservation(
    gym.ObservationWrapper[WrapperObsType, ActType, ObsType],
    gym.utils.RecordConstructorArgs,
):
    def __init__(self, env: gym.Env[ObsType, ActType], epsilon: float = 1e-8):
        """This wrapper will normalize observations such that each observation is centered with unit variance.

        Args:
            env (Env): The environment to apply the wrapper
            epsilon: A stability parameter that is used when scaling the observations.
        """
        gym.utils.RecordConstructorArgs.__init__(self, epsilon=epsilon)
        gym.ObservationWrapper.__init__(self, env)

        assert env.observation_space.shape is not None
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=env.observation_space.shape,
            dtype=np.float32,
        )

        self.obs_rms = RunningMeanStd(
            shape=self.observation_space.shape, dtype=self.observation_space.dtype
        )
        self.epsilon = epsilon
        self._update_running_mean = True

    @property
    def update_running_mean(self) -> bool:
        """Property to freeze/continue the running mean calculation of the observation statistics."""
        return self._update_running_mean

    @update_running_mean.setter
    def update_running_mean(self, setting: bool):
        """Sets the property to freeze/continue the running mean calculation of the observation statistics."""
        self._update_running_mean = setting

    def observation(self, observation: ObsType) -> WrapperObsType:
        """Normalises the observation using the running mean and variance of the observations."""
        if self._update_running_mean:
            self.obs_rms.update(np.array([observation]))
        return np.float32(
            (observation - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon)
        )
    def get_obs_rms(self):
        rms = self.obs_rms
        return (rms.mean, rms.var, rms.count)

    def set_obs_rms(self, obs_rms):
        self.obs_rms.mean = obs_rms[0]
        self.obs_rms.var = obs_rms[1]
        self.obs_rms.count = obs_rms[2]


def make_env(env):
    env = gym.wrappers.TransformAction(env, lambda action: np.clip(action, -1.5, 1.5), env.action_space)
    env = gym.wrappers.FlattenObservation(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.NormalizeReward(env, gamma=GAMMA)
    env = NormalizeObservation(env)
    return env


Transitions = namedtuple(
    "Transitions", ["states", "actions", "log_probs", "values", "advantages", "returns"]
)


def rollout(env, agent, state_dim, action_dim, obs=None, test=False):
    states = torch.zeros((SAMPLE_STEPS, NUM_ENVS, state_dim), dtype=torch.float32).to(
        DEVICE
    )
    actions = torch.zeros((SAMPLE_STEPS, NUM_ENVS, action_dim), dtype=torch.float32).to(
        DEVICE
    )
    log_probs = torch.zeros((SAMPLE_STEPS, NUM_ENVS), dtype=torch.float32).to(DEVICE)
    values = torch.zeros_like(log_probs)
    rewards = torch.zeros_like(log_probs)
    dones = torch.zeros_like(log_probs)
    advantages = torch.zeros_like(log_probs)

    rs = []
    success = []
    next_obs = None
    for step in range(SAMPLE_STEPS):
        x = torch.from_numpy(obs).float().to(DEVICE)
        action, log_prob, _, value = agent(x, test=test)
        states[step] = x
        actions[step] = action
        log_probs[step] = log_prob
        values[step] = value.squeeze(-1)

        action = action.cpu().numpy()
        next_obs, reward, terminated, truncated, info = env.step(action)

        if "final_info" in info:
            rr = info["final_info"]["episode"]["r"]
            rr_ = info["final_info"]["episode"]["_r"]
            ss = info["final_info"]["is_success"]
            ss_ = info["final_info"]["_is_success"]
            rs += rr[rr_].tolist()
            success += ss[ss_].tolist()

        rewards[step] = torch.from_numpy(reward).float().to(DEVICE)
        dones[step] = (
            torch.from_numpy(np.logical_or(terminated, truncated)).float().to(DEVICE)
        )
        obs = next_obs

    next_o = torch.from_numpy(next_obs).float().to(DEVICE)
    next_value = agent.critic_head(agent.shared(next_o)).squeeze(-1)
    next_advantage = 0
    for step in reversed(range(SAMPLE_STEPS)):
        delta = rewards[step] + GAMMA * (1 - dones[step]) * next_value - values[step]
        advantages[step] = delta + GAMMA * GAE_LAMBDA * next_advantage * (
            1 - dones[step]
        )

        next_value = values[step]
        next_advantage = advantages[step]

    returns = values + advantages
    return (
        rs,
        success,
        next_obs,
        Transitions(
            states.flatten(0, 1),
            actions.flatten(0, 1),
            log_probs.flatten(0, 1),
            values.flatten(0, 1),
            advantages.flatten(0, 1),
            returns.flatten(0, 1),
        ),
    )


def ppo_train(transitions, agent, optimizer, writer, update_step=0, global_step=0):
    N = transitions.states.shape[0]
    for epoch in range(EPOCHES):
        indices = torch.randperm(N)
        for chunk_idx in torch.split(indices, MINI_BATCH_SIZE):
            states = transitions.states[chunk_idx]
            actions = transitions.actions[chunk_idx]
            log_probs = transitions.log_probs[chunk_idx]
            returns = transitions.returns[chunk_idx]
            advantages = transitions.advantages[chunk_idx]
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Update policy and value network
            _, new_log_probs, entropy, new_values = agent(states, actions)
            new_values = new_values.squeeze(-1)
            ratio = torch.exp(new_log_probs - log_probs)
            clipped_ratio = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS)
            policy_loss = -torch.min(
                ratio * advantages, clipped_ratio * advantages
            ).mean()
            entropy_loss = -entropy.mean()
            value_loss = F.mse_loss(new_values, returns)

            total_loss = policy_loss + value_loss * 0.5 + entropy_loss * 0.01

            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
            optimizer.step()
            update_step += 1

    writer.add_scalar("losses/value_loss", value_loss.item(), global_step)
    writer.add_scalar("losses/policy_loss", policy_loss.item(), global_step)
    writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
    writer.add_scalar("losses/adv_mean", transitions.advantages.mean().item(), global_step)
    writer.add_scalar("losses/ret_mean", transitions.returns.mean().item(), global_step)
    writer.add_scalar(
        "charts/learning_rate", optimizer.param_groups[0]["lr"], global_step
    )
    return update_step


def train():
    env = gym.make_vec(
        ENV_ID,
        num_envs=NUM_ENVS,
        wrappers=[make_env],
        vectorization_mode="async",
        vector_kwargs={"autoreset_mode": "SameStep"},
        phase=2 if args.next_phase else 1
    )
    state_dim = env.observation_space.shape[1]
    act_dim = env.action_space.shape[1]
    agent = Agent(state_dim, act_dim).to(DEVICE)
    optimizer = torch.optim.AdamW(agent.parameters(), lr=3e-4, eps=1e-5)

    if args.model_path is not None:
        checkpoint = torch.load(args.model_path, weights_only=False)
        agent.load_state_dict(checkpoint["model"])
        obs_rms = checkpoint["obs_rms"][0]
        env.call('set_obs_rms', obs_rms)


    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=TOTAL_STEPS // (NUM_ENVS * SAMPLE_STEPS)
    )
    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    writer = SummaryWriter(f"runs/ppo_{ENV_ID}_{time_str}")
    update_step = 0
    total_timestep = 0
    obs, _ = env.reset()
    last_mean_reward = float('-inf')
    for _ in tqdm(range(TOTAL_STEPS // (NUM_ENVS * SAMPLE_STEPS) + 1)):
        with torch.no_grad():
            rss, success, obs, transitions = rollout(env, agent, state_dim, act_dim, obs)

        total_timestep += NUM_ENVS * SAMPLE_STEPS
        if len(rss) > 0:
            writer.add_scalar("charts/episodic_return", np.mean(rss), total_timestep)

            if np.mean(rss) > last_mean_reward:
                last_mean_reward = np.mean(rss)
                obs_rms = env.call('get_obs_rms')
                torch.save(
                    {
                        "model": agent.state_dict(),
                        "mean_reward": last_mean_reward,
                        "obs_rms": obs_rms,
                        "optimizer": optimizer.state_dict(),
                    },
                    f"runs/ppo_{ENV_ID}_{time_str}/model_best.pt",
                )
        if len(success) > 0:
            writer.add_scalar("charts/success_rate", np.mean(success), total_timestep)
        update_step = ppo_train(
            transitions, agent, optimizer, writer, update_step=update_step, global_step=total_timestep
        )
        lr_scheduler.step()
    env.close()


def test():
    env = gym.make(
        ENV_ID,
        repeat_action=1,
        phase=2 if args.next_phase else 1
    )
    env = make_env(env)

    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    agent = Agent(state_dim, act_dim).to(DEVICE)

    if args.model_path is None:
        raise ValueError("model_path is required for testing")

    data = torch.load(args.model_path, weights_only=False)
    agent.load_state_dict(data["model"])
    obs_rms = data["obs_rms"]
    for env_, rms in zip([env], obs_rms):
        env_.obs_rms.mean = rms[0]
        env_.obs_rms.var = rms[1]
        env_.obs_rms.count = rms[2]
        break
    print("Loaded weights, best episode reward:", data["mean_reward"])

    success_count = 0
    test_count = 100
    for episdoe in range(test_count):
        done = False
        ep_return = 0
        obs, _ = env.reset()
        while True:
            obs_ = torch.from_numpy(obs).unsqueeze(0).float().to(DEVICE)
            with torch.no_grad():
                action_, _, _, _ = agent(obs_, test=True)
                action_ = action_.squeeze(0).cpu().numpy()
            obs, reward, done, truncated, info = env.step(action_)
            success_count += info['is_success']
            ep_return += reward
            if args.render:
                env.render()
            if done or truncated:
                success_info = "Success" if info['is_success'] else "Failed !!!!!!!!!"
                print(f"Episode {episdoe} finished, ep_reward: {ep_return}, {success_info}")
                break
    print(f"Success rate: {success_count / test_count}")
    env.close()


if __name__ == "__main__":
    if args.train:
        train()
    else:
        test()
