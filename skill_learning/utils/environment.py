import gym
import numpy as np
import torch
from .misc import beauty
import d4rl

class EnvironmentWrapper:
    # TODO: Write getattr to access other env attributes
    def __init__(self, env_name):
        self._env = gym.make(env_name)
        self._env_name = env_name
        self._time_step = 0
        # self._env.seed(seed)
        # self._env.action_space.seed(seed)
        # self._env.observation_space.seed(seed)

    def step(self, action):
        if isinstance(action, torch.Tensor):
            action = beauty(action)
        success = ((self.obs[..., :2] - self.goal)**2).sum(-1)**0.5 < 0.5 # 0.5 threshold for maze environments
        self._time_step += 1
        timeout = self._time_step >= self._env.env.spec.max_episode_steps
        return *self._env.step(action), success, timeout

    def reset(self, type='random', resample_dataset=False):
        self._time_step = 0
        return self._env.reset()

    def set_state(self, obs):
        qpos, qvel = obs.chunk(2, dim=-1)
        qpos, qvel = beauty(qpos), beauty(qvel)
        self._env.env.set_state(qpos, qvel)

    def render(self, render='display'):
        if render.lower() == 'display':
            self._env.render()
        elif render.lower() == 'video':
            return self._env.render(mode='rgb_array')

    def run_traj(self, traj_obs, traj_actions=None, render=None):
        frames = []
        obs_list= []

        self.set_state(traj_obs[0])

        if traj_actions is not None:
            for action in traj_actions:
                obs_list.append(self.obs)
                self.step(action)
                if render is not None:
                    frame = self.render(render)
                    if render.lower() == 'video': frames.append(frame)
        else:
            for obs in traj_obs:
                obs_list.append(self.obs)
                self.set_state(obs)
                if render is not None:
                    frame = self.render(render)
                    if render.lower() == 'video': frames.append(frame)

        return frames, obs_list

    def set_target(self, target=None):
        if target is None:
            self._env.set_target()
            self._target = self._env.env._target
            # self._target = self._env.target_goal
        else:
            self._target = target
            self._env.set_target(target)

    @property
    def obs(self):
        return self._env.env._get_obs()

    @property
    def goal(self):
        # return self._env.target_goal
        return np.array(self._env.env._target) if isinstance(self._env.env._target, tuple) else self._env.env._target
