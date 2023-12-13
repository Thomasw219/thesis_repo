from abc import ABC, abstractmethod

import numpy as np
import torch
from torchvision import datasets, transforms
import gym
import d4rl

class GeneratedD4RLDataset:
    def __init__(
            self,
            signal_length=128,
            data_path='data/raw/maze2d-medium-v1.npy',
    ):
        self.signal_length = signal_length
        self.episodes = np.load(data_path, allow_pickle=True)
        self.n_episodes = len(self.episodes)
        self.obs_dim = self.episodes[0]['observations'].shape[-1]
        self.action_dim = self.episodes[0]['actions'].shape[-1]

    def __getitem__(self, index):
        ep = self.episodes[index]
        start_index = np.random.randint(0, ep['observations'].shape[0] - self.signal_length)
        return ep['observations'][start_index:start_index+self.signal_length], ep['actions'][start_index:start_index+self.signal_length]

    def __len__(self):
        return self.n_episodes

    def get_episode(self, index):
        ep = self.episodes[index]
        return ep

class D4RLDataset:
    def __init__(
            self,
            signal_length=128,
            dataset_name='antmaze-large-diverse-v0'
    ):
        self.signal_length = signal_length
        env = gym.make(dataset_name)
        self.dataset_name = dataset_name
        dataset = env.get_dataset()
        if dataset_name == 'antmaze-large-diverse-v0':
            episode_points = [0]
            episode_points.extend([1001 + i for i in range(0, 1000000 - 1000, 1001)])
        elif dataset_name == 'kitchen-mixed-v0' or dataset_name == 'kitchen-partial-v0':
            episode_points = [0]
            episode_points.extend((np.arange(136950)[dataset['terminals']] + 1).tolist())
        elif dataset_name == 'kitchen-complete-v0':
            episode_points = [0]
            episode_points.extend((np.arange(3680)[dataset['terminals']] + 1).tolist())
        else:
            raise NotImplementedError()
        self.episodes = [{k : v[episode_start:episode_end] for k, v in dataset.items()} for episode_start, episode_end in zip(episode_points[:-1], episode_points[1:])]

        min_len = np.inf
        for episode in self.episodes:
            terminations = episode['terminals']
            length = terminations.shape[0]
            min_len = np.minimum(min_len, length)
            assert length > self.signal_length
        print("Min length: ", min_len)
        print("Num episodes: ", len(self.episodes))

        self.obs_dim = self.episodes[0]['observations'].shape[-1]
        self.action_dim = self.episodes[0]['actions'].shape[-1]

    def get_episode(self, index):
        return self.episodes[index]

    def __getitem__(self, index):
        if self.dataset_name == 'kitchen-complete-v0':
            index = np.random.randint(0, len(self.episodes))
        ep = self.episodes[index]
        start_index = np.random.randint(0, ep['observations'].shape[0] - self.signal_length)
        return ep['observations'][start_index:start_index+self.signal_length], ep['actions'][start_index:start_index+self.signal_length]

    def __len__(self):
        if self.dataset_name == 'kitchen-complete-v0':
            return 640
        return len(self.episodes)

def test_maze2d_dataset():
    dataset = GeneratedD4RLDataset()

    from torch.utils.data import DataLoader
    from time import time
    np.random.seed(0)
    t = time()
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)
    obs = next(iter(dataloader))
    print(time() - t)
    print(obs)

def test_d4rl_dataset():
    dataset = D4RLDataset(dataset_name='kitchen-mixed-v0')

    from torch.utils.data import DataLoader
    from time import time
    np.random.seed(0)
    t = time()
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)
    obs, act = next(iter(dataloader))
    print(time() - t)
    print(obs.shape, act.shape)
    print(obs[0, :, 30:])
