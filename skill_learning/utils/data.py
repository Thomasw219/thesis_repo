import os

import d4rl
import gym
import numpy as np
import torch
import tqdm
from .misc import register_data_generator

class Maze2DDataset:
    def __init__(
            self,
            signal_length=128,
            data=None,
            train=True,
            train_ratio=0.9,
    ):
        self.signal_length = signal_length
        self.episodes = data
        self.n_episodes = len(self.episodes)
        if train:
            self.episodes = self.episodes[:int(train_ratio * self.n_episodes)]
        else:
            self.episodes = self.episodes[int(train_ratio * self.n_episodes):]
        self.n_episodes = len(self.episodes)

    def __getitem__(self, index):
        ep = self.episodes[index]
        start_index = np.random.randint(0, ep['observations'].shape[0] - self.signal_length)
        return ep['observations'][start_index:start_index+self.signal_length], ep['actions'][start_index:start_index+self.signal_length], ep['terminals'][start_index:start_index+self.signal_length]

    def __len__(self):
        return self.n_episodes

class KitchenDataset:
    def __init__(
            self,
            signal_length=128,
            data_dir='./data/raw/kitchen-mixed-v0_segmented.npy',
            train=True,
            train_ratio=0.9,
            remove_goal=False,
            seed=0,
    ):
        data = np.load(data_dir, allow_pickle=True)
        self.signal_length = signal_length
        self.episodes = data
        self.n_episodes = len(self.episodes)
        np.random.seed(seed)
        train_episode_indices = np.random.choice(self.n_episodes, int(train_ratio * self.n_episodes), replace=False)
        test_episode_indices = np.array([i for i in range(self.n_episodes) if i not in train_episode_indices])
        if train is None:
            pass
        elif train:
            self.episodes = self.episodes[train_episode_indices]
        else:
            self.episodes = self.episodes[test_episode_indices]
        self.n_episodes = len(self.episodes)
        self.remove_goal = remove_goal
        self.goal_dim = 30
        np.random.seed()

    def __getitem__(self, index, start_index=None):
        ep = self.episodes[index]
        if start_index is None:
            start_index = np.random.randint(-64, ep['observations'].shape[0] - self.signal_length + 64)
            start_index = np.clip(start_index, 0, ep['observations'].shape[0] - self.signal_length)
        if self.remove_goal:
            observations = ep['observations'][start_index:start_index+self.signal_length][..., :-self.goal_dim]
        else:
            observations = ep['observations'][start_index:start_index+self.signal_length]
        seg_probs = ep['segmentation_probs'][start_index:start_index+self.signal_length]
        if start_index == 0:
            seg_probs[0] = 1.0
        return observations, ep['actions'][start_index:start_index+self.signal_length], seg_probs, ep['terminals'][start_index+self.signal_length-1]

    def __len__(self):
        return self.n_episodes

    def all_obs(self):
        if self.remove_goal:
            return np.concatenate([ep['observations'] for ep in self.episodes], axis=0)[..., :-self.goal_dim]
        else:
            return np.concatenate([ep['observations'] for ep in self.episodes], axis=0)

    def all_actions(self):
        return np.concatenate([ep['actions'] for ep in self.episodes], axis=0)

class SegmentedKitchenDataset:
    def __init__(
            self,
            # data_dir='./data/segmented/kitchen-mixed-v0-terminated_masked_0.5.npy',
            data_dir='./data/segmented/kitchen-mixed-v0-terminated_masked.npy',
            train=True,
            train_ratio=0.9,
            remove_goal=False,
    ):
        data = np.load(data_dir, allow_pickle=True)
        self.episodes = data
        self.n_episodes = len(self.episodes)
        train_episode_indices = np.random.choice(self.n_episodes, int(train_ratio * self.n_episodes), replace=False)
        test_episode_indices = np.array([i for i in range(self.n_episodes) if i not in train_episode_indices])
        if train is None:
            pass
        elif train:
            self.episodes = self.episodes[train_episode_indices]
        else:
            if len(test_episode_indices) == 0:
                self.episodes = []
            else:
                self.episodes = self.episodes[test_episode_indices]
        self.n_episodes = len(self.episodes)
        self.remove_goal = remove_goal
        self.goal_dim = 30

    def __getitem__(self, index, start_index=None):
        ep = self.episodes[index]
        if self.remove_goal:
            observations = ep['observations'][..., :-self.goal_dim]
        else:
            observations = ep['observations']
        return observations, ep['actions'], ep['masks'], ep['terminals']

    def __len__(self):
        return self.n_episodes

class SegmentedMaze2DDataset:
    def __init__(
            self,
            data=None,
            train=True,
            train_ratio=0.9,
    ):
        self.episodes = data
        self.n_episodes = len(self.episodes)
        if train is None:
            pass
        if train:
            self.episodes = self.episodes[:int(train_ratio * self.n_episodes)]
        else:
            self.episodes = self.episodes[int(train_ratio * self.n_episodes):]
        self.n_episodes = len(self.episodes)

    def __getitem__(self, index, start_index=None):
        ep = self.episodes[index]
        return ep['observations'], ep['actions'], ep['masks']

    def __len__(self):
        return self.n_episodes

class OrnsteinUhlenbeckProcess:
    def __init__(self, dim, theta, sigma, dt, mu=0):
        self.dim = dim
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self.mu = mu
        self.reset()

    def step(self):
        self.x_t = (
            self.x_t
            + self.theta * (self.mu - self.x_t) * self.dt
            + self.sigma * np.sqrt(self.dt) * np.random.randn(self.dim)
        )
        return self.x_t

    def reset(self, x_t=None):
        self.x_t = x_t if x_t is not None else np.random.randn(self.dim) * self.sigma + self.mu


# Custom dataset generators
@register_data_generator('sine')
def sine_data(batch_size=4, n_time_steps=128, device='cpu'):
    data = []
    t = np.linspace(0, 2 * np.pi, n_time_steps)
    for i in range(batch_size):
        f = np.random.choice([1, 2, 3])
        a = np.random.uniform(0.5, 1.5)
        y = a * np.sin(f * t)
        data.append(y)
    return torch.from_numpy(np.stack(data)).unsqueeze(dim=-1).to(torch.float).to(device)

@register_data_generator('compressible_sine')
def compressible_sine(batch_size=1, n_samples=128, t_end=20, device='cpu'):
    def sample_signal(f_choice, samples_per_t):
        f = np.random.choice(f_choice)
        p = 1 / f
        n_p = np.random.choice([1, 2, 3])
        n_samples = int(samples_per_t * p)  * n_p
        t = np.linspace(0, p*n_p, n_samples)
        if np.random.rand(1) > 0.5:
            s = np.zeros(n_samples)
        else:
            s = np.sin(t * 2 * np.pi * f)
        return s, n_samples

    samples_per_t = n_samples / t_end

    data = []

    for _ in range(batch_size):
        f = np.array([0.1, 0.5, 0.25])
        t = np.linspace(0, t_end, n_samples)
        dt = t_end / n_samples

        s = []
        end = 0
        while(end < n_samples):
            _s, n_s = sample_signal(f, samples_per_t)
            n_s_end = min(n_s, n_samples - end)
            s.extend(list(_s[:n_s_end]))
            end += n_s_end

        data.append(s)
    data = torch.tensor(data, dtype=torch.float, device=device).unsqueeze(dim=-1)
    return data

@register_data_generator('noisy')
def noisy(batch_size=1, n_samples=100, device='cpu'):
    t = torch.arange(n_samples, dtype=torch.float, device=device).unsqueeze(0).expand(batch_size, -1)
    std = torch.square(t - n_samples / 2) / (n_samples / 2)**2 / 2
    data = []
    for _ in range(batch_size):
        signal = np.empty((n_samples,))
        proc = OrnsteinUhlenbeckProcess(1, 0.03, 0.1, 3)
        proc.reset(x_t=0)
        for i in range(n_samples):
            signal[i] = proc.step()
        data.append(signal)
    return torch.clip((torch.tensor(data, device=device, dtype=torch.float) * std).unsqueeze(-1), -1, 1)

@register_data_generator('semi_compressible')
def semi_compressible(batch_size=1, n_samples=128, device='cpu'):
    data = []
    for _ in range(batch_size):
        signal = np.empty(n_samples)
        t = np.arange(64)
        signal[:64] = 0#np.sin(t / 64 * np.pi * 2)
        proc = OrnsteinUhlenbeckProcess(1, 0.03, 0.1, 3)
        proc.reset(x_t=0)
        for i in range(64):
            signal[64 + i] = proc.step()
        data.append(signal)

    return torch.tensor(data, dtype=torch.float, device=device).unsqueeze(dim=-1)

@register_data_generator('binary')
def binary_data(batch_size=1, n_samples=10, device='cpu'):
    return torch.randn(batch_size, n_samples, 1, device=device).sign().to(torch.float)

def sample_signal(f, samples_per_t):
    a = np.random.uniform(0.25, 1)
    p = 1 / f
    n_p = np.random.choice([0.5, 1, 1.5, 2, 2.5, 3])
    n_samples = int(samples_per_t * p  * n_p)
    t = np.linspace(0, p*n_p, n_samples)
    if np.random.rand(1) > 0.3:
        s = np.zeros(n_samples)
    else:
        s = a * np.sin(t * 2 * np.pi * f)
    return s, n_samples

# @register_data_generator('')
def generate_signal(n_samples):
    samples_per_t = 10

    s = []
    end = 0
    while(end < n_samples):
        f = np.random.uniform(.25, .5)
        _s, n_s = sample_signal(f, samples_per_t)
        n_s_end = min(n_s, n_samples - end)
        s.extend(list(_s[:n_s_end]))
        end += n_s_end

    return np.array(s)

@register_data_generator('pseudo_seismic')
def generate_pseudo_seismic_data(batch_size, steps=200):
    data = [generate_signal(steps) for _ in range(batch_size)]
    return np.stack(data, axis=0).reshape(batch_size, steps, 1)

# @register_data_generator('sinusoidal')
def generate_sinusoidal_data(batch_size, steps=200):
    t = np.linspace(0, 2 * np.pi * 10, steps)
    xs = []
    for _ in range(batch_size):
        x = np.random.uniform(0.5, 1) * np.sin((np.random.uniform(1, 3)) * t / 10 + np.random.uniform(0, 2 * np.pi))
        xs.append(x)
    return np.stack(xs, axis=0).reshape(batch_size, steps, 1)


def partition(dataset, split_size):
    op_dataset = dict()
    for k, v in dataset.items():
        op_v = list(map(lambda x: x.numpy(), torch.split(torch.from_numpy(v), split_size)))
        op_dataset[k] = np.stack(op_v[:-1], axis=0) # Drop the last one as it might be smaller than the split size
    return op_dataset


## D4RL dataset utils
def chunk_dataset(env_name, H=128, make_numpy=True):
    env = gym.make(env_name)

    dataset = env.get_dataset(f'data/raw/{env_name}.hdf5')
    episodic_data = d4rl.sequence_dataset(env, dataset=dataset)
    chunk_data = {k: [] for k in dataset.keys()}

    chunk_size_filter = lambda x, s=H: x.shape[0] == s

    # Partition data in chunks of equal length
    for idx, trajectory in tqdm.tqdm(enumerate(episodic_data), desc='Chunking data'):
        l = len(trajectory['terminals'])
        split_idxs = list(range(0, l, H))

        for k, v in trajectory.items():
            assert l == v.shape[0] # sanity check
            chunks = np.split(v, split_idxs)
            chunk_data[k].extend(filter(chunk_size_filter, chunks))

    if make_numpy:
        base_shape = (len(chunk_data['terminals']), H)

        for k in tqdm.tqdm(chunk_data.keys(), desc='Making numpy'):
            chunk_data[k] = updated_v = np.stack(chunk_data[k], axis=0)
            v_shape = updated_v.shape[:2]
            assert v_shape == base_shape, f'Mismatch in {k}. Expected {base_shape}, received {v_shape}'
        # Print data specs
        print(f'Env: {env_name}')
        for k, v in chunk_data.items():
            print(f'{k}: {v.shape}')
    else:
        print(f'Warning: Sanity checks not run.')

    return chunk_data

def split_dataset(env_name):
    """
    Creates a dictionary of lists of arrays. Version 2 clips all lists to the same length. Removes shorter sequences.
    """
    env = gym.make(env_name)

    dataset = env.get_dataset()
    dataset_ = d4rl.qlearning_dataset(env)
    split_data = {}

    diff = abs(dataset_['next_observations'][:, :2] - dataset_['observations'][:, :2]).sum(axis=-1)
    rst_idxs = np.where(diff > 1.)[0]

    new_dones = np.zeros_like(dataset['terminals'])
    new_dones[rst_idxs] = True

    split_idxs = np.where(new_dones == True)[0]

    for key, value in dataset.items():
        # OG data has incorrect timeouts; insert correct ones
        if key == 'timeouts':
            split_data[key] = np.split(new_dones, split_idxs + 1) # adding 1 to split_idxs since .split() doesn't include the last index
        else:
            split_data[key] = np.split(value, split_idxs + 1) # adding 1 to split_idxs since .split() doesn't include the last index

    # Sanity checks
    # last timeout is always true; false otherwise
    for idx, t in enumerate(split_data['timeouts']):
        # last trajectory may not have a timeout # TODO: double check
        if idx == len(split_data['timeouts']) - 1:
            continue
        assert t[-1] == True, f'idx: {idx}'
        assert (t[:-1] == False).all(), f'idx: {idx}'

    # OG and new data should have the same dimensions
    tot_len = lambda data_lis: sum([len(data) for data in data_lis])
    for key, value in dataset.items():
        assert len(value) == tot_len(split_data[key])

    # Clip all lists to the same length
    for key, value in split_data.items():
        l = []
        for v in split_data[key]:
            if len(v) < 1000:
                continue
            l.append(v[:1000])
        split_data[key] = l

    for idx, t in enumerate(split_data['timeouts']):
        # last trajectory may not have a timeout # TODO: double check
        if idx == len(split_data['timeouts']) - 1:
            continue
        t[-1] = True

    # Sanity checks
    # last timeout is always true; false otherwise
    for idx, t in enumerate(split_data['timeouts']):
        # last trajectory may not have a timeout # TODO: double check
        if idx == len(split_data['timeouts']) - 1:
            continue
        assert t[-1] == True, f'idx: {idx}'
        assert (t[:-1] == False).all(), f'idx: {idx}'

    return split_data