import numpy as np
import argparse
import torch
import time

from oracle_terminator.utils import prepare_terminator

def segment_episode(terminator, sub_traj_length, episode, device='cuda:0'):
    ep_len = len(episode['observations'])
    termination_probs = np.zeros(ep_len - 1)
    samples = np.zeros(ep_len - 1)
    for i in range(0, ep_len - sub_traj_length + 1, 10):
        obs = torch.tensor(episode['observations'][i:i+sub_traj_length], device=device).unsqueeze(0)
        actions = torch.tensor(episode['actions'][i:i+sub_traj_length], device=device).unsqueeze(0)
        torch_probs = terminator(obs, actions)
        np_probs = torch_probs.detach().cpu().numpy().squeeze()[:-1]
        termination_probs[i:i+sub_traj_length-1] += np_probs
        samples[i:i+sub_traj_length-1] += np.ones_like(np_probs)

    probs = termination_probs / samples
    return np.concatenate([probs, np.array([1.0])], axis=0)

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='./data/raw/maze2d-large-v1_segmented.npy')
parser.add_argument('--segmented_save_dir', default='./data/segmented/maze2d-large-v1_fixed_length_segmented.npy')
parser.add_argument('--masked_save_dir', default='./data/segmented/maze2d-large-v1_fixed_length_masked.npy')
parser.add_argument('--sub_traj_length', default=128, type=int)
parser.add_argument('--fixed_segment_length', default=29, type=int)
parser.add_argument('--device', default='cuda:0')

args = parser.parse_args()

# terminator = prepare_terminator(args.segmentation_model_dir, 0, 'obs_actions', args.device)

numpy_data = np.load(args.data_dir, allow_pickle=True)

segmented_episodes = []
for i, episode in enumerate(numpy_data):
    t = time.time()
    print(f'Segmenting episode {i} of length {len(episode["observations"])}')
    for j in range(0, len(episode['observations']) - args.fixed_segment_length):
        segmented_episodes.append({k : v[j:j+args.fixed_segment_length] for k, v in episode.items()})

    max_episode_len = max([len(episode['observations']) for episode in segmented_episodes])
    min_episode_len = min([len(episode['observations']) for episode in segmented_episodes])
    avg_episode_len = sum([len(episode['observations']) for episode in segmented_episodes]) / len(segmented_episodes)
    print(f'\tSegmented episode in {time.time() - t} seconds')
    print('\tMax episode length: {}'.format(max_episode_len))
    print('\tMin episode length: {}'.format(min_episode_len))
    print('\tAvg episode length: {}'.format(avg_episode_len))

np.save(args.segmented_save_dir, segmented_episodes)

masked_episodes = []
lengths = [len(episode['observations']) for episode in segmented_episodes]
max_episode_len = args.fixed_segment_length
min_episode_len = args.fixed_segment_length

total_timesteps = 0
count = 0

for episode in segmented_episodes:
    episode_len = len(episode['observations'])
    if episode_len < min_episode_len or episode_len > max_episode_len:
        continue
    total_timesteps += episode_len
    count += 1
    episode['observations'] = np.concatenate([episode['observations'], np.zeros((max_episode_len - episode_len, episode['observations'].shape[-1]))], axis=0)
    episode['actions'] = np.concatenate([episode['actions'], np.zeros((max_episode_len - episode_len, episode['actions'].shape[-1]))], axis=0)
    episode['termination_probs'] = np.concatenate([episode['segmentation_probs'], np.zeros((max_episode_len - episode_len))], axis=0)
    episode['terminals'] = np.concatenate([episode['terminals'], np.zeros((max_episode_len - episode_len))], axis=0)
    episode['masks'] = np.concatenate([np.ones(episode_len), np.zeros((max_episode_len - episode_len))], axis=0)
    masked_episodes.append(episode)

print(len(segmented_episodes), len(masked_episodes))

print('Average episode length: {}'.format(total_timesteps / count))

np.save(args.masked_save_dir, masked_episodes)

