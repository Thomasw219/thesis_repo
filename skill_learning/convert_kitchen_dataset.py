import numpy as np
import argparse
import torch
import time
import os

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='./data/raw/kitchen-mixed-v0_segmented.npy')
parser.add_argument('--segmented_save_dir', default='./data/segmented/kitchen-mixed-v0_segmented.npy')
parser.add_argument('--masked_save_dir', default='./data/segmented/kitchen-mixed-v0_segmented.npy')
parser.add_argument('--sub_traj_length', default=128, type=int)
parser.add_argument('--device', default='cuda:0')

args = parser.parse_args()

numpy_data = np.load(args.data_dir, allow_pickle=True)

segmented_episodes = []
episode_lens = []
for i, episode in enumerate(numpy_data):
    t = time.time()
    print(f'Segmenting episode {i} of length {len(episode["observations"])}')

    episode_points = [0]
    episode_points.extend((np.arange(len(episode['observations']))[episode['segmentation_probs'] > 0.5] + 1).tolist())

    segments = []
    for episode_start, episode_end in zip(episode_points[:-1], episode_points[1:]):
        episode_len = episode_end - episode_start
        episode_lens.append(episode_len)
        for i in range(int(np.ceil(episode_len / 4))):
            print(episode_start + i, episode_end)
            segments.append({k : v[episode_start + i:episode_end] for k, v in episode.items()})

    segmented_episodes.extend(segments)
    # Remove last segment
    # del segmented_episodes[-1] # This dataset was generated in a way such that it doesn't actually end at a terminal state, it just ends in the middle of a "going to goal" segment

    max_episode_len = max([len(episode['observations']) for episode in segmented_episodes])
    min_episode_len = min([len(episode['observations']) for episode in segmented_episodes])
    avg_episode_len = sum([len(episode['observations']) for episode in segmented_episodes]) / len(segmented_episodes)
    print(f'\tSegmented episode in {time.time() - t} seconds')
    print('\tMax episode length: {}'.format(max_episode_len))
    print('\tMin episode length: {}'.format(min_episode_len))
    print('\tAvg episode length: {}'.format(avg_episode_len))

os.makedirs(os.path.dirname(args.segmented_save_dir), exist_ok=True)
np.save(args.segmented_save_dir, segmented_episodes)

masked_episodes = []
max_episode_len = 75
min_episode_len = 5

count = 0
total_timesteps = 0

for episode in segmented_episodes:
    episode_len = len(episode['observations'])
    if episode_len < min_episode_len or episode_len > max_episode_len:
        continue
    total_timesteps += episode_len
    count += 1
    episode['observations'] = np.concatenate([episode['observations'], np.zeros((max_episode_len - episode_len, episode['observations'].shape[-1]))], axis=0)
    episode['actions'] = np.concatenate([episode['actions'], np.zeros((max_episode_len - episode_len, episode['actions'].shape[-1]))], axis=0)
    episode['segmentation_probs'] = np.concatenate([episode['segmentation_probs'], np.zeros((max_episode_len - episode_len))], axis=0)
    episode['terminals'] = np.concatenate([episode['terminals'], np.zeros((max_episode_len - episode_len))], axis=0)
    episode['masks'] = np.concatenate([np.ones(episode_len), np.zeros((max_episode_len - episode_len))], axis=0)
    masked_episodes.append(episode)

print(len(segmented_episodes), len(masked_episodes))

filtered_episode_lens = [episode_len for episode_len in episode_lens if episode_len >= min_episode_len and episode_len <= max_episode_len]

print(f'Average segment length: {np.mean(filtered_episode_lens)}')

np.save(args.masked_save_dir, masked_episodes)

