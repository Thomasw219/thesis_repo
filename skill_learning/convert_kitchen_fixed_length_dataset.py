import numpy as np
import argparse
import torch
import time

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='./data/raw/kitchen-mixed-v0_segmented.npy')
parser.add_argument('--segmented_save_dir', default='./data/segmented/kitchen-mixed-v0-fixed_length.npy')
parser.add_argument('--masked_save_dir', default='./data/segmented/kitchen-mixed-v0-fixed_length.npy')
parser.add_argument('--fixed_segment_length', default=35, type=int)
parser.add_argument('--device', default='cuda:0')

args = parser.parse_args()

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
max_episode_len = args.fixed_segment_length
min_episode_len = args.fixed_segment_length

for episode in segmented_episodes:
    episode_len = len(episode['observations'])
    if episode_len < min_episode_len or episode_len > max_episode_len:
        continue
    episode['observations'] = np.concatenate([episode['observations'], np.zeros((max_episode_len - episode_len, episode['observations'].shape[-1]))], axis=0)
    episode['actions'] = np.concatenate([episode['actions'], np.zeros((max_episode_len - episode_len, episode['actions'].shape[-1]))], axis=0)
    episode['segmentation_probs'] = np.concatenate([episode['segmentation_probs'], np.zeros((max_episode_len - episode_len))], axis=0)
    episode['terminals'] = np.concatenate([episode['terminals'], np.zeros((max_episode_len - episode_len))], axis=0)
    episode['masks'] = np.concatenate([np.ones(episode_len), np.zeros((max_episode_len - episode_len))], axis=0)
    masked_episodes.append(episode)

print(len(segmented_episodes), len(masked_episodes))

np.save(args.masked_save_dir, masked_episodes)


