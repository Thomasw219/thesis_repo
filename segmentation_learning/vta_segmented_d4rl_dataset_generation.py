import gym
import d4rl
import numpy as np
import matplotlib.pyplot as plt
import torch

import time

from data import D4RLDataset, GeneratedD4RLDataset

env_name = 'maze2d-medium-v1'
device = torch.device('cuda:1')

# CHECK THIS IS RIGHT
model_dir = 'maze2d_08-17-2023_21-21-22'
full_model_path = f"logs/vta_maze_eval/{model_dir}/full_model.pt"

model_input_len = 130

def segment_episode(model, episode, device=torch.device('cpu'), skip=10):
    ep_len = len(episode['observations'])
    termination_probs = np.zeros(ep_len)
    samples = np.zeros(ep_len)
    samples[0] = 1.0
    samples[-1] = 1.0
    for i in range(0, ep_len - model_input_len + 1, skip):
        obs = torch.tensor(episode['observations'][i:i+model_input_len], device=device).unsqueeze(0)
        actions = torch.tensor(episode['actions'][i:i+model_input_len], device=device).unsqueeze(0)
        __, _, info = model.get_loss(obs, actions)
        np_probs = info['segmentation_post_probs'].detach().cpu().numpy().squeeze()
        termination_probs[i+2:i+model_input_len] += np_probs
        samples[i+2:i+model_input_len] += np.ones_like(np_probs)

    probs = termination_probs / samples
    probs[-1] = 1
    return probs

model = torch.load(full_model_path, map_location=device)

np.random.seed(1)

if 'maze2d' in env_name:
    dataset = GeneratedD4RLDataset(data_path=f'data/raw/{env_name}.npy')
else:
    dataset = D4RLDataset(dataset_name=env_name)

episodes_with_segmentation = []
total_timesteps = 0
for i in range(len(dataset)):
    t = time.time()
    episode = dataset.get_episode(i)

    with torch.no_grad():
        probs = segment_episode(model, episode, device=device)

    # plt.plot(probs)
    # plt.savefig(f'figures/{env_name}_{model_dir}_segmentation_probs_{i}.png')
    # plt.clf()

    episode['segmentation_probs'] = probs

    episodes_with_segmentation.append(episode)
    total_timesteps += len(episode['observations'])
    print(f"Episode {i} took {time.time() - t} seconds, processed {len(episode['observations'])} timesteps, total timesteps {total_timesteps}")

np.save(f"./data/raw/{env_name}_segmented_{model_dir}.npy", episodes_with_segmentation, allow_pickle=True)

print("Done")
