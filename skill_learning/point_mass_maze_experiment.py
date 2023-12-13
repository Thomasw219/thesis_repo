import signal
import time

import argparse
import gym
import numpy as np
import torch
import yaml
from torch.optim.lr_scheduler import CyclicLR, ReduceLROnPlateau
from torch.utils.data import TensorDataset
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from networks import ModelFactory
from utils import Logger, get_skill_lens, handle_keyboard_interrupt, partition, Maze2DDataset, SegmentedMaze2DDataset

SCHEDULERS = dict(
    plateau=ReduceLROnPlateau,
    cyclic=CyclicLR,
)

def train(cfgs, device_id=-1):

    data = np.load(cfgs['data_dir'], allow_pickle=True)

    if device_id >= 0:
        cfgs['device_id'] = device_id
    device = torch.device('cuda:{}'.format(cfgs['device_id']) if torch.cuda.is_available() else 'cpu')

    # Load raw numpy data into pytorch dataloaders
    obs = torch.from_numpy(np.stack([ep['observations'] for ep in data], axis=0)).to(device)
    actions = torch.from_numpy(np.stack([ep['actions'] for ep in data], axis=0)).to(device)

    # Compute data statistics
    if cfgs['model']['segmented_trajs_training']:
        masks = torch.from_numpy(np.concatenate([ep['masks'] for ep in data], axis=0)).to(device).bool()
        obs_mean = torch.mean(obs.reshape(-1, obs.shape[-1])[masks], dim=0).float()
        obs_std = torch.std(obs.reshape(-1, obs.shape[-1])[masks], dim=0).float()
        actions_mean = torch.mean(actions.reshape(-1, actions.shape[-1])[masks], dim=0).float()
        actions_std = torch.std(actions.reshape(-1, actions.shape[-1])[masks], dim=0).float()
    else:
        obs_mean = obs.reshape(-1, obs.shape[-1]).mean(dim=0)
        obs_std = obs.reshape(-1, obs.shape[-1]).std(dim=0)
        actions_mean = actions.reshape(-1, actions.shape[-1]).mean(dim=0)
        actions_std = actions.reshape(-1, actions.shape[-1]).std(dim=0)

    # Normalize data
    obs = (obs - obs_mean) / obs_std
    actions = (actions - actions_mean) / actions_std

    obs_dim, action_dim = obs.shape[-1], actions.shape[-1]

    model = ModelFactory.create_model(cfgs['model_type'],
                                      load_path=cfgs['load_path'] if cfgs['load_model'] else None,
                                      device=device,
                                      obs_dim=obs_dim,
                                      action_dim=action_dim,
                                      data_specs={
                                            'obs_mean': obs_mean,
                                            'obs_std': obs_std,
                                            'actions_mean': actions_mean,
                                            'actions_std': actions_std,
                                      },
                                      **cfgs['model'],)

    model.train()

    optimizer = torch.optim.Adam(model.parameters(), **cfgs['optimizer'])
    scheduler = SCHEDULERS[cfgs["scheduler_class"]](optimizer, **cfgs[cfgs['scheduler_class']])

    logger = Logger(hps=cfgs, model=model, online=cfgs.get('log_online', True))

    if cfgs['model']['segmented_trajs_training']:
        train_dataset = SegmentedMaze2DDataset(data=data, train=True, train_ratio=cfgs['train_dataset_fraction'])
        test_dataset = SegmentedMaze2DDataset(data=data, train=False, train_ratio=cfgs['train_dataset_fraction'])
    else:
        train_dataset = Maze2DDataset(data=data, signal_length=cfgs['data_sub_traj_len'], train=True, train_ratio=cfgs['train_dataset_fraction'])
        test_dataset = Maze2DDataset(data=data, signal_length=cfgs['data_sub_traj_len'], train=False, train_ratio=cfgs['train_dataset_fraction'])
    train_loader = DataLoader(train_dataset, **cfgs['data_loader'])
    test_loader = DataLoader(test_dataset, **cfgs['data_loader'])

    if not cfgs['model']['segmented_trajs_training']:
        raise NotImplementedError()

    loss = 1e10
    reconstruction_mse = 1e10
    logger.log_model('model', model)

    for epoch in tqdm(range(cfgs['n_epochs']), desc='Epoch', dynamic_ncols=True):

        model.train()
        total_train_loss = 0

        for step, (unnormalized_obs_batch, unnormalized_actions_batch, masks_batch) in (pbar:= tqdm(enumerate(train_loader), leave=False, dynamic_ncols=True)):
            unnormalized_obs_batch = unnormalized_obs_batch.to(device).float()
            unnormalized_actions_batch = unnormalized_actions_batch.to(device).float()
            masks_batch = masks_batch.to(device).bool()

            start_time = time.time()
            if cfgs['model']['normalize_inputs']:
                obs_batch = (unnormalized_obs_batch - obs_mean) / obs_std
                actions_batch = (unnormalized_actions_batch - actions_mean) / actions_std
            else:
                obs_batch = unnormalized_obs_batch
                actions_batch = unnormalized_actions_batch
            loss, metrics, info = model.get_loss(obs_batch, actions_batch, _masks=masks_batch)
            total_train_loss += loss.item()
            reconstruction_mse = metrics['reconstruction_mse_loss']

            optimizer.zero_grad()
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfgs['grad_clip'])

            pbar.set_description(f"Train loss: {loss.item():.2f}, grad: {norm:.2f}")

            logger.log(metrics, mode='train')

            optimizer.step()
            if cfgs["scheduler_class"] == 'plateau':
                scheduler.step(loss)
            else:
                scheduler.step()

            logger.log({
                'lr': scheduler._last_lr[0],
                'epoch': epoch,
                'train/step_time': time.time() - start_time,
                'grad_norm': norm,
            })

            if (logger._global_step % cfgs['model_save_freq']) == 0:
                logger.log_model('model', model)

            logger.step()

        total_test_loss, test_metrics = 0, {}

        model.eval()

        if cfgs['train_dataset_fraction'] < 1.0:
            if (epoch % cfgs['eval_every']) == 0:
                for step, (unnormalized_obs_batch, unnormalized_actions_batch, masks_batch) in (pbar:= tqdm(enumerate(test_loader), leave=False, dynamic_ncols=True)):
                    unnormalized_obs_batch = unnormalized_obs_batch.to(device).float()
                    unnormalized_actions_batch = unnormalized_actions_batch.to(device).float()
                    masks_batch = masks_batch.to(device).bool()
                    obs_batch = (unnormalized_obs_batch - obs_mean) / obs_std
                    actions_batch = (unnormalized_actions_batch - actions_mean) / actions_std

                    start_time = time.time()

                    with torch.no_grad():
                        latest_loss, latest_metrics, info = model.get_loss(obs_batch, actions_batch, _masks=masks_batch)

                    total_test_loss += latest_loss.item()
                    pbar.set_description(f"Test loss: {latest_loss.item():.2f}, grad: {norm:.2f}")

                    for k, v in latest_metrics.items():
                        if not k in test_metrics:
                            test_metrics[k] = v
                        else:
                            test_metrics[k] = test_metrics[k] + v

                # Average test metrics
                total_test_loss /= len(test_loader)
                for k, v in test_metrics.items():
                    test_metrics[k] = v / len(test_loader)
                logger.log(test_metrics, mode='test')
                logger.log({'test/step_time': (time.time() - start_time) / len(test_loader)})

                if logger.is_better_loss(total_test_loss):
                    logger.log_model('best_model', model, overwrite=True)
                logger.log_model('last_model', model, overwrite=True)
        else:
            if (epoch % cfgs['eval_every']) == 0:
                train_loss = total_train_loss / len(train_loader)
                if logger.is_better_loss(train_loss):
                    logger.log_model('best_model', model, overwrite=True)
                logger.log_model('last_model', model, overwrite=True)

        if logger._global_step > cfgs['max_iters']:
            print('Reached max iterations')
            break

if __name__ == '__main__':

    signal.signal(signal.SIGINT, handle_keyboard_interrupt)

    parser = argparse.ArgumentParser()
    parser.add_argument('--device_id', default=-1, type=int)
    args = parser.parse_args()

    with open('./cfgs/supervised_vlsm.yaml', 'r') as f:
        cfgs = yaml.load(f, Loader=yaml.FullLoader)

    train(cfgs, args.device_id)
