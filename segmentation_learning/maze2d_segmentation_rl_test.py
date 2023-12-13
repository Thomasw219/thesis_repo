import os
from datetime import datetime, timezone, timedelta
from time import time

import hydra
from omegaconf import DictConfig
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data import GeneratedD4RLDataset as Dataset
from models import RLSegmentationModel
from utils import make_scheduler

@hydra.main(version_base='1.3', config_path='cfgs', config_name='maze_2d_experiment')
def test_full_prototype(cfg):
    np.random.seed(cfg['np_seed'])
    train_dataset = Dataset(**cfg['train_dataset'])
    train_dataloader = DataLoader(train_dataset, **cfg['dataloader'])

    test_dataset = Dataset(**cfg['test_dataset'])
    test_dataloader = DataLoader(test_dataset, **cfg['dataloader'])

    model = RLSegmentationModel(cfg['model'], obs_dim=train_dataset.obs_dim, action_dim=train_dataset.action_dim, max_seq_len=cfg['train_dataset']['signal_length'])
    model.to(cfg['device'])

    if cfg['model_load_path'] is not None:
        print("MODEL LOADED")
        model.load_state_dict(torch.load(cfg['model_load_path'], map_location=cfg['device']))

    optimizer = get_optimizer(cfg['optimizer'], model)
    temp_scheduler = make_scheduler(cfg['temp_scheduler'])
    # time_loss_weight_scheduler = make_scheduler(cfg['time_loss_weight_scheduler'])
    # state_kl_weight_scheduler = make_scheduler(cfg['state_kl_weight_scheduler'])
    # lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: temp_scheduler.get_value(step))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1)
    timestring = datetime.now(tz=timezone(timedelta(hours=-5))).strftime("_%m-%d-%Y_%H-%M-%S") # EST, No daylight savings
    logger = SummaryWriter(os.path.join(cfg['log_dir'], cfg['name'] + timestring))
    logger.add_text('config', str(cfg))
    logger.add_text('model', str(model))

    epoch_steps = len(train_dataloader)
    global_step = 0
    best_test_loss = np.inf
    for epoch in tqdm(range(cfg['epochs']), desc='Epoch', total=cfg['epochs'], position=0):
        temp = temp_scheduler.get_value(global_step)
        model.set_temperature(temp)
        logger.add_scalar('train/temp', temp, global_step)
        # time_loss_weight = time_loss_weight_scheduler.get_value(epoch)
        # model.set_time_loss_weight(time_loss_weight)
        logger.add_scalar('train/time_loss_weight', model.time_loss_weight, global_step)
        # state_kl_weight = state_kl_weight_scheduler.get_value(epoch)
        # model.set_state_kl_weight(state_kl_weight)
        # logger.add_scalar('train/state_kl_weight', state_kl_weight, global_step)
        model.train()
        for i, (obs, act) in tqdm(enumerate(train_dataloader), desc='Train Batch', position=1, total=len(train_dataloader), leave=False):
            train_start_time = time()
            obs = obs.to(device=cfg['device'], dtype=torch.float32)
            act = act.to(device=cfg['device'], dtype=torch.float32)
            global_step = i + epoch * epoch_steps

            optimizer.zero_grad()
            loss, metrics, info = model.get_loss(obs, act)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['optimizer']['grad_clip'] if 'grad_clip' in cfg['optimizer'] else np.inf)
            if info["segmentation_samples"].grad is not None:
                metrics['segmentation_samples_grad_max'] = torch.max(torch.abs(info["segmentation_samples"].grad))
                metrics['segmentation_samples_grad_avg'] = torch.mean(torch.abs(info["segmentation_samples"].grad))
            optimizer.step()
            lr_scheduler.step()
            metrics['grad_norm'] = grad_norm
            metrics['lr'] = lr_scheduler.get_last_lr()[0]
            metrics['step_time'] = time() - train_start_time

            if global_step % cfg['log_every'] == 0:
                train_metrics = {f'train/{k}' : v for k, v in metrics.items()}
                for k, v in train_metrics.items():
                    logger.add_scalar(k, v, global_step)

            if global_step % cfg['viz_every'] == 0:
                visualize(info, logger, global_step, prefix='train')

        if epoch % 10 == 0:
            with torch.no_grad():
                model.eval()
                metric_list = []
                for i, (obs, act) in tqdm(enumerate(test_dataloader), desc='Test Batch', position=1, total=len(test_dataloader), leave=False):
                    test_start_time = time()
                    obs = obs.to(device=cfg['device'], dtype=torch.float32)
                    act = act.to(device=cfg['device'], dtype=torch.float32)
                    loss, metrics, info = model.get_loss(obs, act)
                    metrics['step_time'] = time() - test_start_time

                metric_list.append(metrics)
                metrics = {k : np.mean([m[k] for m in metric_list]) for k in metric_list[0].keys()}
                metrics = {f'test/{k}' : v for k, v in metrics.items()}
                for k, v in metrics.items():
                    logger.add_scalar(k, v, global_step)

                if metrics['test/loss'] < best_test_loss:
                    best_test_loss = metrics['test/loss']
                    torch.save(model.state_dict(), os.path.join(cfg['log_dir'], cfg['name'] + timestring, 'best_model.pt'))
                    torch.save(model, os.path.join(cfg['log_dir'], cfg['name'] + timestring, 'full_model.pt'))

                visualize(info, logger, global_step, prefix='test')

def get_optimizer(cfg, model):
    if cfg['type'] == 'adam':
        return torch.optim.Adam(model.parameters(), **cfg['params'])
    else:
        raise NotImplementedError(f"Optimizer type {cfg['type']} not implemented")

def plt_prep(tensor):
    return tensor.detach().cpu().numpy().squeeze()

def visualize(info, logger, global_step, n_samples=3, prefix='train'):
    sequence_length = info['ground_truth_obs'].shape[1]
    norm = mpl.colors.Normalize(vmin=1, vmax=sequence_length)
    colors_gt = cm.autumn(norm(np.array([k for k in range(1, sequence_length + 1)])))

    plot_fig = plt.figure(0)
    delta_t_fig = plt.figure(1)
    delta_t_logit_fig = plt.figure(2)
    latent_features_fig = plt.figure(3)
    actions_fig = plt.figure(4)
    for i in range(n_samples):
        # Plot ground truth and reconstruction for n_samples
        plot_ax = plot_fig.add_subplot(n_samples, 1, i+1)
        gt_traj_x = plt_prep(info['ground_truth_obs'][i, :, 0])
        gt_traj_y = plt_prep(info['ground_truth_obs'][i, :, 1])
        segmentations = plt_prep(torch.sigmoid(info['segmentation_post_logits'][i])) > 0.5
        for t in range(sequence_length - 1):
            plot_ax.plot(gt_traj_x[t:t + 2], gt_traj_y[t:t + 2], c=colors_gt[t])
            if segmentations[t]:
                plot_ax.scatter(gt_traj_x[t], gt_traj_y[t], c='k', s=10)
        segmentations = plt_prep(torch.sigmoid(info['segmentation_post_logits'][i]))
        indices = np.arange(segmentations.shape[0])
        # plot_kl = plot_ax.twinx()
        # plot_kl.plot(plt_prep(info['state_kl'][i]), label='kl_divergence', c='r')

        # Plot delta_t for sequence
        delta_t_ax = delta_t_fig.add_subplot(n_samples, 1, i+1)
        delta_t_ax.plot(plt_prep(info['segmentation_samples'][i]), label='delta_t', c='r')

        # Plot delta_t logit for sequence
        delta_t_logit_ax = delta_t_logit_fig.add_subplot(n_samples, 1, i+1)
        delta_t_logit_ax.plot(plt_prep(torch.sigmoid(info['segmentation_post_logits'][i, :, 0])), label='post_prob', c='b')
        delta_t_logit_ax.plot(plt_prep(torch.sigmoid(info['segmentation_prior_logits'][i, :, 0])), label='prior prob', c='r')

        # Plot latent features for sequence
        latent_features_ax = latent_features_fig.add_subplot(n_samples, 1, i+1)
        for j in range(info['abstract_rep'].shape[-1]):
            latent_features_ax.plot(plt_prep(info['abstract_rep'][i, :, j]))

        # Plot actions for sequence
        actions_ax = actions_fig.add_subplot(n_samples, 1, i+1)
        actions_ax.set_ylim(-1, 1)
        actions_ax.plot(plt_prep(info['ground_truth_act'][i, :, 0]), label='action_x', c='r')
        actions_ax.plot(plt_prep(info['reconstructed_act'][i, :, 0]), label='action_x_reconstruction', c='orange')
        actions_ax.plot(plt_prep(info['ground_truth_act'][i, :, 1]), label='action_y', c='b')
        actions_ax.plot(plt_prep(info['reconstructed_act'][i, :, 1]), label='action_y_reconstruction', c='c')
        actions_ax.vlines(indices[segmentations > 0.5], -1, 1, label='segmentations', zorder=0, color='k')

        if i == 0:
            delta_t_ax.legend()
            delta_t_logit_ax.legend()
            actions_ax.legend()

    plot_fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cm.autumn), label="Time step (gt)")

    segmentations_fig = plt.figure(5)
    segmentations = (info['segmentation_post_logits'] > 0.5).squeeze()
    segmentation_x = plt_prep(info['ground_truth_obs'][:, 1:, 0][segmentations])
    segmentation_y = plt_prep(info['ground_truth_obs'][:, 1:, 1][segmentations])
    plt.scatter(segmentation_x, segmentation_y, c='k')


    logger.add_figure(prefix + '/reconstruction', plot_fig, global_step)
    logger.add_figure(prefix + '/delta_t', delta_t_fig, global_step)
    logger.add_figure(prefix + '/delta_t_logit', delta_t_logit_fig, global_step)
    logger.add_figure(prefix + '/latent_features', latent_features_fig, global_step)
    logger.add_figure(prefix + '/actions', actions_fig, global_step)
    logger.add_figure(prefix + '/segmentations', segmentations_fig, global_step)

    plot_fig.clf()
    delta_t_fig.clf()
    delta_t_logit_fig.clf()
    latent_features_fig.clf()
    actions_fig.clf()
    segmentations_fig.clf()

if __name__ == '__main__':
    test_full_prototype()