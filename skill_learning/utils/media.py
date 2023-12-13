
import os

import imageio
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as colors
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from PIL import Image, ImageDraw, ImageFont

beautify_text = lambda x: x.replace('_', ' ').capitalize()
add_text = lambda x, text: ImageDraw.Draw(x).text((28, 36), beautify_text(text), font=ImageFont.load_default(), fill=(255, 255, 255))

def visualize(return_fig=True):
    return None

def make_gif(frames, save_path, label=None):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    frames = [Image.fromarray(image) for image in frames]
    for img in frames:
        if label is not None: add_text(img, label)
    frame_one = frames[0]
    frame_one.save(save_path, format="GIF", append_images=frames,
               save_all=True, duration=50, loop=0)

def make_video(frames, save_path, label=None, fps=20):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    writer = imageio.get_writer(save_path, fps=fps)
    for img in frames:
        pil_img = Image.fromarray(img)
        if label is not None: add_text(pil_img, label)
        writer.append_data(np.asarray(pil_img))
    writer.close()

def plot_trajectories(traj_data, init_data=None, goal_data=None, save_path=None, adaptive_scale=False):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig = plt.figure()
    min_x, min_y, max_x, max_y = 1e10, 1e10, -1e10, -1e10
    plotted_init = False

    def get_plt_kwargs(identifier, color=None, marker=None, alpha=None, z_order=None):
        if 'execution' in identifier:
            return plt.scatter, dict(color='r', marker='.', alpha=0.8, zorder=100)
        elif 'selected_plan' in identifier:
            return plt.scatter, dict(color='g', marker='.', alpha=0.3, zorder=50)
        elif 'topk' in identifier:
            return plt.scatter, dict(color='m', marker='.', alpha=0.1, zorder=10)
        elif 'prior_plan' in identifier:
            return plt.scatter, dict(color='g', alpha=0.8, marker='o', zorder=900)
        elif 'plan' in identifier:
            return plt.scatter, dict(color='b', alpha=0.8, marker='.', zorder=1000)
        else:
            return plt.scatter, dict(color=color, marker=marker)

    for (identifier, data), color, marker in zip(traj_data.items(), colors.CSS4_COLORS.keys(), Line2D.markers.keys()):
        traj = data.get('traj', None)
        milestones = data.get('milestones', None)
        color = data.get('color', color)
        marker = data.get('marker', marker)
        # color, marker, alpha, z_order = get_plt_args(identifier, color, marker)
        plotter, plt_kwargs = get_plt_kwargs(identifier, color, marker)
        # plt_kwargs = dict(color=color, marker=marker, alpha=alpha, zorder=z_order)

        if traj is not None:
            if len(traj.shape) == 2:
                plotter(traj[:, 0], traj[:, 1], label=beautify_text(identifier), **plt_kwargs)
                if not plotted_init:
                    # plt.scatter([traj[0, 0]], [traj[0, 1]], c='k', s=50, label="Initial position", zorder=200, marker='*')
                    if init_data is None:
                        init_data = traj[0, :2]
                    plotted_init = True
                min_x = np.minimum(min_x, (traj[:, 0]).min())
                min_y = np.minimum(min_y, (traj[:, 1]).min())
                max_x = np.maximum(max_x, (traj[:, 0]).max())
                max_y = np.maximum(max_y, (traj[:, 1]).max())
            elif len(traj.shape) == 3:
                trajs = traj
                for i, traj in enumerate(trajs):
                    plotter(traj[:, 0], traj[:, 1], label=beautify_text(identifier) if i == 0 else None, **plt_kwargs)
                    if not plotted_init:
                        # plt.scatter([traj[0, 0]], [traj[0, 1]], c='k', s=50, label="Initial position", zorder=200, marker='*')
                        if init_data is None:
                            init_data = traj[0, :2]
                        plotted_init = True
                    min_x = np.minimum(min_x, (traj[:, 0]).min())
                    min_y = np.minimum(min_y, (traj[:, 1]).min())
                    max_x = np.maximum(max_x, (traj[:, 0]).max())
                    max_y = np.maximum(max_y, (traj[:, 1]).max())
            else:
                raise NotImplementedError(f'Invalid shape for trajectory data: {traj.shape}')
        if milestones is not None:
            if 'plan' not in identifier:
                marker, s = 'x', 150
            else:
                marker, s = '.', 10
            alpha = plt_kwargs.get('alpha', 0.8)
            min_x = np.minimum(min_x, (milestones[..., 0]).min())
            min_y = np.minimum(min_y, (milestones[..., 1]).min())
            max_x = np.maximum(max_x, (milestones[..., 0]).max())
            max_y = np.maximum(max_y, (milestones[..., 1]).max())
            plt.scatter(milestones[..., 0], milestones[..., 1], color=plt_kwargs['color'], marker=marker, s=s, alpha=alpha)

    # Plot initial position
    plt.scatter(*init_data[:2], c='k', s=50, label="Initial position", zorder=200, marker='*')
    # Plot goal position if specified
    if goal_data is not None:
        plt.scatter(*goal_data[:2], c='k', s=50, label="Goal position", zorder=200, marker='X')
        if isinstance(goal_data, np.ndarray):
            min_x = np.minimum(min_x, (goal_data[..., 0]).min())
            min_y = np.minimum(min_y, (goal_data[..., 1]).min())
            max_x = np.maximum(max_x, (goal_data[..., 0]).max())
            max_y = np.maximum(max_y, (goal_data[..., 1]).max())
        else:
            min_x = np.minimum(min_x, goal_data[0])
            min_y = np.minimum(min_y, goal_data[1])
            max_x = np.maximum(max_x, goal_data[0])
            max_y = np.maximum(max_y, goal_data[1])

    if not adaptive_scale:
        min_y = min_x = 0
        max_y = 6.3
        max_x = 6.3
        # max_y = 5.3
        # max_x = 3.3

    range_x = np.abs(max_x - min_x)
    range_y = np.abs(max_y - min_y)
    axis_range = np.maximum(range_x, range_y)
    axis_offset = 0.6 * axis_range
    x_center = (min_x + max_x) / 2
    y_center = (min_y + max_y) / 2
    if 'ground_truth' in traj_data.keys():
        traj = traj_data['ground_truth']['traj']
        x_init, y_init = traj[0, 0], traj[0, 1]
        plt.scatter(x_init, y_init, color='k', label=beautify_text('start'))
    plt.legend()
    plt.xlim(x_center - axis_offset, x_center + axis_offset)
    plt.ylim(y_center - axis_offset, y_center + axis_offset)
    if save_path is not None:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()
    plt.clf()
    plt.close()

def plot_traj(traj_mean, traj_std, traj_gt, skill_times, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # Traj should be (seq_len, feature dim)
    assert traj_mean.shape[:-1] == traj_std.shape[:-1] == traj_gt.shape[:-1]
    sequence_length = traj_mean.shape[0]

    fig = plt.figure()
    norm = mpl.colors.Normalize(vmin=1, vmax=sequence_length)
    colors = cm.winter(norm(np.array([k for k in range(1, sequence_length + 1)])))
    colors_gt = cm.autumn(norm(np.array([k for k in range(1, sequence_length + 1)])))

    x = traj_mean[:, 0]
    y = traj_mean[:, 1]
    x_std = traj_std[:, 0]
    y_std = traj_std[:, 1]
    x_gt = traj_gt[:, 0]
    y_gt = traj_gt[:, 1]
    skill_xs = []
    skill_ys = []
    for t in range(sequence_length - 1):
        plt.plot(x_gt[t:t + 2], y_gt[t:t + 2], c=colors_gt[t])
        plt.plot(x[t:t + 2], y[t:t + 2], c=colors[t])
        plt.gca().add_patch(Ellipse((x[t+1].item(), y[t+1].item()), width=x_std[t+1].item(), height=y_std[t+1].item(), edgecolor=colors[t+1], facecolor='none', alpha=0.3))

        skills_in_step = np.logical_and(t <= skill_times, skill_times <= t + 1)
        if np.any(skills_in_step):
            alphas = skill_times[skills_in_step] - t
            skill_xs.append(((1 - alphas) * x[t] + alphas * x[t + 1]).reshape((-1,)))
            skill_ys.append(((1 - alphas) * y[t] + alphas * y[t + 1]).reshape((-1,)))

    skill_xs.append(x[-1:])
    skill_ys.append(y[-1:])
    plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cm.winter), label="Time step")
    plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cm.autumn), label="Time step (gt)")
    plt.scatter(np.concatenate(skill_xs, axis=0), np.concatenate(skill_ys, axis=0), c='k', s=5, label="Skill Segmentations", zorder=10)
    plt.legend()

    x_mid = (np.max(np.maximum(x, x_gt)) + np.min(np.minimum(x, x_gt))) / 2
    y_mid = (np.max(np.maximum(y, y_gt)) + np.min(np.minimum(y, y_gt))) / 2
    x_range = (np.max(np.maximum(x, x_gt)) - np.min(np.minimum(x, x_gt))) / 2
    y_range = (np.max(np.maximum(y, y_gt)) - np.min(np.minimum(y, y_gt))) / 2
    max_range = np.maximum(x_range, y_range)
    plt.xlim(np.maximum(-10, x_mid - max_range), np.minimum(10, x_mid + max_range))
    plt.ylim(np.maximum(-10, y_mid - max_range), np.minimum(10, y_mid + max_range))

    plt.savefig(save_path, dpi=300)

def plot_milestones(trajs_mean, trajs_std, goal_data=None, save_path=None):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # Traj should be (batch_size, seq_len, feature dim)
    assert trajs_mean.shape[:-1] == trajs_std.shape[:-1]
    # sequence_length = traj_mean.shape[0]
    batch_size, sequence_length, _ = trajs_mean.shape

    fig = plt.figure()
    norm = mpl.colors.Normalize(vmin=1, vmax=sequence_length)
    colors = cm.winter(norm(np.array([k for k in range(1, sequence_length + 1)])))

    for traj_mean, traj_std in zip(trajs_mean, trajs_std):
        x = traj_mean[:, 0]
        y = traj_mean[:, 1]
        # print("x", x)
        # print("y", y)
        x_std = traj_std[:, 0]
        y_std = traj_std[:, 1]
        for t in range(sequence_length - 1):
            plt.plot(x[t:t + 2], y[t:t + 2], color=colors[t])
            # plt.gca().add_patch(Ellipse((x[t+1].item(), y[t+1].item()), width=x_std[t+1].item(), height=y_std[t+1].item(), edgecolor=colors[t+1], facecolor='none', alpha=0.3))

    plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cm.winter), label="Plan step")

    plt.scatter([trajs_mean[0, 0, 0]], [trajs_mean[0, 0, 1]], c='k', s=5, label="Initial Position", zorder=10)
    plt.title('Imagined trajectories')

    x = trajs_mean[..., 0].flatten()
    y = trajs_mean[..., 1].flatten()

    max_x = np.max(x)
    max_y = np.max(y)
    min_x = np.min(x)
    min_y = np.min(y)

    if goal_data is not None:
        plt.scatter(*goal_data[:2], c='k', s=50, label="Goal position", zorder=200, marker='X')
        if isinstance(goal_data, tuple):
            goal_data = np.array(goal_data)
        min_x = np.minimum(min_x, (goal_data[..., 0]).min())
        min_y = np.minimum(min_y, (goal_data[..., 1]).min())
        max_x = np.maximum(max_x, (goal_data[..., 0]).max())
        max_y = np.maximum(max_y, (goal_data[..., 1]).max())

    x_mid = (max_x + min_x) / 2
    y_mid = (max_y + min_y) / 2
    x_range = (max_x - min_x) / 2 + 0.5
    y_range = (max_y - min_y) / 2 + 0.5
    max_range = np.maximum(x_range, y_range)

    plt.xlim(np.maximum(-10, x_mid - max_range), np.minimum(10, x_mid + max_range))
    plt.ylim(np.maximum(-10, y_mid - max_range), np.minimum(10, y_mid + max_range))

    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)
    else:
       plt.show()
    plt.close()


def plot_plan_costs(trajs_mean, trajs_std, goal_data, save_path=None):

    def cost_fn(x, y):
        """Calculate the L2 cost of all plans."""
        _cost = np.sqrt((x[..., 0] - y[..., 0]) ** 2 + (x[..., 1] - y[..., 1]) ** 2)
        return _cost.min(axis=-1)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # Traj should be (batch_size, seq_len, feature dim)
    assert trajs_mean.shape[:-1] == trajs_std.shape[:-1]
    # sequence_length = traj_mean.shape[0]
    batch_size, sequence_length, _ = trajs_mean.shape

    if isinstance(goal_data, tuple):
        goal_data = np.array(goal_data)

    costs = cost_fn(trajs_mean, goal_data)

    fig = plt.figure()
    norm = mpl.colors.Normalize(vmin=costs.min(), vmax=costs.max())
    colors = cm.winter(norm(costs))

    for c, traj_mean, traj_std in zip(colors, trajs_mean, trajs_std):
        x = traj_mean[:, 0]
        y = traj_mean[:, 1]
        # print("x", x)
        # print("y", y)
        x_std = traj_std[:, 0]
        y_std = traj_std[:, 1]
        for t in range(sequence_length - 1):
            plt.plot(x[t:t + 2], y[t:t + 2], color=c)

    plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cm.winter), label="Cost")

    plt.scatter([trajs_mean[0, 0, 0]], [trajs_mean[0, 0, 1]], c='k', s=5, label="Initial Position", zorder=10)
    plt.title('Imagined trajectories')

    x = trajs_mean[..., 0].flatten()
    y = trajs_mean[..., 1].flatten()

    max_x = np.max(x)
    max_y = np.max(y)
    min_x = np.min(x)
    min_y = np.min(y)

    if goal_data is not None:
        plt.scatter(*goal_data[:2], c='k', s=50, label="Goal position", zorder=200, marker='X')
        if isinstance(goal_data, tuple):
            goal_data = np.array(goal_data)
        min_x = np.minimum(min_x, (goal_data[..., 0]).min())
        min_y = np.minimum(min_y, (goal_data[..., 1]).min())
        max_x = np.maximum(max_x, (goal_data[..., 0]).max())
        max_y = np.maximum(max_y, (goal_data[..., 1]).max())

    x_mid = (max_x + min_x) / 2
    y_mid = (max_y + min_y) / 2
    x_range = (max_x - min_x) / 2 + 0.5
    y_range = (max_y - min_y) / 2 + 0.5
    max_range = np.maximum(x_range, y_range)

    plt.xlim(np.maximum(-10, x_mid - max_range), np.minimum(10, x_mid + max_range))
    plt.ylim(np.maximum(-10, y_mid - max_range), np.minimum(10, y_mid + max_range))

    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)
    else:
       plt.show()
    plt.close()

def create_plots(trajs_mean, trajs_std, imagined_traj, executed_traj, inferred_traj, executed_milestones, save_path=None):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # Traj should be (batch_size, seq_len, feature dim)
    def plot(data, plot_type, color, label=None, marker=None, alpha=1.0):
        x = data[:, 0]
        y = data[:, 1]
        plot_kwargs = dict(marker=marker, alpha=alpha)
        if plot_type.lower() == 'scatter':
            for idx, (c, _x, _y) in enumerate(zip(colors, x, y)):
                label = label if idx == 1 else None
                plot_kwargs.update(label=label)
                plt.scatter(_x, _y, c, color=c)
        elif plot_type.lower() == 'line':
             for idx, (c, _x, _y) in enumerate(zip(colors, x, y)):
                label = label if idx == 1 else None
                plot_kwargs.update(label=label)
                plt.plot(_x, _y, c, color=c)

    sequence_length, _ = trajs_mean.shape
    sequence_length_gt, _ = executed_traj.shape
    _, sequence_length_imagination, _ = imagined_traj.shape

    fig = plt.figure()
    norm = mpl.colors.Normalize(vmin=1, vmax=sequence_length)
    norm_gt = mpl.colors.Normalize(vmin=1, vmax=sequence_length_gt)
    norm_imagination = mpl.colors.Normalize(vmin=1, vmax=sequence_length_imagination)
    colors = cm.winter(norm(np.array([k for k in range(1, sequence_length + 1)])))
    colors_gt = cm.autumn(norm_gt(np.array([k for k in range(1, sequence_length_gt + 1)])))
    colors_imagination = cm.winter(norm_imagination(np.array([k for k in range(1, sequence_length_imagination + 1)])))

    x_inferred = inferred_traj[:, 0]
    y_inferred = inferred_traj[:, 1]
    x = trajs_mean[:, 0]
    y = trajs_mean[:, 1]
    imagined_traj = imagined_traj[0]
    x_imagined = imagined_traj[:, 0]
    y_imagined = imagined_traj[:, 1]
    x_executed = executed_traj[:, 0]
    y_executed = executed_traj[:, 1]
    x_std = trajs_std[:, 0]
    y_std = trajs_std[:, 1]
    x_gt_milestones = executed_milestones[:, 0]
    y_gt_milestones = executed_milestones[:, 1]

    for idx, (_x, _y) in enumerate(zip(x_inferred, y_inferred)):
        label = beautify_text('inferred') if idx == 0 else None
        plt.scatter(_x, _y, color='k', marker='.', label=label)
    for idx, (_x, _y, c) in enumerate(zip(x, y, colors)):
        label = beautify_text('imagined milestones') if idx == 0 else None
        plt.scatter(_x, _y, color=c, marker='x', s=120, label=label)
        # plt.gca().add_patch(Ellipse((_x, _y), width=x_std[idx], height=y_std[idx], edgecolor=c, facecolor='none', alpha=0.3))
    for idx, (_x, _y, c) in enumerate(zip(x_imagined, y_imagined, colors_imagination)):
        label = beautify_text('imagined') if idx == 0 else None
        plt.scatter(_x, _y, color=c, s=5, zorder=10, label=label, marker='8', alpha=0.8)
    for idx, (_x, _y, c) in enumerate(zip(x_executed, y_executed, colors_gt)):
        label = beautify_text('executed') if idx == 0 else None
        plt.scatter(_x, _y, color=c, label=label, alpha=0.5)
    for idx, (_x, _y, c) in enumerate(zip(x_gt_milestones, y_gt_milestones, colors_gt[1:])):
        label = beautify_text('executed milestones') if idx == 0 else None
        plt.scatter(_x, _y, color=c, label=label, marker='x', s=120)
    # plt.plot(x_gt, y_gt, c=colors_gt[0], label='executed')
    plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cm.winter), label="Abstract time step")

    x = np.concatenate([x_inferred, x_imagined, x_executed], axis=0).flatten()
    y = np.concatenate([y_inferred, y_imagined, y_executed], axis=0).flatten()
    x_mid = (np.max(x) + np.min(x)) / 2
    y_mid = (np.max(y) + np.min(y)) / 2
    x_range = (np.max(x)- np.min(x))/ 2 + 0.5
    y_range = (np.max(y)- np.min(y))/ 2 + 0.5
    max_range = np.maximum(x_range, y_range)
    plt.xlim(np.maximum(-10, x_mid - max_range), np.minimum(10, x_mid + max_range))
    plt.ylim(np.maximum(-10, y_mid - max_range), np.minimum(10, y_mid + max_range))

    plt.title('Trajectories')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)
    else:
       plt.show()

def plot_actions(actions_means,
                 actions_stds,
                 actions_gt,
                 save_path,
                 skill_lengths=None,
                 gt_skill_lengths=None):

    n_actions = actions_means.shape[1]

    fig, axis = plt.subplots(n_actions, 1, figsize=(6.4, 2.4 * n_actions))

    for i in range(n_actions):
        axis[i].plot(actions_means[:, i], c='b', label='pred')
        axis[i].fill_between(np.arange(actions_means.shape[0]), actions_means[:, i] + actions_stds[:, i], actions_means[:, i] - actions_stds[:, i], color='b', alpha=0.2)
        axis[i].plot(actions_gt[:, i], c='C1', label='gt')
        if skill_lengths is not None:
            axis[i].vlines(skill_lengths, -2, 2, colors='g', label='pred skill boundaries', linewidth=2.5)
        if gt_skill_lengths is not None:
            axis[i].vlines(gt_skill_lengths, -2, 2, colors='k', label='gt skill boundaries')
        axis[i].set_ylim(-2, 2)
        if i == 0:
            axis[i].legend()

    plt.savefig(save_path)
    plt.close()

def plot_states_and_terminal_state_predictions(
    gt_obs,
    pred_obs_means,
    pred_obs_stds,
    pred_obs_times,
    skill_lengths,
    gt_skill_lengths,
    save_path,
):
    n_obs = gt_obs.shape[1]

    fig, axis = plt.subplots(n_obs, 1, figsize=(6.4, 2.4 * n_obs))

    for i in range(n_obs):
        axis[i].plot(gt_obs[:, i], c='C1', label='gt')
        axis[i].scatter(pred_obs_times, pred_obs_means[:, i], c='b', label='pred', zorder=12)
        axis[i].errorbar(pred_obs_times, pred_obs_means[:, i], yerr=pred_obs_stds[:, i], c='c', zorder=11)
        min_val = np.minimum(gt_obs[:, i].min(), pred_obs_means[:, i].min() - pred_obs_stds[:, i].max())
        max_val = np.maximum(gt_obs[:, i].max(), pred_obs_means[:, i].max() + pred_obs_stds[:, i].max())
        if skill_lengths is not None:
            axis[i].vlines(skill_lengths, min_val, max_val, colors='g', label='pred skill boundaries', linewidth=2.5)
        if gt_skill_lengths is not None:
            axis[i].vlines(gt_skill_lengths, min_val, max_val, colors='k', label='gt skill boundaries')
        if i == 0:
            axis[i].legend()

    plt.savefig(save_path)
    plt.close()

from environments import kitchen

def plot_obj_states_and_terminal_state_predictions(
    gt_obs,
    pred_obs_means,
    pred_obs_stds,
    pred_obs_times,
    save_path,
):
    gt_obj_states = gt_obs[..., kitchen.IDX_OFFSET:]
    n_obj_states = gt_obj_states.shape[1]

    fig, axis = plt.subplots(n_obj_states, 1, figsize=(6.4, 2.4 * n_obj_states), constrained_layout=True)

    pred_obj_means = pred_obs_means[:, kitchen.IDX_OFFSET:]
    pred_obj_stds = pred_obs_stds[:, kitchen.IDX_OFFSET:]
    pred_obj_means = np.concatenate([gt_obj_states[:1], pred_obj_means], axis=0)
    pred_obj_stds = np.concatenate([np.zeros_like(gt_obj_states[:1]), pred_obj_stds], axis=0)
    pred_obs_times = np.concatenate([np.array([0]), pred_obs_times], axis=0)

    for i in range(n_obj_states):
        axis[i].plot(gt_obj_states[:, i], c='C1', label='gt')
        axis[i].scatter(pred_obs_times, pred_obj_means[:, i], c='b', label='pred', zorder=12)
        axis[i].errorbar(pred_obs_times, pred_obj_means[:, i], yerr=pred_obj_stds[:, i], c='c', zorder=11)
        if i + kitchen.IDX_OFFSET in kitchen.OBS_INDEX_TO_ELEMENT.keys():
            axis[i].set_title(kitchen.OBS_INDEX_TO_ELEMENT[i + kitchen.IDX_OFFSET])
        if i == 0:
            axis[i].legend()

    plt.savefig(save_path)
    plt.close()

def plot_obj_states_and_terminal_state_predictions_and_plans(
    gt_obs,
    pred_obs_means,
    pred_obs_stds,
    pred_obs_times,
    plan_means,
    # plan_stds,
    plan_times,
    save_path,
):
    gt_obj_states = gt_obs[..., kitchen.IDX_OFFSET:]
    n_obj_states = gt_obj_states.shape[1]

    fig, axis = plt.subplots(n_obj_states, 1, figsize=(6.4, 2.4 * n_obj_states), constrained_layout=True)

    pred_obj_means = pred_obs_means[:, kitchen.IDX_OFFSET:]
    pred_obj_stds = pred_obs_stds[:, kitchen.IDX_OFFSET:]
    plan_means = plan_means[..., kitchen.IDX_OFFSET:]
    pred_obj_means = np.concatenate([gt_obj_states[:1], pred_obj_means], axis=0)
    pred_obj_stds = np.concatenate([np.zeros_like(gt_obj_states[:1]), pred_obj_stds], axis=0)
    pred_obs_times = np.concatenate([np.array([0]), pred_obs_times], axis=0)

    # Plan is of shape (n_samples, plan_len, n_obj_states)
    # Rollout plan times and plan means, stds to plot in one scatter command
    plan_len = plan_means.shape[1]
    plan_times = plan_times[:plan_len]
    plan_times = np.tile(plan_times.reshape((1, plan_len)), (plan_means.shape[0], 1)).reshape((-1,))
    plan_means = plan_means.reshape((-1, n_obj_states))
    # plan_stds = plan_stds.reshape((-1, n_obj_states))
    plan_means = np.clip(plan_means, -2, 2)
    # plan_stds = np.clip(plan_stds, -2, 2)

    for i in range(n_obj_states):
        axis[i].plot(gt_obj_states[:, i], c='C1', label='gt')
        axis[i].scatter(pred_obs_times, pred_obj_means[:, i], c='b', label='pred', zorder=12)
        axis[i].errorbar(pred_obs_times, pred_obj_means[:, i], yerr=pred_obj_stds[:, i], c='c', zorder=11)
        axis[i].scatter(plan_times, plan_means[:, i], c='r', label='plan', zorder=12)
        # axis[i].errorbar(plan_times, plan_means[:, i], yerr=plan_stds[:, i], c='m', zorder=11)

        if i + kitchen.IDX_OFFSET in kitchen.OBS_INDEX_TO_ELEMENT.keys():
            axis[i].set_title(kitchen.OBS_INDEX_TO_ELEMENT[i + kitchen.IDX_OFFSET])
        if i == 0:
            axis[i].legend()

    plt.savefig(save_path)
    plt.close()

def plot_traj_and_terminal_state_predictions(
        gt_obs,
        pred_obs_means,
        pred_obs_stds,
        save_path,

):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    sequence_length = gt_obs.shape[0]
    n_samples = pred_obs_means.shape[0]

    fig = plt.figure()

    gt_norm = mpl.colors.Normalize(vmin=1, vmax=sequence_length)
    gt_colors = cm.autumn(gt_norm(np.array([k for k in range(1, sequence_length + 1)])))

    norm = mpl.colors.Normalize(vmin=1, vmax=n_samples)
    colors = cm.winter(norm(np.array([k for k in range(1, n_samples + 1)])))

    x = pred_obs_means[:, 0]
    y = pred_obs_means[:, 1]
    x_std = pred_obs_stds[:, 0]
    y_std = pred_obs_stds[:, 1]
    x_gt = gt_obs[:, 0]
    y_gt = gt_obs[:, 1]
    for t in range(sequence_length - 1):
        plt.plot(x_gt[t:t + 2], y_gt[t:t + 2], c=gt_colors[t])

    for t in range(n_samples):
        plt.scatter(x[t], y[t], color=colors[t])
        plt.gca().add_patch(Ellipse((x[t].item(), y[t].item()), width=x_std[t].item(), height=y_std[t].item(), edgecolor=colors[t], facecolor='none', alpha=0.3))

        # skills_in_step = np.logical_and(t <= skill_times, skill_times <= t + 1)
        # if np.any(skills_in_step):
        #     alphas = skill_times[skills_in_step] - t
        #     skill_xs.append(((1 - alphas) * x[t] + alphas * x[t + 1]).reshape((-1,)))
        #     skill_ys.append(((1 - alphas) * y[t] + alphas * y[t + 1]).reshape((-1,)))

    plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cm.winter), label="Time step")
    plt.colorbar(cm.ScalarMappable(norm=gt_norm, cmap=cm.autumn), label="Time step (gt)")
    # plt.legend()

    x_mid = (np.max(np.max(np.concatenate([x, x_gt]))) + np.min(np.min(np.concatenate([x, x_gt])))) / 2
    y_mid = (np.max(np.max(np.concatenate([y, y_gt]))) + np.min(np.min(np.concatenate([y, y_gt])))) / 2
    x_range = (np.max(np.max(np.concatenate([x, x_gt]))) - np.min(np.min(np.concatenate([x, x_gt])))) / 2
    y_range = (np.max(np.max(np.concatenate([y, y_gt]))) - np.min(np.min(np.concatenate([y, y_gt])))) / 2

    max_range = np.maximum(x_range, y_range)

    plt.xlim(np.maximum(-10, x_mid - max_range), np.minimum(10, x_mid + max_range))
    plt.ylim(np.maximum(-10, y_mid - max_range), np.minimum(10, y_mid + max_range))

    plt.savefig(save_path, dpi=300)
    plt.clf()
