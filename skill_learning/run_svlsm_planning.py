import argparse
import os
import time
from typing import Dict

import gym
import numpy as np
import torch
import yaml
from statsmodels.stats.proportion import proportion_confint
from tqdm import tqdm

from networks import ModelFactory
from networks.supervised_vlsm import SVLSM
from utils import (EnvironmentWrapper, assemble_dicts, assemble_frames,
                   cem_planner, make_gif, make_video,
                   partition, plot_trajectories, random_planner)
from environments import maze2d

import matplotlib
matplotlib.use('Agg')

def run_planning(cfgs: Dict,
                 model: SVLSM,
                 planner=cem_planner,
                 render: bool=False):

    env = EnvironmentWrapper(cfgs['env_name'])
    env.reset()
    if cfgs['random_goals']:
        env.set_target()

    planning_args = dict(**cfgs,
                         skill_dim=model.skill_dim,
                         device=model.device,)

    rollouts_data = dict(selected_plan=[],
                         plan_data=[],
                         execution_data=[],)

    rollouts_data.update(init_data=env.obs, goal_data=env.goal)

    success = False
    timeout = False
    n_time_step = 0
    start_time = time.time()

    plans = 0
    while not (timeout or success):
        cost_fn = lambda skills: model.get_expected_costs(env.obs,
                                                          env.goal,
                                                          skills,
                                                          cost_fn=maze2d.sparse_cost_fn,
                                                          epsilon_planning=planning_args['epsilon_planning'],
                                                          sample=planning_args['sample_states'])

        if plans < planning_args['max_replans']:
            plan = planner(cost_fn,
                        planning_args,
                        init_state=env.obs,
                        epsilon_to_z=model.epsilon_to_z,)
        else:
            break

        plans += 1

        # Execute `n_skill_executions` skills from the plan
        for i, skill in zip(range(planning_args['n_skill_executions']), plan['skills']):
            skill_data = dict(obs=[],
                             actions=[],
                             next_state=[],
                             reward=[],
                             done=[],
                             frames=[],
                             milestones=[],)

            terminate = False
            model.init_skill(skill)
            while not terminate:
                action = model.act(env.obs, skill, sample=False)
                terminate = model.get_termination(env.obs, skill)

                skill_data['actions'].append(action)
                skill_data['obs'].append(env.obs)

                if len(skill_data['actions']) >= 50:
                    terminate = True

                next_state, reward, done, _, success, timeout = env.step(action)
                n_time_step += 1 # Keep track of steps taken
                model.increment_skill(skill)

                skill_data['next_state'].append(next_state)
                skill_data['reward'].append(reward)
                skill_data['done'].append(done)

                frame = env.render(render)
                if render.lower() == 'video':
                    skill_data['frames'].append(frame)

                if success or timeout:
                    break

            skill_data['milestones'].append(env.obs)

            rollouts_data['selected_plan'].append(plan['selected_plan'])
            if i == 0:
                rollouts_data['plan_data'].append(plan['plan_data'])
            else:
                rollouts_data['plan_data'].append(plan['plan_data'][-1:])
            rollouts_data['execution_data'].append(skill_data)

            if timeout or success:
                break

        rollouts_data['wallclock_time'] = time.time() - start_time
        rollouts_data['n_time_steps'] = float(n_time_step)

    return success, rollouts_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run planning to reach goal')
    # Logging args
    parser.add_argument('-l', '--log_dir', default='./logs', help='logging directory')
    parser.add_argument('-r', '--render', action='store_true', help='render?')
    parser.add_argument('-v', '--video', action='store_true', help='save video?')
    parser.add_argument('-g', '--gif', action='store_true', help='save GIF?')
    parser.add_argument('-p', '--plot_traj', action='store_true', help='plot trajectories?')
    parser.add_argument('-s', '--save_dir', default='./media', help='save path?')
    parser.add_argument('-d', '--device_id', default=0, type=int, help='CUDA device id')
    # Planning args
    parser.add_argument('-b', '--batch_size', default=100, type=int, help='Population size / batch size')
    parser.add_argument('--num_episodes', default=1, type=int)
    parser.add_argument('--plan_length', default=15, type=int)
    parser.add_argument('--num_iters', default=10, type=int, help='Number of CEM iterations')
    parser.add_argument('--keep_frac', default=0.1, type=float)
    parser.add_argument('--skill_std', default=1.0, type=float)
    parser.add_argument('--max_replans', default=1000, type=int, help='max number of replans before failure')
    parser.add_argument('--n_skill_executions', default=1, type=int, help='number of skills to executed from the choosen plan')
    parser.add_argument('--sample_states', action='store_true', help='sample states from the prior?')
    parser.add_argument('--model_name_str', default='best_model', type=str, help='best_model / model_10000 / ...')
    parser.add_argument('--cem_l2_pen', default=0.0, type=float)
    parser.add_argument('--epsilon_planning', action='store_true')
    parser.add_argument('--plot_failures', action='store_true')
    parser.add_argument('--random_goals', action='store_true')

    args = parser.parse_args()

    if args.render:
        from mujoco_py import GlfwContext
        GlfwContext(offscreen=True)

    config_path = os.path.join(args.log_dir, 'configs.yaml')

    with open(config_path, 'r') as f:
        cfgs = yaml.load(f, Loader=yaml.FullLoader)

    cfgs['device_id'] = args.device_id

    data_path = os.path.join(cfgs['data_dir'], cfgs['env_name'] + 'split.npz')
    save_dir = os.path.join(args.save_dir, args.log_dir)

    env = gym.make(cfgs['env_name'])
    data = np.load(cfgs['data_dir'], allow_pickle=True)

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

    os.makedirs(save_dir, exist_ok=True)
    render = 'display' if args.render else 'video' if (args.video or args.gif) else ''

    # Add planning args to cfgs
    cfgs.update(population=args.batch_size,
                max_replans=args.max_replans,
                plan_length=args.plan_length,
                skill_std=args.skill_std,
                keep_frac=args.keep_frac,
                l2_pen=0,
                num_iters=args.num_iters,
                n_skill_executions=args.n_skill_executions,
                epsilon_planning=args.epsilon_planning,
                cem_l2_pen=args.cem_l2_pen,
                sample_states=args.sample_states,
                random_goals=args.random_goals,
                )

    experiment_identifier = 'planning'

    device = torch.device('cuda:{}'.format(cfgs['device_id']) if torch.cuda.is_available() else 'cpu')

    data_specs = dict(
        obs_mean=obs_mean,
        obs_std=obs_std,
        actions_mean=actions_mean,
        actions_std=actions_std,
    )

    obs_dim, action_dim = obs.shape[-1], actions.shape[-1]

    # Create model and load parameters
    model = ModelFactory.create_model('svlsm',
                                      device=device,
                                      load_path=os.path.join(args.log_dir, f'{args.model_name_str}.pth'),
                                      data_specs=data_specs,
                                      obs_dim=obs_dim,
                                      action_dim=action_dim,
                                      skill_length=cfgs['skill_length'] if 'skill_length' in cfgs else 0,
                                      **cfgs['model'])

    model.eval()

    results = dict(n_success=0,
                   n_trials=args.num_episodes,
                   n_time_steps=[],
                   wallclock_times=[],
                   init=[],
                   goal=[],)

    total_timesteps = 0

    for i in (pbar:=tqdm(range(args.num_episodes), dynamic_ncols=True)):
        planner = cem_planner
        with torch.no_grad():
            success, rollouts_data = run_planning(cfgs,
                                                  model,
                                                  planner=planner,
                                                  render=render)
            total_timesteps += rollouts_data['n_time_steps']

        if success:
            results['n_success'] += 1
            results['n_time_steps'].append(rollouts_data['n_time_steps'])
            results['wallclock_times'].append(rollouts_data['wallclock_time'])
            results['init'].append(rollouts_data['init_data'])
            results['goal'].append(rollouts_data['goal_data'])

        ci = proportion_confint(results['n_success'], i + 1)

        pbar.set_description(f'Success: {results["n_success"]}/{i + 1}; CI: {ci}, Average timesteps: {total_timesteps / (i + 1)}')

        if args.video:
            for run_identifier, run_data in rollouts_data.items():
                if not isinstance(run_data, Dict):
                    continue
                elif 'frames' in run_data.keys():
                    make_video(run_data['frames'], save_path=os.path.join(save_dir, experiment_identifier, f'{i}_{run_identifier}.mp4'), label=run_identifier)

        if args.gif:
            for run_identifier, run_data in rollouts_data.items():
                # if not isinstance(run_data, Dict):
                if 'execution' not in run_identifier:
                    continue
                #     continue
                # elif 'frames' in run_data.keys():
                frames = assemble_frames(run_data)
                if len(frames) == 0:
                    continue
                make_gif(frames, save_path=os.path.join(save_dir, experiment_identifier, f'{i}_{run_identifier}.gif'), label=run_identifier)

        plot_every = 1
        plot_idx = 0
        if args.plot_traj or (args.plot_failures and not success):
            for idx in tqdm(range(len(rollouts_data['plan_data'])), desc='plotting', leave=False, dynamic_ncols=True):
                for plan_idx, plan in enumerate(rollouts_data['plan_data'][idx]):
                    if plan_idx % plot_every != 0:
                        continue
                    traj_data = {
                        'execution': dict(**assemble_dicts(rollouts_data['execution_data'][:idx]),),
                        'plan': dict(milestones=plan['all_plans']['milestones']),
                    }

                    save_path_trajs = os.path.join(save_dir, experiment_identifier, str(i), f'trajectories_{plot_idx}.png')
                    plot_idx += 1

                    plot_trajectories(traj_data,
                                        init_data=rollouts_data['init_data'],
                                        goal_data=rollouts_data['goal_data'],
                                        save_path=save_path_trajs)

            save_path_trajs = os.path.join(save_dir, experiment_identifier, str(i), f'trajectories_{plot_idx + plot_every}.png')
            traj_data = {
                'execution': dict(**assemble_dicts(rollouts_data['execution_data']),),
                'plan': dict(),
            }

            plot_trajectories(traj_data,
                                init_data=rollouts_data['init_data'],
                                goal_data=rollouts_data['goal_data'],
                                save_path=save_path_trajs)

    # Save eval results
    save_path = os.path.join(save_dir, experiment_identifier, f'eval_results.pth')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(results, save_path)
