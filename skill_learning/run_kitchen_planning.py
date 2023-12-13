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
                   partition, plot_trajectories, random_planner,
                   KitchenDataset, SegmentedKitchenDataset,
                   plot_obj_states_and_terminal_state_predictions,
                   plot_obj_states_and_terminal_state_predictions_and_plans)
from environments import kitchen

import matplotlib
matplotlib.use('Agg')

def run_planning(cfgs: Dict,
                 model: SVLSM,
                 planner=cem_planner,
                 render: bool=False):

    if not args.random_goals:
        env = gym.make(cfgs['env_name'])
        print('env_name:', cfgs['env_name'])
    else:
        env = gym.make('kitchen-random-v0')
        print('env_name: kitchen-random-v0')
    goal_tasks = env.TASK_ELEMENTS
    print('goal_tasks:', goal_tasks)

    plan_length = 0
    if 'microwave' in goal_tasks:
        plan_length += 2
    if 'kettle' in goal_tasks:
        plan_length += 1
    if 'bottom burner' in goal_tasks:
        plan_length += 2
    if 'top burner' in goal_tasks:
        plan_length += 2
    if 'light switch' in goal_tasks:
        plan_length += 1
    if 'slide cabinet' in goal_tasks:
        plan_length += 1
    if 'hinge cabinet' in goal_tasks:
        plan_length += 1

    planning_args = dict(**cfgs,
                         skill_dim=model.skill_dim,
                         device=model.device,)

    rollouts_data = dict(selected_plan=[],
                         plan_data=[],
                         execution_data=[],)

    obs = env.reset()
    if cfgs['model']['kitchen_remove_goal']:
        obs = obs[:-30]
    # goal_tasks = ['microwave', 'kettle', 'light switch', 'slide cabinet']
    # TASKS = ['bottom burner', 'top burner', 'light switch', 'slide cabinet', 'hinge cabinet', 'microwave', 'kettle']
    # goal_tasks = ['microwave', 'top burner', 'light switch', 'hinge cabinet']
    # goal_tasks = ['microwave', 'top burner', 'bottom burner', 'hinge cabinet']
    # goal_tasks = ['microwave', 'kettle', 'bottom burner', 'light switch']

    success = False
    timeout = False
    n_time_step = 0
    start_time = time.time()

    if cfgs['reward_function_type'] == 'dense':
        kitchen_cost_fn = kitchen.dense_cost_fn
    elif cfgs['reward_function_type'] == 'sparse':
        kitchen_cost_fn = kitchen.sparse_cost_fn
    elif cfgs['reward_function_type'] == 'sparse_random':
        kitchen_cost_fn = kitchen.sparse_random_cost_fn
    else:
        raise NotImplementedError(f'Cost function type {cfgs["reward_function_type"]} not implemented')

    plans = 0
    max_plan_length = planning_args['plan_length']
    # max_plan_length = plan_length if model.skill_length == 0 else planning_args['plan_length']
    print('plan_length:', max_plan_length)
    total_reward = 0
    while not (timeout or success):
        cost_fn = lambda skills: model.get_expected_costs(obs,
                                                          goal_tasks,
                                                          skills,
                                                          cost_fn=kitchen_cost_fn,
                                                          epsilon_planning=planning_args['epsilon_planning'],
                                                          sample=planning_args['sample_states'])

        if plans < planning_args['max_replans']:
            planning_args['plan_length'] = max(1, max_plan_length - plans)
            plan = planner(cost_fn,
                        planning_args,
                        init_state=obs,
                        epsilon_to_z=model.epsilon_to_z,)
        else:
            break

        # Execute `n_skill_executions` skills from the plan
        for i, skill in zip(range(planning_args['n_skill_executions']), plan['skills']):
            print(f'Executing skill {i} from plan {plans}')
            skill_data = dict(obs=[],
                             actions=[],
                             next_state=[],
                             reward=[],
                             done=[],
                             frames=[],
                             milestones=[],)

            terminate = False
            model.init_skill(skill)
            skill_data['predicted_obs_means'], skill_data['predicted_obs_stds'] = model.decode_obs(obs, skill, normalize=model.normalize_inputs)
            skill_data['predicted_obs_means'] = skill_data['predicted_obs_means'].detach().cpu().numpy()
            skill_data['predicted_obs_stds'] = skill_data['predicted_obs_stds'].detach().cpu().numpy()
            while not terminate:
                action = model.act(obs, skill, sample=False)
                terminate = model.get_termination(obs, skill, normalize=model.normalize_inputs)

                skill_data['actions'].append(action)
                skill_data['obs'].append(obs)

                next_state, reward, done, _ = env.step(action)
                total_reward += reward
                n_time_step += 1 # Keep track of steps taken

                if n_time_step >= 280:
                    timeout = True
                if kitchen.check_done(obs, goal_tasks):
                    success = True

                model.increment_skill(skill)

                skill_data['next_state'].append(next_state[:-30] if cfgs['model']['kitchen_remove_goal'] else next_state)
                skill_data['reward'].append(reward)
                skill_data['done'].append(done)

                frame = env.render(mode='rgb_array')
                if render.lower() == 'video':
                    skill_data['frames'].append(frame)

                if success or timeout:
                    break

                obs = next_state
                if cfgs['model']['kitchen_remove_goal']:
                    obs = obs[:-30]

            skill_data['obs'] = np.array(skill_data['obs'])
            skill_data['milestones'].append(obs)

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

        plans += 1

    return success, total_reward, rollouts_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run planning to reach goal')
    # Logging args
    parser.add_argument('-l', '--log_dir', default='./logs', help='logging directory')
    parser.add_argument('-r', '--render', action='store_true', help='render?')
    parser.add_argument('-v', '--video', action='store_true', help='save video?')
    parser.add_argument('-g', '--gif', action='store_true', help='save GIF?')
    parser.add_argument('-p', '--plot', action='store_true', help='plot trajectories?')
    parser.add_argument('-s', '--save_dir', default='./media', help='save path?')
    parser.add_argument('-d', '--device_id', default=0, type=int, help='CUDA device id')
    # Planning args
    parser.add_argument('-b', '--batch_size', default=100, type=int, help='Population size / batch size')
    parser.add_argument('--num_episodes', default=1, type=int)
    parser.add_argument('--plan_length', default=5, type=int)
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
    parser.add_argument('--reward_function_type', default='dense', type=str, help='dense / sparse')
    parser.add_argument('--random_goals', action='store_true')

    args = parser.parse_args()

    if args.render:
        from mujoco_py import GlfwContext
        GlfwContext(offscreen=True)

    config_path = os.path.join(args.log_dir, 'configs.yaml')

    with open(config_path, 'r') as f:
        cfgs = yaml.load(f, Loader=yaml.FullLoader)

    cfgs['device_id'] = args.device_id

    save_dir = os.path.join(args.save_dir, args.log_dir)

    env = gym.make(cfgs['env_name'])

    device = torch.device('cuda:{}'.format(cfgs['device_id']) if torch.cuda.is_available() else 'cpu')

    if cfgs['model']['segmented_trajs_training']:
        train_dataset = SegmentedKitchenDataset(train=True, train_ratio=cfgs['train_dataset_fraction'], remove_goal=cfgs['model']['kitchen_remove_goal'], data_dir=cfgs['data_dir'])
        test_dataset = SegmentedKitchenDataset(train=False, train_ratio=cfgs['train_dataset_fraction'], remove_goal=cfgs['model']['kitchen_remove_goal'], data_dir=cfgs['data_dir'])
        # Load raw numpy data into pytorch dataloaders
        obs = torch.from_numpy(np.stack([ep['observations'] for ep in train_dataset.episodes], axis=0)).to(device)
        if cfgs['model']['kitchen_remove_goal']:
            obs = obs[..., :-30]
        actions = torch.from_numpy(np.stack([ep['actions'] for ep in train_dataset.episodes], axis=0)).to(device)
        masks = torch.from_numpy(np.concatenate([ep['masks'] for ep in train_dataset.episodes], axis=0)).to(device).bool()
        obs_mean = torch.mean(obs.reshape(-1, obs.shape[-1])[masks], dim=0).float()
        obs_std = torch.std(obs.reshape(-1, obs.shape[-1])[masks], dim=0).float()
        actions_mean = torch.mean(actions.reshape(-1, actions.shape[-1])[masks], dim=0).float()
        actions_std = torch.std(actions.reshape(-1, actions.shape[-1])[masks], dim=0).float()
    else:
        train_dataset = KitchenDataset(signal_length=cfgs['data_sub_traj_len'], train=True, train_ratio=cfgs['train_dataset_fraction'], remove_goal=cfgs['model']['kitchen_remove_goal'])
        test_dataset = KitchenDataset(signal_length=cfgs['data_sub_traj_len'], train=False, train_ratio=cfgs['train_dataset_fraction'], remove_goal=cfgs['model']['kitchen_remove_goal'])
        # Load raw numpy data into pytorch dataloaders
        obs = torch.from_numpy(train_dataset.all_obs()).to(device)
        actions = torch.from_numpy(train_dataset.all_actions()).to(device)
        # Compute data statistics
        obs_mean = obs.reshape(-1, obs.shape[-1]).mean(dim=0)
        obs_std = torch.maximum(obs.reshape(-1, obs.shape[-1]).std(dim=0), torch.tensor(1e-6, device=device))
        actions_mean = actions.reshape(-1, actions.shape[-1]).mean(dim=0)
        actions_std = torch.maximum(actions.reshape(-1, actions.shape[-1]).std(dim=0), torch.tensor(1e-6, device=device))

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
                reward_function_type=args.reward_function_type,
                random_goals=args.random_goals,
                )

    experiment_identifier = 'planning' if not args.random_goals else 'random_goal_planning'

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

    total_rewards = 0

    for i in (pbar:=tqdm(range(args.num_episodes), dynamic_ncols=True)):
        planner = cem_planner
        with torch.no_grad():
            success, episode_reward, rollouts_data = run_planning(
                cfgs,
                model,
                planner=planner,
                render=render
            )

        print("Episode reward: ", episode_reward)
        total_rewards += episode_reward / 4 # Normalize by expert score

        if success:
            results['n_success'] += 1
            results['n_time_steps'].append(rollouts_data['n_time_steps'])
            results['wallclock_times'].append(rollouts_data['wallclock_time'])
            # results['init'].append(rollouts_data['init_data'])
            # results['goal'].append(rollouts_data['goal_data'])

        ci = proportion_confint(results['n_success'], i + 1)

        pbar.set_description(f'Success: {results["n_success"]}/{i + 1}; CI: {ci}, Average Score: {total_rewards / (i + 1)}')

        run_dir = os.path.join(save_dir, experiment_identifier, str(i))
        if args.video or args.gif or args.plot:
            os.makedirs(run_dir, exist_ok=True)
        if args.video:
            for run_identifier, run_data in rollouts_data.items():
                # if not isinstance(run_data, Dict):
                if 'execution' not in run_identifier:
                    continue
                #     continue
                # elif 'frames' in run_data.keys():
                frames = assemble_frames(run_data)
                np.save(os.path.join(run_dir, f'{run_identifier}_frames.npy'), frames)
                import ipdb; ipdb.set_trace()
                if len(frames) == 0:
                    continue
                make_video(frames, save_path=os.path.join(run_dir, f'{run_identifier}.mp4'), label=run_identifier)

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
                make_gif(frames, save_path=os.path.join(run_dir, f'{run_identifier}.gif'), label=run_identifier)

        if args.plot:
            execution_data = rollouts_data['execution_data']
            plot_count = 0
            for idx in tqdm(range(len(rollouts_data['plan_data'])), desc='plotting', leave=False, dynamic_ncols=True):
                if idx >= 5:
                    break
                if idx == 0:
                    obs = execution_data[idx]['obs'][:1]
                    completed_pred_obs_means = obs
                    completed_pred_obs_stds = np.zeros_like(obs)
                    completed_skill_times = np.array([0])
                else:
                    obs = np.concatenate([skill['obs'] for skill in execution_data[:idx]], axis=0)
                    completed_pred_obs_means = np.stack([skill['predicted_obs_means'] for skill in execution_data[:idx]], axis=0)
                    completed_pred_obs_stds = np.stack([skill['predicted_obs_stds'] for skill in execution_data[:idx]], axis=0)
                    completed_skill_times = np.cumsum(np.array([len(skill['obs']) for skill in execution_data[:idx]]), axis=0) - 1
                for plan_idx, plan in enumerate(rollouts_data['plan_data'][idx]):
                    plans = plan['all_plans']['milestones']
                    plan_times = np.max(completed_skill_times) + ((np.arange(plans.shape[1]) + 1) * 40)
                    if plot_count % 1 == 0:
                        plot_obj_states_and_terminal_state_predictions_and_plans(
                            obs,
                            completed_pred_obs_means,
                            completed_pred_obs_stds,
                            completed_skill_times,
                            plans,
                            plan_times,
                            os.path.join(run_dir, f'{plot_count}.png')
                        )
                    plot_count += 1

            plot_obj_states_and_terminal_state_predictions(
                np.concatenate([skill['obs'] for skill in execution_data], axis=0),
                np.stack([skill['predicted_obs_means'] for skill in execution_data], axis=0),
                np.stack([skill['predicted_obs_stds'] for skill in execution_data], axis=0),
                np.cumsum(np.array([len(skill['obs']) for skill in execution_data]), axis=0) - 1,
                os.path.join(run_dir, f'plot.png')
            )

    # Save eval results
    save_path = os.path.join(save_dir, experiment_identifier, f'eval_results.pth')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(results, save_path)
