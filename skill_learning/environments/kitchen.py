import numpy as np
import torch

from gym import register
from d4rl.kitchen.kitchen_envs import KitchenBase

OBS_ELEMENT_INDICES = {
    'bottom burner': np.array([11, 12]),
    'top burner': np.array([15, 16]),
    'light switch': np.array([17, 18]),
    'slide cabinet': np.array([19]),
    'hinge cabinet': np.array([20, 21]),
    'microwave': np.array([22]),
    'kettle': np.array([23, 24, 25, 26, 27, 28, 29]),
    }

OBS_ELEMENT_GOALS = {
    'bottom burner': np.array([-0.88, -0.01]),
    'top burner': np.array([-0.92, -0.01]),
    'light switch': np.array([-0.69, -0.05]),
    'slide cabinet': np.array([0.37]),
    'hinge cabinet': np.array([0., 1.45]),
    'microwave': np.array([-0.75]),
    'kettle': np.array([-0.23, 0.75, 1.62, 0.99, 0., 0., -0.06]),
    }

OBS_INDEX_TO_ELEMENT = {
    11 : 'bottom burner',
    12 : 'bottom burner',
    15 : 'top burner',
    16 : 'top burner',
    17 : 'light switch',
    18 : 'light switch',
    19 : 'slide cabinet',
    20 : 'hinge cabinet',
    21 : 'hinge cabinet',
    22 : 'microwave',
    23 : 'kettle',
    24 : 'kettle',
    25 : 'kettle',
    26 : 'kettle',
    27 : 'kettle',
    28 : 'kettle',
    29 : 'kettle',
}

BONUS_THRESH = 0.3

TASKS = ['bottom burner', 'top burner', 'light switch', 'slide cabinet', 'hinge cabinet', 'microwave', 'kettle']
INTS_TO_COMBINATIONS = {
    0: ['bottom burner', 'top burner', 'light switch', 'slide cabinet'],
    1: ['bottom burner', 'top burner', 'light switch', 'hinge cabinet'],
    2: ['bottom burner', 'top burner', 'light switch', 'microwave'],
    3: ['bottom burner', 'top burner', 'light switch', 'kettle'],
    4: ['bottom burner', 'top burner', 'slide cabinet', 'hinge cabinet'],
    5: ['bottom burner', 'top burner', 'slide cabinet', 'microwave'],
    6: ['bottom burner', 'top burner', 'slide cabinet', 'kettle'],
    7: ['bottom burner', 'top burner', 'hinge cabinet', 'microwave'],
    8: ['bottom burner', 'top burner', 'hinge cabinet', 'kettle'],
    9: ['bottom burner', 'top burner', 'microwave', 'kettle'],
    10: ['bottom burner', 'light switch', 'slide cabinet', 'hinge cabinet'],
    11: ['bottom burner', 'light switch', 'slide cabinet', 'microwave'],
    12: ['bottom burner', 'light switch', 'slide cabinet', 'kettle'],
    13: ['bottom burner', 'light switch', 'hinge cabinet', 'microwave'],
    14: ['bottom burner', 'light switch', 'hinge cabinet', 'kettle'],
    15: ['bottom burner', 'light switch', 'microwave', 'kettle'],
    16: ['bottom burner', 'slide cabinet', 'hinge cabinet', 'microwave'],
    17: ['bottom burner', 'slide cabinet', 'hinge cabinet', 'kettle'],
    18: ['bottom burner', 'slide cabinet', 'microwave', 'kettle'],
    19: ['bottom burner', 'hinge cabinet', 'microwave', 'kettle'],
    20: ['top burner', 'light switch', 'slide cabinet', 'hinge cabinet'],
    21: ['top burner', 'light switch', 'slide cabinet', 'microwave'],
    22: ['top burner', 'light switch', 'slide cabinet', 'kettle'],
    23: ['top burner', 'light switch', 'hinge cabinet', 'microwave'],
    24: ['top burner', 'light switch', 'hinge cabinet', 'kettle'],
    25: ['top burner', 'light switch', 'microwave', 'kettle'],
    26: ['top burner', 'slide cabinet', 'hinge cabinet', 'microwave'],
    27: ['top burner', 'slide cabinet', 'hinge cabinet', 'kettle'],
    28: ['top burner', 'slide cabinet', 'microwave', 'kettle'],
    29: ['top burner', 'hinge cabinet', 'microwave', 'kettle'],
    30: ['light switch', 'slide cabinet', 'hinge cabinet', 'microwave'],
    31: ['light switch', 'slide cabinet', 'hinge cabinet', 'kettle'],
    32: ['light switch', 'slide cabinet', 'microwave', 'kettle'],
    33: ['light switch', 'hinge cabinet', 'microwave', 'kettle'],
    34: ['slide cabinet', 'hinge cabinet', 'microwave', 'kettle'],
}
IDX_OFFSET = 9

class KitchenRandom(KitchenBase):
    IDX = 0
    def __init__(self, dataset_url=None, ref_max_score=None, ref_min_score=None, **kwargs):
        self.TASK_ELEMENTS = INTS_TO_COMBINATIONS[KitchenRandom.IDX % len(INTS_TO_COMBINATIONS)]
        KitchenRandom.IDX += 1
        super(KitchenRandom, self).__init__(dataset_url=dataset_url, ref_max_score=ref_max_score, ref_min_score=ref_min_score)

register(
    id='kitchen-random-v0',
    entry_point=KitchenRandom,
    max_episode_steps=280,
    kwargs={
        'ref_min_score': 0.0,
        'ref_max_score': 4.0,
    }
)

def get_num_completions(
        trajectory: torch.Tensor,
        goal_state: list,
    ):
    obj_obs = trajectory[..., 9:9+21]

    completions = torch.zeros_like(obj_obs[..., 0])

    for element in goal_state:
        element_idx = torch.tensor(OBS_ELEMENT_INDICES[element], device=trajectory.device)
        distance = torch.norm(
            obj_obs[..., element_idx - IDX_OFFSET] - torch.tensor(OBS_ELEMENT_GOALS[element], device=trajectory.device),
            dim=-1,
        )
        complete = distance < BONUS_THRESH
        completions += complete

    return completions

def get_all_completions(
        trajectory: torch.Tensor,
    ):
    obj_obs = trajectory[..., 9:9+21]

    completions = torch.zeros_like(obj_obs[..., 0])

    for element in TASKS:
        element_idx = torch.tensor(OBS_ELEMENT_INDICES[element], device=trajectory.device)
        distance = torch.norm(
            obj_obs[..., element_idx - IDX_OFFSET] - torch.tensor(OBS_ELEMENT_GOALS[element], device=trajectory.device),
            dim=-1,
        )
        complete = distance < BONUS_THRESH
        completions += complete

    return completions

def dense_cost_fn(
        trajectory: torch.Tensor,
        goal_state: list,
    ):
    task_completions = get_num_completions(trajectory, goal_state)
    all_completions = get_all_completions(trajectory)

    indices = torch.arange(task_completions.shape[1], device=task_completions.device).unsqueeze(0).expand(task_completions.shape[0], -1) + 1
    complete_indices = indices.clone()
    complete_indices[all_completions < 4] = 1000
    first_4_completions = torch.min(complete_indices, dim=1)[0].unsqueeze(1).expand(-1, task_completions.shape[1])
    keep_indices = indices <= first_4_completions

    task_completions[~keep_indices] = 0
    costs = -task_completions.float()
    timesteps = torch.arange(costs.shape[1], device=costs.device).float().unsqueeze(0) + 1 # (skill_seq_len,)
    costs = costs + timesteps * 0.05 # Add a small cost for each timestep, to prefer reaching the goal faster
    costs = torch.min(costs, dim=1)[0]

    return costs

def sparse_cost_fn(
        trajectory: torch.Tensor,
        goal_state: list,
    ):
    completions = get_num_completions(trajectory, goal_state)

    all_completions = get_all_completions(trajectory)
    indices = torch.arange(all_completions.shape[1], device=all_completions.device).unsqueeze(0).expand(all_completions.shape[0], -1) + 1
    complete_indices = indices.clone()
    complete_indices[all_completions < 4] = 1000
    first_4_completions = torch.min(complete_indices, dim=1)[0].unsqueeze(1).expand(-1, all_completions.shape[1])
    keep_indices = indices <= first_4_completions

    sparse_completions = completions == len(goal_state)
    sparse_completions = -sparse_completions.float()
    sparse_completions[~keep_indices] = 0
    timesteps = torch.arange(sparse_completions.shape[1], device=sparse_completions.device).float().unsqueeze(0) # (skill_seq_len,)
    sparse_completions = sparse_completions + timesteps * 0.05 # Add a small cost for each timestep, to prefer reaching the goal faster
    costs = torch.min(sparse_completions, dim=1)[0]

    return costs

def sparse_random_cost_fn(
        trajectory: torch.Tensor,
        goal_state: list,
    ):
    completions = get_num_completions(trajectory, goal_state)

    all_completions = get_all_completions(trajectory)
    indices = torch.arange(all_completions.shape[1], device=all_completions.device).unsqueeze(0).expand(all_completions.shape[0], -1) + 1
    complete_indices = indices.clone()
    complete_indices[all_completions < 4] = 1000
    first_4_completions = torch.min(complete_indices, dim=1)[0].unsqueeze(1).expand(-1, all_completions.shape[1])
    keep_indices = indices == first_4_completions

    sparse_completions = -completions.float()
    sparse_completions[~keep_indices] = 0
    timesteps = torch.arange(sparse_completions.shape[1], device=sparse_completions.device).float().unsqueeze(0) # (skill_seq_len,)
    sparse_completions = sparse_completions + timesteps * 0.05 # Add a small cost for each timestep, to prefer reaching the goal faster
    costs = torch.min(sparse_completions, dim=1)[0]

    return costs

def check_done(
        obs: np.ndarray,
        goal_state: list,
    ):
    obj_obs = obs[..., 9:9+21]

    completions = 0

    for element in goal_state:
        element_idx = np.array(OBS_ELEMENT_INDICES[element])
        distance = np.linalg.norm(
            obj_obs[..., element_idx - IDX_OFFSET] - np.array(OBS_ELEMENT_GOALS[element]),
        )
        complete = distance < BONUS_THRESH
        completions += complete

    done = completions == len(goal_state)
    return done

def get_completed_tasks(obs):
    final_obs = obs[-1]
    obj_obs = final_obs[9:9+21]

    completions = []
    for task in TASKS:
        element_idx = OBS_ELEMENT_INDICES[task]
        distance = np.linalg.norm(
            obj_obs[element_idx - IDX_OFFSET] - OBS_ELEMENT_GOALS[task],
        )
        complete = distance < BONUS_THRESH
        if complete:
            completions.append(task)

    return completions

if __name__ == '__main__':
    trajs = np.load('../data/raw/kitchen-mixed-v0_segmented.npy', allow_pickle=True)

    task_counts = {}
    for traj in trajs:
        obs = traj['observations']
        completed_tasks = tuple(get_completed_tasks(obs))
        if completed_tasks not in task_counts:
            task_counts[completed_tasks] = 0
        task_counts[completed_tasks] += 1

    for task in task_counts.keys():
        if 'bottom burner' in task and 'top burner' in task and 'microwave' in task and 'hinge cabinet' in task:
            print("THE ONE I'M USING")
        print(task, task_counts[task])

    total = sum(task_counts.values())
    print('total', total)
