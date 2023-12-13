import os
import random
import time

import numpy as np
import torch
import yaml

beauty = lambda x: x.detach().cpu().squeeze().numpy()

_DATA = dict()

def get_skill_lens(real_skill_times):
    skill_times = real_skill_times
    skill_start_times = torch.roll(skill_times, 1, 1)
    skill_start_times[:, 0] = 0
    skill_lengths = skill_times - skill_start_times
    return skill_lengths

def to_single_level_dict(d):
    """Converts a nested dictionary to a single level dictionary. Assumes no repition of keys."""
    res = {}
    for k, v in d.items():
        if isinstance(v, dict):
            res.update(to_single_level_dict(v))
        else:
            res[k] = v
    return res

def register_data_generator(name):
    def _register_data_generator(func):
        _DATA[name] = func
        return func
    return _register_data_generator

def generate_data(name, *args, **kwargs):
    if name not in _DATA:
        raise ValueError(f'{name} not in {_DATA.keys()}')
    return _DATA[name](*args, **kwargs)

def assemble_frames(ip_dicts):
    frames = []
    for d in ip_dicts:
        frames.extend(d.get('frames', []))
    return frames

def assemble_dicts(ip_dicts):
    obs = []
    milestones = []
    for d in ip_dicts:
        obs.append(d.get('obs', None))
        milestones.append(d.get('milestones', None))
    if len(obs) == 0:
        return dict(traj=None, milstones=None)
    return dict(traj=np.concatenate(obs, axis=0), milestones=np.concatenate(milestones, axis=0))

def create_traj_data(data):
    op_data = {}
    for k, v in data.items():
        if isinstance(v, dict):
            op_data[k] = dict(traj=v.get('obs', None), milestones=v.get('milestones', None))
    return op_data

def timeit_wrapper(func):
    def wrap_func(*args, **kwargs):
        t1 = time.time()
        result = func(*args, **kwargs)
        t2 = time.time()
        return (t2 - t1), result
    return wrap_func

def load_cfg(path):
    with open(path, 'r') as f:
        cfgs = yaml.load(f, Loader=yaml.FullLoader)
    return cfgs

def multi_trial_load_cfg(path):
    potential_paths = [f'{path}',
                       os.path.join(path, 'config.yaml'),
                       os.path.join(path, 'configs.yaml'),
                       os.path.join('logs', path),
                       os.path.join('logs', path, 'config.yaml'),
                       os.path.join('logs', path, 'configs.yaml'),]
    for p in potential_paths:
        try: cfgs = load_cfg(p)
        except FileNotFoundError: continue
        return cfgs
    raise FileNotFoundError(f'Could not load config using path {path}')

def compare_dicts(d1, d2):
    if isinstance(d1, str):
        d1 = multi_trial_load_cfg(d1)
    if isinstance(d2, str):
        d2 = multi_trial_load_cfg(d2)

    for k, v in d1.items():
        if k in d2.keys():
            if isinstance(v, dict):
                compare_dicts(v, d2[k])
            elif v != d2[k]:
                print(f'{k}: {v} != {d2[k]}')
    # Print missing keys
    for k in d1.keys():
        if k not in d2.keys():
            print(f'{k} not in d2')
    for k in d2.keys():
        if k not in d1.keys():
            print(f'{k} not in d1')

def handle_keyboard_interrupt(signum, frame):
    res = input("Ctrl-c was pressed. Do you really want to exit? [y/n] ")
    if res.lower() == 'y':
        exit(1)

def seed_all(seed_val: int):
    """Seeds all random number generators."""
    torch.manual_seed(seed_val)
    np.random.seed(seed_val)
    random.seed(seed_val)
