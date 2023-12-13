import torch
import torch.nn as nn


class ModelFactory:
    _registry = {}
    @classmethod
    def register_model(cls, name):
        """Registers a model class with the specified name"""
        def _register(wrapped_class):
            if name in cls._registry:
                raise Exception(f'Executor {name} already exists.')
            cls._registry[name] = wrapped_class
            return wrapped_class
        return _register

    @classmethod
    def create_model(cls, name, device='cpu', load_path=None, *args, **kwargs):
        """Creates model, loads weights if load_path is not None, and moves to the specified device"""
        model = cls._create_model(name, *args, **kwargs)
        model.to(device)
        if load_path is not None:
            model.load_state_dict(torch.load(load_path, map_location=device))
        return model

    @classmethod
    def _create_model(cls, name, *args, **kwargs):
        """Creates an instance of the specified model class registeration name"""
        if name not in cls._registry:
            raise ValueError(f'{name} not in {cls._registry.keys()}')
        model = cls._registry[name](*args, **kwargs)
        return model

def separate_statistics(output):
    mean, std_out = output.chunk(2, dim=-1)
    return mean, torch.nn.functional.softplus(std_out)

def get_activation(activation):
    if activation == 'elu':
        return nn.ELU
    elif activation == 'relu':
        return nn.ReLU
    elif activation == 'sigmoid':
        return nn.Sigmoid
    else:
        return NotImplementedError("Activation not implemented yet")

def interpolate_sample_times_for_ode_solver(original_t, new_t):
    # original_t in shape (batch_size, length)
    # new_t in shape (new_length)
    batch_size = original_t.shape[0]
    n_original_samples = original_t.shape[1]
    n_new_samples = new_t.shape[0]
    device = original_t.device

    idxr = torch.arange(n_original_samples, device=device).unsqueeze(dim=0).expand(batch_size, -1).float()

    interpolated_signal = torch.zeros(batch_size, n_new_samples, device=device)

    idx = torch.arange(batch_size)
    indices = torch.arange(n_original_samples, device=device).unsqueeze(0).expand(batch_size, -1).float()

    for _idx, t in enumerate(new_t):
        # Select the start and end points for interpolation
        clipped_t = torch.min(t, torch.max(original_t[idx, :], dim=1).values).unsqueeze(-1)
        start = torch.argmax((idxr)*(original_t <= t), dim=1)
        end = torch.argmax(((idxr[:, -1:] + 1) - idxr)*(original_t >= clipped_t), dim=1)

        dt = original_t[idx, end] - original_t[idx, start]
        alpha = (clipped_t.squeeze() - original_t[idx, start]) / dt

        interpolated_signal[idx, _idx] = indices[idx, start] * (1-alpha) + indices[idx, end] * (alpha)
        interpolated_signal[start>=end, _idx] = indices[start>=end, start[start>=end]]

    return flatten_queries_and_create_masks(interpolated_signal)

def get_interpolated_indices_batch(original_t, new_t):
    batch_size = original_t.shape[0]
    n_original_samples = original_t.shape[1]
    n_new_samples = new_t.shape[1]
    device = original_t.device

    idxr = torch.arange(n_original_samples, device=device).unsqueeze(dim=0).expand(batch_size, -1).float()

    interpolated_signal = torch.zeros(batch_size, n_new_samples, device=device)

    idx = torch.arange(batch_size)
    indices = torch.arange(n_original_samples, device=device).unsqueeze(0).expand(batch_size, -1).float()

    for _idx in range(new_t.shape[1]):
        t = new_t[:, _idx]
        clipped_t = torch.min(t, torch.max(original_t[idx, :], dim=1).values)
        # Select the start and end points for interpolation
        start = torch.argmax((idxr)*(original_t <= t.unsqueeze(dim=-1)), dim=1)
        end = torch.argmax(((idxr[:, -1:] + 1) - idxr)*(original_t >= clipped_t.unsqueeze(dim=-1)), dim=1)

        dt = original_t[idx, end] - original_t[idx, start]

        alpha = (clipped_t.squeeze() - original_t[idx, start]) / torch.maximum(dt, 1e-12 * torch.ones_like(dt))

        interpolated_signal[idx, _idx] = indices[idx, start] * (1-alpha) + indices[idx, end] * (alpha)
        # Overwrite special cases
        interpolated_signal[start>=end, _idx] = indices[start>=end, start[start>=end]]

    return interpolated_signal

def interpolate_sample_times_for_ode_solver_batch(original_t, new_t):
    interpolated_signal = get_interpolated_indices_batch(original_t, new_t)
    return flatten_queries_and_create_masks(interpolated_signal)

def flatten_queries_and_create_masks(query_matrix):
    batch_size = query_matrix.shape[0]
    device = query_matrix.device
    t_all = torch.flatten(query_matrix)
    traj_masks = []
    for i in range(batch_size):
        mask = torch.cat([torch.zeros_like(query_matrix[0]) if j != i else torch.ones_like(query_matrix[0]) for j in range(batch_size)])
        traj_masks.append(mask)

    t_all, indices = t_all.sort()
    traj_masks = [mask[indices] for mask in traj_masks]
    traj_masks = torch.stack(traj_masks).bool()

    t_q, inverse_indices = torch.unique_consecutive(input=torch.round(t_all, decimals=4), return_inverse=True)
    query_indices = torch.empty_like(query_matrix).long()
    indices = torch.arange(inverse_indices.shape[0], device=device)
    for i, mask in enumerate(traj_masks):
        query_indices[i] = inverse_indices[indices[mask]]
    return t_q, query_indices

def get_real_skill_times(data_t, skill_t):
    batch_size = data_t.shape[0]
    n_original_samples = data_t.shape[1]
    n_new_samples = skill_t.shape[0]
    device = skill_t.device
    interpolated_signal = torch.zeros(batch_size, n_new_samples, device=device)

    idx = torch.arange(batch_size)
    indices = torch.arange(n_original_samples, device=device).unsqueeze(0).expand(batch_size, -1).float()

    for _idx, t in enumerate(skill_t):
        # Select the start and end points for interpolation
        clipped_t = torch.min(t, torch.max(data_t[idx, :], dim=1).values).unsqueeze(-1)
        start = torch.argmax((data_t)*(data_t <= t), dim=1)
        end = torch.argmax(((data_t[:, -1:] + 1) - data_t)*(data_t >= clipped_t), dim=1)

        dt = data_t[idx, end] - data_t[idx, start]
        alpha = (clipped_t.squeeze() - data_t[idx, start]) / torch.maximum(dt, 1e-12 * torch.ones_like(dt))

        interpolated_signal[idx, _idx] = indices[idx, start] * (1-alpha) + indices[idx, end] * (alpha)
        interpolated_signal[start==end, _idx] = indices[start==end, start[start==end]]

    return interpolated_signal

class StandardMLP(nn.Module):
    def __init__(self, input_dim, layer_sizes=[400, 400, 400, 400], output_dim=1, activate_last=False, activation='relu', last_activation=None):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layer_sizes = layer_sizes
        self.activation = get_activation(activation)
        if activate_last:
            self.last_activation = self.activation if last_activation is None else get_activation(last_activation)

        if len(layer_sizes) == 0:
            if not activate_last:
                self.network = nn.Linear(input_dim, output_dim)
            else:
                self.network = nn.Sequential(nn.Linear(input_dim, output_dim), self.last_activation())
            return

        layer_list = [nn.Linear(self.input_dim, self.layer_sizes[0]), self.activation()]
        for i in range(len(self.layer_sizes) - 1):
            layer_list.append(nn.Linear(self.layer_sizes[i], self.layer_sizes[i + 1]))
            layer_list.append(self.activation())
        layer_list.append(nn.Linear(self.layer_sizes[-1], output_dim))
        if activate_last:
            layer_list.append(self.last_activation())

        self.network = nn.Sequential(*layer_list)

    def forward(self, x):
        return self.network.forward(x)

class DataBuffer:
    def __init__(self, action_dim, device):
        self.action_dim = action_dim
        self.device = device
        self.hard_reset()

    def reset(self):
        self._obs = []
        if len(self._actions) == 1:
            self._actions = []
        self._actions = self._actions[-1:]

    def hard_reset(self, reset_val=0.0, init_action=None):
        self._obs = []
        if init_action is None:
            self._actions = [torch.tensor([[[reset_val] * self.action_dim]]).to(self.device)]
        else:
            self._actions = [init_action]

    def push(self, obs=None, actions=None):
        if obs is not None:
            if isinstance(obs, list):
                self._obs.extend(obs)
            else:
                self._obs.append(obs)
        if actions is not None:
            if isinstance(actions, list):
                self._actions.extend(actions)
            else:
                self._actions.append(actions)

    @property
    def assembled(self):
        obs = torch.cat(self._obs, dim=1)
        actions = torch.cat(self._actions, dim=1)
        if obs.shape[1] == actions.shape[1]:
            return torch.cat([obs, actions], dim=-1)
        elif obs.shape[1] == actions.shape[1] - 1:
            return torch.cat([obs, actions[:, :-1]], dim=-1)
        else:
            raise Exception(f'Error with data sequences. Obs len: {len(self._obs)}, actions len: {len(self._actions)}')
    @property
    def seq_len(self):
        return self._obs.shape[1]