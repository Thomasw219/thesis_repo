import torch

def cost_fn(
        trajectory: torch.Tensor,
        goal_state: tuple,
    ):
        goal_state = torch.tensor(goal_state, device=trajectory.device).unsqueeze(0).unsqueeze(0)
        costs = torch.norm(trajectory[..., :2] - goal_state[..., :2], dim=-1) # (batch_size, skill_seq_len)

        # Here starts cost function modifications to make planning more robust
        costs = torch.maximum(costs, torch.ones_like(costs) * 0.5) # Anything inside the goal region is 0.5
        timesteps = torch.arange(costs.shape[1], device=costs.device).float().unsqueeze(0) # (skill_seq_len,)
        costs = costs + timesteps * 0.05 # Add a small cost for each timestep, to prefer reaching the goal faster
        # Here ends cost function modifications

        costs, _ = torch.min(costs, dim=1) # (batch_size,)
        return costs

def sparse_cost_fn(
        trajectory: torch.Tensor,
        goal_state: tuple,
    ):
        goal_state = torch.tensor(goal_state, device=trajectory.device).unsqueeze(0).unsqueeze(0)
        costs = torch.norm(trajectory[..., :2] - goal_state[..., :2], dim=-1) # (batch_size, skill_seq_len)

        costs = torch.where(costs < 0.5, torch.zeros_like(costs), torch.ones_like(costs)) # Anything inside goal region is cost 0, anything outside is cost 1
        timesteps = torch.arange(costs.shape[1], device=costs.device).float().unsqueeze(0) # (skill_seq_len,)
        costs = costs + timesteps * 0.05 # Add a small cost for each timestep, to prefer reaching the goal faster

        # Minimum cost over all timesteps
        costs, _ = torch.min(costs, dim=1) # (batch_size,)
        return costs