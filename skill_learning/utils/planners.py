import torch

def dummy_planner(data):
    """Selects the first skill and trajectory from the given data."""
    plan = dict()
    skills = data['skills'] # (population, n_skills, skill_dim)
    traj_means = data['milestone_means']
    traj_stds = data['milestone_stds']
    plan.update(skills=skills[0]) # TODO: select skills using CEM
    plan.update(milestone_means=traj_means[0])
    plan.update(milestone_stds=traj_stds[0])
    return plan

def random_planner(cost_fn, planning_args, init_state=None, epsilon_to_z=None, init_action=None):
    """Selects a skill within one std of the the learnt skill prior distribution"""

    skill_eps_means = torch.zeros((planning_args['plan_length'], planning_args['skill_dim']), device=planning_args['device'])
    skill_eps_stds = planning_args['skill_std'] * torch.ones((planning_args['plan_length'], planning_args['skill_dim']), device=planning_args['device'])

    skill_seq = skill_eps_means + skill_eps_stds * torch.randn_like(skill_eps_means)
    skill_seq = torch.clip(skill_seq, -1 * torch.ones_like(skill_seq), 1 * torch.ones_like(skill_seq))

    _, imagination_data = cost_fn(skill_seq.unsqueeze(0))

    proposed_plans = dict(obs=imagination_data['obs'],
                          obs_stds=imagination_data['obs_stds'],
                          milestones=imagination_data['milestones'],
                          milestone_stds=imagination_data['milestone_stds'],)

    plan = dict()
    plan_data = [dict(all_plans=proposed_plans,
                      topk_plans=proposed_plans,)]

    if planning_args['epsilon_planning']:
        skill_seq, selected_plan = epsilon_to_z(init_state, skill_seq, init_action=init_action)
    else:
        raise NotImplementedError("Planning type not implemented")

    plan.update(skills=skill_seq.squeeze())
    plan.update(selected_plan=selected_plan)
    plan.update(plan_data=plan_data)

    return plan

def cem_planner(cost_fn, planning_args, init_state=None, epsilon_to_z=None, init_action=None):
    skill_eps_means = torch.zeros((planning_args['plan_length'], planning_args['skill_dim']), device=planning_args['device'])
    skill_eps_stds = planning_args['skill_std'] * torch.ones((planning_args['plan_length'], planning_args['skill_dim']), device=planning_args['device'])

    skill_seq, skill_seq_stds, plan_data = cem(skill_eps_means, skill_eps_stds, cost_fn, planning_args['population'], planning_args['keep_frac'], planning_args['num_iters'], l2_pen=planning_args['cem_l2_pen'])
    skill_seq = skill_seq[:planning_args['plan_length'], :]

    plan = dict()
    if planning_args['epsilon_planning']:
        skill_seq, selected_plan = epsilon_to_z(init_state, skill_seq, init_action=init_action)
    else:
        raise NotImplementedError("Planning type not implemented")

    plan.update(skills=skill_seq.squeeze(0))
    plan.update(selected_plan=selected_plan)
    plan.update(plan_data=plan_data)
    return plan

def cem_iter(x, cost_fn, frac_keep, l2_pen):
    N = x.shape[0]
    k = max(2, int(N*frac_keep)) # k is for keep y'

    # evaluate solution candidates, get sorted inds
    costs, other_data = cost_fn(x)
    l2_cost = l2_pen * torch.mean(torch.mean(x**2, dim=-1), dim=-1)
    costs += l2_cost
    inds = torch.argsort(costs)
    # figure out which inds to keep
    inds_keep = inds[:k]
    # get best k solution candidates & their average cost
    x_topk = x[inds_keep,...]
    cost_topk = torch.mean(costs[inds_keep])
    # print(costs[inds_keep])
    # take mean and stand dev of new solution population
    x_mean = torch.mean(x_topk, dim=0)
    x_std  = torch.std( x_topk, dim=0)

    topk_plans = dict(obs=other_data['obs'][inds_keep.cpu().numpy()],
                     obs_stds=other_data['obs_stds'][inds_keep.cpu().numpy()],
                     milestones=other_data['milestones'][inds_keep.cpu().numpy()],
                     milestone_stds=other_data['milestone_stds'][inds_keep.cpu().numpy()],)

    other_data = dict(topk_plans=topk_plans,
                      all_plans=other_data,)
    # other_data = dict(topk_plans=dict(),
    #                   all_plans=dict(),)

    return x_mean, x_std, other_data

def cem(x_mean, x_std, cost_fn, pop_size, frac_keep, n_iters, l2_pen, eps_bound_stds=1.0):
    device = x_mean.device
    iter_data = []
    for i in range(n_iters):
        x_shape = [pop_size] + list(x_mean.shape)
        x = x_mean + x_std*torch.randn(x_shape, device=device)
        if eps_bound_stds is not None:
            x = torch.clip(x, -1 * eps_bound_stds, eps_bound_stds)
        x_mean, x_std, other_data = cem_iter(x, cost_fn, frac_keep, l2_pen)
        # print(x_mean, x_std)
        iter_data.append(other_data)

    # Warning: other_data only contains the last iterations data
    return x_mean, x_std, iter_data
