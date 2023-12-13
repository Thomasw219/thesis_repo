from typing import Dict, List, Tuple

import numpy as np
import torch

from .infrastructure import ModelFactory, StandardMLP, separate_statistics


class SkillModelBase(torch.nn.Module):
    """Defines shared methods for FLSMs"""
    # TODO: fix error in selecting the correct skill when creating the execution skill vector

    def to_tensor(self, x):
        if isinstance(x, torch.Tensor):
            tensor = x.to(self.device)
        elif isinstance(x, np.ndarray):
            tensor = torch.from_numpy(x).float().to(self.device)
        else:
            tensor = torch.tensor(x).float().to(self.device)
        return tensor

    @property
    def device(self):
        return next(self.parameters()).device

    @staticmethod
    def rsample(mean, std):
        eps = torch.randn_like(mean)
        samples = eps * std + mean
        return samples

    @staticmethod
    def normalize(x, mu, std):
        return (x - mu) / std

    @staticmethod
    def unnormalize(x, mu, std):
        return (x * std + mu)


@ModelFactory.register_model('full_flsm')
class FLSM(SkillModelBase):
    """Fixed Length Skill Model

        Model similar to Ben's FLSM, only using entire trajectory instead of chunks.

    """

    def __init__(self,
                 obs_dim,
                 action_dim,
                 encoder_embedding_dim=32,
                 encoder_mlp_layers=(256, 256),
                 causal_encoder_rnn_hidden_dim=32,
                 noncausal_encoder_rnn_hidden_dim=32,
                 noncausal_encoder_rnn_num_layers=1,
                 skill_dim=32,
                 skill_length=0,
                 skill_prior_layers=(256, 256),
                 skill_posterior_layers=(256, 256),
                 obs_decoder_layers=(256, 256,),
                 actions_decoder_layers=(256, 256,),
                 detach_skills=True,
                 skill_beta=1,
                 obs_loss_coeff=1,
                 actions_loss_coeff=1,
                 khush_loss_weight=0.0,
                 std_min=0.1,
                 obs_std_min=None,
                 actions_std_min=None,
                 init_obs_std_min=0.003,
                 init_actions_std_min=None,
                 data_specs=None,
                 **kwargs):

        super().__init__()

        self.encoder_mlp = StandardMLP(input_dim=obs_dim + action_dim,
                                       layer_sizes=encoder_mlp_layers,
                                       output_dim=encoder_embedding_dim)

        self.noncausal_encoder_rnn = torch.nn.GRU(input_size=encoder_embedding_dim,
                                                 hidden_size=noncausal_encoder_rnn_hidden_dim // 2,
                                                 batch_first=True,
                                                 bidirectional=True,
                                                 num_layers=noncausal_encoder_rnn_num_layers,)

        self.skill_prior = StandardMLP(input_dim=obs_dim,
                                       layer_sizes=skill_prior_layers,
                                       output_dim=2*skill_dim,
                                       activate_last=False,)

        self.skill_posterior = StandardMLP(input_dim=noncausal_encoder_rnn_hidden_dim + obs_dim,
                                           layer_sizes=skill_posterior_layers,
                                           output_dim=2*skill_dim,
                                           activate_last=False,)

        self.skill_prior_func = lambda x: separate_statistics(self.skill_prior(x))
        self.skill_post_func = lambda x: separate_statistics(self.skill_posterior(x))

        self.mlp_obs_decoder = StandardMLP(input_dim=skill_dim + obs_dim,
                                           layer_sizes=obs_decoder_layers,
                                           output_dim=obs_dim*2,
                                           activate_last=False,)

        self.mlp_actions_decoder = StandardMLP(input_dim=skill_dim + obs_dim,
                                               layer_sizes=actions_decoder_layers,
                                               output_dim=action_dim*2,
                                               activate_last=False,)

        # Save required parameters
        self.obs_dim = obs_dim
        self.skill_dim = skill_dim
        self.skill_length = skill_length
        self.action_dim = action_dim

        self.obs_loss_coeff = obs_loss_coeff
        self.actions_loss_coeff = actions_loss_coeff
        self.khush_loss_weight = khush_loss_weight

        self.skill_beta = skill_beta
        self.detach_skills = detach_skills

        self.obs_std_min = obs_std_min if obs_std_min is not None else std_min
        self.actions_std_min = actions_std_min if actions_std_min is not None else std_min

        self.init_obs_std_min = init_obs_std_min
        self.init_actions_std_min = init_actions_std_min

        if data_specs is not None:
            self.obs_mean = data_specs['obs_mean']
            self.obs_std = data_specs['obs_std']
            self.actions_mean = data_specs['actions_mean']
            self.actions_std = data_specs['actions_std']

    def encode(self,
               obs: torch.Tensor,
               actions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Encode observations and actions into skills"""

        # data is of shape (batch_size, seq_len, obs_dim + action_dim)
        data = torch.cat([obs, actions], dim=-1) # (batch_size, seq_len, data_dim:=obs_dim+action_dim)

        device = data.device
        batch_size, seq_len, data_dim = data.shape

        skill_start_idxs = torch.arange(0, seq_len, self.skill_length, device=device) # (n_samples,)

        # Pass data through encoder
        embeddings = self.encoder_mlp.forward(data) # (batch_size, seq_len, encoder_mlp_hidden_dim)

        # Encode trajectories with a noncausal RNN
        h_bi_t, _ = self.noncausal_encoder_rnn(embeddings) # (batch_size, seq_len, noncausal_encoder_rnn_hidden_dim)

        # (batch, n_samples, noncausal_encoder_rnn_hidden_dim)
        h_bi_bar_i = h_bi_t[:, skill_start_idxs, :]

        start_obs = obs[:, skill_start_idxs, :]

        skill_post_means, skill_post_stds = self.skill_post_func(torch.cat([h_bi_bar_i, start_obs], dim=-1))
        skill_prior_means, skill_prior_stds = self.skill_prior_func(start_obs)

        skill_samples = self.rsample(skill_post_means, skill_post_stds)

        skills = dict(prior_mean=skill_prior_means,
                      prior_std=skill_prior_stds,
                      post_mean=skill_post_means,
                      post_std=skill_post_stds,
                      sample=skill_samples,)

        return skills

    def decode(self,
               skill_samples: Dict[str, torch.Tensor],
               gt_obs: torch.Tensor) -> Tuple[torch.Tensor,
                                              torch.Tensor,
                                              torch.Tensor,
                                              torch.Tensor]:

        """Decode actions from skills and observations and end observations from skills"""

        skill_samples = skill_samples['sample']

        execution_skills = torch.repeat_interleave(skill_samples, self.skill_length, dim=1)

        # Decode actions from execution skills and observations
        pred_actions_means, pred_actions_stds = separate_statistics(self.mlp_actions_decoder(torch.cat((execution_skills, gt_obs), dim=-1)))

        if self.detach_skills:
            obs_skills = skill_samples.detach()
        else:
            obs_skills = skill_samples

        # Decode end observations from skills
        pred_obs_means, pred_obs_stds = separate_statistics(self.mlp_obs_decoder(obs_skills))

        # obs_stds_min = torch.ones_like(pred_obs_stds) * self.obs_std_min
        # if self.init_obs_std_min is not None:
        #     obs_stds_min[..., :1, :2] = self.init_obs_std_min
        # actions_stds_min = torch.ones_like(actions_stds) * self.actions_std_min
        # if self.init_actions_std_min is not None:
        #     actions_stds_min[..., :1, :2] = self.init_actions_std_min

        # pred_obs_stds = torch.maximum(pred_obs_stds, obs_stds_min)
        # pred_actions_stds = torch.maximum(actions_stds, actions_stds_min)

        # return pred_obs_means, pred_obs_stds, actions_means, pred_actions_stds
        return pred_obs_means, pred_obs_stds, pred_actions_means, pred_actions_stds

    def forward(self,
               obs: torch.Tensor,
               actions: torch.Tensor,
               skills: List[torch.Tensor]=None) -> Tuple[torch.Tensor,
                                                         torch.Tensor,
                                                         torch.Tensor,
                                                         torch.Tensor,
                                                         Dict[str, torch.Tensor]]:
        """Compress data into skill space and uncompress it back

        ((batch_size, seq_len, obs_dim), (batch_size, seq_len, action_dim)) -> (batch_size, n_samples, skill_dim)

        """

        # Compress the sequence to extract skills and skill lengths
        skills = self.encode(obs, actions)

        # Uncompress observations and actions back from skills
        obs_means, obs_stds, actions_means, actions_stds = self.decode(skills, obs)

        latent_data = dict(skills=skills)

        return obs_means, obs_stds, actions_means, actions_stds, latent_data

    def get_loss(self,
                 obs: torch.Tensor,
                 actions: torch.Tensor) -> Tuple[torch.Tensor,
                                                 Dict[str, torch.Tensor],
                                                 Dict[str, torch.Tensor],]:

        end_obs_means, end_obs_stds, actions_means, actions_stds, latent_data = self.forward(obs, actions)

        batch_size, seq_len, _ = obs.shape
        n_samples = seq_len // self.skill_length

        skill_start_idxs = torch.arange(0, seq_len, self.skill_length, device=self.device) # (n_samples,)
        skill_end_idxs = torch.arange(self.skill_length - 1, seq_len, self.skill_length, device=self.device) # (n_samples,)

        end_obs = obs[:, skill_end_idxs, :]

        # Skill KL divergence
        skill_data = latent_data['skills']
        skill_post_means = skill_data['post_mean']
        skill_post_stds = skill_data['post_std']
        skill_prior_means = skill_data['prior_mean']
        skill_prior_stds = skill_data['prior_std']

        skill_post_dist = torch.distributions.Independent(torch.distributions.Normal(skill_post_means, skill_post_stds), 1)
        skill_prior_dist = torch.distributions.Independent(torch.distributions.Normal(skill_prior_means, skill_prior_stds), 1)

        skill_kl_loss = torch.distributions.kl_divergence(skill_post_dist, skill_prior_dist).mean()

        # Reconstruction loss
        # NLL observation reconstruction; trim the last observation in case it crosses gt
        end_obs_dist = torch.distributions.Independent(torch.distributions.Normal(end_obs_means, end_obs_stds), 1)
        end_obs_reconstruction_mse_loss = torch.nn.functional.mse_loss(end_obs_means[:, :n_samples], end_obs[:, :n_samples])
        end_obs_reconstruction_nll_loss = -end_obs_dist.log_prob(end_obs).mean()
        end_obs_reconstruction_loss = end_obs_reconstruction_nll_loss

        # NLL action reconstruction
        actions_dist = torch.distributions.Independent(torch.distributions.Normal(actions_means, actions_stds), 1)
        actions_reconstruction_mse_loss = torch.nn.functional.mse_loss(actions_means, actions)
        actions_reconstruction_nll_loss = -actions_dist.log_prob(actions).mean()
        actions_reconstruction_loss = actions_reconstruction_nll_loss

        reconstruction_loss = self.obs_loss_coeff * end_obs_reconstruction_loss + self.actions_loss_coeff * actions_reconstruction_loss
        reconstruction_mse_loss = self.obs_loss_coeff * end_obs_reconstruction_mse_loss + self.actions_loss_coeff * actions_reconstruction_mse_loss

        # Total loss
        loss = reconstruction_loss + self.skill_beta * skill_kl_loss

        metrics = dict(loss=loss,
                       reconstruction_loss=reconstruction_loss,
                       skill_kl_loss=skill_kl_loss,

                       obs_reconstruction_nll_loss=end_obs_reconstruction_nll_loss,
                       actions_reconstruction_nll_loss=actions_reconstruction_nll_loss,
                       actions_reconstruction_loss=actions_reconstruction_loss,
                       obs_reconstruction_loss=end_obs_reconstruction_loss,
                       reconstruction_mse_loss=reconstruction_mse_loss,
                       actions_reconstruction_mse_loss=actions_reconstruction_mse_loss,
                       obs_reconstruction_mse_loss=end_obs_reconstruction_mse_loss,)

        info = dict()

        return loss, metrics, info

    def imagine(self,
                obs_raw: np.ndarray,
                skills: List[torch.Tensor],
                sample: bool=True, # Just to be consistent with the interface; was used to sample in dreamer state
                epsilon_sampling: bool=True,
                use_skills: bool=True,) -> Dict[str, torch.Tensor]:
        """Simulate (imagine) given / randomly generated skills execution end points (observations)"""

        if use_skills:
            if isinstance(skills, list):
                n_skills = len(skills)
                batch_size = skills[0].shape[0]
            else:
                batch_size, n_skills, _ = skills.shape

        if self.normalize_inputs:
            obs = self.normalize(self.to_tensor(obs_raw), self.obs_mean, self.obs_std).unsqueeze(0)
        else:
            obs = self.to_tensor(obs_raw).unsqueeze(0)
        if not obs.shape[0] == batch_size:
            obs = obs.repeat(batch_size, 1)

        assert obs.shape == (batch_size, self.obs_dim)

        data = dict(
            milestone_means=[],
            milestone_stds=[],
            skills=[],
        )

        for i in range(n_skills):

            # Planning / posterior execution
            if use_skills:

                if skills is not None:
                    if isinstance(skills, list):
                        skill_eps = skills[i]
                    else:
                        skill_eps = skills[:, i, :]

                if epsilon_sampling:
                    prior_skills_means, prior_skills_stds = self.skill_prior_func(obs)
                    skill_samples = prior_skills_means + prior_skills_stds * skill_eps
                else:
                    skill_samples = skill_eps

            # Imagination condition
            else:
                prior_skills_means, prior_skills_stds = self.skill_prior_func(obs)
                skill_samples = prior_skills_means + prior_skills_stds * skill_eps

            end_obs_means, end_obs_stds = separate_statistics(self.mlp_obs_decoder(torch.cat([skill_samples, obs], dim=-1)))
            end_obs_stds = torch.clamp(end_obs_stds, min=self.obs_std_min)

            end_obs_stds = torch.clamp(end_obs_stds, min=self.obs_std_min)
            if sample:
                if self.unnormalize_outputs and self.normalize_inputs:
                    obs = self.normalize(end_obs_means + end_obs_stds * torch.randn_like(end_obs_stds), self.obs_mean, self.obs_std)
                else:
                    obs = end_obs_means + end_obs_stds * torch.randn_like(end_obs_stds)
            else:
                if self.unnormalize_outputs and self.normalize_inputs:
                    obs = self.normalize(end_obs_means, self.obs_mean, self.obs_std)
                else:
                    obs = end_obs_means

            assert skill_samples.shape == (batch_size, self.skill_dim)
            assert end_obs_means.shape == (batch_size, self.obs_dim)

            # Update the imagination data
            data['milestone_means'].append(end_obs_means)
            data['milestone_stds'].append(end_obs_stds)
            data['skills'].append(skill_samples)

        # Assemble data
        data = {k: torch.stack(v, dim=1) for k, v in data.items()} # all values are (batch_size, n_skills, *)

        return data

    def epsilon_to_z(self,
                     init_state: np.ndarray,
                     skills: torch.Tensor,
                     init_action: torch.Tensor=None) -> Tuple[torch.Tensor,
                                                              Dict[str, np.ndarray]]:
        """Converts epsilon to z for a given initial state and skill"""

        data = self.imagine(init_state, skills=skills.unsqueeze(0), use_skills=True, epsilon_sampling=True)

        if self.unnormalize_outputs:
            post_process = lambda x: x.detach().cpu().numpy().squeeze()
        else:
            post_process = lambda x: self.unnormalize(x, self.obs_mean, self.obs_std).detach().cpu().numpy().squeeze()


        other_data = dict(milestones=post_process(data['milestone_means']),
                          milestone_stds=post_process(data['milestone_stds']),)

        return data['skills'], other_data

    def get_expected_costs(self,
                           init_state: np.ndarray,
                           goal_state: np.ndarray,
                           skills: torch.Tensor,
                           cost_fn,
                           epsilon_planning: bool=True,
                           use_skills: bool=True,
                           sample: bool=False,) -> Tuple[torch.Tensor,
                                                         Dict[str, np.ndarray]]:
        """Calculate expected costs of executing proposed skill plans"""

        init_state = self.to_tensor(init_state)
        if self.unnormalize_outputs:
            goal_state = self.to_tensor(goal_state) if isinstance(goal_state, np.ndarray) else goal_state
        else:
            goal_state = self.normalize(self.to_tensor(goal_state), self.obs_mean[..., :2], self.obs_std[..., :2]) if isinstance(goal_state, np.ndarray) else goal_state

        # Imagine trajectory using the decoder
        data = self.imagine(init_state, skills, sample=sample, epsilon_sampling=epsilon_planning, use_skills=use_skills)
        # data = self.imagine(init_state, skills, sample=False, epsilon_sampling=epsilon_planning, use_skills=use_skills)

        # Calculate cost from predicted terminal states
        terminal_state_means = data['milestone_means']
        terminal_state_stds = data['milestone_stds']

        terminal_states = terminal_state_means
        # TODO: remove this hack
        trajectory = data['milestone_means']
        # trajectory = data['milestone_means']

        assert trajectory.ndim == 3

        # if sample:
        #     terminal_states += terminal_state_stds * torch.randn_like(terminal_state_means)

        costs = cost_fn(trajectory, goal_state)

        if self.unnormalize_outputs:
            post_process = lambda x: x.detach().cpu().numpy()
        else:
            post_process = lambda x: self.unnormalize(x, self.obs_mean, self.obs_std).detach().cpu().numpy()

        other_data = dict(obs=post_process(data['milestone_means']), # This method doesn't produce all intermediate env states
                          obs_stds=post_process(data['milestone_stds']),
                          milestones=post_process(data['milestone_means']),
                          milestone_stds=post_process(data['milestone_stds']),)

        return costs, other_data

    def act(self,
            obs: np.ndarray,
            skill: torch.Tensor,
            sample: bool=False,) -> np.ndarray:
        """Decodes the action from the observation and skill."""

        if self.normalize_inputs:
            obs_ip = self.normalize(self.to_tensor(obs), self.obs_mean, self.obs_std)
        else:
            obs_ip = self.to_tensor(obs)

        actions_means, actions_stds = separate_statistics(self.mlp_actions_decoder(torch.cat((skill, obs_ip), dim=-1)))
        actions_stds = torch.clamp(actions_stds, min=self.actions_std_min)

        if sample:
            action_sample = self.rsample(actions_means, actions_stds)
        else:
            action_sample = actions_means

        if self.unnormalize_outputs:
            return action_sample.detach().cpu().numpy()
        else:
            return self.unnormalize(action_sample, self.actions_mean, self.actions_std).detach().cpu().numpy()
