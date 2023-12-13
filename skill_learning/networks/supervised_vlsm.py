from typing import Dict, List, Tuple

import numpy as np
import torch

from .full_flsm import FLSM
from .infrastructure import ModelFactory, StandardMLP, separate_statistics


@ModelFactory.register_model('svlsm')
class SVLSM(FLSM):
    """Supervised Variable Length Skill Model

    Sequence chunks are provided by an oracle.

    """

    def __init__(self,
                 obs_dim,
                 *args,
                 skill_dim: int=32,
                 noncausal_encoder_rnn_hidden_dim: int=32,
                 skill_posterior_layers: Tuple=(256, 256),
                 termination_decoder_layers: Tuple=(256, 256),
                 termination_loss_coeff: float=1.0,
                 termination_threshold: float=0.5,
                 detach_termination_skills: bool=False,
                 data_specs: Dict=None,
                 **kwargs):

        super().__init__(obs_dim,
                         *args,
                         skill_dim=skill_dim,
                         noncausal_encoder_rnn_hidden_dim=noncausal_encoder_rnn_hidden_dim,
                         skill_posterior_layers=skill_posterior_layers,
                         **kwargs)

        self.noncausal_encoder_rnn_hidden_dim = noncausal_encoder_rnn_hidden_dim

        self.segmented_trajs_training = kwargs['segmented_trajs_training'] if 'segmented_trajs_training' in kwargs else False
        if self.segmented_trajs_training:
            self.forward_gru_cells = torch.nn.ModuleList([torch.nn.GRUCell(input_size=kwargs['encoder_embedding_dim'], hidden_size=noncausal_encoder_rnn_hidden_dim // 2)])
            self.backward_gru_cells = torch.nn.ModuleList([torch.nn.GRUCell(input_size=kwargs['encoder_embedding_dim'], hidden_size=noncausal_encoder_rnn_hidden_dim // 2)])
            self.noncausal_encoder_rnn_num_layers = kwargs['noncausal_encoder_rnn_num_layers']
            for _ in range(self.noncausal_encoder_rnn_num_layers - 1):
                self.forward_gru_cells.append(torch.nn.GRUCell(input_size=noncausal_encoder_rnn_hidden_dim // 2, hidden_size=noncausal_encoder_rnn_hidden_dim // 2))
                self.backward_gru_cells.append(torch.nn.GRUCell(input_size=noncausal_encoder_rnn_hidden_dim // 2, hidden_size=noncausal_encoder_rnn_hidden_dim // 2))

        self.termination_decoder = StandardMLP(input_dim=skill_dim + obs_dim,
                                              layer_sizes=termination_decoder_layers,
                                              output_dim=1,
                                              activate_last=True,
                                              last_activation='sigmoid',)

        self.skill_posterior = StandardMLP(input_dim=noncausal_encoder_rnn_hidden_dim,
                                           layer_sizes=skill_posterior_layers,
                                           output_dim=2*skill_dim,
                                           activate_last=False,)

        self.termination_loss_coeff = termination_loss_coeff
        self.detach_termination_skills = detach_termination_skills
        self.termination_threshold = termination_threshold

        self.train_prior_everywhere = kwargs['train_prior_everywhere'] if 'train_prior_everywhere' in kwargs else False
        self.unnormalize_outputs = kwargs['unnormalize_outputs'] if 'unnormalize_outputs' in kwargs else False
        self.kitchen_remove_goal = kwargs['kitchen_remove_goal'] if 'kitchen_remove_goal' in kwargs else False
        self.normalize_inputs = kwargs['normalize_inputs'] if 'normalize_inputs' in kwargs else True
        self.state_encode_only = kwargs['state_encode_only'] if 'state_encode_only' in kwargs else False
        self.vampprior = kwargs['vampprior'] if 'vampprior' in kwargs else False

        self.termination_loss_ratio = kwargs['termination_loss_ratio'] if 'termination_loss_ratio' in kwargs else None

        if self.vampprior:
            self.vampprior_num_psuedoinputs = kwargs['vampprior_num_pseudoinputs'] if 'vampprior_num_pseudoinputs' in kwargs else -1
            self.improved_estimator = kwargs['improved_estimator'] if 'improved_estimator' in kwargs else False
            self.conditional_psuedoinput_converter = StandardMLP(
                input_dim=obs_dim + kwargs['vampprior_pseudoinput_dim'],
                layer_sizes=kwargs['vampprior_conditional_pseudoinput_converter_layers'],
                output_dim=noncausal_encoder_rnn_hidden_dim,
            )
            self.pseudoinputs = torch.nn.Parameter(torch.randn(kwargs['vampprior_num_pseudoinputs'], kwargs['vampprior_pseudoinput_dim']))


        if self.state_encode_only:
            self.encoder_mlp = StandardMLP(input_dim=obs_dim, layer_sizes=kwargs['encoder_mlp_layers'], output_dim=kwargs['encoder_embedding_dim'])

        if data_specs is not None:
            self.obs_mean = data_specs['obs_mean']
            self.obs_std = data_specs['obs_std']
            self.actions_mean = data_specs['actions_mean']
            self.actions_std = data_specs['actions_std']

    def encode(self,
               obs: torch.Tensor,
               actions: torch.Tensor,
               terminations: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Encode observations and actions into skills"""

        # data is of shape (batch_size, seq_len, obs_dim + action_dim)
        data = torch.cat([obs, actions], dim=-1) # (batch_size, seq_len, data_dim:=obs_dim+action_dim)

        # Pass data through encoder
        if self.state_encode_only:
            embeddings = self.encoder_mlp.forward(obs) # (batch_size, seq_len, encoder_mlp_hidden_dim)
        else:
            embeddings = self.encoder_mlp.forward(data) # (batch_size, seq_len, encoder_mlp_hidden_dim)

        # Encode trajectories with a noncausal RNN
        h_bi_t, _ = self.noncausal_encoder_rnn(embeddings) # (batch_size, seq_len, noncausal_encoder_rnn_hidden_dim)
        h_forward_t = h_bi_t[:, :, :self.noncausal_encoder_rnn_hidden_dim // 2] # (batch_size, seq_len, noncausal_encoder_rnn_hidden_dim // 2)
        h_forward_t_assigned = self.create_execution_skills(h_forward_t, terminations) # (batch_size, seq_len, noncausal_encoder_rnn_hidden_dim // 2)
        h_backward_t = h_bi_t[:, :, self.noncausal_encoder_rnn_hidden_dim // 2:] # (batch_size, seq_len, noncausal_encoder_rnn_hidden_dim // 2)
        shifted_terminations = torch.cat((torch.zeros_like(terminations[:, -1:]), terminations[:, :-1]), dim=1)
        h_backward_t_assigned = self.create_execution_skills(h_backward_t.flip(dims=[1]), shifted_terminations.flip(dims=[1])).flip(dims=[1]) # (batch_size, seq_len, noncausal_encoder_rnn_hidden_dim // 2)
        h_bi_t_assigned = torch.cat((h_forward_t_assigned, h_backward_t_assigned), dim=-1) # (batch_size, seq_len, noncausal_encoder_rnn_hidden_dim)

        # Just compute for all time stamps for now. Only use the masked ones later.
        skill_post_means, skill_post_stds = self.skill_post_func(h_bi_t_assigned) # (batch_size, seq_len, skill_dim)
        if not self.vampprior:
            skill_prior_means, skill_prior_stds = self.skill_prior_func(obs) # (batch_size, seq_len, skill_dim)
        else:
            skill_prior_means, skill_prior_stds = self.skill_post_func(self.conditional_psuedoinput_converter(torch.cat((obs, self.pseudoinputs.unsqueeze(0).unsqueeze(0)), dim=-1))) # (batch_size, seq_len, num_pseudoinputs, skill_dim

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
        execution_skills = skill_samples # (batch_size, seq_len, skill_dim)

        # Decode actions from execution skills and observations
        pred_actions_means, pred_actions_stds = separate_statistics(self.mlp_actions_decoder(torch.cat((execution_skills, gt_obs), dim=-1)))
        pred_actions_stds = torch.clamp(pred_actions_stds, min=self.actions_std_min)
        # pred_action_stds = self.process_stds(pred_action_stds,
        #                                      min=self.actions_std_min,
        #                                      first_min=self.init_actions_std_min,)

        if self.detach_termination_skills:
            termination_skills = execution_skills.detach()
        else:
            termination_skills = execution_skills

        # Decode per time step termination from execution skills and observations
        termination_probs = self.termination_decoder(torch.cat((termination_skills, gt_obs), dim=-1)) # (batch_size, seq_len, 1)

        if self.detach_skills:
            obs_skills = execution_skills.detach()
        else:
            obs_skills = execution_skills

        # Decode end observations from skills
        pred_obs_means, pred_obs_stds = separate_statistics(self.mlp_obs_decoder(torch.cat([obs_skills, gt_obs], dim=-1)))
        pred_obs_stds = torch.clamp(pred_obs_stds, min=self.obs_std_min)
        # pred_obs_stds = self.process_stds(pred_obs_stds,
        #                                      min=self.obs_std_min,
        #                                      first_min=self.init_obs_std_min,)

        return pred_obs_means, pred_obs_stds, pred_actions_means, pred_actions_stds, termination_probs

    def forward(self,
                obs: torch.Tensor,
                actions: torch.Tensor,
                terminations: torch.Tensor=None,
                skills: List[torch.Tensor]=None,
                masks: torch.Tensor=None) -> Tuple[torch.Tensor,
                                                          torch.Tensor,
                                                          torch.Tensor,
                                                          torch.Tensor,
                                                          Dict[str, torch.Tensor]]:
        """Compress data into skill space and uncompress it back"""
        if self.segmented_trajs_training:
            return self.forward_segmented_trajs(obs, actions, masks)

        skills = self.encode(obs, actions, terminations)

        obs_means, obs_stds, actions_means, actions_stds, termination_probs = self.decode(skills, obs)

        latent_data = dict(skills=skills,
                           termination_probs=termination_probs,)

        return obs_means, obs_stds, actions_means, actions_stds, latent_data

    def forward_segmented_trajs(
            self,
            obs: torch.Tensor,
            actions: torch.Tensor,
            masks: torch.Tensor,
    ):
        # Obs of shape (batch_size, seq_len, obs_dim)
        # Actions of shape (batch_size, seq_len, action_dim)
        # Masks of shape (batch_size, seq_len)

        # data is of shape (batch_size, seq_len, obs_dim + action_dim)
        data = torch.cat([obs, actions], dim=-1) # (batch_size, seq_len, data_dim:=obs_dim+action_dim)

        # Pass data through encoder
        if self.state_encode_only:
            embeddings = self.encoder_mlp.forward(obs) # (batch_size, seq_len, encoder_mlp_hidden_dim)
        else:
            embeddings = self.encoder_mlp.forward(data) # (batch_size, seq_len, encoder_mlp_hidden_dim)

        forward_gru_hidden = [torch.zeros((obs.shape[0], self.noncausal_encoder_rnn_hidden_dim // 2)).to(obs.device) for _ in range(self.noncausal_encoder_rnn_num_layers)]
        backward_gru_hidden = [torch.zeros((obs.shape[0], self.noncausal_encoder_rnn_hidden_dim // 2)).to(obs.device) for _ in range(self.noncausal_encoder_rnn_num_layers)]

        broadcastable_masks = masks.unsqueeze(-1) # (batch_size, seq_len, 1)
        for i in range(obs.shape[1]):
            forward_gru_hidden[0] = torch.where(broadcastable_masks[:, i], self.forward_gru_cells[0](embeddings[:, i], forward_gru_hidden[0]), forward_gru_hidden[0])
            backward_gru_hidden[0] = torch.where(broadcastable_masks[:, -(i+1)], self.backward_gru_cells[0](embeddings[:, -(i+1)], backward_gru_hidden[0]), backward_gru_hidden[0])
            for j in range(1, self.noncausal_encoder_rnn_num_layers):
                forward_gru_hidden[j] = torch.where(broadcastable_masks[:, i], self.forward_gru_cells[j](forward_gru_hidden[j-1], forward_gru_hidden[j]), forward_gru_hidden[j])
                backward_gru_hidden[j] = torch.where(broadcastable_masks[:, -(i+1)], self.backward_gru_cells[j](backward_gru_hidden[j-1], backward_gru_hidden[j]), backward_gru_hidden[j])

        gru_hidden = torch.cat((forward_gru_hidden[-1], backward_gru_hidden[-1]), dim=-1) # (batch_size, noncausal_encoder_rnn_hidden_dim)

        skill_post_means, skill_post_stds = self.skill_post_func(gru_hidden) # (batch_size, skill_dim)
        skill_post_means = skill_post_means.unsqueeze(1) # (batch_size, 1, skill_dim)
        skill_post_stds = skill_post_stds.unsqueeze(1) # (batch_size, 1, skill_dim)
        # if skill_post_means.requires_grad:
        #     skill_post_means.register_hook(lambda x: print('skill_post_means grad:', x.isnan().any()))
        #     skill_post_stds.register_hook(lambda x: print('skill_post_stds grad:', x.isnan().any()))
        if not self.vampprior:
            skill_prior_means, skill_prior_stds = self.skill_prior_func(obs) # (batch_size, seq_len, skill_dim)
        else:
            broadcast_obs = obs.unsqueeze(-2).expand(-1, -1, self.vampprior_num_psuedoinputs, -1)
            broadcast_psuedoinputs = self.pseudoinputs.unsqueeze(0).unsqueeze(0).expand(obs.shape[0], obs.shape[1], -1, -1)
            skill_prior_means, skill_prior_stds = self.skill_post_func(self.conditional_psuedoinput_converter(torch.cat((broadcast_obs, broadcast_psuedoinputs), dim=-1))) # (batch_size, seq_len, num_pseudoinputs, skill_dim
            # if skill_prior_means.requires_grad:
            #     skill_prior_means.register_hook(lambda x: print('skill_prior_means grad:', x.isnan().any()))
            #     skill_post_stds.register_hook(lambda x: print('skill_prior_stds grad:', x.isnan().any()))

        skill_sample = self.rsample(skill_post_means, skill_post_stds)

        broadcasted_skill_sample = skill_sample.repeat(1, obs.shape[1], 1) # (batch_size, seq_len, skill_dim)

        pred_action_means, pred_action_stds = separate_statistics(self.mlp_actions_decoder(torch.cat((broadcasted_skill_sample, obs), dim=-1)))
        pred_action_stds = torch.clamp(pred_action_stds, min=self.actions_std_min)

        if self.detach_termination_skills:
            termination_skills = broadcasted_skill_sample.detach()
        else:
            termination_skills = broadcasted_skill_sample

        termination_probs = self.termination_decoder(torch.cat((termination_skills, obs), dim=-1)) # (batch_size, seq_len, 1)

        if self.detach_skills:
            obs_skills = broadcasted_skill_sample.detach()
        else:
            obs_skills = broadcasted_skill_sample
        # Shift obs_skills back one time step, fill in last time step with zeros
        obs_skills = torch.cat((obs_skills[:, :-1], torch.zeros_like(obs_skills[:, :1])), dim=1)

        pred_obs_means, pred_obs_stds = separate_statistics(self.mlp_obs_decoder(torch.cat([obs_skills, obs], dim=-1)))
        pred_obs_stds = torch.clamp(pred_obs_stds, min=self.obs_std_min)

        latent_data = dict(
            skill_post_means=skill_post_means,
            skill_post_stds=skill_post_stds,
            skill_prior_means=skill_prior_means,
            skill_prior_stds=skill_prior_stds,
            skill_sample=skill_sample,
            termination_probs=termination_probs,
        )

        return pred_obs_means, pred_obs_stds, pred_action_means, pred_action_stds, latent_data

    def get_loss_segmented_trajs(
            self,
            obs: torch.Tensor,
            actions: torch.Tensor,
            masks: torch.Tensor,
    ):
        pred_obs_means, pred_obs_stds, pred_actions_means, pred_actions_stds, latent_data = self.forward_segmented_trajs(obs, actions, masks)

        batch_size, seq_len, _ = obs.shape

        skill_post_dist = torch.distributions.Normal(loc=latent_data['skill_post_means'], scale=latent_data['skill_post_stds'])
        skill_prior_dist = torch.distributions.Normal(loc=latent_data['skill_prior_means'], scale=latent_data['skill_prior_stds'])

        sum_masks = masks.sum(dim=1)
        if self.train_prior_everywhere:
            non_terminal_mask = masks.clone()
            non_terminal_mask[torch.arange(batch_size), sum_masks - 1] = 0
            masks_for_prior = non_terminal_mask
        else:
            masks_for_prior = torch.zeros_like(masks)
            masks_for_prior[:, 0] = 1

        if not self.vampprior:
            skill_kl_loss = torch.mean(torch.sum(torch.distributions.kl_divergence(skill_post_dist, skill_prior_dist).sum(dim=-1) * masks_for_prior, dim=1) / masks_for_prior.sum(dim=1))
        else:
            log_q = skill_post_dist.log_prob(latent_data['skill_sample']).sum(dim=-1)
            unsqueezed_samples = latent_data['skill_sample'].unsqueeze(2)
            log_p_components = skill_prior_dist.log_prob(unsqueezed_samples).sum(dim=-1) - torch.log(torch.tensor(self.vampprior_num_psuedoinputs, dtype=torch.float32, device=unsqueezed_samples.device))
            max_component_p = log_p_components.max(dim=-1)[0]

            # log sum exp trick
            log_p = max_component_p + torch.log(torch.exp(log_p_components - max_component_p.unsqueeze(-1)).sum(dim=-1))

            if self.improved_estimator:
                log_r = log_q - log_p
                r = torch.exp(log_r)
                estimator = (r - 1) - torch.log(r)
                skill_kl_loss = torch.mean(torch.sum(estimator * masks_for_prior, dim=1) / masks_for_prior.sum(dim=1))
            else:
                skill_kl_loss = torch.mean(torch.sum((log_q - log_p) * masks_for_prior, dim=1) / masks_for_prior.sum(dim=1))

        if self.unnormalize_outputs and self.normalize_inputs:
            obs = self.unnormalize(obs, self.obs_mean, self.obs_std)
            actions = self.unnormalize(actions, self.actions_mean, self.actions_std)

        obs_dist = torch.distributions.Normal(loc=pred_obs_means, scale=pred_obs_stds)
        actions_dist = torch.distributions.Normal(loc=pred_actions_means, scale=pred_actions_stds)

        # Only predict last masked obs
        last_obs_broadcast = obs[torch.arange(batch_size), sum_masks - 1].unsqueeze(1).repeat(1, seq_len, 1)
        obs_nll_loss = -torch.mean(torch.sum(obs_dist.log_prob(last_obs_broadcast).sum(dim=-1) * masks_for_prior, dim=1) / masks_for_prior.sum(dim=1))
        obs_mse_loss = torch.mean(torch.sum(((obs_dist.mean - last_obs_broadcast) ** 2).sum(dim=-1) * masks_for_prior, dim=1) / masks_for_prior.sum(dim=1))
        obs_reconstruction_loss = obs_nll_loss

        actions_nll_loss = -torch.mean(torch.sum(actions_dist.log_prob(actions).sum(dim=-1) * masks, dim=1) / masks.sum(dim=1))
        actions_mse_loss = torch.mean(torch.sum(((actions_dist.mean - actions) ** 2).sum(dim=-1) * masks, dim=1) / masks.sum(dim=1))
        actions_reconstruction_loss = actions_nll_loss

        reconstruction_loss = self.obs_loss_coeff * obs_reconstruction_loss + self.actions_loss_coeff * actions_reconstruction_loss
        reconstruction_mse_loss = self.obs_loss_coeff * obs_mse_loss + self.actions_loss_coeff * actions_mse_loss

        # Termination prediction loss
        terminations = torch.zeros_like(masks).float()
        terminations[torch.arange(batch_size), sum_masks - 1] = 1
        termination_loss_weights = self.get_termination_loss_weights(terminations, masks, positive_ratio=self.termination_loss_ratio).unsqueeze(dim=-1)
        pred_termination_probs = latent_data['termination_probs'] # (batch_size, seq_len, 1)
        termination_loss = torch.nn.BCELoss(reduction='none')(pred_termination_probs, terminations.unsqueeze(dim=-1)) * termination_loss_weights
        termination_loss = torch.mean(termination_loss.sum(dim=-1).sum(dim=-1) / masks_for_prior.sum(dim=-1))

        # ELBO
        elbo_loss = reconstruction_loss + self.skill_beta * skill_kl_loss

        # Total loss
        loss = elbo_loss + self.termination_loss_coeff * termination_loss

        # Compute useful metrics
        pred_probs = pred_termination_probs.squeeze(dim=-1)
        termination_tp = ((pred_probs >  self.termination_threshold) &  terminations.bool())[masks].sum()
        termination_tn = ((pred_probs <= self.termination_threshold) & ~terminations.bool())[masks].sum()
        termination_fp = ((pred_probs >  self.termination_threshold) & ~terminations.bool())[masks].sum()
        termination_fn = ((pred_probs <= self.termination_threshold) &  terminations.bool())[masks].sum()

        termination_accuracy = (termination_tp + termination_tn) / masks.sum()
        termination_precision = termination_tp / (termination_tp + termination_fp)
        termination_recall = termination_tp / (termination_tp + termination_fn)
        termination_f1 = 2 * (termination_precision * termination_recall) / (termination_precision + termination_recall)
        termination_specificity = termination_tn / (termination_tn + termination_fp)

        metrics = dict(loss=loss,
                       elbo=elbo_loss,
                       reconstruction_loss=reconstruction_loss,
                       skill_kl_loss=skill_kl_loss,
                       termination_loss=termination_loss,

                       termination_precision=termination_precision,
                       termination_accuracy=termination_accuracy,
                       termination_recall=termination_recall,
                       termination_f1=termination_f1,
                       termination_specificity=termination_specificity,

                       obs_reconstruction_nll_loss=obs_nll_loss,
                       actions_reconstruction_nll_loss=actions_nll_loss,
                       actions_reconstruction_loss=actions_reconstruction_loss,
                       obs_reconstruction_loss=obs_reconstruction_loss,
                       reconstruction_mse_loss=reconstruction_mse_loss,
                       actions_reconstruction_mse_loss=actions_mse_loss,
                       obs_reconstruction_mse_loss=obs_mse_loss,)

        info = dict()

        return loss, metrics, info

    def get_loss(self,
                 _obs: torch.Tensor,
                 _actions: torch.Tensor,
                 _terminations: torch.Tensor=None,
                 _masks: torch.Tensor=None) -> Tuple[torch.Tensor,
                                                 Dict[str, torch.Tensor],
                                                 Dict[str, torch.Tensor],]:

        if self.segmented_trajs_training:
            return self.get_loss_segmented_trajs(_obs, _actions, _masks)

        # Train only on terminated trajectories
        mask = self.get_terminated_mask(_terminations)
        obs = _obs[mask]
        actions = _actions[mask]
        terminations = _terminations[mask]

        end_obs_means, end_obs_stds, actions_means, actions_stds, latent_data = self.forward(obs, actions, terminations)

        # While computing loss, we will multiply everything by the mask (terminations)

        batch_size, seq_len, _ = obs.shape

        # Skill KL divergence
        skill_data = latent_data['skills']
        skill_post_means = skill_data['post_mean']
        skill_post_stds = skill_data['post_std']
        skill_prior_means = skill_data['prior_mean']
        skill_prior_stds = skill_data['prior_std']

        skill_post_dist = torch.distributions.normal.Normal(skill_post_means[:, 1:], skill_post_stds[:, 1:]) # We don't need the zeroth posterior because that should be regularized to the negative first prior
        skill_prior_dist = torch.distributions.normal.Normal(skill_prior_means[:, :-1], skill_prior_stds[:, :-1]) # We don't need the last prior because that should be regularized to the posterior after the last posterior

        # Don't train on the last termination because we don't have the posterior after the last prior
        indices = torch.arange(seq_len, device=self.device).unsqueeze(dim=0).expand(batch_size, -1)
        masked_indices = indices * terminations.long()
        max_start_indices = torch.max(masked_indices, dim=-1)[0]
        true_indices = indices < max_start_indices.unsqueeze(dim=-1)
        masked_terminations = terminations * true_indices.float()
        kl_normalizing_factor = (masked_terminations[:, :-1] * (1 / torch.maximum(masked_terminations[:, :-1].sum(dim=-1).unsqueeze(dim=-1), torch.ones_like(masked_terminations[:, :-1].sum(dim=-1).unsqueeze(dim=-1))))).detach() # calculate kl only at termination events and normalize it by the number of termination events
        if self.train_prior_everywhere:
            sum_true_indices = torch.sum(true_indices, dim=-1)
            sum_true_indices = torch.clamp(sum_true_indices, min=1)
            skill_kl_loss = torch.mean((torch.distributions.kl_divergence(skill_post_dist, skill_prior_dist).sum(dim=-1) * true_indices[:, :-1]).sum(dim=-1) / torch.clamp(sum_true_indices - true_indices[:, -1].float(), min=1))
        else:
            skill_kl_loss = (torch.distributions.kl_divergence(skill_post_dist, skill_prior_dist).sum(dim=-1) * kl_normalizing_factor).sum() / batch_size

        # Reconstruction loss
        # NLL observation reconstruction
        end_obs_gt = self.create_execution_skills(obs, terminations)
        if self.unnormalize_outputs and self.normalize_inputs:
            obs = self.unnormalize(obs, self.obs_mean, self.obs_std)
        if self.train_prior_everywhere:
            end_obs_dist = torch.distributions.normal.Normal(end_obs_means, end_obs_stds)
            # Mask that has 0 on the last index where termination is 1 and after
            end_obs_reconstruction_mse_loss = torch.mean((torch.nn.functional.mse_loss(end_obs_means, end_obs_gt, reduction='none').sum(dim=-1) * true_indices).sum(dim=-1) / sum_true_indices)
            end_obs_reconstruction_nll_loss = -torch.mean((end_obs_dist.log_prob(end_obs_gt).sum(dim=-1) * true_indices).sum(dim=-1) / sum_true_indices)
            end_obs_reconstruction_loss = end_obs_reconstruction_nll_loss
        else:
            end_obs_reconstruction_mse_loss = torch.mean((torch.nn.functional.mse_loss(end_obs_means[:, :-1], end_obs_gt[:, 1:], reduction='none').sum(dim=-1) * kl_normalizing_factor).sum(dim=-1))
            end_obs_reconstruction_nll_loss = -torch.mean((torch.distributions.normal.Normal(end_obs_means[:, :-1], end_obs_stds[:, :-1]).log_prob(end_obs_gt[:, 1:]).sum(dim=-1) * kl_normalizing_factor).sum(dim=-1))
            end_obs_reconstruction_loss = end_obs_reconstruction_nll_loss

        # NLL action reconstruction
        if self.unnormalize_outputs and self.normalize_inputs:
            actions = self.unnormalize(actions, self.actions_mean, self.actions_std)
        actions_dist = torch.distributions.normal.Normal(actions_means, actions_stds)
        actions_reconstruction_mse_loss = torch.nn.functional.mse_loss(actions_means, actions) / (batch_size * seq_len)
        actions_reconstruction_nll_loss = -torch.sum(actions_dist.log_prob(actions)) / (batch_size * seq_len)
        actions_reconstruction_loss = actions_reconstruction_nll_loss

        reconstruction_loss = self.obs_loss_coeff * end_obs_reconstruction_loss + self.actions_loss_coeff * actions_reconstruction_loss
        reconstruction_mse_loss = self.obs_loss_coeff * end_obs_reconstruction_mse_loss + self.actions_loss_coeff * actions_reconstruction_mse_loss

        # Termination prediction loss
        termination_loss_weights = self.get_termination_loss_weights(terminations).unsqueeze(dim=-1)
        pred_termination_probs = latent_data['termination_probs'] # (batch_size, seq_len, 1)
        termination_loss = torch.nn.BCELoss(reduction='none')(pred_termination_probs, terminations.unsqueeze(dim=-1)) * termination_loss_weights
        termination_loss = termination_loss.sum() / (batch_size * seq_len)

        # ELBO
        elbo_loss = reconstruction_loss + self.skill_beta * skill_kl_loss

        # Total loss
        loss = elbo_loss + self.termination_loss_coeff * termination_loss

        # Compute useful metrics
        pred_probs = pred_termination_probs.squeeze(dim=-1)
        termination_tp = ((pred_probs >  self.termination_threshold) &  terminations.bool()).sum()
        termination_tn = ((pred_probs <= self.termination_threshold) & ~terminations.bool()).sum()
        termination_fp = ((pred_probs >  self.termination_threshold) & ~terminations.bool()).sum()
        termination_fn = ((pred_probs <= self.termination_threshold) &  terminations.bool()).sum()

        termination_accuracy = (termination_tp + termination_tn) / (batch_size * seq_len)
        termination_precision = termination_tp / (termination_tp + termination_fp)
        termination_recall = termination_tp / (termination_tp + termination_fn)
        termination_f1 = 2 * (termination_precision * termination_recall) / (termination_precision + termination_recall)
        termination_specificity = termination_tn / (termination_tn + termination_fp)

        metrics = dict(loss=loss,
                       elbo=elbo_loss,
                       reconstruction_loss=reconstruction_loss,
                       skill_kl_loss=skill_kl_loss,
                       termination_loss=termination_loss,

                       termination_precision=termination_precision,
                       termination_accuracy=termination_accuracy,
                       termination_recall=termination_recall,
                       termination_f1=termination_f1,
                       termination_specificity=termination_specificity,

                       obs_reconstruction_nll_loss=end_obs_reconstruction_nll_loss,
                       actions_reconstruction_nll_loss=actions_reconstruction_nll_loss,
                       actions_reconstruction_loss=actions_reconstruction_loss,
                       obs_reconstruction_loss=end_obs_reconstruction_loss,
                       reconstruction_mse_loss=reconstruction_mse_loss,
                       actions_reconstruction_mse_loss=actions_reconstruction_mse_loss,
                       obs_reconstruction_mse_loss=end_obs_reconstruction_mse_loss,)

        info = dict()

        return loss, metrics, info

    def get_termination(self,
                        raw_obs: np.ndarray,
                        skill: torch.Tensor,
                        normalize: bool=True) -> np.ndarray:

        if self.skill_length != 0:
            return self.skill_steps >= self.skill_length
        obs = self.to_tensor(raw_obs).unsqueeze(dim=0)
        if normalize:
            if not hasattr(self, 'obs_mean'): raise Exception('Data specs not set during init')
            obs = self.normalize(obs, self.obs_mean, self.obs_std)
        assert obs.ndim == 2

        termination_probs = self.termination_decoder(torch.cat([skill.unsqueeze(dim=0), obs], dim=-1))
        # print("termination_probs", termination_probs)
        terminate = termination_probs > self.termination_threshold

        return terminate.detach().cpu().numpy()

    def get_skill_times(self, termination_probs: torch.Tensor) -> torch.Tensor:
        terminations = (termination_probs > self.termination_threshold).to(torch.int)
        indexes = torch.arange(termination_probs.shape[1], device=termination_probs.device).unsqueeze(dim=0).expand(terminations.shape[0], -1).unsqueeze(dim=-1)
        _raw_sl = indexes * terminations

        _raw_sl = _raw_sl[terminations != 0]
        return _raw_sl

    def compute_obs_loss(self,
                         obs_means: torch.Tensor,
                         obs_stds: torch.Tensor,
                         obs: torch.Tensor,
                         terminations: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        start_idx_mask, end_idx_mask = self.get_idxs_mask(terminations)

        total_nll_loss = 0
        total_mse_loss = 0

        batch_size, seq_len, _ = obs_means.shape

        for i, (slice_means, slice_stds, slice_gts) in enumerate(zip(obs_means, obs_stds, obs)):
            slice_start_idx_mask = start_idx_mask[i]
            slice_end_idx_mask = end_idx_mask[i]

            n_skills = slice_start_idx_mask.sum()

            obs_dist = torch.distributions.normal.Normal(slice_means[slice_start_idx_mask], slice_stds[slice_start_idx_mask])

            slice_nll_loss = -obs_dist.log_prob(slice_gts[slice_end_idx_mask])
            slice_mse_loss = torch.nn.functional.mse_loss(slice_means[slice_start_idx_mask], slice_gts[slice_end_idx_mask], reduction='none')

            total_nll_loss += torch.sum(slice_nll_loss) / n_skills
            total_mse_loss += slice_mse_loss.sum()/ n_skills

        total_nll_loss /= batch_size
        total_mse_loss /= batch_size

        return total_nll_loss, total_mse_loss

    def init_skill(self, skill):
        self.skill_steps = 0

    def increment_skill(self, skill):
        self.skill_steps += 1

    def decode_obs(self, obs, skill, normalize=True):
        if normalize:
            obs_ = self.normalize(self.to_tensor(obs), self.obs_mean, self.obs_std)
        else:
            obs_ = self.to_tensor(obs)
        ips = torch.cat([skill, obs_], dim=-1)
        pred_obs_means, pred_obs_stds = separate_statistics(self.mlp_obs_decoder(ips))
        if not self.unnormalize_outputs:
            pred_obs_means = self.unnormalize(pred_obs_means, self.obs_mean, self.obs_std)
            # TODO: unnormalize stds
        return pred_obs_means, pred_obs_stds

    def decode_action(self, obs, skill, normalize=True, sample=False):
        if self.kitchen_remove_goal:
            obs = obs[..., :-30]
        if normalize:
            obs_ = self.normalize(self.to_tensor(obs), self.obs_mean, self.obs_std)
        else:
            obs_ = self.to_tensor(obs)
        ips = torch.cat([skill, obs_], dim=-1)
        pred_action_means, pred_action_stds = separate_statistics(self.mlp_actions_decoder(ips))
        if not self.unnormalize_outputs:
            pred_action_means = self.unnormalize(pred_action_means, self.actions_mean, self.actions_std)
        if sample:
            return pred_action_means + pred_action_stds * torch.randn_like(pred_action_stds)
        else:
            return pred_action_means

    def decode_termination(self, obs, skill, normalize=True):
        if self.skill_length != 0:
            return self.skill_steps >= self.skill_length
        if self.kitchen_remove_goal:
            obs = obs[..., :-30]
        if normalize:
            obs_ = self.normalize(self.to_tensor(obs), self.obs_mean, self.obs_std)
        else:
            obs_ = self.to_tensor(obs)
        ips = torch.cat([skill, obs_], dim=-1)
        pred_termination_probs = self.termination_decoder(ips)
        # print('pred_termination_probs', pred_termination_probs)
        terminate = pred_termination_probs > self.termination_threshold
        return terminate.squeeze().item()

    @staticmethod
    def process_stds(stds: torch.Tensor,
                     min: float=None,
                     max: float=None,
                     init_min: float=None,
                     init_max: float=None,) -> torch.Tensor:
        stds = torch.clamp(stds, min=min, max=max)
        stds[..., 0] = torch.clamp(stds[..., 0], min=init_min, max=init_max)
        return stds

    @staticmethod
    def get_termination_loss_weights(terminations: torch.Tensor, masks: torch.Tensor, positive_ratio=0.5):
        """Generate weights such that there is ~ 1:1 ratio of positive to negative samples"""
        if positive_ratio == None:
            return torch.clone(masks).float() / masks.sum(dim=-1, keepdim=True)
        n_positives = (terminations * masks).sum(dim=-1)
        n_negatives = masks.sum(dim=-1) - n_positives
        n_positives = torch.ones_like(terminations) * n_positives.unsqueeze(dim=-1)
        n_negatives = torch.ones_like(terminations) * n_negatives.unsqueeze(dim=-1)
        weights = torch.zeros_like(terminations)
        weights[torch.logical_and(terminations.bool(), masks)] = positive_ratio / n_positives[torch.logical_and(terminations.bool(), masks)]
        weights[torch.logical_and(~terminations.bool(), masks)] = (1 - positive_ratio) / n_negatives[torch.logical_and(~terminations.bool(), masks)]
        weights = weights / weights.sum(dim=-1, keepdim=True)
        return weights.detach()

    @staticmethod
    def create_execution_skills(skills: torch.Tensor,
                                raw_terminations: torch.Tensor) -> torch.Tensor:
        execution_skills = []

        terminations = raw_terminations.clone()
        terminations[..., -1] = 1

        indexes = torch.arange(skills.shape[1], device=skills.device) + 1

        get_non_zero = lambda x: x[x != 0]

        for i, (skill_slice, terminations_slice) in enumerate(zip(skills, terminations)):
            unprocessed_sl = get_non_zero(indexes * terminations_slice)

            skill_lens = torch.cat([unprocessed_sl[:1], unprocessed_sl[1:] - unprocessed_sl[:-1]])
            skill_samples = skill_slice[terminations_slice.bool()]
            execution_skills_slice = torch.repeat_interleave(skill_samples, skill_lens.int(), dim=0)
            execution_skills.append(execution_skills_slice)

        return torch.stack(execution_skills, dim=0)

    @staticmethod
    def get_idxs_mask(terminations: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        end_idx_mask = terminations.clone().to(torch.bool)
        batch_size, seq_len = terminations.shape
        device = terminations.device

        start_idx_mask = torch.zeros((batch_size, seq_len + 1), device=terminations.device).to(torch.bool)

        def next_idx_func(_x, max_idx=terminations.shape[1]):
            x = _x.clone()
            x1, x2 = torch.where(x)
            return (x1, torch.clamp(x2+1, 0, max_idx))

        start_idx_mask[next_idx_func(terminations)] = 1
        start_idx_mask[:, 0] = 1

        # Remove the last start idx, since the corresponding skill won't be terminated
        last_starts = start_idx_mask * torch.arange(seq_len + 1, device=device).unsqueeze(dim=0).repeat(batch_size, 1)
        start_idx_mask[torch.where(last_starts == last_starts.max(dim=1)[0].unsqueeze(dim=1))] = False

        start_idx_mask = start_idx_mask[:, :-1]

        # Sanity check that all skills are terminated
        assert (start_idx_mask.sum(dim=-1) == end_idx_mask.sum(dim=-1)).all()

        return start_idx_mask, end_idx_mask

    @staticmethod
    def get_terminated_mask(mask: torch.Tensor) -> torch.Tensor:
        assert mask.ndim == 2 or (mask.ndim == 3 and mask.shape[-1] == 1)
        if mask.ndim == 3:
            raw = mask.squeeze(dim=-1)
        else:
            raw = mask
        non_terminated = raw.sum(dim=-1) != 0
        return non_terminated
