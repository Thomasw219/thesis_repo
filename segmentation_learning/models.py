from collections import deque
import copy

import numpy as np
import torch
import torch.nn as nn

import concrete
import rpr

class FullPrototypeModel(nn.Module):
    def __init__(self, cfg, data_dim, max_seq_len):
        super().__init__()
        self.cfg = cfg
        self.data_dim = data_dim
        self.max_seq_len = max_seq_len
        self.sample = False

        self.temperature = cfg.init_temperature
        self.time_loss_weight = cfg.time_loss_weight
        self.state_kl_weight = cfg.state_transition_kl_weight

        self.encoder = StandardMLP(input_dim=data_dim, **cfg.encoder_params, output_dim=cfg.encoding_dim)

        self.segmentation_mlp_encoder = StandardMLP(input_dim=cfg.encoding_dim, **cfg.segmentation_mlp_encoder_params, output_dim=cfg.segmentation_transformer_dim)
        segmentation_transformer_encoder_layer = rpr.TransformerEncoderLayerRPR(d_model=cfg.segmentation_transformer_dim, **cfg.segmentation_transformer_encoder_layer_params, er_len=max_seq_len)
        self.segmentation_transformer_encoder = rpr.TransformerEncoderRPR(segmentation_transformer_encoder_layer, **cfg.segmentation_transformer_encoder_params)
        self.segmentation_gru = nn.GRUCell(cfg.segmentation_transformer_dim + 1, cfg.segmentation_transformer_dim)
        self.segmentation_post = StandardMLP(input_dim=cfg.segmentation_transformer_dim, **cfg.segmentation_post_params, output_dim=1)

        self.compression_mlp_encoder = StandardMLP(input_dim=cfg.encoding_dim + cfg.segmentation_transformer_dim, **cfg.compression_mlp_encoder_params, output_dim=cfg.temporal_attention_dim)
        # self.compression_mlp_encoder = StandardMLP(input_dim=cfg.encoding_dim + 1, **cfg.compression_mlp_encoder_params, output_dim=cfg.compression_transformer_dim)
        # self.compression_transformer_encoder_layer = rpr.TransformerEncoderLayerRPR(d_model=cfg.compression_transformer_dim, **cfg.compression_transformer_encoder_layer_params, er_len=max_seq_len)
        # self.compression_transfomer = rpr.TransformerEncoderRPR(self.compression_transformer_encoder_layer, **cfg.compression_transformer_params)
        # self.compression_mlp_decoder = StandardMLP(input_dim=cfg.compression_transformer_dim, **cfg.compression_mlp_decoder_params, output_dim=cfg.temporal_attention_dim)
        self.abstract_rep_post = StandardMLP(input_dim=cfg.temporal_attention_dim, **cfg.abstract_rep_post_params, output_dim=cfg.abstract_rep_stoch_dim * 2)

        self.abstract_rep_mlp_encoder = StandardMLP(input_dim=cfg.abstract_rep_stoch_dim, **cfg.abstract_rep_mlp_encoder_params, output_dim=cfg.abstract_rep_transformer_dim)
        abstract_rep_transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=cfg.abstract_rep_transformer_dim, **cfg.abstract_rep_transformer_encoder_layer_params)
        self.abstract_rep_transformer_encoder = nn.TransformerEncoder(abstract_rep_transformer_encoder_layer, **cfg.abstract_rep_transformer_params)
        self.abstract_rep_transformer_nheads = cfg.abstract_rep_transformer_encoder_layer_params['nhead']
        self.abstract_rep_mlp_decoder = StandardMLP(input_dim=cfg.abstract_rep_transformer_dim, **cfg.abstract_rep_mlp_decoder_params, output_dim=cfg.abstract_rep_deter_dim)
        self.abstract_rep_prior = StandardMLP(input_dim=cfg.abstract_rep_deter_dim, **cfg.abstract_rep_prior_params, output_dim=cfg.abstract_rep_stoch_dim * 2)
        self.abstract_rep_dim = cfg.abstract_rep_stoch_dim + cfg.abstract_rep_deter_dim
        # self.abstract_rep_prior = StandardMLP(input_dim=cfg.abstract_rep_stoch_dim, **cfg.abstract_rep_prior_params, output_dim=cfg.abstract_rep_stoch_dim * 2)
        # self.abstract_rep_dim = cfg.abstract_rep_stoch_dim

        self.state_rep_post = StandardMLP(input_dim=cfg.encoding_dim + self.abstract_rep_dim, **cfg.state_rep_post_params, output_dim=cfg.state_rep_stoch_dim * 2)
        self.state_rep_mlp_encoder = StandardMLP(input_dim=cfg.state_rep_stoch_dim  + self.abstract_rep_dim, **cfg.state_rep_mlp_encoder_params, output_dim=cfg.state_rep_transformer_dim)
        state_rep_transformer_encoder_layer = rpr.TransformerEncoderLayerRPR(d_model=cfg.state_rep_transformer_dim, **cfg.state_rep_transformer_encoder_layer_params, er_len=max_seq_len)
        self.state_rep_transformer_encoder = rpr.TransformerEncoderRPR(state_rep_transformer_encoder_layer, **cfg.state_rep_transformer_params)
        self.state_rep_transformer_nheads = cfg.state_rep_transformer_encoder_layer_params['nhead']
        self.state_rep_mlp_decoder = StandardMLP(input_dim=cfg.state_rep_transformer_dim, **cfg.state_rep_mlp_decoder_params, output_dim=cfg.state_rep_deter_dim)
        self.state_rep_prior = StandardMLP(input_dim=cfg.state_rep_deter_dim + self.abstract_rep_dim, **cfg.state_rep_prior_params, output_dim=cfg.state_rep_stoch_dim * 2)
        self.state_rep_dim = cfg.state_rep_stoch_dim + cfg.state_rep_deter_dim

        self.segmentation_prior = StandardMLP(input_dim=self.state_rep_dim, **cfg.segmentation_prior_params, output_dim=1)

        self.decoder = StandardMLP(input_dim=self.state_rep_dim, **cfg.decoder_params, output_dim=data_dim)

    def forward(self, traj, abstract_sample_std_scalar=1.0, state_sample_std_scalar=1.0):
        # traj is a tensor of shape (batch_size, seq_len, data_dim)
        batch_size = traj.shape[0]
        seq_len = traj.shape[1]
        device = traj.device
        assert seq_len <= self.max_seq_len, "Trajectories must be less than length {}".format(self.seq_len)
        encodings = self.encoder(traj)
        broadcast_positional_encoding = self.positional_encoding_dropout(self.positional_encoding[:, :traj.shape[1]].expand(batch_size, -1, -1))

        segmentation_encodings = self.segmentation_mlp_encoder(encodings)
        transformed_segmentation_encodings = self.segmentation_transformer_encoder(torch.transpose(segmentation_encodings, 0, 1)).transpose(0, 1) # TRANSPOSE FOR RPR TRANSFORMER
        segmentation_post_logits = []
        segmentation_post_probs = [torch.ones(batch_size, 1, device=device, dtype=torch.float32)]
        segmentation_samples = [torch.ones(batch_size, 1, device=device, dtype=torch.float32)]
        y_samples = []
        gru_hidden = torch.zeros(batch_size, self.cfg.segmentation_transformer_dim, device=device, dtype=torch.float32)
        for i in range(1, seq_len):
            gru_hidden = self.segmentation_gru(torch.cat([segmentation_post_probs[-1], transformed_segmentation_encodings[:, i, :]], dim=-1), gru_hidden)
            segmentation_post_logit = self.segmentation_post(gru_hidden)
            segmentation_sample, y_sample = concrete.sample_binary_concrete(segmentation_post_logit, self.temperature, hard=self.sample)
            segmentation_post_probs.append(torch.sigmoid(segmentation_post_logit))
            segmentation_post_logits.append(segmentation_post_logit)
            if self.cfg['fix_segmentation_period'] is None:
                segmentation_samples.append(segmentation_sample)
            else:
                if i % self.cfg['fix_segmentation_period'] == 0:
                    segmentation_samples.append(torch.ones_like(segmentation_sample))
                else:
                    segmentation_samples.append(torch.zeros_like(segmentation_sample))
            y_samples.append(y_sample)

        segmentation_post_logits = torch.stack(segmentation_post_logits, dim=1)
        segmentation_samples = torch.stack(segmentation_samples, dim=1)
        y_samples = torch.stack(y_samples, dim=1)

        segmentation_samples = segmentation_samples.squeeze(-1)
        if segmentation_samples.requires_grad:
            segmentation_samples.register_hook(lambda grad: self.cfg.time_grad_scalar * grad)
        # segmentation_samples = torch.sigmoid(segmentation_logits)[..., 0]
        # segmentation_samples = torch.bernoulli(segmentation_samples) + segmentation_samples - segmentation_samples.detach()
        # segmentation_samples = torch.cat([torch.ones_like(segmentation_samples[:, :1]), segmentation_samples], dim=1)
        if segmentation_samples.requires_grad:
            segmentation_samples.retain_grad()
        segment_weights, _, causal_segmentation_attention_mask, abstract_causal_segmentation_attention_mask = self.get_segmentation_attention_masks_probabilistic(segmentation_samples)

        query_encodings = self.query_mlp_encoder(torch.cat([encodings, broadcast_positional_encoding], dim=-1))
        repeated_query_encodings = torch.cat([query_encodings] * seq_len, dim=1).reshape(batch_size, seq_len, seq_len, self.cfg.query_attention_dim)
        attended_query_encodings = torch.sum(segment_weights.unsqueeze(-1) * repeated_query_encodings, dim=2)
        abstract_rep_post_params = self.abstract_rep_post(attended_query_encodings)
        abstract_rep_post_means, abstract_rep_post_stds = abstract_rep_post_params[..., :self.cfg.abstract_rep_stoch_dim], nn.functional.softplus(abstract_rep_post_params[..., self.cfg.abstract_rep_stoch_dim:])
        abstract_rep_stoch_samples = self.reparameterize_segments(abstract_rep_post_means, abstract_rep_post_stds, segmentation_samples, std_scalar=abstract_sample_std_scalar)

        # abstract_rep_encodings = self.abstract_rep_mlp_encoder(torch.cat([abstract_rep_stoch_samples, broadcast_positional_encoding], dim=-1))
        # transformed_abstract_rep_encodings = self.abstract_rep_transformer_encoder(abstract_rep_stoch_samples, abstract_rep_encodings, mask=abstract_causal_segmentation_attention_mask)
        # abstract_rep_deter = self.abstract_rep_mlp_decoder(transformed_abstract_rep_encodings)
        # abstract_rep_prior_params = self.abstract_rep_prior(shift_forward(abstract_rep_deter, 1))
        # abstract_rep_prior_means, abstract_rep_prior_stds = abstract_rep_prior_params[..., :self.cfg.abstract_rep_stoch_dim], nn.functional.softplus(abstract_rep_prior_params[..., self.cfg.abstract_rep_stoch_dim:])
        # abstract_rep = torch.cat([abstract_rep_stoch_samples, abstract_rep_deter], dim=-1)
        abstract_rep_prior_params = self.abstract_rep_prior(shift_forward(abstract_rep_stoch_samples, 1))
        abstract_rep_prior_means, abstract_rep_prior_stds = abstract_rep_prior_params[..., :self.cfg.abstract_rep_stoch_dim], nn.functional.softplus(abstract_rep_prior_params[..., self.cfg.abstract_rep_stoch_dim:])
        abstract_rep = abstract_rep_stoch_samples

        state_rep_post_params = self.state_rep_post(torch.cat([encodings, abstract_rep], dim=-1))
        state_rep_post_means, state_rep_post_stds = state_rep_post_params[..., :self.cfg.state_rep_stoch_dim], nn.functional.softplus(state_rep_post_params[..., self.cfg.state_rep_stoch_dim:])
        state_rep_stoch_samples = self.reparameterize(state_rep_post_means, state_rep_post_stds, std_scalar=state_sample_std_scalar)

        state_rep_encodings = self.state_rep_mlp_encoder(torch.cat([state_rep_stoch_samples, abstract_rep], dim=-1))
        transformed_state_rep_encodings = self.state_rep_transformer_encoder(torch.transpose(state_rep_encodings, 0, 1), mask=causal_segmentation_attention_mask).transpose(0, 1) # TRANSPOSE FOR RPR TRANSFORMER
        state_rep_deter = self.state_rep_mlp_decoder(transformed_state_rep_encodings)
        state_rep_prior_params = self.state_rep_prior(torch.cat([shift_forward(state_rep_deter, 1) * (1 - segmentation_samples).unsqueeze(-1), abstract_rep], dim=-1))
        state_rep_prior_means, state_rep_prior_stds = state_rep_prior_params[..., :self.cfg.state_rep_stoch_dim], nn.functional.softplus(state_rep_prior_params[..., self.cfg.state_rep_stoch_dim:])
        state_rep = torch.cat([state_rep_stoch_samples, state_rep_deter], dim=-1)

        # state_rep = torch.zeros_like(state_rep)
        decoder_input = torch.cat([state_rep, abstract_rep], dim=-1)
        segmentation_prior_logits = self.segmentation_prior(decoder_input)[:, :-1]
        reconstructed_traj = self.decoder(decoder_input)

        return reconstructed_traj, dict(
            segment_weights=segment_weights,
            segmentation_post_logits=segmentation_post_logits,
            segmentation_samples=segmentation_samples,
            y_samples=y_samples,
            abstract_rep_post_means=abstract_rep_post_means,
            abstract_rep_post_stds=abstract_rep_post_stds,
            abstract_rep_prior_means=abstract_rep_prior_means,
            abstract_rep_prior_stds=abstract_rep_prior_stds,
            abstract_rep=abstract_rep,
            state_rep_post_means=state_rep_post_means,
            state_rep_post_stds=state_rep_post_stds,
            state_rep_prior_means=state_rep_prior_means,
            state_rep_prior_stds=state_rep_prior_stds,
            state_rep=state_rep,
            segmentation_prior_logits=segmentation_prior_logits,
        )

    def get_loss(self, traj):
        reconstructed_traj, info = self.forward(traj)
        info['reconstructed_traj'] = reconstructed_traj
        info['ground_truth_traj'] = traj
        reconstruction_loss = nn.functional.mse_loss(traj, reconstructed_traj)
        segmentation_samples = info['segmentation_samples']
        average_compression = 1 / torch.mean(segmentation_samples[:, 1:])

        abstract_rep_post_means, abstract_rep_post_stds = info['abstract_rep_post_means'], info['abstract_rep_post_stds']
        abstract_rep_prior_means, abstract_rep_prior_stds = info['abstract_rep_prior_means'], info['abstract_rep_prior_stds']
        # abstract_rep_prior_means, abstract_rep_prior_stds = torch.zeros_like(abstract_rep_post_means), torch.ones_like(abstract_rep_post_stds)

        # TODO: Don't include time loss factor into KL loss, keep them factorized
        # abstract_rep_kl_loss = torch.mean((self.kl_balance_gaussian(abstract_rep_prior_means, abstract_rep_prior_stds, abstract_rep_post_means, abstract_rep_post_stds, self.cfg.abstract_kl_balance)) * segmentation_samples)
        abs_kl = self.kl_balance_gaussian(abstract_rep_prior_means, abstract_rep_prior_stds, abstract_rep_post_means, abstract_rep_post_stds, self.cfg.abstract_kl_balance)
        abstract_rep_kl_loss = torch.mean(torch.sum(abs_kl * segmentation_samples.detach(), dim=1) / torch.sum(segmentation_samples, dim=1))

        state_rep_post_means, state_rep_post_stds = info['state_rep_post_means'], info['state_rep_post_stds']
        state_rep_prior_means, state_rep_prior_stds = info['state_rep_prior_means'], info['state_rep_prior_stds']

        state_kl = self.kl_balance_gaussian(state_rep_prior_means, state_rep_prior_stds, state_rep_post_means, state_rep_post_stds, self.cfg.state_kl_balance)
        state_rep_kl_loss = torch.mean(state_kl)
        info['state_kl'] = state_kl

        segmentation_post_logits = info['segmentation_post_logits']
        segmentation_prior_logits = info['segmentation_prior_logits']
        segmentation_loss = torch.sigmoid(segmentation_post_logits).mean()
        temp = torch.tensor(self.temperature, device=segmentation_post_logits.device)
        segmentation_kl_loss = torch.mean(concrete.y_kl_divergence(info['y_samples'], segmentation_prior_logits, temp, segmentation_post_logits, temp, kl_balance=self.cfg.segmentation_kl_balance))

        model_loss = self.cfg.reconstruction_loss_weight * reconstruction_loss + \
            self.time_loss_weight * segmentation_loss + \
            self.cfg.abstract_transition_kl_weight * abstract_rep_kl_loss + \
            self.state_kl_weight * state_rep_kl_loss + \
            self.cfg.segmentation_kl_weight * segmentation_kl_loss

        metrics = dict(
            loss=model_loss.item(),
            reconstruction_loss=reconstruction_loss.item(),
            segmentation_loss=segmentation_loss.item(),
            average_compression=torch.minimum(average_compression, torch.tensor(self.max_seq_len, device=average_compression.device)).item(),
            abstract_transition_kl_loss=abstract_rep_kl_loss.item(),
            state_transition_kl_loss=state_rep_kl_loss.item(),
            segmentation_kl_loss=segmentation_kl_loss.item(),
        )

        return model_loss, metrics, info

    def generate(self, batch_size, abstract_sample_std_scalar=1, state_sample_std_scalar=1, generation_length=None, given_segmentations=None, given_abstract_stoch=None, given_state_stoch=None, initial_stoch=None):
        device = self.positional_encoding.device
        if generation_length is None:
            generation_length = self.max_seq_len

        broadcast_positional_encoding = self.positional_encoding_dropout(self.positional_encoding[:, :generation_length].expand(batch_size, -1, -1))

        segmentation_probs = torch.zeros(batch_size, generation_length, device=device, dtype=torch.float32)
        segmentations = torch.zeros(batch_size, generation_length, device=device, dtype=torch.float32)
        segmentation_probs[:, 0] = 1
        segmentations[:, 0] = 1
        if given_segmentations is not None:
            segmentations = given_segmentations

        # abstract_rep = torch.zeros(batch_size, generation_length, self.abstract_rep_dim, device=device, dtype=torch.float32)
        abstract_rep = torch.zeros(batch_size, generation_length, self.cfg.abstract_rep_stoch_dim, device=device, dtype=torch.float32)
        if given_abstract_stoch is not None:
            abstract_rep[..., :self.cfg.abstract_rep_stoch_dim] = given_abstract_stoch

        state_rep = torch.zeros(batch_size, generation_length, self.state_rep_dim, device=device, dtype=torch.float32)
        if given_state_stoch is not None:
            state_rep[..., :self.cfg.state_rep_stoch_dim] = given_state_stoch

        generated_traj = torch.zeros(batch_size, generation_length, self.data_dim, device=device, dtype=torch.float32)

        abstract_stoch_means = torch.zeros(batch_size, generation_length, self.cfg.abstract_rep_stoch_dim, device=device, dtype=torch.float32)
        state_stoch_means = torch.zeros(batch_size, generation_length, self.cfg.state_rep_stoch_dim, device=device, dtype=torch.float32)

        abstract_eps = torch.randn(batch_size, generation_length, self.cfg.abstract_rep_stoch_dim, device=device, dtype=torch.float32)
        abstract_seg_eps = torch.zeros_like(abstract_eps)

        for i in range(generation_length):
            # abstract_rep_prior_params = self.abstract_rep_prior(shift_forward(abstract_rep[:, :i + 1, -self.cfg.abstract_rep_deter_dim:], 1)[:, -1:])
            abstract_rep_prior_params = self.abstract_rep_prior(shift_forward(abstract_rep[:, :i + 1], 1)[:, -1:])
            abstract_rep_prior_means, abstract_rep_prior_stds = abstract_rep_prior_params[..., :self.cfg.abstract_rep_stoch_dim], nn.functional.softplus(abstract_rep_prior_params[..., self.cfg.abstract_rep_stoch_dim:])
            segment = segmentations[:, i:i + 1].unsqueeze(-1)
            if given_abstract_stoch is None:
                if i == 0:
                    abstract_seg_eps[:, i:i + 1] = abstract_eps[:, i:i + 1]
                    if initial_stoch is None:
                        abstract_rep_stoch_samples = abstract_rep_prior_means + abstract_rep_prior_stds * abstract_seg_eps[:, i:i + 1] * abstract_sample_std_scalar
                    else:
                        abstract_rep_stoch_samples = initial_stoch
                else:
                    abstract_seg_eps[:, i:i + 1] = (1 - segment) * abstract_seg_eps[:, i - 1:i] + segment * abstract_eps[:, i:i + 1]
                    abstract_rep_stoch_samples = (1 - segment) * abstract_rep[:, i - 1:i, :self.cfg.abstract_rep_stoch_dim] + segment * (abstract_rep_prior_means + abstract_rep_prior_stds * abstract_seg_eps[:, i:i + 1] * abstract_sample_std_scalar)
                abstract_rep[:, i:i + 1, :self.cfg.abstract_rep_stoch_dim] = abstract_rep_stoch_samples
            abstract_stoch_means[:, i:i + 1] = abstract_rep_prior_means

            segmentation_samples = segmentations[:, :i + 1]
            _, _, causal_segmentation_attention_mask, abstract_causal_segmentation_attention_mask = self.get_segmentation_attention_masks_probabilistic(segmentation_samples)

            # abstract_stoch_hist = abstract_rep[:, :i + 1, :self.cfg.abstract_rep_stoch_dim]
            # abstract_rep_encodings = self.abstract_rep_mlp_encoder(torch.cat([abstract_stoch_hist, broadcast_positional_encoding[:, :i + 1]], dim=-1))
            # transformed_abstract_rep_encodings = self.abstract_rep_transformer_encoder(abstract_stoch_hist, abstract_rep_encodings, mask=abstract_causal_segmentation_attention_mask)
            # abstract_rep_deter = self.abstract_rep_mlp_decoder(transformed_abstract_rep_encodings)
            # abstract_rep[:, i:i + 1, -self.cfg.abstract_rep_deter_dim:] = abstract_rep_deter[:, -1:]

            state_rep_prior_params = self.state_rep_prior(torch.cat([(shift_forward(state_rep[:, :i + 1, -self.cfg.state_rep_deter_dim:], 1))[:, -1:] * (1 - segmentation_samples[:, -1:]).unsqueeze(-1), abstract_rep[:, i:i + 1]], dim=-1))
            state_rep_prior_means, state_rep_prior_stds = state_rep_prior_params[..., :self.cfg.state_rep_stoch_dim], nn.functional.softplus(state_rep_prior_params[..., self.cfg.state_rep_stoch_dim:])
            if given_state_stoch is None:
                state_rep_stoch = state_rep_prior_means + state_rep_prior_stds * torch.randn_like(state_rep_prior_means) * state_sample_std_scalar
                state_rep[:, i:i + 1, :self.cfg.state_rep_stoch_dim] = state_rep_stoch
            state_stoch_means[:, i:i + 1] = state_rep_prior_means

            state_stoch_hist = state_rep[:, :i + 1, :self.cfg.state_rep_stoch_dim]
            abstract_rep_hist = abstract_rep[:, :i + 1]
            state_rep_encodings = self.state_rep_mlp_encoder(torch.cat([state_stoch_hist, abstract_rep_hist], dim=-1))
            transformed_state_rep_encodings = self.state_rep_transformer_encoder(torch.transpose(state_rep_encodings, 0, 1), mask=causal_segmentation_attention_mask).transpose(0, 1)
            state_rep_deter = self.state_rep_mlp_decoder(transformed_state_rep_encodings)
            state_rep[:, i:i + 1, -self.cfg.state_rep_deter_dim:] = state_rep_deter[:, -1:]

            decoder_input = torch.cat([state_rep[:, i:i + 1], abstract_rep[:, i:i + 1]], dim=-1)
            generated_traj[:, i:i + 1] = self.decoder(decoder_input)

            if i < generation_length - 1:
                segmentation_prior_logits = self.segmentation_prior(decoder_input)
                segmentation_samples = torch.distributions.Bernoulli(logits=segmentation_prior_logits).sample().squeeze(-1)
                segmentation_probs[:, i + 1:i + 2] = torch.sigmoid(segmentation_prior_logits).squeeze(-1)
                if given_segmentations is None:
                    if self.cfg['fix_segmentation_period'] is None:
                        segmentations[:, i + 1:i + 2] = segmentation_samples
                    else:
                        if (i + 1) % self.cfg['fix_segmentation_period'] == 0:
                            segmentations[:, i + 1:i + 2] = torch.ones_like(segmentation_samples)
                        else:
                            segmentations[:, i + 1:i + 2] = torch.zeros_like(segmentation_samples)

        return generated_traj, dict(
            abstract_rep=abstract_rep,
            state_rep=state_rep,
            segmentation_samples=segmentations,
            segmentation_probs=segmentation_probs,
            abstract_stoch_means=abstract_stoch_means,
            state_stoch_means=state_stoch_means,
        )

    def get_dist_gaussian(self, means, stds):
        return torch.distributions.Independent(torch.distributions.Normal(means, stds), 1)

    def kl_balance_gaussian(self, prior_means, prior_stds, post_means, post_stds, kl_balance_ratio):
        kl_prior = torch.distributions.kl_divergence(self.get_dist_gaussian(post_means.detach(), post_stds.detach()), self.get_dist_gaussian(prior_means, prior_stds))
        kl_post = torch.distributions.kl_divergence(self.get_dist_gaussian(post_means, post_stds), self.get_dist_gaussian(prior_means.detach(), prior_stds.detach()))
        kl_balanced = kl_balance_ratio * kl_prior + (1 - kl_balance_ratio) * kl_post
        return kl_balanced

    def get_dist_bernoulli(self, logits):
        return torch.distributions.Independent(torch.distributions.Bernoulli(logits=logits), 1)

    def kl_balance_bernoulli(self, prior_logits, post_logits, kl_balance_ratio):
        kl_prior = torch.distributions.kl_divergence(self.get_dist_bernoulli(post_logits.detach()), self.get_dist_bernoulli(prior_logits))
        kl_post = torch.distributions.kl_divergence(self.get_dist_bernoulli(post_logits), self.get_dist_bernoulli(prior_logits.detach()))
        kl_balanced = kl_balance_ratio * kl_prior + (1 - kl_balance_ratio) * kl_post
        return kl_balanced

    def reparameterize(self, means, stds, std_scalar=1.0):
        eps = torch.randn_like(stds)
        return means + eps * stds * std_scalar

    def reparameterize_segments(self, means, stds, segmentations, std_scalar=1.0):
        segmentations = segmentations.unsqueeze(-1)
        eps = torch.randn_like(stds)
        seg_eps = [eps[:, 0]]
        for i in range(1, eps.shape[1]):
            seg_eps.append((1 - segmentations[:, i]) * seg_eps[-1] + segmentations[:, i] * eps[:, i])
        seg_eps = torch.stack(seg_eps, dim=1)
        return means + seg_eps * stds * std_scalar

    def reparameterize_segments_fixed(self, means, stds, normalized_attention_weights, std_scalar=1.0):
        batch_size = means.shape[0]
        seq_len = means.shape[1]
        dim = stds.shape[2]
        eps = torch.randn_like(stds)
        repeated_eps = torch.cat([eps] * seq_len, dim=1).reshape(batch_size, seq_len, seq_len, dim)
        attended_eps = torch.sum(torch.sqrt(normalized_attention_weights).unsqueeze(-1) * repeated_eps, dim=2)
        return means + attended_eps * stds * std_scalar

    def get_segmentation_attention_masks_probabilistic(self, segmentation_probs):
        device = segmentation_probs.device
        batch_size = segmentation_probs.shape[0]
        seq_len = segmentation_probs.shape[1]
        assert seq_len <= self.max_seq_len
        padding = torch.ones(batch_size, seq_len, device=device)
        segmentation_probs_padded = torch.cat([padding, segmentation_probs, padding], dim=1)
        attention_weights = deque([torch.ones(batch_size, seq_len, device=device)])

        forward_p_same_segment = torch.ones(batch_size, seq_len, device=device)
        backward_p_same_segment = torch.ones(batch_size, seq_len, device=device)

        no_segmentation_probs_padded = 1 - segmentation_probs_padded

        base_indices = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0).expand(batch_size, seq_len) + seq_len
        for i in range(seq_len):
            forward_indices = base_indices + i + 1
            backward_indices = base_indices - i

            forward_p_same_segment = forward_p_same_segment * torch.gather(no_segmentation_probs_padded, 1, forward_indices)
            backward_p_same_segment = backward_p_same_segment * torch.gather(no_segmentation_probs_padded, 1, backward_indices)

            attention_weights.append(forward_p_same_segment)
            attention_weights.appendleft(backward_p_same_segment)

        attention_weights = torch.stack(list(attention_weights), dim=-1)
        seq_indices = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0).unsqueeze(1) + (seq_len - torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0).expand(batch_size, seq_len)).unsqueeze(-1)
        attention_weights = torch.gather(attention_weights, 2, seq_indices)

        all_indices = seq_indices - seq_len
        causal_attention_weights = torch.where(torch.zeros(1, device=device) >= all_indices, attention_weights, torch.zeros_like(attention_weights))

        causal_segmentation_attention_mask = causal_attention_weights.unsqueeze(1).expand(batch_size, self.state_rep_transformer_nheads, seq_len, seq_len)
        causal_segmentation_attention_mask = causal_segmentation_attention_mask.reshape(batch_size * self.state_rep_transformer_nheads, seq_len, seq_len)

        abstract_causal_attention_weights = torch.where(torch.zeros(1, device=device) >= all_indices, torch.cat([segmentation_probs] * seq_len, dim=1).reshape(batch_size, seq_len, seq_len), torch.zeros_like(attention_weights))

        abstract_causal_segmentation_attention_mask = abstract_causal_attention_weights.unsqueeze(1).expand(batch_size, self.abstract_rep_transformer_nheads, seq_len, seq_len)
        abstract_causal_segmentation_attention_mask = abstract_causal_segmentation_attention_mask.reshape(batch_size * self.abstract_rep_transformer_nheads, seq_len, seq_len)
        if causal_segmentation_attention_mask.requires_grad:
            # causal_segmentation_attention_mask.register_hook(lambda grad: torch.clamp(torch.nan_to_num(grad, nan=0), min=-1e3, max=1e3))
            causal_segmentation_attention_mask.register_hook(lambda grad: torch.nan_to_num(grad, nan=0))
        if abstract_causal_segmentation_attention_mask.requires_grad:
            abstract_causal_segmentation_attention_mask.register_hook(lambda grad: torch.clamp(torch.nan_to_num(grad, nan=0), min=-1e3, max=1e3))

        normalized_weights = attention_weights / attention_weights.sum(dim=-1, keepdim=True)
        return normalized_weights, None, torch.log(causal_segmentation_attention_mask), torch.log(abstract_causal_segmentation_attention_mask)

    def set_temperature(self, temperature):
        self.temperature = temperature

    def set_time_loss_weight(self, time_loss_weight):
        self.time_loss_weight = time_loss_weight

    def set_state_kl_weight(self, state_kl_weight):
        self.state_kl_weight = state_kl_weight

    def hard_sample(self):
        self.sample = True

    def soft_sample(self):
        self.sample = False

    def train(self, mode=True):
        if mode:
            self.soft_sample()
        else:
            self.hard_sample()
        super().train(mode)

    def eval(self):
        self.hard_sample()
        super().eval()

class RLSegmentationModel(FullPrototypeModel):
    def __init__(self, cfg, obs_dim, action_dim, max_seq_len):
        super().__init__(cfg, obs_dim + action_dim, max_seq_len)
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.state_rep_mlp_encoder = StandardMLP(input_dim=cfg.state_rep_stoch_dim + self.abstract_rep_dim + obs_dim, **cfg.state_rep_mlp_encoder_params, output_dim=cfg.state_rep_transformer_dim)
        self.decoder = StandardMLP(input_dim=self.state_rep_dim, **cfg.decoder_params, output_dim=action_dim)
        self.segmentation_prior = StandardMLP(input_dim=self.state_rep_dim, **cfg.segmentation_prior_params, output_dim=1)

        self.gru_init = StandardMLP(input_dim=self.cfg.segmentation_transformer_dim, layer_sizes=[256, 256], output_dim=self.cfg.segmentation_transformer_dim)
        self.abstract_init = StandardMLP(input_dim=obs_dim, layer_sizes=[256, 256], output_dim=self.cfg.abstract_rep_stoch_dim)

    def forward(self, obs, act, abstract_sample_std_scalar=1.0, state_sample_std_scalar=1.0):
        # traj is a tensor of shape (batch_size, seq_len, data_dim)
        traj = torch.cat([obs, act], dim=-1)
        batch_size = traj.shape[0]
        seq_len = traj.shape[1]
        device = traj.device
        assert seq_len <= self.max_seq_len, "Trajectories must be less than length {}".format(self.seq_len)
        encodings = self.encoder(traj)

        segmentation_encodings = self.segmentation_mlp_encoder(encodings)
        transformed_segmentation_encodings = self.segmentation_transformer_encoder(torch.transpose(segmentation_encodings, 0, 1)).transpose(0, 1) # TRANSPOSE FOR RPR TRANSFORMER
        segmentation_post_logits = []
        segmentation_post_probs = [torch.ones(batch_size, 1, device=device, dtype=torch.float32)]
        segmentation_samples = [torch.ones(batch_size, 1, device=device, dtype=torch.float32)]
        y_samples = []
        # gru_hidden = torch.zeros(batch_size, self.cfg.segmentation_transformer_dim, device=device, dtype=torch.float32)
        gru_hidden = self.gru_init(transformed_segmentation_encodings[:, 0, :])
        for i in range(1, seq_len):
            # gru_hidden = self.segmentation_gru(torch.cat([segmentation_post_probs[-1], transformed_segmentation_encodings[:, i, :]], dim=-1), gru_hidden)
            # Autoregress with segmentation samples not probs
            gru_hidden = self.segmentation_gru(torch.cat([segmentation_samples[-1], transformed_segmentation_encodings[:, i, :]], dim=-1), gru_hidden)
            segmentation_post_logit = self.segmentation_post(gru_hidden)
            if segmentation_post_logit.requires_grad:
                segmentation_post_logit.retain_grad()
            segmentation_sample, y_sample = concrete.sample_binary_concrete(segmentation_post_logit, self.temperature, hard=self.sample)
            segmentation_post_probs.append(torch.sigmoid(segmentation_post_logit))
            segmentation_post_logits.append(segmentation_post_logit)
            if self.cfg['fix_segmentation_period'] is None:
                segmentation_samples.append(segmentation_sample)
            else:
                if i % self.cfg['fix_segmentation_period'] == 0:
                    segmentation_samples.append(torch.ones_like(segmentation_sample))
                else:
                    segmentation_samples.append(torch.zeros_like(segmentation_sample))
            y_samples.append(y_sample)

        segmentation_post_logit_list = segmentation_post_logits
        segmentation_post_logits = torch.stack(segmentation_post_logits, dim=1)
        segmentation_samples = torch.stack(segmentation_samples, dim=1)
        y_samples = torch.stack(y_samples, dim=1)

        segmentation_samples = segmentation_samples.squeeze(-1)
        if segmentation_samples.requires_grad:
            segmentation_samples.register_hook(lambda grad: self.cfg.time_grad_scalar * grad)
        if segmentation_samples.requires_grad:
            segmentation_samples.retain_grad()
        segment_weights, _, causal_segmentation_attention_mask, abstract_causal_segmentation_attention_mask = self.get_segmentation_attention_masks_probabilistic(segmentation_samples)

        attention_encodings = self.compression_mlp_encoder(torch.cat([encodings, transformed_segmentation_encodings], dim=-1))

        # pre_attention_encodings = self.compression_mlp_encoder(torch.cat([encodings, segmentation_samples.unsqueeze(-1)], dim=-1))
        # transformed_pre_attention_encodings = self.compression_transfomer(torch.transpose(pre_attention_encodings, 0, 1)).transpose(0, 1) # TRANSPOSE FOR RPR TRANSFORMER
        # attention_encodings = self.compression_mlp_decoder(transformed_pre_attention_encodings)

        repeated_attention_encodings = torch.cat([attention_encodings] * seq_len, dim=1).reshape(batch_size, seq_len, seq_len, self.cfg.temporal_attention_dim)
        attended_encodings = torch.sum(segment_weights.unsqueeze(-1) * repeated_attention_encodings, dim=2)
        abstract_rep_post_params = self.abstract_rep_post(attended_encodings)
        abstract_rep_post_means, abstract_rep_post_stds = abstract_rep_post_params[..., :self.cfg.abstract_rep_stoch_dim], nn.functional.softplus(abstract_rep_post_params[..., self.cfg.abstract_rep_stoch_dim:])
        # abstract_rep_stoch_samples = self.reparameterize_segments(abstract_rep_post_means, abstract_rep_post_stds, segmentation_samples, std_scalar=abstract_sample_std_scalar)
        abstract_rep_stoch_samples = self.reparameterize_segments_fixed(abstract_rep_post_means, abstract_rep_post_stds, segment_weights.detach(), std_scalar=abstract_sample_std_scalar)

        abstract_init = self.abstract_init(obs[:, 0:1, :])
        abstract_rep_encodings = self.abstract_rep_mlp_encoder(abstract_rep_stoch_samples)
        transformed_abstract_rep_encodings = self.abstract_rep_transformer_encoder(abstract_rep_encodings, mask=abstract_causal_segmentation_attention_mask)
        abstract_rep_deter = self.abstract_rep_mlp_decoder(transformed_abstract_rep_encodings)
        abstract_rep_prior_params = self.abstract_rep_prior(shift_forward(abstract_rep_deter, 1, fill=abstract_init))
        abstract_rep_prior_means, abstract_rep_prior_stds = abstract_rep_prior_params[..., :self.cfg.abstract_rep_stoch_dim], nn.functional.softplus(abstract_rep_prior_params[..., self.cfg.abstract_rep_stoch_dim:])
        abstract_rep = torch.cat([abstract_rep_stoch_samples, abstract_rep_deter], dim=-1)

        state_rep_post_params = self.state_rep_post(torch.cat([encodings, abstract_rep], dim=-1))
        state_rep_post_means, state_rep_post_stds = state_rep_post_params[..., :self.cfg.state_rep_stoch_dim], nn.functional.softplus(state_rep_post_params[..., self.cfg.state_rep_stoch_dim:])
        state_rep_stoch_samples = self.reparameterize(state_rep_post_means, state_rep_post_stds, std_scalar=state_sample_std_scalar)

        state_rep_encodings = self.state_rep_mlp_encoder(torch.cat([state_rep_stoch_samples, abstract_rep, obs], dim=-1))
        transformed_state_rep_encodings = self.state_rep_transformer_encoder(torch.transpose(state_rep_encodings, 0, 1), mask=causal_segmentation_attention_mask).transpose(0, 1) # TRANSPOSE FOR RPR TRANSFORMER
        state_rep_deter = self.state_rep_mlp_decoder(transformed_state_rep_encodings)
        state_rep_prior_params = self.state_rep_prior(torch.cat([shift_forward(state_rep_deter, 1) * (1 - segmentation_samples).unsqueeze(-1), abstract_rep], dim=-1))
        state_rep_prior_means, state_rep_prior_stds = state_rep_prior_params[..., :self.cfg.state_rep_stoch_dim], nn.functional.softplus(state_rep_prior_params[..., self.cfg.state_rep_stoch_dim:])
        state_rep = torch.cat([state_rep_stoch_samples, state_rep_deter], dim=-1)

        # state_rep = torch.zeros_like(state_rep)
        decoder_input = torch.cat([state_rep], dim=-1)
        segmentation_prior_logits = self.segmentation_prior(decoder_input)[:, :-1]
        reconstructed_traj = self.decoder(decoder_input)

        return reconstructed_traj, dict(
            segment_weights=segment_weights,
            segmentation_post_logits=segmentation_post_logits,
            segmentation_post_logit_list=segmentation_post_logit_list,
            segmentation_samples=segmentation_samples,
            y_samples=y_samples,
            abstract_rep_post_means=abstract_rep_post_means,
            abstract_rep_post_stds=abstract_rep_post_stds,
            abstract_rep_prior_means=abstract_rep_prior_means,
            abstract_rep_prior_stds=abstract_rep_prior_stds,
            abstract_rep=abstract_rep,
            state_rep_post_means=state_rep_post_means,
            state_rep_post_stds=state_rep_post_stds,
            state_rep_prior_means=state_rep_prior_means,
            state_rep_prior_stds=state_rep_prior_stds,
            state_rep=state_rep,
            segmentation_prior_logits=segmentation_prior_logits,
        )

    def get_loss(self, obs, act):
        reconstructed_act, info = self.forward(obs, act)
        info['reconstructed_act'] = reconstructed_act
        info['ground_truth_act'] = act
        info['ground_truth_obs'] = obs
        reconstruction_loss = nn.functional.mse_loss(act, reconstructed_act)
        segmentation_samples = info['segmentation_samples']
        average_compression = 1 / torch.mean(segmentation_samples[:, 1:])

        abstract_rep_post_means, abstract_rep_post_stds = info['abstract_rep_post_means'], info['abstract_rep_post_stds']
        abstract_rep_prior_means, abstract_rep_prior_stds = info['abstract_rep_prior_means'], info['abstract_rep_prior_stds']
        # abstract_rep_prior_means, abstract_rep_prior_stds = torch.zeros_like(abstract_rep_post_means), torch.ones_like(abstract_rep_post_stds)

        # TODO: Don't include time loss factor into KL loss, keep them factorized
        # abstract_rep_kl_loss = torch.mean((self.kl_balance_gaussian(abstract_rep_prior_means, abstract_rep_prior_stds, abstract_rep_post_means, abstract_rep_post_stds, self.cfg.abstract_kl_balance)) * segmentation_samples)
        abs_kl = self.kl_balance_gaussian(abstract_rep_prior_means, abstract_rep_prior_stds, abstract_rep_post_means, abstract_rep_post_stds, self.cfg.abstract_kl_balance)
        abstract_rep_kl_loss = torch.mean(torch.sum(abs_kl * segmentation_samples.detach(), dim=1) / torch.sum(segmentation_samples, dim=1))

        state_rep_post_means, state_rep_post_stds = info['state_rep_post_means'], info['state_rep_post_stds']
        state_rep_prior_means, state_rep_prior_stds = info['state_rep_prior_means'], info['state_rep_prior_stds']

        state_kl = self.kl_balance_gaussian(state_rep_prior_means, state_rep_prior_stds, state_rep_post_means, state_rep_post_stds, self.cfg.state_kl_balance)
        state_rep_kl_loss = torch.mean(state_kl)
        info['state_kl'] = state_kl

        segmentation_post_logits = info['segmentation_post_logits']
        segmentation_prior_logits = info['segmentation_prior_logits']
        segmentation_loss = torch.sigmoid(segmentation_post_logits).mean()
        temp = torch.tensor(self.temperature, device=segmentation_post_logits.device)
        segmentation_kl_loss = torch.mean(concrete.y_kl_divergence(info['y_samples'], segmentation_prior_logits, temp, segmentation_post_logits, temp, kl_balance=self.cfg.segmentation_kl_balance))

        model_loss = self.cfg.reconstruction_loss_weight * reconstruction_loss + \
            self.time_loss_weight * segmentation_loss + \
            self.cfg.abstract_transition_kl_weight * abstract_rep_kl_loss + \
            self.state_kl_weight * state_rep_kl_loss + \
            self.cfg.segmentation_kl_weight * segmentation_kl_loss

        metrics = dict(
            loss=model_loss.item(),
            reconstruction_loss=reconstruction_loss.item(),
            segmentation_loss=segmentation_loss.item(),
            average_compression=torch.minimum(average_compression, torch.tensor(self.max_seq_len, device=average_compression.device)).item(),
            abstract_transition_kl_loss=abstract_rep_kl_loss.item(),
            state_transition_kl_loss=state_rep_kl_loss.item(),
            segmentation_kl_loss=segmentation_kl_loss.item(),
        )

        return model_loss, metrics, info

class CorrectRLSegmentationModel(FullPrototypeModel):
    def __init__(self, cfg, obs_dim, action_dim, max_seq_len):
        super().__init__(cfg, obs_dim + action_dim, max_seq_len)
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.state_rep_mlp_encoder = StandardMLP(input_dim=cfg.state_rep_stoch_dim + self.abstract_rep_dim, **cfg.state_rep_mlp_encoder_params, output_dim=cfg.state_rep_transformer_dim)
        self.decoder = StandardMLP(input_dim=self.state_rep_dim, **cfg.decoder_params, output_dim=action_dim + obs_dim)
        self.segmentation_prior = StandardMLP(input_dim=self.state_rep_dim, **cfg.segmentation_prior_params, output_dim=1)

        self.gru_init = StandardMLP(input_dim=self.cfg.segmentation_transformer_dim, layer_sizes=[256, 256], output_dim=self.cfg.segmentation_transformer_dim)
        self.abstract_init = StandardMLP(input_dim=obs_dim, layer_sizes=[256, 256], output_dim=self.cfg.abstract_rep_stoch_dim)

    def forward(self, obs, act, abstract_sample_std_scalar=1.0, state_sample_std_scalar=1.0):
        # traj is a tensor of shape (batch_size, seq_len, data_dim)
        traj = torch.cat([obs, act], dim=-1)
        batch_size = traj.shape[0]
        seq_len = traj.shape[1]
        device = traj.device
        assert seq_len <= self.max_seq_len, "Trajectories must be less than length {}".format(self.seq_len)
        encodings = self.encoder(traj)

        segmentation_encodings = self.segmentation_mlp_encoder(encodings)
        transformed_segmentation_encodings = self.segmentation_transformer_encoder(torch.transpose(segmentation_encodings, 0, 1)).transpose(0, 1) # TRANSPOSE FOR RPR TRANSFORMER
        segmentation_post_logits = []
        segmentation_post_probs = [torch.ones(batch_size, 1, device=device, dtype=torch.float32)]
        segmentation_samples = [torch.ones(batch_size, 1, device=device, dtype=torch.float32)]
        y_samples = []
        # gru_hidden = torch.zeros(batch_size, self.cfg.segmentation_transformer_dim, device=device, dtype=torch.float32)
        gru_hidden = self.gru_init(transformed_segmentation_encodings[:, 0, :])
        for i in range(1, seq_len):
            # gru_hidden = self.segmentation_gru(torch.cat([segmentation_post_probs[-1], transformed_segmentation_encodings[:, i, :]], dim=-1), gru_hidden)
            # Autoregress with segmentation samples not probs
            gru_hidden = self.segmentation_gru(torch.cat([segmentation_samples[-1], transformed_segmentation_encodings[:, i, :]], dim=-1), gru_hidden)
            segmentation_post_logit = self.segmentation_post(gru_hidden)
            if segmentation_post_logit.requires_grad:
                segmentation_post_logit.retain_grad()
            segmentation_sample, y_sample = concrete.sample_binary_concrete(segmentation_post_logit, self.temperature, hard=self.sample)
            segmentation_post_probs.append(torch.sigmoid(segmentation_post_logit))
            segmentation_post_logits.append(segmentation_post_logit)
            if self.cfg['fix_segmentation_period'] is None:
                segmentation_samples.append(segmentation_sample)
            else:
                if i % self.cfg['fix_segmentation_period'] == 0:
                    segmentation_samples.append(torch.ones_like(segmentation_sample))
                else:
                    segmentation_samples.append(torch.zeros_like(segmentation_sample))
            y_samples.append(y_sample)

        segmentation_post_logit_list = segmentation_post_logits
        segmentation_post_logits = torch.stack(segmentation_post_logits, dim=1)
        segmentation_samples = torch.stack(segmentation_samples, dim=1)
        y_samples = torch.stack(y_samples, dim=1)

        segmentation_samples = segmentation_samples.squeeze(-1)
        if segmentation_samples.requires_grad:
            segmentation_samples.register_hook(lambda grad: self.cfg.time_grad_scalar * grad)
        if segmentation_samples.requires_grad:
            segmentation_samples.retain_grad()
        segment_weights, _, causal_segmentation_attention_mask, abstract_causal_segmentation_attention_mask = self.get_segmentation_attention_masks_probabilistic(segmentation_samples)

        attention_encodings = self.compression_mlp_encoder(torch.cat([encodings, transformed_segmentation_encodings], dim=-1))

        # pre_attention_encodings = self.compression_mlp_encoder(torch.cat([encodings, segmentation_samples.unsqueeze(-1)], dim=-1))
        # transformed_pre_attention_encodings = self.compression_transfomer(torch.transpose(pre_attention_encodings, 0, 1)).transpose(0, 1) # TRANSPOSE FOR RPR TRANSFORMER
        # attention_encodings = self.compression_mlp_decoder(transformed_pre_attention_encodings)

        repeated_attention_encodings = torch.cat([attention_encodings] * seq_len, dim=1).reshape(batch_size, seq_len, seq_len, self.cfg.temporal_attention_dim)
        attended_encodings = torch.sum(segment_weights.unsqueeze(-1) * repeated_attention_encodings, dim=2)
        abstract_rep_post_params = self.abstract_rep_post(attended_encodings)
        abstract_rep_post_means, abstract_rep_post_stds = abstract_rep_post_params[..., :self.cfg.abstract_rep_stoch_dim], nn.functional.softplus(abstract_rep_post_params[..., self.cfg.abstract_rep_stoch_dim:])
        abstract_rep_stoch_samples = self.reparameterize_segments(abstract_rep_post_means, abstract_rep_post_stds, segmentation_samples, std_scalar=abstract_sample_std_scalar)

        abstract_init = self.abstract_init(obs[:, 0:1, :])
        abstract_rep_encodings = self.abstract_rep_mlp_encoder(abstract_rep_stoch_samples)
        transformed_abstract_rep_encodings = self.abstract_rep_transformer_encoder(abstract_rep_encodings, mask=abstract_causal_segmentation_attention_mask)
        abstract_rep_deter = self.abstract_rep_mlp_decoder(transformed_abstract_rep_encodings)
        abstract_rep_prior_params = self.abstract_rep_prior(shift_forward(abstract_rep_deter, 1, fill=abstract_init))
        abstract_rep_prior_means, abstract_rep_prior_stds = abstract_rep_prior_params[..., :self.cfg.abstract_rep_stoch_dim], nn.functional.softplus(abstract_rep_prior_params[..., self.cfg.abstract_rep_stoch_dim:])
        abstract_rep = torch.cat([abstract_rep_stoch_samples, abstract_rep_deter], dim=-1)

        state_rep_post_params = self.state_rep_post(torch.cat([encodings, abstract_rep], dim=-1))
        state_rep_post_means, state_rep_post_stds = state_rep_post_params[..., :self.cfg.state_rep_stoch_dim], nn.functional.softplus(state_rep_post_params[..., self.cfg.state_rep_stoch_dim:])
        state_rep_stoch_samples = self.reparameterize(state_rep_post_means, state_rep_post_stds, std_scalar=state_sample_std_scalar)

        state_rep_encodings = self.state_rep_mlp_encoder(torch.cat([state_rep_stoch_samples, abstract_rep], dim=-1))
        transformed_state_rep_encodings = self.state_rep_transformer_encoder(torch.transpose(state_rep_encodings, 0, 1), mask=causal_segmentation_attention_mask).transpose(0, 1) # TRANSPOSE FOR RPR TRANSFORMER
        state_rep_deter = self.state_rep_mlp_decoder(transformed_state_rep_encodings)
        state_rep_prior_params = self.state_rep_prior(torch.cat([shift_forward(state_rep_deter, 1) * (1 - segmentation_samples).unsqueeze(-1), abstract_rep], dim=-1))
        state_rep_prior_means, state_rep_prior_stds = state_rep_prior_params[..., :self.cfg.state_rep_stoch_dim], nn.functional.softplus(state_rep_prior_params[..., self.cfg.state_rep_stoch_dim:])
        state_rep = torch.cat([state_rep_stoch_samples, state_rep_deter], dim=-1)

        # state_rep = torch.zeros_like(state_rep)
        decoder_input = torch.cat([state_rep], dim=-1)
        segmentation_prior_logits = self.segmentation_prior(decoder_input)[:, :-1]
        reconstructed_traj = self.decoder(decoder_input)

        return reconstructed_traj, dict(
            segment_weights=segment_weights,
            segmentation_post_logits=segmentation_post_logits,
            segmentation_post_logit_list=segmentation_post_logit_list,
            segmentation_samples=segmentation_samples,
            y_samples=y_samples,
            abstract_rep_post_means=abstract_rep_post_means,
            abstract_rep_post_stds=abstract_rep_post_stds,
            abstract_rep_prior_means=abstract_rep_prior_means,
            abstract_rep_prior_stds=abstract_rep_prior_stds,
            abstract_rep=abstract_rep,
            state_rep_post_means=state_rep_post_means,
            state_rep_post_stds=state_rep_post_stds,
            state_rep_prior_means=state_rep_prior_means,
            state_rep_prior_stds=state_rep_prior_stds,
            state_rep=state_rep,
            segmentation_prior_logits=segmentation_prior_logits,
        )

    def get_loss(self, obs, act):
        reconstructed_traj, info = self.forward(obs, act)
        reconstructed_act = reconstructed_traj[..., :self.action_dim]
        reconstructed_obs = reconstructed_traj[..., self.action_dim:]
        info['reconstructed_act'] = reconstructed_act
        info['reconstructed_obs'] = reconstructed_obs
        info['ground_truth_act'] = act
        info['ground_truth_obs'] = obs
        reconstruction_loss = nn.functional.mse_loss(act, reconstructed_act) + nn.functional.mse_loss(obs, reconstructed_obs)
        segmentation_samples = info['segmentation_samples']
        average_compression = 1 / torch.mean(segmentation_samples[:, 1:])

        abstract_rep_post_means, abstract_rep_post_stds = info['abstract_rep_post_means'], info['abstract_rep_post_stds']
        abstract_rep_prior_means, abstract_rep_prior_stds = info['abstract_rep_prior_means'], info['abstract_rep_prior_stds']
        # abstract_rep_prior_means, abstract_rep_prior_stds = torch.zeros_like(abstract_rep_post_means), torch.ones_like(abstract_rep_post_stds)

        # TODO: Don't include time loss factor into KL loss, keep them factorized
        # abstract_rep_kl_loss = torch.mean((self.kl_balance_gaussian(abstract_rep_prior_means, abstract_rep_prior_stds, abstract_rep_post_means, abstract_rep_post_stds, self.cfg.abstract_kl_balance)) * segmentation_samples)
        abs_kl = self.kl_balance_gaussian(abstract_rep_prior_means, abstract_rep_prior_stds, abstract_rep_post_means, abstract_rep_post_stds, self.cfg.abstract_kl_balance)
        abstract_rep_kl_loss = torch.mean(torch.sum(abs_kl * segmentation_samples.detach(), dim=1) / torch.sum(segmentation_samples, dim=1))

        state_rep_post_means, state_rep_post_stds = info['state_rep_post_means'], info['state_rep_post_stds']
        state_rep_prior_means, state_rep_prior_stds = info['state_rep_prior_means'], info['state_rep_prior_stds']

        state_kl = self.kl_balance_gaussian(state_rep_prior_means, state_rep_prior_stds, state_rep_post_means, state_rep_post_stds, self.cfg.state_kl_balance)
        state_rep_kl_loss = torch.mean(state_kl)
        info['state_kl'] = state_kl

        segmentation_post_logits = info['segmentation_post_logits']
        segmentation_prior_logits = info['segmentation_prior_logits']
        segmentation_loss = torch.sigmoid(segmentation_post_logits).mean()
        temp = torch.tensor(self.temperature, device=segmentation_post_logits.device)
        segmentation_kl_loss = torch.mean(concrete.y_kl_divergence(info['y_samples'], segmentation_prior_logits, temp, segmentation_post_logits, temp, kl_balance=self.cfg.segmentation_kl_balance))

        model_loss = self.cfg.reconstruction_loss_weight * reconstruction_loss + \
            self.time_loss_weight * segmentation_loss + \
            self.cfg.abstract_transition_kl_weight * abstract_rep_kl_loss + \
            self.state_kl_weight * state_rep_kl_loss + \
            self.cfg.segmentation_kl_weight * segmentation_kl_loss

        metrics = dict(
            loss=model_loss.item(),
            reconstruction_loss=reconstruction_loss.item(),
            segmentation_loss=segmentation_loss.item(),
            average_compression=torch.minimum(average_compression, torch.tensor(self.max_seq_len, device=average_compression.device)).item(),
            abstract_transition_kl_loss=abstract_rep_kl_loss.item(),
            state_transition_kl_loss=state_rep_kl_loss.item(),
            segmentation_kl_loss=segmentation_kl_loss.item(),
        )

        return model_loss, metrics, info

def get_activation(activation):
    if activation == 'elu':
        return nn.ELU
    elif activation == 'relu':
        return nn.ReLU
    else:
        return NotImplementedError("Activation not implemented yet")

def shift_forward(x, shift, fill=None):
    if fill is None:
        return torch.cat([torch.zeros_like(x[:, -shift:]), x[:, :-shift]], dim=1)
    else:
        assert fill.shape[1] == shift
        return torch.cat([fill, x[:, :-shift]], dim=1)

class StandardMLP(nn.Module):
    def __init__(self, input_dim, layer_sizes=[400, 400, 400, 400], output_dim=1, activate_last=False, activation='elu'):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layer_sizes = layer_sizes
        self.activation = get_activation(activation)

        if len(layer_sizes) == 0:
            if not activate_last:
                self.network = nn.Linear(input_dim, output_dim)
            else:
                self.network = nn.Sequential(nn.Linear(input_dim, output_dim), self.activation())
            return

        layer_list = [nn.Linear(self.input_dim, self.layer_sizes[0]), self.activation()]
        for i in range(len(self.layer_sizes) - 1):
            layer_list.append(nn.Linear(self.layer_sizes[i], self.layer_sizes[i + 1]))
            layer_list.append(self.activation())
        layer_list.append(nn.Linear(self.layer_sizes[-1], output_dim))
        if activate_last:
            layer_list.append(self.activation())

        self.network = nn.Sequential(*layer_list)

    def forward(self, x):
        return self.network.forward(x)

class CNNEncoder(nn.Module):
    def __init__(self, img_shape):
        super().__init__()
        self.img_shape = img_shape

        self.feature = nn.Sequential(
            nn.Conv2d(img_shape[0], 32, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.feature(x)
        x = x.reshape(x.shape[0], -1)
        return x

    def get_output_dim(self):
        return 1024

class CNNDecoder(nn.Module):
    def __init__(self, feat_dim, img_shape):
        super().__init__()
        self.feat_dim = feat_dim
        self.img_shape = img_shape

        self.linear = nn.Linear(feat_dim, 1024)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1024, 128, 5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 6, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, img_shape[0], 6, stride=2)
        )

    def forward(self, x):
        x = self.linear(x)
        x = x.reshape(x.shape[0], -1, 1, 1)
        mean = self.decoder(x)
        return mean

    def get_mse(self, feats, batch_obs):
        mean = self.forward(feats)
        loss = nn.MSELoss()(mean, batch_obs)
        return loss
