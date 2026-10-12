from typing import Tuple, List

import torch
from collections import defaultdict


class TargetMixture:
    def __init__(self, variance = None, latents=None, noisy_latents=None, reference_batches=None):
        self.variance = variance
        self.reference_batches = reference_batches
        if latents is None:
            self.latents = latents
        else:
            self.latents = latents.detach()

        if noisy_latents is None:
            self.noisy_latents = noisy_latents
        else:
            self.noisy_latents = noisy_latents.detach()
        self.current_weights_per_expert = None
        self.current_expert_indices = None

    def update(self, variance, latents, noisy_latents):
        self.variance = variance
        self.latents = latents.detach()
        self.noisy_latents = noisy_latents.detach()

    def sample_closest_latents(self, reference_latents, noisy_indices, k=1) -> Tuple[torch.Tensor, torch.Tensor]:
        device = self.noisy_latents.device
        reference_latents = reference_latents.to(device)

        B, C, H, W = self.noisy_latents.shape
        D = C * H * W

        noisy_indices = torch.tensor(noisy_indices, device=device, dtype=torch.long)

        var_noisy = self.variance.index_select(
            0, torch.as_tensor(noisy_indices, device=self.latents.device)).squeeze().unsqueeze(0)
        noisy_latents_flat  = self.noisy_latents.view(B, D).index_select(dim=0, index=noisy_indices)
        ref_flat = reference_latents.view(reference_latents.shape[0], D).to(device)

        # squared distances
        diff = noisy_latents_flat.unsqueeze(0) - ref_flat.unsqueeze(1)
        sq_dist = diff.square().mean(dim=-1)

        logits = -sq_dist / (2 * var_noisy)
        invalid = ~torch.isfinite(logits)
        logits = logits.masked_fill(invalid, float("-inf"))

        probs = torch.softmax(logits, dim=0).mean(1)
        idx = torch.multinomial(probs, k, replacement=False)

        return reference_latents.index_select(dim=0, index=idx),probs.index_select(dim=0, index=idx).softmax(0)

    def sample_furthest_latents(self, reference_batch, noisy_indices, k=1) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        device = self.noisy_latents.device
        reference_latents = reference_batch["img"].to(device)
        reference_embeddings = reference_batch["emb"].to(device)
        reference_embeddings_pooled = reference_batch["emb_pooled"].to(device)

        B, C, H, W = self.noisy_latents.shape
        D = C * H * W

        noisy_indices = torch.tensor(noisy_indices, device=device, dtype=torch.long)

        var_noisy = self.variance.index_select(
            0, torch.as_tensor(noisy_indices, device=self.latents.device)).squeeze().unsqueeze(0)
        noisy_latents_flat = self.noisy_latents.view(B, D).index_select(dim=0, index=noisy_indices)
        ref_flat = reference_latents.view(reference_latents.shape[0], D).to(device)

        # squared distances
        diff = noisy_latents_flat.unsqueeze(0) - ref_flat.unsqueeze(1)
        sq_dist = diff.square().mean(dim=-1)

        logits = sq_dist / (2 * var_noisy)
        invalid = ~torch.isfinite(logits)
        logits = logits.masked_fill(invalid, float("-inf"))

        probs = torch.softmax(logits, dim=0).mean(1)
        idx = torch.multinomial(probs, k, replacement=False)

        selected_embs = reference_embeddings.index_select(dim=0, index=idx)
        selected_embs_pooled = reference_embeddings_pooled.index_select(dim=0, index=idx)

        return reference_latents.index_select(dim=0, index=idx), selected_embs, selected_embs_pooled, probs.index_select(dim=0, index=idx).softmax(0)

    def sample_reference_batches_by_labels(self, data_pairs, labels, k, scaling_factor, shift_factor):
        latents, embeddings = data_pairs
        idx_by_class = defaultdict(list)
        for i, y in labels.items():
            if int(y) >= 0:
                idx_by_class[int(y)].append(i)

        reference_batches = {}

        for cls, idxs in idx_by_class.items():
            reference_batches[cls] = {}
            if len(idxs) >= k:
                chosen = [idxs[j] for j in torch.randperm(len(idxs))[:k].tolist()]
            else:
                chosen = [idxs[j] for j in torch.randint(0, len(idxs), (k,)).tolist()]

            samples = [(latents[i].latent_dist.sample() - shift_factor) * scaling_factor for i in chosen]
            batch_emb_out = [embeddings[i] for i in chosen]
            batch_emb =  [out.prompt_embeds for out in batch_emb_out]
            batch_emb_pooled = [out.pooled_prompt_embeds for out in batch_emb_out]

            if len(samples) > 0 and torch.is_tensor(samples[0]):
                batch_img = torch.stack(samples, dim=0)
            else:
                batch_img = samples

            reference_batches[cls]["img"] = batch_img
            reference_batches[cls]["emb"] =  torch.stack(batch_emb, dim=0)
            reference_batches[cls]["emb_pooled"] =  torch.stack(batch_emb_pooled, dim=0)

        self.reference_batches = reference_batches
        return reference_batches

