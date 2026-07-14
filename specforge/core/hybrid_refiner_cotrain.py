#!/usr/bin/env python3
# coding=utf-8
"""Training wrapper for the RIPPLE-vs-DSpark head-to-head, in the SpecForge framework.

Reuses the DFlash drafter + block alignment (`compute_block_features`) and swaps in
`HybridRefinerHead` (which produces a logit BIAS on the drafter's parallel readout) trained with
DSpark's exact ce+l1(TV)+decay objective (`combined_training_loss`). One class trains EITHER head
(Markov special case OR r-bottleneck SGU) by flags, so the ablation is apples-to-apples: same
drafter backbone, same data, same block, same loss -- only the head architecture differs.

  base_logits    = frozen target lm_head( drafter_hidden )      (drafter's parallel logits, full V)
  refined_logits = base_logits + head.bias(h, g, prev_token)    (Markov or SGU)
  target_logits  = frozen target lm_head( target_hidden )       (full-vocab teacher dist for L1/TV)

NOT modified: dflash_refiner.py / dflash_refiner_cotrain.py. Pass a `CoTrainFeatureExtractor`
(from dflash_refiner_cotrain) so gradients reach the drafter; the target lm_head/embed stay frozen.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from specforge.core.hybrid_refiner_head import (
    HybridRefinerHead,
    combined_training_loss,
    markov_style_loss,
)


class OnlineHybridRefinerCoTrain(nn.Module):
    """DFlash drafter (optionally co-trained) + HybridRefinerHead, trained with DSpark's loss."""

    def __init__(self, feature_extractor, *,
                 markov_rank: int = 256, sgu_enabled: bool = True, use_hidden: bool = True,
                 use_perpos: bool = True, use_global: bool = True,
                 use_residual: bool = True, use_mixer: bool = True, use_mlp: bool = True,
                 mlp_ratio: int = 4, input_mode: str = "concat", mixer_init: str = "eye",
                 use_norm: bool = True, use_mix_out: bool = True, shared_readout_rank: int = 0,
                 base_readout_rank: int = 0,
                 l1_alpha: float = 0.9, ce_alpha: float = 0.1, loss_decay_gamma: float = 4.0,
                 consistency_weight: float = 0.0, cotrain_drafter: bool = True):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.block_size = feature_extractor.block_size
        V, H = feature_extractor.lm_head.weight.shape  # lm_head: Linear(H, V) -> weight (V, H)
        # SHARED low-rank lm_head: if shared_readout_rank>0 the head uses ONE low-rank readout (w2 + ro_down)
        # for BOTH the drafter base AND the refined output: base = w2(ro_down(h)), refined = w2(ro_down(h)+code).
        # The code dim (markov_rank) is forced to the shared rank. TEACHER target_logits ALWAYS stays full
        # lm_head. Calibration-fit via init_shared_readout(). 0 -> full frozen lm_head base (no compression).
        # SEPARATE low-rank base (non-shared): base_readout_rank>0 gives the drafter base its OWN low-rank
        # readout at that (larger) rank while the refiner keeps markov_rank -> base accept + refiner latency
        # decoupled. Mutually exclusive with shared_readout_rank. EITHER -> head computes base internally.
        self.shared_readout_rank = int(shared_readout_rank)
        self.base_readout_rank = int(base_readout_rank)
        # rank >= hidden: no compression possible -> fall back to the FULL frozen lm_head base (no readout,
        # no training). Makes base_readout_rank=H the exact full-rank control. (head normalizes the same way.)
        if self.base_readout_rank >= H:
            self.base_readout_rank = 0
        assert not (self.shared_readout_rank > 0 and self.base_readout_rank > 0), \
            "shared_readout_rank (shared) and base_readout_rank (separate) are mutually exclusive"
        self.use_lowrank_base = self.shared_readout_rank > 0 or self.base_readout_rank > 0
        head_rank = self.shared_readout_rank if self.shared_readout_rank > 0 else markov_rank
        self.refiner = HybridRefinerHead(
            V, H, self.block_size, markov_rank=head_rank, sgu_enabled=sgu_enabled,
            use_hidden=use_hidden, use_perpos=use_perpos, use_global=use_global,
            use_residual=use_residual, use_mixer=use_mixer,
            use_mlp=use_mlp, mlp_ratio=mlp_ratio, input_mode=input_mode, mixer_init=mixer_init,
            use_norm=use_norm, use_mix_out=use_mix_out,
            shared_readout=self.shared_readout_rank > 0, base_readout_rank=self.base_readout_rank,
        )
        self.l1_alpha = float(l1_alpha)
        self.ce_alpha = float(ce_alpha)
        self.loss_decay_gamma = loss_decay_gamma
        self.consistency_weight = float(consistency_weight)

        # Freeze the whole feature extractor; optionally un-freeze the drafter backbone (co-train).
        for p in self.feature_extractor.parameters():
            p.requires_grad = False
        self._cotrain_drafter_params = 0
        if cotrain_drafter:
            for p in self.feature_extractor.draft_model.parameters():
                p.requires_grad = True
                self._cotrain_drafter_params += p.numel()

    @torch.no_grad()
    def init_shared_readout(self, covariance, rotate=False):
        """Calibration (data-PCA) init of the head's SHARED low-rank readout (w2 + ro_down) from
        C = sum_i h_i h_i^T. After: base = w2(ro_down(h)) ~= lm_head(h); trainable (co-adapts)."""
        assert self.shared_readout_rank > 0, "init_shared_readout requires shared_readout_rank > 0"
        self.refiner.fit_shared_readout(covariance, self.feature_extractor.lm_head.weight, rotate=rotate)

    @torch.no_grad()
    def init_base_readout(self, covariance, rotate=False):
        """Calibration (data-PCA) init of the head's SEPARATE low-rank base readout (base_down + base_up)
        from C = sum_i h_i h_i^T. After: base = base_up(base_down(h)) ~= lm_head(h); trainable (co-adapts)."""
        assert self.base_readout_rank > 0, "init_base_readout requires base_readout_rank > 0"
        self.refiner.fit_base_readout(covariance, self.feature_extractor.lm_head.weight, rotate=rotate)

    def _base_logits(self, h):
        """Non-shared drafter readout = full frozen target lm_head. (Shared mode computes base inside the
        head's forward, so this is only used by the non-shared path / accept eval.)"""
        return self.feature_extractor.lm_head(h)

    def _decay_weight(self, valid):
        """Per-position loss weight = binary_mask * exp(-(k-1)_+/gamma). The (k-1) shift accounts
        for OUR block's anchor at position 0 (masked) so the FIRST prediction (k=1) gets weight 1 --
        matching DSpark's semantics (its block pos 0 = first prediction, weight 1). See existing
        dflash_refiner_cotrain convention."""
        w = valid
        if self.loss_decay_gamma:
            kk = torch.arange(self.block_size, device=valid.device)
            decay = torch.exp(-(kk - 1).clamp(min=0).float() / self.loss_decay_gamma)
            w = w * decay.view(1, self.block_size)
        return w

    def forward(self, input_ids, hidden_states, loss_mask, final_hidden=None, lambda_base: float = 0.0):
        """Returns (loss, accuracy). `hidden_states` = drafter's captured MIDDLE layers (for the
        drafter forward); `final_hidden` = target's POST-NORM last hidden (B,seq,H) for the L1/TV
        teacher distribution -- REQUIRED (lm_head on `hidden_states` would be the wrong layer/dim).
        lambda_base>0 adds an optional drafter-anchor term; 0 = pure DSpark-style."""
        assert final_hidden is not None, (
            "final_hidden (target post-norm hidden) is required for the L1/TV target distribution; "
            "pass target_output.final_hidden (HF backend provides it)."
        )
        fe = self.feature_extractor
        f = fe.compute_block_features(input_ids, hidden_states, loss_mask)

        B, _, H = f.output_hidden.shape
        block = self.block_size
        n = f.output_hidden.size(1) // block
        BN = B * n

        h = f.output_hidden.view(B, n, block, H).reshape(BN, block, H)
        g = h.mean(dim=1, keepdim=True).expand(-1, block, -1)  # block-mean summary (per existing code)

        tgt = f.target_ids.reshape(BN, block)                  # (BN, block) ground-truth tokens
        prev_tok = tgt.roll(shifts=1, dims=1)                  # prev[k] = gt[k-1]
        am1_idx = (f.anchor_positions - 1).clamp(min=0)
        prev_tok[:, 0] = torch.gather(input_ids, 1, am1_idx).reshape(BN)  # token before block

        # SHARED: base is computed INSIDE the head's forward (via w2(ro_down(h))); pass None. Else: full lm_head.
        base_in = None if self.use_lowrank_base else self._base_logits(h)

        # Full-vocab teacher distribution: target lm_head( target POST-NORM hidden that predicts each
        # block token ). target_ids[k] is at seq position anchor+k -> predicting hidden is at anchor+k-1.
        # Use `final_hidden` (post-norm, lm_head-compatible), NOT `hidden_states` (drafter middle layers).
        with torch.no_grad():
            Hf = final_hidden.shape[-1]
            offs = torch.arange(block, device=h.device).view(1, 1, block)
            predict_idx = (f.anchor_positions.unsqueeze(-1) + offs - 1).clamp(min=0)  # (B, n, block)
            gidx = predict_idx.reshape(B, n * block, 1).expand(-1, -1, Hf)
            target_hidden = torch.gather(final_hidden, 1, gidx).view(BN, block, Hf)
            target_logits = fe.lm_head(target_hidden)          # (BN, block, V) frozen

        valid = f.binary_mask.reshape(BN, block)               # 0/1 (position 0 = anchor already masked)
        # FAITHFUL to DSpark build_eval_mask: PREFIX (cumprod) mask -- once a block position is
        # invalid (OOB / non-assistant), mask ALL following positions (they're unreachable in a valid
        # left-to-right accept). DSpark cumprods over its whole block (pos0=first pred); OUR pos0 is
        # the anchor (already 0), so cumprod over the PREDICTION positions 1: only.
        valid = valid.clone()
        valid[:, 1:] = valid[:, 1:].cumprod(dim=1)
        w = self._decay_weight(valid)                          # decay folded in -> pass gamma=None below

        loss, terms = combined_training_loss(
            self.refiner, base_in, h, g, prev_tok, target_logits, tgt, w,
            consistency_weight=self.consistency_weight,
            l1_alpha=self.l1_alpha, ce_alpha=self.ce_alpha, loss_decay_gamma=None,
        )
        base_logits = terms["base"]   # real base: shared -> w2(ro_down(h)); else -> echoed lm_head(h)

        # Drafter-anchor: ADDITIVE markov_style_loss(base) = ce_alpha*CE + l1_alpha*L1 (== CE + TV, the
        # SAME DSpark loss the refiner gets), weighted by lambda_base -> keeps the co-trained drafter a
        # valid standalone drafter matching the target's FULL distribution (not just argmax).
        # Total = markov_style_loss(refined) + lambda_base * markov_style_loss(base).
        if lambda_base > 0.0:
            V = base_logits.shape[-1]
            base_loss, _, _ = markov_style_loss(
                base_logits.reshape(-1, V), target_logits.reshape(-1, V), tgt.reshape(-1),
                w.reshape(-1), self.l1_alpha, self.ce_alpha)
            loss = loss + lambda_base * base_loss

        with torch.no_grad():
            pred = terms["refined"].reshape(-1, base_logits.shape[-1]).argmax(dim=-1)
            flat_tgt = tgt.reshape(-1)
            flat_valid = valid.reshape(-1)
            accuracy = ((pred == flat_tgt) & (flat_valid > 0.5)).sum().float() / (flat_valid.sum() + 1e-6)

        return loss, accuracy

    @torch.no_grad()
    def accept_lengths(self, input_ids, hidden_states, loss_mask, jacobi_passes: int = None):
        """Mean accepted draft length (refiner, drafter) on the SAME sampled blocks -- the honest
        free-running inference proxy. INFERENCE is head-faithful:
          * Markov (sgu off): SEQUENTIAL free-running, one position/step (== DSpark sample_block).
          * SGU   (sgu on):   PARALLEL Jacobi, `jacobi_passes` passes refining all positions at once.
        Accept length = leading run of slots 1..L-1 whose argmax matches the ground-truth (cumprod)."""
        fe = self.feature_extractor
        f = fe.compute_block_features(input_ids, hidden_states, loss_mask)
        B, _, H = f.output_hidden.shape
        block = self.block_size
        n = f.output_hidden.size(1) // block
        BN = B * n
        device = input_ids.device

        h = f.output_hidden.view(B, n, block, H).reshape(BN, block, H)
        g = h.mean(dim=1, keepdim=True).expand(-1, block, -1)
        tgt = f.target_ids.reshape(BN, block)                  # [:,0] = real anchor
        # SHARED: base via the head's shared readout w2(ro_down(h)); else full lm_head. (eval, no_grad ->
        # calling the head method under SHARD_GRAD_OP full params is fine.)
        base_logits = (self.refiner.readout_base(h) if self.use_lowrank_base
                       else self._base_logits(h))               # [BN, block, V]

        anchors = f.anchor_positions.reshape(BN, 1)
        pos_abs = anchors + torch.arange(block, device=device).view(1, block)
        valid = pos_abs < f.seq_len
        keep = f.block_keep_mask.reshape(BN).float()

        def _accept(pred):
            match = (pred[:, 1:] == tgt[:, 1:]) & valid[:, 1:]
            accept = match.float().cumprod(dim=1).sum(dim=1)
            return (accept * keep).sum() / keep.sum().clamp(min=1.0)

        drafter_accept = _accept(base_logits.argmax(dim=-1))   # parallel-argmax baseline (no head)

        am1_idx = (f.anchor_positions - 1).clamp(min=0)
        tok_am1 = torch.gather(input_ids, 1, am1_idx).reshape(BN)
        if self.refiner.sgu_enabled:
            # PARALLEL Jacobi: seed from the drafter's OWN parallel prediction (free-running, NOT the
            # ground truth), then refine ALL slots per pass. Slot 0 stays the given anchor.
            K = jacobi_passes if jacobi_passes is not None else (block - 1)
            pred = base_logits.argmax(dim=-1)
            pred[:, 0] = tgt[:, 0]
            for _ in range(K):
                prev_tok = pred.roll(shifts=1, dims=1)
                prev_tok[:, 0] = tok_am1
                new_pred = self.refiner(base_logits, h, g, prev_tok)[1].argmax(dim=-1)   # [1] = refined
                new_pred[:, 0] = tgt[:, 0]
                pred = new_pred
        else:
            # SEQUENTIAL free-running (DSpark Markov): decide slots left-to-right; each step reads the
            # already-decided prev (slot 0 = anchor). pred[k>=1] init value is overwritten before use.
            pred = tgt.clone()
            for k in range(1, block):
                prev_tok = pred.roll(shifts=1, dims=1)
                prev_tok[:, 0] = tok_am1
                pred[:, k] = self.refiner(base_logits, h, g, prev_tok)[1][:, k, :].argmax(dim=-1)   # [1]=refined
        return _accept(pred), drafter_accept
