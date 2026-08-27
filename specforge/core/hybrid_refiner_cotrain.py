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
    markov_loss_shared_target,
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
                 base_readout_rank: int = 0, input_scale: float = 1.0, input_relu: bool = False,
                 l1_alpha: float = 0.9, ce_alpha: float = 0.1, loss_decay_gamma: float = 4.0,
                 consistency_weight: float = 0.0, consistency_passes: int = 1,
                 grad_checkpoint_loss: bool = True, cotrain_drafter: bool = True,
                 block_convention: str = "fillin", anchor_full_weight: bool = False,
                 confidence_alpha: float = 0.0, confidence_mode: str = "trajectory",
                 confidence_detach: bool = False):
        super().__init__()
        # Confidence head (opt-in). alpha=0.0 (default): NO module, NO loss term --
        # bit-identical to pre-confidence behaviour; existing checkpoints resume cleanly.
        # alpha>0: BCE(conf_pred, per-slot acceptance prob) at this weight, JOINTLY TRAINED
        # (official-style: gradients flow into backbone/head via the conf input features)
        # unless confidence_detach=True. Mode: "trajectory" (our causal-mixed latent input)
        # or "dspark" (official cat([h, W1[prev]]) into Linear(H+r,1)).
        self.confidence_alpha = float(confidence_alpha)
        self.confidence_mode = str(confidence_mode)
        self.confidence_detach = bool(confidence_detach)
        assert block_convention in ("fillin", "shift"), block_convention
        # True: the lambda_base drafter-anchor uses UNDECAYED (validity-only) weights, so deep
        # slots are anchored at full strength. False (default): historical decayed anchor.
        self.anchor_full_weight = bool(anchor_full_weight)
        # shift = the deepseek/DeepSpec label alignment (slot k predicts the token AFTER its
        # position; every slot supervised; prev[0] = the anchor token). Must MATCH the value set
        # on the feature extractor -- targets and prev/decay/eval indexing move together.
        self.block_convention = block_convention
        # The extractor owns the target alignment; stamp it so compute_block_features branches.
        feature_extractor.block_convention = block_convention
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
            input_scale=input_scale, input_relu=input_relu,
            confidence_mode=self.confidence_mode if self.confidence_alpha > 0.0 else "off",
        )
        self.l1_alpha = float(l1_alpha)
        self.ce_alpha = float(ce_alpha)
        self.loss_decay_gamma = loss_decay_gamma
        self.consistency_weight = float(consistency_weight)
        # Number of free-running Jacobi rounds the consistency term covers (1 = old behaviour).
        self.consistency_passes = max(1, int(consistency_passes))
        # Recompute each loss term's (N,V) fp32 intermediates in backward instead of storing them.
        self.grad_checkpoint_loss = bool(grad_checkpoint_loss)

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
        """Per-position loss weight = binary_mask * exp(-k'/gamma), where k' is the PREDICTION
        index. fillin: the anchor sits at slot 0 (masked), the first prediction is slot 1 ->
        k' = (k-1)_+ so it gets weight 1 (matches DSpark semantics). shift: slot 0 IS the first
        prediction -> k' = k directly."""
        w = valid
        if self.loss_decay_gamma:
            kk = torch.arange(self.block_size, device=valid.device)
            if self.block_convention == "shift":
                exponent = kk.float()
            else:
                exponent = (kk - 1).clamp(min=0).float()
            decay = torch.exp(-exponent / self.loss_decay_gamma)
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

        shift = self.block_convention == "shift"
        tgt = f.target_ids.reshape(BN, block)                  # (BN, block) ground-truth tokens
        prev_tok = tgt.roll(shifts=1, dims=1)                  # prev[k] = gt[k-1]
        # prev for the FIRST slot: fillin -> slot 0 carries the anchor itself, so its prev is
        # the token BEFORE the block; shift -> slot 0 predicts anchor+1, so its prev IS the
        # anchor token (always known at inference -- same first_prev as DSpark's markov chain).
        first_prev_idx = f.anchor_positions if shift else (f.anchor_positions - 1).clamp(min=0)
        prev_tok[:, 0] = torch.gather(input_ids, 1, first_prev_idx).reshape(BN)

        # SHARED: base is computed INSIDE the head's forward (via w2(ro_down(h))); pass None. Else: full lm_head.
        base_in = None if self.use_lowrank_base else self._base_logits(h)

        # Full-vocab teacher distribution: target lm_head( target POST-NORM hidden that predicts each
        # block token ). target_ids[k] is at seq position anchor+k -> predicting hidden is at anchor+k-1.
        # Use `final_hidden` (post-norm, lm_head-compatible), NOT `hidden_states` (drafter middle layers).
        with torch.no_grad():
            Hf = final_hidden.shape[-1]
            offs = torch.arange(block, device=h.device).view(1, 1, block)
            # The teacher hidden that PREDICTS slot k's label token: the label sits at
            # anchor+k (fillin) / anchor+k+1 (shift), so its predicting post-norm hidden sits
            # one position earlier: anchor+k-1 (fillin) / anchor+k (shift).
            predict_off = 0 if shift else -1
            predict_idx = (f.anchor_positions.unsqueeze(-1) + offs + predict_off).clamp(min=0)
            gidx = predict_idx.reshape(B, n * block, 1).expand(-1, -1, Hf)
            target_hidden = torch.gather(final_hidden, 1, gidx).view(BN, block, Hf)
            target_logits = fe.lm_head(target_hidden)          # (BN, block, V) frozen

        valid = f.binary_mask.reshape(BN, block)
        # FAITHFUL to DSpark build_eval_mask: PREFIX (cumprod) mask -- once a block position is
        # invalid (OOB / non-assistant), mask ALL following positions (they're unreachable in a
        # valid left-to-right accept). shift: every slot is a prediction -> cumprod the whole
        # block (exactly DSpark). fillin: slot 0 is the anchor (already 0 in binary_mask), so
        # cumprod over the PREDICTION positions 1: only.
        valid = valid.clone()
        if shift:
            valid = valid.cumprod(dim=1)
        else:
            valid[:, 1:] = valid[:, 1:].cumprod(dim=1)
        w = self._decay_weight(valid)                          # decay folded in -> pass gamma=None below

        loss, terms = combined_training_loss(
            self.refiner, base_in, h, g, prev_tok, target_logits, tgt, w,
            consistency_weight=self.consistency_weight,
            consistency_passes=self.consistency_passes,
            grad_checkpoint_loss=self.grad_checkpoint_loss,
            l1_alpha=self.l1_alpha, ce_alpha=self.ce_alpha, loss_decay_gamma=None,
            pin_slot0=not shift,
        )
        base_logits = terms["base"]   # real base: shared -> w2(ro_down(h)); else -> echoed lm_head(h)
        # --- loss-term diagnostics for record_metrics (kept as TENSORS so the train loop can
        # all_reduce them across ranks the same way it does loss/accuracy). Detached already.
        self.last_loss_terms = {"tf": terms["tf"], "ce": terms["ce"], "l1": terms["l1"],
                                "cons": terms["cons"]}
        # Per-Jacobi-round consistency losses. If these are all EQUAL the rollout is already sitting
        # on a fixed point (bias too small to flip any argmax -- the ReZero/zeros-init regime), so the
        # extra rounds cost compute and add no signal. record_metrics logs their spread for exactly this.
        cpp = terms.get("cons_per_pass") or []
        self.last_cons_per_pass = torch.stack(cpp).detach() if cpp else None

        # Drafter-anchor: ADDITIVE markov_style_loss(base) = ce_alpha*CE + l1_alpha*L1 (== CE + TV, the
        # SAME DSpark loss the refiner gets), weighted by lambda_base -> keeps the co-trained drafter a
        # valid standalone drafter matching the target's FULL distribution (not just argmax).
        # Total = markov_style_loss(refined) + lambda_base * markov_style_loss(base).
        if lambda_base > 0.0:
            V = base_logits.shape[-1]
            # Anchor weights: DECAYED by default (historical recipes). anchor_full_weight drops
            # the exp(-k/gamma) decay from the ANCHOR ONLY -- the head's learning loss keeps it
            # (the decay mirrors the prefix-product value structure of speculative accept).
            # Rationale: the anchor PRESERVES the warm-started drafter, and preservation should
            # not be depth-discounted -- measured erosion is depth-graded exactly where the
            # decayed anchor is weakest (slot-6 weight 0.22; standalone drafter accept@6 fell
            # 0.52 -> 0.20 by 40k steps while the fork's teacher-forced greedy metric ROSE).
            w_anchor = valid if self.anchor_full_weight else w
            # Reuses terms["target_probs"] (same target_logits -> same softmax) and honours the
            # loss-checkpoint flag: this call holds the same 7.24 GiB of fp32 (N,V) intermediates
            # as any other round, so it must be checkpointed too.
            base_loss, _, _ = markov_loss_shared_target(
                base_logits.reshape(-1, V), terms["target_probs"], tgt.reshape(-1),
                w_anchor.reshape(-1), self.l1_alpha, self.ce_alpha, self.grad_checkpoint_loss)
            loss = loss + lambda_base * base_loss

        # Confidence head (opt-in): BCE(conf_pred, per-slot acceptance prob).
        # Target = 1 - TV(softmax(refined_tf), target_probs) -- the speculative-sampling
        # acceptance probability of THIS slot's refined proposal (official DSpark's exact
        # label), computed from tensors the loss already produced (no extra data, no extra
        # target forward; target detached). JOINTLY TRAINED by default: the BCE gradients
        # flow through the conf input features into the head/drafter, official-style.
        if self.confidence_alpha > 0.0:
            Vc = base_logits.shape[-1]
            conf_logits = self.refiner.confidence_logits(
                h, g, prev_tok, detach=self.confidence_detach).squeeze(-1)  # (BN, block)
            with torch.no_grad():
                ref_p = torch.softmax(terms["refined"].reshape(-1, Vc).float(), dim=-1)
                tv = 0.5 * (ref_p - terms["target_probs"].reshape(-1, Vc)).abs().sum(dim=-1)
                conf_target = (1.0 - tv).clamp_(0.0, 1.0).reshape(BN, block)
            conf_bce = F.binary_cross_entropy_with_logits(
                conf_logits.float(), conf_target, reduction="none")
            conf_loss = (conf_bce * w).sum() / w.sum().clamp(min=1.0)
            loss = loss + self.confidence_alpha * conf_loss
            self.last_loss_terms["conf"] = conf_loss.detach()

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
        # base logits. LOW-RANK readout MUST go through the head's FORWARD (self.refiner(...)) so FSDP
        # all-gathers the refiner params: a DIRECT self.refiner.readout_base(h) call does NOT trigger the
        # gather, so under multi-GPU SHARD_GRAD_OP base_down/base_up (or ro_down/w2) are the sharded 0-size
        # placeholders -> "vec (0)" size-mismatch crash at eval (1-GPU NO_SHARD masks it). Full lm_head base
        # is FROZEN in the feature_extractor (NOT FSDP-wrapped) -> safe to call directly.
        if self.use_lowrank_base:
            base_logits = self.refiner(None, h, g, tgt)[0]       # [0]=base=readout_base(h); prev irrelevant to base
        else:
            base_logits = self._base_logits(h)                   # [BN, block, V]

        shift = self.block_convention == "shift"
        anchors = f.anchor_positions.reshape(BN, 1)
        # Absolute position of each slot's LABEL token: anchor+k (fillin) / anchor+k+1 (shift).
        label_off = 1 if shift else 0
        pos_abs = anchors + torch.arange(block, device=device).view(1, block) + label_off
        valid = pos_abs < f.seq_len
        keep = f.block_keep_mask.reshape(BN).float()
        # Prediction slots: shift -> every slot; fillin -> slots 1: (slot 0 is the anchor).
        p0 = 0 if shift else 1

        def _accept(pred):
            match = (pred[:, p0:] == tgt[:, p0:]) & valid[:, p0:]
            accept = match.float().cumprod(dim=1).sum(dim=1)
            return (accept * keep).sum() / keep.sum().clamp(min=1.0)

        drafter_accept = _accept(base_logits.argmax(dim=-1))   # parallel-argmax baseline (no head)

        # first_prev: fillin -> token BEFORE the anchor (slot 0 carries the anchor itself);
        # shift -> the anchor token (slot 0 predicts anchor+1). Matches the training-side chain.
        first_prev_idx = f.anchor_positions if shift else (f.anchor_positions - 1).clamp(min=0)
        tok_first_prev = torch.gather(input_ids, 1, first_prev_idx).reshape(BN)
        if self.refiner.sgu_enabled:
            # PARALLEL Jacobi: seed from the drafter's OWN parallel prediction (free-running, NOT
            # the ground truth), then refine ALL slots per pass. fillin pins slot 0 to the given
            # anchor; shift has no known slot to pin (all are predictions).
            K = jacobi_passes if jacobi_passes is not None else (block - 1)
            pred = base_logits.argmax(dim=-1)
            if not shift:
                pred[:, 0] = tgt[:, 0]
            for _ in range(K):
                prev_tok = pred.roll(shifts=1, dims=1)
                prev_tok[:, 0] = tok_first_prev
                new_pred = self.refiner(base_logits, h, g, prev_tok)[1].argmax(dim=-1)   # [1] = refined
                if not shift:
                    new_pred[:, 0] = tgt[:, 0]
                pred = new_pred
        else:
            # SEQUENTIAL free-running (DSpark Markov): decide slots left-to-right; each step reads
            # the already-decided prev. fillin starts at slot 1 (slot 0 = given anchor); shift
            # starts at slot 0 (its prev is the anchor token). pred init values beyond the start
            # are overwritten before the chain reads them.
            pred = tgt.clone()
            for k in range(p0, block):
                prev_tok = pred.roll(shifts=1, dims=1)
                prev_tok[:, 0] = tok_first_prev
                pred[:, k] = self.refiner(base_logits, h, g, prev_tok)[1][:, k, :].argmax(dim=-1)   # [1]=refined
        return _accept(pred), drafter_accept
