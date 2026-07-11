#!/usr/bin/env python3
# coding=utf-8
"""Unified low-rank refiner head for the RIPPLE-vs-DSpark head-to-head, in ONE module.

The head produces a per-position logit BIAS added to the base readout  base = lm_head(h):

    bias = W2( markov_latent  +  res_gate * SGU_refine([down_h(h); down_g(g); markov_latent]) )
    refined_logits = base + bias

where  markov_latent = W1[prev_token]  (r-dim).  W1 [V,r] / W2 [r,V] are the SAME projections
DSpark's Markov head uses -- so DSpark's VanillaMarkov is the STRICT SPECIAL CASE of this head
with the SGU turned off (bias = W2(W1[prev]) exactly).  The r-bottleneck SGU (channel-wise causal
mixer + MLP) lives BETWEEN W1(down) and W2(up), so the whole refiner runs in r=256 dim instead of
H=4096 -> ~16x lighter refiner core, total head ~= DSpark's Markov (~80M, dominated by W1+W2).

Flags give the full ablation matrix (all trained in the SAME SpecForge framework -> fair, only the
head differs):
  * sgu_enabled=False              -> pure Markov (faithful DSpark VanillaMarkov).
  * sgu_enabled=True, use_residual -> ReZero residual so it STARTS as Markov, learns SGU on top
    (Markov = exact special case at res_gate=0). This is the residual ablation.
  * use_hidden=True/False          -> three sources [h,g,prev] vs token-only [prev].

Training loss `markov_style_loss` reproduces DSpark's recipe: 0.9 * L1(TV of draft vs target dist)
+ 0.1 * CE(draft vs target token). Use the SAME loss for BOTH heads so the comparison isolates the
head architecture (NOT the loss).

Inference (not here; wire into the generate/eval path): pure-Markov -> sequential (exact prev token,
faithful DSpark); SGU -> parallel K-pass Jacobi (prev = previous pass's token).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        h = x.float()
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight.float() * h).type_as(x)


class ChannelWiseCausalMix(nn.Module):
    """Per-channel learned lower-triangular token mix (SGU's mixer), at `channels` dim.
    u[b,k,c] = sum_{j<=k} L[c,k,j] x[b,j,c]. + tril mask.

    init: 'eye'    -> identity token-mix (each pos attends to itself) at init;
          'zeros'  -> no mixing at init, rely on the sublayer residual for identity
                      (then optionally fold at inference: L += I on the diagonal);
          'random' -> normal(0, initializer_range)."""

    def __init__(self, channels: int, block_size: int, init: str = "eye",
                 initializer_range: float = 0.02):
        super().__init__()
        if init == "eye":
            L0 = torch.eye(block_size).unsqueeze(0).repeat(channels, 1, 1)
        elif init == "zeros":
            L0 = torch.zeros(channels, block_size, block_size)
        elif init == "random":
            L0 = torch.randn(channels, block_size, block_size) * initializer_range
        else:
            raise ValueError(f"mixer init must be eye/zeros/random, got {init!r}")
        self.L = nn.Parameter(L0)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):  # x: (BN, block, C)
        Lm = self.L * self.tril  # (C, block, block) causal
        return torch.bmm(Lm, x.permute(2, 1, 0)).permute(2, 1, 0)


class RDimMLP(nn.Module):
    """SwiGLU MLP in the r-dim bottleneck."""

    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden, bias=False)
        self.up_proj = nn.Linear(dim, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dim, bias=False)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))


class HybridRefinerHead(nn.Module):
    """See module docstring. Produces refined_logits = base_logits + bias."""

    def __init__(self, vocab_size: int, hidden_size: int, block_size: int, markov_rank: int = 256,
                 sgu_enabled: bool = True, use_hidden: bool = True, use_residual: bool = True,
                 use_mixer: bool = True, use_mlp: bool = True, mlp_ratio: int = 4,
                 input_mode: str = "concat", mixer_init: str = "eye",
                 initializer_range: float = 0.02):
        super().__init__()
        r = markov_rank
        self.vocab_size = int(vocab_size)
        self.hidden_size = int(hidden_size)
        self.markov_rank = r
        self.initializer_range = float(initializer_range)
        # NOTE: DSpark's released Qwen3-8B also trains a confidence head (alpha=1.0) for inference
        # SCHEDULING; we deliberately OMIT it from BOTH heads. It's orthogonal to the fixed-block
        # accept-rate comparison, and excluding it identically keeps the head-vs-head test clean.
        self.sgu_enabled = bool(sgu_enabled)
        self.use_hidden = bool(use_hidden)
        self.use_residual = bool(use_residual)
        # mixer = cross-position channel mix (long-range); mlp = per-position nonlinearity.
        # Ablate independently: MLP-only head = use_mixer=False; mixer-only = use_mlp=False.
        self.use_mixer = bool(use_mixer)
        self.use_mlp = bool(use_mlp)
        # input_mode: how h/g enter the r-dim refiner.
        #   'concat' -> in_proj( [down_h(h); down_g(g); W1[prev]] )              (3r -> r)
        #   'add'    -> down_hg( [h;g] ) + W1[prev]   (single 2H->r down, token added residual-style)
        self.input_mode = str(input_mode)
        assert self.input_mode in ("concat", "add"), self.input_mode
        self.mixer_init = str(mixer_init)
        if self.sgu_enabled:
            assert self.use_mixer or self.use_mlp, "SGU needs at least one of mixer / MLP."

        # Markov projections -- shared by the Markov special case AND the SGU (W1=down, W2=up).
        self.w1 = nn.Embedding(vocab_size, r)          # prev token -> r  (down / Markov embedding)
        self.w2 = nn.Linear(r, vocab_size, bias=False)  # r -> V           (up / readout)

        if self.sgu_enabled:
            if self.use_hidden and self.input_mode == "add":
                self.down_hg = nn.Linear(2 * hidden_size, r, bias=False)  # [h;g] -> r; token added
            elif self.use_hidden:  # concat
                self.down_h = nn.Linear(hidden_size, r, bias=False)  # h -> r
                self.down_g = nn.Linear(hidden_size, r, bias=False)  # g (block summary) -> r
                self.in_proj = nn.Linear(3 * r, r, bias=False)
            else:  # token-only
                self.in_proj = nn.Linear(r, r, bias=False)
            if self.use_mixer:
                self.input_norm = RMSNorm(r)
                self.mix = ChannelWiseCausalMix(r, block_size, init=self.mixer_init,
                                                initializer_range=initializer_range)
                self.mix_out = nn.Linear(r, r, bias=False)
            if self.use_mlp:
                self.post_norm = RMSNorm(r)
                self.mlp = RDimMLP(r, r * mlp_ratio)
            self.out_norm = RMSNorm(r)
            if self.use_residual:
                self.res_gate = nn.Parameter(torch.zeros(1))  # ReZero: starts as pure Markov

        self.reset_parameters()

    def reset_parameters(self):
        """FAITHFUL to DeepSpec: Qwen3PreTrainedModel._init_weights -> every Linear/Embedding is
        normal_(0, initializer_range). This makes the Markov bias W2(W1[prev]) start ~0 (product of
        two small matrices), i.e. the head starts as 'no correction'. PyTorch defaults (Embedding
        std=1, Linear kaiming) would give a huge initial bias -- do NOT rely on them."""
        std = self.initializer_range
        nn.init.normal_(self.w1.weight, mean=0.0, std=std)   # markov_w1  (== DeepSpec)
        nn.init.normal_(self.w2.weight, mean=0.0, std=std)   # markov_w2  (== DeepSpec)
        if self.sgu_enabled:
            if self.use_hidden and self.input_mode == "add":
                nn.init.normal_(self.down_hg.weight, mean=0.0, std=std)
            elif self.use_hidden:  # concat
                nn.init.normal_(self.down_h.weight, mean=0.0, std=std)
                nn.init.normal_(self.down_g.weight, mean=0.0, std=std)
                nn.init.normal_(self.in_proj.weight, mean=0.0, std=std)
            else:  # token-only
                nn.init.normal_(self.in_proj.weight, mean=0.0, std=std)
            if self.use_mixer:
                nn.init.normal_(self.mix_out.weight, mean=0.0, std=std)
                # mixer L stays eye-init (identity token-mix); norms stay ones; res_gate stays zero.
            if self.use_mlp:
                nn.init.normal_(self.mlp.gate_proj.weight, mean=0.0, std=std)
                nn.init.normal_(self.mlp.up_proj.weight, mean=0.0, std=std)
                nn.init.normal_(self.mlp.down_proj.weight, mean=0.0, std=std)

    def _refine(self, markov_latent, h, g):
        """r-dim SGU refinement over the block. markov_latent/h/g are per-(BN,block)."""
        if self.use_hidden and self.input_mode == "add":
            x = self.down_hg(torch.cat([h, g], dim=-1)) + markov_latent  # add token residual-style
        elif self.use_hidden:  # concat
            x = self.in_proj(torch.cat([self.down_h(h), self.down_g(g), markov_latent], dim=-1))
        else:  # token-only
            x = self.in_proj(markov_latent)
        if self.use_mixer:
            x = x + self.mix_out(self.mix(self.input_norm(x)))  # mixer sublayer (cross-position)
        if self.use_mlp:
            x = x + self.mlp(self.post_norm(x))                 # MLP sublayer (per-position)
        return self.out_norm(x)

    def compute_latent(self, h, g, prev_token_ids):
        """The r-dim latent whose W2-projection is the bias. Markov = latent = W1[prev]."""
        markov_latent = self.w1(prev_token_ids.long())      # (BN, block, r)
        if not self.sgu_enabled:
            return markov_latent                            # pure Markov (DSpark VanillaMarkov)
        refined = self._refine(markov_latent, h, g)
        if self.use_residual:
            return markov_latent + self.res_gate * refined  # starts as Markov, learns SGU on top
        return refined

    def bias(self, h, g, prev_token_ids):
        return self.w2(self.compute_latent(h, g, prev_token_ids))  # (BN, block, V)

    def forward(self, base_logits, h, g, prev_token_ids):
        """base_logits=(BN,block,V)=lm_head(h). h,g=(BN,block,H). prev_token_ids=(BN,block)."""
        return base_logits + self.bias(h, g, prev_token_ids)


def markov_style_loss(draft_logits, target_logits, target_tokens, loss_weight,
                      l1_alpha: float = 0.9, ce_alpha: float = 0.1):
    """DSpark-style loss (loss.py): l1_alpha * L1(softmax(draft) - softmax(target)) [= TV distance,
    which is 1 - accept_rate] + ce_alpha * CE(draft, target_token). Use for BOTH heads.

    draft_logits, target_logits: (N, V).  target_tokens: (N,) long.  loss_weight: (N,) float 0/1.
    Returns (loss, ce_detached, l1_detached)."""
    denom = loss_weight.sum().clamp(min=1.0)
    ce = (F.cross_entropy(draft_logits, target_tokens, reduction="none") * loss_weight).sum() / denom
    dp = draft_logits.float().softmax(dim=-1)
    tp = target_logits.float().softmax(dim=-1)
    l1 = ((dp - tp).abs().sum(dim=-1) * loss_weight).sum() / denom  # sum|dp-tp| = 2*TV
    loss = ce_alpha * ce + l1_alpha * l1
    return loss, ce.detach(), l1.detach()


def build_loss_weight_mask(eval_mask, loss_decay_gamma=4.0):
    """FAITHFUL to DSpark `_build_loss_weight_mask`: eval_mask(float) * exp(-position/gamma) over the
    block axis (last dim). gamma=4.0 is DSpark's Qwen3-8B setting (pos0 weight 1.0 -> pos6 ~0.22).
    Pass loss_decay_gamma=None/<=0 to disable."""
    w = eval_mask.float()
    if loss_decay_gamma is not None and loss_decay_gamma > 0:
        block = w.shape[-1]
        pos = torch.arange(block, device=w.device, dtype=torch.float32)
        decay = torch.exp(-pos / float(loss_decay_gamma)).view(*([1] * (w.dim() - 1)), block)
        w = w * decay
    return w


def combined_training_loss(head, base_logits, h, g, gt_prev_ids, target_logits, target_tokens,
                           loss_weight, consistency_weight: float = 0.0,
                           l1_alpha: float = 0.9, ce_alpha: float = 0.1, loss_decay_gamma: float = 4.0):
    """Teacher-forced markov_style_loss  +  optional CONSISTENCY term (our Jacobi self-consistency):
    a second pass conditioned on the DRAFTER's OWN argmax prev (what the first Jacobi pass sees at
    inference), so the refiner is trained robust to imperfect self-predicted prev.

      total = loss_tf  +  consistency_weight * loss_seed

    loss_decay_gamma=4.0 reproduces DSpark's per-position loss decay (exp(-pos/gamma)); set None to
    disable. consistency_weight=0 -> pure teacher-forced (unify to DSpark). Shapes: base_logits/
    target_logits (BN,block,V); h/g (BN,block,H); gt_prev_ids/target_tokens (BN,block); loss_weight
    (BN,block) = eval_mask."""
    V = base_logits.shape[-1]
    w = build_loss_weight_mask(loss_weight, loss_decay_gamma).reshape(-1)  # == DSpark decayed mask
    tgt_l = target_logits.reshape(-1, V)
    tgt_t = target_tokens.reshape(-1).long()

    refined_tf = head(base_logits, h, g, gt_prev_ids)  # teacher-forced pass
    loss_tf, ce_tf, l1_tf = markov_style_loss(refined_tf.reshape(-1, V), tgt_l, tgt_t, w,
                                              l1_alpha, ce_alpha)
    total = loss_tf
    loss_cons = base_logits.new_zeros(())
    if consistency_weight > 0.0:
        # Seed prev = drafter's own argmax (== first Jacobi pass at inference), KEEPING the two known
        # slots (token-before-block, anchor) from the teacher-forced prev. Matches dflash_refiner_cotrain.
        block = base_logits.shape[1]
        base_pred = base_logits.detach().argmax(dim=-1)          # (BN, block) drafter argmax
        seed_prev = gt_prev_ids.clone()
        if block > 2:
            seed_prev[:, 2:] = base_pred[:, 1:block - 1]         # slots 0,1 stay correct
        refined_seed = head(base_logits, h, g, seed_prev)
        loss_cons, _, _ = markov_style_loss(refined_seed.reshape(-1, V), tgt_l, tgt_t, w,
                                            l1_alpha, ce_alpha)
        total = loss_tf + consistency_weight * loss_cons
    return total, {"tf": loss_tf.detach(), "ce": ce_tf, "l1": l1_tf, "cons": loss_cons.detach(),
                   "refined": refined_tf}  # refined_tf reused for the accuracy metric (no recompute)
