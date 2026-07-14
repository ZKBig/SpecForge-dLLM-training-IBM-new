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
                 sgu_enabled: bool = True, use_hidden: bool = True,
                 use_perpos: bool = True, use_global: bool = True, use_residual: bool = True,
                 use_mixer: bool = True, use_mlp: bool = True, mlp_ratio: int = 4,
                 input_mode: str = "concat", mixer_init: str = "eye",
                 use_norm: bool = True, use_mix_out: bool = True,
                 shared_readout: bool = False, base_readout_rank: int = 0,
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
        # Hidden sources split into two ABLATABLE flags: use_perpos (per-position drafter hidden h) and
        # use_global (block-summary g = mean_pos(h)). use_hidden=False forces token-only (both off) for
        # backward compat; otherwise each is independent -> heads: both (default), perpos-only, global-only.
        if not bool(use_hidden):
            use_perpos = use_global = False
        self.use_perpos = bool(use_perpos)
        self.use_global = bool(use_global)
        self.use_hidden = self.use_perpos or self.use_global   # any hidden source active
        self.use_residual = bool(use_residual)
        # mixer = cross-position channel mix (long-range); mlp = per-position nonlinearity.
        # Ablate independently: MLP-only head = use_mixer=False; mixer-only = use_mlp=False.
        self.use_mixer = bool(use_mixer)
        self.use_mlp = bool(use_mlp)
        # Sublayer machinery knobs (item ④ / norm ablation):
        #   use_norm=False    -> drop input_norm/post_norm/out_norm (the 3 RMSNorms). 1-layer + ReZero
        #                        block probably doesn't need pre-norm; also removes the norm that breaks fold.
        #   use_mix_out=False -> drop the post-mixer Linear, so the mixer sublayer becomes x + mix(x).
        #   use_norm=False AND use_mix_out=False  ==>  mixer = x + L*x  (pure linear token-mix;
        #                        foldable at inference as (I+L)*x -> zeros/random-init ≡ identity-init).
        self.use_norm = bool(use_norm)
        self.use_mix_out = bool(use_mix_out)
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

        # SHARED low-rank readout: reuse w2 (r->V) as the SINGLE shared 'up' for BOTH base and refined.
        # ro_down (H->r) reads the drafter hidden h so base = w2(ro_down(h)) and refined = base + w2(code)
        # = w2(ro_down(h)+code). r == the shared rank. Calibration-fit via fit_shared_readout().
        self.shared_readout = bool(shared_readout)
        if self.shared_readout:
            self.ro_down = nn.Linear(hidden_size, r, bias=False)

        # SEPARATE low-rank base readout (non-shared): the drafter base gets its OWN low-rank readout at
        # base_readout_rank (e.g. 512/1024) DECOUPLED from the refiner's markov_rank w2. Lets the base keep
        # a HIGH rank (for accept) while the refiner stays small (for latency). base = base_up(base_down(h)),
        # calibration-fit via fit_base_readout(). Mutually exclusive with shared_readout.
        self.base_readout_rank = int(base_readout_rank)
        # rank >= hidden_size: a full-rank readout has NO compression (up(down(h)) can equal lm_head(h)
        # EXACTLY), so there's zero latency point AND no reason to train a 638M readout -> fall back to the
        # full FROZEN lm_head base (no module, no training). Makes --lowrank-base-rank 4096 the exact
        # full-rank control (base = frozen lm_head, identical to non-low-rank mode).
        if self.base_readout_rank >= hidden_size:
            self.base_readout_rank = 0
        assert not (self.shared_readout and self.base_readout_rank > 0), \
            "shared_readout and base_readout_rank are mutually exclusive"
        if self.base_readout_rank > 0:
            self.base_down = nn.Linear(hidden_size, self.base_readout_rank, bias=False)
            self.base_up = nn.Linear(self.base_readout_rank, vocab_size, bias=False)

        if self.sgu_enabled:
            n_hid = int(self.use_perpos) + int(self.use_global)   # 0 / 1 / 2 active hidden sources
            if n_hid == 2 and self.input_mode == "add":
                self.down_hg = nn.Linear(2 * hidden_size, r, bias=False)  # both + add: fused [h;g] -> r (unchanged)
            elif self.use_hidden:  # concat (any n_hid), OR add with a single hidden source
                if self.use_perpos:
                    self.down_h = nn.Linear(hidden_size, r, bias=False)  # h -> r
                if self.use_global:
                    self.down_g = nn.Linear(hidden_size, r, bias=False)  # g (block summary) -> r
                if self.input_mode == "concat":
                    self.in_proj = nn.Linear((1 + n_hid) * r, r, bias=False)  # markov + n_hid hidden -> r
                # add + single source: no in_proj (x = down_x(x) + markov_latent)
            else:  # token-only
                self.in_proj = nn.Linear(r, r, bias=False)
            if self.use_mixer:
                if self.use_norm:
                    self.input_norm = RMSNorm(r)
                self.mix = ChannelWiseCausalMix(r, block_size, init=self.mixer_init,
                                                initializer_range=initializer_range)
                if self.use_mix_out:
                    self.mix_out = nn.Linear(r, r, bias=False)
            if self.use_mlp:
                if self.use_norm:
                    self.post_norm = RMSNorm(r)
                self.mlp = RDimMLP(r, r * mlp_ratio)
            if self.use_norm:
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
        # SHARED mode: w2 (the shared 'up') is calibration-fit later (large ~lm_head scale), so the code
        # W1[prev] MUST start at ZERO -> code=0 at init -> refined = base (else up(W1[prev]) blows up).
        if self.shared_readout:
            nn.init.zeros_(self.w1.weight)                   # code starts 0 -> refined == base at init
            nn.init.normal_(self.ro_down.weight, mean=0.0, std=std)   # overwritten by fit_shared_readout
        else:
            nn.init.normal_(self.w1.weight, mean=0.0, std=std)   # markov_w1  (== DeepSpec)
        nn.init.normal_(self.w2.weight, mean=0.0, std=std)   # markov_w2 (== DeepSpec; overwritten by fit if shared)
        if self.base_readout_rank > 0:
            nn.init.normal_(self.base_down.weight, mean=0.0, std=std)  # both overwritten by fit_base_readout;
            nn.init.zeros_(self.base_up.weight)                        # zeros -> base=0 until calibration-fit
        if self.sgu_enabled:
            for _name in ("down_hg", "down_h", "down_g", "in_proj"):   # whichever input projections exist
                _mod = getattr(self, _name, None)
                if _mod is not None:
                    nn.init.normal_(_mod.weight, mean=0.0, std=std)
            if self.use_mixer and self.use_mix_out:
                nn.init.normal_(self.mix_out.weight, mean=0.0, std=std)
                # mixer L stays init'd per mixer_init; norms (if any) stay ones; res_gate stays zero.
            if self.use_mlp:
                nn.init.normal_(self.mlp.gate_proj.weight, mean=0.0, std=std)
                nn.init.normal_(self.mlp.up_proj.weight, mean=0.0, std=std)
                nn.init.normal_(self.mlp.down_proj.weight, mean=0.0, std=std)

    def _refine(self, markov_latent, h, g):
        """r-dim SGU refinement over the block. markov_latent/h/g are per-(BN,block)."""
        if self.use_hidden and self.input_mode == "add":
            if hasattr(self, "down_hg"):                        # both sources -> fused [h;g]
                hid = self.down_hg(torch.cat([h, g], dim=-1))
            else:                                               # single source (perpos-only or global-only)
                hid = self.down_h(h) if self.use_perpos else self.down_g(g)
            x = hid + markov_latent                             # add token residual-style
        elif self.use_hidden:  # concat: active hidden projections + markov_latent -> in_proj
            parts = []
            if self.use_perpos:
                parts.append(self.down_h(h))
            if self.use_global:
                parts.append(self.down_g(g))
            parts.append(markov_latent)
            x = self.in_proj(torch.cat(parts, dim=-1))
        else:  # token-only
            x = self.in_proj(markov_latent)
        if self.use_mixer:
            m = self.input_norm(x) if self.use_norm else x
            m = self.mix(m)
            if self.use_mix_out:
                m = self.mix_out(m)
            x = x + m                                           # no-norm + no-mix_out => x + L*x (foldable)
        if self.use_mlp:
            p = self.post_norm(x) if self.use_norm else x
            x = x + self.mlp(p)                                 # MLP sublayer (per-position)
        return self.out_norm(x) if self.use_norm else x

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

    def readout_base(self, h):
        """Low-rank drafter base logits from hidden h. SHARED: w2(ro_down(h)) (reuses the refiner's up).
        SEPARATE: base_up(base_down(h)) (own readout at base_readout_rank, decoupled from the refiner)."""
        if self.shared_readout:
            return self.w2(self.ro_down(h))
        return self.base_up(self.base_down(h))

    def forward(self, base_logits, h, g, prev_token_ids):
        """Returns (base_logits, refined_logits).
        SHARED: base is computed INTERNALLY via the shared readout (the base_logits arg is ignored) so both
        base and refined go through the SAME w2 -> refined = w2(ro_down(h)+code). Else: base_logits is the
        full frozen lm_head(h) passed in. Computing base inside forward keeps its grad under FSDP."""
        base = self.readout_base(h) if (self.shared_readout or self.base_readout_rank > 0) else base_logits
        return base, base + self.bias(h, g, prev_token_ids)

    @torch.no_grad()
    def fit_shared_readout(self, cov, lm_head_weight, rotate=False):
        """Calibration / data-PCA init of the SHARED readout: ro_down = P^T, w2 = W @ P, where P = top-r
        eigenvectors of the activation covariance C = sum h h^T. Then base = w2(ro_down(h)) = W(P P^T)h ~=
        lm_head(h) (h lives in span(P)). Same fold as LowRankReadout.fit_from_covariance, applied to (ro_down, w2)."""
        assert self.shared_readout, "fit_shared_readout requires shared_readout=True"
        r = self.markov_rank
        dev = lm_head_weight.device
        evals, evecs = torch.linalg.eigh(cov.float().to(dev))          # ascending, symmetric PSD
        P = evecs[:, -r:]                                              # top-r principal directions (H, r)
        if rotate:
            Q, _ = torch.linalg.qr(torch.randn(r, r, device=dev, dtype=P.dtype))
            P = P @ Q
        up = lm_head_weight.detach().float() @ P                       # (V, r) = W P
        self.ro_down.weight.copy_(P.t().to(self.ro_down.weight))       # (r, H) = P^T
        self.w2.weight.copy_(up.to(self.w2.weight))                    # (V, r) = shared up

    @torch.no_grad()
    def fit_base_readout(self, cov, lm_head_weight, rotate=False):
        """Calibration / data-PCA init of the SEPARATE base readout (non-shared): base_down = P^T,
        base_up = W @ P with P = top-(base_readout_rank) eigvecs of C. => base_up(base_down(h)) ~= lm_head(h).
        Same fit as fit_shared_readout, applied to (base_down, base_up) at the (larger) base rank."""
        assert self.base_readout_rank > 0, "fit_base_readout requires base_readout_rank > 0"
        r = self.base_readout_rank
        dev = lm_head_weight.device
        evals, evecs = torch.linalg.eigh(cov.float().to(dev))          # ascending, symmetric PSD
        P = evecs[:, -r:]                                              # top-r principal directions (H, r)
        if rotate:
            Q, _ = torch.linalg.qr(torch.randn(r, r, device=dev, dtype=P.dtype))
            P = P @ Q
        up = lm_head_weight.detach().float() @ P                       # (V, r) = W P
        self.base_down.weight.copy_(P.t().to(self.base_down.weight))   # (r, H) = P^T
        self.base_up.weight.copy_(up.to(self.base_up.weight))          # (V, r)


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
    V = target_logits.shape[-1]           # base_logits is None in shared mode; target_logits has the same V
    w = build_loss_weight_mask(loss_weight, loss_decay_gamma).reshape(-1)  # == DSpark decayed mask
    tgt_l = target_logits.reshape(-1, V)
    tgt_t = target_tokens.reshape(-1).long()

    base_tf, refined_tf = head(base_logits, h, g, gt_prev_ids)  # teacher-forced pass; base_tf = shared base or echo
    loss_tf, ce_tf, l1_tf = markov_style_loss(refined_tf.reshape(-1, V), tgt_l, tgt_t, w,
                                              l1_alpha, ce_alpha)
    total = loss_tf
    loss_cons = refined_tf.new_zeros(())   # base_logits may be None (shared); refined_tf is always a tensor
    if consistency_weight > 0.0:
        # Seed prev = drafter's own argmax (== first Jacobi pass at inference), KEEPING the two known
        # slots (token-before-block, anchor) from the teacher-forced prev. Matches dflash_refiner_cotrain.
        block = base_tf.shape[1]                                 # base_tf = shared base or echoed lm_head(h)
        base_pred = base_tf.detach().argmax(dim=-1)              # (BN, block) drafter argmax
        seed_prev = gt_prev_ids.clone()
        if block > 2:
            seed_prev[:, 2:] = base_pred[:, 1:block - 1]         # slots 0,1 stay correct
        _, refined_seed = head(base_logits, h, g, seed_prev)
        loss_cons, _, _ = markov_style_loss(refined_seed.reshape(-1, V), tgt_l, tgt_t, w,
                                            l1_alpha, ce_alpha)
        total = loss_tf + consistency_weight * loss_cons
    return total, {"tf": loss_tf.detach(), "ce": ce_tf, "l1": l1_tf, "cons": loss_cons.detach(),
                   "refined": refined_tf,   # reused for the accuracy metric (no recompute)
                   "base": base_tf}         # shared: head-computed base (=w2(ro_down(h))); else: echoed lm_head(h)
