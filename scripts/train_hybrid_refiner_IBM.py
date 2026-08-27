#!/usr/bin/env python3
# coding=utf-8
"""CO-TRAIN the AR refiner together with the DFlash drafter backbone.

This is the co-train ABLATION counterpart of `train_dflash_refiner.py` (frozen drafter).
It is a SEPARATE script and shares no code path with the frozen trainer beyond imports.

Differences vs the frozen trainer:
  - feature extractor is `CoTrainFeatureExtractor` (drafter forward keeps autograd);
  - model is `OnlineDFlashRefinerCoTrain` (drafter backbone un-frozen; target lm_head/embed frozen);
  - BOTH the refiner head AND the drafter backbone are FSDP-wrapped and optimized;
  - checkpoints save refiner_state_dict + draft_state_dict (+ optimizer/scheduler).

The drafter is still initialized from --dflash-model-path (e.g. z-lab/Qwen3-8B-DFlash-b16),
so frozen vs co-train differ only in whether that backbone keeps training -> a clean ablation.
"""

import argparse
import logging
import math
import os
import sys
import time
import warnings
from datetime import timedelta
from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from accelerate.utils import set_seed
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy, StateDictType
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from datasets import load_dataset
from specforge.args import TrackerArgs
from specforge.core.dflash_refiner_cotrain import (
    CoTrainBF16Optimizer,
    CoTrainFeatureExtractor,
)
from specforge.core.hybrid_refiner_cotrain import OnlineHybridRefinerCoTrain
from specforge.data import build_eagle3_dataset, prepare_dp_dataloaders
from specforge.distributed import destroy_distributed, get_dp_group, init_distributed
from specforge.modeling.draft.dflash import DFlashDraftModel
from specforge.modeling.target.dflash_target_model import get_dflash_target_model
from specforge.modeling.target.target_utils import TargetEmbeddingsAndHead
from specforge.optimizer import BF16Optimizer
from specforge.tracker import create_tracker, get_tracker_class
from specforge.utils import get_last_checkpoint, print_on_rank0, print_with_rank

# rank-0 log file handle (opened in main once output_dir is known). log_print writes to BOTH stdout
# (flushed, so tqdm's \r bar can't hide it) AND this file (so `grep` always recovers everything),
# bypassing the logging module entirely (which silently dropped output before).
_LOGF = None


def log_print(msg):
    if int(os.environ.get("RANK", "0")) != 0:
        return
    print(msg, flush=True)
    if _LOGF is not None:
        _LOGF.write(str(msg) + "\n")
        _LOGF.flush()


def parse_args():
    parser = argparse.ArgumentParser(description="Co-train DFlash AR Refiner + drafter")

    model_group = parser.add_argument_group("model")
    model_group.add_argument("--target-model-path", type=str, required=True)
    model_group.add_argument(
        "--dflash-model-path",
        type=str,
        required=True,
        help="Path/HF-repo of the DFlash drafter to INITIALIZE from (then co-trained), "
        "e.g. z-lab/Qwen3-8B-DFlash-b16. With --random-init-drafter only its CONFIG (arch) is used.",
    )
    model_group.add_argument(
        "--random-init-drafter", action="store_true",
        help="Setting 2 (DSpark-style): build the drafter RANDOMLY from --dflash-model-path's config "
        "(no checkpoint) and train from scratch. Needs lots of data (full Open-PerfectBlend).",
    )
    model_group.add_argument(
        "--attention-backend",
        type=str,
        default="flex_attention",
        choices=["eager", "sdpa", "flex_attention"],
    )
    model_group.add_argument("--num-anchors", type=int, default=512)
    model_group.add_argument("--trust-remote-code", action="store_true")
    model_group.add_argument(
        "--embedding-key", type=str, default=None,
        help="Embedding weight key in the target model.",
    )
    model_group.add_argument(
        "--lm-head-key", type=str, default=None,
        help="LM head weight key in the target model.",
    )

    refiner_group = parser.add_argument_group("refiner")
    # ---- hybrid head (RIPPLE-vs-DSpark) ----
    refiner_group.add_argument("--markov-rank", type=int, default=256,
                               help="r: Markov W1/W2 rank AND SGU bottleneck width (DSpark uses 256).")
    refiner_group.add_argument("--markov-only", action="store_true",
                               help="Train the pure Markov head (DSpark VanillaMarkov special case; SGU off).")
    refiner_group.add_argument("--anchor-full-weight", action="store_true",
                               help="Apply the lambda_base drafter-anchor with UNDECAYED weights (validity only), "
                               "anchoring deep slots at full strength. The head's learning loss keeps its decay. "
                               "Fixes the depth-graded standalone-drafter erosion measured on the b7shift arms "
                               "(accept@6 0.52->0.20 by 40k steps under the decayed anchor, whose slot-6 weight is 0.22).")
    refiner_group.add_argument("--confidence-alpha", type=float, default=0.0,
                               help="Weight of the confidence head's BCE loss (0 = OFF: no module, no loss, "
                               "bit-identical to pre-confidence runs). JOINTLY TRAINED like official DSpark "
                               "(alpha=1.0 there): the BCE gradients flow through the conf input features "
                               "into the head/drafter backbone (auxiliary 'will this be accepted?' task).")
    refiner_group.add_argument("--confidence-mode", choices=("trajectory", "dspark"), default="trajectory",
                               help="Conf input. trajectory (ours): the refiner's causal-mixed latent -- slot k "
                               "sees ALL previous block tokens (DSpark's 1-token input is measurably "
                               "overconfident at depth: official 14B ECE 0.019@0 -> 0.101@6). dspark "
                               "(official): cat([h, W1[prev]]) into Linear(H+r,1), for faithful repro arms.")
    refiner_group.add_argument("--confidence-detach", action="store_true",
                               help="Cut the conf gradients at its input features: train ONLY the conf module "
                               "(inference-only calibration; the main objective is untouched). Default OFF = "
                               "joint training.")
    refiner_group.add_argument("--block-convention", choices=("fillin", "shift"), default="fillin",
                               help="Block label alignment. fillin (default, z-lab style): slot k predicts "
                               "the token AT anchor+k; anchor slot excluded -> block_size-1 drafts. Matches "
                               "z-lab DFlash warm starts (b16 arms). shift (deepseek/DeepSpec style): slot k "
                               "predicts the token AFTER its position; every slot supervised -> block_size "
                               "drafts. REQUIRED when warm-starting from deepseek-ai/dflash_*_block7: those "
                               "are shift-native, and retrofitting them to fillin permanently costs deep-"
                               "position accept (measured: standalone drafter capped at ~3.9 vs native 5.43).")
    refiner_group.add_argument("--no-hidden", action="store_true",
                               help="Token-only SGU input (drop h/g; use prev-token W1 only).")
    refiner_group.add_argument("--no-perpos", action="store_true",
                               help="Drop the per-position hidden h from the SGU input (keep global g + token). "
                                    "-> global-only head. Ignored if --no-hidden.")
    refiner_group.add_argument("--no-global", action="store_true",
                               help="Drop the block-summary g (=mean_pos h) from the SGU input (keep per-position "
                                    "h + token). -> perpos-only head. Ignored if --no-hidden.")
    refiner_group.add_argument("--no-residual", action="store_true",
                               help="Drop the outer ReZero Markov-residual (Markov no longer the init).")
    refiner_group.add_argument("--input-scale", type=float, default=1.0,
                               help="INIT-ONLY initial-latent scale: scales the input-projection init std by "
                                    "this (x starts smaller, then trains at normal rate). 1.0 = unchanged; "
                                    "~0.32 makes an ADD head's initial x match a CONCAT head's -> tests whether "
                                    "concat's edge is the small INITIAL latent, not the in_proj itself.")
    refiner_group.add_argument("--input-relu", action="store_true",
                               help="Add a ReLU between the first projections (down_*/W1) and in_proj (concat "
                                    "mode). Breaks the linear fold -> concat becomes a genuine 2-layer MLP "
                                    "(more expressive than add), motivating the double projection. No-op for add.")
    refiner_group.add_argument("--no-mixer", action="store_true", help="MLP-only SGU (no channel mixer).")
    refiner_group.add_argument("--no-mlp", action="store_true", help="Mixer-only SGU (no MLP).")
    refiner_group.add_argument("--mlp-ratio", type=int, default=4, help="SGU MLP hidden = ratio * r.")
    refiner_group.add_argument("--input-mode", type=str, default="concat", choices=["concat", "add"],
                               help="'concat' in_proj[down_h;down_g;W1] vs 'add' down_hg[h;g]+W1.")
    refiner_group.add_argument("--mixer-init", type=str, default="eye", choices=["eye", "zeros", "random"])
    refiner_group.add_argument("--no-sublayer-norm", action="store_true",
                               help="Drop input_norm/post_norm/out_norm (the 3 RMSNorms). 1-layer + ReZero "
                               "block likely needs no pre-norm; also removes the norm that blocks fold(I).")
    refiner_group.add_argument("--no-mix-out", action="store_true",
                               help="Drop the post-mixer Linear -> mixer sublayer becomes x + mix(x). "
                               "With --no-sublayer-norm this makes the mixer exactly x + L*x (foldable).")
    refiner_group.add_argument("--l1-alpha", type=float, default=0.9, help="DSpark L1/TV weight.")
    refiner_group.add_argument("--ce-alpha", type=float, default=0.1, help="DSpark CE weight.")
    refiner_group.add_argument("--no-cotrain-drafter", action="store_true",
                               help="Freeze the drafter backbone (train head only).")
    refiner_group.add_argument(
        "--window-size", type=int, default=0,
        help="Local context window the refiner cross-attends to (0 = pure v1).",
    )
    refiner_group.add_argument("--num-refiner-layers", type=int, default=1)
    refiner_group.add_argument(
        "--mlp-intermediate", type=int, default=None,
        help="Refiner MLP intermediate size. None = Qwen3 default (full); >0 = shrink to this; "
        "<=0 = drop the MLP entirely. For MLP-size ablation / cost reduction.",
    )
    refiner_group.add_argument(
        "--lowrank-lmhead-rank", type=int, default=0,
        help="If >0, read out as base lm_head(h) + low-rank(refined-h) with this rank (Domino-style "
        "base+correction; only the small delta is low-ranked so argmax is preserved). 0 = full lm_head.",
    )
    refiner_group.add_argument(
        "--lowrank-lmhead-init", type=str, default="svd", choices=["svd", "zero"],
        help="Low-rank head init: 'svd' = warm-start so up@down ~= lm_head (starts ~equivalent to "
        "full); 'zero' = correction starts at 0 (readout == drafter at step 0).",
    )
    refiner_group.add_argument(
        "--lowrank-lmhead-shared-rank", type=int, default=0,
        help="If >0, use a SHARED low-rank lm_head of this rank for BOTH the base readout LR(h) AND the "
        "refined readout LR(refined) -- NO full lm_head anywhere (cracks the base-lm_head latency floor). "
        "Exclusive with --lowrank-lmhead-rank (that only low-ranks the delta). Try 512.",
    )
    refiner_group.add_argument(
        "--lowrank-base-rank", type=int, default=0,
        help="SEPARATE (non-shared) low-rank base readout of this rank for the DRAFTER base ONLY, while the "
        "refiner keeps --markov-rank. Decouples base accept (use 512/1024) from refiner latency (256). Uses "
        "the SAME inline calibration as --lowrank-lmhead-shared-rank. Mutually exclusive with it.",
    )
    refiner_group.add_argument(
        "--lowrank-lmhead-shared-cov", type=str, default=None,
        help="Path to an activation-covariance .pt (collect_covariance.py --calib-target hidden) to "
        "DATA-PCA-init the shared low-rank lm_head. If the path EXISTS it is loaded; if it does NOT exist "
        "(or is omitted) the covariance is COLLECTED INLINE from --lowrank-calib-batches training batches "
        "and saved to this path (if given). The whole point: hidden activations are low-rank even though "
        "the weight is not.",
    )
    refiner_group.add_argument(
        "--lowrank-calib-batches", type=int, default=32,
        help="#training batches to accumulate the activation covariance C=sum h h^T for the INLINE "
        "DATA-PCA init of the shared low-rank base readout (used when --lowrank-lmhead-shared-rank>0 and "
        "no existing --lowrank-lmhead-shared-cov file).",
    )
    refiner_group.add_argument(
        "--lowrank-calib-rotate", action="store_true",
        help="Random-rotate the PCA subspace when fitting the low-rank base readout (bf16 precision).",
    )
    refiner_group.add_argument(
        "--consistency-weight", type=float, default=0.0,
        help="Consistency loss weight. >0 supervises the refiner on the prev it actually sees during "
        "free-running Jacobi decoding (not the ground-truth prev), so it learns to converge in fewer "
        "passes. See --consistency-passes for how many rounds. 0 = off. Try ~0.5. NOTE the target of "
        "every round is the GROUND TRUTH, i.e. unrolled scheduled sampling -- NOT CLLM fixed-point "
        "consistency, which would distill each round toward the trajectory's own converged output.",
    )
    refiner_group.add_argument(
        "--no-loss-grad-checkpoint", action="store_false", dest="loss_grad_checkpoint",
        help="Disable activation checkpointing of the loss terms (ON by default). Each loss call "
        "otherwise keeps two fp32 (N,V) tensors for backward -- 3.62 GiB each at N=BN*block=6400, "
        "V=152k -- whose only product is a scalar and which are all recomputable from the bf16 "
        "logits. Checkpointing them cuts ~7.24 GiB per call: with --consistency-passes 3 that is "
        "~50 GiB -> ~18 GiB (50 GiB OOMs on an 80 GiB card), and it drops the marginal cost of a "
        "Jacobi round from 9.06 to 1.81 GiB. Recompute is exact, so results are bit-identical; the "
        "cost is one extra softmax per loss call in backward. Turn off only to measure that cost.",
    )
    refiner_group.add_argument(
        "--consistency-passes", type=int, default=3,
        help="How many free-running Jacobi rounds the consistency term covers (K). Round 0 seeds from "
        "the drafter's parallel argmax; each later round re-feeds the refiner its own argmax, matching "
        "pass K of inference (which itself defaults to block_size-1 passes). The K per-round losses are "
        "AVERAGED, so --consistency-weight keeps its meaning as K changes and K=1 is numerically "
        "identical to the old single-pass term. Costs one extra refiner forward/backward per round and "
        "peak activation memory grows ~linearly in K (each round holds its own (BN,block,V) logits plus "
        "an fp32 softmax) -- if you OOM, lower K first.",
    )
    refiner_group.add_argument(
        "--use-residual-gate",
        action="store_true",
        help="Refine via a gated residual on the drafter hidden "
        "(head_in = h[k] + gate*refined). Off = use refiner output directly.",
    )
    refiner_group.add_argument(
        "--residual-gate-init", type=float, default=0.0,
        help="Initial value of the residual gate (0 = starts == drafter / ReZero).",
    )
    refiner_group.add_argument(
        "--freeze-residual-gate", action="store_true",
        help="Do NOT train the residual gate; fix it at --residual-gate-init.",
    )
    refiner_group.add_argument(
        "--loss-decay-gamma", type=float, default=None,
        help="If set, weight block-position k loss by exp(-(k-1)/gamma).",
    )
    refiner_group.add_argument(
        "--drafter-lr-scale", type=float, default=1.0,
        help="Scale the drafter LR relative to the head LR (e.g. 0.1 = drafter learns 10x "
        "slower). !=1.0 -> a 2-param-group optimizer; =1.0 -> single shared LR.",
    )
    refiner_group.add_argument(
        "--lambda-base-start", type=float, default=1.0,
        help="Initial weight of base_loss = CE(lm_head(drafter_hidden)) = the drafter's own "
        "prediction. Anchors the co-trained drafter (Domino's trick) so training doesn't collapse. "
        "loss=(1-lambda)*refined+lambda*base. Set 0 to disable (pure refiner loss, may diverge).",
    )
    refiner_group.add_argument(
        "--lambda-base-decay-ratio", type=float, default=1.0,
        help="Fraction of total steps over which lambda_base decays linearly to 0 "
        "(1.0 = decay across the whole run, Domino default).",
    )

    refiner_group.add_argument(
        "--lambda-base-floor", type=float, default=0.0,
        help="Final lambda_base after decay (default 0.0 = old behavior). Set >0 (e.g. 0.05-0.1) to "
        "KEEP anchoring the drafter as a valid standalone drafter (good par-K seed) to the end.",
    )
    # --- refiner architecture ablations (all optional, default == current) ---
    refiner_group.add_argument(
        "--mixer-type", type=str, default="attention", choices=["attention", "sgu"],
        help="causal mixer: 'attention' (Qwen3 self-attn) | 'sgu' (channel-wise lower-triangular "
        "causal mix — cheap, fixed, block-internal; pair with --gate-type perpos).",
    )
    refiner_group.add_argument(
        "--pool-type", type=str, default="mean", choices=["mean", "xattn"],
        help="global pool token: 'mean' (mean-pool) | 'xattn' (cross-attention over the dflash output set).",
    )
    refiner_group.add_argument(
        "--gate-type", type=str, default="scalar", choices=["scalar", "perpos"],
        help="residual gate: 'scalar' (one global ReZero) | 'perpos' (input-dependent sigmoid(w.h) "
        "per position — learns WHERE to refine; needs --use-residual-gate).",
    )
    refiner_group.add_argument(
        "--zero-init-oproj", action="store_true",
        help="Stability route 1 (entry fix): zero-init the o_proj of EVERY randomly-init "
        "data-dependent attention module — the attention mixer AND the xattn pool — so each "
        "starts as a no-op and grows in from 0 instead of injecting harmful random scramble at "
        "step 0 (which slams the perpos gate shut). No-op for sgu mixer / mean pool.",
    )
    refiner_group.add_argument(
        "--gate-floor", type=float, default=0.0,
        help="Stability route 2 (trap fix): perpos gate floor eps in g = eps + (1-eps)*sigmoid(w.h). "
        "Keeps g>=eps so the mixer's gradient (~g) is never throttled to 0 (breaks gate-collapse). "
        "0.0 = off (current behavior); try 0.1.",
    )
    refiner_group.add_argument(
        "--gate-bias-init", type=float, default=-5.0,
        help="Stability route 3 (soft start): perpos gate bias init. -5.0 => g~0 (starts == drafter, "
        "but mutes the mixer's early gradient); 0.0 => g~0.5 (strong early gradient so the mixer can "
        "learn before the gate shuts; accept an initial score drop). No floor — gate may still close.",
    )

    dataset_group = parser.add_argument_group("dataset")
    dataset_group.add_argument("--train-data-path", type=str, required=True)
    dataset_group.add_argument("--eval-data-path", type=str, default=None)
    dataset_group.add_argument("--chat-template", type=str, default="qwen")
    dataset_group.add_argument("--is-preformatted", action="store_true")
    dataset_group.add_argument("--dataloader-num-workers", type=int, default=8)
    dataset_group.add_argument(
        "--build-dataset-num-proc", type=int,
        default=int(os.environ.get("SPECFORGE_DATA_NUM_PROC", 8)),
    )

    training_group = parser.add_argument_group("training")
    training_group.add_argument("--num-epochs", type=int, default=6)
    training_group.add_argument("--batch-size", type=int, default=1)
    training_group.add_argument("--learning-rate", type=float, default=6e-4)
    training_group.add_argument("--max-length", type=int, default=3072)
    training_group.add_argument("--warmup-ratio", type=float, default=0.04)
    training_group.add_argument("--max-grad-norm", type=float, default=1.0)
    training_group.add_argument("--accumulation-steps", type=int, default=1)
    training_group.add_argument("--seed", type=int, default=42)
    training_group.add_argument("--resume", action="store_true")

    output_group = parser.add_argument_group("output")
    output_group.add_argument("--output-dir", type=str, required=True)
    output_group.add_argument("--cache-dir", type=str, default="./cache")
    output_group.add_argument("--log-interval", type=int, default=50)
    output_group.add_argument("--save-interval", type=int, default=1000)
    output_group.add_argument("--keep-last-n-checkpoints", type=int, default=4,
                              help="After saving, delete older epoch_*_step_* dirs, keeping only the "
                              "last N (each ckpt ~16GB: drafter+head+8 optim shards). 0 = keep all "
                              "(WILL fill /gpfs). Latest is always kept so --resume works.")
    output_group.add_argument("--eval-interval", type=int, default=1000)

    optimization_group = parser.add_argument_group("optimization")
    optimization_group.add_argument("--tp-size", type=int, default=1)

    tracker_group = parser.add_argument_group("tracker")
    TrackerArgs.add_args(tracker_group)

    dist_group = parser.add_argument_group("distributed")
    dist_group.add_argument("--dist-timeout", type=int, default=30)

    args = parser.parse_args()

    # Fail fast HERE, before init_distributed() and before any model is loaded.
    # The tracker is otherwise only constructed at create_tracker() much further down,
    # so a missing W&B key surfaced only AFTER NCCL was up: rank 0 raised, the other 15
    # ranks then blocked 30 min in store->get() waiting for its ncclUniqueId, and all 16
    # ended up in the sleep-forever handler -> ~2h of 16 GPUs burned on a config typo
    # (job 677663). validate_args() also recovers the key from ~/.netrc, i.e. from a
    # previous `wandb login`, and prints which of the 3 ways to supply it are available.
    tracker_class = get_tracker_class(args.report_to)
    if tracker_class is not None:
        tracker_class.validate_args(parser, args)

    return args


# CPU-only group used ONLY to serialize dataset-cache construction (see _data_barrier).
_DATA_PREP_PG = None


def _data_barrier(tag):
    """Barrier for the rank-0-first dataset build. Deliberately NOT the default group.

    The default group is NCCL: a barrier there busy-waits on the GPU and is bounded by
    --dist-timeout (minutes), while rank 0 may spend hours tokenizing the corpus. So we
    use a dedicated gloo group with a long timeout instead.
    """
    global _DATA_PREP_PG
    if not dist.is_initialized():
        return
    if _DATA_PREP_PG is None:
        # Collective: every rank must reach this, which they do (build_dataloader is
        # called unconditionally by all ranks at the same point in main()).
        _DATA_PREP_PG = dist.new_group(backend="gloo", timeout=timedelta(hours=12))
    # NOT print_with_rank: that routes to logger.info, whose output never reaches the
    # job logs (not even init_distributed's "bind to device"). This barrier can hold for
    # tens of minutes while rank 0 tokenizes, so it MUST be visible or the run looks hung.
    rank = dist.get_rank()
    print(f"[data][rank{rank}] waiting at barrier: {tag}", flush=True)
    t0 = time.time()
    dist.barrier(group=_DATA_PREP_PG)
    print(f"[data][rank{rank}] released after {time.time() - t0:.1f}s: {tag}", flush=True)


def _build_split(args, tokenizer, block_size, data_path, cache_key):
    """load_dataset + tokenize + filter for one split. All three steps write to SHARED
    on-disk caches, so this must only ever run under the rank-0-first guard below."""
    dataset = load_dataset("json", data_files=data_path)["train"]
    eagle3 = build_eagle3_dataset(
        dataset=dataset,
        tokenizer=tokenizer,
        chat_template=args.chat_template,
        max_length=args.max_length,
        is_preformatted=args.is_preformatted,
        cache_dir=os.path.join(args.cache_dir, "processed_dataset"),
        cache_key=cache_key,
        num_proc=args.build_dataset_num_proc,
    )
    min_loss_tokens = 2 * block_size
    original_size = len(eagle3)
    eagle3 = eagle3.filter(lambda x: x["loss_mask"].sum() >= min_loss_tokens)
    # log_print, not print_on_rank0 -- the latter goes to logger.info, which is dropped.
    log_print(f"[data] filtered {data_path}: {original_size} -> {len(eagle3)} samples")
    return eagle3


def build_dataloader(args, tokenizer, block_size) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Build the train (+ optional eval) dataloader.

    RANK-0-FIRST. Every step below (the `load_dataset` arrow cache, the
    build_eagle3_dataset `.pkl`, and the `.filter()` cache) writes to a path derived
    only from its inputs -- so it is IDENTICAL across ranks. If all ranks run it
    concurrently they write the same files at the same time. HF's FileLock serializes
    ranks within a node but NOT across nodes (advisory flock on GPFS is node-local), so
    on a multi-node run one node's writer renames the `.incomplete` staging dir out from
    under the other's and that rank dies with FileNotFoundError / "I/O operation on
    closed file". Global rank 0 builds; everyone else waits and then hits pure cache.
    Must be *global* rank 0, not local rank -- one-writer-per-node is exactly the
    configuration that breaks.
    """
    import hashlib

    def _key(data_path):
        s = f"{data_path}-{args.max_length}-{args.chat_template}-{args.target_model_path}"
        return hashlib.md5(s.encode()).hexdigest()

    rank = dist.get_rank() if dist.is_initialized() else 0
    splits = {}

    if rank == 0:
        splits["train"] = _build_split(
            args, tokenizer, block_size, args.train_data_path, _key(args.train_data_path)
        )
        if args.eval_data_path:
            splits["eval"] = _build_split(
                args, tokenizer, block_size, args.eval_data_path, _key(args.eval_data_path)
            )
    _data_barrier("dataset cache built by rank 0")
    if rank != 0:
        # Same paths, now fully populated -> these are cache hits, no writes.
        splits["train"] = _build_split(
            args, tokenizer, block_size, args.train_data_path, _key(args.train_data_path)
        )
        if args.eval_data_path:
            splits["eval"] = _build_split(
                args, tokenizer, block_size, args.eval_data_path, _key(args.eval_data_path)
            )

    train_dataloader = prepare_dp_dataloaders(
        splits["train"],
        args.batch_size,
        num_workers=args.dataloader_num_workers,
        shuffle=True,
        process_group=get_dp_group(),
    )

    eval_dataloader = None
    if args.eval_data_path:
        eval_dataloader = prepare_dp_dataloaders(
            splits["eval"],
            args.batch_size,
            num_workers=args.dataloader_num_workers,
            shuffle=False,
            process_group=get_dp_group(),
        )
    return train_dataloader, eval_dataloader


def save_checkpoint(args, epoch, step, refiner_fsdp, draft_fsdp, optimizer, base_readout_fsdp=None):
    """Save the refiner AND the co-trained drafter (+ optimizer/scheduler state).

    Both state_dict() calls are collectives -> every rank must enter them; only rank 0
    writes. The drafter is saved as a FULL_STATE_DICT so it can be reloaded into a fresh
    DFlashDraftModel(config) for benchmarking.
    """
    save_dir = os.path.join(args.output_dir, f"epoch_{epoch}_step_{step}")
    if dist.get_rank() == 0:
        os.makedirs(save_dir, exist_ok=True)
    dist.barrier()

    with FSDP.state_dict_type(refiner_fsdp, StateDictType.FULL_STATE_DICT):
        refiner_state_dict = refiner_fsdp.state_dict()
    with FSDP.state_dict_type(draft_fsdp, StateDictType.FULL_STATE_DICT):
        draft_state_dict = draft_fsdp.state_dict()
    base_readout_state_dict = None                       # collective -> ALL ranks enter if present
    if base_readout_fsdp is not None:
        with FSDP.state_dict_type(base_readout_fsdp, StateDictType.FULL_STATE_DICT):
            base_readout_state_dict = base_readout_fsdp.state_dict()

    if dist.get_rank() == 0:
        torch.save(
            {
                "epoch": epoch,
                "global_step": step,
                "args": args,
                "window_size": args.window_size,
                "num_refiner_layers": args.num_refiner_layers,
                "mixer_type": args.mixer_type,
                "pool_type": args.pool_type,
                "gate_type": args.gate_type,
                "zero_init_oproj": args.zero_init_oproj,
                "gate_floor": args.gate_floor,
                "gate_bias_init": args.gate_bias_init,
                "mlp_intermediate": args.mlp_intermediate,
                "lowrank_lmhead_rank": args.lowrank_lmhead_rank,
                "lowrank_lmhead_init": args.lowrank_lmhead_init,
                "lowrank_lmhead_shared_rank": args.lowrank_lmhead_shared_rank,
                "refiner_state_dict": refiner_state_dict,
                "draft_state_dict": draft_state_dict,
                "base_readout_state_dict": base_readout_state_dict,   # low-rank base readout (None if off)
                # Scheduler is REPLICATED across ranks -> store it in the rank-0 file.
                # The optimizer (Adam) state is FSDP-SHARDED (per-rank, different sizes)
                # because fp32_params clone the LOCAL FSDP shards -> saved per-rank below.
                "scheduler_state_dict": optimizer.scheduler.state_dict(),
            },
            os.path.join(save_dir, "refiner_cotrain.pt"),
        )
        print_on_rank0(f"Saved co-train checkpoint (refiner+drafter) to {save_dir}")

    # Every rank writes its OWN optimizer (Adam) shard. save_dir already exists for all
    # ranks (rank-0 mkdir + barrier above). On resume each rank reloads its matching shard.
    torch.save(
        {"optimizer_state_dict": optimizer.optimizer.state_dict()},
        os.path.join(save_dir, f"optim_rank{dist.get_rank()}.pt"),
    )
    dist.barrier()

    # Rotation: keep only the last N checkpoints so /gpfs doesn't fill up (each ckpt ~16GB =
    # drafter 1.05B + head + 8 optim shards). Latest (highest global_step) is always kept -> --resume ok.
    keep = getattr(args, "keep_last_n_checkpoints", 0)
    if dist.get_rank() == 0 and keep and keep > 0:
        import re
        import shutil
        ckpts = [d for d in os.listdir(args.output_dir)
                 if re.fullmatch(r"epoch_\d+_step_\d+", d)
                 and os.path.isdir(os.path.join(args.output_dir, d))]
        ckpts.sort(key=lambda d: int(d.rsplit("_step_", 1)[1]))   # global_step is monotonic across epochs
        for old in ckpts[:-keep]:
            shutil.rmtree(os.path.join(args.output_dir, old), ignore_errors=True)
            print_on_rank0(f"[ckpt-rotate] removed old checkpoint {old} (keep last {keep})")
    dist.barrier()


def record_metrics(args, loss, accuracy, global_step, tracker, optimizer, train_dataloader,
                   loss_terms=None, cons_per_pass=None):
    """loss_terms: {'tf','ce','l1','cons'} rank-averaged scalars. cons_per_pass: per-Jacobi-round
    consistency losses (len == --consistency-passes), also rank-averaged, or None when consistency
    is off."""
    logdict = {"train/lr": optimizer.get_learning_rate(), "train/loss": loss, "train/accuracy": accuracy}
    # Topology-independent x-axes for cross-run overlays (official-code runs log the same
    # keys): optim_step = parameter updates, samples = sequences consumed. Select either
    # as the x-axis in the wandb UI to compare arms with different world sizes/accum.
    logdict["progress/optim_step"] = global_step // max(1, args.accumulation_steps)
    logdict["progress/samples"] = global_step * args.batch_size * dist.get_world_size()
    extra = ""
    if loss_terms:
        logdict.update({f"train/{k}": v for k, v in loss_terms.items()})
        extra += f", tf: {loss_terms['tf']:.4f}, ce: {loss_terms['ce']:.4f}, l1: {loss_terms['l1']:.4f}"
        if args.consistency_weight > 0:
            extra += f", cons: {loss_terms['cons']:.4f}"
        if "conf" in loss_terms:
            extra += f", conf: {loss_terms['conf']:.4f}"
    if cons_per_pass:
        for j, v in enumerate(cons_per_pass):
            logdict[f"train/cons_pass{j}"] = v
        # Spread across rounds is the honest "is K>1 doing anything?" signal: ~0 means the Jacobi
        # rollout hit a fixed point at round 0, so every extra round is a duplicate of round 0.
        spread = max(cons_per_pass) - min(cons_per_pass)
        logdict["train/cons_spread"] = spread
        extra += "  cons/round=[" + " ".join(f"{v:.4f}" for v in cons_per_pass) + f"] spread={spread:.2e}"
        if len(cons_per_pass) > 1 and spread < 1e-6:
            extra += "  <- FLAT: rollout at fixed point, K>1 adding nothing"
    log_print(
        f"Train - Step {global_step} "
        f"[{global_step}/{args.num_epochs * len(train_dataloader) // args.accumulation_steps}?], "
        f"Loss: {loss:.4f}, Acc: {accuracy:.4f}{extra}"
    )
    tracker.log(logdict, step=global_step)


@torch.no_grad()
def run_eval(refiner_model, eval_dataloader, target_model, max_batches=50):
    """Mean free-running accept length: (refiner, drafter baseline) over a few eval batches."""
    r_tot, d_tot, cnt = 0.0, 0.0, 0
    for i, data in enumerate(eval_dataloader):
        if i >= max_batches:
            break
        input_ids = data["input_ids"].cuda()
        attention_mask = data["attention_mask"].cuda()
        loss_mask = data["loss_mask"].cuda()
        out = target_model.generate_dflash_data(input_ids, attention_mask, loss_mask)
        r, d = refiner_model.accept_lengths(input_ids, out.hidden_states.cuda(), loss_mask)
        r_tot += r.item()
        d_tot += d.item()
        cnt += 1
    vals = torch.tensor([r_tot, d_tot, float(cnt)], device="cuda")
    dist.all_reduce(vals)
    total = vals[2].item()
    if total == 0:
        return 0.0, 0.0
    return (vals[0] / total).item(), (vals[1] / total).item()


def main():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    warnings.filterwarnings(
        "ignore",
        "The .grad attribute of a Tensor that is not a leaf Tensor is being accessed",
    )

    args = parse_args()
    set_seed(args.seed)
    init_distributed(timeout=args.dist_timeout, tp_size=args.tp_size)
    print_with_rank("Initialized distributed")

    if dist.get_rank() == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        global _LOGF
        _LOGF = open(os.path.join(args.output_dir, "train.log"), "a", buffering=1)  # line-buffered

        # class _Tee:
        #     def __init__(self, *streams):
        #         self.streams = streams

        #     def write(self, msg):
        #         for s in self.streams:
        #             s.write(msg)
        #             s.flush()

        #     def flush(self):
        #         for s in self.streams:
        #             s.flush()

        #     def isatty(self):
        #         # reflect the REAL underlying stream: wandb shows its progress bar when this
        #         # runs in a terminal, and auto-suppresses it when output is piped to a file
        #         # (where a dynamic \r/ANSI bar would just be log garbage). Without this method
        #         # at all, wandb.init crashes with AttributeError on the _Tee.
        #         return self.streams[0].isatty()

        #     def __getattr__(self, name):
        #         # delegate anything else (fileno, encoding, buffer, ...) to the real stream
        #         # so libraries probing the stream don't hit AttributeError.
        #         return getattr(self.streams[0], name)

        # sys.stdout = _Tee(sys.stdout, _logf)
        # sys.stderr = _Tee(sys.stderr, _logf)
        # print(f"[tee] logging to {os.path.join(args.output_dir, 'train.log')}")

    # --- DFlash drafter (initialized from checkpoint, then CO-TRAINED) ---
    print_on_rank0(f"Loading DFlash drafter (to co-train) from {args.dflash_model_path}")
    if args.random_init_drafter:
        # Setting 2 (DSpark-style): build the drafter from config with RANDOM init (NO checkpoint).
        # Only the architecture (layers / target_layer_ids / block_size / mask_token) comes from the
        # --dflash-model-path config; weights are fresh -> drafter trained from scratch with the head.
        _dcfg = DFlashDraftModel.config_class.from_pretrained(args.dflash_model_path)
        draft_model = DFlashDraftModel(_dcfg).cuda().to(torch.bfloat16)
        print_on_rank0(f"[random-init-drafter] RANDOM weights; arch-only from {args.dflash_model_path}")
    else:
        draft_model = (
            DFlashDraftModel.from_pretrained(args.dflash_model_path, torch_dtype=torch.bfloat16)
            .cuda()
            .to(torch.bfloat16)
        )
    draft_model.config._attn_implementation = args.attention_backend
    block_size = draft_model.block_size
    # mask_token_id = (draft_model.config.dflash_config or {}).get("mask_token_id", None)
    mask_token_id = (getattr(draft_model.config, "dflash_config", None) or {}).get("mask_token_id", None) or getattr(draft_model.config, "mask_token_id", None)
    if mask_token_id is None:
        mask_token_id = draft_model.mask_token_id
    print_on_rank0(
        f"block_size={block_size} target_layer_ids={draft_model.target_layer_ids} "
        f"mask_token_id={mask_token_id}"
    )

    # --- target model (produces hidden_states online) ---
    print_on_rank0(f"Loading target model from {args.target_model_path}")
    target_model = get_dflash_target_model(
        pretrained_model_name_or_path=args.target_model_path,
        backend="hf",
        torch_dtype=torch.bfloat16,
        device="cuda",
        trust_remote_code=args.trust_remote_code,
    )
    target_model.set_capture_layers(draft_model.target_layer_ids)

    tokenizer = AutoTokenizer.from_pretrained(args.target_model_path)

    # --- data ---
    train_dataloader, eval_dataloader = build_dataloader(args, tokenizer, block_size)
    if int(os.environ.get("RANK", "0")) == 0:
        _n_eval = len(eval_dataloader) if eval_dataloader is not None else 0
        print(f"[setup] eval_data_path={args.eval_data_path!r}  eval_dataloader="
              f"{'NONE -> EVAL WILL BE SKIPPED' if eval_dataloader is None else f'{_n_eval} batches'}",
              flush=True)
    steps_per_epoch = math.ceil(len(train_dataloader) / args.accumulation_steps)
    total_steps = args.num_epochs * steps_per_epoch
    print_on_rank0(f"Total training steps: {total_steps}")

    # --- target embed / lm_head (reused, frozen) ---
    print_on_rank0("Loading target embeddings and head...")
    target_components = TargetEmbeddingsAndHead.from_pretrained(
        args.target_model_path,
        embed_key=args.embedding_key,
        lm_head_key=args.lm_head_key,
        device="cuda",
        trust_remote_code=args.trust_remote_code,
    )

    # --- co-train feature extractor (grad flows to drafter) + refiner ---
    feature_extractor = CoTrainFeatureExtractor(
        draft_model=draft_model,
        target_lm_head=target_components.lm_head.cuda().to(torch.bfloat16),
        target_embed_tokens=target_components.embed_tokens.cuda().to(torch.bfloat16),
        mask_token_id=mask_token_id,
        block_size=block_size,
        attention_backend=args.attention_backend,
        num_anchors=args.num_anchors,
        loss_decay_gamma=None,
    ).cuda()

    refiner_model = OnlineHybridRefinerCoTrain(
        feature_extractor,
        markov_rank=args.markov_rank,
        sgu_enabled=not args.markov_only,
        use_hidden=not args.no_hidden,
        use_perpos=not args.no_perpos,
        use_global=not args.no_global,
        use_residual=not args.no_residual,
        use_mixer=not args.no_mixer,
        use_mlp=not args.no_mlp,
        mlp_ratio=args.mlp_ratio,
        input_mode=args.input_mode,
        mixer_init=args.mixer_init,
        use_norm=not args.no_sublayer_norm,
        use_mix_out=not args.no_mix_out,
        shared_readout_rank=args.lowrank_lmhead_shared_rank,
        base_readout_rank=args.lowrank_base_rank,
        input_scale=args.input_scale,
        input_relu=args.input_relu,
        l1_alpha=args.l1_alpha,
        ce_alpha=args.ce_alpha,
        loss_decay_gamma=args.loss_decay_gamma,
        consistency_weight=args.consistency_weight,
        consistency_passes=args.consistency_passes,
        grad_checkpoint_loss=args.loss_grad_checkpoint,
        cotrain_drafter=not args.no_cotrain_drafter,
        block_convention=args.block_convention,
        anchor_full_weight=args.anchor_full_weight,
        confidence_alpha=args.confidence_alpha,
        confidence_mode=args.confidence_mode,
        confidence_detach=args.confidence_detach,
    ).cuda()
    refiner_model.refiner = refiner_model.refiner.to(torch.bfloat16)
    if int(os.environ.get("RANK", "0")) == 0:
        head = "Markov (DSpark special case)" if args.markov_only else "r-bottleneck SGU"
        print(
            "\n==================== HYBRID HEAD ARCH ====================\n"
            f"  head         = {head}\n"
            f"  markov_rank  = {args.markov_rank}   input_mode = {args.input_mode}   mixer_init = {args.mixer_init}\n"
            f"  use_perpos={not args.no_hidden and not args.no_perpos}  use_global={not args.no_hidden and not args.no_global}  "
            f"use_mixer={not args.no_mixer}  use_mlp={not args.no_mlp}  "
            f"use_residual={not args.no_residual}  mlp_ratio={args.mlp_ratio}\n"
            f"  loss: ce_alpha={args.ce_alpha} l1_alpha={args.l1_alpha} decay_gamma={args.loss_decay_gamma} "
            f"consistency_weight={args.consistency_weight} x{args.consistency_passes} Jacobi rounds (mean)"
            f"  loss_ckpt={args.loss_grad_checkpoint}   (DSpark: ce0.1/l1_0.9/gamma4)\n"
            f"  lambda_base  = {args.lambda_base_start}->{args.lambda_base_floor} (ratio {args.lambda_base_decay_ratio})   "
            f"drafter_lr_scale = {args.drafter_lr_scale}  cotrain_drafter={not args.no_cotrain_drafter}\n"
            f"  head params  = {sum(p.numel() for p in refiner_model.refiner.parameters()):,} | "
            f"co-trained drafter params = {refiner_model._cotrain_drafter_params:,}\n"
            "=========================================================\n",
            flush=True,
        )

    # rank >= hidden -> the wrapper normalized base_readout_rank to 0 (full FROZEN lm_head base, no training).
    _H_lr = refiner_model.feature_extractor.lm_head.weight.shape[1]
    if args.lowrank_base_rank > 0 and refiner_model.base_readout_rank == 0:
        log_print(f"[lowrank-base] --lowrank-base-rank {args.lowrank_base_rank} >= hidden {_H_lr} -> FULL frozen "
                  f"lm_head base (no low-rank readout, no training; = the full-rank control).")

    # --- LOW-RANK BASE READOUT (data-PCA / calibration init) -- BEFORE FSDP so the drafter is unsharded ---
    # Trigger on use_lowrank_base so a rank>=H fallback (-> full lm_head) correctly SKIPS calibration/fit.
    if refiner_model.use_lowrank_base:
        assert args.drafter_lr_scale == 1.0, \
            "--lowrank-lmhead-shared-rank / --lowrank-base-rank support single-LR only (set --drafter-lr-scale 1.0)."
        V_lr, H_lr = refiner_model.feature_extractor.lm_head.weight.shape   # (V, H)
        cov_path = args.lowrank_lmhead_shared_cov
        if cov_path and os.path.exists(cov_path):
            _c = torch.load(cov_path, map_location="cpu", weights_only=False)
            C = (_c["C"] if isinstance(_c, dict) else _c).float().cuda()
            log_print(f"[lowrank-base] loaded calibration covariance from {cov_path}")
        else:
            log_print(f"[lowrank-base] collecting activation covariance over "
                      f"{args.lowrank_calib_batches} batches (h=drafter hidden) ...")
            C = torch.zeros(H_lr, H_lr, dtype=torch.float64, device="cuda")
            n_vec = 0
            refiner_model.eval()
            with torch.no_grad():
                for bi, data in enumerate(train_dataloader):
                    if bi >= args.lowrank_calib_batches:
                        break
                    iid = data["input_ids"].cuda()
                    am = data["attention_mask"].cuda()
                    lm = data["loss_mask"].cuda()
                    tout = target_model.generate_dflash_data(iid, am, lm)
                    f = refiner_model.feature_extractor.compute_block_features(
                        iid, tout.hidden_states.cuda(), lm)
                    hh = f.output_hidden.reshape(-1, H_lr).double()          # (N, H) readout inputs
                    C += hh.t() @ hh
                    n_vec += hh.shape[0]
            refiner_model.train()
            if dist.get_world_size() > 1:
                dist.all_reduce(C)                                          # global covariance across DP ranks
            C = C.float()
            log_print(f"[lowrank-base] covariance collected (~{n_vec} vectors/rank)")
            if cov_path and dist.get_rank() == 0:
                os.makedirs(os.path.dirname(cov_path) or ".", exist_ok=True)
                torch.save({"C": C.cpu(), "hidden_size": H_lr}, cov_path)
                log_print(f"[lowrank-base] saved covariance -> {cov_path}")
        # eigh sign/degeneracy AND --lowrank-calib-rotate's randn(r,r) can differ per GPU -> force rank-0's
        # fit onto all ranks so FSDP shards a consistent tensor. (set_seed makes the rest identical already.)
        if args.lowrank_lmhead_shared_rank > 0:
            refiner_model.init_shared_readout(C, rotate=args.lowrank_calib_rotate)   # fits head.w2 (up) + ro_down
            fit_params = (refiner_model.refiner.w2.weight, refiner_model.refiner.ro_down.weight)
            ready_msg = (f"[lowrank-base] SHARED low-rank readout ready: rank={args.lowrank_lmhead_shared_rank} "
                         f"ro_down{tuple(refiner_model.refiner.ro_down.weight.shape)} "
                         f"w2(up){tuple(refiner_model.refiner.w2.weight.shape)}")
        else:
            refiner_model.init_base_readout(C, rotate=args.lowrank_calib_rotate)     # fits head.base_down + base_up
            fit_params = (refiner_model.refiner.base_up.weight, refiner_model.refiner.base_down.weight)
            ready_msg = (f"[lowrank-base] SEPARATE low-rank base readout ready: rank={args.lowrank_base_rank} "
                         f"base_down{tuple(refiner_model.refiner.base_down.weight.shape)} "
                         f"base_up{tuple(refiner_model.refiner.base_up.weight.shape)} (refiner rank={args.markov_rank})")
        if dist.get_world_size() > 1:
            for p in fit_params:
                dist.broadcast(p.data, src=0)
        del C
        log_print(ready_msg)

    # FSDP-wrap BOTH trainable modules: the refiner head AND the drafter backbone.
    refiner_model.refiner = FSDP(
        refiner_model.refiner,
        use_orig_params=True,
        mixed_precision=MixedPrecision(param_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16),
        sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
    )
    refiner_model.feature_extractor.draft_model = FSDP(
        refiner_model.feature_extractor.draft_model,
        use_orig_params=True,
        mixed_precision=MixedPrecision(param_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16),
        sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
    )
    # NOTE: the SHARED low-rank readout (w2 + ro_down) lives INSIDE the head (refiner) -> it's already
    # FSDP-wrapped above, already in the optimizer (refiner), and saved in refiner_state_dict. No extra unit.
    print_with_rank("Initialized FSDP (refiner head + drafter)"
                    + (f"  [shared low-rank readout r={args.lowrank_lmhead_shared_rank}]"
                       if args.lowrank_lmhead_shared_rank > 0 else ""))

    # optimizer over both FSDP units. drafter-lr-scale != 1.0 -> 2 param groups
    # (head at lr, drafter at lr*scale); == 1.0 -> single shared LR.
    if args.drafter_lr_scale != 1.0:
        optimizer = CoTrainBF16Optimizer(
            refiner_model.refiner,
            refiner_model.feature_extractor.draft_model,
            lr=args.learning_rate,
            drafter_lr_scale=args.drafter_lr_scale,
            max_grad_norm=args.max_grad_norm,
            warmup_ratio=args.warmup_ratio,
            total_steps=total_steps,
        )
        print_on_rank0(f"Optimizer: 2 groups (head lr={args.learning_rate}, "
                       f"drafter lr={args.learning_rate * args.drafter_lr_scale})")
    else:
        # refiner already contains the shared low-rank readout (w2 + ro_down) -> trained here, no extra module.
        trainable = nn.ModuleList(
            [refiner_model.refiner, refiner_model.feature_extractor.draft_model]
        )
        optimizer = BF16Optimizer(
            trainable,
            lr=args.learning_rate,
            max_grad_norm=args.max_grad_norm,
            warmup_ratio=args.warmup_ratio,
            total_steps=total_steps,
        )

    # --- resume (refiner + drafter + optimizer) ---
    start_epoch, global_step = 0, 0
    if args.resume and os.path.isdir(args.output_dir):
        last_ckpt, ckpt_info = get_last_checkpoint(args.output_dir)
        ckpt_file = os.path.join(last_ckpt, "refiner_cotrain.pt") if last_ckpt else None
        if ckpt_file and os.path.exists(ckpt_file):
            state = torch.load(ckpt_file, map_location="cpu", weights_only=False)
            with FSDP.state_dict_type(refiner_model.refiner, StateDictType.FULL_STATE_DICT):
                # strict=False so a structure change (e.g. adding the shared low-rank lm_head to an old
                # ckpt) keeps the freshly DATA-PCA-init'd shared_lmhead + ignores a stale lowrank_head,
                # instead of crashing. A matching ckpt loads identically (no missing/unexpected).
                _ld = refiner_model.refiner.load_state_dict(state["refiner_state_dict"], strict=False)
            if int(os.environ.get("RANK", "0")) == 0 and (_ld.missing_keys or _ld.unexpected_keys):
                print(f"[resume] refiner load: missing={_ld.missing_keys[:6]} "
                      f"unexpected={_ld.unexpected_keys[:6]}", flush=True)
            if "draft_state_dict" in state:
                with FSDP.state_dict_type(
                    refiner_model.feature_extractor.draft_model, StateDictType.FULL_STATE_DICT
                ):
                    refiner_model.feature_extractor.draft_model.load_state_dict(
                        state["draft_state_dict"]
                    )
            # (shared low-rank readout is inside refiner -> already restored via refiner_state_dict above.)
            # CRITICAL: the BF16 optimizer holds fp32 MASTER copies cloned at construction
            # (pre-resume init weights). The load_state_dict calls above updated the model's
            # p.data but NOT these masters. On the first step(), p.data.copy_(master) would
            # overwrite the freshly-loaded checkpoint weights with the stale init masters and
            # silently revert the model (train loss looks fine for one step, then eval craters).
            # Re-sync the masters from the just-loaded model params.
            with torch.no_grad():
                for p, mp in zip(optimizer.model_params, optimizer.fp32_params):
                    mp.data.copy_(p.data.to(torch.float32))
            # Scheduler is replicated -> always restore (continues LR schedule exactly).
            if "scheduler_state_dict" in state:
                optimizer.scheduler.load_state_dict(state["scheduler_state_dict"])
            # Optimizer (Adam) state is FSDP-SHARDED -> each rank loads ITS OWN shard.
            # Loading rank-0's state into every rank crashes (size mismatch). Old
            # checkpoints without per-rank shards just reset Adam moments (small dip).
            optim_file = os.path.join(last_ckpt, f"optim_rank{dist.get_rank()}.pt")
            if os.path.exists(optim_file):
                opt_state = torch.load(optim_file, map_location="cpu", weights_only=False)
                optimizer.optimizer.load_state_dict(opt_state["optimizer_state_dict"])
                print_on_rank0("Restored per-rank optimizer (Adam) shards + scheduler.")
            else:
                print_on_rank0("[warn] no per-rank optim shards (old checkpoint); "
                               "Adam moments reset (small dip), scheduler restored.")
            start_epoch = state["epoch"]
            global_step = state["global_step"]
            print_on_rank0(f"Resumed from epoch {start_epoch}, step {global_step}")
    skip_steps = global_step - start_epoch * len(train_dataloader)

    tracker = create_tracker(args, args.output_dir)
    last_time = time.time()

    # --- ACTUAL-STATE verification (assert reality, not echo args); fail-fast on a broken setup ---
    _errs = []
    _head_tr = sum(p.numel() for p in refiner_model.refiner.parameters() if p.requires_grad)
    _drf_tr = sum(p.numel() for p in refiner_model.feature_extractor.draft_model.parameters() if p.requires_grad)
    _opt_tr = sum(p.numel() for p in getattr(optimizer, "fp32_params", []))
    _n_eval_b = len(eval_dataloader) if eval_dataloader is not None else 0
    _w1, _w2 = refiner_model.refiner.w1.weight, refiner_model.refiner.w2.weight
    _V = refiner_model.feature_extractor.lm_head.weight.shape[0]
    if _n_eval_b == 0:
        _errs.append("eval_dataloader EMPTY -> eval will be SKIPPED (check --eval-data-path).")
    if _head_tr == 0:
        _errs.append("head has ZERO trainable params -> head won't learn.")
    if args.no_cotrain_drafter and _drf_tr != 0:
        _errs.append(f"--no-cotrain-drafter set but drafter has {_drf_tr:,} trainable params (freeze failed).")
    if (not args.no_cotrain_drafter) and _drf_tr == 0:
        _errs.append("cotrain_drafter=True but drafter has ZERO trainable params (should be co-trained).")
    if args.no_cotrain_drafter and args.lambda_base_start > 0:
        _errs.append(f"--no-cotrain-drafter + lambda_base_start={args.lambda_base_start}>0: base_loss on a FROZEN "
                     "drafter has NO gradient; with lambda_base>0 the head is starved. Set --lambda-base-start 0.")
    if _opt_tr != _head_tr + _drf_tr:
        _errs.append(f"optimizer holds {_opt_tr:,} params but trainable = {_head_tr + _drf_tr:,} (optimizer mis-wired).")
    # NOTE: W1/W2 shapes are only LOGGED, not asserted -- under FSDP sharding they may be the local
    # shard, not the full (V, r), so a hard equality check would false-crash on multi-GPU.
    log_print(
        "[verify-actual-state]\n"
        f"  head trainable   = {_head_tr:,}   (requires_grad params in refiner)\n"
        f"  drafter trainable= {_drf_tr:,}   (cotrain={not args.no_cotrain_drafter})\n"
        f"  optimizer params = {_opt_tr:,}   (must == head+drafter trainable = {_head_tr + _drf_tr:,})\n"
        f"  optim current lr = {optimizer.optimizer.param_groups[0]['lr']:.2e}  (warms up from ~0)\n"
        f"  head W1={tuple(_w1.shape)} W2={tuple(_w2.shape)}  (expect V={_V}, r={args.markov_rank})\n"
        f"  eval batches     = {_n_eval_b}\n"
        f"  first-batch loss = (see first Train-Step below; must be finite)"
    )
    if _errs:
        raise RuntimeError("SETUP VERIFICATION FAILED:\n  - " + "\n  - ".join(_errs))
    log_print("[verify-actual-state] ALL CHECKS PASSED\n")

    _ws = dist.get_world_size()
    _gbs = args.batch_size * _ws * args.accumulation_steps           # global batch in SEQUENCES
    _n_eval = len(eval_dataloader) if eval_dataloader is not None else 0
    log_print(
        "\n==================== TRAINING SETUP ====================\n"
        f"  world_size(GPUs) = {_ws}   local_batch = {args.batch_size}   accum_steps = {args.accumulation_steps}\n"
        f"  global_batch     = {_gbs} sequences  (= {_gbs} x num_anchors {args.num_anchors} = {_gbs*args.num_anchors} BLOCKS/step)\n"
        f"  lr = {args.learning_rate}   warmup_ratio = {args.warmup_ratio}   "
        f"weight_decay = {optimizer.optimizer.param_groups[0]['weight_decay']}   "
        f"max_grad_norm = {args.max_grad_norm}\n"
        f"  block_size = {block_size}   num_anchors = {args.num_anchors}   max_length = {args.max_length}\n"
        f"  num_epochs = {args.num_epochs}   steps/epoch = {len(train_dataloader)}   total_optim_steps = {total_steps}\n"
        f"  train_batches = {len(train_dataloader)}   eval_batches = {_n_eval}"
        f"{'  <-- WARNING: eval_dataloader is EMPTY/NONE, eval will be skipped!' if _n_eval == 0 else ''}\n"
        f"  loss: ce={args.ce_alpha} l1={args.l1_alpha} decay_gamma={args.loss_decay_gamma}  "
        f"lambda_base={args.lambda_base_start}->{args.lambda_base_floor}  consistency={args.consistency_weight}\n"
        f"  cotrain_drafter = {not args.no_cotrain_drafter}   drafter_lr_scale = {args.drafter_lr_scale}\n"
        f"  target = {args.target_model_path}\n"
        f"  drafter = {args.dflash_model_path}   target_layer_ids = {draft_model.target_layer_ids}\n"
        f"  train_data = {args.train_data_path}\n"
        f"  eval_data  = {args.eval_data_path}\n"
        f"  output_dir = {args.output_dir}\n"
        f"  tracker(report_to) = {args.report_to}   log_file = {os.path.join(args.output_dir, 'train.log')}\n"
        "=======================================================\n"
    )
    print_on_rank0(f"Starting co-train from epoch {start_epoch}, step {global_step}")

    for epoch in range(start_epoch, args.num_epochs):
        train_dataloader.sampler.set_epoch(epoch)
        refiner_model.refiner.train()
        # everything frozen-eval first, then put ONLY the drafter back in train mode
        feature_extractor.eval()
        refiner_model.feature_extractor.draft_model.train()

        progress_bar = (
            tqdm(train_dataloader, desc=f"Co-train Epoch {epoch}", leave=True)
            if dist.get_rank() == 0
            else train_dataloader
        )

        for step_in_epoch, data in enumerate(progress_bar):
            if epoch == start_epoch and step_in_epoch < skip_steps:
                continue
            global_step += 1

            input_ids = data["input_ids"].cuda()
            attention_mask = data["attention_mask"].cuda()
            loss_mask = data["loss_mask"].cuda()

            target_output = target_model.generate_dflash_data(input_ids, attention_mask, loss_mask)
            hidden_states = target_output.hidden_states.cuda()
            final_hidden = target_output.final_hidden.cuda()  # post-norm target hidden for L1/TV target

            # base_loss curriculum: lambda_base decays start -> 0 over decay_ratio*total_steps
            decay_steps = max(1, int(total_steps * args.lambda_base_decay_ratio))
            # lambda_base = max(0.0, args.lambda_base_start * (1.0 - min(global_step / decay_steps, 1.0)))
            _p = min(global_step / decay_steps, 1.0)   # 0 -> 1 over decay
            lambda_base = args.lambda_base_start * (1.0 - _p) + args.lambda_base_floor * _p

            loss, accuracy = refiner_model(
                input_ids=input_ids, hidden_states=hidden_states, loss_mask=loss_mask,
                final_hidden=final_hidden, lambda_base=lambda_base,
            )
            (loss / args.accumulation_steps).backward()
            if global_step % args.accumulation_steps == 0:
                optimizer.step()

            if global_step % args.log_interval == 0:
                ws = dist.get_world_size()
                loss_log, acc_log = loss.clone(), accuracy.clone()
                dist.all_reduce(loss_log)
                # FAIL FAST on non-finite loss. The verify block promises "first-batch loss must
                # be finite" but nothing ever enforced it: the markov-b7 run trained 900 steps of
                # pure NaN before a human noticed. loss_log is the SAME all-reduced value on every
                # rank, so either all ranks raise together (clean torchrun teardown, no straggler
                # hang) or none do. NaN poisons fp32 optimizer state -- aborting is always right.
                if not torch.isfinite(loss_log):
                    raise RuntimeError(
                        f"NON-FINITE all-reduced loss ({loss_log.item()}) at step {global_step}. "
                        "Weights/optimizer state are already poisoned -- do NOT --resume from a "
                        "checkpoint saved after this point. If the model/config is unchanged from "
                        "a run that trained cleanly, suspect a faulty GPU on one of these hosts "
                        "and resubmit (or exclude them with bsub -R)."
                    )
                dist.all_reduce(acc_log)
                loss_log /= ws
                acc_log /= ws
                # Loss breakdown + per-Jacobi-round consistency, rank-averaged the same way.
                # Every rank runs the SAME number of all_reduce calls here: last_loss_terms always has
                # the same 4 keys and last_cons_per_pass is None/len-K identically on all ranks (both
                # derive from args), so this cannot desync the collective.
                terms_log = {}
                # "conf" exists iff --confidence-alpha > 0 (same args on every rank -> the
                # all_reduce count below stays identical across ranks; no desync possible).
                _term_keys = ("tf", "ce", "l1", "cons") + (("conf",) if args.confidence_alpha > 0 else ())
                for k in _term_keys:
                    t = refiner_model.last_loss_terms[k].detach().clone().float()
                    dist.all_reduce(t)
                    terms_log[k] = (t / ws).item()
                cpp_log = None
                if refiner_model.last_cons_per_pass is not None:
                    t = refiner_model.last_cons_per_pass.detach().clone().float()
                    dist.all_reduce(t)
                    cpp_log = (t / ws).tolist()
                record_metrics(args, loss_log.item(), acc_log.item(), global_step, tracker, optimizer,
                               train_dataloader, loss_terms=terms_log, cons_per_pass=cpp_log)
                tracker.log({"train/lambda_base": lambda_base}, step=global_step)

            if eval_dataloader is not None and global_step % args.eval_interval == 0:
                log_print(f"\n[eval] running at step {global_step} ...")
                refiner_model.feature_extractor.draft_model.eval()
                r_acc, d_acc = run_eval(refiner_model, eval_dataloader, target_model)
                refiner_model.feature_extractor.draft_model.train()
                tracker.log(
                    {"eval/refiner_accept_len": r_acc, "eval/drafter_accept_len": d_acc},
                    step=global_step,
                )
                log_print(f"===== Eval - Step {global_step}: refiner_accept={r_acc:.3f} "
                          f"drafter_accept={d_acc:.3f} =====\n")

            if dist.get_rank() == 0:
                elapsed = time.time() - last_time
                last_time = time.time()
                progress_bar.set_postfix(
                    {"loss": f"{loss.item():.4f}", "acc": f"{accuracy.item():.4f}", "iter_time": f"{elapsed:.2f}s"}
                )

            if global_step % args.save_interval == 0:
                save_checkpoint(
                    args, epoch, global_step,
                    refiner_model.refiner, refiner_model.feature_extractor.draft_model,
                    optimizer,
                )

    save_checkpoint(
        args, args.num_epochs, global_step,
        refiner_model.refiner, refiner_model.feature_extractor.draft_model,
        optimizer,
    )
    tracker.close()
    destroy_distributed()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import time, traceback  # hold the crashed process so the pod isn't deleted before we can inspect
        traceback.print_exc()  # SHOW the real error instead of swallowing it
        print("*******SLEEPING (crashed — traceback above)********", flush=True)
        time.sleep(10 ** 9)
