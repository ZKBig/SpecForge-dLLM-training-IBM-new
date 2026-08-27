#!/usr/bin/env python3
"""Probe the SGLang DFlash target backend on a Qwen3.5 model.

Answers, with printed shapes instead of code reading, the questions that block porting
train_hybrid_refiner_IBM.py to Qwen3.5:

  Q1  Does the capture hook exist and fire?  (set_dflash_layers_to_capture, added in sglang 0.5.12)
  Q2  Is `hidden_states` the CONCATENATION of the captured layers, at FULL sequence length?
      -> expected (batch, seq, len(TARGET_LAYERS) * target_hidden)
  Q3  Is `final_hidden` populated?  The hybrid refiner's L1/TV loss asserts on it, and the
      SGLang backend currently never sets it -- this probe confirms whether that is really the
      case at runtime, and what the alternative sources look like.
  Q4  Where does the target's final norm live, and can we reach it to build a post-norm
      full-sequence hidden ourselves?

Run on a GPU node:
    cd /proj/checkpoints/zwang619/SpecForge-dLLM-training-IBM
    E=/proj/checkpoints/zwang619/miniconda3/envs/dLLM_35b/bin
    $E/torchrun --standalone --nproc_per_node 1 probe_sglang_qwen35.py

Read-only: loads the target model and runs one forward. Trains nothing, writes nothing.
"""
import os
import sys

import torch
import torch.distributed as dist

MODEL = os.environ.get("PROBE_MODEL", "/proj/checkpoints/ashishagr/model_downloads/qwen3.5-9b")
# 9B has 32 layers; spread 5 capture points like the 35B config does over its 40.
TARGET_LAYERS = [int(x) for x in os.environ.get("PROBE_LAYERS", "1,8,15,22,29").split(",")]
SEQ = int(os.environ.get("PROBE_SEQ", "256"))
BATCH = int(os.environ.get("PROBE_BATCH", "2"))


def banner(s):
    print(f"\n{'=' * 70}\n{s}\n{'=' * 70}", flush=True)


from specforge.distributed import init_distributed
from specforge.modeling.target.dflash_target_model import get_dflash_target_model

init_distributed(timeout=30, tp_size=1)
rank = dist.get_rank()
if rank == 0:
    banner(f"model={MODEL}\ncapture layers={TARGET_LAYERS}  batch={BATCH} seq={SEQ}")

target = get_dflash_target_model(
    pretrained_model_name_or_path=MODEL,
    backend="sglang",
    torch_dtype=torch.bfloat16,
    device=None,
    trust_remote_code=True,
)

# ---- Q1: does the capture hook exist and fire?
banner("Q1  capture hook")
inner = target.model_runner.model
print(f"  model class          : {type(inner).__name__}")
for h in ("set_dflash_layers_to_capture", "set_eagle3_layers_to_capture"):
    print(f"  has {h:32}: {hasattr(inner, h)}")
target.set_capture_layers(TARGET_LAYERS)          # raises if neither hook exists

# ---- Q4: locate the final norm (needed if we must build post-norm hidden ourselves)
banner("Q4  where is the final norm (LANGUAGE tower only)")
print("  top-level children:", [n for n, _ in inner.named_children()])
for n, m in inner.named_children():
    print(f"    {n}: {[c for c, _ in m.named_children()][:8]}")
H_txt = target.model_runner.model_config.hidden_size
print(f"\n  looking for norms with weight dim == hidden_size ({H_txt}), excluding per-layer "
      f"and vision-tower norms:")
hits = []
for path, mod in inner.named_modules():
    if not type(mod).__name__.endswith(("RMSNorm", "LayerNorm")):
        continue
    if "visual" in path or ".layers." in path or "vision" in path:
        continue                                   # vision tower / the 32 per-layer norms
    w = getattr(mod, "weight", None)
    if w is not None and w.numel() == H_txt:
        hits.append((path, type(mod).__name__, tuple(w.shape)))
for p, cls, shape in hits:
    print(f"    {p:56} {cls:16} weight={shape}")
if not hits:
    print("    (none found -- widen the filter)")

# ---- Q2/Q3: run one forward and inspect what comes back
banner("Q2/Q3  one forward, what does the backend return")
torch.manual_seed(0)
vocab = target.model_runner.model_config.vocab_size
input_ids = torch.randint(100, min(vocab, 100000), (BATCH, SEQ), dtype=torch.long)
attention_mask = torch.ones(BATCH, SEQ, dtype=torch.long)
loss_mask = torch.ones(BATCH, SEQ, dtype=torch.long)

out = target.generate_dflash_data(input_ids, attention_mask, loss_mask)

def describe(name, t):
    if t is None:
        print(f"  {name:16} = None")
    elif torch.is_tensor(t):
        print(f"  {name:16} = shape {tuple(t.shape)}  dtype {t.dtype}")
    else:
        print(f"  {name:16} = {type(t)}")

for f in ("hidden_states", "final_hidden", "input_ids", "attention_mask", "loss_mask"):
    describe(f, getattr(out, f, None))

H = target.model_runner.model_config.hidden_size
print(f"\n  target hidden_size   = {H}")
print(f"  len(TARGET_LAYERS)   = {len(TARGET_LAYERS)}")
print(f"  expected concat width= {len(TARGET_LAYERS) * H}")
if out.hidden_states is not None:
    w = out.hidden_states.shape[-1]
    print(f"  actual width         = {w}  -> "
          f"{'CONCATENATED (capture works)' if w == len(TARGET_LAYERS) * H else ('SINGLE LAYER (capture NOT working)' if w == H else 'UNEXPECTED')}")
    print(f"  seq dim              = {out.hidden_states.shape[1]} (expected {SEQ} for full-sequence)")

banner("VERDICT")
ok_cap = out.hidden_states is not None and out.hidden_states.shape[-1] == len(TARGET_LAYERS) * H
ok_seq = out.hidden_states is not None and out.hidden_states.shape[1] == SEQ
print(f"  drafter input (multi-layer, full seq) : {'OK' if ok_cap and ok_seq else 'BROKEN'}")
print(f"  refiner final_hidden available        : {'OK' if out.final_hidden is not None else 'MISSING -> must be added'}")
if out.final_hidden is None:
    print("\n  -> The hybrid refiner cannot run on this backend as-is; it asserts on final_hidden.")
    print("     Next step is to source a full-sequence POST-NORM hidden. Two candidates:")
    print("       (a) add the last layer to the capture list and apply the final norm above manually")
    print("       (b) extend the sglang backend to also return the post-norm hidden")

dist.destroy_process_group()
