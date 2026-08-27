#!/usr/bin/env bash
# LSF wrapper for Qwen3.5-35B-A3B sgu_dspark ONLINE training on ONE node (8 H100).
#
#   cd /proj/checkpoints/zwang619/SpecForge-dLLM-training-IBM
#   bsub -q normal -G grp_ai_compiler_design -M 2000G -hl -n 1 \
#     -J xpress/train-35b-sgu-dspark-1node -gpu "num=8/task:mode=exclusive_process" \
#     -R "select[hname != 'p1-r04-n2']" \
#     -oo /proj/checkpoints/zwang619/results/xpress/train-35b-sgu-dspark-1node/output.log \
#     -eo /proj/checkpoints/zwang619/results/xpress/train-35b-sgu-dspark-1node/err.log \
#     bash train-35b-sgu-dspark-1node.sh
#
# One task, no blaunch: managed_local keeps every process on this node.
#
# This is a VALIDATION run, not the production one. It answers the questions the 2-node attempts
# never reached: real per-card memory, real producer->consumer throughput, warm-start
# correctness, and checkpoint/resume mechanics. Its checkpoints are NOT interchangeable with the
# 2-node run's -- dp_size is 6 here vs 8 there, and in online mode the LR horizon is DERIVED
# from dp_size, so the schedules differ and the consumer validates that contract on resume.
# Hence a separate run_id and output_dir.
#
# Unlike the 2-node path this does NOT go through examples/disagg/run_qwen3_8b_dflash_disagg_2node.sh
# (which hard-checks `NUM_NODES must be 2`). deployment.disaggregated.managed_local makes
# specforge itself supervise the Mooncake services, the patched SGLang capture server (TP2 on
# cuda 0-1), the producer and the 6-rank consumer in one process tree. So there is no RUN_ROOT,
# no Mooncake store_id, and no NODE_RANK -- all three of which caused failures on the 2-node path.
#
# GPU split: capture server TP2 on cuda 0,1 (the target has num_key_value_heads=2, so TP>2 would
# replicate KV heads); trainer on cuda 2-7.
set -Eeuo pipefail

SPECFORGE=/proj/checkpoints/zwang619/SpecForge
ENV_BIN=/proj/checkpoints/zwang619/miniconda3/envs/dLLM_35b/bin
export PATH="${ENV_BIN}:${PATH}"                       # mooncake_master + specforge live here
export HF_HOME=/proj/checkpoints/zwang619/.cache/huggingface

# ---- Triton JIT and sglang /tmp writers (both cost a 2-node attempt each) ----
# Job 746443 died because Triton JIT'd a launcher stub through /usr/lib64/ccache/gcc into /tmp
# and gcc could not write there; 746742 died in numa_utils.py:53 with ENOSPC on a hardcoded
# /tmp path. Same root cause, one full node-local /tmp. Triton honours TMPDIR (build.py:89) and
# $CC (build.py:26); the numactl wrapper does not, but the whole block is gated on
# SGLANG_NUMA_BIND_V2 (environ.py:738, EnvBool default True). Keep TMPDIR SHORT: AF_UNIX socket
# names cap at 108 bytes and torch/sglang create sockets under it.
export TRITON_CACHE_DIR="${HOME}/.triton/cache"
export TMPDIR="/dev/shm/${USER}/tmp"
export CC=/usr/bin/gcc
export SGLANG_NUMA_BIND_V2=0

CONFIG="${SPECFORGE}/examples/configs/qwen3.5-35b-a3b-sgu-dspark-1node.yaml"
RUN_ID="qwen35-35b-sgu-dspark-1node"
STABLE_ROOT="/proj/checkpoints/zwang619/results/xpress/train-35b-sgu-dspark-1node"
STABLE_OUT="${STABLE_ROOT}/output"                # must match output_dir in the YAML
STABLE_STATE="${STABLE_ROOT}/consumer-state"      # holds consumer.sqlite, the durable ledger
LEDGER="${STABLE_STATE}/consumer.sqlite"
# Per-attempt, unlike the two above: control_dir carries this attempt's lifecycle markers and
# ref channel, which must never be inherited from a dead attempt.
CONTROL_DIR="${STABLE_ROOT}/control-${LSB_JOBID:-manual}"
# z-lab's DFlash backbone + this head's own init, merged offline by
# SpecForge/scripts/build_sgu_warm_start.py. The raw z-lab checkpoint cannot be used: the
# warm-start loader excuses only absent EMBEDDING keys and ignores the provider's
# allowed_missing_checkpoint_keys, so the 10 from-scratch SGU head keys are rejected.
WARM_START=/proj/checkpoints/zwang619/warm_start/qwen3.5-35b-a3b-sgu-dspark-init

# NOT CONTROL_DIR: managed_local's supervisor creates it itself and refuses one that already
# exists ("managed_local requires a fresh control_dir") -- the same create-or-fail lock pattern
# that cost the 2-node path its first attempt with consumer-state.
mkdir -p "${STABLE_OUT}" "${STABLE_STATE}" "${TMPDIR}" "${TRITON_CACHE_DIR}"

# ---- /tmp preflight: fail in seconds naming the host, not 2 minutes in with an opaque error ----
_probe="/tmp/.specforge_tmp_probe.$$"
if ! dd if=/dev/zero of="${_probe}" bs=1M count=8 status=none 2>/dev/null; then
    rm -f "${_probe}" 2>/dev/null || true
    echo "FATAL: /tmp is unusable on $(hostname | cut -d. -f1) -- cannot write 8 MiB."
    echo "FATAL: sglang hardcodes /tmp in its startup path, so this node cannot run the server."
    echo "FATAL: resubmit excluding it, e.g.  -R \"select[hname != '$(hostname | cut -d. -f1)']\""
    df -h /tmp 2>&1 | sed 's/^/FATAL: /'
    exit 1
fi
rm -f "${_probe}"

# ---- warm start vs resume, decided from what is on disk ----
# resume_from must name a CHECKPOINT DIR, not the output dir: the trainer's resolve_resume_dir
# would expand '*-latest', but the producer-side reconciliation in launch.py builds its path as
# os.path.join(resume_from, "training_state.pt") with no such expansion.
RESUME_DIR=""
if [[ -f "${STABLE_OUT}/${RUN_ID}-latest/training_state.pt" ]]; then
    RESUME_DIR="${STABLE_OUT}/${RUN_ID}-latest"
else
    # Completeness filtered BEFORE taking the maximum: a save interrupted by preemption leaves a
    # step dir with no training_state.pt, and "take the max, then check" would reject that torn
    # dir and silently declare the run FRESH -- discarding every checkpoint AND the ledger.
    best=""
    for d in "${STABLE_OUT}/${RUN_ID}"-step*; do
        [[ -f "${d}/training_state.pt" ]] || continue      # also skips the unmatched glob
        s=${d##*-step}
        [[ "${s}" =~ ^[0-9]+$ ]] || continue
        if [[ -z "${best}" || "${s}" -gt "${best}" ]]; then best="${s}"; fi
    done
    [[ -n "${best}" ]] && RESUME_DIR="${STABLE_OUT}/${RUN_ID}-step${best}"
fi

OVERRIDES=("deployment.disaggregated.control_dir=${CONTROL_DIR}")
if [[ -n "${RESUME_DIR}" ]]; then
    # draft_checkpoint_path and resume_from are mutually exclusive in the schema, and no override
    # value can clear the former once set ('', '~' and 'None' were each tested), which is why the
    # YAML omits it and the launcher injects exactly one of the two.
    OVERRIDES+=("training.resume_from=${RESUME_DIR}")
    MODE="RESUME from ${RESUME_DIR}"
else
    OVERRIDES+=("model.draft_checkpoint_path=${WARM_START}")
    MODE="FRESH (warm start from ${WARM_START})"
    # A first attempt that committed samples but died before its first checkpoint leaves a
    # non-empty ledger with nothing to reconcile against; launch.py rejects that outright
    # ("a fresh online attempt cannot reuse a ledger"). Retire it -- there is no trained state
    # to protect, only the warm start we are about to redo.
    if [[ -f "${LEDGER}" || -d "${STABLE_STATE}/inboxes" ]]; then
        echo "[wrapper] no checkpoint but consumer state exists -> retiring stale ledger + inboxes"
        # inboxes too, not just the sqlite: the per-rank inbox files live at stable paths and
        # carry .closed markers and .consumed_count offsets; a fresh consumer that reads a
        # stale .closed marker sees instant end-of-stream instead of this attempt's refs.
        rm -f "${LEDGER}" "${LEDGER}-wal" "${LEDGER}-shm"
        rm -rf "${STABLE_STATE}/inboxes"
    fi
fi

echo "[wrapper] host=$(hostname | cut -d. -f1)  /tmp $(df -h /tmp | awk 'NR==2{print $4" free"}')"
echo "[wrapper] mode=${MODE}"
echo "[wrapper] config=${CONFIG}"
echo "[wrapper] output_dir=${STABLE_OUT}   (stable)"
echo "[wrapper] ledger=${LEDGER}   (stable)"
echo "[wrapper] control_dir=${CONTROL_DIR}   (fresh per attempt)"
echo "[wrapper] gpus: server=0,1 (TP2)  trainer=2-7 (dp6)"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader | sed 's/^/[wrapper] gpu /'

# --role defaults to auto, which managed_local requires (it rejects an explicit producer/consumer
# split); specforge then supervises mooncake + capture server + producer + 6-rank consumer itself.
cd "${SPECFORGE}"
exec specforge train -c "${CONFIG}" "${OVERRIDES[@]}"
