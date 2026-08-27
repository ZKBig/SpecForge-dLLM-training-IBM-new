#!/usr/bin/env bash
# LSF wrapper for Qwen3.5-35B-A3B sgu_dspark ONLINE training on FOUR nodes (32 H100).
# Runs ONCE PER NODE via blaunch. Topology:
#   cluster rank 0:    mooncake master + patched SGLang server (TP2, cuda 0-1) + CPU producer
#   cluster ranks 1-3: trainers, ONE torchrun world (nnodes=3, dp24), rendezvous at rank 1
#
#   cd /proj/checkpoints/zwang619/SpecForge-dLLM-training-IBM
#   bsub -q normal -G grp_ai_compiler_design -M 2000G -hl -n 4 -J xpress/train-35b-sgu-dspark-4node \
#     -gpu "num=8/task:mode=exclusive_process" -R "select[hname != 'p1-r04-n2']" \
#     -oo /proj/checkpoints/zwang619/results/xpress/train-35b-sgu-dspark-4node/output.log \
#     -eo /proj/checkpoints/zwang619/results/xpress/train-35b-sgu-dspark-4node/err.log \
#     blaunch bash train-35b-sgu-dspark-4node.sh
#
# Delegates to OUR 4-node launcher (run_disagg_4node.sh, forked from upstream's 2-node one:
# NUM_NODES==4, trainer ranks 1-3 join one torchrun world via --node-rank, per-node
# consumer.done.<rank> markers, and the consumer_pid EXIT-trap bug fixed).
#
# Inherits every hard-won behavior of the proven 2-node wrapper -- stable-vs-fresh state
# split, RCLI_* identity, Triton/tmp env, /tmp preflight, FRESH/RESUME auto-detection,
# ledger marker alignment, orphaned-ref release -- with two 4-node-specific changes:
#   * resume_from / draft_checkpoint_path go to ALL trainer nodes (gate NODE_RANK != 0, not
#     == 1): the three trainer nodes form ONE torchrun world and their configs must be
#     IDENTICAL, or ranks desynchronize. The producer still gets neither (schema forbids
#     resume_from on the producer role). Ledger surgery stays single-writer on rank 1.
#   * FRESH warm-starts from the 2-NODE run's newest checkpoint (weights only; specforge
#     format, strategy stamp 'sgu_dspark', all 79 keys verified) so the dp8 progress carries
#     over. dp24 changes the LR-schedule contract, so the 2-node checkpoints cannot be
#     resumed here -- weights-only warm start is exactly the right (and only) bridge.
set -Eeuo pipefail

SPECFORGE=/proj/checkpoints/zwang619/SpecForge
ENV_BIN=/proj/checkpoints/zwang619/miniconda3/envs/dLLM_35b/bin
export PATH="${ENV_BIN}:${PATH}"
export HF_HOME=/proj/checkpoints/zwang619/.cache/huggingface

# ---- Triton JIT and sglang /tmp writers (each cost a 2-node attempt) ----
export TRITON_CACHE_DIR="${HOME}/.triton/cache"
export TMPDIR="/dev/shm/${USER}/tmp"          # short: AF_UNIX names cap at 108 bytes
export CC=/usr/bin/gcc                         # not the ccache shim ($HOME/.ccache never existed)
export SGLANG_NUMA_BIND_V2=0                   # numa_utils writes a hardcoded /tmp script

# ---- cluster identity ----
NUM_NODES=$(echo ${LSB_MCPU_HOSTS} | tr ' ' '\n' | sed 'n; d' | wc -w)
NODE_RANK=$(($(echo ${LSB_MCPU_HOSTS} | tr ' ' '\n' | sed 'n; d' | grep -n -m1 $(hostname | cut -d'.' -f1) | cut -d':' -f1)-1))
export RCLI_NODE_RANK="${NODE_RANK}"
export RCLI_NUM_NODES="${NUM_NODES}"
export -n NODE_RANK NUM_NODES 2>/dev/null || true   # NODE_RANK means something else to specforge

HEAD_HOST=$(echo ${LSB_MCPU_HOSTS} | tr ' ' '\n' | sed 'n; d' | sed -n '1p')
TRAINER_HEAD_HOST=$(echo ${LSB_MCPU_HOSTS} | tr ' ' '\n' | sed 'n; d' | sed -n '2p')
# ahostsv4: `getent hosts` can return an unroutable IPv6 link-local first on this cluster.
export HEAD_IP=$(getent ahostsv4 "${HEAD_HOST}" | awk '{print $1; exit}')
TRAINER_HEAD_IP=$(getent ahostsv4 "${TRAINER_HEAD_HOST}" | awk '{print $1; exit}')
for pair in "HEAD_IP:${HEAD_IP}" "TRAINER_HEAD_IP:${TRAINER_HEAD_IP}"; do
    v=${pair#*:}
    if [[ ! "${v}" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        echo "FATAL: could not resolve an IPv4 address for ${pair%%:*} (got '${v}')"
        exit 1
    fi
done

# ---- per-attempt identity (launcher demands virgin RUN_ROOT; Mooncake needs a fresh store) ----
export DISAGG_STORE_ID="qwen35-35b-sgu-dspark-4node-${LSB_JOBID:-manual}"
export DISAGG_RUN_ROOT="/proj/checkpoints/zwang619/disagg_runs/${DISAGG_STORE_ID}"
# Disposable per-attempt lock dir (the launcher's atomic mkdir); the REAL ledger location is
# the config override below. Node-local by design -- each trainer node takes its own lock.
export DISAGG_CONSUMER_STATE_DIR="/dev/shm/${USER}/specforge/${DISAGG_STORE_ID}/lock"

# ---- run-lifetime identity: survives every attempt ----
STABLE_RUN_ID="qwen35-35b-sgu-dspark-4node"
STABLE_ROOT="/proj/checkpoints/zwang619/results/xpress/train-35b-sgu-dspark-4node"
STABLE_OUT="${STABLE_ROOT}/output"
STABLE_STATE="${STABLE_ROOT}/consumer-state"
LEDGER="${STABLE_STATE}/consumer.sqlite"
# The 2-node run's newest checkpoint = weights-only warm start for the FIRST 4-node attempt.
TWONODE_OUT="/proj/checkpoints/zwang619/results/xpress/train-35b-sgu-dspark/output"

# ---- 35B server settings (identical to the proven 2-node run) ----
export CONFIG="${SPECFORGE}/examples/configs/qwen3.5-35b-a3b-sgu-dspark-4node-online.yaml"
export RUN_LABEL="qwen35-35b-sgu-dspark-4node"
export TARGET_MODEL_PATH=/proj/checkpoints/ashishagr/model_downloads/qwen3.5-35b-a3b
export SERVER_GPUS="0,1"          # num_key_value_heads=2 -> TP2 is this model's natural split
export SERVER_TP=2
export SERVER_MEM_FRACTION=0.85
export CAPTURE_LAYER_IDS="1 6 11 16 22 27 32 37"
export SERVER_EXTRA_ARGS="--disable-fast-image-processor"   # VL warmup vs EXCLUSIVE_PROCESS
export TRAINER_GPUS="0,1,2,3,4,5,6,7"
export TRAINER_NPROC=8
export APPLY_SGLANG_CAPTURE_PATCH=0
export START_TIMEOUT_S=3600
export PEER_TIMEOUT_S=3600

mkdir -p "$(dirname "${DISAGG_RUN_ROOT}")" "${STABLE_OUT}" "${STABLE_STATE}" \
         "${TMPDIR}" "${TRITON_CACHE_DIR}"

# ---- /tmp preflight: fail in seconds naming the host ----
_probe="/tmp/.specforge_tmp_probe.$$"
if ! dd if=/dev/zero of="${_probe}" bs=1M count=8 status=none 2>/dev/null; then
    rm -f "${_probe}" 2>/dev/null || true
    echo "FATAL: /tmp is unusable on $(hostname | cut -d. -f1) -- cannot write 8 MiB."
    echo "FATAL: resubmit excluding it:  -R \"select[hname != '$(hostname | cut -d. -f1)']\""
    df -h /tmp 2>&1 | sed 's/^/FATAL: /'
    exit 1
fi
rm -f "${_probe}"
echo "[wrapper] /tmp OK on $(hostname | cut -d. -f1) ($(df -h /tmp | awk 'NR==2{print $4" free"}'))"

# ---- warm start vs resume, decided from THIS run's own output dir ----
RESUME_DIR=""
if [[ -f "${STABLE_OUT}/${STABLE_RUN_ID}-latest/training_state.pt" ]]; then
    RESUME_DIR="${STABLE_OUT}/${STABLE_RUN_ID}-latest"
else
    # Completeness filtered BEFORE taking the maximum (a torn save must not demote us to FRESH).
    best=""
    for d in "${STABLE_OUT}/${STABLE_RUN_ID}"-step*; do
        [[ -f "${d}/training_state.pt" ]] || continue
        s=${d##*-step}
        [[ "${s}" =~ ^[0-9]+$ ]] || continue
        if [[ -z "${best}" || "${s}" -gt "${best}" ]]; then best="${s}"; fi
    done
    [[ -n "${best}" ]] && RESUME_DIR="${STABLE_OUT}/${STABLE_RUN_ID}-step${best}"
fi

OVERRIDES=(
    "run_id=${STABLE_RUN_ID}"
    "output_dir=${STABLE_OUT}"
    "deployment.disaggregated.consumer_state_dir=${STABLE_STATE}"
    "deployment.trainer.nnodes=3"
    "deployment.trainer.nproc_per_node=${TRAINER_NPROC}"
    "deployment.trainer.master_addr=${TRAINER_HEAD_IP}"
)
if [[ -n "${RESUME_DIR}" ]]; then
    # ALL trainer nodes (identical torchrun-world configs); never the producer (schema forbids).
    if [[ "${NODE_RANK}" != "0" ]]; then
        OVERRIDES+=("training.resume_from=${RESUME_DIR}")
    fi
    # Ledger surgery once, on rank 1 (single writer).
    if [[ "${NODE_RANK}" == "1" ]]; then
        CKPT_REAL=$(readlink -f "${RESUME_DIR}")
        CKPT_STEP=${CKPT_REAL##*-step}
        if [[ "${CKPT_STEP}" =~ ^[0-9]+$ && -f "${LEDGER}" ]]; then
            "${ENV_BIN}/python" - "${LEDGER}" "${CKPT_STEP}" <<'PYEOF'
import json, sqlite3, sys
db, step = sys.argv[1], int(sys.argv[2])
conn = sqlite3.connect(db)
row = conn.execute("SELECT v FROM marker WHERE k='global_step'").fetchone()
cur = json.loads(row[0]) if row else None
if cur is not None and cur > step:
    conn.execute("INSERT OR REPLACE INTO marker (k, v) VALUES ('global_step', ?)",
                 (json.dumps(step),))
    conn.commit()
    print(f"[wrapper] ledger marker rewound {cur} -> {step}; "
          f"samples acked during steps {step+1}..{cur} are forfeited (bounded by save_interval)")
elif cur is not None and cur < step:
    print(f"[wrapper] WARNING: ledger marker {cur} is BEHIND checkpoint {step}; "
          f"not touching it -- reconciliation will fail; investigate the ledger")
    raise SystemExit(0)
else:
    print(f"[wrapper] ledger marker already aligned at {cur}")
# In-flight refs' tensors died with the previous attempt's Mooncake store; release them.
orphans = conn.execute(
    "SELECT COUNT(*) FROM committed WHERE sample_id NOT IN (SELECT sample_id FROM acked)"
).fetchone()[0]
if orphans:
    conn.execute(
        "INSERT OR IGNORE INTO acked (sample_id) "
        "SELECT sample_id FROM committed WHERE sample_id NOT IN (SELECT sample_id FROM acked)")
    conn.commit()
    print(f"[wrapper] {orphans} committed-but-unacked refs marked acked "
          f"(their tensors died with the previous attempt's Mooncake store)")
conn.close()
PYEOF
        fi
    fi
    MODE="RESUME from ${RESUME_DIR}"
else
    # FRESH: weights-only warm start from the 2-node run's newest complete checkpoint.
    # Specforge-format checkpoints are accepted directly by the warm-start loader (it reads
    # draft_state_dict and checks the strategy stamp, which is 'sgu_dspark' since the rename).
    WARM_SRC=""
    if [[ -f "${TWONODE_OUT}/qwen35-35b-sgu-dspark-latest/training_state.pt" ]]; then
        WARM_SRC=$(readlink -f "${TWONODE_OUT}/qwen35-35b-sgu-dspark-latest")
    else
        best=""
        for d in "${TWONODE_OUT}/qwen35-35b-sgu-dspark"-step*; do
            [[ -f "${d}/training_state.pt" ]] || continue
            s=${d##*-step}
            [[ "${s}" =~ ^[0-9]+$ ]] || continue
            if [[ -z "${best}" || "${s}" -gt "${best}" ]]; then best="${s}"; fi
        done
        [[ -n "${best}" ]] && WARM_SRC="${TWONODE_OUT}/qwen35-35b-sgu-dspark-step${best}"
    fi
    if [[ -z "${WARM_SRC}" ]]; then
        # No 2-node checkpoint anywhere -> fall back to the offline z-lab+SGU merge.
        WARM_SRC=/proj/checkpoints/zwang619/warm_start/qwen3.5-35b-a3b-sgu-dspark-init
    fi
    if [[ "${NODE_RANK}" != "0" ]]; then
        OVERRIDES+=("model.draft_checkpoint_path=${WARM_SRC}")
    fi
    MODE="FRESH (warm start from ${WARM_SRC})"
    if [[ ( -f "${LEDGER}" || -d "${STABLE_STATE}/inboxes" ) && "${NODE_RANK}" == "1" ]]; then
        echo "[wrapper] no checkpoint but consumer state exists -> retiring stale ledger + inboxes"
        rm -f "${LEDGER}" "${LEDGER}-wal" "${LEDGER}-shm"
        rm -rf "${STABLE_STATE}/inboxes"
    fi
fi

echo "[wrapper] host=$(hostname | cut -d. -f1) rank=${NODE_RANK}/${NUM_NODES} head=${HEAD_IP} trainer_head=${TRAINER_HEAD_IP}"
echo "[wrapper] mode=${MODE}"
echo "[wrapper] run_root=${DISAGG_RUN_ROOT}   (fresh per attempt)"
echo "[wrapper] output_dir=${STABLE_OUT}   (stable)"
echo "[wrapper] ledger=${LEDGER}   (stable)"
echo "[wrapper] config=${CONFIG}"

exec bash "$(dirname "${BASH_SOURCE[0]}")/run_disagg_4node.sh" "${OVERRIDES[@]}"
