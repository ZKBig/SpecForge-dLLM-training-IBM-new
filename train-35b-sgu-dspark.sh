#!/usr/bin/env bash
# LSF wrapper for Qwen3.5-35B-A3B sgu_dspark ONLINE training (submit.sh/blaunch style: this
# script runs ONCE PER NODE). It only derives this cluster's identity (node rank, head IP,
# run root), decides warm-start-vs-resume, and then delegates to upstream's own 2-node
# launcher, which owns the real orchestration -- Mooncake master, patched SGLang server,
# producer/consumer handshake:
#
#     SpecForge/examples/disagg/run_qwen3_8b_dflash_disagg_2node.sh
#
# Topology (upstream's design, not ours):
#   rank 0 = mooncake_master + patched SGLang server (GPUs 0,1 via TP2) + CPU producer
#   rank 1 = consumer/trainer (GPUs 0-7)
# Control state travels through the shared RUN_ROOT on GPFS; tensors go through Mooncake.
#
#   cd /proj/checkpoints/zwang619/SpecForge-dLLM-training-IBM
#   bash submit.sh 2 xpress train-35b-sgu-dspark
#
# ---------------------------------------------------------------------------------------
# RESUMABILITY (re-submit the same command; the script figures out which mode it is in)
# ---------------------------------------------------------------------------------------
# Two sets of state pull in opposite directions, which is why the paths below are split:
#
#   FRESH per attempt -- the launcher refuses to reuse an existing RUN_ROOT, and Mooncake
#   needs an unused store_id, so RUN_ROOT / control_dir / store_id are stamped with $LSB_JOBID.
#
#   STABLE across attempts -- all three of these, or resume is impossible:
#     * output_dir  : checkpoints live at $STABLE_OUT/{run_id}-step{N} with a {run_id}-latest
#                     pointer. The old script put this under $RUN_ROOT, so every attempt wrote
#                     its checkpoints somewhere new and none could ever see the previous ones.
#     * run_id      : it is the checkpoint-name PREFIX. Left equal to the jobid-stamped
#                     DISAGG_STORE_ID (the launcher's default), attempt 2 would drop a second
#                     '*-latest' into the same output_dir and CheckpointManager.resolve_resume_dir
#                     raises "contains multiple complete *-latest runs". So run_id is pinned to
#                     a jobid-free constant and only store_id keeps the jobid.
#     * consumer_state_dir : holds consumer.sqlite, the durable ack ledger. Online resume is
#                     NOT weights-only -- launch.py reconciles that ledger to build the skip
#                     list of already-consumed prompts, and it hard-fails unless the ledger's
#                     durable marker equals the checkpoint's global_step exactly. Pointing it
#                     at /dev/shm/<jobid>/... (node-local AND jobid-stamped, so wiped on
#                     preemption) means attempt 2 meets an empty ledger and dies with "durable
#                     marker global_step=None is behind checkpoint ... global_step=N". It now
#                     lives on GPFS. SQLite's hardcoded journal_mode=WAL was verified working
#                     on this GPFS mount (write, reopen, recover); only one process (consumer
#                     dp_rank 0) ever opens it. NOTE it is passed as a CONFIG OVERRIDE, not via
#                     DISAGG_CONSUMER_STATE_DIR -- see the lock-directory comment below.
#
# Preemption on this cluster is Suspended -> Running (verified in bhist: 45 min in, suspended
# by a higher-priority job, running again 49 s later), i.e. SIGSTOP/SIGCONT with the process
# intact -- not a requeue. So a jobid-stamped RUN_ROOT is safe: a hard kill ends the job and
# you resubmit by hand, which draws a new LSB_JOBID.
#
# Mode selection is automatic:
#   no checkpoint yet -> model.draft_checkpoint_path=z-lab/... (weights-only warm start)
#   checkpoint found  -> training.resume_from=<that checkpoint dir>
# Never both: the schema declares them mutually exclusive and NO override value can clear
# draft_checkpoint_path once set ('', '~', 'None' were each tested and all still trip the
# check), which is why the YAML no longer contains it at all.
#
# Do NOT change seed, deployment.trainer.nproc_per_node, batch_size or accumulation_steps
# between attempts: the online prompt plan is a seeded shuffle and the consumer validates the
# producer's schedule contract against the checkpoint, so a change invalidates every existing
# checkpoint.
set -Eeuo pipefail

SPECFORGE=/proj/checkpoints/zwang619/SpecForge
ENV_BIN=/proj/checkpoints/zwang619/miniconda3/envs/dLLM_35b/bin
export PATH="${ENV_BIN}:${PATH}"                       # mooncake_master + specforge live here
export HF_HOME=/proj/checkpoints/zwang619/.cache/huggingface

# ---- Triton JIT: off the node-local /tmp and off the ccache shim ----
# Job 746443 died in CUDA-graph capture, not from memory: capturing the flashattention
# metadata kernel makes Triton JIT a launcher stub, and
#   subprocess.CalledProcessError: ['/usr/lib64/ccache/gcc', '/tmp/tmpnxarfvim/...c', ...]
#                                  returned non-zero exit status 1
# killed both TP ranks. Two node-local variables were in that command, neither ours:
#   * /tmp -- writability varies per compute node here (the same reason Python's tempfile
#     falls back to CWD on these nodes). TMPDIR moves the compile out of it. /dev/shm is
#     already known-writable on these nodes. Keep this path SHORT: AF_UNIX socket names are
#     capped at 108 bytes and torch/sglang create sockets under TMPDIR, so a long
#     jobid-stamped path would risk truncation for no benefit -- this is transient scratch
#     and needs no per-attempt isolation (tempfile still generates unique names inside).
#   * /usr/lib64/ccache/gcc -- `which gcc` finds the ccache shim first, and $HOME/.ccache has
#     never actually been created. Triton reads $CC (runtime/build.py:26) before falling back
#     to which(), so pointing CC at the real gcc takes ccache out of the picture entirely.
# TRITON_CACHE_DIR is set explicitly to what the default already resolves to
# (knobs.py: TRITON_HOME defaults to ~/), so the 22 launcher .so files already cached there
# keep being reused -- compile_module_from_src returns from cache before ever calling gcc.
# This matters on BOTH nodes: the trainer's flex_attention backend JITs through Triton too.
export TRITON_CACHE_DIR="${HOME}/.triton/cache"
export TMPDIR="/dev/shm/${USER}/tmp"
export CC=/usr/bin/gcc

# ---- the other /tmp writer, which TMPDIR cannot redirect ----
# Job 746742 (same node pair as 746443) died even earlier, before any weight loading:
#   numa_utils.py:53 path.write_text(script) -> OSError: [Errno 28] No space left on device
# That is DISK, not memory (RAM exhaustion would surface as CUDA OOM or a kernel OOM kill, and
# a read-only mount would be EROFS, not ENOSPC), and numa_utils.py:51 hardcodes
#   f"/tmp/sglang_temp_file_{time}_{rand}.sh"
# so TMPDIR has no effect on it. The whole numactl wrapper is gated on SGLANG_NUMA_BIND_V2
# (environ.py:738, EnvBool default True, accepts "0"), so turning it off removes the only
# hardcoded-/tmp write on the startup critical path. Cost: the scheduler processes are no
# longer NUMA-pinned near their GPU -- a throughput optimization, not a correctness feature.
export SGLANG_NUMA_BIND_V2=0

# ---- cluster identity, same derivation as the training scripts ----
NUM_NODES=$(echo ${LSB_MCPU_HOSTS} | tr ' ' '\n' | sed 'n; d' | wc -w)
NODE_RANK=$(($(echo ${LSB_MCPU_HOSTS} | tr ' ' '\n' | sed 'n; d' | grep -n -m1 $(hostname | cut -d'.' -f1) | cut -d':' -f1)-1))
# RCLI_NODE_RANK, and deliberately NOT an exported NODE_RANK. Both sides read $NODE_RANK and
# mean different things by it: the launcher wants this node's rank in the 2-node CLUSTER (0 =
# server, 1 = trainer), while specforge's _resolved_node_rank() reads the same variable as the
# rank within the TRAINER group and validates it against deployment.trainer.nnodes, which is 1
# here. Exporting NODE_RANK=1 therefore killed the consumer with
#   ValueError: node_rank=1 must be in [0, 1)
# and upstream's finish() EXIT trap then masked it with "consumer_pid: unbound variable".
# The launcher accepts RCLI_NODE_RANK as its own input (line 14: NODE_RANK="${NODE_RANK:-...}"),
# so routing our cluster rank through it leaves specforge to see no NODE_RANK at all and fall
# back to None -- which launch_plan treats as rank 0 by design (`node_rank in (None, 0)`).
# NUM_NODES goes through its own RCLI_ alias for symmetry (launcher line 15) and to keep our
# cluster topology out of the child's env entirely.
export RCLI_NODE_RANK="${NODE_RANK}"
export RCLI_NUM_NODES="${NUM_NODES}"
# Not paranoia: in bash a plain assignment to an ALREADY-EXPORTED name keeps the export
# attribute, so if anything upstream of us (LSF, a profile) exported NODE_RANK, the lines above
# would still leak it to the consumer. Strip the attribute explicitly, keeping the local value.
export -n NODE_RANK NUM_NODES 2>/dev/null || true
HEAD_HOST=$(echo ${LSB_MCPU_HOSTS} | tr ' ' '\n' | head -n 1)
# ahostsv4, not `getent hosts`: the latter can return an IPv6 link-local address first
# (fe80::...), which is both unroutable and, unbracketed, produces a malformed
# http://fe80::...:30000 URL in the Mooncake/server endpoints. Observed on this cluster.
export HEAD_IP=$(getent ahostsv4 "${HEAD_HOST}" | awk '{print $1; exit}')
if [[ ! "${HEAD_IP}" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "FATAL: could not resolve an IPv4 address for head host ${HEAD_HOST} (got '${HEAD_IP}')"
    exit 1
fi

# ---- per-attempt identity: one fresh Mooncake store + control dir per LSF attempt ----
export DISAGG_STORE_ID="qwen35-35b-sgu-dspark-${LSB_JOBID:-manual}"
export DISAGG_RUN_ROOT="/proj/checkpoints/zwang619/disagg_runs/${DISAGG_STORE_ID}"
# The launcher uses a NON-recursive `mkdir "$CONSUMER_STATE_DIR"` as an atomic create-or-fail
# lock ("consumer state already exists; choose a fresh DISAGG_CONSUMER_STATE_DIR"), so this
# path MUST NOT exist at start -- which is flatly incompatible with keeping the ledger here.
# So this stays a disposable per-attempt lock directory and the real ledger location is set
# separately, by overriding deployment.disaggregated.consumer_state_dir below. Python only
# ever uses that config value to derive <dir>/consumer.sqlite; nothing else lands in it.
export DISAGG_CONSUMER_STATE_DIR="/dev/shm/${USER}/specforge/${DISAGG_STORE_ID}/lock"

# ---- run-lifetime identity: survives every attempt (see RESUMABILITY above) ----
STABLE_RUN_ID="qwen35-35b-sgu-dspark"
# Deliberately the SCRIPT's name, not STABLE_RUN_ID: submit.sh derives the log directory as
# results/$2/$3 from the script name, so this puts output.log, err.log, output/ (checkpoints)
# and consumer-state/ in one place instead of two sibling directories.
STABLE_ROOT="/proj/checkpoints/zwang619/results/xpress/train-35b-sgu-dspark"
STABLE_OUT="${STABLE_ROOT}/output"
STABLE_STATE="${STABLE_ROOT}/consumer-state"      # holds consumer.sqlite, the durable ledger
LEDGER="${STABLE_STATE}/consumer.sqlite"
# z-lab's DFlash backbone + the SGU head's own init, merged offline (see the FRESH branch below).
WARM_START=/proj/checkpoints/zwang619/warm_start/qwen3.5-35b-a3b-sgu-dspark-init

# ---- 35B-specific server settings ----
export CONFIG="${SPECFORGE}/examples/configs/qwen3.5-35b-a3b-sgu-dspark-online.yaml"
export RUN_LABEL="qwen35-35b-sgu-dspark"
export TARGET_MODEL_PATH=/proj/checkpoints/ashishagr/model_downloads/qwen3.5-35b-a3b
# TP2 is not a compromise, it is what this architecture wants: the target has
# num_key_value_heads=2, so TP>2 would replicate KV heads, and moe_intermediate_size=512
# shards badly beyond 2. 67 GiB of MoE weights over 2 cards is ~33.5 GiB each.
export SERVER_GPUS="0,1"
export SERVER_TP=2
export SERVER_MEM_FRACTION=0.85
# The 8 layers z-lab's drafter was trained to consume (drafter fc width = 8 x 2048 = 16384).
# Must match dflash_config.target_layer_ids in configs/qwen3.5-35b-a3b-sgu-dspark.json.
export CAPTURE_LAYER_IDS="1 6 11 16 22 27 32 37"
# Qwen3.5 is a VL architecture: SGLang's warmup sends an IMAGE request and the fast image
# processor opens a SECOND CUDA context from the TokenizerManager process, which dies with
# cudaErrorDevicesUnavailable on LSF's EXCLUSIVE_PROCESS GPUs. Verified during data generation.
export SERVER_EXTRA_ARGS="--disable-fast-image-processor"
export TRAINER_GPUS="0,1,2,3,4,5,6,7"
export TRAINER_NPROC=8            # must equal deployment.trainer.nproc_per_node in the YAML
# The patch is already applied to this env's site-packages; re-applying per job is wasteful and
# would race between the two nodes on a shared filesystem.
export APPLY_SGLANG_CAPTURE_PATCH=0
export START_TIMEOUT_S=3600       # 67 GiB from GPFS + MoE init is slow
export PEER_TIMEOUT_S=3600

# NOT DISAGG_CONSUMER_STATE_DIR: creating that is exactly what trips the launcher's lock.
mkdir -p "$(dirname "${DISAGG_RUN_ROOT}")" "${STABLE_OUT}" "${STABLE_STATE}" \
         "${TMPDIR}" "${TRITON_CACHE_DIR}"

# ---- decide warm start vs resume from what is actually on disk ----
# resume_from must name a CHECKPOINT DIR, not the output dir: the trainer's resolve_resume_dir
# would expand '*-latest' for us, but the producer-side reconciliation in launch.py builds its
# path as os.path.join(resume_from, "training_state.pt") with no such expansion.
RESUME_DIR=""
if [[ -f "${STABLE_OUT}/${STABLE_RUN_ID}-latest/training_state.pt" ]]; then
    RESUME_DIR="${STABLE_OUT}/${STABLE_RUN_ID}-latest"
else
    # No usable '-latest' pointer (symlink lost, or it points at an incomplete dir): fall back
    # to the highest COMPLETE step dir, matching resolve_resume_dir's own fallback.
    # Completeness is filtered BEFORE taking the maximum, not after: a save interrupted by
    # preemption leaves a step dir with no training_state.pt, and "take the max, then check"
    # would see that torn dir, reject it, and silently declare the whole run FRESH -- throwing
    # away every checkpoint and retiring the ledger along with it.
    best=""
    for d in "${STABLE_OUT}/${STABLE_RUN_ID}"-step*; do
        [[ -f "${d}/training_state.pt" ]] || continue      # also skips the unmatched glob
        s=${d##*-step}
        [[ "${s}" =~ ^[0-9]+$ ]] || continue
        if [[ -z "${best}" || "${s}" -gt "${best}" ]]; then best="${s}"; fi
    done
    [[ -n "${best}" ]] && RESUME_DIR="${STABLE_OUT}/${STABLE_RUN_ID}-step${best}"
fi

OVERRIDES=(
    "run_id=${STABLE_RUN_ID}"
    "output_dir=${STABLE_OUT}"
    # Takes the ledger back from the launcher's throwaway lock dir. This is the single source
    # of truth: both _consumer_database_path() and launch_plan's DISAGG_DB derive from it, so
    # overriding the config value keeps every consumer of the path consistent.
    "deployment.disaggregated.consumer_state_dir=${STABLE_STATE}"
)
if [[ -n "${RESUME_DIR}" ]]; then
    # CONSUMER-ONLY, hence the NODE_RANK gate: resume is a trainer-role concept upstream. The
    # ledger reconciliation (skip list, durable-marker == checkpoint-step check) runs in the
    # consumer's dp_rank 0; the producer just regenerates the identical seeded prompt plan and
    # streams. Feeding resume_from to the producer command too (this script's trailing args go
    # to BOTH roles) dies in _config_for_role("producer")'s re-validation:
    #   ValueError: training.resume_from is valid only for a trainer role
    # and upstream's own error text confirms the design: "--role both cannot resume a
    # disaggregated producer; use --role consumer". (Cost one attempt: job 768173.)
    if [[ "${NODE_RANK}" == "1" ]]; then
        OVERRIDES+=("training.resume_from=${RESUME_DIR}")
        # ---- align the ledger's durable marker with the checkpoint ----
        # The consumer acks every optimizer step (controller.py: ack_fn at the step boundary),
        # so after a mid-interval kill the marker is AHEAD of the newest checkpoint (observed:
        # marker 556 vs checkpoint 500), and launch.py's reconciliation demands strict equality:
        #   RuntimeError: durable marker global_step=556 is ahead of checkpoint ... =500
        # A true rewind is impossible -- the acked table stores bare sample_ids with no step
        # provenance -- so the only losing move is bounded and deliberate: set the marker back
        # to the checkpoint step and accept that samples acked in the gap stay acked. Their
        # gradient contribution is rolled back with the weights and they will not be retrained:
        # at most save_interval * dp * batch = 500*8 = 4000 samples (~0.3% of the epoch) lost
        # per preemption. The alternative is no resume at all. Marker BEHIND checkpoint is NOT
        # auto-fixed: that means a lost/wrong ledger, which needs eyes, not silent repair.
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
    conn.execute(
        "INSERT OR REPLACE INTO marker (k, v) VALUES ('global_step', ?)",
        (json.dumps(step),),
    )
    conn.commit()
    print(f"[wrapper] ledger marker rewound {cur} -> {step}; "
          f"samples acked during steps {step+1}..{cur} are forfeited (bounded by save_interval)")
elif cur is not None and cur < step:
    print(f"[wrapper] WARNING: ledger marker {cur} is BEHIND checkpoint {step}; "
          f"not touching it -- reconciliation will fail; investigate the ledger")
    raise SystemExit(0)
else:
    print(f"[wrapper] ledger marker already aligned at {cur}")
# Committed-but-unacked refs (the in-flight window at kill time) are REQUEUED by
# reconcile_on_restart for at-least-once training -- but their feature tensors lived in the
# dead attempt's Mooncake store (store_id is jobid-stamped; the master restarts per attempt
# and "Storage root directory is not set. persisting data is disabled"), so the consumer's
# first fetch dies with
#   KeyError: sample ... feature 'hidden_states' not available (freed, stale, or never written)
# (cost one attempt: job 768801). Upstream's requeue design assumes a feature store that
# outlives the attempt; ours never does. Marking these refs acked makes reconciliation
# RELEASE them instead (skipped via skip_ids, prompt accounting settled) -- forfeiting at
# most the in-flight watermark (256 refs, ~0.02% of the epoch) per preemption, on top of the
# marker forfeit above.
orphans = conn.execute(
    "SELECT COUNT(*) FROM committed WHERE sample_id NOT IN (SELECT sample_id FROM acked)"
).fetchone()[0]
if orphans:
    conn.execute(
        "INSERT OR IGNORE INTO acked (sample_id) "
        "SELECT sample_id FROM committed WHERE sample_id NOT IN (SELECT sample_id FROM acked)"
    )
    conn.commit()
    print(f"[wrapper] {orphans} committed-but-unacked refs marked acked "
          f"(their tensors died with the previous attempt's Mooncake store)")
conn.close()
PYEOF
        fi
    fi
    MODE="RESUME from ${RESUME_DIR}"
else
    # NOT z-lab/Qwen3.5-35B-A3B-DFlash directly. model_loading.py excuses only ABSENT EMBEDDING
    # keys on a warm start (allow_missing_embedding) and ignores the provider's
    # allowed_missing_checkpoint_keys, which governs resume_from only -- so the raw z-lab
    # checkpoint failed on all 8 ranks with
    #   ValueError: warm-start checkpoint '...' is missing draft weights required by this
    #   architecture: ['markov_head.sgu.down_g.weight', ... 10 keys]
    # since the SGU head is new here and no DFlash/DSpark checkpoint can contain it.
    # WARM_START below is that checkpoint plus the head's own initialization, merged offline by
    # SpecForge/scripts/build_sgu_warm_start.py, so its key set is complete (79/79) and the
    # strict check passes on its own terms. Verified through the real warm_start_draft_model:
    # 0 missing, 0 unexpected, all 69 backbone tensors bit-identical to z-lab, and mix.L all
    # zeros so the refiner starts as a near-identity pass-through.
    OVERRIDES+=("model.draft_checkpoint_path=${WARM_START}")
    MODE="FRESH (warm start from ${WARM_START})"
    # A first attempt that committed samples but died before its first checkpoint leaves a
    # non-empty ledger with no checkpoint to reconcile against. launch.py rejects that outright
    # ("a fresh online attempt cannot reuse a ledger"), so retire it -- there is no trained
    # state to protect, only the warm start we are about to redo.
    if [[ ( -f "${LEDGER}" || -d "${STABLE_STATE}/inboxes" ) && "${NODE_RANK}" == "1" ]]; then
        echo "[wrapper] no checkpoint but consumer state exists -> retiring stale ledger + inboxes"
        # inboxes too, not just the sqlite: the per-rank inbox files live at stable paths and
        # carry .closed markers and .consumed_count offsets; a fresh consumer that reads a
        # stale .closed marker sees instant end-of-stream instead of this attempt's refs.
        # (Found on the 1-node validation run; same layout here.)
        rm -f "${LEDGER}" "${LEDGER}-wal" "${LEDGER}-shm"
        rm -rf "${STABLE_STATE}/inboxes"
    fi
fi

# ---- /tmp preflight ----
# Two attempts died on a full node-local /tmp in two different places (Triton's gcc could not
# write its .so; sglang's numactl script hit ENOSPC), both ~2 minutes in and both with errors
# that name neither /tmp nor the host. We cannot fix another node's full /tmp and we cannot
# redirect every hardcoded /tmp path in sglang, so at least fail in seconds and say WHICH host
# to exclude -- that is the established remedy here (submit.sh carries a commented-out
# `-R "select[hname != ...]"` line for exactly this).
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
echo "[wrapper] /tmp OK on $(hostname | cut -d. -f1) ($(df -h /tmp | awk 'NR==2{print $4" free"}'))"

echo "[wrapper] host=$(hostname) rank=${NODE_RANK}/${NUM_NODES} head=${HEAD_IP}"
echo "[wrapper] mode=${MODE}"
echo "[wrapper] run_root=${DISAGG_RUN_ROOT}   (fresh per attempt)"
echo "[wrapper] output_dir=${STABLE_OUT}   (stable)"
echo "[wrapper] ledger=${LEDGER}   (stable)"
echo "[wrapper] config=${CONFIG}"

# Trailing args land AFTER the launcher's COMMON_OVERRIDES on every `specforge train` command
# line (producer and consumer alike), and later overrides win -- that is how output_dir/run_id
# get taken back from $RUN_ROOT. store_id keeps the jobid because the launcher sets it from
# DISAGG_STORE_ID, which we are not touching here.
exec bash "${SPECFORGE}/examples/disagg/run_qwen3_8b_dflash_disagg_2node.sh" "${OVERRIDES[@]}"
