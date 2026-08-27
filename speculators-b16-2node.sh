# FORMAL speculators XPress b16 run, TWO nodes (runs once per node via submit.sh/blaunch):
#   EVERY node: its own vLLM verifier (GPU 0) + 7 trainer ranks (GPUs 1-7)
#   cluster rank 0 additionally: one-time data prep (marker-gated)
# => 14 trainer ranks, one torchrun world, rendezvous at rank 0.
#
# WHY per-node vLLM (do not "simplify" to one shared server): the hs_connector returns
# hidden-state HANDLES that are paths under the SERVER's local /tmp/hidden_states. A trainer
# can only read a handle written on ITS OWN node. The first 2-node attempt pointed node 1's
# trainers at node 0's vLLM -> 100% "Failed to load hidden states for handle /tmp/..." on
# node 1 (1428 dead samples, half the world training on nothing). localhost endpoints make
# every handle local by construction, and double generation throughput as a bonus.
#
# FORK-PARITY ALIGNED (2026-08-13): backbone-lr-scale 1.0 (fork b16 golden arm ran the
# drafter at FULL lr, not 0.1), base-anchor 0.6 -> floor 0.2 linear (the golden arm's
# lambda schedule), --decayed-loss-norm (normalize by decayed weight sum like the fork;
# fixes the ~4x loss-scale difference and its grad-clip interaction), --no-packing
# (1 conversation/rank/step like the fork's bs1 -- kills the ~4.4x token-packing
# batch inflation), tv weight 1.8 (fork's l1=|p-q|=2*TV at 0.9 => tv 1.8 makes the
# objective IDENTICAL; 0.9 was half the fork's distribution-matching weight).
# Fork-name wandb aliases emitted by the trainer: train/ce, train/l1, train/tf,
# train/cons, train/lambda_base, train/accuracy, eval/{refiner,drafter}_accept_len.
# Known accepted deviations: gb = 7*(num_nodes) convs/step (28 at 4 nodes, vs fork
# gb32 -- GPU0/node is the vLLM verifier; no accum knob), vLLM online hiddens,
# data order, internal val protocol (~1.1 offset vs fork eval).
# Recipe = the fork's validated b16 arm, translated: z-lab b16 warm start (fill-in native),
# xpress rank 256, K=3 consistency 0.3, base anchor 0.6 DECAYED (validated for z-lab b16 --
# no --base-anchor-full-weight, per the b16 evidence), backbone-lr x0.1, lr 6e-4, warmup 4%,
# 10 epochs, full 1.35M-row regen dataset (the same jsonl the fork arms train on).
# wandb: project ripple-dspark (same as every other arm; WANDB_PROJECT env -- the wandb
# handler passes no explicit project, so the env var wins). Metric name mapping vs the fork:
#   fork eval/refiner_accept_len  <->  speculators val/refiner_accept_len
#   fork eval/drafter_accept_len  <->  speculators val/drafter_accept_len   (the guardrail)
#
#   bash submit.sh 2 xpress speculators-b16-2node
set -Eeuo pipefail

PY=/proj/checkpoints/zwang619/miniconda3/envs/speculators_train/bin/python
cd /proj/checkpoints/zwang619/speculators

export FLASHINFER_DISABLE_VERSION_CHECK=1
export HF_HOME=/proj/checkpoints/zwang619/.cache/huggingface
export TMPDIR=/dev/shm/${USER}/tmp && mkdir -p "$TMPDIR"
: "${WANDB_API_KEY:?export WANDB_API_KEY in your environment before running}"
export WANDB_PROJECT=ripple-dspark

NUM_NODES=$(echo ${LSB_MCPU_HOSTS} | tr ' ' '\n' | sed 'n; d' | wc -w)
NODE_RANK=$(($(echo ${LSB_MCPU_HOSTS} | tr ' ' '\n' | sed 'n; d' | grep -n -m1 $(hostname | cut -d'.' -f1) | cut -d':' -f1)-1))
HEAD_HOST=$(echo ${LSB_MCPU_HOSTS} | tr ' ' '\n' | sed 'n; d' | sed -n '1p')
HEAD_IP=$(getent ahostsv4 "${HEAD_HOST}" | awk '{print $1; exit}')
[[ "${HEAD_IP}" =~ ^[0-9.]+$ ]] || { echo "FATAL: cannot resolve ${HEAD_HOST}"; exit 1; }

MODEL="Qwen/Qwen3-8B"
OUT_ROOT=/proj/checkpoints/zwang619/speculators_out
BACKBONE=$OUT_ROOT/dflash_b16_zlab_converted        # already converted+morphed by the trial
DATA_JSONL=/proj/checkpoints/daviswer/results/specu/refiner_train_nothink.jsonl
OUTPUT_DIR=$OUT_ROOT/xpress_b16_zlab_full
RUN_NAME=xpress-b16-speculators-2node
VLLM_PORT=8300
RDZV_PORT=29610
SEQ_LENGTH=4096
TARGET_LAYER_IDS="1 9 17 25 33"
DATA_READY=$OUTPUT_DIR/.data_ready
mkdir -p "$OUTPUT_DIR"

echo "[node $NODE_RANK/$NUM_NODES] host=$(hostname -s) head=$HEAD_IP"

if [[ "$NODE_RANK" == "0" ]]; then
    # backbone must exist (trial created it); refuse to run without it rather than re-convert
    [[ -f "$BACKBONE/config.json" ]] || { echo "FATAL: converted backbone missing: $BACKBONE"; exit 1; }
    grep -q '"speculators_model_type": "xpress"' "$BACKBONE/config.json" || \
        $PY examples/train/xpress_b16_zlab_morph_config.py "$BACKBONE"

    if [[ ! -f "$DATA_READY" ]]; then
        echo "[node 0] preparing FULL dataset (one-time; resubmits skip via marker)"
        $PY scripts/prepare_data.py \
            --model "$MODEL" --data "$DATA_JSONL" --output "$OUTPUT_DIR" \
            --max-samples 1348810 --seq-length "$SEQ_LENGTH"
        touch "$DATA_READY"
    fi
else
    echo "[node $NODE_RANK] waiting for data marker"
    for i in $(seq 1 360); do [[ -f "$DATA_READY" ]] && break; sleep 10; done
    [[ -f "$DATA_READY" ]] || { echo "FATAL: data never became ready"; exit 1; }
fi

# leftover handles from a previous run on this node would slowly fill /tmp; start clean
rm -rf /tmp/hidden_states

echo "[node $NODE_RANK] launching LOCAL vLLM verifier on GPU 0"
CUDA_VISIBLE_DEVICES=0 $PY scripts/launch_vllm.py "$MODEL" \
    --target-layer-ids $TARGET_LAYER_IDS \
    --port "$VLLM_PORT" &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null || true' EXIT
for i in $(seq 1 120); do
    curl -fsS "http://localhost:${VLLM_PORT}/v1/models" >/dev/null 2>&1 && break
    kill -0 $VLLM_PID 2>/dev/null || { echo "FATAL: vLLM died during startup"; exit 1; }
    sleep 5
done
curl -fsS "http://localhost:${VLLM_PORT}/v1/models" >/dev/null || { echo "FATAL: vLLM not ready"; exit 1; }
echo "[node $NODE_RANK] local vLLM ready"
TRAIN_CUDA="1,2,3,4,5,6,7"

# RENDEZVOUS RACE GUARD: each node reaches this point only after ITS OWN vLLM is ready,
# and vLLM startup times differ by minutes across nodes. torchrun's c10d CONNECT phase
# gives up after ~60s if the head's store isn't listening yet (measured: node1 died with
# RendezvousConnectionError at +240s while node0 was still loading its vLLM). Non-head
# nodes therefore poll the head's rendezvous port and only start torchrun once node 0
# has opened it. On timeout we proceed anyway and let torchrun produce the real error.
if [[ "$NODE_RANK" != "0" ]]; then
    echo "[node $NODE_RANK] waiting for head rendezvous port ${HEAD_IP}:${RDZV_PORT}"
    for i in $(seq 1 240); do
        (exec 3<>/dev/tcp/${HEAD_IP}/${RDZV_PORT}) 2>/dev/null && { exec 3>&- 2>/dev/null; break; }
        sleep 5
    done
fi

echo "[node $NODE_RANK] starting 7 trainer ranks on GPUs $TRAIN_CUDA"
# expandable segments for the TRAINERS ONLY (anti-fragmentation; the run sits ~3GB from
# the ceiling and died to a step-7 OOM once). Must NOT be exported globally: vLLM's
# hidden-states connector refuses to start under expandable_segments (pydantic
# validation error) -- that is exactly how run 865xxx failed repeatedly.
CUDA_VISIBLE_DEVICES="$TRAIN_CUDA" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    $PY -m torch.distributed.run \
    --nnodes "$NUM_NODES" --node_rank "$NODE_RANK" \
    --rdzv_id xpressb16 --rdzv_backend c10d --rdzv_endpoint "${HEAD_IP}:${RDZV_PORT}" \
    --nproc_per_node 7 \
    scripts/train.py \
    --verifier-name-or-path "$MODEL" \
    --from-pretrained "$BACKBONE" \
    --data-path "$OUTPUT_DIR" \
    --vllm-endpoint "http://localhost:${VLLM_PORT}/v1" \
    --save-path "$OUTPUT_DIR/checkpoints" \
    --epochs 10 \
    --lr 6e-4 \
    --scheduler-warmup-ratio 0.04 \
    --optimizer adamw \
    --backbone-lr-scale 1.0 \
    --total-seq-len "$SEQ_LENGTH" \
    --speculator-type xpress \
    --max-anchors 400 \
    --target-layer-ids $TARGET_LAYER_IDS \
    --xpress-rank 256 \
    --consistency-weight 0.3 \
    --consistency-passes 3 \
    --base-anchor-weight 0.6 \
    --base-anchor-floor 0.2 \
    --decayed-loss-norm \
    --no-packing \
    --loss-fn '{"ce": 0.1, "tv": 1.8}' \
    --on-missing generate \
    --on-generate delete \
    --logger wandb \
    --run-name "$RUN_NAME"
