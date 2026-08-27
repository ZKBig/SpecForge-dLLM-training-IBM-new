#!/usr/bin/env bash
# 2-node Qwen3-14B data generation, submit.sh/blaunch style: this script runs ONCE PER NODE.
# Each node computes its rank from LSB_MCPU_HOSTS (same derivation as the training scripts),
# carves its own input shard (line k goes to node k%NNODES -- deterministic, no cross-node
# race), serves 8 local sglang workers, regenerates its shard, and tears down.
#
#   cd /proj/checkpoints/zwang619/SpecForge-dLLM-training-IBM
#   bash submit.sh 2 datagen datagen-qwen3-14b
#
# Preemption-safe: --resume + per-shard outputs; just resubmit. After BOTH shards finish:
#   cat /proj/checkpoints/zwang619/DeepSpec/train_datasets/qwen3_14b/regen.shard*.jsonl \
#     > /proj/checkpoints/zwang619/DeepSpec/train_datasets/qwen3_14b/perfectblend_train_regen.jsonl
set -euo pipefail

ENV_BIN=/proj/checkpoints/zwang619/miniconda3/envs/dLLM_35b/bin
export PATH="${ENV_BIN}:${PATH}"
export HF_HOME=/proj/checkpoints/zwang619/.cache/huggingface
cd /proj/checkpoints/zwang619/DeepSpec

# ---- which node am I (0-based), how many nodes ----
NNODES=$(echo ${LSB_MCPU_HOSTS} | tr ' ' '\n' | sed 'n; d' | wc -w)
NODE_RANK=$(($(echo ${LSB_MCPU_HOSTS} | tr ' ' '\n' | sed 'n; d' | grep -n -m1 $(hostname | cut -d'.' -f1) | cut -d':' -f1)-1))
echo "[datagen] host=$(hostname) rank=${NODE_RANK}/${NNODES}"

# ---- my input shard (idempotent: each node writes only its own file) ----
shard_dir=train_datasets/qwen3_14b
mkdir -p "${shard_dir}" logs/sglang_qwen3_14b_node${NODE_RANK}
shard_in=${shard_dir}/input.shard${NODE_RANK}-of-${NNODES}.jsonl
if [[ ! -s "${shard_in}" ]]; then
    awk -v n="${NNODES}" -v r="${NODE_RANK}" 'NR % n == r' \
        train_datasets/perfectblend_train.jsonl > "${shard_in}.tmp"
    mv "${shard_in}.tmp" "${shard_in}"
fi
echo "[datagen] shard ${NODE_RANK}: $(wc -l < ${shard_in}) rows"

# ---- 8 local workers ----
ports=(30000 30001 30002 30003 30004 30005 30006 30007)
pids=()
cleanup() { for p in "${pids[@]:-}"; do kill "$p" > /dev/null 2>&1 || true; done; wait || true; }
trap cleanup INT TERM EXIT
for gpu in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=${gpu} sglang serve \
        --model-path Qwen/Qwen3-14B \
        --host 0.0.0.0 --port $((30000 + gpu)) --nccl-port $((31000 + gpu)) \
        --dtype bfloat16 --mem-fraction-static 0.9 \
        > logs/sglang_qwen3_14b_node${NODE_RANK}/worker_gpu${gpu}.log 2>&1 &
    pids+=("$!")
done

deadline=$((SECONDS + 1800))
for port in "${ports[@]}"; do
    until curl -sf "http://127.0.0.1:${port}/health" > /dev/null 2>&1; do
        ((SECONDS >= deadline)) && { echo "[datagen] FATAL: port ${port} not healthy"; exit 1; }
        sleep 10
    done
    echo "[datagen] node ${NODE_RANK} port ${port} healthy"
done

# ---- regenerate my shard (concurrency 128 = 16 in-flight per worker; sglang batches them) ----
python scripts/data/generate_train_data.py \
    --model Qwen/Qwen3-14B \
    --server-address 127.0.0.1:30000 127.0.0.1:30001 127.0.0.1:30002 127.0.0.1:30003 \
                     127.0.0.1:30004 127.0.0.1:30005 127.0.0.1:30006 127.0.0.1:30007 \
    --concurrency 128 \
    --temperature 0.7 --top-p 0.8 --top-k 20 --min-p 0 \
    --max-tokens 4096 \
    --disable-thinking \
    --resume \
    --input-file-path "${shard_in}" \
    --output-file-path "${shard_dir}/regen.shard${NODE_RANK}-of-${NNODES}.jsonl"

echo "[datagen] node ${NODE_RANK} DONE"
