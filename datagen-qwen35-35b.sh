#!/usr/bin/env bash
# Multi-node Qwen3.5-35B-A3B data generation, submit.sh/blaunch style (runs ONCE PER NODE).
# Per node: 4 sglang workers x TP2 (67 GiB MoE weights need a GPU pair each), then the node's
# input shard is regenerated against the 4 local servers. Same shard/resume mechanics as
# datagen-qwen3-14b.sh.
#
#   cd /proj/checkpoints/zwang619/SpecForge-dLLM-training-IBM
#   bash submit.sh 2 datagen datagen-qwen35-35b
#
# After ALL shards finish:
#   cat /proj/checkpoints/zwang619/DeepSpec/train_datasets/qwen35_35b_a3b/regen.shard*.jsonl \
#     > /proj/checkpoints/zwang619/DeepSpec/train_datasets/qwen35_35b_a3b/perfectblend_train_regen.jsonl
set -euo pipefail

ENV_BIN=/proj/checkpoints/zwang619/miniconda3/envs/dLLM_35b/bin
export PATH="${ENV_BIN}:${PATH}"
export HF_HOME=/proj/checkpoints/zwang619/.cache/huggingface
cd /proj/checkpoints/zwang619/DeepSpec

model=/proj/checkpoints/ashishagr/model_downloads/qwen3.5-35b-a3b

NNODES=$(echo ${LSB_MCPU_HOSTS} | tr ' ' '\n' | sed 'n; d' | wc -w)
NODE_RANK=$(($(echo ${LSB_MCPU_HOSTS} | tr ' ' '\n' | sed 'n; d' | grep -n -m1 $(hostname | cut -d'.' -f1) | cut -d':' -f1)-1))
echo "[datagen-35b] host=$(hostname) rank=${NODE_RANK}/${NNODES}"

shard_dir=train_datasets/qwen35_35b_a3b
mkdir -p "${shard_dir}" logs/sglang_qwen35_node${NODE_RANK}
shard_in=${shard_dir}/input.shard${NODE_RANK}-of-${NNODES}.jsonl
if [[ ! -s "${shard_in}" ]]; then
    awk -v n="${NNODES}" -v r="${NODE_RANK}" 'NR % n == r' \
        train_datasets/perfectblend_train.jsonl > "${shard_in}.tmp"
    mv "${shard_in}.tmp" "${shard_in}"
fi
echo "[datagen-35b] shard ${NODE_RANK}: $(wc -l < ${shard_in}) rows"

# ---- 4 local workers x TP2 ----
ports=(30000 30001 30002 30003)
pids=()
cleanup() { for p in "${pids[@]:-}"; do kill "$p" > /dev/null 2>&1 || true; done; wait || true; }
trap cleanup INT TERM EXIT
for w in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$((w * 2)),$((w * 2 + 1)) sglang serve \
        --model-path "${model}" \
        --host 0.0.0.0 --port $((30000 + w)) --nccl-port $((31000 + w)) \
        --tp-size 2 \
        --dtype bfloat16 --mem-fraction-static 0.85 \
        --disable-fast-image-processor \
        > logs/sglang_qwen35_node${NODE_RANK}/worker_${w}.log 2>&1 &
    pids+=("$!")
done
# disable-fast-image-processor: VL warmup would otherwise open a second CUDA context from the
# TokenizerManager process and die on LSF's EXCLUSIVE_PROCESS GPUs (verified in the smoke test).

deadline=$((SECONDS + 2400))
for port in "${ports[@]}"; do
    until curl -sf "http://127.0.0.1:${port}/health" > /dev/null 2>&1; do
        ((SECONDS >= deadline)) && { echo "[datagen-35b] FATAL: port ${port} not healthy after 40min"; exit 1; }
        sleep 15
    done
    echo "[datagen-35b] node ${NODE_RANK} port ${port} healthy"
done

# concurrency 128 over 4 servers = 32 in-flight per TP2 replica
python scripts/data/generate_train_data.py \
    --model "${model}" \
    --server-address 127.0.0.1:30000 127.0.0.1:30001 127.0.0.1:30002 127.0.0.1:30003 \
    --concurrency 128 \
    --temperature 0.7 --top-p 0.8 --top-k 20 --min-p 0 \
    --max-tokens 4096 \
    --disable-thinking \
    --resume \
    --input-file-path "${shard_in}" \
    --output-file-path "${shard_dir}/regen.shard${NODE_RANK}-of-${NNODES}.jsonl"

echo "[datagen-35b] node ${NODE_RANK} DONE"
