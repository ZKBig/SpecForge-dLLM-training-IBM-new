# PAPER-LEVEL REPRODUCTION: official code, FULL 1.35M dataset, official 10-epoch schedule
# (~26k optim steps), gb512, FOUR nodes (32 GPUs, ~2.3 days). Variant via $VARIANT:
#   conf1 (default) = official as-is (confidence alpha 1.0)  -- THE paper recipe
#   conf0           = confidence disabled, everything else identical
# Endpoint verdict: package eval vs deepseek-ai/dspark_qwen3_8b_block7 direct (our-protocol
# reference to be measured) and the paper's number as external reference.
set -Eeuo pipefail
VARIANT=${VARIANT:-conf1}

export PATH=/proj/checkpoints/zwang619/miniconda3/envs/deepspec_eval/bin:$PATH
export HF_HOME=/proj/checkpoints/zwang619/.cache/huggingface
export TOKENIZERS_PARALLELISM=false
: "${WANDB_API_KEY:?export WANDB_API_KEY in your environment before running}"
export DEEPSPEC_WANDB_PROJECT=ripple-dspark
export DEEPSPEC_WANDB_NAME=deepspec-official-FULL-${VARIANT}-4node
cd /proj/checkpoints/zwang619/DeepSpec

export MASTER_ADDR=$(echo ${LSB_MCPU_HOSTS} | tr ' ' '\n' | sed 'n; d' | sed -n '1p')
export MASTER_PORT=$(( 22000 + (${LSB_JOBID:-0} % 40000) ))
export WORLD_SIZE=$(echo ${LSB_MCPU_HOSTS} | tr ' ' '\n' | sed 'n; d' | wc -w)
export RANK=$(($(echo ${LSB_MCPU_HOSTS} | tr ' ' '\n' | sed 'n; d' | grep -n -m1 $(hostname | cut -d'.' -f1) | cut -d':' -f1)-1))
echo "[node $RANK/$WORLD_SIZE] master=$MASTER_ADDR:$MASTER_PORT variant=$VARIANT FULL DATA"

IDX=train_datasets/qwen3_8b_full_online/refiner_train_nothink.jsonl.online_index_ml4096_mt14.json
for i in $(seq 1 240); do [[ -f "$IDX" ]] && break; echo "waiting for FULL online index ($i)"; sleep 30; done
[[ -f "$IDX" ]] || { echo "FATAL: full index never appeared"; exit 1; }

OPTS=(--opts "exp_name=dspark8b_online_full_${VARIANT}")
if [[ "$VARIANT" == "conf0" ]]; then
    OPTS+=(--opts "model.confidence_head_alpha=0.0")
fi

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py \
    --config config/dspark/dspark_qwen3_8b_online_full.py \
    "${OPTS[@]}"
