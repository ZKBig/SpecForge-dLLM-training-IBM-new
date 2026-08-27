# OFFICIAL-CODE subset run, FOUR nodes (32 GPUs; ~4x faster than the 1-node variant).
# Their init_dist natively supports this: env RANK = node rank, WORLD_SIZE = node count,
# global rank = RANK*8 + local_rank. gb512 stays exact: the schedule computer derives
# accum = 512 / (32 ranks x local_bs 1) = 16 automatically.
#   bash submit.sh 4 xpress deepspec-online-subset-4node        (defaults to conf1)
# For the conf0 variant submit with a wrapper or edit VARIANT below.
set -Eeuo pipefail
VARIANT=${VARIANT:-conf1}

export PATH=/proj/checkpoints/zwang619/miniconda3/envs/deepspec_eval/bin:$PATH
export HF_HOME=/proj/checkpoints/zwang619/.cache/huggingface
export TOKENIZERS_PARALLELISM=false
: "${WANDB_API_KEY:?export WANDB_API_KEY in your environment before running}"
export DEEPSPEC_WANDB_PROJECT=ripple-dspark
export DEEPSPEC_WANDB_NAME=deepspec-official-online100k-${VARIANT}-4node
cd /proj/checkpoints/zwang619/DeepSpec

# node topology from LSF (same derivation as the fork scripts)
export MASTER_ADDR=$(echo ${LSB_MCPU_HOSTS} | tr ' ' '\n' | sed 'n; d' | sed -n '1p')
export MASTER_PORT=$(( 21000 + (${LSB_JOBID:-0} % 40000) ))
export WORLD_SIZE=$(echo ${LSB_MCPU_HOSTS} | tr ' ' '\n' | sed 'n; d' | wc -w)
export RANK=$(($(echo ${LSB_MCPU_HOSTS} | tr ' ' '\n' | sed 'n; d' | grep -n -m1 $(hostname | cut -d'.' -f1) | cut -d':' -f1)-1))
echo "[node $RANK/$WORLD_SIZE] master=$MASTER_ADDR:$MASTER_PORT variant=$VARIANT"

IDX=train_datasets/qwen3_8b_subset100k/refiner_train_100k.jsonl.online_index_ml4096_mt14.json
[[ -f "$IDX" ]] || { echo "FATAL: online index missing"; exit 1; }

OPTS=(--opts "exp_name=dspark8b_online100k_${VARIANT}_4node")
if [[ "$VARIANT" == "conf0" ]]; then
    OPTS+=(--opts "model.confidence_head_alpha=0.0")
fi

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py \
    --config config/dspark/dspark_qwen3_8b_online100k.py \
    "${OPTS[@]}"
