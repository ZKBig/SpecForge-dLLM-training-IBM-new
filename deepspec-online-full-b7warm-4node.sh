# EROSION TEST ARM: official code, FULL data, official conf1 recipe, but the drafter is
# WARM-STARTED from the saturated deepseek dflash b7 checkpoint (5.43 T1 standalone).
# Watch wandb eval/drafter_accept_len: down from ~5.4 => erosion is intrinsic to the joint
# objective (fork exonerated); flat/up => fork-implementation damage -> investigate.
# Replaces the stopped conf1 scratch arm (proven effective, 6.37@10k, ckpts kept).
set -Eeuo pipefail

export PATH=/proj/checkpoints/zwang619/miniconda3/envs/deepspec_eval/bin:$PATH
export HF_HOME=/proj/checkpoints/zwang619/.cache/huggingface
export TOKENIZERS_PARALLELISM=false
: "${WANDB_API_KEY:?export WANDB_API_KEY in your environment before running}"
export DEEPSPEC_WANDB_PROJECT=ripple-dspark
export DEEPSPEC_WANDB_NAME=deepspec-official-FULL-b7warm-4node
cd /proj/checkpoints/zwang619/DeepSpec

export MASTER_ADDR=$(echo ${LSB_MCPU_HOSTS} | tr ' ' '\n' | sed 'n; d' | sed -n '1p')
export MASTER_PORT=$(( 22000 + (${LSB_JOBID:-0} % 40000) ))
export WORLD_SIZE=$(echo ${LSB_MCPU_HOSTS} | tr ' ' '\n' | sed 'n; d' | wc -w)
export RANK=$(($(echo ${LSB_MCPU_HOSTS} | tr ' ' '\n' | sed 'n; d' | grep -n -m1 $(hostname | cut -d'.' -f1) | cut -d':' -f1)-1))
echo "[node $RANK/$WORLD_SIZE] master=$MASTER_ADDR:$MASTER_PORT B7WARM EROSION TEST"

IDX=train_datasets/qwen3_8b_full_online/refiner_train_nothink.jsonl.online_index_ml4096_mt14.json
[[ -f "$IDX" ]] || { echo "FATAL: full index missing"; exit 1; }

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py \
    --config config/dspark/dspark_qwen3_8b_online_full_b7warm.py
