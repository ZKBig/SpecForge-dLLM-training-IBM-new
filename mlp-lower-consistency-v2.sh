# export PYTHONFAULTHANDLER=1
# export NCCL_DEBUG="INFO"
# export NCCL_DEBUG_FILE="$LOG_PATH/NCCL_DEBUG.%h.%p.txt"
# export NCCL_TOPO_DUMP_FILE="$LOG_PATH/NCCL_TOP.%h.xml"
export NCCL_SOCKET_IFNAME="ib,bond"
export NCCL_IB_CUDA_SUPPORT=1

export QUACK_COMPILE_WORKERS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=${2:-1800}
export TORCH_FR_BUFFER_SIZE=2097152

# WANDB_API_KEY comes from the environment

MASTER_ADDRESS=$(echo ${LSB_MCPU_HOSTS} | tr ' ' '\n' | head -n 1)
MASTER_PORT=$(( 20000 + (${LSB_JOBID:-0} % 40000) ))
NNODES=$(echo ${LSB_MCPU_HOSTS} | tr ' ' '\n' | sed 'n; d' | wc -w)
GPUS_PER_NODE=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -w)
NODE_RANK=$(($(echo ${LSB_MCPU_HOSTS} | tr ' ' '\n' | sed 'n; d' | grep -n -m1 $(echo $HOSTNAME | cut -d'.' -f1) | cut -d':' -f1)-1))

TOKENIZERS_PARALLELISM=false \
torchrun --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --nproc_per_node=$GPUS_PER_NODE \
    --rdzv_id=101 \
    --rdzv_endpoint=$MASTER_ADDRESS:$MASTER_PORT \
    -m scripts.train_dflash_refiner_cotrain \
    --target-model-path /proj/checkpoints/daviswer/results/specu/checkpoints/Qwen3-8B \
    --dflash-model-path z-lab/Qwen3-8B-DFlash-b16 \
    --train-data-path /proj/checkpoints/daviswer/results/specu/refiner_train_nothink.jsonl \
    --eval-data-path /proj/checkpoints/daviswer/results/specu/refiner_eval_nothink.jsonl \
    --output-dir /proj/checkpoints/daviswer/results/specu/mlp-low-consistency-v2 \
    --cache-dir /proj/checkpoints/daviswer/ \
    --chat-template qwen3-instruct \
    --window-size 0 \
    --use-residual-gate \
    --mixer-type sgu \
    --pool-type mean \
    --gate-type scalar \
    --num-refiner-layers 1 \
    --use-residual-gate \
    --mlp-intermediate 2048 \
    --consistency-weight 0.3 \
    --lambda-base-floor 0.2 \
    --lambda-base-start 0.6 \
    --lambda-base-decay-ratio 0.5 \
    --drafter-lr-scale 0.1 \
    --attention-backend flex_attention \
    --num-anchors 512 \
    --num-epochs 8 \
    --batch-size 1 \
    --learning-rate 6e-4 \
    --warmup-ratio 0.04 \
    --max-grad-norm 1.0 \
    --max-length 4096 \
    --tp-size 1 \
    --log-interval 50 \
    --save-interval 2000 \
    --eval-interval 1000 \
    --resume \
    --lowrank-lmhead-rank 256 \
    --lowrank-lmhead-init svd \
    --report-to wandb \
    --wandb-project specforge-dflash-refine \
    --wandb-name refiner-nothink-cotrain-sgu-mlp-low-rank4-consistency-v2 \
    > >(stdbuf -oL sed "s/^/[$(hostname)] /") \
    2> >(stdbuf -oL sed "s/^/[$(hostname)] /" >&2)
