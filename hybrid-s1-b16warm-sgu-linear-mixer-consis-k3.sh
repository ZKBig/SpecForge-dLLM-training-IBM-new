# K=3 multi-round Jacobi consistency. Identical to hybrid-s1-b16warm-sgu-linear-mixer-consis.sh
# except for --consistency-passes / --output-dir / --wandb-name, so it is a clean A/B against it.
#
# submit with:   bash submit.sh 2 xpress hybrid-s1-b16warm-sgu-linear-mixer-consis-k3
#   (submit.sh runs `bash $3.sh`, so this FILENAME must equal the job name, and logs land in
#    /proj/checkpoints/zwang619/results/xpress/hybrid-s1-b16warm-sgu-linear-mixer-consis-k3/)

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

: "${WANDB_API_KEY:?export WANDB_API_KEY in your environment before running}"

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
    -m scripts.train_hybrid_refiner_IBM \
    --l1-alpha 0.9 \
    --ce-alpha 0.1 \
    --loss-decay-gamma 4.0 \
    --markov-rank 256 \
    --learning-rate 0.0006 \
    --num-anchors 400 \
    --max-length 4096 \
    --batch-size 1 \
    --accumulation-steps 1 \
    --num-epochs 10 \
    --eval-interval 1000 \
    --target-model-path /proj/checkpoints/daviswer/results/specu/checkpoints/Qwen3-8B \
    --train-data-path /proj/checkpoints/daviswer/results/specu/refiner_train_nothink.jsonl \
    --eval-data-path /proj/checkpoints/daviswer/results/specu/refiner_eval_nothink.jsonl \
    --cache-dir /proj/checkpoints/zwang619/ \
    --chat-template qwen \
    --attention-backend sdpa \
    --warmup-ratio 0.04 \
    --max-grad-norm 1.0 \
    --tp-size 1 \
    --log-interval 50 \
    --save-interval 10000 \
    --dflash-model-path z-lab/Qwen3-8B-DFlash-b16 \
    --lambda-base-start 0.6 \
    --lambda-base-floor 0.2 \
    --lambda-base-decay-ratio 1.0 \
    --no-sublayer-norm \
    --no-mix-out \
    --no-residual \
    --consistency-weight 0.3 \
    --consistency-passes 3 \
    --mixer-init zeros \
    --output-dir /proj/checkpoints/zwang619/hybrid_out/xpress-consis-k3 \
    --report-to wandb \
    --wandb-project ripple-dspark \
    --wandb-name xpress-consis-k3 \
    --resume \
    > >(stdbuf -oL sed "s/^/[$(hostname)] /") \
    2> >(stdbuf -oL sed "s/^/[$(hostname)] /" >&2)
