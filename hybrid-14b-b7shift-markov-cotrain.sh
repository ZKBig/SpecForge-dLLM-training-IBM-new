# SHIFT-CONVENTION variant: identical to hybrid-14b-b7warm-markov-cotrain.sh except --block-convention shift
# (+ output/wandb names). deepseek-ai/dflash_qwen3_14b_block7 is shift-native, so this arm
# warm-starts WITHOUT the fillin-retrofit hole (8B gate A: shift drafter_accept 4.04 at step
# 1000 vs the fillin arm's 210k-step ceiling 3.84). All 7 slots draft.
#
# Qwen3-14B + b7 DFlash drafter, pure MARKOV head (DSpark VanillaMarkov special case),
# TEACHER-FORCED ONLY (no consistency -> no Jacobi rollout in training; eval accept_lengths
# uses the SEQUENTIAL DSpark-style branch for a Markov head), drafter CO-TRAINED.
#
# Head-vs-head partner of hybrid-14b-b7warm-sgu-consis-k3.sh: identical target, drafter, data,
# optimizer, batch, num-anchors and lambda_base -- the only differences are
#   + --markov-only                              (SGU off: bias = W2(W1[prev]) exactly)
#   - --consistency-weight/--consistency-passes   (teacher-forced only; weight defaults to 0)
#   - --no-sublayer-norm --no-mix-out --no-residual --mixer-init zeros  (SGU internals, inert)
# so any difference in accept length is attributable to the head alone.
#
# See the SGU script's header for the verified 14B model/drafter/data facts.
#
#   bash submit.sh 2 xpress hybrid-14b-b7warm-markov-cotrain

export NCCL_SOCKET_IFNAME="ib,bond"
export NCCL_IB_CUDA_SUPPORT=1

export QUACK_COMPILE_WORKERS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=${2:-1800}
export TORCH_FR_BUFFER_SIZE=2097152

export HF_HOME=/proj/checkpoints/zwang619/.cache/huggingface
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
    --block-convention shift \
    --l1-alpha 0.9 \
    --ce-alpha 0.1 \
    --loss-decay-gamma 4.0 \
    --markov-rank 256 \
    --markov-only \
    --learning-rate 0.0006 \
    --num-anchors 400 \
    --max-length 4096 \
    --batch-size 2 \
    --accumulation-steps 1 \
    --num-epochs 10 \
    --eval-interval 1000 \
    --target-model-path Qwen/Qwen3-14B \
    --train-data-path /proj/checkpoints/zwang619/DeepSpec/train_datasets/qwen3_14b/refiner_train_nothink.jsonl \
    --eval-data-path /proj/checkpoints/zwang619/DeepSpec/train_datasets/qwen3_14b/refiner_eval_nothink.jsonl \
    --cache-dir /proj/checkpoints/zwang619/ \
    --chat-template qwen \
    --attention-backend sdpa \
    --warmup-ratio 0.04 \
    --max-grad-norm 1.0 \
    --tp-size 1 \
    --drafter-lr-scale 0.1 \
    --log-interval 50 \
    --save-interval 10000 \
    --dflash-model-path deepseek-ai/dflash_qwen3_14b_block7 \
    --lambda-base-start 0.6 \
    --lambda-base-floor 0.6 \
    --lambda-base-decay-ratio 1.0 \
    --output-dir /proj/checkpoints/zwang619/hybrid_out/xpress-14b-markov-b7shift-cotrain \
    --report-to wandb \
    --wandb-project ripple-dspark \
    --wandb-name xpress-14b-markov-b7shift-cotrain \
    --resume \
    > >(stdbuf -oL sed "s/^/[$(hostname)] /") \
    2> >(stdbuf -oL sed "s/^/[$(hostname)] /" >&2)
